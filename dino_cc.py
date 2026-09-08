import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np

from vec import IVec2


class SimCCPoseHead(nn.Module):
    def __init__(self, in_channels:int, num_joints:int, grid_size:IVec2,
                 out_size:IVec2, split_ratio:float):
        super().__init__()
        self.in_channels = in_channels
        self.out_size = out_size
        self.grid_size = grid_size
        self.num_joints = num_joints
        self.split_ratio = split_ratio
        
        # Continuous sub-pixel coordinate bins
        self.out_bins_size = IVec2(int(out_size.x * split_ratio), int(out_size.y * split_ratio))
        
        # Neck: Adapt transformer features and capture local neighborhood context
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
            # Depthwise convolution: stabilizes ambiguous loose/dark fabrics
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU()
        )

        # X-Branch: Pool along height -> project to horizontal bins
        self.mlp_x = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(start_dim=2),
            nn.Linear(self.grid_size.x, self.out_bins_size.x)
        )
        self.fc_x = nn.Linear(256, num_joints)

        # Y-Branch: Pool along width -> project to vertical bins
        self.mlp_y = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Flatten(start_dim=2),
            nn.Linear(self.grid_size.y, self.out_bins_size.y)
        )

        self.fc_y = nn.Linear(256, num_joints)

    def forward(self, patch_tokens):
        # patch_tokens: [B, 768, 768] (without CLS token)
        B, N, C = patch_tokens.shape
        
        # Reshape to spatial feature map: [B, C, grid_y, grid_x]
        x_2d = patch_tokens.permute(0, 2, 1).reshape(B, C, self.grid_size.y, self.grid_size.x)
        feat = self.neck(x_2d) # [B, 256, grid_y, grid_x]

        # Classify X coordinate
        feat_x = self.mlp_x(feat).permute(0, 2, 1)  # [B, out_w_bins, 256]
        pred_x = self.fc_x(feat_x).permute(0, 2, 1) # [B, num_joints, out_w_bins]

        # Classify Y coordinate
        feat_y = self.mlp_y(feat).permute(0, 2, 1)  # [B, out_h_bins, 256]
        pred_y = self.fc_y(feat_y).permute(0, 2, 1) # [B, num_joints, out_h_bins]

        return pred_x, pred_y


class DinoCC(nn.Module):
    def __init__(self, num_joints:int, img_size:IVec2, freeze_backbone: bool = True):
        super().__init__()
        self.patch_size = 14
        self.num_joints = num_joints
        self.img_size = img_size
        self.grid_size = IVec2(self.img_size.x // self.patch_size, self.img_size.y // self.patch_size)

        # Load pre-trained ViT backbone
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # Freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Create custom head
        self.head = SimCCPoseHead(
            in_channels=self.backbone.config.hidden_size,
            num_joints=self.num_joints,
            grid_size=self.grid_size,
            out_size=IVec2(self.img_size.x, self.img_size.y),
            split_ratio=2.0
        )


    def forward(self, pixel_values):
        # Extract features through DINOv2
        outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        
        # Hugging Face token index 0 is [CLS]; tokens 1: are the spatial patches
        patch_tokens = outputs.last_hidden_state[:, 1:, :] # [B, N, 768]
        
        # Predict 1D coordinates
        pred_x, pred_y = self.head(patch_tokens)
        return pred_x, pred_y



def generate_simcc_labels(keypoints, img_size: IVec2, split_ratio=2.0, sigma=6.0):
    """
    Args:
        keypoints: Array-like of shape [K, 3] with rows: [x, y, visibility]
                   Coordinates must be relative to the cropped 336x448 image.
        img_size: IVec2(x=336, y=448)
        split_ratio: Sub-pixel scaling factor (2.0 = 0.5px bins)
        sigma: Standard deviation of the Gaussian label distribution in bin units
    Returns:
        target_x: [K, out_bins_x]
        target_y: [K, out_bins_y]
        weights:  [K] Loss weight per joint based on visibility
    """
    num_joints = len(keypoints)
    bins_x = int(img_size.x * split_ratio)
    bins_y = int(img_size.y * split_ratio)

    target_x = np.zeros((num_joints, bins_x), dtype=np.float32)
    target_y = np.zeros((num_joints, bins_y), dtype=np.float32)
    weights = np.zeros((num_joints,), dtype=np.float32)

    x_bins = np.arange(bins_x)
    y_bins = np.arange(bins_y)

    for k, (x, y, v) in enumerate(keypoints):
        # Skip joints marked outside frame or invalid
        if v == 0 or x < 0 or y < 0 or x >= img_size.x or y >= img_size.y:
            continue

        # Scale continuous pixels to sub-pixel bin centers
        mu_x = x * split_ratio
        mu_y = y * split_ratio

        # Continuous 1D Gaussian distributions
        gx = np.exp(-((x_bins - mu_x) ** 2) / (2.0 * sigma ** 2))
        gy = np.exp(-((y_bins - mu_y) ** 2) / (2.0 * sigma ** 2))

        # Normalize to valid probability density
        target_x[k] = gx / (gx.sum() + 1e-8)
        target_y[k] = gy / (gy.sum() + 1e-8)

        # Visibility weighting: full weight if visible, 0.35 if occluded by partner/clothing
        weights[k] = 1.0 if v == 2 else 0.35

    return (
        torch.from_numpy(target_x),
        torch.from_numpy(target_y),
        torch.from_numpy(weights)
    )


class SimCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # batchmean computes the mathematically rigorous KL-divergence over batches
        self.kl = nn.KLDivLoss(reduction='none')

    def forward(self, pred_x, pred_y, target_x, target_y, weights):
        """
        pred_x:   [B, K, bins_x] (raw logits)
        pred_y:   [B, K, bins_y] (raw logits)
        target_x: [B, K, bins_x] (normalized probability density)
        target_y: [B, K, bins_y] (normalized probability density)
        weights:  [B, K]
        """
        # Convert logits to log-probabilities
        log_pred_x = F.log_softmax(pred_x, dim=-1)
        log_pred_y = F.log_softmax(pred_y, dim=-1)

        # Calculate KL divergence per joint: sum across the bin dimension
        loss_x = self.kl(log_pred_x, target_x).sum(dim=-1) # [B, K]
        loss_y = self.kl(log_pred_y, target_y).sum(dim=-1) # [B, K]

        # Weight by joint visibility
        loss = (loss_x + loss_y) * weights # [B, K]

        # Normalize across non-zero weighted joints
        valid_joints = weights.sum() + 1e-6
        return loss.sum() / valid_joints


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")

    num_joints = 19
    img_size = IVec2(336, 448) # width=336, height=448

    processor = AutoImageProcessor.from_pretrained(
        'facebook/dinov2-base', 
        size={"height": 448, "width": 336},
        crop_size={"height": 448, "width": 336},
        do_center_crop=False
    )
    
    # 1. Instantiate model and move to GPU
    model = DinoCC(num_joints=num_joints, img_size=img_size, freeze_backbone=True).to(device)
    criterion = SimCCLoss().to(device)

    # 2. Setup AdamW optimizer only over the head & neck parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    # 3. Load sample image and prepare input batch
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device) # Shape: [1, 3, 448, 336]

    # 4. Generate dummy target keypoints [K, 3] -> (x, y, visibility)
    # Simulating 19 keypoints within the 336x448 crop
    dummy_kpts = np.zeros((num_joints, 3), dtype=np.float32)
    for i in range(num_joints):
        dummy_kpts[i] = [50.0 + i * 12.0, 100.0 + i * 15.0, 2 if i % 4 != 0 else 1]

    # Convert coordinates to SimCC 1D Gaussian targets
    target_x, target_y, weights = generate_simcc_labels(dummy_kpts, img_size=img_size, split_ratio=2.0)
    
    # Add batch dimension and send to device: [1, K, bins]
    target_x = target_x.unsqueeze(0).to(device)
    target_y = target_y.unsqueeze(0).to(device)
    weights = weights.unsqueeze(0).to(device)

    # 5. Run Training Loop & Backprop
    model.train()
    print("\nStarting verification backpropagation loop...")
    for step in range(1, 26):
        optimizer.zero_grad()

        # Forward pass
        pred_x, pred_y = model(pixel_values)

        # Compute KL Divergence loss
        loss = criterion(pred_x, pred_y, target_x, target_y, weights)

        # Backpropagate and update gradients
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d} | Loss: {loss.item():.5f}")

    # 6. Verify predicted coordinates after gradient descent
    model.eval()
    with torch.no_grad():
        p_x, p_y = model(pixel_values)
        pred_coord_x = torch.argmax(p_x, dim=-1).float() / 2.0
        pred_coord_y = torch.argmax(p_y, dim=-1).float() / 2.0

    print("\n--- Target vs Learned Prediction Check (Joint 0) ---")
    print(f"Ground Truth : ({dummy_kpts[0][0]:.1f}, {dummy_kpts[0][1]:.1f})")
    print(f"Model Learned: ({pred_coord_x[0, 0].item():.1f}, {pred_coord_y[0, 0].item():.1f})")








"""
Input image
    [B, 3, 448, 336]  (height, width)
            |
            v
DINOv2 ViT (14 x 14 patches)
    patch grid: 32 x 24 (grid_y x grid_x)
    last_hidden_state: [B, 769, 768]
            |
            v
Remove CLS token
    patch_tokens: [B, 768, 768]
            |
            v
Reshape tokens to spatial feature map
    [B, 768, 32, 24]  (channels, grid_y, grid_x)
            |
            v
SimCCPoseHead
    Neck: 1x1 projection -> GroupNorm/GELU
          -> depthwise 3x3 -> 1x1 -> GroupNorm/GELU
    features: [B, 256, 32, 24]
            |
       +----+----+
       |         |
       v         v
    X branch  Y branch
    pool H    pool W
    Linear    Linear
       |         |
       v         v
    pred_x    pred_y
 [B, K, 672] [B, K, 896]
   (x bins)    (y bins)
"""