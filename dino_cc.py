import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

from vec import IVec2

class BaseSimCCHead(nn.Module, ABC):
  """Formal interface for any SimCC head variant."""

  @abstractmethod
  def forward(
      self, patch_tokens: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Args:

        patch_tokens: [B, N, C]

    Returns:
        (pred_x, pred_y): ([B, K, W_bins], [B, K, H_bins])
    """
    pass


class DepthwiseSimCCHead(BaseSimCCHead):
  """Your current head with 1x1 + 3x3 depthwise + 1x1 neck."""

  def __init__(
      self,
      in_channels: int,
      num_joints: int,
      grid_size: IVec2,
      out_size: IVec2,
      split_ratio: float = 2.0,
      neck_dim: int = 256,
      **kwargs,
  ):
    super().__init__()
    self.grid_size = grid_size
    self.neck_dim = neck_dim
    self.out_bins_size = IVec2(
        int(out_size.x * split_ratio), int(out_size.y * split_ratio)
    )

    self.neck = nn.Sequential(
        nn.Conv2d(in_channels, neck_dim, kernel_size=1, bias=False),
        nn.GroupNorm(32, neck_dim),
        nn.GELU(),
        nn.Conv2d(
            neck_dim,
            neck_dim,
            kernel_size=3,
            padding=1,
            groups=neck_dim,
            bias=False,
        ),
        nn.Conv2d(neck_dim, neck_dim, kernel_size=1, bias=False),
        nn.GroupNorm(32, neck_dim),
        nn.GELU(),
    )

    self.mlp_x = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, None)),
        nn.Flatten(start_dim=2),
        nn.Linear(self.grid_size.x, self.out_bins_size.x),
    )
    self.fc_x = nn.Linear(neck_dim, num_joints)

    self.mlp_y = nn.Sequential(
        nn.AdaptiveAvgPool2d((None, 1)),
        nn.Flatten(start_dim=2),
        nn.Linear(self.grid_size.y, self.out_bins_size.y),
    )
    self.fc_y = nn.Linear(neck_dim, num_joints)

  def forward(self, patch_tokens: torch.Tensor):
    B, _, C = patch_tokens.shape
    x_2d = patch_tokens.permute(0, 2, 1).reshape(
        B, C, self.grid_size.y, self.grid_size.x
    )
    feat = self.neck(x_2d)

    # Note: transpose(1, 2) is cleaner than permute(0, 2, 1)
    feat_x = self.mlp_x(feat).transpose(1, 2)
    pred_x = self.fc_x(feat_x).transpose(1, 2)

    feat_y = self.mlp_y(feat).transpose(1, 2)
    pred_y = self.fc_y(feat_y).transpose(1, 2)

    return pred_x, pred_y


class _JointQueryDecoderLayer(nn.Module):
  """Cross-attention over patch memory only (no query–query mixing)."""

  def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    self.norm_q = nn.LayerNorm(dim)
    self.norm_kv = nn.LayerNorm(dim)
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_ffn = nn.LayerNorm(dim)
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )

  def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    q = self.norm_q(queries)
    kv = self.norm_kv(memory)
    attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
    queries = queries + attn_out
    queries = queries + self.ffn(self.norm_ffn(queries))
    return queries


class _JointQuerySelfAttnDecoderLayer(nn.Module):
  """DETR-style layer: self-attn among joint queries, then cross-attn to patches."""

  def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    self.norm_self = nn.LayerNorm(dim)
    self.self_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_q = nn.LayerNorm(dim)
    self.norm_kv = nn.LayerNorm(dim)
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_ffn = nn.LayerNorm(dim)
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )

  def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    q_self = self.norm_self(queries)
    self_out, _ = self.self_attn(q_self, q_self, q_self, need_weights=False)
    queries = queries + self_out

    q = self.norm_q(queries)
    kv = self.norm_kv(memory)
    cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
    queries = queries + cross_out
    queries = queries + self.ffn(self.norm_ffn(queries))
    return queries


class _BaseJointQuerySimCCHead(BaseSimCCHead):
  """Shared joint-query → SimCC head; subclasses set ``_layer_cls``."""

  _layer_cls = _JointQueryDecoderLayer

  def __init__(
      self,
      in_channels: int,
      num_joints: int,
      grid_size: IVec2,
      out_size: IVec2,
      split_ratio: float = 2.0,
      neck_dim: int = 256,
      num_heads: int = 8,
      num_layers: int = 2,
      dropout: float = 0.0,
      **kwargs,
  ):
    super().__init__()
    if neck_dim % num_heads != 0:
      raise ValueError(f'neck_dim ({neck_dim}) must be divisible by num_heads ({num_heads})')

    self.grid_size = grid_size
    self.neck_dim = neck_dim
    self.num_joints = num_joints
    self.out_bins_size = IVec2(
        int(out_size.x * split_ratio), int(out_size.y * split_ratio)
    )

    self.input_proj = nn.Linear(in_channels, neck_dim)
    self.joint_queries = nn.Parameter(torch.randn(num_joints, neck_dim) * 0.02)
    self.patch_pos_embed = nn.Parameter(
        torch.randn(1, grid_size.x * grid_size.y, neck_dim) * 0.02
    )
    self.layers = nn.ModuleList(
        [self._layer_cls(neck_dim, num_heads, dropout) for _ in range(num_layers)]
    )
    self.fc_x = nn.Linear(neck_dim, self.out_bins_size.x)
    self.fc_y = nn.Linear(neck_dim, self.out_bins_size.y)

  def forward(self, patch_tokens: torch.Tensor):
    B, N, _ = patch_tokens.shape
    expected_n = self.grid_size.x * self.grid_size.y
    if N != expected_n:
      raise ValueError(
          f'Expected {expected_n} patch tokens for grid {self.grid_size.x}x{self.grid_size.y}, got {N}'
      )

    memory = self.input_proj(patch_tokens) + self.patch_pos_embed
    queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)
    for layer in self.layers:
      queries = layer(queries, memory)

    pred_x = self.fc_x(queries)
    pred_y = self.fc_y(queries)
    return pred_x, pred_y


class JointQuerySimCCHead(_BaseJointQuerySimCCHead):
  """Learnable joint queries with cross-attention over patch tokens, then SimCC X/Y heads."""

  _layer_cls = _JointQueryDecoderLayer


class JointQuerySelfAttnSimCCHead(_BaseJointQuerySimCCHead):
  """Joint queries with self-attention among joints, then cross-attention to patches + SimCC."""

  _layer_cls = _JointQuerySelfAttnDecoderLayer


HEAD_REGISTRY = {
    'depthwise_simcc': DepthwiseSimCCHead,
    'joint_query_simcc': JointQuerySimCCHead,
    'joint_query_self_attn_simcc': JointQuerySelfAttnSimCCHead,
}


class DinoCC(nn.Module):
    def __init__(self, num_joints: int, img_size: IVec2, freeze_backbone: bool = True,
                 split_ratio: float = 2.0, neck_dim: int = 256,
                 backbone_name: str = 'facebook/dinov2-base',
                 head_name: str = 'depthwise_simcc',
                 head_kwargs: dict | None = None):
        super().__init__()
        self.num_joints = num_joints
        self.img_size = img_size
        self.backbone_name = backbone_name
        self.head_name = head_name

        # Load pre-trained ViT backbone
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = getattr(self.backbone.config, 'num_register_tokens', 0) or 0

        if self.img_size.x % self.patch_size != 0 or self.img_size.y % self.patch_size != 0:
            raise ValueError(
                f'Image size {self.img_size.x}x{self.img_size.y} must be divisible by '
                f'backbone patch size {self.patch_size}'
            )

        self.grid_size = IVec2(self.img_size.x // self.patch_size, self.img_size.y // self.patch_size)

        # Freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if head_name not in HEAD_REGISTRY:
            raise ValueError(
                f'Unknown head_name={head_name!r}; expected one of {sorted(HEAD_REGISTRY)}'
            )
        head_cls = HEAD_REGISTRY[head_name]
        kwargs = {
            'in_channels': self.backbone.config.hidden_size,
            'num_joints': self.num_joints,
            'grid_size': self.grid_size,
            'out_size': IVec2(self.img_size.x, self.img_size.y),
            'split_ratio': split_ratio,
            'neck_dim': neck_dim,
        }
        if head_kwargs:
            kwargs.update(head_kwargs)
        self.head = head_cls(**kwargs)

    def load_weights(self, checkpoint_path, map_location='cpu', strict=True):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.load_state_dict(state_dict, strict=strict)
        return checkpoint


    def forward(self, pixel_values):
        # DINOv2 needs pos-encoding interpolation for non-native resolutions; DINOv3 uses RoPE
        if getattr(self.backbone.config, 'model_type', None) == 'dinov2':
            outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        # Token 0 is [CLS]; DINOv3 (and some DINOv2 variants) insert register tokens before patches
        patch_start = 1 + self.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, patch_start:, :]  # [B, N, C]

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