import argparse
import random
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from dino_cc import DinoCC, SimCCLoss, generate_simcc_labels
from vec import IVec2


IMAGE_SIZE = IVec2(336, 448)
KEYPOINT_IDS = list(range(6, 12)) + list(range(15, 29))
KEYPOINT_INDEX = {point_id: index for index, point_id in enumerate(KEYPOINT_IDS)}
NUM_JOINTS = len(KEYPOINT_IDS)
BBOX_PADDING = 0.05
LEFT_RIGHT_PAIRS = (
    (6, 9),
    (7, 10),
    (8, 11),
    (15, 18),
    (16, 19),
    (17, 20),
    (21, 25),
    (22, 26),
    (23, 27),
    (24, 28),
)
JOINT_LOSS_WEIGHTS = {
    6: 2.0,
    7: 2.5,
    8: 2.0,
    9: 2.0,
    10: 2.5,
    11: 2.0,
    15: 1.5,
    16: 2.5,
    17: 1.5,
    18: 1.5,
    19: 2.5,
    20: 1.5,
    21: 1.25,
    22: 1.25,
    23: 1.25,
    24: 1.25,
    25: 1.25,
    26: 1.25,
    27: 1.25,
    28: 1.25,
}
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_person(person):
    keypoints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    for point in person.findall('keypoint'):
        point_id = int(point.get('id'))
        if point_id not in KEYPOINT_INDEX:
            continue
        joint_id = KEYPOINT_INDEX[point_id]
        x, y = (float(value) for value in point.get('x').split(',')) if ',' in point.get('x', '') else (float(point.get('x')), float(point.get('y')))
        keypoints[joint_id] = [x, y, float(point.get('visibility', '0'))]
    bbox = tuple(
        float(person.get(name))
        for name in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2')
    )
    return keypoints, bbox


def horizontally_flip_keypoints(keypoints):
    flipped = keypoints.copy()
    flipped[:, 0] = IMAGE_SIZE.x - 1 - flipped[:, 0]
    for left_id, right_id in LEFT_RIGHT_PAIRS:
        left_index = KEYPOINT_INDEX[left_id]
        right_index = KEYPOINT_INDEX[right_id]
        flipped[[left_index, right_index]] = flipped[[right_index, left_index]]
    return flipped


class KeypointDataset(Dataset):
    def __init__(self, annotation_file, task_ids, training=False, repeats=1, samples=None):
        self.annotation_file = Path(annotation_file)
        self.image_root = self.annotation_file.parent
        self.training = training
        self.repeats = repeats if training else 1
        if samples is None:
            root = ET.parse(self.annotation_file).getroot()
            samples = []
            for video in root.findall('./project/videos/video'):
                if video.get('id') not in task_ids:
                    continue
                for frame in video.findall('frame'):
                    image_path = self.image_root / frame.get('image')
                    for person in frame.findall('person'):
                        keypoints, bbox = parse_person(person)
                        samples.append((image_path, keypoints, bbox))

        self.samples = list(samples)

        if not self.samples:
            raise ValueError(f'No samples found for task IDs: {sorted(task_ids)}')

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        image_path, keypoints, bbox = self.samples[index % len(self.samples)]
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        crop_width = max(
            bbox_width * (1 + 2 * BBOX_PADDING),
            bbox_height * IMAGE_SIZE.x / IMAGE_SIZE.y,
        )
        crop_height = max(
            bbox_height * (1 + 2 * BBOX_PADDING),
            crop_width * IMAGE_SIZE.y / IMAGE_SIZE.x,
        )
        crop_width = min(crop_width, original_width)
        crop_height = min(crop_height, original_height)
        crop_x1 = max(0.0, min(center_x - crop_width / 2, original_width - crop_width))
        crop_y1 = max(0.0, min(center_y - crop_height / 2, original_height - crop_height))
        image = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
        image = image.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)

        keypoints = keypoints.copy()
        keypoints[:, 0] = (keypoints[:, 0] - crop_x1) * IMAGE_SIZE.x / crop_width
        keypoints[:, 1] = (keypoints[:, 1] - crop_y1) * IMAGE_SIZE.y / crop_height

        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                keypoints = horizontally_flip_keypoints(keypoints)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))

        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
        return pixels, torch.from_numpy(keypoints)


def make_loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_targets(keypoints, device):
    targets = [
        generate_simcc_labels(sample.numpy(), IMAGE_SIZE)
        for sample in keypoints
    ]
    target_x, target_y, weights = zip(*targets)
    joint_weights = torch.tensor(
        [JOINT_LOSS_WEIGHTS[point_id] for point_id in KEYPOINT_IDS],
        dtype=torch.float32,
    )
    return (
        torch.stack(target_x).to(device),
        torch.stack(target_y).to(device),
        torch.stack(weights).to(device) * joint_weights.to(device),
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler, training):
    model.train(training)
    total_loss = 0.0

    for pixels, keypoints in loader:
        pixels = pixels.to(device, non_blocking=True)
        target_x, target_y, weights = make_targets(keypoints, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y = model(pixels)
            loss = criterion(pred_x, pred_y, target_x, target_y, weights)

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * pixels.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Train DinoCC on the cleaned keypoint dataset.')
    parser.add_argument('--annotations', default='datasets/cleaned_dataset_1.0_9-9/annotations/annotations.xml')
    parser.add_argument('--output', default='weights/dino_cc_cleaned.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--augmentation-repeats', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=6)
    args = parser.parse_args()

    set_seed(args.seed)
    annotation_file = Path(args.annotations)
    root = ET.parse(annotation_file).getroot()
    task_ids = [video.get('id') for video in root.findall('./project/videos/video')]
    full_dataset = KeypointDataset(annotation_file, set(task_ids))
    image_groups = {}
    for index, (image_path, _, _) in enumerate(full_dataset.samples):
        image_groups.setdefault(image_path, []).append(index)
    image_paths = list(image_groups)
    random.shuffle(image_paths)
    split_index = max(1, min(len(image_paths) - 1, round(len(image_paths) * 0.8)))
    train_indices = [index for path in image_paths[:split_index] for index in image_groups[path]]
    test_indices = [index for path in image_paths[split_index:] for index in image_groups[path]]
    train_samples = [full_dataset.samples[index] for index in train_indices]
    test_samples = [full_dataset.samples[index] for index in test_indices]
    train_dataset = KeypointDataset(
        annotation_file,
        set(task_ids),
        training=True,
        repeats=args.augmentation_repeats,
        samples=train_samples,
    )
    test_dataset = KeypointDataset(
        annotation_file,
        set(task_ids),
        samples=test_samples,
    )
    train_loader = make_loader(train_dataset, args.batch_size, args.workers)
    test_loader = make_loader(test_dataset, args.batch_size, args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DinoCC(NUM_JOINTS, IMAGE_SIZE, freeze_backbone=True).to(device)
    criterion = SimCCLoss().to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f'Training on {device}; '
        f'train images={split_index}/{len(image_paths)}, '
        f'test images={len(image_paths) - split_index}/{len(image_paths)}'
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        val_loss = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False)
        scheduler.step()
        print(f'Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'num_joints': NUM_JOINTS,
                    'keypoint_ids': KEYPOINT_IDS,
                    'joint_loss_weights': JOINT_LOSS_WEIGHTS,
                    'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
                    'split': '80/20 by image',
                    'epoch': epoch,
                    'val_loss': val_loss,
                },
                output_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f'Early stopping after {args.patience} epochs without improvement.')
                break

    final_path = output_path.with_name(f'{output_path.stem}_final{output_path.suffix}')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'num_joints': NUM_JOINTS,
            'keypoint_ids': KEYPOINT_IDS,
            'joint_loss_weights': JOINT_LOSS_WEIGHTS,
            'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
            'split': '80/20 by image',
            'epoch': args.epochs,
            'test_loss': val_loss,
        },
        final_path,
    )
    print(f'Saved best checkpoint to {output_path}')
    print(f'Saved final checkpoint to {final_path}')


if __name__ == '__main__':
    main()