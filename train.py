import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from dataclasses import dataclass
from requests import head
import yaml
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from dino_cc import DinoCC, SimCCLoss, generate_simcc_labels

from vec import IVec2


@dataclass
class DatasetConfig:
    name: str
    annotation_version: str
    annotation_file: str 

@dataclass
class ImageConfig:
    width: int
    height: int
    bbox_padding: float

@dataclass
class BackboneConfig:
    name: str
    type: str
    patch_size: int
    hidden_size: int

@dataclass
class HeadConfig:
    name: str
    type: str
    keypoint_format_name: str
    keypoints: dict
    in_channels: int
    out_channels: int
    normalization: str
    groups: int
    activation: str
    custom: Any

@dataclass
class SimCCConfig:
    pointwise_convolutions: int
    depthwise_kernel: int
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    pooling: str

@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    train_test_split: float
    gradient_clip_norm: float
    augmentation_repeats: int

@dataclass
class keypoint_mapping:
    name: str
    num_keypoints: int
    keypoints: dict


IMAGE_SIZE = IVec2(336, 448)
KEYPOINT_IDS = list(range(6, 12)) + list(range(15, 29))
KEYPOINT_INDEX = {point_id: index for index, point_id in enumerate(KEYPOINT_IDS)}
NUM_JOINTS = len(KEYPOINT_IDS)
BBOX_PADDING = 0.05
SIMCC_SPLIT_RATIO = 2.0
SIMCC_SIGMA = 6.0
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
    bbox_element = person.find('bbox')
    if bbox_element is not None:
        bbox = tuple(float(bbox_element.get(name)) for name in ('x1', 'y1', 'x2', 'y2'))
    else:
        bbox = tuple(float(person.get(name)) for name in ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'))
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
    def __init__(self, annotation_file: Path, task_ids, training=False, repeats=1, samples=None):
        self.image_root = annotation_file.parent
        self.training = training
        self.repeats = repeats if training else 1
        if samples is None:
            root = ET.parse(annotation_file).getroot()
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
        generate_simcc_labels(sample.numpy(), IMAGE_SIZE, split_ratio=SIMCC_SPLIT_RATIO, sigma=SIMCC_SIGMA)
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


def run_epoch(model, loader, criterion, optimizer, device, scaler, training, gradient_clip_norm):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * pixels.size(0)

    return total_loss / len(loader.dataset)



def create_run_dir(runs_dir, config_path):
    run_date = datetime.now().strftime('%Y_%m_%d')

    run_number = 1
    while (runs_dir / f'{run_date}_{run_number:03d}').exists():
        run_number += 1

    run_dir = runs_dir / f'{run_date}_{run_number:03d}'
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config_path = run_dir / 'config.yaml'
    shutil.copy2(config_path, run_config_path)

    return run_dir, run_config_path


def load_keypoint_format(format_name):
    mapping_path = Path('mapping')
    standard_map = mapping_path / 'standard.yaml'
    format_map = mapping_path / f'{format_name}.yaml'

    if not format_map.exists():
        raise ValueError(f'Keypoint format mapping file not found: {format_map}')
    with open(format_map, 'r') as f:
        format_mapping_cfg = keypoint_mapping(**yaml.safe_load(f))

    return format_mapping_cfg.keypoints

def get_annotation_file_path(annotation_version: str) -> Path:
    # Parse version parts (e.g., '1.2.0' -> major='1', minor='2', patch='0')
    parts = annotation_version.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Expected semantic version format 'X.Y.Z', got: {annotation_version}"
        )

    major, minor, _ = parts
    major_dir = f"{major}.X"
    minor_dir = f"{major}.{minor}.X"

    annotation_file = (
        Path("dataset")
        / "versions"
        / major_dir
        / minor_dir
        / annotation_version
        / "cleaned_annotations.xml"
    )

    if not annotation_file.exists():
        raise ValueError(f"Annotation file not found: {annotation_file}")

    return annotation_file


def parse_config(raw_config):
    image_cfg = ImageConfig(**raw_config['image'])
    backbone_cfg = BackboneConfig(**raw_config['backbone'])
    training_cfg = TrainingConfig(**raw_config['training'])


    dataset_cfg = DatasetConfig(**raw_config['dataset'], annotation_file='')
    annotation_path = Path('annotations')
    dataset_cfg.annotation_file = str(annotation_path / f'annotations_{dataset_cfg.annotation_version}.xml')


    head_cfg = HeadConfig(**raw_config['head'], keypoints={})
    head_cfg.keypoints = load_keypoint_format(head_cfg.keypoint_format_name)

    match (head_cfg.type):
        case 'simcc':
            head_cfg.custom = SimCCConfig(**head_cfg.custom)
        case _:
            raise ValueError(f'Unsupported head type: {head_cfg.type}')


    return {
        'dataset': dataset_cfg,
        'image': image_cfg,
        'backbone': backbone_cfg,
        'head': head_cfg,
        'training': training_cfg,
    }

def main():
    parser = argparse.ArgumentParser(description='Train DinoCC on the cleaned keypoint dataset.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, encoding='utf-8') as config_file:
        raw_config = yaml.safe_load(config_file)
    config = parse_config(raw_config)

    runs_dir = Path('training/runs')
    run_dir, run_config_path = create_run_dir(runs_dir, config_path)

    dataset_cfg: DatasetConfig = config['dataset']
    image_cfg: ImageConfig = config['image']
    training_cfg: TrainingConfig = config['training']
    head_cfg: HeadConfig = config['head']
    backbone_cfg: BackboneConfig = config['backbone']

    global IMAGE_SIZE, KEYPOINTS_BY_NAME, KEYPOINTS_BY_ID, KEYPOINT_INDEX, NUM_JOINTS, BBOX_PADDING, SIMCC_SPLIT_RATIO, SIMCC_SIGMA
    IMAGE_SIZE = IVec2(image_cfg.width, image_cfg.height)
    KEYPOINTS_BY_NAME = head_cfg.keypoints
    KEYPOINTS_BY_ID = {id: {'name': name, 'standard_id': standard_id} for name, (id, standard_id) in KEYPOINTS_BY_NAME.items()}
    NUM_JOINTS = len(head_cfg.keypoints)
    BBOX_PADDING = image_cfg.bbox_padding


    set_seed(training_cfg.seed)

    annotation_file = get_annotation_file_path(dataset_cfg.annotation_version)
    root = ET.parse(annotation_file).getroot()
    task_ids = [video.get('id') for video in root.findall('./project/videos/video')]
    full_dataset = KeypointDataset(annotation_file, set(task_ids))
    image_groups = {}
    for index, (image_path, _, _) in enumerate(full_dataset.samples):
        image_groups.setdefault(image_path, []).append(index)
    image_paths = list(image_groups)
    random.shuffle(image_paths)
    split_index = max(1, min(len(image_paths) - 1, round(len(image_paths) * training_cfg.train_test_split)))
    train_indices = [index for path in image_paths[:split_index] for index in image_groups[path]]
    test_indices = [index for path in image_paths[split_index:] for index in image_groups[path]]
    train_samples = [full_dataset.samples[index] for index in train_indices]
    test_samples = [full_dataset.samples[index] for index in test_indices]
    train_dataset = KeypointDataset(
        annotation_file,
        set(task_ids),
        training=True,
        repeats=training_cfg.augmentation_repeats,
        samples=train_samples,
    )
    test_dataset = KeypointDataset(
        annotation_file,
        set(task_ids),
        samples=test_samples,
    )
    train_loader = make_loader(train_dataset, training_cfg.batch_size, args.workers)
    test_loader = make_loader(test_dataset, training_cfg.batch_size, args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    match backbone_cfg.name:
        case 'dino_v2_base':
            match head_cfg.type:
                case 'simcc':
                    model = DinoCC(
                        NUM_JOINTS,
                        IMAGE_SIZE,
                        freeze_backbone=True,
                        split_ratio=SIMCC_SPLIT_RATIO,
                        neck_dim=head_cfg.out_channels,
                    ).to(device)
                    criterion = SimCCLoss().to(device)
                case _:
                    raise ValueError(f'Unsupported head type for backbone {backbone_cfg.name}: {head_cfg.type}')
        case _:
            raise ValueError(f'Unsupported backbone type: {backbone_cfg.name}')


    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.max_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_path = run_dir / 'best.pt'
    print(
        f'Training on {device}; '
        f'train images={split_index}/{len(image_paths)}, '
        f'test images={len(image_paths) - split_index}/{len(image_paths)}'
    )

    for epoch in range(1, training_cfg.max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True, training_cfg.gradient_clip_norm)
        val_loss = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False, training_cfg.gradient_clip_norm)
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
                    'config_path': str(run_config_path),
                    'seed': training_cfg.seed,
                    'split': f'{training_cfg.train_test_split:.2f} by image',
                    'epoch': epoch,
                    'val_loss': val_loss,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_cfg.early_stopping_patience:
                print(f"Early stopping after {training_cfg.early_stopping_patience} epochs without improvement.")
                break

    final_path = run_dir / 'final.pt'
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'num_joints': NUM_JOINTS,
            'keypoint_ids': KEYPOINT_IDS,
            'joint_loss_weights': JOINT_LOSS_WEIGHTS,
            'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
            'config_path': str(run_config_path),
            'seed': training_cfg.seed,
            'split': f'{training_cfg.train_test_split:.2f} by image',
            'epoch': epoch,
            'test_loss': val_loss,
        },
        final_path,
    )
    print(f'Saved best checkpoint to {best_path}')
    print(f'Saved final checkpoint to {final_path}')


if __name__ == '__main__':
    main()