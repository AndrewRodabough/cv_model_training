import argparse
import csv
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from dataclasses import dataclass
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset

from dino_cc import DinoCC, SimCCLoss, generate_simcc_labels

from vec import IVec2


class StdoutCsvTee:
    """Mirror stdout line-by-line into a CSV log while still printing to the console."""

    def __init__(self, stream, csv_path: Path):
        self.stream = stream
        self._buffer = ''
        self._file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['timestamp', 'message'])
        self._file.flush()

    def write(self, data):
        self.stream.write(data)
        if not isinstance(data, str):
            data = str(data)
        self._buffer += data
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._writer.writerow([datetime.now().isoformat(timespec='seconds'), line])
            self._file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self._file.flush()

    def close(self):
        if self._buffer:
            self._writer.writerow([datetime.now().isoformat(timespec='seconds'), self._buffer])
            self._buffer = ''
            self._file.flush()
        self._file.close()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


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
    left_right_pairs: list | None = None

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
class JointQuerySimCCConfig:
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0

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
    joint_loss_weights: dict[str, float]
    worst_test_overlays: int = 10
    # frame: random images; stratified_clip: every clip in both sets; clip: whole videos
    split_mode: str = 'frame'
    # Train-time augmentation (applied when KeypointDataset.training=True)
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.25
    aug_shift_frac: float = 0.1
    aug_rotation_deg: float = 12.0
    aug_color_jitter: float = 0.25

@dataclass
class keypoint_mapping:
    name: str
    num_keypoints: int
    keypoints: dict
    left_right_pairs: list | None = None


IMAGE_SIZE = IVec2(384, 512)
KEYPOINT_IDS = list(range(6, 12)) + list(range(15, 29))
KEYPOINT_INDEX = {point_id: index for index, point_id in enumerate(KEYPOINT_IDS)}
NUM_JOINTS = len(KEYPOINT_IDS)
KEYPOINTS_BY_NAME: dict = {}
BBOX_PADDING = 0.05
SIMCC_SPLIT_RATIO = 2.0
SIMCC_SIGMA = 6.0
LEFT_RIGHT_PAIRS: list[tuple[int, int]] = []
JOINT_LOSS_WEIGHTS = torch.empty(0)
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
AUG_SCALE_MIN = 0.85
AUG_SCALE_MAX = 1.25
AUG_SHIFT_FRAC = 0.1
AUG_ROTATION_DEG = 12.0
AUG_COLOR_JITTER = 0.25





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
    for left_index, right_index in LEFT_RIGHT_PAIRS:
        flipped[[left_index, right_index]] = flipped[[right_index, left_index]]
    return flipped


def rotate_image_and_keypoints(image: Image.Image, keypoints: np.ndarray, angle_deg: float):
    """Rotate crop and keypoints together (CCW, same as PIL). Mark OOB joints invisible."""
    if abs(angle_deg) < 1e-6:
        return image, keypoints

    rotated = image.rotate(angle_deg, resample=Image.Resampling.BILINEAR, expand=False)
    width, height = image.size
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    out = keypoints.copy()
    x = keypoints[:, 0] - cx
    y = keypoints[:, 1] - cy
    out[:, 0] = cx + x * cos_t - y * sin_t
    out[:, 1] = cy + x * sin_t + y * cos_t
    out_of_bounds = (
        (out[:, 0] < 0)
        | (out[:, 1] < 0)
        | (out[:, 0] >= width)
        | (out[:, 1] >= height)
    )
    out[out_of_bounds, 2] = 0.0
    return rotated, out


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

        if self.training:
            scale = random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX)
            shift_x_frac = random.uniform(-AUG_SHIFT_FRAC, AUG_SHIFT_FRAC)
            shift_y_frac = random.uniform(-AUG_SHIFT_FRAC, AUG_SHIFT_FRAC)
        else:
            scale = 1.0
            shift_x_frac = 0.0
            shift_y_frac = 0.0

        crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
            bbox,
            original_width,
            original_height,
            scale=scale,
            shift_x_frac=shift_x_frac,
            shift_y_frac=shift_y_frac,
        )
        image = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
        image = image.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)

        keypoints = transform_keypoints_to_crop(
            keypoints, crop_x1, crop_y1, crop_width, crop_height
        )

        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                keypoints = horizontally_flip_keypoints(keypoints)
            if AUG_ROTATION_DEG > 0:
                angle = random.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG)
                image, keypoints = rotate_image_and_keypoints(image, keypoints, angle)
            color_lo = 1.0 - AUG_COLOR_JITTER
            color_hi = 1.0 + AUG_COLOR_JITTER
            image = ImageEnhance.Brightness(image).enhance(random.uniform(color_lo, color_hi))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(color_lo, color_hi))
            image = ImageEnhance.Color(image).enhance(random.uniform(color_lo, color_hi))

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


def build_joint_loss_weights(weights_by_name: dict[str, float], keypoints_by_name: dict) -> torch.Tensor:
    missing = sorted(name for name in keypoints_by_name if name not in weights_by_name)
    if missing:
        raise ValueError(f'Missing joint_loss_weights for: {missing}')
    ordered = sorted(keypoints_by_name.items(), key=lambda item: item[1]['id'])
    return torch.tensor([float(weights_by_name[name]) for name, _ in ordered], dtype=torch.float32)


def build_left_right_index_pairs(pairs: list | None, keypoints_by_name: dict) -> list[tuple[int, int]]:
    if not pairs:
        raise ValueError('left_right_pairs missing from keypoint format mapping')
    index_pairs = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f'Expected [left, right] pair, got: {pair}')
        left_name, right_name = pair
        for name in (left_name, right_name):
            if name not in keypoints_by_name:
                raise ValueError(f'Unknown keypoint in left_right_pairs: {name}')
        index_pairs.append(
            (keypoints_by_name[left_name]['id'], keypoints_by_name[right_name]['id'])
        )
    return index_pairs


def setup_from_config(config):
    """Apply image / SimCC / keypoint globals from the parsed config."""
    global IMAGE_SIZE, KEYPOINT_IDS, KEYPOINT_INDEX, NUM_JOINTS, KEYPOINTS_BY_NAME
    global BBOX_PADDING, SIMCC_SPLIT_RATIO, SIMCC_SIGMA
    global LEFT_RIGHT_PAIRS, JOINT_LOSS_WEIGHTS
    global AUG_SCALE_MIN, AUG_SCALE_MAX, AUG_SHIFT_FRAC, AUG_ROTATION_DEG, AUG_COLOR_JITTER

    image_cfg: ImageConfig = config['image']
    head_cfg: HeadConfig = config['head']
    backbone_cfg: BackboneConfig = config['backbone']
    training_cfg: TrainingConfig = config['training']

    if (
        image_cfg.width % backbone_cfg.patch_size != 0
        or image_cfg.height % backbone_cfg.patch_size != 0
    ):
        raise ValueError(
            f'Image size {image_cfg.width}x{image_cfg.height} must be divisible by '
            f'backbone patch_size {backbone_cfg.patch_size}'
        )
    if head_cfg.in_channels != backbone_cfg.hidden_size:
        raise ValueError(
            f'head.in_channels ({head_cfg.in_channels}) must match '
            f'backbone.hidden_size ({backbone_cfg.hidden_size})'
        )

    IMAGE_SIZE = IVec2(image_cfg.width, image_cfg.height)
    BBOX_PADDING = image_cfg.bbox_padding
    AUG_SCALE_MIN = training_cfg.aug_scale_min
    AUG_SCALE_MAX = training_cfg.aug_scale_max
    AUG_SHIFT_FRAC = training_cfg.aug_shift_frac
    AUG_ROTATION_DEG = training_cfg.aug_rotation_deg
    AUG_COLOR_JITTER = training_cfg.aug_color_jitter
    if AUG_SCALE_MIN <= 0 or AUG_SCALE_MAX < AUG_SCALE_MIN:
        raise ValueError(
            f'Invalid aug scale range: [{AUG_SCALE_MIN}, {AUG_SCALE_MAX}]'
        )
    if AUG_SHIFT_FRAC < 0:
        raise ValueError(f'aug_shift_frac must be >= 0, got {AUG_SHIFT_FRAC}')
    if AUG_ROTATION_DEG < 0:
        raise ValueError(f'aug_rotation_deg must be >= 0, got {AUG_ROTATION_DEG}')
    if AUG_COLOR_JITTER < 0 or AUG_COLOR_JITTER >= 1:
        raise ValueError(f'aug_color_jitter must be in [0, 1), got {AUG_COLOR_JITTER}')

    ordered = sorted(head_cfg.keypoints.items(), key=lambda item: item[1]['id'])
    local_ids = [meta['id'] for _, meta in ordered]
    if local_ids != list(range(len(local_ids))):
        raise ValueError(
            f'Keypoint local ids must be contiguous 0..K-1, got: {local_ids}'
        )
    KEYPOINTS_BY_NAME = head_cfg.keypoints
    KEYPOINT_IDS = [meta['standard_id'] for _, meta in ordered]
    KEYPOINT_INDEX = {meta['standard_id']: meta['id'] for _, meta in ordered}
    NUM_JOINTS = len(ordered)

    if head_cfg.type == 'simcc':
        simcc = head_cfg.custom
        SIMCC_SPLIT_RATIO = simcc.split_ratio
        SIMCC_SIGMA = simcc.gaussian_sigma
        expected_x = int(image_cfg.width * simcc.split_ratio)
        expected_y = int(image_cfg.height * simcc.split_ratio)
        if simcc.x_bins != expected_x or simcc.y_bins != expected_y:
            raise ValueError(
                f'SimCC bins must equal image_size * split_ratio '
                f'(expected x_bins={expected_x}, y_bins={expected_y}; '
                f'got x_bins={simcc.x_bins}, y_bins={simcc.y_bins})'
            )

    JOINT_LOSS_WEIGHTS = build_joint_loss_weights(
        training_cfg.joint_loss_weights, head_cfg.keypoints
    )
    LEFT_RIGHT_PAIRS = build_left_right_index_pairs(
        head_cfg.left_right_pairs, head_cfg.keypoints
    )


def compute_padded_crop(
    bbox,
    original_width,
    original_height,
    *,
    scale: float = 1.0,
    shift_x_frac: float = 0.0,
    shift_y_frac: float = 0.0,
):
    """Return (crop_x1, crop_y1, crop_width, crop_height) matching KeypointDataset.

    ``scale`` zooms the crop ( >1 = zoom out). ``shift_*_frac`` offsets the crop
    center by a fraction of the (pre-clamp) crop size.
    """
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
    crop_width = crop_width * scale
    crop_height = crop_height * scale
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    center_x = center_x + shift_x_frac * crop_width
    center_y = center_y + shift_y_frac * crop_height
    crop_x1 = max(0.0, min(center_x - crop_width / 2, original_width - crop_width))
    crop_y1 = max(0.0, min(center_y - crop_height / 2, original_height - crop_height))
    return crop_x1, crop_y1, crop_width, crop_height


def transform_keypoints_to_crop(keypoints, crop_x1, crop_y1, crop_width, crop_height):
    transformed = keypoints.copy()
    transformed[:, 0] = (transformed[:, 0] - crop_x1) * IMAGE_SIZE.x / crop_width
    transformed[:, 1] = (transformed[:, 1] - crop_y1) * IMAGE_SIZE.y / crop_height
    return transformed


def clip_id_from_image_path(image_path) -> str:
    name = Path(image_path).name
    if name.startswith('task_') and '_frame_' in name:
        return name.split('_frame_', 1)[0]
    return str(Path(image_path).parent)


def split_image_paths(image_paths, train_ratio, seed):
    paths = list(image_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if len(paths) == 1:
        return paths, []
    if not paths:
        return [], []
    split_index = max(1, min(len(paths) - 1, round(len(paths) * train_ratio)))
    return paths[:split_index], paths[split_index:]


def split_samples_by_image(samples, train_ratio, seed, split_mode='frame'):
    """Split person-samples into train/test.

    Modes:
      - frame: shuffle unique images globally (legacy).
      - stratified_clip: within each task/clip, split images by ratio (every clip appears in train).
      - clip: assign whole clips to train or test.
    """
    image_groups: dict = {}
    for sample in samples:
        image_groups.setdefault(sample[0], []).append(sample)

    split_mode = (split_mode or 'frame').strip().lower()
    if split_mode == 'frame':
        train_paths, test_paths = split_image_paths(list(image_groups), train_ratio, seed)
        if not train_paths or not test_paths:
            raise ValueError('split_mode=frame needs at least 2 images to form train and test sets')
    elif split_mode == 'stratified_clip':
        clip_to_images: dict[str, list] = {}
        for path in image_groups:
            clip_to_images.setdefault(clip_id_from_image_path(path), []).append(path)
        train_paths, test_paths = [], []
        for clip_id, paths in sorted(clip_to_images.items()):
            clip_seed = random.Random(f'{seed}:{clip_id}').randint(0, 2**31 - 1)
            clip_train, clip_test = split_image_paths(paths, train_ratio, clip_seed)
            if not clip_test and len(paths) >= 2:
                clip_test.append(clip_train.pop())
            if not clip_train and clip_test:
                # Prefer keeping rare single-frame clips in train so they still teach.
                clip_train.append(clip_test.pop())
            train_paths.extend(clip_train)
            test_paths.extend(clip_test)
        if not train_paths or not test_paths:
            raise ValueError(
                'split_mode=stratified_clip could not form both train and test sets; '
                'need more frames per clip or a different split_mode'
            )
    elif split_mode == 'clip':
        clip_to_images = {}
        for path in image_groups:
            clip_to_images.setdefault(clip_id_from_image_path(path), []).append(path)
        clip_ids = list(clip_to_images)
        if len(clip_ids) < 2:
            raise ValueError('split_mode=clip requires at least 2 clips')
        rng = random.Random(seed)
        rng.shuffle(clip_ids)
        split_index = max(1, min(len(clip_ids) - 1, round(len(clip_ids) * train_ratio)))
        train_clips = set(clip_ids[:split_index])
        test_clips = set(clip_ids[split_index:])
        train_paths = [path for clip_id in train_clips for path in clip_to_images[clip_id]]
        test_paths = [path for clip_id in test_clips for path in clip_to_images[clip_id]]
    else:
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; expected 'frame', 'stratified_clip', or 'clip'"
        )

    train_samples = [sample for path in train_paths for sample in image_groups[path]]
    test_samples = [sample for path in test_paths for sample in image_groups[path]]
    return train_samples, test_samples, train_paths, test_paths


def make_targets(keypoints, device):
    targets = [
        generate_simcc_labels(sample.numpy(), IMAGE_SIZE, split_ratio=SIMCC_SPLIT_RATIO, sigma=SIMCC_SIGMA)
        for sample in keypoints
    ]
    target_x, target_y, weights = zip(*targets)
    return (
        torch.stack(target_x).to(device),
        torch.stack(target_y).to(device),
        torch.stack(weights).to(device) * JOINT_LOSS_WEIGHTS.to(device),
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


def load_keypoint_format(format_name) -> keypoint_mapping:
    mapping_path = Path('mapping')
    format_map = mapping_path / f'{format_name}.yaml'

    if not format_map.exists():
        raise ValueError(f'Keypoint format mapping file not found: {format_map}')
    with open(format_map, 'r') as f:
        return keypoint_mapping(**yaml.safe_load(f))

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
    keypoint_format = load_keypoint_format(head_cfg.keypoint_format_name)
    head_cfg.keypoints = keypoint_format.keypoints
    head_cfg.left_right_pairs = keypoint_format.left_right_pairs

    if head_cfg.type != 'simcc':
        raise ValueError(f'Unsupported head type: {head_cfg.type}')

    match head_cfg.name:
        case 'depthwise_simcc':
            head_cfg.custom = SimCCConfig(**head_cfg.custom)
        case 'joint_query_simcc' | 'joint_query_self_attn_simcc':
            head_cfg.custom = JointQuerySimCCConfig(**head_cfg.custom)
        case _:
            raise ValueError(
                f'Unsupported head name: {head_cfg.name}; '
                f"expected 'depthwise_simcc', 'joint_query_simcc', "
                f"or 'joint_query_self_attn_simcc'"
            )

    return {
        'dataset': dataset_cfg,
        'image': image_cfg,
        'backbone': backbone_cfg,
        'head': head_cfg,
        'training': training_cfg,
    }


def build_skeleton_connections(keypoints_by_name: dict) -> list[tuple[int, int]]:
    name_links = (
        ('shoulder_l', 'elbow_l'),
        ('elbow_l', 'wrist_l'),
        ('shoulder_r', 'elbow_r'),
        ('elbow_r', 'wrist_r'),
        ('shoulder_l', 'shoulder_r'),
        ('shoulder_l', 'hip_l'),
        ('shoulder_r', 'hip_r'),
        ('hip_l', 'hip_r'),
        ('hip_l', 'knee_l'),
        ('knee_l', 'ankle_l'),
        ('hip_r', 'knee_r'),
        ('knee_r', 'ankle_r'),
        ('ankle_l', 'toe_b_l'),
        ('ankle_l', 'toe_s_l'),
        ('ankle_l', 'heel_l'),
        ('ankle_l', 'heel_spike_l'),
        ('ankle_r', 'toe_b_r'),
        ('ankle_r', 'toe_s_r'),
        ('ankle_r', 'heel_r'),
        ('ankle_r', 'heel_spike_r'),
    )
    connections = []
    for left_name, right_name in name_links:
        if left_name not in keypoints_by_name or right_name not in keypoints_by_name:
            continue
        connections.append(
            (
                keypoints_by_name[left_name]['standard_id'],
                keypoints_by_name[right_name]['standard_id'],
            )
        )
    return connections


def per_sample_simcc_loss(pred_x, pred_y, target_x, target_y, weights):
    """Return per-sample SimCC KL loss with shape [B]."""
    log_pred_x = torch.nn.functional.log_softmax(pred_x, dim=-1)
    log_pred_y = torch.nn.functional.log_softmax(pred_y, dim=-1)
    kl = torch.nn.functional.kl_div(log_pred_x, target_x, reduction='none').sum(dim=-1)
    kl = kl + torch.nn.functional.kl_div(log_pred_y, target_y, reduction='none').sum(dim=-1)
    weighted = kl * weights
    return weighted.sum(dim=-1) / (weights.sum(dim=-1) + 1e-6)


@torch.inference_mode()
def rank_test_samples_by_loss(model, samples, device, batch_size):
    """Score every test person-sample by SimCC loss; highest loss first."""
    model.eval()
    ranked = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        pixels_batch = []
        keypoints_batch = []

        for image_path, keypoints, bbox in batch:
            image = Image.open(image_path).convert('RGB')
            crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
                bbox, image.width, image.height
            )
            crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
            crop = crop.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
            pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
            crop_keypoints = transform_keypoints_to_crop(
                keypoints, crop_x1, crop_y1, crop_width, crop_height
            )
            pixels_batch.append(pixels)
            keypoints_batch.append(torch.from_numpy(crop_keypoints))

        pixels_tensor = torch.stack(pixels_batch).to(device)
        keypoints_tensor = torch.stack(keypoints_batch)
        target_x, target_y, weights = make_targets(keypoints_tensor, device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y = model(pixels_tensor)
            sample_losses = per_sample_simcc_loss(pred_x, pred_y, target_x, target_y, weights)

        coordinates = torch.stack(
            (
                pred_x.argmax(dim=-1).float() / SIMCC_SPLIT_RATIO,
                pred_y.argmax(dim=-1).float() / SIMCC_SPLIT_RATIO,
            ),
            dim=-1,
        ).cpu().numpy()

        for offset, loss in enumerate(sample_losses.detach().cpu().tolist()):
            sample_index = start + offset
            ranked.append(
                {
                    'index': sample_index,
                    'loss': float(loss),
                    'image_path': Path(batch[offset][0]),
                    'keypoints': batch[offset][1],
                    'bbox': batch[offset][2],
                    'coordinates': coordinates[offset],
                }
            )

    ranked.sort(key=lambda item: item['loss'], reverse=True)
    return ranked


def draw_prediction_overlay(image_path, bbox, coordinates, output_path, loss=None):
    image = Image.open(image_path).convert('RGB')
    crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
        bbox, image.width, image.height
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height),
        outline='yellow',
        width=1,
    )

    points = []
    for x, y in coordinates:
        points.append(
            (
                crop_x1 + float(x) * crop_width / IMAGE_SIZE.x,
                crop_y1 + float(y) * crop_height / IMAGE_SIZE.y,
            )
        )

    point_by_id = dict(zip(KEYPOINT_IDS, points))
    for first_id, second_id in build_skeleton_connections(KEYPOINTS_BY_NAME):
        first = point_by_id.get(first_id)
        second = point_by_id.get(second_id)
        if first is not None and second is not None:
            draw.line((first, second), fill='lime', width=1)

    for index, (x, y) in enumerate(points):
        radius = 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='white')
        draw.text(
            (x + 4, y - 4),
            str(KEYPOINT_IDS[index]),
            fill='white',
            stroke_width=1,
            stroke_fill='black',
        )

    if loss is not None:
        draw.text(
            (crop_x1 + 4, max(0.0, crop_y1 - 18)),
            f'loss={loss:.4f}',
            fill='yellow',
            stroke_width=1,
            stroke_fill='black',
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def export_worst_test_overlays(model, checkpoint_path, test_samples, output_dir, top_n, device, batch_size):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    ranked = rank_test_samples_by_loss(model, test_samples, device, batch_size)
    worst = ranked[: max(0, min(top_n, len(ranked)))]

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f'checkpoint: {checkpoint_path}',
        f'test_samples: {len(test_samples)}',
        f'top_n: {len(worst)}',
        '',
    ]
    for rank, item in enumerate(worst, start=1):
        stem = item['image_path'].stem
        output_path = output_dir / f'{rank:02d}_loss_{item["loss"]:.4f}_{stem}.jpg'
        draw_prediction_overlay(
            item['image_path'],
            item['bbox'],
            item['coordinates'],
            output_path,
            loss=item['loss'],
        )
        summary_lines.append(
            f'{rank:02d}\tloss={item["loss"]:.6f}\t{item["image_path"]}\tbbox={item["bbox"]}'
        )
        print(f'  [{output_dir.name}] {rank:02d}/{len(worst)} loss={item["loss"]:.4f} -> {output_path.name}')

    (output_dir / 'worst_summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    return worst


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

    log_tee = StdoutCsvTee(sys.stdout, run_dir / 'log.csv')
    sys.stdout = log_tee
    try:
        _run_training(args, config, run_dir, run_config_path)
    finally:
        sys.stdout = log_tee.stream
        log_tee.close()


def _run_training(args, config, run_dir, run_config_path):
    dataset_cfg: DatasetConfig = config['dataset']
    training_cfg: TrainingConfig = config['training']
    head_cfg: HeadConfig = config['head']
    backbone_cfg: BackboneConfig = config['backbone']

    setup_from_config(config)
    set_seed(training_cfg.seed)

    annotation_file = get_annotation_file_path(dataset_cfg.annotation_version)
    root = ET.parse(annotation_file).getroot()
    task_ids = [video.get('id') for video in root.findall('./project/videos/video')]
    full_dataset = KeypointDataset(annotation_file, set(task_ids))
    train_samples, test_samples, train_image_paths, test_image_paths = split_samples_by_image(
        full_dataset.samples,
        training_cfg.train_test_split,
        training_cfg.seed,
        split_mode=training_cfg.split_mode,
    )
    split_index = len(train_image_paths)
    image_paths = train_image_paths + test_image_paths
    train_clips = sorted({clip_id_from_image_path(path) for path in train_image_paths})
    test_clips = sorted({clip_id_from_image_path(path) for path in test_image_paths})
    split_label = f'{training_cfg.train_test_split:.2f} by {training_cfg.split_mode}'
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

    backbone_hf_ids = {
        'dino_v2_base': 'facebook/dinov2-base',
        'dino_v3_base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    }

    match backbone_cfg.name:
        case 'dino_v2_base' | 'dino_v3_base':
            match head_cfg.type:
                case 'simcc':
                    head_kwargs = {}
                    if head_cfg.name in ('joint_query_simcc', 'joint_query_self_attn_simcc'):
                        head_kwargs = {
                            'num_heads': head_cfg.custom.num_heads,
                            'num_layers': head_cfg.custom.num_layers,
                            'dropout': head_cfg.custom.dropout,
                        }
                    model = DinoCC(
                        NUM_JOINTS,
                        IMAGE_SIZE,
                        freeze_backbone=True,
                        split_ratio=SIMCC_SPLIT_RATIO,
                        neck_dim=head_cfg.out_channels,
                        backbone_name=backbone_hf_ids[backbone_cfg.name],
                        head_name=head_cfg.name,
                        head_kwargs=head_kwargs,
                    ).to(device)
                    criterion = SimCCLoss().to(device)
                case _:
                    raise ValueError(f'Unsupported head type for backbone {backbone_cfg.name}: {head_cfg.type}')
        case _:
            raise ValueError(f'Unsupported backbone type: {backbone_cfg.name}')

    head_checkpoint_meta = {
        'head_name': head_cfg.name,
        'head_kwargs': (
            head_kwargs
            if head_cfg.name in ('joint_query_simcc', 'joint_query_self_attn_simcc')
            else {}
        ),
        'neck_dim': head_cfg.out_channels,
        'split_ratio': SIMCC_SPLIT_RATIO,
    }


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
        f'Training on {device}; split_mode={training_cfg.split_mode}; '
        f'train images={split_index}/{len(image_paths)}, '
        f'test images={len(image_paths) - split_index}/{len(image_paths)}; '
        f'train clips={train_clips}; test clips={test_clips}'
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
                    'joint_loss_weights': training_cfg.joint_loss_weights,
                    'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
                    'backbone_name': backbone_hf_ids[backbone_cfg.name],
                    'head_name': head_checkpoint_meta['head_name'],
                    'head_kwargs': head_checkpoint_meta['head_kwargs'],
                    'neck_dim': head_checkpoint_meta['neck_dim'],
                    'split_ratio': head_checkpoint_meta['split_ratio'],
                    'config_path': str(run_config_path),
                    'seed': training_cfg.seed,
                    'split': split_label,
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
            'joint_loss_weights': training_cfg.joint_loss_weights,
            'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
            'backbone_name': backbone_hf_ids[backbone_cfg.name],
            'head_name': head_checkpoint_meta['head_name'],
            'head_kwargs': head_checkpoint_meta['head_kwargs'],
            'neck_dim': head_checkpoint_meta['neck_dim'],
            'split_ratio': head_checkpoint_meta['split_ratio'],
            'config_path': str(run_config_path),
            'seed': training_cfg.seed,
            'split': split_label,
            'epoch': epoch,
            'test_loss': val_loss,
        },
        final_path,
    )
    print(f'Saved best checkpoint to {best_path}')
    print(f'Saved final checkpoint to {final_path}')

    if training_cfg.worst_test_overlays > 0:
        print(
            f'Exporting top {training_cfg.worst_test_overlays} worst test overlays '
            f'for best and final checkpoints...'
        )
        for checkpoint_name, checkpoint_path in (('best', best_path), ('final', final_path)):
            if not checkpoint_path.exists():
                print(f'Skipping {checkpoint_name}: missing {checkpoint_path}')
                continue
            export_worst_test_overlays(
                model=model,
                checkpoint_path=checkpoint_path,
                test_samples=test_samples,
                output_dir=run_dir / checkpoint_name,
                top_n=training_cfg.worst_test_overlays,
                device=device,
                batch_size=training_cfg.batch_size,
            )


if __name__ == '__main__':
    main()