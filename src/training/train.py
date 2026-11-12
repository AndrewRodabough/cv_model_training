from argparse import ArgumentParser
import gc
import random
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.training.dino_cc import (
    DinoCC,
    SimCCLoss,
    build_lr_shared_pair_bias_index,
    generate_simcc_labels,
)
from src.training.coco_wholebody_dataset import CocoWholeBodyConfig
from src.config.image_config import ImageConfig
from src.config.backbone_config import BackboneConfig
from src.config.head_config import HeadConfig
from src.config.simcc_config import SimCCConfig
from src.config.joint_query_simcc_config import JointQuerySimCCConfig
from src.config.couple_joint_query_simcc_config import CoupleJointQuerySimCCConfig
from src.config.training_config import TrainingConfig
from src.config.dataset_config import DatasetConfig
from src.config.initialization_config import InitializationConfig, PairBiasInitConfig
from src.config.keypoint_mapping_config import keypoint_mapping
from src.util.stdout_csv_tee import StdoutCsvTee
from src.util.vec import IVec2
from src.util.dir import POSE_ESTIMATOR_RUN_DIR, DEFAULT_POSE_ESTIMATOR_CONFIG_PATH
from src.util.get_version import get_pose_estimator_version_folder
from src.training import run_artifacts

from src.model_review.metrics import (
    align_keypoints_to_num_joints,
    body_part_oks_fields,
    compute_oks,
    decode_simcc,
    format_metrics_summary,
    joint_score_for_visibility,
    part_mean_oks_from_metrics,
    person_scale_in_crop,
    resolve_body_part_indices,
    summarize_pose_metrics,
    write_metrics_json,
)
from src.training.coco_wholebody_dataset import (
    build_coco_wholebody_samples,
    mean_labeled_joints,
    mix_dance_with_coco,
    target_coco_count,
)
from src.training.temporal_dataset import (
    TemporalWindowDataset,
    build_contiguous_windows,
    split_windows_stratified,
    summarize_window_sources,
)

IMAGE_SIZE = IVec2(384, 512)
KEYPOINT_IDS = list(range(6, 12)) + list(range(15, 29))
KEYPOINT_INDEX = {point_id: index for index, point_id in enumerate(KEYPOINT_IDS)}
NUM_JOINTS = len(KEYPOINT_IDS)
PERSON_JOINTS = NUM_JOINTS
KEYPOINTS_BY_NAME: dict = {}
BBOX_PADDING = 0.05
SIMCC_SPLIT_RATIO = 2.0
SIMCC_SIGMA = 6.0
SIMCC_SIGMA_OCCLUDED: float | None = None
OCCLUDED_JOINT_WEIGHT = 0.35
LEFT_RIGHT_PAIRS: list[tuple[int, int]] = []
LEFT_RIGHT_PAIR_IS_LOWER: list[bool] = []
JOINT_LOSS_WEIGHTS = torch.empty(0)
OKS_SIGMAS = torch.empty(0)
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
AUG_SCALE_MIN = 0.85
AUG_SCALE_MAX = 1.25
AUG_SHIFT_FRAC = 0.1
AUG_ROTATION_DEG = 12.0
AUG_COLOR_JITTER = 0.25
USE_ROLE = False
ROLE_DROPOUT = 0.1
ROLE_DIM = 2
COUPLE_MODE = False
COUPLE_MIN_IOU = 0.3
COUPLE_HEAD_NAME = 'couple_joint_query_self_attn_simcc'
TEMPORAL_HEAD_NAME = 'temporal_joint_query_self_attn_simcc'
WINDOW_SIZE = 0
WINDOW_STRIDE = 1
MIN_TEMPORAL_WINDOWS = 16
TEMPORAL_MODE = False
MAX_PARTNER_IOU: float | None = None
MIN_PARTNER_IOU: float | None = None
USE_SPATIAL_PROMPT = False
DEEP_SUPERVISION = False
DEEP_SUPERVISION_AUX_WEIGHT = 0.5
XY_CONSISTENCY_LOSS_WEIGHT = 0.0
BONE_LOSS_WEIGHT = 0.0
LR_SWAP_LOSS_WEIGHT = 0.0
LR_SWAP_MARGIN = 0.02
LR_SWAP_SCALE_FRAC = 1.0
LR_SWAP_FOOT_PAIR_WEIGHT = 1.0
LR_SWAP_HARD_MIX = 0.0
CHIRALITY_LOSS_WEIGHT = 0.0
CHIRALITY_MIN_SEP = 0.03
CHIRALITY_FOOT_PAIR_WEIGHT = 1.0
OKS_LOSS_WEIGHT = 0.0
PRIORITY_PART_WEIGHTS: dict[str, float] = {}
BONE_PAIRS: torch.Tensor = torch.empty(0, 2, dtype=torch.long)

# Directed skeleton edges used for bone-vector loss and overlays (name → name).
SKELETON_BONE_NAME_LINKS = (
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
    ('heel_l', 'toe_b_l'),
    ('heel_l', 'toe_s_l'),
    ('ankle_l', 'heel_l'),
    ('heel_l', 'heel_spike_l'),
    ('toe_b_l', 'toe_s_l'),
    ('heel_r', 'toe_b_r'),
    ('heel_r', 'toe_s_r'),
    ('ankle_r', 'heel_r'),
    ('heel_r', 'heel_spike_r'),
    ('toe_b_r', 'toe_s_r'),
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_role(role: str | None) -> torch.Tensor:
    """Encode track role as [is_lead, is_follow] bits. Unknown → [0, 0]."""
    bits = torch.zeros(ROLE_DIM, dtype=torch.float32)
    if role is None:
        return bits
    normalized = role.strip().lower()
    if normalized == 'lead':
        bits[0] = 1.0
    elif normalized == 'follow':
        bits[1] = 1.0
    elif normalized:
        raise ValueError(f'Unknown role {role!r}; expected lead, follow, or empty')
    return bits


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def union_bbox(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def parse_person(person):
    keypoints = np.zeros((PERSON_JOINTS, 3), dtype=np.float32)
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
    role = person.get('role')
    return keypoints, bbox, role


def horizontally_flip_keypoints(keypoints):
    flipped = keypoints.copy()
    flipped[:, 0] = IMAGE_SIZE.x - 1 - flipped[:, 0]
    for left_index, right_index in LEFT_RIGHT_PAIRS:
        flipped[[left_index, right_index]] = flipped[[right_index, left_index]]
    return flipped


def horizontally_flip_couple_keypoints(keypoints):
    """Flip stacked [lead|follow] keypoints; banks stay lead then follow."""
    lead = horizontally_flip_keypoints(keypoints[:PERSON_JOINTS])
    follow = horizontally_flip_keypoints(keypoints[PERSON_JOINTS:])
    return np.concatenate([lead, follow], axis=0)


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
    def __init__(
        self,
        annotation_file: Path,
        task_ids,
        training=False,
        repeats=1,
        samples=None,
        coco_paths: set[str] | None = None,
    ):
        self.image_root = annotation_file.parent
        self.training = training
        self.repeats = repeats if training else 1
        self.coco_paths = set(coco_paths or ())
        if samples is None:
            samples = build_person_samples(
                annotation_file,
                task_ids,
                max_partner_iou=MAX_PARTNER_IOU,
                min_partner_iou=MIN_PARTNER_IOU,
            )

        self.samples = list(samples)

        if not self.samples:
            raise ValueError(f'No samples found for task IDs: {sorted(task_ids)}')

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        sample = self.samples[index % len(self.samples)]
        image_path, keypoints, bbox, role = sample[:4]
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
        role_bits = encode_role(role)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        spatial = crop_normalized_bbox_center(
            cx, cy, crop_x1, crop_y1, crop_width, crop_height
        )
        bbox_scale = person_scale_in_crop(
            bbox,
            crop_width,
            crop_height,
            IMAGE_SIZE.x,
            IMAGE_SIZE.y,
        )
        is_coco = 1.0 if str(Path(image_path).resolve()) in self.coco_paths else 0.0
        return (
            pixels,
            torch.from_numpy(keypoints),
            role_bits,
            spatial,
            torch.tensor(bbox_scale, dtype=torch.float32),
            torch.tensor(is_coco, dtype=torch.float32),
        )


def build_person_samples(
    annotation_file: Path,
    task_ids,
    max_partner_iou: float | None = None,
    min_partner_iou: float | None = None,
):
    """Build single-person samples; optionally filter by partner-overlap IoU.

    Each sample is
    ``(image_path, keypoints, bbox, role, task_id, frame_number, track_id)``.
    When ``max_partner_iou`` is set, keep a person only if their max IoU with any
    other person in the same frame is ``<= max_partner_iou`` (solo → IoU 0).
    When ``min_partner_iou`` is set, keep only if that max IoU is ``>= min_partner_iou``.
    """
    root = ET.parse(annotation_file).getroot()
    image_root = Path(annotation_file).parent
    samples = []
    n_total = 0
    n_dropped = 0
    for video in root.findall('./project/videos/video'):
        task_id = video.get('id')
        if task_id not in task_ids:
            continue
        for frame in video.findall('frame'):
            image_name = frame.get('image')
            if image_name is None:
                raise ValueError('Frame is missing its image attribute')
            image_path = image_root / image_name
            frame_number = frame.get('number')
            if frame_number is None:
                frame_number = _frame_index_from_image_path(image_path)
            else:
                frame_number = int(frame_number)
            people = []
            for person in frame.findall('person'):
                keypoints, bbox, role = parse_person(person)
                track_id = person.get('track_id') or ''
                if USE_ROLE and role is None:
                    raise ValueError(
                        f'Missing role on person in {image_path} '
                        f'(track_id={track_id})'
                    )
                people.append((keypoints, bbox, role, track_id))
            for index, (keypoints, bbox, role, track_id) in enumerate(people):
                n_total += 1
                if max_partner_iou is not None or min_partner_iou is not None:
                    partner_iou = 0.0
                    for other_i, (_, other_bbox, _, _) in enumerate(people):
                        if other_i == index:
                            continue
                        partner_iou = max(partner_iou, bbox_iou(bbox, other_bbox))
                    if max_partner_iou is not None and partner_iou > max_partner_iou:
                        n_dropped += 1
                        continue
                    if min_partner_iou is not None and partner_iou < min_partner_iou:
                        n_dropped += 1
                        continue
                samples.append(
                    (image_path, keypoints, bbox, role, task_id, frame_number, track_id)
                )
    if not samples:
        raise ValueError(
            f'No person samples for task IDs {sorted(task_ids)}'
            + (
                f' after partner IoU filter '
                f'(max={max_partner_iou}, min={min_partner_iou}; '
                f'dropped {n_dropped}/{n_total})'
                if max_partner_iou is not None or min_partner_iou is not None
                else ''
            )
        )
    if max_partner_iou is not None or min_partner_iou is not None:
        print(
            f'Person IoU filter max_partner_iou={max_partner_iou} '
            f'min_partner_iou={min_partner_iou}: '
            f'kept {len(samples)}/{n_total}, dropped {n_dropped}'
        )
    return samples


def build_couple_samples(annotation_file: Path, task_ids, min_iou: float):
    """Build lead+follow pair samples with union bbox, filtered by IoU.

    Each sample is:
      (image_path, lead_kps, follow_kps, union_bbox, lead_bbox, follow_bbox, iou)
    """
    root = ET.parse(annotation_file).getroot()
    image_root = Path(annotation_file).parent
    samples = []
    for video in root.findall('./project/videos/video'):
        if video.get('id') not in task_ids:
            continue
        for frame in video.findall('frame'):
            image_name = frame.get('image')
            if image_name is None:
                raise ValueError('Frame is missing its image attribute')
            image_path = image_root / image_name
            leads = []
            follows = []
            for person in frame.findall('person'):
                keypoints, bbox, role = parse_person(person)
                if role is None:
                    continue
                normalized = role.strip().lower()
                if normalized == 'lead':
                    leads.append((keypoints, bbox))
                elif normalized == 'follow':
                    follows.append((keypoints, bbox))
            if not leads or not follows:
                continue
            best = None
            best_iou = -1.0
            for lead_kps, lead_bbox in leads:
                for follow_kps, follow_bbox in follows:
                    iou = bbox_iou(lead_bbox, follow_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best = (lead_kps, follow_kps, lead_bbox, follow_bbox)
            if best is None or best_iou < min_iou:
                continue
            lead_kps, follow_kps, lead_bbox, follow_bbox = best
            samples.append(
                (
                    image_path,
                    lead_kps,
                    follow_kps,
                    union_bbox(lead_bbox, follow_bbox),
                    lead_bbox,
                    follow_bbox,
                    float(best_iou),
                )
            )
    if not samples:
        raise ValueError(
            f'No couple samples with IoU >= {min_iou} for task IDs: {sorted(task_ids)}'
        )
    return samples


class CoupleKeypointDataset(Dataset):
    """Union-crop dataset for stacked [lead|follow] keypoints (2K joints)."""

    def __init__(self, annotation_file: Path, task_ids, training=False, repeats=1, samples=None):
        self.training = training
        self.repeats = repeats if training else 1
        if samples is None:
            samples = build_couple_samples(annotation_file, task_ids, COUPLE_MIN_IOU)
        self.samples = list(samples)
        if not self.samples:
            raise ValueError(f'No couple samples found for task IDs: {sorted(task_ids)}')

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        (
            image_path,
            lead_kps,
            follow_kps,
            union_box,
            _,
            _,
            _,
        ) = self.samples[index % len(self.samples)]
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
            union_box,
            original_width,
            original_height,
            scale=scale,
            shift_x_frac=shift_x_frac,
            shift_y_frac=shift_y_frac,
        )
        image = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
        image = image.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)

        lead = transform_keypoints_to_crop(
            lead_kps, crop_x1, crop_y1, crop_width, crop_height
        )
        follow = transform_keypoints_to_crop(
            follow_kps, crop_x1, crop_y1, crop_width, crop_height
        )
        keypoints = np.concatenate([lead, follow], axis=0)

        if self.training:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                keypoints = horizontally_flip_couple_keypoints(keypoints)
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
        # Dummy role zeros so the shared train loop can unpack (pixels, kps, role).
        role_bits = torch.zeros(ROLE_DIM, dtype=torch.float32)
        spatial = torch.zeros(2, dtype=torch.float32)
        return (
            pixels,
            torch.from_numpy(keypoints),
            role_bits,
            spatial,
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),  # is_coco (couples are dance-only)
        )


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


def build_foot_anchor_pairs(keypoints_by_name: dict) -> list[tuple[int, int]]:
    """Map each foot tip local id → same-side ankle local id for soft query anchoring."""
    pairs = []
    for name, meta in sorted(keypoints_by_name.items(), key=lambda item: item[1]['id']):
        if not (name.startswith('toe_') or name.startswith('heel')):
            continue
        if name.endswith('_l'):
            ankle_name = 'ankle_l'
        elif name.endswith('_r'):
            ankle_name = 'ankle_r'
        else:
            raise ValueError(f'Foot keypoint {name!r} missing _l/_r suffix')
        if ankle_name not in keypoints_by_name:
            raise ValueError(f'Cannot anchor {name!r}: missing {ankle_name!r}')
        pairs.append((int(meta['id']), int(keypoints_by_name[ankle_name]['id'])))
    if not pairs:
        raise ValueError('soft_foot_anchor enabled but no toe_/heel* keypoints found')
    return pairs


def build_bone_index_pairs(keypoints_by_name: dict) -> list[tuple[int, int]]:
    """Local-id directed bones for vector loss (parent → child)."""
    pairs = []
    for parent_name, child_name in SKELETON_BONE_NAME_LINKS:
        if parent_name not in keypoints_by_name or child_name not in keypoints_by_name:
            continue
        pairs.append(
            (
                int(keypoints_by_name[parent_name]['id']),
                int(keypoints_by_name[child_name]['id']),
            )
        )
    if not pairs:
        raise ValueError('No bone pairs resolved from keypoint format mapping')
    return pairs


def expand_pair_bias_joint_name(name: str, keypoints_by_name: dict) -> list[str]:
    """Resolve a joint name or L/R stem to concrete keypoint names."""
    if name in keypoints_by_name:
        return [name]
    left, right = f'{name}_l', f'{name}_r'
    if left in keypoints_by_name and right in keypoints_by_name:
        return [left, right]
    raise ValueError(
        f'Unknown pair-bias joint {name!r}: not in keypoint format and no '
        f'{left!r}/{right!r} stem expansion'
    )


def expand_pair_bias_edge(
    joint_a: str, joint_b: str, keypoints_by_name: dict
) -> list[tuple[str, str]]:
    """Expand config edge names into concrete undirected name pairs.

    Stems like ``knee``/``ankle`` expand to same-side pairs (l–l, r–r).
    A sided joint + center (``ear`` + ``glabella``) fans out to both sides.
    Exact names (``ear_l``, ``ear_r``) stay as a single edge.
    """
    forms_a = expand_pair_bias_joint_name(joint_a, keypoints_by_name)
    forms_b = expand_pair_bias_joint_name(joint_b, keypoints_by_name)
    if len(forms_a) == 1 and len(forms_b) == 1:
        return [(forms_a[0], forms_b[0])]
    if len(forms_a) == 2 and len(forms_b) == 2:
        return [(forms_a[0], forms_b[0]), (forms_a[1], forms_b[1])]
    if len(forms_a) == 2 and len(forms_b) == 1:
        return [(a, forms_b[0]) for a in forms_a]
    if len(forms_a) == 1 and len(forms_b) == 2:
        return [(forms_a[0], b) for b in forms_b]
    raise ValueError(
        f'Cannot expand pair-bias edge ({joint_a!r}, {joint_b!r}) with forms '
        f'{forms_a} / {forms_b}'
    )


def build_pair_bias_from_edges(
    pair_bias_cfg: PairBiasInitConfig,
    keypoints_by_name: dict,
    left_right_pairs: list | None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Resolve named edges → bone index pairs + per shared-type init values."""
    if not pair_bias_cfg.edges:
        raise ValueError('initialization.pair_bias.edges is empty')
    bone_pairs: list[tuple[int, int]] = []
    bone_inits: list[float] = []
    for edge in pair_bias_cfg.edges:
        init = (
            float(pair_bias_cfg.default_init)
            if edge.init is None
            else float(edge.init)
        )
        for name_a, name_b in expand_pair_bias_edge(
            edge.joints[0], edge.joints[1], keypoints_by_name
        ):
            bone_pairs.append(
                (
                    int(keypoints_by_name[name_a]['id']),
                    int(keypoints_by_name[name_b]['id']),
                )
            )
            bone_inits.append(init)

    lr_pairs = build_left_right_index_pairs(left_right_pairs, keypoints_by_name)
    num_joints = len(keypoints_by_name)
    bias_index = build_lr_shared_pair_bias_index(num_joints, bone_pairs, lr_pairs)
    num_types = int(bias_index.max().item()) + 1
    type_inits: list[float | None] = [None] * num_types
    for (i, j), init in zip(bone_pairs, bone_inits):
        tid = int(bias_index[i, j].item())
        if tid < 0:
            raise ValueError(f'pair-bias edge ({i}, {j}) missing from index')
        prev = type_inits[tid]
        if prev is not None and abs(prev - init) > 1e-6:
            raise ValueError(
                f'Conflicting pair_bias init for shared L/R edge type {tid}: '
                f'{prev} vs {init} (mirror edges must share one init)'
            )
        type_inits[tid] = init
    resolved = [
        float(pair_bias_cfg.default_init) if v is None else float(v) for v in type_inits
    ]
    return bone_pairs, resolved


def build_couple_foot_anchor_pairs(keypoints_by_name: dict) -> list[tuple[int, int]]:
    """Duplicate single-person foot anchors into lead and follow banks."""
    person_pairs = build_foot_anchor_pairs(keypoints_by_name)
    follow_offset = len(keypoints_by_name)
    return list(person_pairs) + [
        (foot + follow_offset, ankle + follow_offset) for foot, ankle in person_pairs
    ]


def build_joint_query_head_kwargs(
    head_cfg: HeadConfig,
    initialization: InitializationConfig | None = None,
) -> dict:
    custom = head_cfg.custom
    use_pair_bias = bool(getattr(custom, 'use_pair_bias', False))
    pair_cfg = None if initialization is None else initialization.pair_bias
    if pair_cfg is not None and pair_cfg.edges:
        use_pair_bias = True
    kwargs = {
        'num_heads': custom.num_heads,
        'num_layers': custom.num_layers,
        'dropout': custom.dropout,
        'soft_foot_anchor': custom.soft_foot_anchor,
        'soft_foot_anchor_init': custom.soft_foot_anchor_init,
        'per_layer_role_reinject': getattr(custom, 'per_layer_role_reinject', False),
        'use_spatial_prompt': getattr(custom, 'use_spatial_prompt', False),
        'deep_supervision': getattr(custom, 'deep_supervision', False),
        'use_joint_pos_embed': getattr(custom, 'use_joint_pos_embed', False),
        'use_pair_bias': use_pair_bias,
        'pair_bias_init': float(getattr(custom, 'pair_bias_init', 2.0)),
        'use_local_cross_attn': getattr(custom, 'use_local_cross_attn', False),
        'local_attn_sigma': float(getattr(custom, 'local_attn_sigma', 2.0)),
        'local_attn_strength': float(getattr(custom, 'local_attn_strength', 2.0)),
        'local_attn_center_mode': str(
            getattr(custom, 'local_attn_center_mode', 'pred')
        ),
        'local_attn_gt_mix': float(getattr(custom, 'local_attn_gt_mix', 0.7)),
    }
    if head_cfg.name == TEMPORAL_HEAD_NAME:
        kwargs['temporal_num_layers'] = int(getattr(custom, 'temporal_num_layers', 2))
        kwargs['temporal_num_heads'] = int(getattr(custom, 'temporal_num_heads', 8))
        kwargs['temporal_dropout'] = float(getattr(custom, 'temporal_dropout', 0.0))
        kwargs['window_size'] = int(getattr(custom, 'window_size', 5))
        kwargs['temporal_mode'] = str(
            getattr(custom, 'temporal_mode', 'peer_self_attn')
        )
    if head_cfg.name == COUPLE_HEAD_NAME:
        kwargs['person_joints'] = PERSON_JOINTS
        kwargs['use_role'] = False
        kwargs['role_film'] = False
        kwargs['use_pair_bias'] = False
        if custom.soft_foot_anchor:
            kwargs['foot_anchor_pairs'] = build_couple_foot_anchor_pairs(head_cfg.keypoints)
    else:
        kwargs['use_role'] = custom.use_role
        kwargs['role_film'] = custom.role_film
        if custom.soft_foot_anchor:
            kwargs['foot_anchor_pairs'] = build_foot_anchor_pairs(head_cfg.keypoints)
        if kwargs['use_pair_bias']:
            kwargs['pair_bias_lr_pairs'] = build_left_right_index_pairs(
                head_cfg.left_right_pairs, head_cfg.keypoints
            )
            if pair_cfg is not None and pair_cfg.edges:
                bones, type_inits = build_pair_bias_from_edges(
                    pair_cfg, head_cfg.keypoints, head_cfg.left_right_pairs
                )
                kwargs['pair_bias_bones'] = bones
                kwargs['pair_bias_type_inits'] = type_inits
                kwargs['pair_bias_init'] = float(pair_cfg.default_init)
            else:
                kwargs['pair_bias_bones'] = build_bone_index_pairs(head_cfg.keypoints)
    return kwargs


def build_depthwise_head_kwargs(head_cfg: HeadConfig) -> dict:
    return {
        'use_role': head_cfg.custom.use_role,
        'role_film': head_cfg.custom.role_film,
    }


def build_oks_sigmas(oks_sigmas: dict[str, float] | None, keypoints_by_name: dict) -> torch.Tensor:
    """Build [K] OKS sigma tensor ordered by local joint id."""
    ordered = sorted(keypoints_by_name.items(), key=lambda item: item[1]['id'])
    if not oks_sigmas:
        raise ValueError(
            'oks_sigmas missing from keypoint format mapping; required when compute_ap is enabled'
        )
    missing = [name for name, _ in ordered if name not in oks_sigmas]
    if missing:
        raise ValueError(f'oks_sigmas missing joints: {missing}')
    return torch.tensor([float(oks_sigmas[name]) for name, _ in ordered], dtype=torch.float32)


def setup_from_config(config):
    """Apply image / SimCC / keypoint globals from the parsed config."""
    global IMAGE_SIZE, KEYPOINT_IDS, KEYPOINT_INDEX, NUM_JOINTS, PERSON_JOINTS, KEYPOINTS_BY_NAME
    global BBOX_PADDING, SIMCC_SPLIT_RATIO, SIMCC_SIGMA, SIMCC_SIGMA_OCCLUDED
    global OCCLUDED_JOINT_WEIGHT
    global LEFT_RIGHT_PAIRS, LEFT_RIGHT_PAIR_IS_LOWER, JOINT_LOSS_WEIGHTS, OKS_SIGMAS
    global AUG_SCALE_MIN, AUG_SCALE_MAX, AUG_SHIFT_FRAC, AUG_ROTATION_DEG, AUG_COLOR_JITTER
    global USE_ROLE, ROLE_DROPOUT, COUPLE_MODE, COUPLE_MIN_IOU
    global MAX_PARTNER_IOU, MIN_PARTNER_IOU
    global USE_SPATIAL_PROMPT, DEEP_SUPERVISION, DEEP_SUPERVISION_AUX_WEIGHT
    global XY_CONSISTENCY_LOSS_WEIGHT, BONE_LOSS_WEIGHT, LR_SWAP_LOSS_WEIGHT, LR_SWAP_MARGIN
    global LR_SWAP_SCALE_FRAC, LR_SWAP_FOOT_PAIR_WEIGHT, LR_SWAP_HARD_MIX
    global CHIRALITY_LOSS_WEIGHT, CHIRALITY_MIN_SEP, CHIRALITY_FOOT_PAIR_WEIGHT
    global OKS_LOSS_WEIGHT
    global PRIORITY_PART_WEIGHTS, BONE_PAIRS
    global WINDOW_SIZE, WINDOW_STRIDE, MIN_TEMPORAL_WINDOWS, TEMPORAL_MODE

    image_cfg: ImageConfig = config['image']
    head_cfg: HeadConfig = config['head']
    backbone_cfg: BackboneConfig = config['backbone']
    training_cfg: TrainingConfig = config['training']

    WINDOW_SIZE = int(getattr(training_cfg, 'window_size', 0) or 0)
    WINDOW_STRIDE = int(getattr(training_cfg, 'window_stride', 1) or 1)
    MIN_TEMPORAL_WINDOWS = int(getattr(training_cfg, 'min_temporal_windows', 16) or 16)
    TEMPORAL_MODE = head_cfg.name == TEMPORAL_HEAD_NAME or WINDOW_SIZE > 1
    if TEMPORAL_MODE:
        if head_cfg.name != TEMPORAL_HEAD_NAME:
            raise ValueError(
                f'Temporal window_size={WINDOW_SIZE} requires head.name='
                f'{TEMPORAL_HEAD_NAME!r}, got {head_cfg.name!r}'
            )
        if WINDOW_SIZE < 2:
            WINDOW_SIZE = int(getattr(head_cfg.custom, 'window_size', 5))
        if WINDOW_SIZE < 2:
            raise ValueError(f'Temporal training requires window_size >= 2, got {WINDOW_SIZE}')
        if WINDOW_STRIDE < 1:
            raise ValueError(f'window_stride must be >= 1, got {WINDOW_STRIDE}')
        # Keep head + training window sizes aligned for checkpoint meta / time embed.
        head_cfg.custom.window_size = WINDOW_SIZE
        if getattr(config.get('dataset'), 'coco_wholebody', None) is not None:
            coco_cfg = config['dataset'].coco_wholebody
            if coco_cfg is not None and coco_cfg.enabled:
                raise ValueError(
                    'Temporal training does not support coco_wholebody mixing; '
                    'disable dataset.coco_wholebody.enabled'
                )

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

    COUPLE_MODE = head_cfg.name == COUPLE_HEAD_NAME
    if COUPLE_MODE:
        USE_ROLE = False
        ROLE_DROPOUT = 0.0
        COUPLE_MIN_IOU = float(getattr(head_cfg.custom, 'couple_min_iou', 0.3))
        MAX_PARTNER_IOU = None
        MIN_PARTNER_IOU = None
        if not 0.0 <= COUPLE_MIN_IOU <= 1.0:
            raise ValueError(f'couple_min_iou must be in [0, 1], got {COUPLE_MIN_IOU}')
    else:
        USE_ROLE = bool(getattr(head_cfg.custom, 'use_role', False))
        ROLE_DROPOUT = float(getattr(head_cfg.custom, 'role_dropout', 0.1))
        COUPLE_MIN_IOU = 0.3
        if not 0.0 <= ROLE_DROPOUT <= 1.0:
            raise ValueError(f'role_dropout must be in [0, 1], got {ROLE_DROPOUT}')
        max_partner = getattr(training_cfg, 'max_partner_iou', None)
        if max_partner is None:
            MAX_PARTNER_IOU = None
        else:
            MAX_PARTNER_IOU = float(max_partner)
            if not 0.0 <= MAX_PARTNER_IOU <= 1.0:
                raise ValueError(
                    f'max_partner_iou must be in [0, 1], got {MAX_PARTNER_IOU}'
                )
        min_partner = getattr(training_cfg, 'min_partner_iou', None)
        if min_partner is None:
            MIN_PARTNER_IOU = None
        else:
            MIN_PARTNER_IOU = float(min_partner)
            if not 0.0 <= MIN_PARTNER_IOU <= 1.0:
                raise ValueError(
                    f'min_partner_iou must be in [0, 1], got {MIN_PARTNER_IOU}'
                )
        if (
            MAX_PARTNER_IOU is not None
            and MIN_PARTNER_IOU is not None
            and MIN_PARTNER_IOU > MAX_PARTNER_IOU
        ):
            raise ValueError(
                f'min_partner_iou ({MIN_PARTNER_IOU}) cannot exceed '
                f'max_partner_iou ({MAX_PARTNER_IOU})'
            )
        custom = head_cfg.custom
        USE_SPATIAL_PROMPT = bool(getattr(custom, 'use_spatial_prompt', False))
        DEEP_SUPERVISION = bool(getattr(custom, 'deep_supervision', False))
        DEEP_SUPERVISION_AUX_WEIGHT = float(
            getattr(custom, 'deep_supervision_aux_weight', 0.5)
        )
        XY_CONSISTENCY_LOSS_WEIGHT = float(
            getattr(custom, 'xy_consistency_loss_weight', 0.0)
        )
        if DEEP_SUPERVISION_AUX_WEIGHT < 0:
            raise ValueError(
                f'deep_supervision_aux_weight must be >= 0, got {DEEP_SUPERVISION_AUX_WEIGHT}'
            )
        if XY_CONSISTENCY_LOSS_WEIGHT < 0:
            raise ValueError(
                f'xy_consistency_loss_weight must be >= 0, got {XY_CONSISTENCY_LOSS_WEIGHT}'
            )

    if COUPLE_MODE:
        USE_SPATIAL_PROMPT = False
        DEEP_SUPERVISION = False
        XY_CONSISTENCY_LOSS_WEIGHT = 0.0

    BONE_LOSS_WEIGHT = float(getattr(training_cfg, 'bone_loss_weight', 0.0))
    if BONE_LOSS_WEIGHT < 0:
        raise ValueError(f'bone_loss_weight must be >= 0, got {BONE_LOSS_WEIGHT}')
    LR_SWAP_LOSS_WEIGHT = float(getattr(training_cfg, 'lr_swap_loss_weight', 0.0))
    if LR_SWAP_LOSS_WEIGHT < 0:
        raise ValueError(f'lr_swap_loss_weight must be >= 0, got {LR_SWAP_LOSS_WEIGHT}')
    LR_SWAP_MARGIN = float(getattr(training_cfg, 'lr_swap_margin', 0.02))
    if LR_SWAP_MARGIN < 0:
        raise ValueError(f'lr_swap_margin must be >= 0, got {LR_SWAP_MARGIN}')
    # Smaller denom → stronger hinge. 1.0 = full image diagonal (old behavior).
    LR_SWAP_SCALE_FRAC = float(getattr(training_cfg, 'lr_swap_scale_frac', 1.0))
    if not 0.01 <= LR_SWAP_SCALE_FRAC <= 1.0:
        raise ValueError(
            f'lr_swap_scale_frac must be in [0.01, 1], got {LR_SWAP_SCALE_FRAC}'
        )
    LR_SWAP_FOOT_PAIR_WEIGHT = float(getattr(training_cfg, 'lr_swap_foot_pair_weight', 1.0))
    if LR_SWAP_FOOT_PAIR_WEIGHT < 0:
        raise ValueError(
            f'lr_swap_foot_pair_weight must be >= 0, got {LR_SWAP_FOOT_PAIR_WEIGHT}'
        )
    LR_SWAP_HARD_MIX = float(getattr(training_cfg, 'lr_swap_hard_mix', 0.0))
    if not 0.0 <= LR_SWAP_HARD_MIX <= 1.0:
        raise ValueError(f'lr_swap_hard_mix must be in [0, 1], got {LR_SWAP_HARD_MIX}')
    CHIRALITY_LOSS_WEIGHT = float(getattr(training_cfg, 'chirality_loss_weight', 0.0))
    if CHIRALITY_LOSS_WEIGHT < 0:
        raise ValueError(f'chirality_loss_weight must be >= 0, got {CHIRALITY_LOSS_WEIGHT}')
    CHIRALITY_MIN_SEP = float(getattr(training_cfg, 'chirality_min_sep', 0.03))
    if CHIRALITY_MIN_SEP < 0:
        raise ValueError(f'chirality_min_sep must be >= 0, got {CHIRALITY_MIN_SEP}')
    CHIRALITY_FOOT_PAIR_WEIGHT = float(
        getattr(training_cfg, 'chirality_foot_pair_weight', 1.0)
    )
    if CHIRALITY_FOOT_PAIR_WEIGHT < 0:
        raise ValueError(
            f'chirality_foot_pair_weight must be >= 0, got {CHIRALITY_FOOT_PAIR_WEIGHT}'
        )
    OKS_LOSS_WEIGHT = float(getattr(training_cfg, 'oks_loss_weight', 0.0))
    if OKS_LOSS_WEIGHT < 0:
        raise ValueError(f'oks_loss_weight must be >= 0, got {OKS_LOSS_WEIGHT}')
    PRIORITY_PART_WEIGHTS = {
        str(name): float(weight)
        for name, weight in getattr(training_cfg, 'priority_part_weights', {}).items()
    }
    if set(PRIORITY_PART_WEIGHTS) != {'feet', 'legs', 'arms', 'head'}:
        raise ValueError(
            'priority_part_weights must define exactly feet, legs, arms, and head'
        )
    if any(weight < 0 for weight in PRIORITY_PART_WEIGHTS.values()):
        raise ValueError('priority_part_weights must be non-negative')
    if sum(PRIORITY_PART_WEIGHTS.values()) <= 0:
        raise ValueError('priority_part_weights must have a positive sum')
    checkpoint_metric = str(getattr(training_cfg, 'checkpoint_metric', 'val_loss')).lower()
    if checkpoint_metric not in ('val_loss', 'ap', 'priority_oks'):
        raise ValueError(
            'checkpoint_metric must be val_loss, ap, or priority_oks, '
            f'got {training_cfg.checkpoint_metric!r}'
        )
    if checkpoint_metric == 'ap' and not training_cfg.compute_ap:
        raise ValueError('checkpoint_metric=ap requires compute_ap=true')

    ordered = sorted(head_cfg.keypoints.items(), key=lambda item: item[1]['id'])
    local_ids = [meta['id'] for _, meta in ordered]
    if local_ids != list(range(len(local_ids))):
        raise ValueError(
            f'Keypoint local ids must be contiguous 0..K-1, got: {local_ids}'
        )
    KEYPOINTS_BY_NAME = head_cfg.keypoints
    KEYPOINT_IDS = [meta['standard_id'] for _, meta in ordered]
    KEYPOINT_INDEX = {meta['standard_id']: meta['id'] for _, meta in ordered}
    PERSON_JOINTS = len(ordered)
    NUM_JOINTS = PERSON_JOINTS * 2 if COUPLE_MODE else PERSON_JOINTS

    if head_cfg.type == 'simcc':
        simcc = head_cfg.custom
        SIMCC_SPLIT_RATIO = simcc.split_ratio
        SIMCC_SIGMA = float(simcc.gaussian_sigma)
        occ = getattr(simcc, 'gaussian_sigma_occluded', None)
        SIMCC_SIGMA_OCCLUDED = float(occ) if occ is not None else None
        if SIMCC_SIGMA_OCCLUDED is not None and SIMCC_SIGMA_OCCLUDED <= 0:
            raise ValueError(
                f'gaussian_sigma_occluded must be > 0, got {SIMCC_SIGMA_OCCLUDED}'
            )
        OCCLUDED_JOINT_WEIGHT = float(getattr(simcc, 'occluded_joint_weight', 0.35))
        if not 0.0 < OCCLUDED_JOINT_WEIGHT <= 1.0:
            raise ValueError(
                f'occluded_joint_weight must be in (0, 1], got {OCCLUDED_JOINT_WEIGHT}'
            )
        expected_x = int(image_cfg.width * simcc.split_ratio)
        expected_y = int(image_cfg.height * simcc.split_ratio)
        if simcc.x_bins != expected_x or simcc.y_bins != expected_y:
            raise ValueError(
                f'SimCC bins must equal image_size * split_ratio '
                f'(expected x_bins={expected_x}, y_bins={expected_y}; '
                f'got x_bins={simcc.x_bins}, y_bins={simcc.y_bins})'
            )

    person_loss_weights = build_joint_loss_weights(
        training_cfg.joint_loss_weights, head_cfg.keypoints
    )
    JOINT_LOSS_WEIGHTS = (
        torch.cat([person_loss_weights, person_loss_weights], dim=0)
        if COUPLE_MODE
        else person_loss_weights
    )
    LEFT_RIGHT_PAIRS = build_left_right_index_pairs(
        head_cfg.left_right_pairs, head_cfg.keypoints
    )
    _id_to_name = {int(meta['id']): name for name, meta in head_cfg.keypoints.items()}
    _foot_tokens = (
        'ankle', 'knee', 'hip', 'heel', 'toe_b', 'toe_s', 'heel_spike'
    )

    def _lr_pair_is_lower_body(left_i: int, right_i: int) -> bool:
        for jid in (left_i, right_i):
            name = _id_to_name.get(int(jid), '')
            if any(tok in name for tok in _foot_tokens):
                return True
        return False

    LEFT_RIGHT_PAIR_IS_LOWER = [
        _lr_pair_is_lower_body(left_i, right_i) for left_i, right_i in LEFT_RIGHT_PAIRS
    ]
    person_bones = build_bone_index_pairs(head_cfg.keypoints)
    if COUPLE_MODE:
        follow_offset = PERSON_JOINTS
        BONE_PAIRS = torch.tensor(
            person_bones
            + [(i + follow_offset, j + follow_offset) for i, j in person_bones],
            dtype=torch.long,
        )
    else:
        BONE_PAIRS = torch.tensor(person_bones, dtype=torch.long)
    if head_cfg.oks_sigmas:
        # Always person-sized; couple eval splits banks before OKS.
        OKS_SIGMAS = build_oks_sigmas(head_cfg.oks_sigmas, head_cfg.keypoints)
    else:
        OKS_SIGMAS = torch.empty(0)


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


def crop_normalized_bbox_center(
    center_x: float,
    center_y: float,
    crop_x1: float,
    crop_y1: float,
    crop_width: float,
    crop_height: float,
) -> torch.Tensor:
    """Person bbox center in crop-normalized [0, 1]^2 coordinates."""
    cx = (center_x - crop_x1) / max(crop_width, 1e-6)
    cy = (center_y - crop_y1) / max(crop_height, 1e-6)
    cx = float(min(max(cx, 0.0), 1.0))
    cy = float(min(max(cy, 0.0), 1.0))
    return torch.tensor([cx, cy], dtype=torch.float32)


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


def _frame_index_from_image_path(image_path) -> int:
    name = Path(image_path).name
    if '_frame_' in name:
        stem = name.rsplit('.', 1)[0]
        suffix = stem.split('_frame_', 1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def cap_samples_per_clip(
    samples,
    max_frames_per_clip: int | None,
    seed: int,
    *,
    mode: str = 'even',
    log: bool = True,
):
    """Keep at most ``max_frames_per_clip`` unique images per task/clip.

    Clips at or under the cap keep every frame. Oversize clips are subsampled:

    - ``mode='even'``: evenly spaced along sorted frame index (fixed val/test).
    - ``mode='random'``: ``seed``-driven sample without replacement (train rolling).
    """
    if max_frames_per_clip is None:
        return samples
    if max_frames_per_clip < 1:
        raise ValueError(f'max_frames_per_clip must be >= 1, got {max_frames_per_clip}')
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ('even', 'random'):
        raise ValueError(f"cap_samples_per_clip mode must be 'even' or 'random', got {mode!r}")

    by_clip: dict[str, dict] = {}
    order: list[str] = []
    for sample in samples:
        image_path = sample[0]
        clip_id = clip_id_from_image_path(image_path)
        if clip_id not in by_clip:
            by_clip[clip_id] = {}
            order.append(clip_id)
        by_clip[clip_id].setdefault(image_path, []).append(sample)

    rng = random.Random(seed)
    kept = []
    dropped_frames = 0
    rolled_clips = 0
    short_clips = 0
    for clip_id in order:
        image_map = by_clip[clip_id]
        paths = sorted(image_map, key=_frame_index_from_image_path)
        if len(paths) <= max_frames_per_clip:
            selected = paths
            short_clips += 1
        else:
            n = len(paths)
            if mode_norm == 'random':
                selected = sorted(
                    rng.sample(paths, max_frames_per_clip),
                    key=_frame_index_from_image_path,
                )
            else:
                # Evenly spaced indices over [0, n-1], inclusive endpoints.
                idxs = sorted(
                    {
                        int(round(i * (n - 1) / (max_frames_per_clip - 1)))
                        for i in range(max_frames_per_clip)
                    }
                )
                # If rounding collapsed duplicates, fill gaps from unused frames.
                if len(idxs) < max_frames_per_clip:
                    unused = [j for j in range(n) if j not in idxs]
                    need = max_frames_per_clip - len(idxs)
                    idxs = sorted(idxs + unused[:need])
                selected = [paths[j] for j in idxs]
            dropped_frames += n - len(selected)
            rolled_clips += 1
        for path in selected:
            kept.extend(image_map[path])

    if log and (dropped_frames or mode_norm == 'random'):
        print(
            f'max_frames_per_clip={max_frames_per_clip} mode={mode_norm}: '
            f'dropped {dropped_frames} frames from {rolled_clips} oversize clips '
            f'({short_clips} clips kept in full); '
            f'kept {sum(1 for _ in {s[0] for s in kept})}/'
            f'{sum(1 for _ in {s[0] for s in samples})} images, '
            f'{len(kept)}/{len(samples)} person-samples'
        )
    return kept


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


def _normalize_clip_id(clip_id: str) -> str:
    text = str(clip_id).strip()
    if text.isdigit():
        return f'task_{text}'
    if text.startswith('task_'):
        return text
    return text


def split_samples_by_image(
    samples, train_ratio, seed, split_mode='frame', test_clips=None
):
    """Split person-samples into train/test.

    Modes:
      - frame: shuffle unique images globally (legacy).
      - stratified_clip: within each task/clip, split images by ratio (every
        non-holdout clip appears in train and test). If ``test_clips`` is set,
        those clips are held out entirely; remaining clips are stratified.
      - clip: assign whole clips to train or test.
        If ``test_clips`` is set, those clips are held out and the rest train
        (``train_ratio`` is ignored for assignment).
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
        holdout_clips = {_normalize_clip_id(c) for c in (test_clips or [])}
        if holdout_clips:
            available = set(clip_to_images)
            missing = sorted(holdout_clips - available)
            if missing:
                raise ValueError(
                    f'test_clips not found in dataset: {missing}; available={sorted(available)}'
                )
            if holdout_clips == available:
                raise ValueError('test_clips cannot include every clip')
        train_paths, test_paths = [], []
        for clip_id, paths in sorted(clip_to_images.items()):
            if _normalize_clip_id(clip_id) in holdout_clips:
                test_paths.extend(paths)
                continue
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
        if test_clips:
            wanted = {_normalize_clip_id(c) for c in test_clips}
            available = set(clip_ids)
            missing = sorted(wanted - available)
            if missing:
                raise ValueError(
                    f'test_clips not found in dataset: {missing}; available={sorted(available)}'
                )
            if wanted == available:
                raise ValueError('test_clips cannot include every clip')
            test_clip_set = wanted
            train_clip_set = available - wanted
        else:
            rng = random.Random(seed)
            shuffled = list(clip_ids)
            rng.shuffle(shuffled)
            split_index = max(1, min(len(shuffled) - 1, round(len(shuffled) * train_ratio)))
            train_clip_set = set(shuffled[:split_index])
            test_clip_set = set(shuffled[split_index:])
        train_paths = [path for clip_id in train_clip_set for path in clip_to_images[clip_id]]
        test_paths = [path for clip_id in test_clip_set for path in clip_to_images[clip_id]]
    else:
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}; expected 'frame', 'stratified_clip', or 'clip'"
        )

    train_samples = [sample for path in train_paths for sample in image_groups[path]]
    test_samples = [sample for path in test_paths for sample in image_groups[path]]
    return train_samples, test_samples, train_paths, test_paths


def make_targets(keypoints, device):
    targets = [
        generate_simcc_labels(
            sample.numpy(),
            IMAGE_SIZE,
            split_ratio=SIMCC_SPLIT_RATIO,
            sigma=SIMCC_SIGMA,
            sigma_occluded=SIMCC_SIGMA_OCCLUDED,
            occluded_weight=OCCLUDED_JOINT_WEIGHT,
        )
        for sample in keypoints
    ]
    target_x, target_y, weights = zip(*targets)
    return (
        torch.stack(target_x).to(device),
        torch.stack(target_y).to(device),
        torch.stack(weights).to(device) * JOINT_LOSS_WEIGHTS.to(device),
    )


def maybe_drop_role(role: torch.Tensor, training: bool) -> torch.Tensor:
    """Randomly zero role bits during training (role dropout)."""
    if not training or not USE_ROLE or ROLE_DROPOUT <= 0:
        return role
    keep = torch.rand(role.shape[0], 1, device=role.device) >= ROLE_DROPOUT
    return role * keep.to(role.dtype)


def keypoints_to_local_attn_gt(keypoints: torch.Tensor):
    """Crop-pixel keypoints [B,K,3] → (xy_normed [B,K,2], valid [B,K])."""
    xy = keypoints[..., :2].to(dtype=torch.float32)
    xy_normed = torch.stack(
        (
            xy[..., 0] / max(float(IMAGE_SIZE.x), 1.0),
            xy[..., 1] / max(float(IMAGE_SIZE.y), 1.0),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)
    valid = keypoints[..., 2] > 0
    return xy_normed, valid


def forward_model(model, pixels, role, spatial, keypoints=None):
    """Run DinoCC; returns (pred_x, pred_y, aux_list | None).

    When ``keypoints`` is provided and the head uses GT/mix local-attn centering,
    pass crop-normalized GT so train-time cross-attn gets a soft prior near labels.
    Val/infer omit ``keypoints`` → pred-centered (or no) bias.

    ``pixels`` may be ``[B,3,H,W]`` or temporal ``[B,T,3,H,W]`` (preds flattened to B*T).
    """
    joint_xy_normed = joint_valid = None
    if keypoints is not None:
        head = getattr(model, 'head', None)
        mode = str(getattr(head, 'local_attn_center_mode', 'pred')).lower()
        if getattr(head, 'use_local_cross_attn', False) and mode in ('gt', 'mix'):
            flat_kps = keypoints
            if flat_kps.ndim == 4:
                bsz, t_len, n_joints, _ = flat_kps.shape
                flat_kps = flat_kps.reshape(bsz * t_len, n_joints, 3)
            joint_xy_normed, joint_valid = keypoints_to_local_attn_gt(flat_kps)
            if keypoints.ndim == 4:
                bsz, t_len, n_joints, _ = keypoints.shape
                joint_xy_normed = joint_xy_normed.view(bsz, t_len, n_joints, 2)
                joint_valid = joint_valid.view(bsz, t_len, n_joints)
    out = model(
        pixels,
        role=role if USE_ROLE else None,
        spatial=spatial if USE_SPATIAL_PROMPT else None,
        joint_xy_normed=joint_xy_normed,
        joint_valid=joint_valid,
    )
    if isinstance(out, tuple) and len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], None


def flatten_temporal_batch(pixels, keypoints, role, spatial, bbox_scales):
    """Flatten ``[B,T,...]`` tensors to ``[B*T,...]`` for loss; return ``(flat, batch_size)``."""
    if pixels.ndim != 5:
        return pixels, keypoints, role, spatial, bbox_scales, pixels.shape[0]
    batch, t_len = pixels.shape[:2]
    flat_pixels = pixels  # model accepts 5D; keep for forward
    flat_keypoints = keypoints.reshape(batch * t_len, keypoints.shape[2], keypoints.shape[3])
    flat_role = role.reshape(batch * t_len, role.shape[-1])
    flat_spatial = spatial.reshape(batch * t_len, spatial.shape[-1])
    flat_scales = bbox_scales.reshape(batch * t_len)
    return flat_pixels, flat_keypoints, flat_role, flat_spatial, flat_scales, batch


def load_state_dict_partial(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load matching keys/shapes from a (possibly single-frame) teacher checkpoint."""
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # unexpected from filtered path should be empty; missing = new temporal params etc.
    if skipped_shape:
        print(
            f'Partial init skipped {len(skipped_shape)} shape-mismatched keys '
            f'(e.g. {skipped_shape[:3]})'
        )
    return list(missing), list(unexpected)


def soft_decode_simcc_coords(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    split_ratio: float,
) -> torch.Tensor:
    """Differentiable SimCC decode → [B, K, 2] crop pixels via expectation."""
    px = torch.nn.functional.softmax(pred_x.float(), dim=-1)
    py = torch.nn.functional.softmax(pred_y.float(), dim=-1)
    bins_x = torch.arange(pred_x.shape[-1], device=pred_x.device, dtype=px.dtype)
    bins_y = torch.arange(pred_y.shape[-1], device=pred_y.device, dtype=py.dtype)
    x = (px * bins_x).sum(dim=-1) / split_ratio
    y = (py * bins_y).sum(dim=-1) / split_ratio
    return torch.stack((x, y), dim=-1)


def bone_vector_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    bone_pairs: torch.Tensor,
) -> torch.Tensor:
    """L1 on predicted vs GT bone vectors (normalized by image diagonal)."""
    if bone_pairs.numel() == 0:
        return pred_xy.new_zeros(())
    pairs = bone_pairs.to(device=pred_xy.device)
    parent = pairs[:, 0]
    child = pairs[:, 1]
    pred_vec = pred_xy[:, child] - pred_xy[:, parent]  # [B, N, 2]
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    gt_vec = gt_xy[:, child] - gt_xy[:, parent]
    vis = gt_keypoints[..., 2].to(device=pred_xy.device)
    valid = (vis[:, parent] > 0) & (vis[:, child] > 0)  # [B, N]
    err = (pred_vec - gt_vec).abs().sum(dim=-1)  # [B, N]
    diag = (IMAGE_SIZE.x ** 2 + IMAGE_SIZE.y ** 2) ** 0.5
    err = err / max(float(diag), 1e-6)
    return (err * valid).sum() / (valid.sum() + 1e-6)


def differentiable_oks_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    weights: torch.Tensor,
    bbox_scales: torch.Tensor,
) -> torch.Tensor:
    """Approximate ``1 - OKS`` using soft-decoded coordinates and bbox scale."""
    if OKS_SIGMAS.numel() == 0:
        return pred_xy.new_zeros(())
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    visibility = gt_keypoints[..., 2].to(device=pred_xy.device)
    valid = (visibility > 0) & (weights > 0)
    if not bool(valid.any()):
        return pred_xy.new_zeros(())
    scale = bbox_scales.to(device=pred_xy.device, dtype=pred_xy.dtype).clamp(min=1.0)
    sigmas = OKS_SIGMAS.to(device=pred_xy.device, dtype=pred_xy.dtype).view(1, -1)
    dist_sq = ((pred_xy - gt_xy) ** 2).sum(dim=-1)
    oks = torch.exp(-dist_sq / (2.0 * scale[:, None].square() * sigmas.square() + 1e-6))
    weighted = (oks * weights * valid).sum()
    return 1.0 - weighted / (weights * valid).sum().clamp(min=1e-6)


def lr_swap_assignment_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    left_right_pairs: list[tuple[int, int]],
    margin: float | None = None,
) -> torch.Tensor:
    """Per-pair L/R swap hinge with lower-body emphasis and hard-example mix.

    ``relu((d_correct - d_swapped) / (diag * scale_frac) + margin)`` — smaller
    ``scale_frac`` makes assignment errors compete with SimCC KL. Foot/leg pairs
    can be up-weighted; ``hard_mix`` blends in the mean over near-swap samples
    so clean batches don't dilute the gradient.
    """
    if not left_right_pairs:
        return pred_xy.new_zeros(())
    margin = LR_SWAP_MARGIN if margin is None else float(margin)
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    vis = gt_keypoints[..., 2].to(device=pred_xy.device)
    diag = (IMAGE_SIZE.x ** 2 + IMAGE_SIZE.y ** 2) ** 0.5
    scale = max(float(diag) * float(LR_SWAP_SCALE_FRAC), 1e-6)

    pair_losses = []
    pair_weights = []
    for pair_i, (left_i, right_i) in enumerate(left_right_pairs):
        valid = (vis[:, left_i] > 0) & (vis[:, right_i] > 0)
        if not bool(valid.any()):
            continue
        pred_l = pred_xy[:, left_i]
        pred_r = pred_xy[:, right_i]
        gt_l = gt_xy[:, left_i]
        gt_r = gt_xy[:, right_i]
        correct = (pred_l - gt_l).abs().sum(dim=-1) + (pred_r - gt_r).abs().sum(dim=-1)
        swapped = (pred_l - gt_r).abs().sum(dim=-1) + (pred_r - gt_l).abs().sum(dim=-1)
        hinge = torch.relu((correct - swapped) / scale + margin)
        pair_losses.append(hinge[valid].mean())
        is_lower = (
            LEFT_RIGHT_PAIR_IS_LOWER[pair_i]
            if pair_i < len(LEFT_RIGHT_PAIR_IS_LOWER)
            else False
        )
        pair_weights.append(
            float(LR_SWAP_FOOT_PAIR_WEIGHT) if is_lower else 1.0
        )

    if not pair_losses:
        return pred_xy.new_zeros(())
    losses = torch.stack(pair_losses)
    weights = pred_xy.new_tensor(pair_weights)
    soft = (losses * weights).sum() / weights.sum().clamp(min=1e-6)

    if LR_SWAP_HARD_MIX <= 0:
        return soft
    # Recompute batch-level hinge for hard mining (lower-body weighted pairs only).
    d_correct = pred_xy.new_zeros(pred_xy.shape[0])
    d_swapped = pred_xy.new_zeros(pred_xy.shape[0])
    w_count = pred_xy.new_zeros(pred_xy.shape[0])
    for pair_i, (left_i, right_i) in enumerate(left_right_pairs):
        valid = (vis[:, left_i] > 0) & (vis[:, right_i] > 0)
        if not bool(valid.any()):
            continue
        is_lower = (
            LEFT_RIGHT_PAIR_IS_LOWER[pair_i]
            if pair_i < len(LEFT_RIGHT_PAIR_IS_LOWER)
            else False
        )
        pw = float(LR_SWAP_FOOT_PAIR_WEIGHT) if is_lower else 1.0
        correct = (pred_xy[:, left_i] - gt_xy[:, left_i]).abs().sum(dim=-1) + (
            pred_xy[:, right_i] - gt_xy[:, right_i]
        ).abs().sum(dim=-1)
        swapped = (pred_xy[:, left_i] - gt_xy[:, right_i]).abs().sum(dim=-1) + (
            pred_xy[:, right_i] - gt_xy[:, left_i]
        ).abs().sum(dim=-1)
        d_correct = d_correct + correct * valid.float() * pw
        d_swapped = d_swapped + swapped * valid.float() * pw
        w_count = w_count + valid.float() * pw
    has = w_count > 0
    if not bool(has.any()):
        return soft
    hinge_b = torch.relu(
        (d_correct / w_count.clamp(min=1e-6) - d_swapped / w_count.clamp(min=1e-6))
        / scale
        + margin
    )
    hard = hinge_b[has & (hinge_b > margin * 0.5)]
    if hard.numel() == 0:
        return soft
    return (1.0 - float(LR_SWAP_HARD_MIX)) * soft + float(LR_SWAP_HARD_MIX) * hard.mean()


def lr_chirality_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    left_right_pairs: list[tuple[int, int]],
    min_sep: float | None = None,
) -> torch.Tensor:
    """Push L/R vectors to match GT orientation: ``relu(-cos(pred_R-L, gt_R-L))``.

    Skips pairs whose GT separation is below ``min_sep * image_diag`` so closed
    feet do not inject noisy chirality gradients. Lower-body pairs can be boosted.
    """
    if not left_right_pairs:
        return pred_xy.new_zeros(())
    min_sep = CHIRALITY_MIN_SEP if min_sep is None else float(min_sep)
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    vis = gt_keypoints[..., 2].to(device=pred_xy.device)
    diag = max((IMAGE_SIZE.x ** 2 + IMAGE_SIZE.y ** 2) ** 0.5, 1e-6)
    min_norm = min_sep * diag

    losses = []
    weights = []
    for pair_i, (left_i, right_i) in enumerate(left_right_pairs):
        valid = (vis[:, left_i] > 0) & (vis[:, right_i] > 0)
        gt_vec = gt_xy[:, right_i] - gt_xy[:, left_i]
        gt_norm = gt_vec.norm(dim=-1)
        valid = valid & (gt_norm >= min_norm)
        if not bool(valid.any()):
            continue
        pred_vec = pred_xy[:, right_i] - pred_xy[:, left_i]
        pred_norm = pred_vec.norm(dim=-1).clamp(min=1e-6)
        cos = (pred_vec * gt_vec).sum(dim=-1) / (pred_norm * gt_norm.clamp(min=1e-6))
        # 0 when aligned (cos>0), up to 1 when fully anti-aligned (swapped).
        losses.append(torch.relu(-cos)[valid].mean())
        is_lower = (
            LEFT_RIGHT_PAIR_IS_LOWER[pair_i]
            if pair_i < len(LEFT_RIGHT_PAIR_IS_LOWER)
            else False
        )
        weights.append(
            float(CHIRALITY_FOOT_PAIR_WEIGHT) if is_lower else 1.0
        )
    if not losses:
        return pred_xy.new_zeros(())
    loss_t = torch.stack(losses)
    weight_t = pred_xy.new_tensor(weights)
    return (loss_t * weight_t).sum() / weight_t.sum().clamp(min=1e-6)


def simcc_xy_consistency_loss(pred_x, pred_y, target_x, target_y, weights):
    """KL between factored pred marginals and factored GT SimCC (outer product)."""
    px = torch.nn.functional.softmax(pred_x, dim=-1)
    py = torch.nn.functional.softmax(pred_y, dim=-1)
    pred_2d = px.unsqueeze(-1) * py.unsqueeze(-2)
    tgt_2d = target_x.unsqueeze(-1) * target_y.unsqueeze(-2)
    tgt_2d = tgt_2d / (tgt_2d.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    log_pred = torch.log(pred_2d.clamp(min=1e-8))
    kl = torch.nn.functional.kl_div(log_pred, tgt_2d, reduction='none').sum(dim=(-2, -1))
    return (kl * weights).sum() / (weights.sum() + 1e-6)


def compute_simcc_training_loss(
    criterion,
    pred_x,
    pred_y,
    target_x,
    target_y,
    weights,
    aux_preds,
    gt_keypoints=None,
    bbox_scales=None,
    return_components=False,
):
    components = {
        'simcc_kl': criterion(pred_x, pred_y, target_x, target_y, weights),
        'deep_supervision': pred_x.new_zeros(()),
        'xy_consistency': pred_x.new_zeros(()),
        'oks': pred_x.new_zeros(()),
        'bone': pred_x.new_zeros(()),
        'lr_swap': pred_x.new_zeros(()),
        'chirality': pred_x.new_zeros(()),
    }
    loss = components['simcc_kl']
    if aux_preds:
        for aux_x, aux_y in aux_preds:
            components['deep_supervision'] = components['deep_supervision'] + criterion(
                aux_x, aux_y, target_x, target_y, weights
            )
        components['deep_supervision'] = (
            components['deep_supervision'] * DEEP_SUPERVISION_AUX_WEIGHT
        )
        loss = loss + components['deep_supervision']
    if XY_CONSISTENCY_LOSS_WEIGHT > 0:
        components['xy_consistency'] = XY_CONSISTENCY_LOSS_WEIGHT * simcc_xy_consistency_loss(
            pred_x, pred_y, target_x, target_y, weights
        )
        loss = loss + components['xy_consistency']
    need_xy = gt_keypoints is not None and (
        OKS_LOSS_WEIGHT > 0
        or BONE_LOSS_WEIGHT > 0
        or LR_SWAP_LOSS_WEIGHT > 0
        or CHIRALITY_LOSS_WEIGHT > 0
    )
    if need_xy:
        assert gt_keypoints is not None
        pred_xy = soft_decode_simcc_coords(pred_x, pred_y, SIMCC_SPLIT_RATIO)
        if OKS_LOSS_WEIGHT > 0:
            if bbox_scales is None:
                raise ValueError('bbox_scales are required when oks_loss_weight > 0')
            components['oks'] = OKS_LOSS_WEIGHT * differentiable_oks_loss(
                pred_xy, gt_keypoints, weights, bbox_scales
            )
            loss = loss + components['oks']
        if BONE_LOSS_WEIGHT > 0:
            components['bone'] = BONE_LOSS_WEIGHT * bone_vector_loss(
                pred_xy, gt_keypoints, BONE_PAIRS
            )
            loss = loss + components['bone']
        if LR_SWAP_LOSS_WEIGHT > 0:
            components['lr_swap'] = LR_SWAP_LOSS_WEIGHT * lr_swap_assignment_loss(
                pred_xy, gt_keypoints, LEFT_RIGHT_PAIRS
            )
            loss = loss + components['lr_swap']
        if CHIRALITY_LOSS_WEIGHT > 0:
            components['chirality'] = CHIRALITY_LOSS_WEIGHT * lr_chirality_loss(
                pred_xy, gt_keypoints, LEFT_RIGHT_PAIRS
            )
            loss = loss + components['chirality']
    if return_components:
        return loss, components
    return loss


def simcc_kl_per_sample(pred_x, pred_y, target_x, target_y, weights) -> torch.Tensor:
    """Per-example SimCC KL (same reduction as SimCCLoss, but kept over batch)."""
    log_pred_x = F.log_softmax(pred_x.float(), dim=-1)
    log_pred_y = F.log_softmax(pred_y.float(), dim=-1)
    loss_x = F.kl_div(log_pred_x, target_x.float(), reduction='none').sum(dim=-1)
    loss_y = F.kl_div(log_pred_y, target_y.float(), reduction='none').sum(dim=-1)
    weighted = (loss_x + loss_y) * weights.float()
    return weighted.sum(dim=-1) / (weights.float().sum(dim=-1) + 1e-6)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    training,
    gradient_clip_norm,
    return_components=False,
    track_source_loss=False,
):
    model.train(training)
    total_loss = 0.0
    component_totals: dict[str, float] = {}
    dance_loss_sum = 0.0
    dance_count = 0.0
    coco_loss_sum = 0.0
    coco_count = 0.0

    for batch in loader:
        pixels, keypoints, role, spatial, bbox_scales, is_coco = batch
        pixels = pixels.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        role = role.to(device, non_blocking=True)
        spatial = spatial.to(device, non_blocking=True)
        bbox_scales = bbox_scales.to(device, non_blocking=True)
        is_coco = is_coco.to(device, non_blocking=True)

        (
            pixels_fwd,
            keypoints_flat,
            role_flat,
            spatial_flat,
            scales_flat,
            batch_size,
        ) = flatten_temporal_batch(pixels, keypoints, role, spatial, bbox_scales)
        # Role dropout over windows (or frames); flatten first when temporal.
        if pixels.ndim == 5:
            role_for_drop = role_flat
        else:
            role_for_drop = role
        role_dropped = maybe_drop_role(role_for_drop, training)
        if pixels.ndim == 5:
            role_fwd = role_dropped.view(batch_size, pixels.shape[1], -1)
            spatial_fwd = spatial_flat.view(batch_size, pixels.shape[1], -1)
            keypoints_fwd = keypoints
            scales_loss = scales_flat
            keypoints_loss = keypoints_flat
        else:
            role_fwd = role_dropped
            spatial_fwd = spatial
            keypoints_fwd = keypoints
            scales_loss = bbox_scales
            keypoints_loss = keypoints

        target_x, target_y, weights = make_targets(keypoints_loss.detach().cpu(), device)
        gt_keypoints = (
            keypoints_loss
            if (
                OKS_LOSS_WEIGHT > 0
                or BONE_LOSS_WEIGHT > 0
                or LR_SWAP_LOSS_WEIGHT > 0
                or CHIRALITY_LOSS_WEIGHT > 0
            )
            else None
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y, aux_preds = forward_model(
                model,
                pixels_fwd,
                role_fwd,
                spatial_fwd,
                keypoints=(keypoints_fwd if training else None),
            )
            loss_result = compute_simcc_training_loss(
                criterion,
                pred_x,
                pred_y,
                target_x,
                target_y,
                weights,
                aux_preds,
                gt_keypoints=gt_keypoints,
                bbox_scales=scales_loss,
                return_components=return_components,
            )
            if return_components:
                loss, components = loss_result
            else:
                loss = loss_result

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * batch_size
        if return_components:
            for name, value in components.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0) + value.item() * batch_size
                )
        if track_source_loss and training:
            with torch.no_grad():
                per = simcc_kl_per_sample(pred_x, pred_y, target_x, target_y, weights)
                if pixels.ndim == 5:
                    # Aggregate frame losses → window then apply is_coco (scalar per window).
                    t_len = pixels.shape[1]
                    per = per.view(batch_size, t_len).mean(dim=1)
                dance_mask = is_coco < 0.5
                coco_mask = ~dance_mask
                if dance_mask.any():
                    dance_loss_sum += float(per[dance_mask].sum().item())
                    dance_count += float(dance_mask.sum().item())
                if coco_mask.any():
                    coco_loss_sum += float(per[coco_mask].sum().item())
                    coco_count += float(coco_mask.sum().item())

    mean_loss = total_loss / len(loader.dataset)
    source_losses = None
    if track_source_loss:
        source_losses = {
            'dance': (dance_loss_sum / dance_count) if dance_count > 0 else None,
            'coco': (coco_loss_sum / coco_count) if coco_count > 0 else None,
        }
    if return_components:
        comps = {
            name: value / len(loader.dataset)
            for name, value in component_totals.items()
        }
        if track_source_loss:
            return mean_loss, comps, source_losses
        return mean_loss, comps
    if track_source_loss:
        return mean_loss, source_losses
    return mean_loss


def priority_oks_score(metrics: dict[str, Any]) -> float:
    """Weighted mean body-part OKS for the user's feet-first objective."""
    parts = metrics.get('custom', {}).get('body_parts', {}).get('parts', {})
    weighted_sum = 0.0
    weight_sum = 0.0
    for part, weight in PRIORITY_PART_WEIGHTS.items():
        value = parts.get(part, {}).get('mean_OKS')
        if value is not None and weight > 0:
            weighted_sum += weight * float(value)
            weight_sum += weight
    return weighted_sum / weight_sum if weight_sum else 0.0



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
    version_dir = get_pose_estimator_version_folder(annotation_version)
    annotation_file = version_dir / 'cleaned_annotations.xml'
    if not annotation_file.is_file():
        raise FileNotFoundError(f'Cleaned annotations not found: {annotation_file}')
    return annotation_file


def parse_config(raw_config):
    image_cfg = ImageConfig(**raw_config['image'])
    backbone_cfg = BackboneConfig(**raw_config['backbone'])
    training_cfg = TrainingConfig(**raw_config['training'])
    initialization_cfg = InitializationConfig.from_raw(raw_config.get('initialization'))

    dataset_raw = dict(raw_config['dataset'])
    coco_raw = dataset_raw.pop('coco_wholebody', None)
    coco_cfg = CocoWholeBodyConfig.from_raw(coco_raw)
    coco_cfg.validate()

    dataset_cfg = DatasetConfig(**dataset_raw, annotation_file='', coco_wholebody=coco_cfg)
    annotation_path = Path('annotations')
    dataset_cfg.annotation_file = str(
        annotation_path / f'annotations_{dataset_cfg.annotation_version}.xml'
    )

    head_cfg = HeadConfig(**raw_config['head'], keypoints={})
    keypoint_format = load_keypoint_format(head_cfg.keypoint_format_name)
    head_cfg.keypoints = keypoint_format.keypoints
    head_cfg.left_right_pairs = keypoint_format.left_right_pairs
    head_cfg.oks_sigmas = keypoint_format.oks_sigmas

    if head_cfg.type != 'simcc':
        raise ValueError(f'Unsupported head type: {head_cfg.type}')

    match head_cfg.name:
        case 'depthwise_simcc':
            head_cfg.custom = SimCCConfig(**head_cfg.custom)
        case (
            'joint_query_simcc'
            | 'joint_query_self_attn_simcc'
            | 'temporal_joint_query_self_attn_simcc'
        ):
            head_cfg.custom = JointQuerySimCCConfig(**head_cfg.custom)
        case 'couple_joint_query_self_attn_simcc':
            head_cfg.custom = CoupleJointQuerySimCCConfig(**head_cfg.custom)
        case _:
            raise ValueError(
                f'Unsupported head name: {head_cfg.name}; '
                f"expected 'depthwise_simcc', 'joint_query_simcc', "
                f"'joint_query_self_attn_simcc', "
                f"'temporal_joint_query_self_attn_simcc', "
                f"or 'couple_joint_query_self_attn_simcc'"
            )

    if coco_cfg.enabled and head_cfg.name == 'couple_joint_query_self_attn_simcc':
        raise ValueError(
            'coco_wholebody mixing is only supported for single-person heads '
            '(not couple_joint_query_self_attn_simcc)'
        )

    return {
        'dataset': dataset_cfg,
        'image': image_cfg,
        'backbone': backbone_cfg,
        'head': head_cfg,
        'training': training_cfg,
        'initialization': initialization_cfg,
    }


BACKBONE_HF_IDS = {
    'dino_v2_base': 'facebook/dinov2-base',
    'dino_v3_base': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'vitpose_plus_base': 'usyd-community/vitpose-plus-base',
    'vitpose_base': 'usyd-community/vitpose-base',
}


def build_model(config, device):
    """Construct a DinoCC model from a parsed config on ``device``."""
    head_cfg: HeadConfig = config['head']
    backbone_cfg: BackboneConfig = config['backbone']
    initialization_cfg: InitializationConfig | None = config.get('initialization')

    match backbone_cfg.name:
        case 'dino_v2_base' | 'dino_v3_base' | 'vitpose_plus_base' | 'vitpose_base':
            match head_cfg.type:
                case 'simcc':
                    joint_query_heads = (
                        'joint_query_simcc',
                        'joint_query_self_attn_simcc',
                        TEMPORAL_HEAD_NAME,
                        COUPLE_HEAD_NAME,
                    )
                    if head_cfg.name in joint_query_heads:
                        head_kwargs = build_joint_query_head_kwargs(
                            head_cfg, initialization_cfg
                        )
                    elif head_cfg.name == 'depthwise_simcc':
                        head_kwargs = build_depthwise_head_kwargs(head_cfg)
                    else:
                        head_kwargs = {}
                    model = DinoCC(
                        NUM_JOINTS,
                        IMAGE_SIZE,
                        freeze_backbone=backbone_cfg.freeze,
                        split_ratio=SIMCC_SPLIT_RATIO,
                        neck_dim=head_cfg.out_channels,
                        backbone_name=BACKBONE_HF_IDS[backbone_cfg.name],
                        head_name=head_cfg.name,
                        head_kwargs=head_kwargs,
                        unfreeze_last_blocks=(
                            None if backbone_cfg.freeze else backbone_cfg.unfreeze_last_blocks
                        ),
                    ).to(device)
                    if hasattr(model.backbone, 'dataset_index'):
                        model.backbone.dataset_index = int(backbone_cfg.dataset_index)
                    return model
                case _:
                    raise ValueError(
                        f'Unsupported head type for backbone {backbone_cfg.name}: {head_cfg.type}'
                    )
        case _:
            raise ValueError(f'Unsupported backbone type: {backbone_cfg.name}')


def build_optimizer(model, training_cfg: TrainingConfig, backbone_cfg: BackboneConfig):
    """AdamW with optional lower backbone LR and optional higher pair-bias LR."""
    head_lr = training_cfg.learning_rate
    pair_bias_lr = getattr(training_cfg, 'pair_bias_learning_rate', None)
    if pair_bias_lr is not None and pair_bias_lr <= 0:
        raise ValueError(f'pair_bias_learning_rate must be > 0, got {pair_bias_lr}')

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    pair_bias_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith('backbone.'):
            continue
        if 'pair_bias_values' in name:
            pair_bias_params.append(param)
        else:
            head_params.append(param)

    if pair_bias_lr is not None and not pair_bias_params:
        print(
            'Warning: pair_bias_learning_rate is set but no trainable '
            'pair_bias_values params were found; ignoring separate pair-bias LR'
        )
        pair_bias_lr = None

    # When a dedicated pair-bias LR is set, keep those scalars out of weight decay.
    use_separate_pair_bias = pair_bias_lr is not None and bool(pair_bias_params)
    if not use_separate_pair_bias and pair_bias_params:
        head_params.extend(pair_bias_params)
        pair_bias_params = []

    param_groups = []
    backbone_lr = None
    if backbone_params:
        backbone_lr = (
            backbone_cfg.learning_rate
            if backbone_cfg.learning_rate is not None
            else head_lr * 0.1
        )
        param_groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})
    if use_separate_pair_bias:
        param_groups.append(
            {
                'params': pair_bias_params,
                'lr': float(pair_bias_lr),
                'weight_decay': 0.0,
            }
        )
        print(
            f'pair_bias LR={float(pair_bias_lr):g} '
            f'(n_tensors={len(pair_bias_params)}, weight_decay=0)'
        )

    if not param_groups:
        raise ValueError('build_optimizer found no trainable parameters')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=training_cfg.weight_decay)
    return optimizer, backbone_lr


def load_test_samples(config):
    """Rebuild the train/test split and return the test person/couple samples."""
    dataset_cfg: DatasetConfig = config['dataset']
    training_cfg: TrainingConfig = config['training']
    annotation_file = get_annotation_file_path(dataset_cfg.annotation_version)
    root = ET.parse(annotation_file).getroot()
    task_ids = [video.get('id') for video in root.findall('./project/videos/video')]
    if COUPLE_MODE:
        full_samples = build_couple_samples(annotation_file, set(task_ids), COUPLE_MIN_IOU)
    else:
        full_samples = build_person_samples(
            annotation_file,
            set(task_ids),
            max_partner_iou=MAX_PARTNER_IOU,
            min_partner_iou=MIN_PARTNER_IOU,
        )
    max_frames_per_clip = getattr(training_cfg, 'max_frames_per_clip', None)
    dance_rolling = bool(getattr(training_cfg, 'rolling_max_frames_per_clip', False))
    if not dance_rolling:
        full_samples = cap_samples_per_clip(
            full_samples,
            max_frames_per_clip,
            training_cfg.seed,
            mode='even',
            log=False,
        )
    _, test_samples, _, _ = split_samples_by_image(
        full_samples,
        training_cfg.train_test_split,
        training_cfg.seed,
        split_mode=training_cfg.split_mode,
        test_clips=getattr(training_cfg, 'test_clips', None),
    )
    if dance_rolling:
        test_samples = cap_samples_per_clip(
            test_samples,
            max_frames_per_clip,
            training_cfg.seed + 7,
            mode='even',
            log=False,
        )
    return test_samples


def build_skeleton_connections(keypoints_by_name: dict) -> list[tuple[int, int]]:
    connections = []
    for left_name, right_name in SKELETON_BONE_NAME_LINKS:
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


def _joint_oks_vector(
    pred: np.ndarray,
    gt: np.ndarray,
    sigmas: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Per-joint OKS values; NaN when visibility == 0."""
    out = np.full((len(gt),), np.nan, dtype=np.float32)
    for index in range(len(gt)):
        if gt[index, 2] <= 0:
            continue
        dist_sq = float(np.sum((pred[index] - gt[index, :2]) ** 2))
        denom = 2.0 * float(scale) ** 2 * float(sigmas[index]) ** 2 + 1e-9
        out[index] = float(np.exp(-dist_sq / denom))
    return out


@torch.inference_mode()
def collect_val_epoch_samples(model, samples, device, batch_size) -> list[dict[str, Any]]:
    """All val samples with pred/gt coords, SimCC loss, and OKS (when sigmas exist)."""
    model.eval()
    collected: list[dict[str, Any]] = []
    sigmas = (
        OKS_SIGMAS.detach().cpu().numpy()
        if OKS_SIGMAS.numel() == NUM_JOINTS
        else None
    )

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        pixels_batch = []
        keypoints_batch = []
        role_batch = []
        spatial_batch = []
        prepared = []

        for sample in batch:
            if COUPLE_MODE:
                (
                    image_path,
                    lead_kps,
                    follow_kps,
                    union_box,
                    lead_bbox,
                    follow_bbox,
                    iou,
                ) = sample
                image = Image.open(image_path).convert('RGB')
                crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
                    union_box, image.width, image.height
                )
                crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
                crop = crop.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
                pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
                lead = transform_keypoints_to_crop(
                    lead_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                follow = transform_keypoints_to_crop(
                    follow_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                crop_keypoints = np.concatenate([lead, follow], axis=0)
                role_bits = torch.zeros(ROLE_DIM, dtype=torch.float32)
                spatial_bits = torch.zeros(2, dtype=torch.float32)
                lead_scale = person_scale_in_crop(
                    lead_bbox, crop_width, crop_height, IMAGE_SIZE.x, IMAGE_SIZE.y
                )
                follow_scale = person_scale_in_crop(
                    follow_bbox, crop_width, crop_height, IMAGE_SIZE.x, IMAGE_SIZE.y
                )
                prepared.append(
                    {
                        'image_path': Path(image_path),
                        'bbox': union_box,
                        'role': None,
                        'lead_scale': lead_scale,
                        'follow_scale': follow_scale,
                        'iou': iou,
                    }
                )
            else:
                image_path, keypoints, bbox, role = sample[:4]
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
                role_bits = encode_role(role)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                spatial_bits = crop_normalized_bbox_center(
                    cx, cy, crop_x1, crop_y1, crop_width, crop_height
                )
                prepared.append(
                    {
                        'image_path': Path(image_path),
                        'bbox': bbox,
                        'role': role,
                        'scale': person_scale_in_crop(
                            bbox, crop_width, crop_height, IMAGE_SIZE.x, IMAGE_SIZE.y
                        ),
                    }
                )

            pixels_batch.append(pixels)
            keypoints_batch.append(torch.from_numpy(crop_keypoints))
            role_batch.append(role_bits)
            spatial_batch.append(spatial_bits)

        pixels_tensor = torch.stack(pixels_batch).to(device)
        keypoints_tensor = torch.stack(keypoints_batch)
        role_tensor = torch.stack(role_batch).to(device)
        spatial_tensor = torch.stack(spatial_batch).to(device)
        target_x, target_y, weights = make_targets(keypoints_tensor, device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y, _ = forward_model(model, pixels_tensor, role_tensor, spatial_tensor)
            sample_losses = per_sample_simcc_loss(pred_x, pred_y, target_x, target_y, weights)
            coords, _joint_conf = decode_simcc(pred_x, pred_y, SIMCC_SPLIT_RATIO)

        coords_np = coords.detach().cpu().numpy()
        keypoints_np = keypoints_tensor.numpy()
        losses_np = sample_losses.detach().cpu().numpy()

        for offset, meta in enumerate(prepared):
            image_path = meta['image_path']
            if COUPLE_MODE:
                # Store full concatenated pose; OKS is mean of lead/follow person OKS.
                pred = coords_np[offset]
                gt = keypoints_np[offset]
                loss = float(losses_np[offset])
                oks_val = None
                joint_oks = np.full((pred.shape[0],), np.nan, dtype=np.float32)
                if sigmas is not None:
                    lead_oks = compute_oks(
                        pred[:PERSON_JOINTS],
                        gt[:PERSON_JOINTS],
                        sigmas,
                        meta['lead_scale'],
                    )
                    follow_oks = compute_oks(
                        pred[PERSON_JOINTS:],
                        gt[PERSON_JOINTS:],
                        sigmas,
                        meta['follow_scale'],
                    )
                    vals = [v for v in (lead_oks, follow_oks) if v is not None]
                    oks_val = float(np.mean(vals)) if vals else None
                    joint_oks[:PERSON_JOINTS] = _joint_oks_vector(
                        pred[:PERSON_JOINTS], gt[:PERSON_JOINTS], sigmas, meta['lead_scale']
                    )
                    joint_oks[PERSON_JOINTS:] = _joint_oks_vector(
                        pred[PERSON_JOINTS:],
                        gt[PERSON_JOINTS:],
                        sigmas,
                        meta['follow_scale'],
                    )
            else:
                pred = coords_np[offset]
                gt = keypoints_np[offset]
                loss = float(losses_np[offset])
                oks_val = None
                joint_oks = np.full((pred.shape[0],), np.nan, dtype=np.float32)
                if sigmas is not None:
                    oks_val = compute_oks(pred, gt, sigmas, meta['scale'])
                    oks_val = float(oks_val) if oks_val is not None else None
                    joint_oks = _joint_oks_vector(pred, gt, sigmas, meta['scale'])

            collected.append(
                {
                    'index': start + offset,
                    'image_path': str(image_path),
                    'image_stem': image_path.stem,
                    'bbox': tuple(float(v) for v in meta['bbox']),
                    'role': meta['role'],
                    'loss': loss,
                    'oks': oks_val,
                    'pred_xy': pred.astype(np.float32),
                    'gt_xy': gt[:, :2].astype(np.float32),
                    'visibility': gt[:, 2].astype(np.int16),
                    'joint_oks': joint_oks,
                }
            )

    return collected


def export_epoch_val_artifacts(
    model,
    test_samples,
    run_dir: Path,
    epoch: int,
    device,
    batch_size: int,
    overlay_top_n: int,
    watchlist_frames: list[str] | None,
) -> list[dict[str, Any]]:
    """Write val_preds npz + worst/watchlist overlays for one epoch."""
    samples = collect_val_epoch_samples(model, test_samples, device, batch_size)
    run_artifacts.save_val_preds_npz(run_dir, epoch, samples)

    by_stem = {item['image_stem']: item for item in samples}
    ranked = sorted(samples, key=lambda item: item['loss'], reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked[: max(0, overlay_top_n)]:
        key = f"{item['image_stem']}|{item['index']}"
        if key not in seen:
            selected.append(item)
            seen.add(key)
    for stem in watchlist_frames or []:
        item = by_stem.get(stem)
        if item is None:
            continue
        key = f"{item['image_stem']}|{item['index']}"
        if key not in seen:
            selected.append(item)
            seen.add(key)

    if selected:
        out_dir = run_artifacts.val_overlays_dir(run_dir, epoch)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = [f'epoch: {epoch}', f'overlays: {len(selected)}', '']
        for rank, item in enumerate(selected, start=1):
            tag = 'watch' if (watchlist_frames and item['image_stem'] in watchlist_frames) else 'worst'
            oks_part = (
                f'_oks_{item["oks"]:.4f}' if item.get('oks') is not None else ''
            )
            name = (
                f'{rank:02d}_{tag}_loss_{item["loss"]:.4f}'
                f'{oks_part}_{item["image_stem"]}.jpg'
            )
            draw_prediction_overlay(
                item['image_path'],
                item['bbox'],
                item['pred_xy'],
                out_dir / name,
                loss=item['loss'],
            )
            summary.append(
                f'{rank:02d}\t{tag}\tloss={item["loss"]:.6f}\t'
                f'oks={item.get("oks")}\t{item["image_path"]}'
            )
        (out_dir / 'overlay_summary.txt').write_text(
            '\n'.join(summary) + '\n', encoding='utf-8'
        )
    return samples


def val_samples_json_records(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-friendly records for end-of-run val_samples.json."""
    records = []
    for item in samples:
        joint_oks = item['joint_oks']
        records.append(
            {
                'index': item['index'],
                'image_path': item['image_path'],
                'image_stem': item['image_stem'],
                'bbox': list(item['bbox']),
                'role': item['role'],
                'loss': float(item['loss']),
                'oks': None if item.get('oks') is None else float(item['oks']),
                'pred_xy': np.asarray(item['pred_xy']).tolist(),
                'gt_xy': np.asarray(item['gt_xy']).tolist(),
                'visibility': np.asarray(item['visibility']).astype(int).tolist(),
                'joint_oks': [
                    None if not np.isfinite(v) else float(v)
                    for v in np.asarray(joint_oks).tolist()
                ],
            }
        )
    return records


@torch.inference_mode()
def rank_test_samples_by_loss(model, samples, device, batch_size):
    """Score every test sample by SimCC loss; highest loss first."""
    model.eval()
    ranked = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        pixels_batch = []
        keypoints_batch = []
        role_batch = []
        spatial_batch = []
        prepared = []

        for sample in batch:
            if COUPLE_MODE:
                (
                    image_path,
                    lead_kps,
                    follow_kps,
                    union_box,
                    lead_bbox,
                    follow_bbox,
                    iou,
                ) = sample
                image = Image.open(image_path).convert('RGB')
                crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
                    union_box, image.width, image.height
                )
                crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
                crop = crop.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
                pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
                lead = transform_keypoints_to_crop(
                    lead_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                follow = transform_keypoints_to_crop(
                    follow_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                crop_keypoints = np.concatenate([lead, follow], axis=0)
                role_bits = torch.zeros(ROLE_DIM, dtype=torch.float32)
                spatial_bits = torch.zeros(2, dtype=torch.float32)
                prepared.append(
                    {
                        'image_path': Path(image_path),
                        'bbox': union_box,
                        'lead_bbox': lead_bbox,
                        'follow_bbox': follow_bbox,
                        'iou': iou,
                        'role': None,
                    }
                )
            else:
                image_path, keypoints, bbox, role = sample[:4]
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
                role_bits = encode_role(role)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                spatial_bits = crop_normalized_bbox_center(
                    cx, cy, crop_x1, crop_y1, crop_width, crop_height
                )
                prepared.append(
                    {
                        'image_path': Path(image_path),
                        'bbox': bbox,
                        'role': role,
                    }
                )

            pixels_batch.append(pixels)
            keypoints_batch.append(torch.from_numpy(crop_keypoints))
            role_batch.append(role_bits)
            spatial_batch.append(spatial_bits)

        pixels_tensor = torch.stack(pixels_batch).to(device)
        keypoints_tensor = torch.stack(keypoints_batch)
        role_tensor = torch.stack(role_batch).to(device)
        spatial_tensor = torch.stack(spatial_batch).to(device)
        target_x, target_y, weights = make_targets(keypoints_tensor, device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y, _ = forward_model(model, pixels_tensor, role_tensor, spatial_tensor)
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
            item = {
                'index': sample_index,
                'loss': float(loss),
                'image_path': prepared[offset]['image_path'],
                'bbox': prepared[offset]['bbox'],
                'role': prepared[offset]['role'],
                'coordinates': coordinates[offset],
            }
            if COUPLE_MODE:
                item['lead_bbox'] = prepared[offset]['lead_bbox']
                item['follow_bbox'] = prepared[offset]['follow_bbox']
                item['iou'] = prepared[offset]['iou']
            ranked.append(item)

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

    connections = build_skeleton_connections(KEYPOINTS_BY_NAME)
    banks = (
        [(0, PERSON_JOINTS, 'lime', 'red'), (PERSON_JOINTS, 2 * PERSON_JOINTS, 'cyan', 'magenta')]
        if COUPLE_MODE and len(coordinates) == 2 * PERSON_JOINTS
        else [(0, len(coordinates), 'lime', 'red')]
    )

    for start, end, line_color, point_color in banks:
        bank_coords = coordinates[start:end]
        points = []
        for x, y in bank_coords:
            points.append(
                (
                    crop_x1 + float(x) * crop_width / IMAGE_SIZE.x,
                    crop_y1 + float(y) * crop_height / IMAGE_SIZE.y,
                )
            )
        point_by_id = dict(zip(KEYPOINT_IDS, points))
        for first_id, second_id in connections:
            first = point_by_id.get(first_id)
            second = point_by_id.get(second_id)
            if first is not None and second is not None:
                draw.line((first, second), fill=line_color, width=1)

        for index, (x, y) in enumerate(points):
            radius = 2
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=point_color,
                outline='white',
            )
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


def _ordered_joint_names() -> list[str]:
    ordered = sorted(KEYPOINTS_BY_NAME.items(), key=lambda item: item[1]['id'])
    return [name for name, _ in ordered]


@torch.inference_mode()
def export_query_attn_maps(
    model,
    checkpoint_path,
    test_samples,
    run_dir: Path,
    checkpoint_name: str,
    top_n: int,
    watchlist_frames: list[str] | None,
    device,
    batch_size: int,
) -> list[Path]:
    """Save last-layer cross-attn [K,H,W] for worst + watchlist under ``{best,final}/attn``."""
    from src.training.dino_cc import _BaseJointQuerySimCCHead

    if COUPLE_MODE:
        print(f'  [{checkpoint_name}] skip query attn (couple mode)')
        return []
    if not isinstance(getattr(model, 'head', None), _BaseJointQuerySimCCHead):
        print(f'  [{checkpoint_name}] skip query attn (not a joint-query head)')
        return []

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ranked = rank_test_samples_by_loss(model, test_samples, device, batch_size)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked[: max(0, min(top_n, len(ranked)))]:
        stem = item['image_path'].stem
        if stem not in seen:
            selected.append(item)
            seen.add(stem)
    stem_to_item = {item['image_path'].stem: item for item in ranked}
    for stem in watchlist_frames or []:
        item = stem_to_item.get(stem)
        if item is None or stem in seen:
            continue
        selected.append(item)
        seen.add(stem)

    joint_names = _ordered_joint_names()
    grid_w = int(model.grid_size.x)
    grid_h = int(model.grid_size.y)
    out_paths: list[Path] = []
    attn_root = run_artifacts.attn_dir(run_dir, checkpoint_name)
    if attn_root.exists():
        shutil.rmtree(attn_root)
    attn_root.mkdir(parents=True, exist_ok=True)

    # One-at-a-time to keep attn weight memory bounded on small GPUs.
    for item in selected:
        sample = test_samples[item['index']]
        image_path, keypoints, bbox, role = sample[:4]
        image = Image.open(image_path).convert('RGB')
        crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
            bbox, image.width, image.height
        )
        crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
        crop = crop.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
        pixels = ((pixels - IMAGE_MEAN) / IMAGE_STD).unsqueeze(0).to(device)
        role_tensor = encode_role(role).unsqueeze(0).to(device)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        spatial_tensor = crop_normalized_bbox_center(
            cx, cy, crop_x1, crop_y1, crop_width, crop_height
        ).unsqueeze(0).to(device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            out = model(
                pixels,
                role=role_tensor if USE_ROLE else None,
                spatial=spatial_tensor if USE_SPATIAL_PROMPT else None,
                return_cross_attn=True,
            )
        if len(out) == 4:
            pred_x, pred_y, _aux, cross_w = out
        else:
            pred_x, pred_y, cross_w = out
        if cross_w is None:
            print(f'  [{checkpoint_name}] no cross-attn for {Path(image_path).stem}')
            continue
        # cross_w: [1, K, N] → [K, H, W] with H=grid_h, W=grid_w (row-major)
        attn = cross_w[0].detach().float().cpu().numpy().reshape(-1, grid_h, grid_w)
        pred_xy = (
            torch.stack(
                (
                    pred_x.argmax(dim=-1).float() / SIMCC_SPLIT_RATIO,
                    pred_y.argmax(dim=-1).float() / SIMCC_SPLIT_RATIO,
                ),
                dim=-1,
            )[0]
            .cpu()
            .numpy()
        )
        stem = Path(image_path).stem
        path = run_artifacts.save_attn_npz(
            run_dir,
            checkpoint_name,
            stem=stem,
            attn_khw=attn,
            joint_names=joint_names,
            image_path=str(image_path),
            bbox=bbox,
            pred_xy=pred_xy,
            grid_wh=(grid_w, grid_h),
        )
        out_paths.append(path)
        print(f'  [{checkpoint_name}] attn {stem} -> {path.name}')

    (attn_root / 'attn_summary.txt').write_text(
        '\n'.join(
            [
                f'checkpoint: {checkpoint_path}',
                f'samples: {len(out_paths)}',
                f'grid: {grid_w}x{grid_h}',
                f'joints: {len(joint_names)}',
                '',
                *[p.name for p in out_paths],
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    return out_paths


@torch.inference_mode()
def evaluate_checkpoint_metrics(
    model,
    checkpoint_path,
    test_samples,
    device,
    batch_size,
    annotation_version: str | None = None,
):
    """Score best/final checkpoint with OKS-AP (primary) and custom suites.

    If ``checkpoint_path`` is None, evaluates the current in-memory weights
    (used for per-epoch AP checkpoint selection).
    """
    if OKS_SIGMAS.numel() == 0:
        raise ValueError('OKS_SIGMAS is empty; keypoint format must define oks_sigmas')

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sigmas = OKS_SIGMAS.detach().cpu().numpy()
    part_indices = resolve_body_part_indices(KEYPOINTS_BY_NAME)
    records = []
    ownership_records: list[dict[str, Any]] = []

    frame_index = None
    if not COUPLE_MODE and annotation_version:
        # Lazy import avoids train <-> eval_ownership cycle at module load.
        from src.model_review.eval_ownership import build_frame_index

        annotation_file = get_annotation_file_path(annotation_version)
        root = ET.parse(annotation_file).getroot()
        task_ids = {
            task_id
            for video in root.findall('./project/videos/video')
            if (task_id := video.get('id')) is not None
        }
        frame_index = build_frame_index(annotation_file, task_ids)

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        pixels_batch = []
        keypoints_batch = []
        role_batch = []
        spatial_batch = []
        meta = []

        for sample in batch:
            if COUPLE_MODE:
                (
                    image_path,
                    lead_kps,
                    follow_kps,
                    union_box,
                    lead_bbox,
                    follow_bbox,
                    _,
                ) = sample
                image = Image.open(image_path).convert('RGB')
                crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
                    union_box, image.width, image.height
                )
                crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
                crop = crop.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
                pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
                pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
                lead = transform_keypoints_to_crop(
                    lead_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                follow = transform_keypoints_to_crop(
                    follow_kps, crop_x1, crop_y1, crop_width, crop_height
                )
                crop_keypoints = np.concatenate([lead, follow], axis=0)
                role_bits = torch.zeros(ROLE_DIM, dtype=torch.float32)
                spatial_bits = torch.zeros(2, dtype=torch.float32)
                lead_scale = person_scale_in_crop(
                    lead_bbox, crop_width, crop_height, IMAGE_SIZE.x, IMAGE_SIZE.y
                )
                follow_scale = person_scale_in_crop(
                    follow_bbox, crop_width, crop_height, IMAGE_SIZE.x, IMAGE_SIZE.y
                )
                meta.append(
                    {
                        'image_path': image_path,
                        'lead_scale': lead_scale,
                        'follow_scale': follow_scale,
                    }
                )
            else:
                image_path, keypoints, bbox, role = sample[:4]
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
                role_bits = encode_role(role)
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                spatial_bits = crop_normalized_bbox_center(
                    cx, cy, crop_x1, crop_y1, crop_width, crop_height
                )
                meta.append(
                    {
                        'image_path': image_path,
                        'bbox': bbox,
                        'role': role,
                        'crop_kps': crop_keypoints,
                        'crop_geom': (crop_x1, crop_y1, crop_width, crop_height),
                        'scale': person_scale_in_crop(
                            bbox,
                            crop_width,
                            crop_height,
                            IMAGE_SIZE.x,
                            IMAGE_SIZE.y,
                        ),
                    }
                )

            pixels_batch.append(pixels)
            keypoints_batch.append(torch.from_numpy(crop_keypoints))
            role_batch.append(role_bits)
            spatial_batch.append(spatial_bits)

        pixels_tensor = torch.stack(pixels_batch).to(device)
        keypoints_tensor = torch.stack(keypoints_batch)
        role_tensor = torch.stack(role_batch).to(device)
        spatial_tensor = torch.stack(spatial_batch).to(device)
        target_x, target_y, weights = make_targets(keypoints_tensor, device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y, _ = forward_model(model, pixels_tensor, role_tensor, spatial_tensor)
            sample_losses = per_sample_simcc_loss(pred_x, pred_y, target_x, target_y, weights)
            coords, joint_conf = decode_simcc(pred_x, pred_y, SIMCC_SPLIT_RATIO)

        coords_np = coords.detach().cpu().numpy()
        conf_np = joint_conf.detach().cpu().numpy()
        keypoints_np = keypoints_tensor.numpy()
        losses_np = sample_losses.detach().cpu().numpy()
        pred_x_np = pred_x.detach().float().cpu().numpy()
        pred_y_np = pred_y.detach().float().cpu().numpy()

        for offset, item in enumerate(meta):
            image_path = item['image_path']
            video_id = clip_id_from_image_path(image_path)
            if COUPLE_MODE:
                for role_name, bank_slice, scale in (
                    ('lead', slice(0, PERSON_JOINTS), item['lead_scale']),
                    ('follow', slice(PERSON_JOINTS, 2 * PERSON_JOINTS), item['follow_scale']),
                ):
                    gt = keypoints_np[offset, bank_slice]
                    pred = coords_np[offset, bank_slice]
                    conf = conf_np[offset, bank_slice]
                    records.append(
                        _pose_metric_record(
                            video_id=f'{video_id}:{role_name}',
                            pred=pred,
                            gt=gt,
                            conf=conf,
                            sigmas=sigmas,
                            scale=scale,
                            loss=float(losses_np[offset]) * 0.5,
                            part_indices=part_indices,
                        )
                    )
            else:
                gt = keypoints_np[offset]
                records.append(
                    _pose_metric_record(
                        video_id=video_id,
                        pred=coords_np[offset],
                        gt=gt,
                        conf=conf_np[offset],
                        sigmas=sigmas,
                        scale=item['scale'],
                        loss=float(losses_np[offset]),
                        part_indices=part_indices,
                    )
                )
                if frame_index is not None:
                    ownership_records.append(
                        _ownership_metric_record(
                            item=item,
                            pred_xy=coords_np[offset],
                            pred_x=pred_x_np[offset],
                            pred_y=pred_y_np[offset],
                            sigmas=sigmas,
                            frame_index=frame_index,
                        )
                    )

    ownership_summary = None
    if ownership_records:
        from src.model_review.ownership_metrics import summarize_ownership_records

        ownership_summary = summarize_ownership_records(ownership_records)

    keypoint_names = [
        name
        for name, _ in sorted(KEYPOINTS_BY_NAME.items(), key=lambda item: item[1]['id'])
    ]
    return summarize_pose_metrics(
        records,
        ownership=ownership_summary,
        keypoint_names=keypoint_names,
    )


def _pose_metric_record(
    video_id: str,
    pred: np.ndarray,
    gt: np.ndarray,
    conf: np.ndarray,
    sigmas: np.ndarray,
    scale: float,
    loss: float,
    part_indices: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Primary OKS-AP fields plus visibility-split / body-part OKS for custom metrics."""
    n_visible = int(np.sum(gt[:, 2] == 2))
    n_occluded = int(np.sum(gt[:, 2] == 1))
    record = {
        'video_id': video_id,
        'oks': compute_oks(pred, gt, sigmas, scale),
        'loss': loss,
        'score': joint_score_for_visibility(conf, gt),
        'oks_visible': compute_oks(pred, gt, sigmas, scale, visibility_eq=2),
        'score_visible': joint_score_for_visibility(conf, gt, visibility_eq=2),
        'n_visible': n_visible,
        'oks_occluded': compute_oks(pred, gt, sigmas, scale, visibility_eq=1),
        'score_occluded': joint_score_for_visibility(conf, gt, visibility_eq=1),
        'n_occluded': n_occluded,
        'per_keypoints': [
            {
                'oks': (
                    float(np.exp(
                        -float(np.sum((pred[index] - gt[index, :2]) ** 2))
                        / (2.0 * float(scale) ** 2 * float(sigmas[index]) ** 2 + 1e-9)
                    ))
                    if gt[index, 2] > 0
                    else None
                ),
                'score': float(conf[index]),
                'visibility': int(gt[index, 2]),
            }
            for index in range(len(gt))
        ],
    }
    if part_indices:
        record['parts'] = body_part_oks_fields(
            pred, gt, conf, sigmas, scale, part_indices
        )
    return record


def _ownership_metric_record(
    item: dict[str, Any],
    pred_xy: np.ndarray,
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    sigmas: np.ndarray,
    frame_index: dict,
    swap_partner_oks_floor: float = 0.25,
    collapse_self_tau: float = 0.5,
) -> dict[str, Any]:
    """One-sample swap / collapse / xy-mix record (same defs as eval_ownership)."""
    from src.model_review.eval_ownership import is_same_person
    from src.model_review.ownership_metrics import iou_bucket, joint_xy_mix_fraction, pick_partner

    image_path = Path(item['image_path']).resolve()
    self_bbox = item['bbox']
    people = frame_index.get(image_path, [])
    if not people:
        # Fall back to unresolved path keys from older indexes.
        people = frame_index.get(Path(item['image_path']), [])
    others = [
        (kps, bbox, role)
        for kps, bbox, role in people
        if not is_same_person(bbox, self_bbox)
    ]
    crop_kps = align_keypoints_to_num_joints(item['crop_kps'], pred_xy.shape[0])
    oks_self = compute_oks(pred_xy, crop_kps, sigmas, item['scale'])
    partner_ctx = pick_partner(self_bbox, others)
    if partner_ctx is None:
        return {
            'image_path': str(image_path),
            'role': item['role'],
            'iou_bucket': iou_bucket(0.0),
            'max_partner_iou': 0.0,
            'has_partner': False,
            'swap': False,
            'collapse': False,
            'oks_self': float(oks_self) if oks_self is not None else 0.0,
            'oks_partner': 0.0,
            'xy_mix_rate': 0.0,
            'xy_mix_joints': 0,
        }

    cx1, cy1, cw, ch = item['crop_geom']
    partner_kps = align_keypoints_to_num_joints(
        partner_ctx.partner_kps, pred_xy.shape[0]
    )
    partner_crop = transform_keypoints_to_crop(partner_kps, cx1, cy1, cw, ch)
    partner_scale = person_scale_in_crop(
        partner_ctx.partner_bbox, cw, ch, IMAGE_SIZE.x, IMAGE_SIZE.y
    )
    oks_partner = compute_oks(pred_xy, partner_crop, sigmas, partner_scale)
    oks_self_f = float(oks_self) if oks_self is not None else 0.0
    oks_partner_f = float(oks_partner) if oks_partner is not None else 0.0
    swap = oks_partner_f > oks_self_f and oks_partner_f >= swap_partner_oks_floor
    collapse = oks_self_f < collapse_self_tau and oks_partner_f > oks_self_f
    xy_mix_rate, xy_n = joint_xy_mix_fraction(
        pred_x, pred_y, crop_kps, partner_crop, SIMCC_SPLIT_RATIO
    )
    return {
        'image_path': str(image_path),
        'role': item['role'],
        'iou_bucket': iou_bucket(partner_ctx.max_partner_iou),
        'max_partner_iou': partner_ctx.max_partner_iou,
        'has_partner': True,
        'swap': bool(swap),
        'collapse': bool(collapse),
        'oks_self': oks_self_f,
        'oks_partner': oks_partner_f,
        'xy_mix_rate': float(xy_mix_rate),
        'xy_mix_joints': int(xy_n),
    }


def export_checkpoint_metrics(
    model,
    checkpoint_path,
    test_samples,
    output_dir,
    device,
    batch_size,
    annotation_version: str | None = None,
):
    metrics = evaluate_checkpoint_metrics(
        model,
        checkpoint_path,
        test_samples,
        device,
        batch_size,
        annotation_version=annotation_version,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(metrics, output_dir / 'metrics.json')
    print(format_metrics_summary(output_dir.name, metrics))
    return metrics


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
    if COUPLE_MODE:
        full_samples = build_couple_samples(annotation_file, set(task_ids), COUPLE_MIN_IOU)
    else:
        full_samples = build_person_samples(
            annotation_file,
            set(task_ids),
            max_partner_iou=MAX_PARTNER_IOU,
            min_partner_iou=MIN_PARTNER_IOU,
        )
    max_frames_per_clip = getattr(training_cfg, 'max_frames_per_clip', None)
    dance_rolling = bool(getattr(training_cfg, 'rolling_max_frames_per_clip', False))
    if dance_rolling:
        if max_frames_per_clip is None:
            raise ValueError(
                'rolling_max_frames_per_clip=true requires training.max_frames_per_clip'
            )
        if TEMPORAL_MODE:
            raise ValueError(
                'rolling_max_frames_per_clip is incompatible with temporal windows; '
                'set max_frames_per_clip=null for temporal runs or disable rolling'
            )
    else:
        # Fixed even-spaced cap before the split (historical behavior).
        full_samples = cap_samples_per_clip(
            full_samples,
            max_frames_per_clip,
            training_cfg.seed,
            mode='even',
        )
    coco_rolling = False
    coco_pool: list = []
    n_coco_train = 0
    coco_train_paths: set = set()
    dance_train_pool: list = []
    dance_train_only: list = []

    if TEMPORAL_MODE:
        if COUPLE_MODE:
            raise ValueError('Temporal training is not supported with the couple head')
        all_windows = build_contiguous_windows(
            full_samples, WINDOW_SIZE, window_stride=WINDOW_STRIDE
        )
        sources = summarize_window_sources(all_windows)
        print(
            f'Temporal windows: T={WINDOW_SIZE} stride={WINDOW_STRIDE} '
            f'total={len(all_windows)} by_clip={sources}'
        )
        if len(all_windows) < MIN_TEMPORAL_WINDOWS:
            raise ValueError(
                f'Need at least {MIN_TEMPORAL_WINDOWS} contiguous windows for temporal '
                f'training, found {len(all_windows)}. Densify more clips (every-frame labels) '
                f'or lower training.min_temporal_windows. Sources: {sources}'
            )
        train_windows, test_windows = split_windows_stratified(
            all_windows, training_cfg.train_test_split, training_cfg.seed
        )
        print(
            f'Temporal split: train_windows={len(train_windows)} '
            f'test_windows={len(test_windows)} '
            f'(chronological holdout per track, ratio={training_cfg.train_test_split})'
        )
        train_samples = train_windows
        test_samples = test_windows
        # Single-frame views of window centers for AP / overlays / ownership.
        metric_test_samples = [window[len(window) // 2] for window in test_windows]
        n_dance_train = len(train_windows)
        train_image_paths = [w[len(w) // 2][0] for w in train_windows]
        test_image_paths = [w[len(w) // 2][0] for w in test_windows]
        split_label = (
            f'{training_cfg.train_test_split:.2f} by temporal_window '
            f'T={WINDOW_SIZE}'
        )
    else:
        train_samples, test_samples, train_image_paths, test_image_paths = split_samples_by_image(
            full_samples,
            training_cfg.train_test_split,
            training_cfg.seed,
            split_mode=training_cfg.split_mode,
            test_clips=getattr(training_cfg, 'test_clips', None),
        )
        if dance_rolling:
            # Stable val: fixed even-spaced cap. Train pool stays full; subsampled each epoch.
            test_samples = cap_samples_per_clip(
                test_samples,
                max_frames_per_clip,
                training_cfg.seed + 7,
                mode='even',
            )
            test_image_paths = sorted({sample[0] for sample in test_samples})
            dance_train_pool = list(train_samples)
            train_samples = cap_samples_per_clip(
                dance_train_pool,
                max_frames_per_clip,
                training_cfg.seed + 1000,
                mode='random',
            )
            train_image_paths = sorted({sample[0] for sample in train_samples})
            print(
                f'Dance rolling: max_frames_per_clip={max_frames_per_clip}; '
                f'train_pool_person_samples={len(dance_train_pool)} '
                f'epoch_train_person_samples={len(train_samples)} '
                f'test_person_samples={len(test_samples)}'
            )
        n_dance_train = len(train_samples)
        coco_cfg = getattr(dataset_cfg, 'coco_wholebody', None)
        coco_rolling = bool(coco_cfg is not None and coco_cfg.enabled and coco_cfg.rolling)
        dance_train_only = list(train_samples)
        if coco_cfg is not None and coco_cfg.enabled:
            if COUPLE_MODE:
                raise ValueError(
                    'coco_wholebody mixing requires a single-person head '
                    '(not couple_joint_query_self_attn_simcc)'
                )
            target_coco = target_coco_count(n_dance_train, coco_cfg.dance_fraction)
            # Keep a full COCO pool whenever either side rolls so remix can hit the
            # target dance_fraction as dance size changes epoch-to-epoch.
            remix_each_epoch = coco_rolling or dance_rolling
            build_target = None if remix_each_epoch else (target_coco if target_coco > 0 else None)
            coco_pool = build_coco_wholebody_samples(
                coco_cfg,
                head_cfg.keypoints,
                PERSON_JOINTS,
                seed=training_cfg.seed,
                target_count=build_target,
            )
            if remix_each_epoch:
                gc.collect()
            if dance_rolling and not coco_rolling:
                print(
                    'Dance rolling: building full COCO pool and remixing each epoch '
                    'so dance_fraction stays on target (coco_wholebody.rolling was false)'
                )
            train_samples, n_coco_train, coco_train_paths = mix_dance_with_coco(
                dance_train_only,
                coco_pool,
                coco_cfg.dance_fraction,
                seed=training_cfg.seed + 17 + (1 if remix_each_epoch else 0),
            )
            missing_images = [
                path
                for path, *_ in coco_pool[: min(32, len(coco_pool))]
                if not Path(path).is_file()
            ]
            if missing_images:
                raise FileNotFoundError(
                    f'COCO image(s) missing (showing up to 32): {missing_images[:5]}'
                )
            dance_labeled = mean_labeled_joints(dance_train_only)
            realized = n_dance_train / max(1, len(train_samples))
            print(
                f'COCO mix: rolling={coco_rolling}, dance_rolling={dance_rolling}, '
                f'dance_train={n_dance_train}, '
                f'coco_pool={len(coco_pool)}, coco_per_epoch={n_coco_train}, '
                f'total_train={len(train_samples)}, realized_dance_fraction={realized:.3f} '
                f'(target={coco_cfg.dance_fraction:.3f}); '
                f'mean_labeled_joints dance={dance_labeled:.1f} '
                f'coco_pool={mean_labeled_joints(coco_pool):.1f}'
            )
        split_label = f'{training_cfg.train_test_split:.2f} by {training_cfg.split_mode}'
        if dance_rolling:
            split_label += f' + dance_rolling(max={max_frames_per_clip})'
        if n_coco_train:
            rolling_tag = ', rolling' if coco_rolling else ''
            split_label += (
                f' + coco_wholebody (dance_frac≈{n_dance_train / len(train_samples):.2f}'
                f'{rolling_tag})'
            )
        metric_test_samples = test_samples

    dance_train_image_paths = list(train_image_paths)
    split_index = len(dance_train_image_paths)
    image_paths = dance_train_image_paths + test_image_paths
    train_clips = sorted({clip_id_from_image_path(path) for path in dance_train_image_paths})
    test_clips = sorted({clip_id_from_image_path(path) for path in test_image_paths})

    def _make_temporal_dataset(windows, training: bool):
        return TemporalWindowDataset(
            windows,
            training=training,
            repeats=training_cfg.augmentation_repeats if training else 1,
            image_size=IMAGE_SIZE,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
            bbox_padding=BBOX_PADDING,
            compute_padded_crop=compute_padded_crop,
            transform_keypoints_to_crop=transform_keypoints_to_crop,
            encode_role=encode_role,
            crop_normalized_bbox_center=crop_normalized_bbox_center,
            horizontally_flip_keypoints=horizontally_flip_keypoints,
            rotate_image_and_keypoints=rotate_image_and_keypoints,
            aug_scale_min=AUG_SCALE_MIN,
            aug_scale_max=AUG_SCALE_MAX,
            aug_shift_frac=AUG_SHIFT_FRAC,
            aug_rotation_deg=AUG_ROTATION_DEG,
            aug_color_jitter=AUG_COLOR_JITTER,
        )

    if TEMPORAL_MODE:
        train_dataset = _make_temporal_dataset(train_samples, training=True)
        test_dataset = _make_temporal_dataset(test_samples, training=False)
    else:
        dataset_cls = CoupleKeypointDataset if COUPLE_MODE else KeypointDataset
        train_dataset_kwargs = {
            'annotation_file': annotation_file,
            'task_ids': set(task_ids),
            'training': True,
            'repeats': training_cfg.augmentation_repeats,
            'samples': train_samples,
        }
        if not COUPLE_MODE:
            train_dataset_kwargs['coco_paths'] = coco_train_paths
        train_dataset = dataset_cls(**train_dataset_kwargs)
        test_dataset = dataset_cls(
            annotation_file,
            set(task_ids),
            samples=test_samples,
        )
    # Rolling keeps a large COCO pool in-process. Prefer a couple workers for GPU
    # feed rate; cap hard so we don't multiply the ~130k-sample pool into an OOM.
    remix_coco_each_epoch = bool(coco_pool) and (coco_rolling or dance_rolling)
    train_workers = min(2, args.workers) if remix_coco_each_epoch else args.workers
    if remix_coco_each_epoch and train_workers != args.workers:
        print(
            f'COCO rolling pool: using train_workers={train_workers} '
            f'(requested --workers={args.workers}) with pool={len(coco_pool)}'
        )
    train_loader = make_loader(train_dataset, training_cfg.batch_size, train_workers)
    test_loader = make_loader(test_dataset, training_cfg.batch_size, args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config, device)
    initialization_cfg: InitializationConfig = config.get(
        'initialization', InitializationConfig()
    )
    init_checkpoint = getattr(args, 'init_checkpoint', None) or initialization_cfg.checkpoint
    if init_checkpoint:
        init_path = Path(init_checkpoint).resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f'init checkpoint not found: {init_path}')
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        if TEMPORAL_MODE:
            missing, unexpected = load_state_dict_partial(model, ckpt['model_state_dict'])
            print(
                f'Loaded init checkpoint {init_path} (partial for temporal head) '
                f'(epoch={ckpt.get("epoch", "?")}; missing={len(missing)} '
                f'unexpected={len(unexpected)}; pair-bias noise inits skipped)'
            )
        else:
            missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=True)
            print(
                f'Loaded init checkpoint {init_path} '
                f'(epoch={ckpt.get("epoch", "?")}; missing={len(missing)} unexpected={len(unexpected)}; '
                f'pair-bias noise inits skipped)'
            )
    elif (
        initialization_cfg.pair_bias is not None
        and initialization_cfg.pair_bias.edges
        and getattr(model.head, 'pair_bias_values', None) is not None
    ):
        vals = model.head.pair_bias_values.detach().cpu().tolist()
        print(
            f'Noise init: pair_bias {len(vals)} shared types '
            f'(min={min(vals):.3f}, max={max(vals):.3f})'
        )
    criterion = SimCCLoss().to(device)

    joint_query_heads = (
        'joint_query_simcc',
        'joint_query_self_attn_simcc',
        TEMPORAL_HEAD_NAME,
        COUPLE_HEAD_NAME,
    )
    head_checkpoint_meta = {
        'head_name': head_cfg.name,
        'head_kwargs': (
            build_joint_query_head_kwargs(head_cfg, initialization_cfg)
            if head_cfg.name in joint_query_heads
            else build_depthwise_head_kwargs(head_cfg)
            if head_cfg.name == 'depthwise_simcc'
            else {}
        ),
        'neck_dim': head_cfg.out_channels,
        'split_ratio': SIMCC_SPLIT_RATIO,
        'use_role': USE_ROLE,
        'role_encoding': 'lead_follow_bits',
        'couple_mode': COUPLE_MODE,
        'couple_min_iou': COUPLE_MIN_IOU if COUPLE_MODE else None,
        'max_partner_iou': MAX_PARTNER_IOU,
        'min_partner_iou': MIN_PARTNER_IOU,
        'person_joints': PERSON_JOINTS,
    }


    optimizer, backbone_lr = build_optimizer(model, training_cfg, backbone_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg.max_epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    checkpoint_metric = str(getattr(training_cfg, 'checkpoint_metric', 'val_loss')).lower()
    save_val_preds_every_n = int(getattr(training_cfg, 'save_val_preds_every_n_epochs', 1))
    epoch_overlay_top_n = int(getattr(training_cfg, 'epoch_overlay_top_n', 8))
    watchlist_frames = list(getattr(training_cfg, 'watchlist_frames', None) or [])
    best_val_loss = float('inf')
    epoch = 0
    val_loss = float('inf')
    train_loss = float('inf')
    val_loss_components = {
        'simcc_kl': 0.0,
        'deep_supervision': 0.0,
        'oks': 0.0,
        'bone': 0.0,
        'lr_swap': 0.0,
        'chirality': 0.0,
    }
    best_val_ap = -1.0
    best_priority_oks = -1.0
    epochs_without_improvement = 0
    best_path = run_dir / 'best.pt'
    stop_requested = {'flag': False}

    def _request_stop(signum, _frame):
        stop_requested['flag'] = True
        print(f'Received signal {signum}; will stop after the current epoch.')

    previous_sigint = signal.signal(signal.SIGINT, _request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)
    run_artifacts.write_control(run_dir, 'none')
    run_artifacts.write_status(run_dir, 'running', epoch=0, message='training started')
    couple_msg = (
        f' couple_mode=True couple_min_iou={COUPLE_MIN_IOU} pair_samples={len(full_samples)}'
        if COUPLE_MODE
        else (
            f' max_partner_iou={MAX_PARTNER_IOU}'
            f' min_partner_iou={MIN_PARTNER_IOU}'
            f' person_samples={len(full_samples)}'
            if MAX_PARTNER_IOU is not None or MIN_PARTNER_IOU is not None
            else f' person_samples={len(full_samples)}'
        )
    )
    init_msg = (
        f'; init_checkpoint={Path(init_checkpoint).resolve()}'
        if init_checkpoint
        else ' init=noise'
    )
    bb_train = sum(p.requires_grad for p in model.backbone.parameters())
    bb_total = sum(1 for _ in model.backbone.parameters())
    pair_bias_lr = getattr(training_cfg, 'pair_bias_learning_rate', None)
    backbone_msg = (
        f' backbone={backbone_cfg.name} freeze={backbone_cfg.freeze}'
        f' unfreeze_last_blocks={backbone_cfg.unfreeze_last_blocks}'
        f' trainable_tensors={bb_train}/{bb_total}'
        f'{"" if backbone_lr is None else f" backbone_lr={backbone_lr}"}'
        f'{"" if pair_bias_lr is None else f" pair_bias_lr={pair_bias_lr}"}'
        f'{"" if not dance_rolling else f" dance_rolling=max{max_frames_per_clip}"}'
    )
    print(
        f'Training on {device}; split_mode={training_cfg.split_mode}; '
        f'train images={split_index}/{len(image_paths)}, '
        f'test images={len(image_paths) - split_index}/{len(image_paths)}; '
        f'train clips={train_clips}; test clips={test_clips}; '
        f'use_role={USE_ROLE} role_dropout={ROLE_DROPOUT if USE_ROLE else 0.0}; '
        f'bone_loss_weight={BONE_LOSS_WEIGHT}; '
        f'lr_swap_loss_weight={LR_SWAP_LOSS_WEIGHT}; '
        f'lr_swap_margin={LR_SWAP_MARGIN}; '
        f'lr_swap_scale_frac={LR_SWAP_SCALE_FRAC}; '
        f'lr_swap_foot_pair_weight={LR_SWAP_FOOT_PAIR_WEIGHT}; '
        f'lr_swap_hard_mix={LR_SWAP_HARD_MIX}; '
        f'chirality_loss_weight={CHIRALITY_LOSS_WEIGHT}; '
        f'chirality_min_sep={CHIRALITY_MIN_SEP}; '
        f'chirality_foot_pair_weight={CHIRALITY_FOOT_PAIR_WEIGHT}; '
        f'oks_loss_weight={OKS_LOSS_WEIGHT}; '
        f'checkpoint_metric={checkpoint_metric}; '
        f'lr={training_cfg.learning_rate}; '
        f'simcc_sigma={SIMCC_SIGMA}'
        f'{"" if SIMCC_SIGMA_OCCLUDED is None else f"/{SIMCC_SIGMA_OCCLUDED}_occ"}; '
        f'occ_weight={OCCLUDED_JOINT_WEIGHT}; '
        f'num_joints={NUM_JOINTS}{couple_msg}{backbone_msg}{init_msg}'
    )

    stopped_early = False
    last_val_samples: list[dict[str, Any]] | None = None
    try:
        for epoch in range(1, training_cfg.max_epochs + 1):
            control = run_artifacts.poll_control_between_epochs(
                run_dir,
                epoch=epoch - 1,
                stop_requested=lambda: stop_requested['flag'],
            )
            if control == 'stop':
                print('Stop requested; ending training before next epoch.')
                stopped_early = True
                if epoch > 1:
                    epoch -= 1
                break

            run_artifacts.write_status(
                run_dir, 'running', epoch=epoch, message=f'epoch {epoch} train'
            )
            if dance_rolling or coco_rolling or remix_coco_each_epoch:
                if dance_rolling:
                    dance_train_only = cap_samples_per_clip(
                        dance_train_pool,
                        max_frames_per_clip,
                        training_cfg.seed + 1000 + epoch,
                        mode='random',
                        log=False,
                    )
                    n_dance_train = len(dance_train_only)
                if coco_pool and (coco_rolling or dance_rolling):
                    train_samples, n_coco_train, coco_train_paths = mix_dance_with_coco(
                        dance_train_only,
                        coco_pool,
                        coco_cfg.dance_fraction if coco_cfg is not None else 0.0,
                        seed=training_cfg.seed + 17 + epoch,
                    )
                else:
                    train_samples = list(dance_train_only)
                    n_coco_train = 0
                    coco_train_paths = set()
                train_dataset.samples = train_samples
                if hasattr(train_dataset, 'coco_paths'):
                    train_dataset.coco_paths = coco_train_paths
                train_loader = make_loader(
                    train_dataset, training_cfg.batch_size, train_workers
                )
                parts = [f'epoch={epoch:03d}']
                if dance_rolling:
                    parts.append(
                        f'dance={n_dance_train}/{len(dance_train_pool)}'
                    )
                if coco_pool and (coco_rolling or dance_rolling):
                    parts.append(f'coco={n_coco_train}/{len(coco_pool)}')
                parts.append(f'total_train={len(train_samples)}')
                print('Train rolling: ' + ' '.join(parts))
            train_loss, train_source_losses = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler,
                True,
                training_cfg.gradient_clip_norm,
                track_source_loss=True,
            )
            train_loss_dance = (
                None if train_source_losses is None else train_source_losses.get('dance')
            )
            train_loss_coco = (
                None if train_source_losses is None else train_source_losses.get('coco')
            )
            run_artifacts.write_status(
                run_dir, 'running', epoch=epoch, message=f'epoch {epoch} val'
            )
            val_loss, val_loss_components = run_epoch(
                model,
                test_loader,
                criterion,
                optimizer,
                device,
                scaler,
                False,
                training_cfg.gradient_clip_norm,
                return_components=True,
            )
            scheduler.step()

            val_ap = None
            val_priority = None
            val_part_oks = None
            val_oks_percentile_bands = None
            if checkpoint_metric in ('ap', 'priority_oks'):
                # Skip ownership suite during the epoch loop for speed.
                val_metrics = evaluate_checkpoint_metrics(
                    model,
                    checkpoint_path=None,
                    test_samples=metric_test_samples,
                    device=device,
                    batch_size=training_cfg.batch_size,
                    annotation_version=None,
                )
                val_ap = float(val_metrics['AP'])
                val_priority = priority_oks_score(val_metrics)
                val_part_oks = part_mean_oks_from_metrics(val_metrics)
                val_oks_percentile_bands = val_metrics.get('oks_percentile_bands')
                source_msg = ''
                if train_loss_dance is not None or train_loss_coco is not None:
                    dance_s = (
                        f'{train_loss_dance:.5f}'
                        if train_loss_dance is not None
                        else 'n/a'
                    )
                    coco_s = (
                        f'{train_loss_coco:.5f}' if train_loss_coco is not None else 'n/a'
                    )
                    source_msg = f' | train_simcc dance={dance_s} coco={coco_s}'
                print(
                    f'Epoch {epoch:03d} | train_loss={train_loss:.5f} | '
                    f'val_loss={val_loss:.5f} | val_AP={val_ap:.4f} | '
                    f'val_priority_OKS={val_priority:.4f}{source_msg} | '
                    f'val_components='
                    f'simcc={val_loss_components["simcc_kl"]:.5f},'
                    f'deep={val_loss_components["deep_supervision"]:.5f},'
                    f'oks={val_loss_components["oks"]:.5f},'
                    f'bone={val_loss_components["bone"]:.5f},'
                    f'lr_swap={val_loss_components["lr_swap"]:.5f},'
                    f'chirality={val_loss_components["chirality"]:.5f}'
                )
                if checkpoint_metric == 'priority_oks':
                    improved = val_priority > best_priority_oks + 1e-6
                else:
                    improved = val_ap > best_val_ap + 1e-6
            else:
                source_msg = ''
                if train_loss_dance is not None or train_loss_coco is not None:
                    dance_s = (
                        f'{train_loss_dance:.5f}'
                        if train_loss_dance is not None
                        else 'n/a'
                    )
                    coco_s = (
                        f'{train_loss_coco:.5f}' if train_loss_coco is not None else 'n/a'
                    )
                    source_msg = f' | train_simcc dance={dance_s} coco={coco_s}'
                print(
                    f'Epoch {epoch:03d} | train_loss={train_loss:.5f} | '
                    f'val_loss={val_loss:.5f}{source_msg} | '
                    f'val_components='
                    f'simcc={val_loss_components["simcc_kl"]:.5f},'
                    f'deep={val_loss_components["deep_supervision"]:.5f},'
                    f'oks={val_loss_components["oks"]:.5f},'
                    f'bone={val_loss_components["bone"]:.5f},'
                    f'lr_swap={val_loss_components["lr_swap"]:.5f},'
                    f'chirality={val_loss_components["chirality"]:.5f}'
                )
                improved = val_loss < best_val_loss

            if improved:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if val_ap is not None and val_ap > best_val_ap:
                    best_val_ap = val_ap
                if val_priority is not None and val_priority > best_priority_oks:
                    best_priority_oks = val_priority
                epochs_without_improvement = 0
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'num_joints': NUM_JOINTS,
                        'person_joints': PERSON_JOINTS,
                        'keypoint_ids': KEYPOINT_IDS,
                        'joint_loss_weights': training_cfg.joint_loss_weights,
                        'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
                        'backbone_name': BACKBONE_HF_IDS[backbone_cfg.name],
                        'head_name': head_checkpoint_meta['head_name'],
                        'head_kwargs': head_checkpoint_meta['head_kwargs'],
                        'neck_dim': head_checkpoint_meta['neck_dim'],
                        'split_ratio': head_checkpoint_meta['split_ratio'],
                        'use_role': head_checkpoint_meta['use_role'],
                        'role_encoding': head_checkpoint_meta['role_encoding'],
                        'couple_mode': head_checkpoint_meta['couple_mode'],
                        'couple_min_iou': head_checkpoint_meta['couple_min_iou'],
                        'max_partner_iou': head_checkpoint_meta['max_partner_iou'],
                        'min_partner_iou': head_checkpoint_meta['min_partner_iou'],
                        'config_path': str(run_config_path),
                        'seed': training_cfg.seed,
                        'split': split_label,
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_ap': val_ap,
                        'val_priority_oks': val_priority,
                        'checkpoint_metric': checkpoint_metric,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1

            current_lr = float(optimizer.param_groups[0]['lr'])
            run_artifacts.append_history(
                run_dir,
                {
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'train_loss_dance': train_loss_dance,
                    'train_loss_coco': train_loss_coco,
                    'val_loss': float(val_loss),
                    'val_components': {
                        key: float(val_loss_components[key])
                        for key in (
                            'simcc_kl',
                            'deep_supervision',
                            'oks',
                            'bone',
                            'lr_swap',
                            'chirality',
                        )
                    },
                    'val_ap': val_ap,
                    'val_priority_oks': val_priority,
                    'val_part_oks': val_part_oks,
                    'val_oks_percentile_bands': val_oks_percentile_bands,
                    'lr': current_lr,
                    'is_best': bool(improved),
                    'checkpoint_metric': checkpoint_metric,
                },
            )

            should_save_preds = (
                save_val_preds_every_n > 0
                and epoch % save_val_preds_every_n == 0
            )
            if should_save_preds:
                run_artifacts.write_status(
                    run_dir,
                    'running',
                    epoch=epoch,
                    message=f'epoch {epoch} val preds',
                )
                last_val_samples = export_epoch_val_artifacts(
                    model=model,
                    test_samples=metric_test_samples,
                    run_dir=run_dir,
                    epoch=epoch,
                    device=device,
                    batch_size=training_cfg.batch_size,
                    overlay_top_n=epoch_overlay_top_n,
                    watchlist_frames=watchlist_frames,
                )

            if epochs_without_improvement >= training_cfg.early_stopping_patience:
                print(
                    f'Early stopping after {training_cfg.early_stopping_patience} epochs '
                    f'without {checkpoint_metric} improvement.'
                )
                break

            if stop_requested['flag'] or run_artifacts.read_control(run_dir) == 'stop':
                print('Stop requested; ending training after this epoch.')
                stopped_early = True
                run_artifacts.clear_control(run_dir)
                break

        final_path = run_dir / 'final.pt'
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'num_joints': NUM_JOINTS,
                'person_joints': PERSON_JOINTS,
                'keypoint_ids': KEYPOINT_IDS,
                'joint_loss_weights': training_cfg.joint_loss_weights,
                'image_size': (IMAGE_SIZE.x, IMAGE_SIZE.y),
                'backbone_name': BACKBONE_HF_IDS[backbone_cfg.name],
                'head_name': head_checkpoint_meta['head_name'],
                'head_kwargs': head_checkpoint_meta['head_kwargs'],
                'neck_dim': head_checkpoint_meta['neck_dim'],
                'split_ratio': head_checkpoint_meta['split_ratio'],
                'use_role': head_checkpoint_meta['use_role'],
                'role_encoding': head_checkpoint_meta['role_encoding'],
                'couple_mode': head_checkpoint_meta['couple_mode'],
                'couple_min_iou': head_checkpoint_meta['couple_min_iou'],
                'max_partner_iou': head_checkpoint_meta['max_partner_iou'],
                'min_partner_iou': head_checkpoint_meta['min_partner_iou'],
                'config_path': str(run_config_path),
                'seed': training_cfg.seed,
                'split': split_label,
                'epoch': epoch,
                'test_loss': val_loss,
                'stopped_early': stopped_early,
            },
            final_path,
        )
        print(f'Saved best checkpoint to {best_path}')
        print(f'Saved final checkpoint to {final_path}')

        if last_val_samples is None or (
            save_val_preds_every_n > 0 and epoch % save_val_preds_every_n != 0
        ):
            last_val_samples = export_epoch_val_artifacts(
                model=model,
                test_samples=metric_test_samples,
                run_dir=run_dir,
                epoch=epoch,
                device=device,
                batch_size=training_cfg.batch_size,
                overlay_top_n=epoch_overlay_top_n,
                watchlist_frames=watchlist_frames,
            )
        run_artifacts.save_val_samples_json(
            run_dir, val_samples_json_records(last_val_samples)
        )
        print(f'Wrote {run_dir / run_artifacts.VAL_SAMPLES_NAME}')

        if training_cfg.worst_test_overlays > 0:
            print(
                f'Exporting top {training_cfg.worst_test_overlays} worst test overlays '
                f'for best and final checkpoints...'
            )
            for checkpoint_name, checkpoint_path in (
                ('best', best_path),
                ('final', final_path),
            ):
                if not checkpoint_path.exists():
                    print(f'Skipping {checkpoint_name}: missing {checkpoint_path}')
                    continue
                export_worst_test_overlays(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_samples=metric_test_samples,
                    output_dir=run_dir / checkpoint_name,
                    top_n=training_cfg.worst_test_overlays,
                    device=device,
                    batch_size=training_cfg.batch_size,
                )
                print(
                    f'Exporting query cross-attn maps for {checkpoint_name} '
                    f'(worst {training_cfg.worst_test_overlays} + watchlist)...'
                )
                export_query_attn_maps(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_samples=metric_test_samples,
                    run_dir=run_dir,
                    checkpoint_name=checkpoint_name,
                    top_n=training_cfg.worst_test_overlays,
                    watchlist_frames=watchlist_frames,
                    device=device,
                    batch_size=training_cfg.batch_size,
                )

        if training_cfg.compute_ap:
            if OKS_SIGMAS.numel() == 0:
                raise ValueError(
                    'compute_ap=true requires oks_sigmas in the keypoint format mapping '
                    f'({head_cfg.keypoint_format_name})'
                )
            print('Evaluating OKS-AP on test set for best and final checkpoints...')
            for checkpoint_name, checkpoint_path in (
                ('best', best_path),
                ('final', final_path),
            ):
                if not checkpoint_path.exists():
                    print(f'Skipping AP for {checkpoint_name}: missing {checkpoint_path}')
                    continue
                export_checkpoint_metrics(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    test_samples=metric_test_samples,
                    output_dir=run_dir / checkpoint_name,
                    device=device,
                    batch_size=training_cfg.batch_size,
                    annotation_version=dataset_cfg.annotation_version,
                )

        run_artifacts.write_status(
            run_dir,
            'done',
            epoch=epoch,
            message='stopped early' if stopped_early else 'training complete',
        )
    except Exception as exc:
        run_artifacts.write_status(
            run_dir, 'error', epoch=epoch, message=f'{type(exc).__name__}: {exc}'
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def main():
    parser = ArgumentParser(description='Train DinoCC on the cleaned keypoint dataset.')
    parser.add_argument('--config', default=DEFAULT_POSE_ESTIMATOR_CONFIG_PATH)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help='Load model weights from this .pt (fresh optimizer; for continued training).',
    )
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as config_file:
        raw_config = yaml.safe_load(config_file)
    config = parse_config(raw_config)

    run_dir, run_config_path = create_run_dir(POSE_ESTIMATOR_RUN_DIR, args.config)
    print(f'Run directory: {run_dir.resolve()}')

    log_tee = StdoutCsvTee(sys.stdout, run_dir / 'log.csv')
    sys.stdout = log_tee
    try:
        print(f'Run directory: {run_dir.resolve()}')
        _run_training(args, config, run_dir, run_config_path)
    finally:
        sys.stdout = log_tee.stream
        log_tee.close()


if __name__ == '__main__':
    main()