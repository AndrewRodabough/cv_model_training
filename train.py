import argparse
import csv
import gc
import random
import shutil
import subprocess
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
from metrics import (
    align_keypoints_to_num_joints,
    body_part_oks_fields,
    compute_oks,
    decode_simcc,
    format_metrics_summary,
    joint_score_for_visibility,
    person_scale_in_crop,
    resolve_body_part_indices,
    summarize_pose_metrics,
    write_metrics_json,
)
from coco_wholebody_dataset import (
    CocoWholeBodyConfig,
    build_coco_wholebody_samples,
    mean_labeled_joints,
    mix_dance_with_coco,
    target_coco_count,
)

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
    coco_wholebody: CocoWholeBodyConfig | None = None

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
    # If True, backbone params are frozen. If False, train all (or last N blocks).
    freeze: bool = True
    # When freeze=False: if set, only the last N encoder blocks (+ LN) are trainable.
    unfreeze_last_blocks: int | None = None
    # Optional backbone LR (AdamW param group). None → training.learning_rate when
    # fully frozen head-only; when backbone is trainable defaults to 0.1× head LR.
    learning_rate: float | None = None
    # VitPose++ MoE expert index (COCO=0). Ignored for non-MoE backbones.
    dataset_index: int = 0

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
    oks_sigmas: dict[str, float] | None = None

@dataclass
class SimCCConfig:
    pointwise_convolutions: int
    depthwise_kernel: int
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    pooling: str
    use_role: bool = False
    role_dropout: float = 0.1
    role_film: bool = True
    gaussian_sigma_occluded: float | None = None
    occluded_joint_weight: float = 0.35

@dataclass
class JointQuerySimCCConfig:
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    # Wider SimCC label for occluded (v==1); None → same as gaussian_sigma
    gaussian_sigma_occluded: float | None = None
    # Loss weight for occluded joints (visible always 1.0)
    occluded_joint_weight: float = 0.35
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    # Soft-anchor foot queries to same-side ankle: q_foot += σ(α) * W(q_ankle)
    soft_foot_anchor: bool = False
    soft_foot_anchor_init: float = 0.4
    use_role: bool = False
    role_dropout: float = 0.1
    # If false: query bias only (no FiLM on patch memory)
    role_film: bool = True
    # Per-layer q/k conditioning: role + joint id (+ optional spatial) every decoder layer
    per_layer_role_reinject: bool = False
    use_spatial_prompt: bool = False
    deep_supervision: bool = False
    deep_supervision_aux_weight: float = 0.5
    xy_consistency_loss_weight: float = 0.0
    # Phase B: embed soft-decoded joint (x,y) into queries between decoder layers
    use_joint_pos_embed: bool = False
    # Learned self-attn bias on skeleton edges; L/R mirror edges share params
    use_pair_bias: bool = False
    pair_bias_init: float = 2.0
    # Soft-decoded joint XY → Gaussian bias on cross-attn over patch grid
    use_local_cross_attn: bool = False
    local_attn_sigma: float = 2.0
    local_attn_strength: float = 2.0


@dataclass
class CoupleJointQuerySimCCConfig:
    """Couple head: 2K queries on union crops of lead+follow pairs."""

    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    gaussian_sigma_occluded: float | None = None
    occluded_joint_weight: float = 0.35
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    soft_foot_anchor: bool = False
    soft_foot_anchor_init: float = 0.4
    # Keep only lead+follow pairs with bbox IoU >= this threshold
    couple_min_iou: float = 0.3

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
    compute_ap: bool = True
    # frame: random images; stratified_clip: every clip in both sets; clip: whole videos
    split_mode: str = 'frame'
    # Optional explicit holdout clip ids for split_mode=clip (e.g. ['task_15', 'task_10']).
    # When set, overrides random train_test_split clip assignment.
    test_clips: list[str] | None = None
    # Cap unique frames kept per task/clip (evenly spaced by frame index). None = no cap.
    max_frames_per_clip: int | None = None
    # Train-time augmentation (applied when KeypointDataset.training=True)
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.25
    aug_shift_frac: float = 0.1
    aug_rotation_deg: float = 12.0
    aug_color_jitter: float = 0.25
    # Single-person only: drop samples whose max IoU with another person in-frame
    # exceeds this. None = no filter. Solo people count as IoU 0.
    max_partner_iou: float | None = None
    # Single-person only: drop samples whose max IoU with another person in-frame
    # is below this. None = no filter. Solo people (IoU 0) are dropped when set.
    min_partner_iou: float | None = None
    # Phase A: weight on L1 bone-vector loss (0 = off). Applied on soft-decoded SimCC coords.
    bone_loss_weight: float = 0.0
    # Penalize L/R joint assignment when swapped assignment fits GT better (0 = off).
    lr_swap_loss_weight: float = 0.0
    # Margin (fraction of image diagonal) for joint L/R swap hinge.
    lr_swap_margin: float = 0.02
    # Chirality: relu(-cos(pred_R-L, gt_R-L)) on L/R pairs with enough GT separation.
    chirality_loss_weight: float = 0.0
    # Skip chirality when ||gt_R - gt_L|| / image_diag is below this (closed feet).
    chirality_min_sep: float = 0.03
    # What drives best.pt + early stopping: val_loss (default) or ap (OKS-AP on test).
    checkpoint_metric: str = 'val_loss'

@dataclass
class keypoint_mapping:
    name: str
    num_keypoints: int
    keypoints: dict
    left_right_pairs: list | None = None
    oks_sigmas: dict[str, float] | None = None


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
MAX_PARTNER_IOU: float | None = None
MIN_PARTNER_IOU: float | None = None
USE_SPATIAL_PROMPT = False
DEEP_SUPERVISION = False
DEEP_SUPERVISION_AUX_WEIGHT = 0.5
XY_CONSISTENCY_LOSS_WEIGHT = 0.0
BONE_LOSS_WEIGHT = 0.0
LR_SWAP_LOSS_WEIGHT = 0.0
LR_SWAP_MARGIN = 0.02
CHIRALITY_LOSS_WEIGHT = 0.0
CHIRALITY_MIN_SEP = 0.03
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
    def __init__(self, annotation_file: Path, task_ids, training=False, repeats=1, samples=None):
        self.image_root = annotation_file.parent
        self.training = training
        self.repeats = repeats if training else 1
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
        image_path, keypoints, bbox, role = self.samples[index % len(self.samples)]
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
        return pixels, torch.from_numpy(keypoints), role_bits, spatial


def build_person_samples(
    annotation_file: Path,
    task_ids,
    max_partner_iou: float | None = None,
    min_partner_iou: float | None = None,
):
    """Build single-person samples; optionally filter by partner-overlap IoU.

    Each sample is ``(image_path, keypoints, bbox, role)``.
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
        if video.get('id') not in task_ids:
            continue
        for frame in video.findall('frame'):
            image_path = image_root / frame.get('image')
            people = []
            for person in frame.findall('person'):
                keypoints, bbox, role = parse_person(person)
                if USE_ROLE and role is None:
                    raise ValueError(
                        f'Missing role on person in {image_path} '
                        f'(track_id={person.get("track_id")})'
                    )
                people.append((keypoints, bbox, role))
            for index, (keypoints, bbox, role) in enumerate(people):
                n_total += 1
                if max_partner_iou is not None or min_partner_iou is not None:
                    partner_iou = 0.0
                    for other_i, (_ok, other_bbox, _or) in enumerate(people):
                        if other_i == index:
                            continue
                        partner_iou = max(partner_iou, bbox_iou(bbox, other_bbox))
                    if max_partner_iou is not None and partner_iou > max_partner_iou:
                        n_dropped += 1
                        continue
                    if min_partner_iou is not None and partner_iou < min_partner_iou:
                        n_dropped += 1
                        continue
                samples.append((image_path, keypoints, bbox, role))
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
            image_path = image_root / frame.get('image')
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
            _lead_bbox,
            _follow_bbox,
            _iou,
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
        return pixels, torch.from_numpy(keypoints), role_bits, spatial


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


def build_couple_foot_anchor_pairs(keypoints_by_name: dict) -> list[tuple[int, int]]:
    """Duplicate single-person foot anchors into lead and follow banks."""
    person_pairs = build_foot_anchor_pairs(keypoints_by_name)
    follow_offset = len(keypoints_by_name)
    return list(person_pairs) + [
        (foot + follow_offset, ankle + follow_offset) for foot, ankle in person_pairs
    ]


def build_joint_query_head_kwargs(head_cfg: HeadConfig) -> dict:
    custom = head_cfg.custom
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
        'use_pair_bias': getattr(custom, 'use_pair_bias', False),
        'pair_bias_init': float(getattr(custom, 'pair_bias_init', 2.0)),
        'use_local_cross_attn': getattr(custom, 'use_local_cross_attn', False),
        'local_attn_sigma': float(getattr(custom, 'local_attn_sigma', 2.0)),
        'local_attn_strength': float(getattr(custom, 'local_attn_strength', 2.0)),
    }
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
            kwargs['pair_bias_bones'] = build_bone_index_pairs(head_cfg.keypoints)
            kwargs['pair_bias_lr_pairs'] = build_left_right_index_pairs(
                head_cfg.left_right_pairs, head_cfg.keypoints
            )
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
    global LEFT_RIGHT_PAIRS, JOINT_LOSS_WEIGHTS, OKS_SIGMAS
    global AUG_SCALE_MIN, AUG_SCALE_MAX, AUG_SHIFT_FRAC, AUG_ROTATION_DEG, AUG_COLOR_JITTER
    global USE_ROLE, ROLE_DROPOUT, COUPLE_MODE, COUPLE_MIN_IOU
    global MAX_PARTNER_IOU, MIN_PARTNER_IOU
    global USE_SPATIAL_PROMPT, DEEP_SUPERVISION, DEEP_SUPERVISION_AUX_WEIGHT
    global XY_CONSISTENCY_LOSS_WEIGHT, BONE_LOSS_WEIGHT, LR_SWAP_LOSS_WEIGHT, LR_SWAP_MARGIN
    global CHIRALITY_LOSS_WEIGHT, CHIRALITY_MIN_SEP, BONE_PAIRS

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
    CHIRALITY_LOSS_WEIGHT = float(getattr(training_cfg, 'chirality_loss_weight', 0.0))
    if CHIRALITY_LOSS_WEIGHT < 0:
        raise ValueError(f'chirality_loss_weight must be >= 0, got {CHIRALITY_LOSS_WEIGHT}')
    CHIRALITY_MIN_SEP = float(getattr(training_cfg, 'chirality_min_sep', 0.03))
    if CHIRALITY_MIN_SEP < 0:
        raise ValueError(f'chirality_min_sep must be >= 0, got {CHIRALITY_MIN_SEP}')
    checkpoint_metric = str(getattr(training_cfg, 'checkpoint_metric', 'val_loss')).lower()
    if checkpoint_metric not in ('val_loss', 'ap'):
        raise ValueError(
            f"checkpoint_metric must be 'val_loss' or 'ap', got {training_cfg.checkpoint_metric!r}"
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


def cap_samples_per_clip(samples, max_frames_per_clip: int | None, seed: int):
    """Keep at most ``max_frames_per_clip`` unique images per task/clip.

    When a clip exceeds the cap, frames are chosen evenly along sorted frame
    index (deterministic; ``seed`` reserved for future random modes).
    """
    if max_frames_per_clip is None:
        return samples
    if max_frames_per_clip < 1:
        raise ValueError(f'max_frames_per_clip must be >= 1, got {max_frames_per_clip}')

    by_clip: dict[str, dict] = {}
    order: list[str] = []
    for sample in samples:
        image_path = sample[0]
        clip_id = clip_id_from_image_path(image_path)
        if clip_id not in by_clip:
            by_clip[clip_id] = {}
            order.append(clip_id)
        by_clip[clip_id].setdefault(image_path, []).append(sample)

    kept = []
    dropped_frames = 0
    for clip_id in order:
        image_map = by_clip[clip_id]
        paths = sorted(image_map, key=_frame_index_from_image_path)
        if len(paths) <= max_frames_per_clip:
            selected = paths
        else:
            # Evenly spaced indices over [0, n-1], inclusive endpoints.
            n = len(paths)
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
        for path in selected:
            kept.extend(image_map[path])

    if dropped_frames:
        print(
            f'max_frames_per_clip={max_frames_per_clip}: '
            f'dropped {dropped_frames} frames; '
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
      - stratified_clip: within each task/clip, split images by ratio (every clip appears in train).
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


def forward_model(model, pixels, role, spatial):
    """Run DinoCC; returns (pred_x, pred_y, aux_list | None)."""
    out = model(
        pixels,
        role=role if USE_ROLE else None,
        spatial=spatial if USE_SPATIAL_PROMPT else None,
    )
    if isinstance(out, tuple) and len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], None


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


def lr_swap_assignment_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    left_right_pairs: list[tuple[int, int]],
    margin: float | None = None,
) -> torch.Tensor:
    """Joint margin hinge: penalize when full L/R swap fits GT better than correct.

    Aggregates L1 over all valid pairs, then
    ``relu((d_correct - d_swapped) / diag + margin)`` so a partial/full swap
    that beats the correct assignment is expensive, while mushy ties still get
    a margin push toward the labeled side.
    """
    if not left_right_pairs:
        return pred_xy.new_zeros(())
    margin = LR_SWAP_MARGIN if margin is None else float(margin)
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    vis = gt_keypoints[..., 2].to(device=pred_xy.device)
    diag = (IMAGE_SIZE.x ** 2 + IMAGE_SIZE.y ** 2) ** 0.5
    scale = max(float(diag), 1e-6)

    d_correct = pred_xy.new_zeros(pred_xy.shape[0])
    d_swapped = pred_xy.new_zeros(pred_xy.shape[0])
    pair_count = pred_xy.new_zeros(pred_xy.shape[0])
    for left_i, right_i in left_right_pairs:
        valid = (vis[:, left_i] > 0) & (vis[:, right_i] > 0)
        if not bool(valid.any()):
            continue
        pred_l = pred_xy[:, left_i]
        pred_r = pred_xy[:, right_i]
        gt_l = gt_xy[:, left_i]
        gt_r = gt_xy[:, right_i]
        correct = (pred_l - gt_l).abs().sum(dim=-1) + (pred_r - gt_r).abs().sum(dim=-1)
        swapped = (pred_l - gt_r).abs().sum(dim=-1) + (pred_r - gt_l).abs().sum(dim=-1)
        d_correct = d_correct + correct * valid.float()
        d_swapped = d_swapped + swapped * valid.float()
        pair_count = pair_count + valid.float()

    has_pairs = pair_count > 0
    if not bool(has_pairs.any()):
        return pred_xy.new_zeros(())
    # Normalize by #pairs so margin is comparable across visibility patterns.
    d_correct = d_correct / pair_count.clamp(min=1.0)
    d_swapped = d_swapped / pair_count.clamp(min=1.0)
    hinge = torch.relu((d_correct - d_swapped) / scale + margin)
    return hinge[has_pairs].mean()


def lr_chirality_loss(
    pred_xy: torch.Tensor,
    gt_keypoints: torch.Tensor,
    left_right_pairs: list[tuple[int, int]],
    min_sep: float | None = None,
) -> torch.Tensor:
    """Push L/R vectors to match GT orientation: ``relu(-cos(pred_R-L, gt_R-L))``.

    Skips pairs whose GT separation is below ``min_sep * image_diag`` so closed
    feet do not inject noisy chirality gradients.
    """
    if not left_right_pairs:
        return pred_xy.new_zeros(())
    min_sep = CHIRALITY_MIN_SEP if min_sep is None else float(min_sep)
    gt_xy = gt_keypoints[..., :2].to(device=pred_xy.device, dtype=pred_xy.dtype)
    vis = gt_keypoints[..., 2].to(device=pred_xy.device)
    diag = max((IMAGE_SIZE.x ** 2 + IMAGE_SIZE.y ** 2) ** 0.5, 1e-6)
    min_norm = min_sep * diag

    losses = []
    for left_i, right_i in left_right_pairs:
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
    if not losses:
        return pred_xy.new_zeros(())
    return torch.stack(losses).mean()


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
):
    loss = criterion(pred_x, pred_y, target_x, target_y, weights)
    if aux_preds:
        for aux_x, aux_y in aux_preds:
            loss = loss + DEEP_SUPERVISION_AUX_WEIGHT * criterion(
                aux_x, aux_y, target_x, target_y, weights
            )
    if XY_CONSISTENCY_LOSS_WEIGHT > 0:
        loss = loss + XY_CONSISTENCY_LOSS_WEIGHT * simcc_xy_consistency_loss(
            pred_x, pred_y, target_x, target_y, weights
        )
    need_xy = gt_keypoints is not None and (
        BONE_LOSS_WEIGHT > 0 or LR_SWAP_LOSS_WEIGHT > 0 or CHIRALITY_LOSS_WEIGHT > 0
    )
    if need_xy:
        pred_xy = soft_decode_simcc_coords(pred_x, pred_y, SIMCC_SPLIT_RATIO)
        if BONE_LOSS_WEIGHT > 0:
            loss = loss + BONE_LOSS_WEIGHT * bone_vector_loss(
                pred_xy, gt_keypoints, BONE_PAIRS
            )
        if LR_SWAP_LOSS_WEIGHT > 0:
            loss = loss + LR_SWAP_LOSS_WEIGHT * lr_swap_assignment_loss(
                pred_xy, gt_keypoints, LEFT_RIGHT_PAIRS
            )
        if CHIRALITY_LOSS_WEIGHT > 0:
            loss = loss + CHIRALITY_LOSS_WEIGHT * lr_chirality_loss(
                pred_xy, gt_keypoints, LEFT_RIGHT_PAIRS
            )
    return loss


def run_epoch(model, loader, criterion, optimizer, device, scaler, training, gradient_clip_norm):
    model.train(training)
    total_loss = 0.0

    for pixels, keypoints, role, spatial in loader:
        pixels = pixels.to(device, non_blocking=True)
        role = role.to(device, non_blocking=True)
        spatial = spatial.to(device, non_blocking=True)
        role = maybe_drop_role(role, training)
        target_x, target_y, weights = make_targets(keypoints, device)
        gt_keypoints = (
            keypoints.to(device, non_blocking=True)
            if (BONE_LOSS_WEIGHT > 0 or LR_SWAP_LOSS_WEIGHT > 0)
            else None
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y, aux_preds = forward_model(model, pixels, role, spatial)
            loss = compute_simcc_training_loss(
                criterion,
                pred_x,
                pred_y,
                target_x,
                target_y,
                weights,
                aux_preds,
                gt_keypoints=gt_keypoints,
            )

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
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Expected semantic version format 'X.Y.Z', got: {annotation_version}"
        )

    major, minor, _ = parts
    major_dir = f"{major}.X"
    minor_dir = f"{major}.{minor}.X"

    dataset_dir = Path("dataset") / "versions" / major_dir / minor_dir / annotation_version
    annotation_file = dataset_dir / "cleaned_annotations.xml"
    raw_annotations = dataset_dir / "annotations.xml"

    if annotation_file.is_file():
        return annotation_file

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset version directory not found: {dataset_dir}")
    if not raw_annotations.is_file():
        raise FileNotFoundError(
            f"cleaned_annotations.xml missing and no annotations.xml to build from: {raw_annotations}"
        )

    print(
        f"cleaned_annotations.xml not found for {annotation_version}; "
        f"running dataset_parser.py {annotation_version} ..."
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "dataset_parser.py"), annotation_version],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dataset_parser.py failed with exit code {result.returncode} "
            f"while building {annotation_file}"
        )
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"dataset_parser.py finished but cleaned annotations still missing: {annotation_file}"
        )
    return annotation_file


def parse_config(raw_config):
    image_cfg = ImageConfig(**raw_config['image'])
    backbone_cfg = BackboneConfig(**raw_config['backbone'])
    training_cfg = TrainingConfig(**raw_config['training'])

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
        case 'joint_query_simcc' | 'joint_query_self_attn_simcc':
            head_cfg.custom = JointQuerySimCCConfig(**head_cfg.custom)
        case 'couple_joint_query_self_attn_simcc':
            head_cfg.custom = CoupleJointQuerySimCCConfig(**head_cfg.custom)
        case _:
            raise ValueError(
                f'Unsupported head name: {head_cfg.name}; '
                f"expected 'depthwise_simcc', 'joint_query_simcc', "
                f"'joint_query_self_attn_simcc', or 'couple_joint_query_self_attn_simcc'"
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

    match backbone_cfg.name:
        case 'dino_v2_base' | 'dino_v3_base' | 'vitpose_plus_base' | 'vitpose_base':
            match head_cfg.type:
                case 'simcc':
                    joint_query_heads = (
                        'joint_query_simcc',
                        'joint_query_self_attn_simcc',
                        COUPLE_HEAD_NAME,
                    )
                    if head_cfg.name in joint_query_heads:
                        head_kwargs = build_joint_query_head_kwargs(head_cfg)
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
    """AdamW with optional lower LR for trainable backbone params."""
    head_lr = training_cfg.learning_rate
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')]
    if backbone_params:
        backbone_lr = (
            backbone_cfg.learning_rate
            if backbone_cfg.learning_rate is not None
            else head_lr * 0.1
        )
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=training_cfg.weight_decay)
        return optimizer, backbone_lr
    optimizer = torch.optim.AdamW(
        head_params,
        lr=head_lr,
        weight_decay=training_cfg.weight_decay,
    )
    return optimizer, None


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
    full_samples = cap_samples_per_clip(
        full_samples,
        getattr(training_cfg, 'max_frames_per_clip', None),
        training_cfg.seed,
    )
    _, test_samples, _, _ = split_samples_by_image(
        full_samples,
        training_cfg.train_test_split,
        training_cfg.seed,
        split_mode=training_cfg.split_mode,
        test_clips=getattr(training_cfg, 'test_clips', None),
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
                image_path, keypoints, bbox, role = sample
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
            pred_x, pred_y, _aux = forward_model(model, pixels_tensor, role_tensor, spatial_tensor)
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
        from eval_ownership import build_frame_index

        annotation_file = get_annotation_file_path(annotation_version)
        root = ET.parse(annotation_file).getroot()
        task_ids = {video.get('id') for video in root.findall('./project/videos/video')}
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
                    _iou,
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
                image_path, keypoints, bbox, role = sample
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
            pred_x, pred_y, _aux = forward_model(model, pixels_tensor, role_tensor, spatial_tensor)
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
        from ownership_metrics import summarize_ownership_records

        ownership_summary = summarize_ownership_records(ownership_records)

    return summarize_pose_metrics(records, ownership=ownership_summary)


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
    from eval_ownership import is_same_person
    from ownership_metrics import iou_bucket, joint_xy_mix_fraction, pick_partner

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


def main():
    parser = argparse.ArgumentParser(description='Train DinoCC on the cleaned keypoint dataset.')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help='Load model weights from this .pt (fresh optimizer; for continued training).',
    )
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
    if COUPLE_MODE:
        full_samples = build_couple_samples(annotation_file, set(task_ids), COUPLE_MIN_IOU)
    else:
        full_samples = build_person_samples(
            annotation_file,
            set(task_ids),
            max_partner_iou=MAX_PARTNER_IOU,
            min_partner_iou=MIN_PARTNER_IOU,
        )
    full_samples = cap_samples_per_clip(
        full_samples,
        getattr(training_cfg, 'max_frames_per_clip', None),
        training_cfg.seed,
    )
    train_samples, test_samples, train_image_paths, test_image_paths = split_samples_by_image(
        full_samples,
        training_cfg.train_test_split,
        training_cfg.seed,
        split_mode=training_cfg.split_mode,
        test_clips=getattr(training_cfg, 'test_clips', None),
    )
    n_dance_train = len(train_samples)
    n_coco_train = 0
    coco_cfg = getattr(dataset_cfg, 'coco_wholebody', None)
    coco_rolling = bool(coco_cfg is not None and coco_cfg.enabled and coco_cfg.rolling)
    dance_train_only = list(train_samples)
    coco_pool: list = []
    if coco_cfg is not None and coco_cfg.enabled:
        if COUPLE_MODE:
            raise ValueError(
                'coco_wholebody mixing requires a single-person head '
                '(not couple_joint_query_self_attn_simcc)'
            )
        target_coco = target_coco_count(n_dance_train, coco_cfg.dance_fraction)
        # Rolling keeps the full valid pool (max_samples still caps it); fixed mode
        # reservoir-samples down to the per-epoch target once at startup.
        build_target = None if coco_rolling else (target_coco if target_coco > 0 else None)
        coco_pool = build_coco_wholebody_samples(
            coco_cfg,
            head_cfg.keypoints,
            PERSON_JOINTS,
            seed=training_cfg.seed,
            target_count=build_target,
        )
        if coco_rolling:
            gc.collect()
        train_samples, n_coco_train = mix_dance_with_coco(
            dance_train_only,
            coco_pool,
            coco_cfg.dance_fraction,
            seed=training_cfg.seed + 17 + (1 if coco_rolling else 0),
        )
        missing_images = [
            path for path, *_ in coco_pool[: min(32, len(coco_pool))] if not Path(path).is_file()
        ]
        if missing_images:
            raise FileNotFoundError(
                f'COCO image(s) missing (showing up to 32): {missing_images[:5]}'
            )
        dance_labeled = mean_labeled_joints(dance_train_only)
        realized = n_dance_train / max(1, len(train_samples))
        print(
            f'COCO mix: rolling={coco_rolling}, dance_train={n_dance_train}, '
            f'coco_pool={len(coco_pool)}, coco_per_epoch={n_coco_train}, '
            f'total_train={len(train_samples)}, realized_dance_fraction={realized:.3f} '
            f'(target={coco_cfg.dance_fraction:.3f}); '
            f'mean_labeled_joints dance={dance_labeled:.1f} '
            f'coco_pool={mean_labeled_joints(coco_pool):.1f}'
        )

    dance_train_image_paths = list(train_image_paths)
    split_index = len(dance_train_image_paths)
    image_paths = dance_train_image_paths + test_image_paths
    train_clips = sorted({clip_id_from_image_path(path) for path in dance_train_image_paths})
    test_clips = sorted({clip_id_from_image_path(path) for path in test_image_paths})
    split_label = f'{training_cfg.train_test_split:.2f} by {training_cfg.split_mode}'
    if n_coco_train:
        rolling_tag = ', rolling' if coco_rolling else ''
        split_label += (
            f' + coco_wholebody (dance_frac≈{n_dance_train / len(train_samples):.2f}'
            f'{rolling_tag})'
        )
    dataset_cls = CoupleKeypointDataset if COUPLE_MODE else KeypointDataset
    train_dataset = dataset_cls(
        annotation_file,
        set(task_ids),
        training=True,
        repeats=training_cfg.augmentation_repeats,
        samples=train_samples,
    )
    test_dataset = dataset_cls(
        annotation_file,
        set(task_ids),
        samples=test_samples,
    )
    # Rolling keeps a large COCO pool in-process. Prefer a couple workers for GPU
    # feed rate; cap hard so we don't multiply the ~130k-sample pool into an OOM.
    train_workers = min(2, args.workers) if coco_rolling else args.workers
    if coco_rolling and train_workers != args.workers:
        print(
            f'COCO rolling: using train_workers={train_workers} '
            f'(requested --workers={args.workers}) with pool={len(coco_pool)}'
        )
    train_loader = make_loader(train_dataset, training_cfg.batch_size, train_workers)
    test_loader = make_loader(test_dataset, training_cfg.batch_size, args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config, device)
    if getattr(args, 'init_checkpoint', None):
        init_path = Path(args.init_checkpoint).resolve()
        if not init_path.is_file():
            raise FileNotFoundError(f'--init-checkpoint not found: {init_path}')
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=True)
        print(
            f'Loaded init checkpoint {init_path} '
            f'(epoch={ckpt.get("epoch", "?")}; missing={len(missing)} unexpected={len(unexpected)})'
        )
    criterion = SimCCLoss().to(device)

    joint_query_heads = (
        'joint_query_simcc',
        'joint_query_self_attn_simcc',
        COUPLE_HEAD_NAME,
    )
    head_checkpoint_meta = {
        'head_name': head_cfg.name,
        'head_kwargs': (
            build_joint_query_head_kwargs(head_cfg)
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
    best_val_loss = float('inf')
    best_val_ap = -1.0
    epochs_without_improvement = 0
    best_path = run_dir / 'best.pt'
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
        f'; init_checkpoint={Path(args.init_checkpoint).resolve()}'
        if getattr(args, 'init_checkpoint', None)
        else ''
    )
    bb_train = sum(p.requires_grad for p in model.backbone.parameters())
    bb_total = sum(1 for _ in model.backbone.parameters())
    backbone_msg = (
        f' backbone={backbone_cfg.name} freeze={backbone_cfg.freeze}'
        f' unfreeze_last_blocks={backbone_cfg.unfreeze_last_blocks}'
        f' trainable_tensors={bb_train}/{bb_total}'
        f'{"" if backbone_lr is None else f" backbone_lr={backbone_lr}"}'
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
        f'chirality_loss_weight={CHIRALITY_LOSS_WEIGHT}; '
        f'chirality_min_sep={CHIRALITY_MIN_SEP}; '
        f'checkpoint_metric={checkpoint_metric}; '
        f'lr={training_cfg.learning_rate}; '
        f'simcc_sigma={SIMCC_SIGMA}'
        f'{"" if SIMCC_SIGMA_OCCLUDED is None else f"/{SIMCC_SIGMA_OCCLUDED}_occ"}; '
        f'occ_weight={OCCLUDED_JOINT_WEIGHT}; '
        f'num_joints={NUM_JOINTS}{couple_msg}{backbone_msg}{init_msg}'
    )

    for epoch in range(1, training_cfg.max_epochs + 1):
        if coco_rolling:
            train_samples, n_coco_train = mix_dance_with_coco(
                dance_train_only,
                coco_pool,
                coco_cfg.dance_fraction,
                seed=training_cfg.seed + 17 + epoch,
            )
            train_dataset.samples = train_samples
            train_loader = make_loader(train_dataset, training_cfg.batch_size, train_workers)
            print(
                f'COCO rolling: epoch={epoch:03d} coco={n_coco_train}/{len(coco_pool)} '
                f'total_train={len(train_samples)}'
            )
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, True, training_cfg.gradient_clip_norm)
        val_loss = run_epoch(model, test_loader, criterion, optimizer, device, scaler, False, training_cfg.gradient_clip_norm)
        scheduler.step()

        val_ap = None
        if checkpoint_metric == 'ap':
            # Skip ownership suite during the epoch loop for speed.
            val_metrics = evaluate_checkpoint_metrics(
                model,
                checkpoint_path=None,
                test_samples=test_samples,
                device=device,
                batch_size=training_cfg.batch_size,
                annotation_version=None,
            )
            val_ap = float(val_metrics['AP'])
            print(
                f'Epoch {epoch:03d} | train_loss={train_loss:.5f} | '
                f'val_loss={val_loss:.5f} | val_AP={val_ap:.4f}'
            )
            improved = val_ap > best_val_ap + 1e-6
        else:
            print(f'Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}')
            improved = val_loss < best_val_loss

        if improved:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_ap is not None and val_ap > best_val_ap:
                best_val_ap = val_ap
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
                    'checkpoint_metric': checkpoint_metric,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_cfg.early_stopping_patience:
                print(
                    f'Early stopping after {training_cfg.early_stopping_patience} epochs '
                    f'without {checkpoint_metric} improvement.'
                )
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

    if training_cfg.compute_ap:
        if OKS_SIGMAS.numel() == 0:
            raise ValueError(
                'compute_ap=true requires oks_sigmas in the keypoint format mapping '
                f'({head_cfg.keypoint_format_name})'
            )
        print('Evaluating OKS-AP on test set for best and final checkpoints...')
        for checkpoint_name, checkpoint_path in (('best', best_path), ('final', final_path)):
            if not checkpoint_path.exists():
                print(f'Skipping AP for {checkpoint_name}: missing {checkpoint_path}')
                continue
            export_checkpoint_metrics(
                model=model,
                checkpoint_path=checkpoint_path,
                test_samples=test_samples,
                output_dir=run_dir / checkpoint_name,
                device=device,
                batch_size=training_cfg.batch_size,
                annotation_version=dataset_cfg.annotation_version,
            )


if __name__ == '__main__':
    main()