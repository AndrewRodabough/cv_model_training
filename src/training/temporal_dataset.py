"""Contiguous temporal window sampling for dense pose clips."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

from src.model_review.metrics import person_scale_in_crop
from src.util.vec import IVec2


def unpack_person_sample(sample: tuple) -> tuple:
    """Return ``(image_path, keypoints, bbox, role, task_id, frame_number, track_id)``.

    Legacy 4-tuples (e.g. COCO) get ``task_id=None``, ``frame_number=0``, ``track_id=''``.
    """
    image_path, keypoints, bbox, role = sample[:4]
    if len(sample) >= 7:
        return image_path, keypoints, bbox, role, sample[4], int(sample[5]), str(sample[6])
    return image_path, keypoints, bbox, role, None, 0, ''


def sample_has_temporal_meta(sample: tuple) -> bool:
    return len(sample) >= 7 and sample[4] is not None


def build_contiguous_windows(
    samples: list[tuple],
    window_size: int,
    window_stride: int = 1,
) -> list[list[tuple]]:
    """Group person samples into contiguous frame windows of length ``window_size``.

    A window is emitted only when frame numbers form ``f, f+1, …, f+T-1`` for the
    same ``(task_id, track_id)``.
    """
    if window_size < 1:
        raise ValueError(f'window_size must be >= 1, got {window_size}')
    if window_stride < 1:
        raise ValueError(f'window_stride must be >= 1, got {window_stride}')

    by_track: dict[tuple[Any, str], list[tuple]] = defaultdict(list)
    skipped = 0
    for sample in samples:
        if not sample_has_temporal_meta(sample):
            skipped += 1
            continue
        _path, _kps, _bbox, _role, task_id, frame_number, track_id = unpack_person_sample(
            sample
        )
        by_track[(task_id, track_id)].append(sample)

    windows: list[list[tuple]] = []
    for (_task_id, _track_id), track_samples in by_track.items():
        # One sample per frame (last wins if duplicates).
        by_frame: dict[int, tuple] = {}
        for sample in track_samples:
            frame_number = int(sample[5])
            by_frame[frame_number] = sample
        frames = sorted(by_frame)
        # Candidate starts: every annotated frame; keep those with a full contiguous run.
        for start_frame in frames:
            if window_stride > 1 and (start_frame - frames[0]) % window_stride != 0:
                continue
            needed = list(range(start_frame, start_frame + window_size))
            if all(f in by_frame for f in needed):
                windows.append([by_frame[f] for f in needed])

    return windows


def summarize_window_sources(windows: list[list[tuple]]) -> dict[str, int]:
    """Count windows per ``task_N`` clip id."""
    counts: dict[str, int] = defaultdict(int)
    for window in windows:
        task_id = window[0][4]
        counts[f'task_{task_id}'] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def split_windows_stratified(
    windows: list[list[tuple]],
    train_ratio: float,
    seed: int,
) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """Hold out the last ``1 - train_ratio`` windows per track (by start frame)."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f'train_ratio must be in (0, 1), got {train_ratio}')

    by_track: dict[tuple[Any, str], list[list[tuple]]] = defaultdict(list)
    for window in windows:
        task_id = window[0][4]
        track_id = str(window[0][6])
        by_track[(task_id, track_id)].append(window)

    train: list[list[tuple]] = []
    test: list[list[tuple]] = []
    for key in sorted(by_track, key=lambda item: (str(item[0]), item[1])):
        track_windows = sorted(by_track[key], key=lambda w: int(w[0][5]))
        n = len(track_windows)
        if n == 1:
            train.extend(track_windows)
            continue
        n_train = max(1, min(n - 1, int(round(n * train_ratio))))
        # Chronological holdout of the tail (seed reserved for future random modes).
        _ = seed
        train.extend(track_windows[:n_train])
        test.extend(track_windows[n_train:])
    return train, test


def _union_bbox(bboxes: list[tuple[float, float, float, float]]):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


class TemporalWindowDataset(Dataset):
    """Load contiguous T-frame person windows with a shared crop + shared aug."""

    def __init__(
        self,
        windows: list[list[tuple]],
        *,
        training: bool,
        repeats: int = 1,
        image_size: IVec2,
        image_mean: torch.Tensor,
        image_std: torch.Tensor,
        bbox_padding: float,
        compute_padded_crop,
        transform_keypoints_to_crop,
        encode_role,
        crop_normalized_bbox_center,
        horizontally_flip_keypoints,
        rotate_image_and_keypoints,
        aug_scale_min: float,
        aug_scale_max: float,
        aug_shift_frac: float,
        aug_rotation_deg: float,
        aug_color_jitter: float,
        use_union_bbox: bool = True,
    ):
        if not windows:
            raise ValueError('TemporalWindowDataset requires at least one window')
        self.windows = list(windows)
        self.training = training
        self.repeats = repeats if training else 1
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.bbox_padding = bbox_padding
        self.compute_padded_crop = compute_padded_crop
        self.transform_keypoints_to_crop = transform_keypoints_to_crop
        self.encode_role = encode_role
        self.crop_normalized_bbox_center = crop_normalized_bbox_center
        self.horizontally_flip_keypoints = horizontally_flip_keypoints
        self.rotate_image_and_keypoints = rotate_image_and_keypoints
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_shift_frac = aug_shift_frac
        self.aug_rotation_deg = aug_rotation_deg
        self.aug_color_jitter = aug_color_jitter
        self.use_union_bbox = use_union_bbox
        self.window_size = len(windows[0])
        for window in windows:
            if len(window) != self.window_size:
                raise ValueError('All windows must share the same length')

    def __len__(self) -> int:
        return len(self.windows) * self.repeats

    def __getitem__(self, index: int):
        window = self.windows[index % len(self.windows)]
        center = window[len(window) // 2]
        bboxes = [sample[2] for sample in window]
        crop_bbox = _union_bbox(bboxes) if self.use_union_bbox else center[2]

        # Shared image size from center frame (all frames of a clip share resolution).
        center_image = Image.open(center[0]).convert('RGB')
        original_width, original_height = center_image.size
        center_image.close()

        if self.training:
            scale = random.uniform(self.aug_scale_min, self.aug_scale_max)
            shift_x_frac = random.uniform(-self.aug_shift_frac, self.aug_shift_frac)
            shift_y_frac = random.uniform(-self.aug_shift_frac, self.aug_shift_frac)
            do_flip = random.random() < 0.5
            angle = (
                random.uniform(-self.aug_rotation_deg, self.aug_rotation_deg)
                if self.aug_rotation_deg > 0
                else 0.0
            )
            color_lo = 1.0 - self.aug_color_jitter
            color_hi = 1.0 + self.aug_color_jitter
            brightness = random.uniform(color_lo, color_hi)
            contrast = random.uniform(color_lo, color_hi)
            color = random.uniform(color_lo, color_hi)
        else:
            scale = 1.0
            shift_x_frac = 0.0
            shift_y_frac = 0.0
            do_flip = False
            angle = 0.0
            brightness = contrast = color = 1.0

        crop_x1, crop_y1, crop_width, crop_height = self.compute_padded_crop(
            crop_bbox,
            original_width,
            original_height,
            scale=scale,
            shift_x_frac=shift_x_frac,
            shift_y_frac=shift_y_frac,
        )

        pixels_list = []
        keypoints_list = []
        role_list = []
        spatial_list = []
        scale_list = []

        for sample in window:
            image_path, keypoints, bbox, role, *_rest = unpack_person_sample(sample)
            image = Image.open(image_path).convert('RGB')
            if image.size != (original_width, original_height):
                raise ValueError(
                    f'Frame size mismatch in temporal window: {image_path} '
                    f'has {image.size}, expected {(original_width, original_height)}'
                )
            image = image.crop(
                (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)
            )
            image = image.resize(
                (self.image_size.x, self.image_size.y), Image.Resampling.BILINEAR
            )
            kps = self.transform_keypoints_to_crop(
                keypoints, crop_x1, crop_y1, crop_width, crop_height
            )
            if do_flip:
                image = ImageOps.mirror(image)
                kps = self.horizontally_flip_keypoints(kps)
            if angle != 0.0:
                image, kps = self.rotate_image_and_keypoints(image, kps, angle)
            if self.training:
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
                image = ImageEnhance.Color(image).enhance(color)

            pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
            pixels = (pixels - self.image_mean) / self.image_std
            role_bits = self.encode_role(role)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            spatial = self.crop_normalized_bbox_center(
                cx, cy, crop_x1, crop_y1, crop_width, crop_height
            )
            bbox_scale = person_scale_in_crop(
                bbox,
                crop_width,
                crop_height,
                self.image_size.x,
                self.image_size.y,
            )
            pixels_list.append(pixels)
            keypoints_list.append(torch.from_numpy(kps))
            role_list.append(role_bits)
            spatial_list.append(spatial)
            scale_list.append(torch.tensor(bbox_scale, dtype=torch.float32))

        return (
            torch.stack(pixels_list, dim=0),  # [T, 3, H, W]
            torch.stack(keypoints_list, dim=0),  # [T, K, 3]
            torch.stack(role_list, dim=0),  # [T, 2]
            torch.stack(spatial_list, dim=0),  # [T, 2]
            torch.stack(scale_list, dim=0),  # [T]
            torch.tensor(0.0, dtype=torch.float32),  # is_coco
        )
