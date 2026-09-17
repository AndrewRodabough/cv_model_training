#!/usr/bin/env python3
"""Tier 1–2 pipeline verification: automated checks + visual overlay gallery."""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dino_cc import generate_simcc_labels
import train as train_mod
from train import (
    IMAGE_MEAN,
    IMAGE_STD,
    KeypointDataset,
    compute_padded_crop,
    get_annotation_file_path,
    horizontally_flip_keypoints,
    parse_config,
    set_seed,
    setup_from_config,
    split_samples_by_image,
    transform_keypoints_to_crop,
)


class CheckResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def ok(self, name: str, detail: str = ''):
        self.passed += 1
        suffix = f' — {detail}' if detail else ''
        print(f'  PASS  {name}{suffix}')

    def fail(self, name: str, detail: str):
        self.failed += 1
        print(f'  FAIL  {name} — {detail}')

    def warn(self, name: str, detail: str):
        self.warnings += 1
        print(f'  WARN  {name} — {detail}')


def joint_names_by_id(keypoints_by_name: dict) -> dict[int, str]:
    return {meta['id']: name for name, meta in keypoints_by_name.items()}


def denormalize_pixels(pixels: torch.Tensor) -> Image.Image:
    image = pixels.detach().cpu() * IMAGE_STD + IMAGE_MEAN
    image = (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image, mode='RGB')


def draw_keypoints(image: Image.Image, keypoints, names: dict[int, str], color='lime'):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None
    for index, (x, y, visibility) in enumerate(keypoints):
        if visibility <= 0:
            continue
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline='white')
        label = names.get(index, str(index))
        draw.text((x + 5, y - 8), label, fill='white', font=font, stroke_width=2, stroke_fill='black')
    return image


def load_config(config_path: Path):
    with open(config_path, encoding='utf-8') as config_file:
        raw = yaml.safe_load(config_file)
    return parse_config(raw)


def check_config_mapping(config, results: CheckResult):
    print('\n[1] Config / mapping')
    head = config['head']
    training = config['training']
    dataset = config['dataset']

    missing = sorted(name for name in head.keypoints if name not in training.joint_loss_weights)
    if missing:
        results.fail('joint_loss_weights coverage', f'missing: {missing}')
    else:
        results.ok('joint_loss_weights coverage', f'{len(head.keypoints)} joints')

    try:
        pairs = train_mod.build_left_right_index_pairs(head.left_right_pairs, head.keypoints)
        results.ok('left_right_pairs', f'{len(pairs)} pairs')
    except ValueError as error:
        results.fail('left_right_pairs', str(error))

    mapping_count = len(head.keypoints)
    if train_mod.NUM_JOINTS == mapping_count:
        results.ok('NUM_JOINTS', str(train_mod.NUM_JOINTS))
    else:
        results.fail(
            'NUM_JOINTS',
            f'NUM_JOINTS={train_mod.NUM_JOINTS} vs mapping={mapping_count}',
        )

    try:
        annotation_file = get_annotation_file_path(dataset.annotation_version)
        results.ok('annotation path', str(annotation_file))
    except ValueError as error:
        results.fail('annotation path', str(error))
        return None
    return annotation_file


def check_keypoint_id_mapping(config, results: CheckResult):
    print('\n[3] Keypoint id mapping')
    keypoints = config['head'].keypoints
    ordered = sorted(keypoints.items(), key=lambda item: item[1]['id'])
    local_ids = [meta['id'] for _, meta in ordered]
    expected = list(range(len(ordered)))
    if local_ids == expected:
        results.ok('local id order', f'0..{len(ordered) - 1}')
    else:
        results.fail('local id order', f'got {local_ids}')

    for name, meta in ordered:
        standard_id = meta['standard_id']
        local_id = meta['id']
        if train_mod.KEYPOINT_INDEX.get(standard_id) != local_id:
            results.fail(
                f'mapping {name}',
                f'standard_id={standard_id} expected local {local_id}, '
                f'KEYPOINT_INDEX has {train_mod.KEYPOINT_INDEX.get(standard_id)}',
            )
        elif train_mod.KEYPOINT_IDS[local_id] != standard_id:
            results.fail(
                f'KEYPOINT_IDS[{local_id}]',
                f'expected {standard_id}, got {train_mod.KEYPOINT_IDS[local_id]}',
            )
        else:
            results.ok(f'{name}: standard_id {standard_id} -> local {local_id}')


def check_annotation_images(annotation_file: Path, results: CheckResult):
    print('\n[2] Annotation ↔ images')
    root = ET.parse(annotation_file).getroot()
    image_root = annotation_file.parent
    missing = []
    person_count = 0
    frame_paths = []
    for video in root.findall('./project/videos/video'):
        for frame in video.findall('frame'):
            rel = frame.get('image')
            path = image_root / rel
            frame_paths.append(path)
            if not path.exists():
                missing.append(str(path))
            person_count += len(frame.findall('person'))

    if missing:
        results.fail('image files exist', f'{len(missing)} missing (e.g. {missing[0]})')
    else:
        results.ok('image files exist', f'{len(frame_paths)} frames')

    task_ids = {video.get('id') for video in root.findall('./project/videos/video')}
    dataset = KeypointDataset(annotation_file, task_ids)
    if len(dataset.samples) == person_count:
        results.ok('sample count', f'{person_count} persons')
    else:
        results.fail(
            'sample count',
            f'dataset={len(dataset.samples)} vs xml persons={person_count}',
        )

    oob = 0
    nonfinite = 0
    visible = 0
    for image_path, keypoints, _bbox in dataset.samples:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            continue
        for x, y, visibility in keypoints:
            if visibility <= 0:
                continue
            visible += 1
            if not (np.isfinite(x) and np.isfinite(y)):
                nonfinite += 1
            elif x < 0 or y < 0 or x >= width or y >= height:
                oob += 1

    if nonfinite:
        results.fail('visible keypoints finite', f'{nonfinite}/{visible} non-finite')
    else:
        results.ok('visible keypoints finite', f'{visible} checked')

    if visible and oob / visible > 0.25:
        results.fail('out-of-image keypoints', f'{oob}/{visible} ({100 * oob / visible:.1f}%)')
    elif oob:
        results.warn('out-of-image keypoints', f'{oob}/{visible} (reported only)')
    else:
        results.ok('out-of-image keypoints', '0')

    return dataset, task_ids


def check_crop_geometry(dataset: KeypointDataset, results: CheckResult, n: int = 16):
    print('\n[4] Crop / resize geometry')
    indices = list(range(min(n, len(dataset.samples))))
    eval_dataset = KeypointDataset(
        Path('.'),
        set(),
        training=False,
        samples=dataset.samples,
    )
    max_err = 0.0
    for index in indices:
        image_path, keypoints, bbox = dataset.samples[index]
        pixels, out_keypoints = eval_dataset[index]
        with Image.open(image_path) as image:
            width, height = image.size
        crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(bbox, width, height)
        expected = transform_keypoints_to_crop(keypoints, crop_x1, crop_y1, crop_width, crop_height)
        err = float(np.max(np.abs(out_keypoints.numpy() - expected)))
        max_err = max(max_err, err)

        if pixels.shape != (3, train_mod.IMAGE_SIZE.y, train_mod.IMAGE_SIZE.x):
            results.fail(
                f'sample {index} pixel shape',
                f'got {tuple(pixels.shape)}, expected (3, {train_mod.IMAGE_SIZE.y}, {train_mod.IMAGE_SIZE.x})',
            )
            return
        if out_keypoints.shape != (train_mod.NUM_JOINTS, 3):
            results.fail(
                f'sample {index} keypoint shape',
                f'got {tuple(out_keypoints.shape)}, expected ({train_mod.NUM_JOINTS}, 3)',
            )
            return

    if max_err <= 1e-3:
        results.ok('crop keypoint transform', f'max abs err={max_err:.2e} on {len(indices)} samples')
    else:
        results.fail('crop keypoint transform', f'max abs err={max_err:.2e} > 1e-3')

    results.ok(
        'tensor shapes',
        f'pixels (3, {train_mod.IMAGE_SIZE.y}, {train_mod.IMAGE_SIZE.x}), '
        f'keypoints ({train_mod.NUM_JOINTS}, 3)',
    )


def check_horizontal_flip(results: CheckResult):
    print('\n[5] Horizontal flip + L/R swap')
    keypoints = np.zeros((train_mod.NUM_JOINTS, 3), dtype=np.float32)
    for index in range(train_mod.NUM_JOINTS):
        keypoints[index] = [10.0 + index, 20.0 + 2 * index, 1.0 if index % 2 == 0 else 2.0]

    if train_mod.LEFT_RIGHT_PAIRS:
        left, right = train_mod.LEFT_RIGHT_PAIRS[0]
        keypoints[left, 2] = 2.0
        keypoints[right, 2] = 1.0
        keypoints[left, 0] = 40.0
        keypoints[right, 0] = 200.0
        keypoints[left, 1] = 50.0
        keypoints[right, 1] = 60.0

    flipped = horizontally_flip_keypoints(keypoints)
    expected = keypoints.copy()
    expected[:, 0] = train_mod.IMAGE_SIZE.x - 1 - expected[:, 0]
    for left, right in train_mod.LEFT_RIGHT_PAIRS:
        expected[[left, right]] = expected[[right, left]]

    if np.allclose(flipped, expected):
        results.ok('x flip + L/R swap', f'{len(train_mod.LEFT_RIGHT_PAIRS)} pairs, W={train_mod.IMAGE_SIZE.x}')
    else:
        results.fail('x flip + L/R swap', f'max err={float(np.max(np.abs(flipped - expected))):.4f}')

    vis_ok = True
    for left, right in train_mod.LEFT_RIGHT_PAIRS:
        if flipped[left, 2] != keypoints[right, 2] or flipped[right, 2] != keypoints[left, 2]:
            vis_ok = False
    if vis_ok:
        results.ok('visibility travels with joint')
    else:
        results.fail('visibility travels with joint', 'visibility not swapped with pair')


def check_repeats_and_split(dataset: KeypointDataset, config, task_ids, results: CheckResult):
    print('\n[6] Augmentation repeats + split')
    training = config['training']
    train_samples, test_samples, train_paths, test_paths = split_samples_by_image(
        dataset.samples,
        training.train_test_split,
        training.seed,
    )
    train_dataset = KeypointDataset(
        Path('.'),
        task_ids,
        training=True,
        repeats=training.augmentation_repeats,
        samples=train_samples,
    )
    test_dataset = KeypointDataset(Path('.'), task_ids, samples=test_samples)

    expected_train_len = len(train_samples) * training.augmentation_repeats
    if len(train_dataset) == expected_train_len:
        results.ok(
            'train length with repeats',
            f'{len(train_samples)} * {training.augmentation_repeats} = {expected_train_len}',
        )
    else:
        results.fail(
            'train length with repeats',
            f'got {len(train_dataset)}, expected {expected_train_len}',
        )

    if len(test_dataset) == len(test_samples) and not test_dataset.training:
        results.ok('val length / no training flag', f'{len(test_dataset)} samples')
    else:
        results.fail(
            'val length / no training flag',
            f'len={len(test_dataset)} training={test_dataset.training}',
        )

    overlap = set(train_paths) & set(test_paths)
    if overlap:
        results.fail('split image leak', f'{len(overlap)} paths in both splits')
    else:
        results.ok('split image leak', 'no shared image paths')

    image_groups: dict = {}
    for sample in dataset.samples:
        image_groups.setdefault(sample[0], []).append(sample)
    multi = [path for path, group in image_groups.items() if len(group) > 1]
    split_broken = []
    for path in multi:
        in_train = any(sample[0] == path for sample in train_samples)
        in_test = any(sample[0] == path for sample in test_samples)
        if in_train and in_test:
            split_broken.append(str(path))
    if split_broken:
        results.fail('multi-person cohesion', f'{len(split_broken)} images split across sets')
    else:
        results.ok('multi-person cohesion', f'{len(multi)} multi-person images checked')


def check_simcc_targets(config, results: CheckResult):
    print('\n[7] SimCC targets')
    simcc = config['head'].custom
    if abs(train_mod.SIMCC_SPLIT_RATIO - simcc.split_ratio) > 1e-9:
        results.fail(
            'SIMCC_SPLIT_RATIO global',
            f'global={train_mod.SIMCC_SPLIT_RATIO} config={simcc.split_ratio}',
        )
    else:
        results.ok('SIMCC_SPLIT_RATIO global', str(train_mod.SIMCC_SPLIT_RATIO))

    if abs(train_mod.SIMCC_SIGMA - simcc.gaussian_sigma) > 1e-9:
        results.fail(
            'SIMCC_SIGMA global',
            f'global={train_mod.SIMCC_SIGMA} config={simcc.gaussian_sigma}',
        )
    else:
        results.ok('SIMCC_SIGMA global', str(train_mod.SIMCC_SIGMA))

    keypoints = np.zeros((train_mod.NUM_JOINTS, 3), dtype=np.float32)
    x, y = 100.0, 150.0
    keypoints[0] = [x, y, 2.0]
    keypoints[1] = [x, y, 1.0]
    keypoints[2] = [x, y, 0.0]
    keypoints[3] = [-5.0, y, 2.0]

    target_x, target_y, weights = generate_simcc_labels(
        keypoints,
        train_mod.IMAGE_SIZE,
        split_ratio=train_mod.SIMCC_SPLIT_RATIO,
        sigma=train_mod.SIMCC_SIGMA,
    )

    expected_bins_x = int(train_mod.IMAGE_SIZE.x * train_mod.SIMCC_SPLIT_RATIO)
    expected_bins_y = int(train_mod.IMAGE_SIZE.y * train_mod.SIMCC_SPLIT_RATIO)
    if target_x.shape[1] == expected_bins_x and target_y.shape[1] == expected_bins_y:
        results.ok('SimCC bin sizes', f'x={expected_bins_x} y={expected_bins_y}')
    else:
        results.fail(
            'SimCC bin sizes',
            f'got x={target_x.shape[1]} y={target_y.shape[1]}, '
            f'expected {expected_bins_x}/{expected_bins_y}',
        )

    peak_x = int(target_x[0].argmax())
    peak_y = int(target_y[0].argmax())
    mu_x = x * train_mod.SIMCC_SPLIT_RATIO
    mu_y = y * train_mod.SIMCC_SPLIT_RATIO
    if abs(peak_x - mu_x) <= 1.0 and abs(peak_y - mu_y) <= 1.0:
        results.ok('visible peak bins', f'peak=({peak_x},{peak_y}) mu=({mu_x:.1f},{mu_y:.1f})')
    else:
        results.fail(
            'visible peak bins',
            f'peak=({peak_x},{peak_y}) expected near ({mu_x:.1f},{mu_y:.1f})',
        )

    if float(weights[0]) == 1.0:
        results.ok('visible weight', '1.0')
    else:
        results.fail('visible weight', f'got {float(weights[0])}')

    if abs(float(weights[1]) - 0.35) < 1e-6:
        results.ok('occluded weight', '0.35')
    else:
        results.fail('occluded weight', f'got {float(weights[1])}')

    if float(weights[2]) == 0.0 and float(weights[3]) == 0.0:
        results.ok('invisible / OOB weight', '0.0')
    else:
        results.fail(
            'invisible / OOB weight',
            f'invisible={float(weights[2])} oob={float(weights[3])}',
        )


def run_checks(config) -> tuple[CheckResult, KeypointDataset | None, set | None]:
    results = CheckResult()
    annotation_file = check_config_mapping(config, results)
    if annotation_file is None:
        return results, None, None

    dataset, task_ids = check_annotation_images(annotation_file, results)
    check_keypoint_id_mapping(config, results)
    check_crop_geometry(dataset, results)
    check_horizontal_flip(results)
    check_repeats_and_split(dataset, config, task_ids, results)
    check_simcc_targets(config, results)
    return results, dataset, task_ids


def select_gallery_indices(samples, num_samples: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_image: dict = {}
    for index, (image_path, _kp, bbox) in enumerate(samples):
        by_image.setdefault(image_path, []).append((index, bbox))

    multi = [indices for indices in by_image.values() if len(indices) > 1]
    multi_indices = [index for group in multi for index, _ in group]
    rng.shuffle(multi_indices)

    border = []
    for index, (image_path, _kp, bbox) in enumerate(samples):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2:
            border.append(index)

    selected: list[int] = []
    half = max(1, num_samples // 2)
    for index in multi_indices:
        if len(selected) >= half:
            break
        if index not in selected:
            selected.append(index)

    for index in border:
        if len(selected) >= num_samples:
            break
        if index not in selected:
            selected.append(index)
            break  # at least one near-border if present

    remaining = [index for index in range(len(samples)) if index not in selected]
    rng.shuffle(remaining)
    for index in remaining:
        if len(selected) >= num_samples:
            break
        selected.append(index)

    return selected[:num_samples]


def write_gallery(dataset: KeypointDataset, config, out_dir: Path, num_samples: int):
    print('\n[Visual] Writing overlay gallery')
    raw_dir = out_dir / 'raw'
    crop_dir = out_dir / 'crop'
    flip_dir = out_dir / 'flip'
    for directory in (raw_dir, crop_dir, flip_dir):
        directory.mkdir(parents=True, exist_ok=True)

    names = joint_names_by_id(config['head'].keypoints)
    seed = config['training'].seed
    indices = select_gallery_indices(dataset.samples, num_samples, seed)
    eval_dataset = KeypointDataset(Path('.'), set(), training=False, samples=dataset.samples)

    for rank, sample_index in enumerate(indices):
        image_path, keypoints, bbox = dataset.samples[sample_index]
        stem = f'{rank:03d}_{image_path.stem}_p{sample_index}'

        # Raw annotation-space overlay
        with Image.open(image_path) as original:
            raw = original.convert('RGB')
        draw = ImageDraw.Draw(raw)
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1, y1, x2, y2), outline='yellow', width=3)
        draw_keypoints(raw, keypoints, names, color='cyan')
        raw.save(raw_dir / f'{stem}.jpg', quality=92)

        # Training crop (no aug)
        pixels, crop_keypoints = eval_dataset[sample_index]
        crop_image = denormalize_pixels(pixels)
        draw_keypoints(crop_image, crop_keypoints.numpy(), names, color='lime')
        crop_image.save(crop_dir / f'{stem}.jpg', quality=92)

        # Forced flip side-by-side
        flipped_image = ImageOps.mirror(denormalize_pixels(pixels))
        flipped_keypoints = horizontally_flip_keypoints(crop_keypoints.numpy())
        left = denormalize_pixels(pixels)
        draw_keypoints(left, crop_keypoints.numpy(), names, color='lime')
        right = flipped_image
        draw_keypoints(right, flipped_keypoints, names, color='orange')
        side = Image.new('RGB', (left.width * 2 + 8, left.height), color=(30, 30, 30))
        side.paste(left, (0, 0))
        side.paste(right, (left.width + 8, 0))
        side.save(flip_dir / f'{stem}.jpg', quality=92)

    print(f'  Wrote {len(indices)} samples to {out_dir}/{{raw,crop,flip}}/')


def print_checklist():
    print(
        '\nManual review checklist:\n'
        '  1. Raw: skeleton/bbox attached to the correct person (not another dancer).\n'
        '  2. Crop: full body visible with padding; feet/wrists not clipped unless bbox was wrong.\n'
        '  3. Flip: left/right joint names swap sides (check wrists and feet).\n'
        '  4. Multi-person frames: each sample matches its bbox.\n'
        '  5. Near-border crops: still aligned, not empty or wildly shifted.'
    )


def main():
    parser = argparse.ArgumentParser(description='Verify data pipeline (Tier 1 checks + Tier 2 overlays).')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out', type=Path, default=Path('temp/pipeline_verify'))
    parser.add_argument('--num-samples', type=int, default=30)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--checks-only', action='store_true')
    mode.add_argument('--visual-only', action='store_true')
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    setup_from_config(config)
    set_seed(config['training'].seed)

    results = CheckResult()
    dataset = None
    task_ids = None

    if not args.visual_only:
        results, dataset, task_ids = run_checks(config)
        print(
            f'\nSummary: {results.passed} passed, {results.failed} failed, '
            f'{results.warnings} warnings'
        )
    else:
        annotation_file = get_annotation_file_path(config['dataset'].annotation_version)
        root = ET.parse(annotation_file).getroot()
        task_ids = {video.get('id') for video in root.findall('./project/videos/video')}
        dataset = KeypointDataset(annotation_file, task_ids)
        results = CheckResult()

    if not args.checks_only and dataset is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        write_gallery(dataset, config, args.out, args.num_samples)
        print_checklist()

    if results.failed and not args.visual_only:
        sys.exit(1)


if __name__ == '__main__':
    main()
