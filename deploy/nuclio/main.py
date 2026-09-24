"""CVAT Nuclio handler for DinoCC dance_20 pose estimation."""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from dino_cc import DinoCC
from vec import IVec2

FUNCTION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = Path(os.environ.get('DINOCC_WEIGHTS', str(FUNCTION_DIR / 'weights' / 'best.pt')))
BACKBONE_DIR = Path(os.environ.get('DINOCC_BACKBONE', str(FUNCTION_DIR / 'backbone')))
MAPPING_PATH = FUNCTION_DIR / 'mapping' / 'dance_20.yaml'

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEFAULT_BBOX_PADDING = 0.05
# Joints below this SimCC confidence are marked outside for CVAT review.
OUTSIDE_CONFIDENCE = float(os.environ.get('DINOCC_OUTSIDE_CONF', '0.05'))


def load_keypoint_names(mapping_path: Path = MAPPING_PATH) -> list[str]:
    mapping = yaml.safe_load(mapping_path.read_text())
    ordered = sorted(mapping['keypoints'].items(), key=lambda item: item[1]['id'])
    return [name for name, _ in ordered]


def compute_padded_crop(bbox, original_width, original_height, image_size, bbox_padding):
    """Match training/predict crop: padded person box, aspect locked to model input."""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(
        bbox_width * (1 + 2 * bbox_padding),
        bbox_height * image_size.x / image_size.y,
    )
    crop_height = max(
        bbox_height * (1 + 2 * bbox_padding),
        crop_width * image_size.y / image_size.x,
    )
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    crop_x1 = max(0.0, min(center_x - crop_width / 2, original_width - crop_width))
    crop_y1 = max(0.0, min(center_y - crop_height / 2, original_height - crop_height))
    return crop_x1, crop_y1, crop_width, crop_height


def parse_regions(data, image_width, image_height):
    """Return list of xyxy bboxes from CVAT regions, or one full-frame fallback."""
    regions = data.get('regions') or []
    boxes = []
    for region in regions:
        points = region.get('points')
        if not points or len(points) < 4:
            continue
        x1, y1, x2, y2 = map(float, points[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        boxes.append((0.0, 0.0, float(image_width), float(image_height)))
    return boxes


def prepare_crop_tensor(image: Image.Image, crop_info, image_size: IVec2, device):
    crop_x1, crop_y1, crop_width, crop_height = crop_info
    cropped = image.crop(
        (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)
    )
    resized = cropped.resize((image_size.x, image_size.y), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return pixels.unsqueeze(0).to(device)


def decode_simcc(pred_x, pred_y, split_ratio: float):
    """Return crop-pixel coords [K, 2] and per-joint confidence [K]."""
    # Geometric mean of max Softmax mass on X/Y axes — calibrated enough for CVAT UI.
    conf = torch.sqrt(
        pred_x.softmax(dim=-1).amax(dim=-1) * pred_y.softmax(dim=-1).amax(dim=-1)
    )
    coords = torch.stack(
        (
            pred_x.argmax(dim=-1).float() / split_ratio,
            pred_y.argmax(dim=-1).float() / split_ratio,
        ),
        dim=-1,
    )
    return coords, conf


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    keypoint_ids = list(checkpoint['keypoint_ids'])
    image_w, image_h = checkpoint['image_size']
    image_size = IVec2(int(image_w), int(image_h))
    split_ratio = float(checkpoint.get('split_ratio', 2.0))
    head_kwargs = checkpoint.get('head_kwargs') or {}

    backbone_name = str(BACKBONE_DIR) if BACKBONE_DIR.is_dir() else checkpoint['backbone_name']
    model = DinoCC(
        num_joints=int(checkpoint['num_joints']),
        img_size=image_size,
        freeze_backbone=True,
        split_ratio=split_ratio,
        neck_dim=int(checkpoint.get('neck_dim', 256)),
        backbone_name=backbone_name,
        head_name=checkpoint.get('head_name', 'depthwise_simcc'),
        head_kwargs=head_kwargs,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint, keypoint_ids, image_size, split_ratio


def init_context(context):
    context.logger.info('Initializing DinoCC dance_20...')
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f'Missing checkpoint at {WEIGHTS_PATH}. '
            'Run deploy/nuclio/prepare_assets.sh or set DINOCC_WEIGHTS.'
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    context.logger.info(f'Using device: {device}')
    context.logger.info(f'Loading weights: {WEIGHTS_PATH}')

    model, checkpoint, keypoint_ids, image_size, split_ratio = build_model_from_checkpoint(
        WEIGHTS_PATH, device
    )
    keypoint_names = load_keypoint_names()
    if len(keypoint_names) != len(keypoint_ids):
        raise ValueError(
            f'Mapping has {len(keypoint_names)} joints but checkpoint has {len(keypoint_ids)}'
        )

    context.user_data.model = model
    context.user_data.device = device
    context.user_data.keypoint_ids = keypoint_ids
    context.user_data.keypoint_names = keypoint_names
    context.user_data.image_size = image_size
    context.user_data.split_ratio = split_ratio
    context.user_data.bbox_padding = float(
        checkpoint.get('bbox_padding', DEFAULT_BBOX_PADDING)
    )
    context.logger.info(
        f"Loaded {checkpoint.get('head_name')} / {checkpoint.get('backbone_name')} "
        f"image={image_size.x}x{image_size.y} joints={keypoint_names}"
    )


def handler(context, event):
    try:
        data = event.body
        if isinstance(data, (bytes, bytearray)):
            data = json.loads(data.decode('utf-8'))
        elif isinstance(data, str):
            data = json.loads(data)

        image = Image.open(io.BytesIO(base64.b64decode(data['image']))).convert('RGB')
        boxes = parse_regions(data, image.width, image.height)
        context.logger.info(f'Running pose on {len(boxes)} region(s)')

        model = context.user_data.model
        device = context.user_data.device
        image_size = context.user_data.image_size
        split_ratio = context.user_data.split_ratio
        bbox_padding = context.user_data.bbox_padding
        names = context.user_data.keypoint_names

        skeletons = []
        with torch.inference_mode():
            for bbox in boxes:
                crop_info = compute_padded_crop(
                    bbox, image.width, image.height, image_size, bbox_padding
                )
                pixels = prepare_crop_tensor(image, crop_info, image_size, device)
                pred_x, pred_y = model(pixels)
                coords, conf = decode_simcc(pred_x[0], pred_y[0], split_ratio)
                coords = coords.cpu().numpy()
                conf = conf.cpu().numpy()

                crop_x1, crop_y1, crop_width, crop_height = crop_info
                elements = []
                for index, name in enumerate(names):
                    x = float(crop_x1 + coords[index, 0] * crop_width / image_size.x)
                    y = float(crop_y1 + coords[index, 1] * crop_height / image_size.y)
                    score = float(conf[index])
                    elements.append(
                        {
                            'label': name,
                            'type': 'points',
                            'points': [x, y],
                            'confidence': f'{score:.4f}',
                            'outside': 0.0 if score >= OUTSIDE_CONFIDENCE else 1.0,
                        }
                    )

                skeletons.append(
                    {
                        'confidence': f'{float(np.mean(conf)):.4f}',
                        'label': 'body',
                        'type': 'skeleton',
                        'elements': elements,
                    }
                )

        return context.Response(
            body=json.dumps(skeletons),
            content_type='application/json',
            status_code=200,
        )
    except Exception as exc:
        context.logger.error(f'Inference error: {exc}')
        context.logger.error(traceback.format_exc())
        return context.Response(
            body=json.dumps([]),
            content_type='application/json',
            status_code=200,
        )
