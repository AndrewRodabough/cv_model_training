import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from src.training.dino_cc import DinoCC
from src.training.train import (
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    KEYPOINT_IDS,
    BBOX_PADDING,
    crop_normalized_bbox_center,
    encode_role,
)
from src.util.vec import IVec2


SKELETON_CONNECTIONS = (
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (15, 16),
    (16, 17),
    (18, 19),
    (19, 20),
    (25, 23),
    (25, 24),
    (15, 25),
    (25, 26),
    (23, 24),
    (29, 27),
    (29, 28),
    (16, 29),
    (29, 30),
    (27, 28),
)


def parse_bbox(values):
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError('BBox must contain exactly four values: x1 y1 x2 y2.')
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError('BBox must satisfy x2 > x1 and y2 > y1.')
    return tuple(values)


def crop_image(image, bbox):
    if bbox is None:
        return image, (0.0, 0.0, float(image.width), float(image.height))

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
    crop_width = min(crop_width, image.width)
    crop_height = min(crop_height, image.height)
    crop_x1 = max(0.0, min(center_x - crop_width / 2, image.width - crop_width))
    crop_y1 = max(0.0, min(center_y - crop_height / 2, image.height - crop_height))
    return image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height)), (
        crop_x1,
        crop_y1,
        crop_width,
        crop_height,
    )


def prepare_image(image, _crop_info):
    resized = image.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return pixels.unsqueeze(0), _crop_info


def draw_prediction(
    image,
    coordinates,
    crop_info,
    output_path,
    person_joints=None,
    keypoint_ids=None,
):
    keypoint_ids = KEYPOINT_IDS if keypoint_ids is None else keypoint_ids
    crop_x1, crop_y1, crop_width, crop_height = crop_info
    draw = ImageDraw.Draw(image)
    if crop_width != image.width or crop_height != image.height:
        draw.rectangle(
            (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height),
            outline='yellow',
            width=1,
        )

    if person_joints is None:
        person_joints = len(KEYPOINT_IDS)
    banks = (
        [(0, person_joints, 'lime', 'red'), (person_joints, 2 * person_joints, 'cyan', 'magenta')]
        if len(coordinates) == 2 * person_joints
        else [(0, len(coordinates), 'lime', 'red')]
    )

    for start, end, line_color, point_color in banks:
        bank_coords = coordinates[start:end]
        points = []
        for x, y in bank_coords:
            original_x = crop_x1 + x * crop_width / IMAGE_SIZE.x
            original_y = crop_y1 + y * crop_height / IMAGE_SIZE.y
            points.append((original_x, original_y))

        point_by_id = dict(zip(keypoint_ids, points))
        for first_id, second_id in SKELETON_CONNECTIONS:
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
                str(keypoint_ids[index]),
                fill='white',
                stroke_width=1,
                stroke_fill='black',
            )

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Overlay DinoCC keypoint predictions on an image.')
    parser.add_argument('weights', type=Path, help='Path to a DinoCC checkpoint.')
    parser.add_argument('image', type=Path, help='Path to the image to process.')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'), help='Optional person (or union) bbox in source-image pixels.')
    parser.add_argument(
        '--role',
        choices=('lead', 'follow'),
        default=None,
        help='Optional dance role for role-conditioned checkpoints ([is_lead, is_follow]).',
    )
    parser.add_argument('--output', type=Path, default=Path('prediction_overlay.jpg'))
    args = parser.parse_args()

    if args.bbox is None:
        print(
            'Warning: no bbox supplied. This model was trained on padded person '
            'crops, so full-frame predictions may have incorrect scale. '
            'Use --bbox X1 Y1 X2 Y2 for reliable inference.'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.weights, map_location=device)
    checkpoint_keypoint_ids = list(checkpoint.get('keypoint_ids', KEYPOINT_IDS))

    couple_mode = bool(checkpoint.get('couple_mode', False)) or (
        checkpoint.get('head_name') == 'couple_joint_query_self_attn_simcc'
    )
    person_joints = int(checkpoint.get('person_joints', len(checkpoint_keypoint_ids)))
    if len(checkpoint_keypoint_ids) != person_joints:
        raise ValueError(
            f'Checkpoint keypoint metadata has {len(checkpoint_keypoint_ids)} IDs '
            f'but person_joints={person_joints}.'
        )
    num_joints = int(checkpoint.get('num_joints', 2 * person_joints if couple_mode else person_joints))

    model = DinoCC(
        num_joints=num_joints,
        img_size=IVec2(IMAGE_SIZE.x, IMAGE_SIZE.y),
        backbone_name=checkpoint.get('backbone_name', 'facebook/dinov2-base'),
        head_name=checkpoint.get('head_name', 'depthwise_simcc'),
        head_kwargs=checkpoint.get('head_kwargs'),
        neck_dim=checkpoint.get('neck_dim', 256),
        split_ratio=checkpoint.get('split_ratio', 2.0),
    ).to(device)
    model.load_weights(args.weights, map_location=str(device))
    model.eval()

    image = Image.open(args.image).convert('RGB')
    bbox = parse_bbox(args.bbox)
    cropped, crop_info = crop_image(image, bbox)
    pixels, _ = prepare_image(cropped, (0.0, 0.0, float(cropped.width), float(cropped.height)))
    pixels = pixels.to(device)
    use_role = bool(checkpoint.get('use_role', False)) or bool(
        (checkpoint.get('head_kwargs') or {}).get('use_role', False)
    )
    if couple_mode:
        use_role = False
        if args.bbox is None:
            print(
                'Warning: couple-head checkpoints expect a union lead+follow bbox; '
                'full-frame input may be unreliable.'
            )
    role = encode_role(args.role).unsqueeze(0).to(device) if use_role else None
    if use_role and args.role is None:
        print('Warning: role-conditioned checkpoint but --role not set; using [0, 0].')

    head_kwargs = checkpoint.get('head_kwargs') or {}
    use_spatial = bool(head_kwargs.get('use_spatial_prompt', False))
    spatial = None
    if use_spatial:
        if bbox is None:
            raise ValueError(
                'This checkpoint uses spatial prompting; --bbox is required.'
            )

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        x1, y1, w, h = crop_info
        spatial = crop_normalized_bbox_center(cx, cy, x1, y1, w, h).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(pixels, role=role, spatial=spatial)
        pred_x, pred_y = out[0], out[1]
        coordinates = torch.stack(
            (
                pred_x[0].argmax(dim=-1).float() / 2.0,
                pred_y[0].argmax(dim=-1).float() / 2.0,
            ),
            dim=1,
        ).cpu().numpy()

    draw_prediction(
        image,
        coordinates,
        crop_info,
        args.output,
        person_joints=person_joints if couple_mode else None,
        keypoint_ids=checkpoint_keypoint_ids,
    )
    print(f'Saved overlay to {args.output}')


if __name__ == '__main__':
    main()
