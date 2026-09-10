import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from dino_cc import DinoCC
from train import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, KEYPOINT_IDS, BBOX_PADDING
from vec import IVec2


SKELETON_CONNECTIONS = (
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (15, 16),
    (16, 17),
    (18, 19),
    (19, 20),
    (17, 21),
    (17, 22),
    (17, 23),
    (17, 24),
    (20, 25),
    (20, 26),
    (20, 27),
    (20, 28),
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


def prepare_image(image, crop_info):
    crop_x1, crop_y1, crop_width, crop_height = crop_info
    resized = image.resize((IMAGE_SIZE.x, IMAGE_SIZE.y), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1) / 255.0
    pixels = (pixels - IMAGE_MEAN) / IMAGE_STD
    return pixels.unsqueeze(0), crop_info


def draw_prediction(image, coordinates, crop_info, output_path):
    crop_x1, crop_y1, crop_width, crop_height = crop_info
    draw = ImageDraw.Draw(image)
    if crop_width != image.width or crop_height != image.height:
        draw.rectangle(
            (crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height),
            outline='yellow',
            width=3,
        )

    points = []
    for x, y in coordinates:
        original_x = crop_x1 + x * crop_width / IMAGE_SIZE.x
        original_y = crop_y1 + y * crop_height / IMAGE_SIZE.y
        points.append((original_x, original_y))

    for index, (x, y) in enumerate(points):
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='white')
        draw.text((x + 6, y - 6), str(KEYPOINT_IDS[index]), fill='white', stroke_width=2, stroke_fill='black')

    point_by_id = dict(zip(KEYPOINT_IDS, points))
    for first_id, second_id in SKELETON_CONNECTIONS:
        first = point_by_id.get(first_id)
        second = point_by_id.get(second_id)
        if first is not None and second is not None:
            draw.line((first, second), fill='lime', width=2)

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Overlay DinoCC keypoint predictions on an image.')
    parser.add_argument('weights', type=Path, help='Path to a DinoCC checkpoint.')
    parser.add_argument('image', type=Path, help='Path to the image to process.')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'), help='Optional person bbox in source-image pixels.')
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
    checkpoint_keypoint_ids = checkpoint.get('keypoint_ids', KEYPOINT_IDS)
    if list(checkpoint_keypoint_ids) != KEYPOINT_IDS:
        raise ValueError(
            f'Checkpoint keypoints {checkpoint_keypoint_ids} do not match this inference script {KEYPOINT_IDS}.'
        )

    model = DinoCC(num_joints=len(KEYPOINT_IDS), img_size=IVec2(IMAGE_SIZE.x, IMAGE_SIZE.y)).to(device)
    model.load_weights(args.weights, map_location=device)
    model.eval()

    image = Image.open(args.image).convert('RGB')
    bbox = parse_bbox(args.bbox)
    cropped, crop_info = crop_image(image, bbox)
    pixels, _ = prepare_image(cropped, (0.0, 0.0, float(cropped.width), float(cropped.height)))
    pixels = pixels.to(device)

    with torch.inference_mode():
        pred_x, pred_y = model(pixels)
        coordinates = torch.stack(
            (
                pred_x[0].argmax(dim=-1).float() / 2.0,
                pred_y[0].argmax(dim=-1).float() / 2.0,
            ),
            dim=1,
        ).cpu().numpy()

    draw_prediction(image, coordinates, crop_info, args.output)
    print(f'Saved overlay to {args.output}')


if __name__ == '__main__':
    main()
