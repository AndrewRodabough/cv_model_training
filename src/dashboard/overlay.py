"""Lightweight skeleton overlay drawing for the dashboard (no torch/train import)."""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from PIL import Image, ImageDraw

# Keep in sync with src/training/train.py SKELETON_BONE_NAME_LINKS.
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

# Normalize every crop so the longest side hits this size (upscale small boxes too).
DEFAULT_DISPLAY_SIDE = 960
DEFAULT_DISPLAY_MAX_HEIGHT = 960


def load_keypoint_format(mapping_path: Path) -> dict[str, Any]:
    with open(mapping_path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def skeleton_connections(keypoints_by_name: dict) -> list[tuple[int, int]]:
    connections = []
    for parent_name, child_name in SKELETON_BONE_NAME_LINKS:
        if parent_name not in keypoints_by_name or child_name not in keypoints_by_name:
            continue
        connections.append(
            (
                int(keypoints_by_name[parent_name]['id']),
                int(keypoints_by_name[child_name]['id']),
            )
        )
    return connections


def compute_padded_crop(
    bbox,
    original_width: int,
    original_height: int,
    image_width: int,
    image_height: int,
    bbox_padding: float,
):
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_width = max(
        bbox_width * (1 + 2 * bbox_padding),
        bbox_height * image_width / image_height,
    )
    crop_height = max(
        bbox_height * (1 + 2 * bbox_padding),
        crop_width * image_height / image_width,
    )
    crop_width = min(crop_width, original_width)
    crop_height = min(crop_height, original_height)
    crop_x1 = max(0.0, min(center_x - crop_width / 2, original_width - crop_width))
    crop_y1 = max(0.0, min(center_y - crop_height / 2, original_height - crop_height))
    return crop_x1, crop_y1, crop_width, crop_height


@dataclass(frozen=True)
class CropView:
    """Cached person crop for fast pose redraws across epochs."""

    base: Image.Image  # RGB, display-sized
    model_w: int
    model_h: int


def make_crop_view(
    image_path: str | Path,
    bbox,
    *,
    image_size: tuple[int, int],
    bbox_padding: float = 0.08,
    display_side: int = DEFAULT_DISPLAY_SIDE,
) -> CropView:
    image = Image.open(image_path).convert('RGB')
    image_w, image_h = image_size
    crop_x1, crop_y1, crop_width, crop_height = compute_padded_crop(
        bbox,
        image.width,
        image.height,
        image_w,
        image_h,
        bbox_padding,
    )
    left = int(round(crop_x1))
    top = int(round(crop_y1))
    right = int(round(crop_x1 + crop_width))
    bottom = int(round(crop_y1 + crop_height))
    crop = image.crop((left, top, right, bottom))
    # Upscale small person boxes and downscale large ones to a consistent display size.
    if display_side > 0 and max(crop.size) > 0:
        scale = float(display_side) / float(max(crop.size))
        if abs(scale - 1.0) > 1e-3:
            crop = crop.resize(
                (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
                Image.Resampling.LANCZOS,
            )
    return CropView(base=crop, model_w=image_w, model_h=image_h)


def keypoint_ids_sorted(keypoints_by_name: dict) -> list[int]:
    return [
        int(meta['id'])
        for _, meta in sorted(keypoints_by_name.items(), key=lambda item: item[1]['id'])
    ]


def model_coords_to_display(
    view: CropView,
    coordinates: Sequence[Sequence[float]],
) -> list[tuple[float, float]]:
    disp_w, disp_h = view.base.size
    return [
        (
            float(x) * disp_w / view.model_w,
            float(y) * disp_h / view.model_h,
        )
        for x, y in coordinates
    ]


def image_to_data_url(image: Image.Image, *, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}'


def pose_overlay_svg(
    width: int,
    height: int,
    points: Sequence[tuple[float, float]],
    *,
    keypoint_ids: Sequence[int],
    connections: Sequence[tuple[int, int]],
    loss: float | None = None,
    oks: float | None = None,
    title: str | None = None,
) -> str:
    """Transparent SVG pose layer drawn in display-crop coordinates."""
    point_by_id = dict(zip(keypoint_ids, points))
    # Fill the photo box exactly (parent #pose-overlay-slot is already absolute).
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" '
        f'style="display:block;width:100%;height:100%;overflow:visible;'
        f'pointer-events:none;">'
    ]
    for first_id, second_id in connections:
        first = point_by_id.get(first_id)
        second = point_by_id.get(second_id)
        if first is None or second is None:
            continue
        parts.append(
            f'<line x1="{first[0]:.2f}" y1="{first[1]:.2f}" '
            f'x2="{second[0]:.2f}" y2="{second[1]:.2f}" '
            f'stroke="lime" stroke-width="2" stroke-linecap="round" />'
        )
    for index, (x, y) in enumerate(points):
        kid = keypoint_ids[index] if index < len(keypoint_ids) else index
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" '
            f'fill="red" stroke="white" stroke-width="1" />'
            f'<text x="{x + 4:.2f}" y="{y - 4:.2f}" '
            f'fill="white" font-size="11" font-family="sans-serif">{kid}</text>'
        )
    label_bits = []
    if title:
        label_bits.append(title)
    if loss is not None:
        label_bits.append(f'loss={loss:.4f}')
    if oks is not None:
        label_bits.append(f'oks={oks:.4f}')
    if label_bits:
        label = html.escape(' | '.join(label_bits))
        parts.append(
            f'<text x="6" y="16" fill="yellow" font-size="13" '
            f'font-family="sans-serif" stroke="black" stroke-width="0.4">{label}</text>'
        )
    parts.append('</svg>')
    return ''.join(parts)


def compose_pose_viewer_html(
    data_url: str,
    width: int,
    height: int,
    overlay_svg: str,
    *,
    max_height: int = DEFAULT_DISPLAY_MAX_HEIGHT,
) -> str:
    """Photo + SVG in one shrink-wrapped box so they share the same alignment.

    ``#pose-overlay-slot`` is updated in-place on epoch scrub so the ``<img>``
    node is not replaced (no photo flicker).
    """
    return (
        f'<div id="pose-viewer" style="position:relative;display:inline-block;'
        f'width:min(100%,{width}px);max-width:100%;line-height:0;background:#111;">'
        f'<img id="pose-base-img" src="{data_url}" width="{width}" height="{height}" '
        f'alt="frame crop" '
        f'style="display:block;width:100%;height:auto;'
        f'max-height:min(90vh,{max_height}px);" />'
        f'<div id="pose-overlay-slot" style="position:absolute;left:0;top:0;'
        f'width:100%;height:100%;pointer-events:none;">{overlay_svg}</div>'
        f'</div>'
    )


def empty_pose_viewer_html() -> str:
    return (
        '<div style="color:#888;padding:1rem;line-height:1.4;">'
        'Select a frame to load the photo + keypoint layers.'
        '</div>'
    )


def draw_pose_on_crop(
    view: CropView,
    coordinates,
    *,
    keypoints_by_name: dict,
    loss: float | None = None,
    oks: float | None = None,
    title: str | None = None,
) -> Image.Image:
    """Draw skeleton onto a copy of the cached crop (fallback / non-layered use)."""
    image = view.base.copy()
    draw = ImageDraw.Draw(image)
    keypoint_ids = keypoint_ids_sorted(keypoints_by_name)
    connections = skeleton_connections(keypoints_by_name)
    points = model_coords_to_display(view, coordinates)
    point_by_id = dict(zip(keypoint_ids, points))
    for first_id, second_id in connections:
        first = point_by_id.get(first_id)
        second = point_by_id.get(second_id)
        if first is not None and second is not None:
            draw.line((first, second), fill='lime', width=2)
    for index, (x, y) in enumerate(points):
        radius = 3
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill='red',
            outline='white',
        )
        draw.text((x + 4, y - 4), str(keypoint_ids[index]), fill='white')

    label_bits = []
    if title:
        label_bits.append(title)
    if loss is not None:
        label_bits.append(f'loss={loss:.4f}')
    if oks is not None:
        label_bits.append(f'oks={oks:.4f}')
    if label_bits:
        draw.text((4, 4), ' | '.join(label_bits), fill='yellow')
    return image


def draw_prediction_overlay(
    image_path: str | Path,
    bbox,
    coordinates,
    *,
    keypoints_by_name: dict,
    image_size: tuple[int, int],
    bbox_padding: float = 0.08,
    loss: float | None = None,
    oks: float | None = None,
    title: str | None = None,
    display_side: int = DEFAULT_DISPLAY_SIDE,
) -> Image.Image:
    """Load → crop → draw. Prefer make_crop_view + draw_pose_on_crop when scrubbing."""
    view = make_crop_view(
        image_path,
        bbox,
        image_size=image_size,
        bbox_padding=bbox_padding,
        display_side=display_side,
    )
    return draw_pose_on_crop(
        view,
        coordinates,
        keypoints_by_name=keypoints_by_name,
        loss=loss,
        oks=oks,
        title=title,
    )
