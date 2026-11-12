"""Render joint-query cross-attn heatmaps (PIL only — no torch)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.dashboard.overlay import make_crop_view


def _jet_rgba(values: np.ndarray) -> np.ndarray:
    """Map [0,1] → RGBA uint8 with a simple matplotlib-like jet (no mpl dep)."""
    x = np.clip(values.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    a = np.clip(0.15 + 0.75 * x, 0.0, 0.9)
    out = np.stack([r, g, b, a], axis=-1)
    return (out * 255.0).astype(np.uint8)


def normalize_attn_map(attn_hw: np.ndarray, *, percentile: float = 99.0) -> np.ndarray:
    """Normalize a single-joint [H,W] map to [0,1] with robust peak scaling."""
    arr = np.asarray(attn_hw, dtype=np.float32)
    flat = arr.reshape(-1)
    if flat.size == 0:
        return arr
    lo = float(np.min(flat))
    hi = float(np.percentile(flat, percentile))
    if hi <= lo + 1e-12:
        hi = float(np.max(flat))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def render_attn_on_crop(
    image_path: str | Path,
    bbox,
    attn_hw: np.ndarray,
    *,
    image_size: tuple[int, int],
    bbox_padding: float = 0.08,
    max_side: int = 960,
    title: str | None = None,
) -> Image.Image:
    """Overlay one joint's patch-grid attn on the person crop."""
    view = make_crop_view(
        image_path,
        bbox,
        image_size=image_size,
        bbox_padding=bbox_padding,
        display_side=max_side,
    )
    base = view.base.convert('RGBA')
    norm = normalize_attn_map(attn_hw)
    heat = Image.fromarray(_jet_rgba(norm), mode='RGBA')
    heat = heat.resize(base.size, Image.Resampling.BILINEAR)
    blended = Image.alpha_composite(base, heat).convert('RGB')
    if title:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(blended)
        draw.text((6, 6), title, fill='yellow')
    return blended


def attn_joint_index(joint_names: Sequence[str], joint: str) -> int:
    names = [str(n) for n in joint_names]
    if joint in names:
        return names.index(joint)
    raise KeyError(f'Joint {joint!r} not in {names}')
