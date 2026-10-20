from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path

@dataclass
class CocoWholeBodyConfig:
    enabled: bool = False
    ann_file: str = ''
    image_root: str = ''
    dance_fraction: float = 0.25
    max_samples: int | None = None
    min_labeled_joints: int = 5
    proxy_glabella_from_nose: bool = True
    proxy_heel_spike_from_heel: bool = True
    # When true, keep a COCO pool and draw a fresh dance_fraction-sized subset each epoch.
    rolling: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> CocoWholeBodyConfig:
        if not raw:
            return cls(enabled=False)
        return cls(
            enabled=bool(raw.get('enabled', False)),
            ann_file=str(raw.get('ann_file', '')),
            image_root=str(raw.get('image_root', '')),
            dance_fraction=float(raw.get('dance_fraction', 0.25)),
            max_samples=(
                None
                if raw.get('max_samples') is None
                else int(raw['max_samples'])
            ),
            min_labeled_joints=int(raw.get('min_labeled_joints', 5)),
            proxy_glabella_from_nose=bool(raw.get('proxy_glabella_from_nose', True)),
            proxy_heel_spike_from_heel=bool(raw.get('proxy_heel_spike_from_heel', True)),
            rolling=bool(raw.get('rolling', False)),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.ann_file:
            raise ValueError('coco_wholebody.ann_file is required when enabled')
        if not self.image_root:
            raise ValueError('coco_wholebody.image_root is required when enabled')
        if not (0.0 < self.dance_fraction <= 1.0):
            raise ValueError(
                f'coco_wholebody.dance_fraction must be in (0, 1], got {self.dance_fraction}'
            )
        if self.min_labeled_joints < 1:
            raise ValueError(
                f'coco_wholebody.min_labeled_joints must be >= 1, got {self.min_labeled_joints}'
            )
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(
                f'coco_wholebody.max_samples must be >= 1 or null, got {self.max_samples}'
            )
        ann_path = Path(self.ann_file)
        image_root = Path(self.image_root)
        if not ann_path.is_file():
            raise FileNotFoundError(f'COCO WholeBody ann_file not found: {ann_path}')
        if not image_root.is_dir():
            raise FileNotFoundError(f'COCO image_root not found: {image_root}')
