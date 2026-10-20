from dataclasses import dataclass
from typing import Any

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