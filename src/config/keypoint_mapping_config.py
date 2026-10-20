from dataclasses import dataclass

@dataclass
class keypoint_mapping:
    name: str
    num_keypoints: int
    keypoints: dict
    left_right_pairs: list | None = None
    oks_sigmas: dict[str, float] | None = None