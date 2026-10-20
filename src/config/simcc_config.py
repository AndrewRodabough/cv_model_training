from dataclasses import dataclass

@dataclass
class SimCCConfig:
    pointwise_convolutions: int
    depthwise_kernel: int
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    pooling: str
    use_role: bool = False
    role_dropout: float = 0.1
    role_film: bool = True
    gaussian_sigma_occluded: float | None = None
    occluded_joint_weight: float = 0.35