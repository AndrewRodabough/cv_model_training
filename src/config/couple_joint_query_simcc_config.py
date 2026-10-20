from dataclasses import dataclass

@dataclass
class CoupleJointQuerySimCCConfig:
    """Couple head: 2K queries on union crops of lead+follow pairs."""

    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    gaussian_sigma_occluded: float | None = None
    occluded_joint_weight: float = 0.35
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    soft_foot_anchor: bool = False
    soft_foot_anchor_init: float = 0.4
    # Keep only lead+follow pairs with bbox IoU >= this threshold
    couple_min_iou: float = 0.3