from dataclasses import dataclass

@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    train_test_split: float
    gradient_clip_norm: float
    augmentation_repeats: int
    joint_loss_weights: dict[str, float]
    worst_test_overlays: int = 10
    compute_ap: bool = True
    # frame: random images; stratified_clip: every clip in both sets; clip: whole videos
    split_mode: str = 'frame'
    # Optional explicit holdout clip ids for split_mode=clip (e.g. ['task_15', 'task_10']).
    # When set, overrides random train_test_split clip assignment.
    test_clips: list[str] | None = None
    # Cap unique frames kept per task/clip (evenly spaced by frame index). None = no cap.
    max_frames_per_clip: int | None = None
    # Train-time augmentation (applied when KeypointDataset.training=True)
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.25
    aug_shift_frac: float = 0.1
    aug_rotation_deg: float = 12.0
    aug_color_jitter: float = 0.25
    # Single-person only: drop samples whose max IoU with another person in-frame
    # exceeds this. None = no filter. Solo people count as IoU 0.
    max_partner_iou: float | None = None
    # Single-person only: drop samples whose max IoU with another person in-frame
    # is below this. None = no filter. Solo people (IoU 0) are dropped when set.
    min_partner_iou: float | None = None
    # Phase A: weight on L1 bone-vector loss (0 = off). Applied on soft-decoded SimCC coords.
    bone_loss_weight: float = 0.0
    # Penalize L/R joint assignment when swapped assignment fits GT better (0 = off).
    lr_swap_loss_weight: float = 0.0
    # Margin (fraction of image diagonal) for joint L/R swap hinge.
    lr_swap_margin: float = 0.02
    # Chirality: relu(-cos(pred_R-L, gt_R-L)) on L/R pairs with enough GT separation.
    chirality_loss_weight: float = 0.0
    # Skip chirality when ||gt_R - gt_L|| / image_diag is below this (closed feet).
    chirality_min_sep: float = 0.03
    # What drives best.pt + early stopping: val_loss (default) or ap (OKS-AP on test).
    checkpoint_metric: str = 'val_loss'
