from dataclasses import dataclass, field

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
    # Optional full-clip holdouts for split_mode=clip or stratified_clip
    # (e.g. ['task_14', 'task_57']). For clip: overrides random assignment.
    # For stratified_clip: those clips are all-test; other clips stay stratified.
    test_clips: list[str] | None = None
    # Cap unique frames kept per task/clip. None = no cap.
    # Clips with ≤ this many frames keep all frames; larger clips are subsampled.
    max_frames_per_clip: int | None = None
    # If true, resample oversize train clips each epoch (random frames up to the cap).
    # Short clips stay fully included. Val/test still use a fixed even-spaced cap.
    # Requires max_frames_per_clip. Incompatible with temporal windows.
    rolling_max_frames_per_clip: bool = False
    # Optional AdamW LR for head.pair_bias_values only. None = share head learning_rate.
    # When set, pair-bias params use this LR and weight_decay=0 (bias scalars).
    pair_bias_learning_rate: float | None = None
    # Temporal windows (0 / None = single-frame training). Requires contiguous frames.
    window_size: int = 0
    window_stride: int = 1
    # Minimum windows required to start a temporal run (fail loudly otherwise).
    min_temporal_windows: int = 16
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
    # Denom = image_diag * scale_frac (smaller → stronger hinge vs SimCC KL).
    lr_swap_scale_frac: float = 1.0
    # Extra weight on hip/knee/ankle/foot L/R pairs inside swap loss.
    lr_swap_foot_pair_weight: float = 1.0
    # Blend mean over near-swap samples into swap loss (0 = soft mean only).
    lr_swap_hard_mix: float = 0.0
    # Chirality: relu(-cos(pred_R-L, gt_R-L)) on L/R pairs with enough GT separation.
    chirality_loss_weight: float = 0.0
    # Skip chirality when ||gt_R - gt_L|| / image_diag is below this (closed feet).
    chirality_min_sep: float = 0.03
    # Extra weight on lower-body pairs inside chirality loss.
    chirality_foot_pair_weight: float = 1.0
    # Differentiable OKS-shaped coordinate loss on soft-decoded joint locations.
    oks_loss_weight: float = 0.0
    # Body-part weights used by the priority_oks checkpoint metric.
    priority_part_weights: dict[str, float] = field(
        default_factory=lambda: {'feet': 0.4, 'legs': 0.3, 'arms': 0.2, 'head': 0.1}
    )
    # What drives best.pt + early stopping: val_loss, ap, or priority_oks.
    checkpoint_metric: str = 'val_loss'
    # Dashboard / observability: write val_preds/epoch_XXX.npz every N epochs (0 = off).
    save_val_preds_every_n_epochs: int = 1
    # Worst-N overlay JPGs under val_overlays/epoch_XXX/ each time preds are saved.
    epoch_overlay_top_n: int = 8
    # Always render overlays for these image stems (e.g. task_45_frame_000384).
    watchlist_frames: list[str] = field(default_factory=list)
