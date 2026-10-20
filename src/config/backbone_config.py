from dataclasses import dataclass

@dataclass
class BackboneConfig:
    name: str
    type: str
    patch_size: int
    hidden_size: int
    # If True, backbone params are frozen. If False, train all (or last N blocks).
    freeze: bool = True
    # When freeze=False: if set, only the last N encoder blocks (+ LN) are trainable.
    unfreeze_last_blocks: int | None = None
    # Optional backbone LR (AdamW param group). None → training.learning_rate when
    # fully frozen head-only; when backbone is trainable defaults to 0.1× head LR.
    learning_rate: float | None = None
    # VitPose++ MoE expert index (COCO=0). Ignored for non-MoE backbones.
    dataset_index: int = 0