from dataclasses import dataclass

@dataclass
class JointQuerySimCCConfig:
    split_ratio: float
    x_bins: int
    y_bins: int
    gaussian_sigma: float
    # Wider SimCC label for occluded (v==1); None → same as gaussian_sigma
    gaussian_sigma_occluded: float | None = None
    # Loss weight for occluded joints (visible always 1.0)
    occluded_joint_weight: float = 0.35
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.0
    # Soft-anchor foot queries to same-side ankle: q_foot += σ(α) * W(q_ankle)
    soft_foot_anchor: bool = False
    soft_foot_anchor_init: float = 0.4
    use_role: bool = False
    role_dropout: float = 0.1
    # If false: query bias only (no FiLM on patch memory)
    role_film: bool = True
    # Per-layer q/k conditioning: role + joint id (+ optional spatial) every decoder layer
    per_layer_role_reinject: bool = False
    use_spatial_prompt: bool = False
    deep_supervision: bool = False
    deep_supervision_aux_weight: float = 0.5
    xy_consistency_loss_weight: float = 0.0
    # Phase B: embed soft-decoded joint (x,y) into queries between decoder layers
    use_joint_pos_embed: bool = False
    # Learned self-attn bias on skeleton edges; L/R mirror edges share params
    use_pair_bias: bool = False
    pair_bias_init: float = 2.0
    # Soft-decoded joint XY → Gaussian bias on cross-attn over patch grid
    use_local_cross_attn: bool = False
    local_attn_sigma: float = 2.0
    local_attn_strength: float = 2.0