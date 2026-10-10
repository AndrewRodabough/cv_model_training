import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Tuple

from vec import IVec2


def load_backbone(backbone_name: str):
  """Load a HF ViT backbone from a Hub id or local directory.

  A local directory with only ``config.json`` (no weight files) uses
  ``from_config`` so deploy packages can ship architecture metadata and load
  weights exclusively from a training checkpoint.
  """
  backbone_path = Path(backbone_name)
  if backbone_path.is_dir():
    config = AutoConfig.from_pretrained(backbone_path)
    weight_names = ('model.safetensors', 'pytorch_model.bin', 'model.bin', 'model.pt')
    if any((backbone_path / name).exists() for name in weight_names):
      return AutoModel.from_pretrained(backbone_path)
    return AutoModel.from_config(config)
  return AutoModel.from_pretrained(backbone_name)

class BaseSimCCHead(nn.Module, ABC):
  """Formal interface for any SimCC head variant."""

  @abstractmethod
  def forward(
      self, patch_tokens: torch.Tensor, role: torch.Tensor | None = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Args:

        patch_tokens: [B, N, C]
        role: optional [B, 2] lead/follow bits (is_lead, is_follow)

    Returns:
        (pred_x, pred_y): ([B, K, W_bins], [B, K, H_bins])
    """
    pass


ROLE_DIM = 2


class RoleConditioner(nn.Module):
  """Map lead/follow bits to a query bias and optional FiLM (gamma, beta) on features."""

  def __init__(self, dim: int, use_film: bool = True):
    super().__init__()
    self.use_film = use_film
    self.query_embed = nn.Linear(ROLE_DIM, dim, bias=False)
    # Small nonzero query bias so roles differ immediately.
    nn.init.normal_(self.query_embed.weight, std=0.02)
    if use_film:
      self.film = nn.Linear(ROLE_DIM, 2 * dim)
      # FiLM starts as identity (gamma=0, beta=0).
      nn.init.zeros_(self.film.weight)
      nn.init.zeros_(self.film.bias)
    else:
      self.film = None

  def bias_queries(self, queries: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    # queries: [B, K, D], role: [B, 2]
    return queries + self.query_embed(role).unsqueeze(1)

  def film_features(self, features: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    if self.film is None:
      return features
    # features: [B, N, D] or [B, D, H, W]; role: [B, 2]
    gamma, beta = self.film(role).chunk(2, dim=-1)
    if features.dim() == 4:
      # Conv feature map [B, D, H, W]
      gamma = gamma.unsqueeze(-1).unsqueeze(-1)
      beta = beta.unsqueeze(-1).unsqueeze(-1)
    else:
      gamma = gamma.unsqueeze(1)
      beta = beta.unsqueeze(1)
    return features * (1.0 + gamma) + beta

  def resolve_role(
      self, role: torch.Tensor | None, batch_size: int, device, dtype
  ) -> torch.Tensor:
    if role is None:
      return torch.zeros(batch_size, ROLE_DIM, device=device, dtype=dtype)
    if role.shape != (batch_size, ROLE_DIM):
      raise ValueError(
          f'Expected role shape ({batch_size}, {ROLE_DIM}), got {tuple(role.shape)}'
      )
    return role.to(device=device, dtype=dtype)


class DepthwiseSimCCHead(BaseSimCCHead):
  """Your current head with 1x1 + 3x3 depthwise + 1x1 neck."""

  def __init__(
      self,
      in_channels: int,
      num_joints: int,
      grid_size: IVec2,
      out_size: IVec2,
      split_ratio: float = 2.0,
      neck_dim: int = 256,
      use_role: bool = False,
      role_film: bool = True,
      **kwargs,
  ):
    super().__init__()
    self.grid_size = grid_size
    self.neck_dim = neck_dim
    self.use_role = use_role
    self.role_film = role_film
    self.out_bins_size = IVec2(
        int(out_size.x * split_ratio), int(out_size.y * split_ratio)
    )

    self.neck = nn.Sequential(
        nn.Conv2d(in_channels, neck_dim, kernel_size=1, bias=False),
        nn.GroupNorm(32, neck_dim),
        nn.GELU(),
        nn.Conv2d(
            neck_dim,
            neck_dim,
            kernel_size=3,
            padding=1,
            groups=neck_dim,
            bias=False,
        ),
        nn.Conv2d(neck_dim, neck_dim, kernel_size=1, bias=False),
        nn.GroupNorm(32, neck_dim),
        nn.GELU(),
    )

    self.mlp_x = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, None)),
        nn.Flatten(start_dim=2),
        nn.Linear(self.grid_size.x, self.out_bins_size.x),
    )
    self.fc_x = nn.Linear(neck_dim, num_joints)

    self.mlp_y = nn.Sequential(
        nn.AdaptiveAvgPool2d((None, 1)),
        nn.Flatten(start_dim=2),
        nn.Linear(self.grid_size.y, self.out_bins_size.y),
    )
    self.fc_y = nn.Linear(neck_dim, num_joints)
    # Depthwise has no joint queries; role conditioning is FiLM-only.
    self.role_conditioner = (
        RoleConditioner(neck_dim, use_film=True) if use_role and role_film else None
    )

  def forward(self, patch_tokens: torch.Tensor, role: torch.Tensor | None = None, **kwargs):
    B, _, C = patch_tokens.shape
    x_2d = patch_tokens.permute(0, 2, 1).reshape(
        B, C, self.grid_size.y, self.grid_size.x
    )
    feat = self.neck(x_2d)
    if self.role_conditioner is not None:
      role = self.role_conditioner.resolve_role(role, B, feat.device, feat.dtype)
      feat = self.role_conditioner.film_features(feat, role)

    # Note: transpose(1, 2) is cleaner than permute(0, 2, 1)
    feat_x = self.mlp_x(feat).transpose(1, 2)
    pred_x = self.fc_x(feat_x).transpose(1, 2)

    feat_y = self.mlp_y(feat).transpose(1, 2)
    pred_y = self.fc_y(feat_y).transpose(1, 2)

    return pred_x, pred_y


class _JointQueryDecoderLayer(nn.Module):
  """Cross-attention over patch memory only (no query–query mixing)."""

  def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    self.norm_q = nn.LayerNorm(dim)
    self.norm_kv = nn.LayerNorm(dim)
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_ffn = nn.LayerNorm(dim)
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )

  def forward(
      self,
      queries: torch.Tensor,
      memory: torch.Tensor,
      cond: torch.Tensor | None = None,
      self_attn_bias: torch.Tensor | None = None,
      cross_attn_bias: torch.Tensor | None = None,
  ) -> torch.Tensor:
    q_base = self.norm_q(queries)
    q = q_base + cond if cond is not None else q_base
    kv = self.norm_kv(memory)
    attn_out, _ = self.cross_attn(
        q, kv, kv, attn_mask=cross_attn_bias, need_weights=False
    )
    queries = queries + attn_out
    queries = queries + self.ffn(self.norm_ffn(queries))
    return queries


class _JointQuerySelfAttnDecoderLayer(nn.Module):
  """DETR-style layer: self-attn among joint queries, then cross-attn to patches."""

  def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
    super().__init__()
    self.norm_self = nn.LayerNorm(dim)
    self.self_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_q = nn.LayerNorm(dim)
    self.norm_kv = nn.LayerNorm(dim)
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.norm_ffn = nn.LayerNorm(dim)
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )

  def forward(
      self,
      queries: torch.Tensor,
      memory: torch.Tensor,
      cond: torch.Tensor | None = None,
      self_attn_bias: torch.Tensor | None = None,
      cross_attn_bias: torch.Tensor | None = None,
  ) -> torch.Tensor:
    q_content = self.norm_self(queries)
    if cond is not None:
      q_self = q_content + cond
      k_self = q_content + cond
      self_out, _ = self.self_attn(
          q_self, k_self, q_content, attn_mask=self_attn_bias, need_weights=False
      )
    else:
      self_out, _ = self.self_attn(
          q_content,
          q_content,
          q_content,
          attn_mask=self_attn_bias,
          need_weights=False,
      )
    queries = queries + self_out

    q_base = self.norm_q(queries)
    q = q_base + cond if cond is not None else q_base
    kv = self.norm_kv(memory)
    cross_out, _ = self.cross_attn(
        q, kv, kv, attn_mask=cross_attn_bias, need_weights=False
    )
    queries = queries + cross_out
    queries = queries + self.ffn(self.norm_ffn(queries))
    return queries


def build_lr_shared_pair_bias_index(
    num_joints: int,
    bone_pairs: list[tuple[int, int]],
    left_right_pairs: list[tuple[int, int]],
) -> torch.Tensor:
  """Build [K, K] long index into shared bias params (−1 = unused).

  Undirected bone edges share a type. Left and right mirror edges share one type
  so L/R structure is tied (helps side swaps).
  """
  mirror = list(range(num_joints))
  for left_i, right_i in left_right_pairs:
    mirror[left_i] = right_i
    mirror[right_i] = left_i

  def undirected(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i <= j else (j, i)

  def canonical(i: int, j: int) -> tuple[int, int]:
    e1 = undirected(i, j)
    e2 = undirected(mirror[i], mirror[j])
    return e1 if e1 <= e2 else e2

  type_ids: dict[tuple[int, int], int] = {}
  for parent, child in bone_pairs:
    if not (0 <= parent < num_joints and 0 <= child < num_joints):
      raise ValueError(f'bone pair ({parent}, {child}) out of range for K={num_joints}')
    key = canonical(parent, child)
    if key not in type_ids:
      type_ids[key] = len(type_ids)

  index = torch.full((num_joints, num_joints), -1, dtype=torch.long)
  for parent, child in bone_pairs:
    tid = type_ids[canonical(parent, child)]
    index[parent, child] = tid
    index[child, parent] = tid
  return index


class _BaseJointQuerySimCCHead(BaseSimCCHead):
  """Shared joint-query → SimCC head; subclasses set ``_layer_cls``.

  Optional soft foot anchoring (``soft_foot_anchor=True``):
      q_foot = q_foot_free + σ(α) * W(q_ankle)
  before the decoder. Feet keep a free query; the ankle term is a soft prior.
  Self-attn remains bidirectional afterward. ``foot_anchor_pairs`` is a list of
  ``(foot_local_id, ankle_local_id)``.
  """

  _layer_cls = _JointQueryDecoderLayer

  def __init__(
      self,
      in_channels: int,
      num_joints: int,
      grid_size: IVec2,
      out_size: IVec2,
      split_ratio: float = 2.0,
      neck_dim: int = 256,
      num_heads: int = 8,
      num_layers: int = 2,
      dropout: float = 0.0,
      soft_foot_anchor: bool = False,
      soft_foot_anchor_init: float = 0.4,
      foot_anchor_pairs: list[tuple[int, int]] | None = None,
      use_role: bool = False,
      role_film: bool = True,
      per_layer_role_reinject: bool = False,
      use_spatial_prompt: bool = False,
      deep_supervision: bool = False,
      use_joint_pos_embed: bool = False,
      use_pair_bias: bool = False,
      pair_bias_bones: list[tuple[int, int]] | None = None,
      pair_bias_lr_pairs: list[tuple[int, int]] | None = None,
      pair_bias_init: float = 2.0,
      use_local_cross_attn: bool = False,
      local_attn_sigma: float = 2.0,
      local_attn_strength: float = 2.0,
      **kwargs,
  ):
    super().__init__()
    if neck_dim % num_heads != 0:
      raise ValueError(f'neck_dim ({neck_dim}) must be divisible by num_heads ({num_heads})')
    if soft_foot_anchor and not 0.0 < soft_foot_anchor_init < 1.0:
      raise ValueError(
          f'soft_foot_anchor_init must be in (0, 1), got {soft_foot_anchor_init}'
      )
    if use_local_cross_attn and local_attn_sigma <= 0:
      raise ValueError(f'local_attn_sigma must be > 0, got {local_attn_sigma}')

    self.grid_size = grid_size
    self.neck_dim = neck_dim
    self.num_heads = num_heads
    self.num_joints = num_joints
    self.soft_foot_anchor = soft_foot_anchor
    self.use_role = use_role
    self.role_film = role_film
    self.per_layer_role_reinject = per_layer_role_reinject
    self.use_spatial_prompt = use_spatial_prompt
    self.deep_supervision = deep_supervision
    self.use_joint_pos_embed = use_joint_pos_embed
    self.use_pair_bias = use_pair_bias
    self.use_local_cross_attn = use_local_cross_attn
    self.local_attn_sigma = float(local_attn_sigma)
    self.local_attn_strength = float(local_attn_strength)
    self.split_ratio = float(split_ratio)
    self.out_size = out_size
    self.out_bins_size = IVec2(
        int(out_size.x * split_ratio), int(out_size.y * split_ratio)
    )

    self.input_proj = nn.Linear(in_channels, neck_dim)
    self.joint_queries = nn.Parameter(torch.randn(num_joints, neck_dim) * 0.02)
    self.patch_pos_embed = nn.Parameter(
        torch.randn(1, grid_size.x * grid_size.y, neck_dim) * 0.02
    )
    self.layers = nn.ModuleList(
        [self._layer_cls(neck_dim, num_heads, dropout) for _ in range(num_layers)]
    )
    self.fc_x = nn.Linear(neck_dim, self.out_bins_size.x)
    self.fc_y = nn.Linear(neck_dim, self.out_bins_size.y)
    self.role_conditioner = (
        RoleConditioner(neck_dim, use_film=role_film) if use_role else None
    )
    if per_layer_role_reinject:
      self.joint_id_embed = nn.Embedding(num_joints, neck_dim)
      nn.init.normal_(self.joint_id_embed.weight, std=0.02)
    else:
      self.joint_id_embed = None
    if use_spatial_prompt:
      self.spatial_mlp = nn.Sequential(
          nn.Linear(2, neck_dim),
          nn.GELU(),
          nn.Linear(neck_dim, neck_dim),
      )
    else:
      self.spatial_mlp = None
    if use_joint_pos_embed:
      # Map crop-normalized (x, y) ∈ [0, 1]^2 → query residual.
      self.joint_pos_mlp = nn.Sequential(
          nn.Linear(2, neck_dim),
          nn.GELU(),
          nn.Linear(neck_dim, neck_dim),
      )
      nn.init.zeros_(self.joint_pos_mlp[-1].weight)
      nn.init.zeros_(self.joint_pos_mlp[-1].bias)
    else:
      self.joint_pos_mlp = None
    if use_pair_bias:
      if not pair_bias_bones:
        raise ValueError('use_pair_bias=True requires non-empty pair_bias_bones')
      lr_pairs = list(pair_bias_lr_pairs or [])
      bias_index = build_lr_shared_pair_bias_index(
          num_joints, list(pair_bias_bones), lr_pairs
      )
      num_types = int(bias_index.max().item()) + 1
      if num_types <= 0:
        raise ValueError('use_pair_bias=True produced zero shared edge types')
      self.register_buffer('pair_bias_index', bias_index, persistent=True)
      self.pair_bias_values = nn.Parameter(
          torch.full((num_types,), float(pair_bias_init), dtype=torch.float32)
      )
    else:
      self.pair_bias_index = None
      self.pair_bias_values = None

    if soft_foot_anchor:
      if not foot_anchor_pairs:
        raise ValueError('soft_foot_anchor=True requires non-empty foot_anchor_pairs')
      foot_ids, ankle_ids = zip(*foot_anchor_pairs)
      for foot_i, ankle_i in foot_anchor_pairs:
        if not (0 <= foot_i < num_joints and 0 <= ankle_i < num_joints):
          raise ValueError(
              f'foot_anchor pair ({foot_i}, {ankle_i}) out of range for '
              f'num_joints={num_joints}'
          )
      self.register_buffer(
          'foot_anchor_foot_ids', torch.tensor(foot_ids, dtype=torch.long), persistent=True
      )
      self.register_buffer(
          'foot_anchor_ankle_ids', torch.tensor(ankle_ids, dtype=torch.long), persistent=True
      )
      self.ankle_to_foot = nn.Linear(neck_dim, neck_dim, bias=False)
      nn.init.eye_(self.ankle_to_foot.weight)
      # σ(logit) starts at soft_foot_anchor_init; shared gate keeps param count tiny
      init = float(soft_foot_anchor_init)
      logit = math.log(init / (1.0 - init))
      self.foot_anchor_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
    else:
      self.foot_anchor_foot_ids = None
      self.foot_anchor_ankle_ids = None
      self.ankle_to_foot = None
      self.foot_anchor_logit = None

  def _build_queries(self, batch_size: int) -> torch.Tensor:
    queries = self.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)
    if not self.soft_foot_anchor:
      return queries

    # Clone so we can write foot slots without touching the shared expand view
    queries = queries.clone()
    foot = queries[:, self.foot_anchor_foot_ids]
    ankle = queries[:, self.foot_anchor_ankle_ids]
    gate = torch.sigmoid(self.foot_anchor_logit)
    queries[:, self.foot_anchor_foot_ids] = foot + gate * self.ankle_to_foot(ankle)
    return queries

  def _build_layer_cond(
      self,
      batch_size: int,
      role: torch.Tensor | None,
      spatial: torch.Tensor | None,
      device,
      dtype,
  ) -> torch.Tensor | None:
    if not self.per_layer_role_reinject and not self.use_spatial_prompt:
      return None
    cond = torch.zeros(
        batch_size, self.num_joints, self.neck_dim, device=device, dtype=dtype
    )
    if self.per_layer_role_reinject and self.joint_id_embed is not None:
      joint_ids = torch.arange(self.num_joints, device=device, dtype=torch.long)
      cond = cond + self.joint_id_embed(joint_ids).unsqueeze(0)
      if self.use_role and self.role_conditioner is not None:
        role = self.role_conditioner.resolve_role(role, batch_size, device, dtype)
        cond = cond + self.role_conditioner.query_embed(role).unsqueeze(1)
    if self.use_spatial_prompt and self.spatial_mlp is not None:
      if spatial is None:
        spatial = torch.zeros(batch_size, 2, device=device, dtype=dtype)
      else:
        spatial = spatial.to(device=device, dtype=dtype)
      cond = cond + self.spatial_mlp(spatial).unsqueeze(1)
    return cond

  def _materialize_pair_bias(self, dtype, device) -> torch.Tensor | None:
    if not self.use_pair_bias or self.pair_bias_values is None or self.pair_bias_index is None:
      return None
    # Additive attention mask: skeleton edges start positive so neighbors attend first.
    bias = torch.zeros(
        self.num_joints, self.num_joints, device=device, dtype=dtype
    )
    valid = self.pair_bias_index >= 0
    bias[valid] = self.pair_bias_values[self.pair_bias_index[valid]].to(dtype=dtype)
    return bias

  def _soft_decode_normed(
      self, pred_x: torch.Tensor, pred_y: torch.Tensor
  ) -> torch.Tensor:
    """Soft-argmax SimCC → crop-normalized [B, K, 2] in [0, 1]."""
    px = F.softmax(pred_x.float(), dim=-1)
    py = F.softmax(pred_y.float(), dim=-1)
    bins_x = torch.arange(pred_x.shape[-1], device=pred_x.device, dtype=px.dtype)
    bins_y = torch.arange(pred_y.shape[-1], device=pred_y.device, dtype=py.dtype)
    x = (px * bins_x).sum(dim=-1) / self.split_ratio / max(self.out_size.x, 1)
    y = (py * bins_y).sum(dim=-1) / self.split_ratio / max(self.out_size.y, 1)
    return torch.stack((x, y), dim=-1).clamp(0.0, 1.0)

  def _gaussian_cross_attn_bias(
      self, normed_xy: torch.Tensor, batch_size: int, dtype, device
  ) -> torch.Tensor:
    """Additive cross-attn bias [B*heads, K, N] peaked at each joint's patch.

    ``normed_xy`` is crop-normalized [B, K, 2] (x, y) in [0, 1]. Distances are
    in patch-grid units; ``local_attn_sigma`` is the Gaussian width in patches.
    """
    gw = self.grid_size.x
    gh = self.grid_size.y
    xs = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw
    ys = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh
    # meshgrid ij → (y, x); stack as (x, y) to match soft-decode order
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    patch_xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)  # [N, 2]
    scale = torch.tensor([gw, gh], device=device, dtype=dtype)
    diff = normed_xy.to(dtype=dtype).unsqueeze(2) - patch_xy.view(1, 1, -1, 2)
    dist_sq = (diff * scale).pow(2).sum(dim=-1)  # [B, K, N]
    sigma = self.local_attn_sigma
    bias = self.local_attn_strength * torch.exp(-dist_sq / (2.0 * sigma * sigma))
    # MHA float attn_mask expects [B * num_heads, tgt_len, src_len]
    return (
        bias.unsqueeze(1)
        .expand(batch_size, self.num_heads, self.num_joints, -1)
        .reshape(batch_size * self.num_heads, self.num_joints, -1)
    )

  def forward(
      self,
      patch_tokens: torch.Tensor,
      role: torch.Tensor | None = None,
      spatial: torch.Tensor | None = None,
  ):
    B, N, _ = patch_tokens.shape
    expected_n = self.grid_size.x * self.grid_size.y
    if N != expected_n:
      raise ValueError(
          f'Expected {expected_n} patch tokens for grid {self.grid_size.x}x{self.grid_size.y}, got {N}'
      )

    memory = self.input_proj(patch_tokens) + self.patch_pos_embed
    queries = self._build_queries(B)
    if self.role_conditioner is not None:
      role = self.role_conditioner.resolve_role(role, B, memory.device, memory.dtype)
      queries = self.role_conditioner.bias_queries(queries, role)
      if self.role_conditioner.use_film:
        memory = self.role_conditioner.film_features(memory, role)
    cond = self._build_layer_cond(B, role, spatial, memory.device, memory.dtype)
    self_attn_bias = self._materialize_pair_bias(memory.dtype, memory.device)
    aux_preds: list[tuple[torch.Tensor, torch.Tensor]] = []
    prev_pred_x = prev_pred_y = None
    need_layer_preds = (
        self.deep_supervision or self.use_joint_pos_embed or self.use_local_cross_attn
    )
    for layer in self.layers:
      normed = None
      if prev_pred_x is not None and prev_pred_y is not None:
        # Detach so early layers get gradients from deep-sup, not through feedback.
        normed = self._soft_decode_normed(prev_pred_x.detach(), prev_pred_y.detach())
      if (
          self.use_joint_pos_embed
          and self.joint_pos_mlp is not None
          and normed is not None
      ):
        queries = queries + self.joint_pos_mlp(normed.to(dtype=queries.dtype))
      cross_attn_bias = None
      if self.use_local_cross_attn and normed is not None:
        cross_attn_bias = self._gaussian_cross_attn_bias(
            normed, B, memory.dtype, memory.device
        )
      queries = layer(
          queries,
          memory,
          cond,
          self_attn_bias=self_attn_bias,
          cross_attn_bias=cross_attn_bias,
      )
      if need_layer_preds:
        pred_x_i = self.fc_x(queries)
        pred_y_i = self.fc_y(queries)
        prev_pred_x, prev_pred_y = pred_x_i, pred_y_i
        if self.deep_supervision:
          aux_preds.append((pred_x_i, pred_y_i))

    if self.deep_supervision:
      pred_x, pred_y = aux_preds[-1]
      early_aux = aux_preds[:-1] if len(aux_preds) > 1 else []
      return pred_x, pred_y, early_aux

    pred_x = self.fc_x(queries)
    pred_y = self.fc_y(queries)
    return pred_x, pred_y


class JointQuerySimCCHead(_BaseJointQuerySimCCHead):
  """Learnable joint queries with cross-attention over patch tokens, then SimCC X/Y heads."""

  _layer_cls = _JointQueryDecoderLayer


class JointQuerySelfAttnSimCCHead(_BaseJointQuerySimCCHead):
  """Joint queries with self-attention among joints, then cross-attention to patches + SimCC."""

  _layer_cls = _JointQuerySelfAttnDecoderLayer


class CoupleJointQuerySelfAttnSimCCHead(_BaseJointQuerySimCCHead):
  """Paired lead|follow joint-query head (2K queries) with person-bank embeddings.

  Query layout: ``[0:K)`` = lead, ``[K:2K)`` = follow. Self-attn across both banks
  can resolve limb ownership on overlapping union crops. Role conditioning is
  disabled — bank identity replaces ``[is_lead, is_follow]`` bits.
  """

  _layer_cls = _JointQuerySelfAttnDecoderLayer

  def __init__(
      self,
      in_channels: int,
      num_joints: int,
      grid_size: IVec2,
      out_size: IVec2,
      split_ratio: float = 2.0,
      neck_dim: int = 256,
      num_heads: int = 8,
      num_layers: int = 2,
      dropout: float = 0.0,
      soft_foot_anchor: bool = False,
      soft_foot_anchor_init: float = 0.4,
      foot_anchor_pairs: list[tuple[int, int]] | None = None,
      person_joints: int | None = None,
      use_role: bool = False,
      role_film: bool = True,
      **kwargs,
  ):
    if num_joints % 2 != 0:
      raise ValueError(
          f'Couple head expects even num_joints (2 * person_joints), got {num_joints}'
      )
    resolved_person_joints = (
        int(person_joints) if person_joints is not None else num_joints // 2
    )
    if num_joints != 2 * resolved_person_joints:
      raise ValueError(
          f'num_joints ({num_joints}) must equal 2 * person_joints ({resolved_person_joints})'
      )
    # Bank embeddings carry identity; never enable role FiLM/bias on this head.
    super().__init__(
        in_channels=in_channels,
        num_joints=num_joints,
        grid_size=grid_size,
        out_size=out_size,
        split_ratio=split_ratio,
        neck_dim=neck_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        soft_foot_anchor=soft_foot_anchor,
        soft_foot_anchor_init=soft_foot_anchor_init,
        foot_anchor_pairs=foot_anchor_pairs,
        use_role=False,
        role_film=False,
        **kwargs,
    )
    self.person_joints = resolved_person_joints
    self.lead_embed = nn.Parameter(torch.randn(1, 1, neck_dim) * 0.02)
    self.follow_embed = nn.Parameter(torch.randn(1, 1, neck_dim) * 0.02)

  def _build_queries(self, batch_size: int) -> torch.Tensor:
    queries = super()._build_queries(batch_size)
    if not self.soft_foot_anchor:
      queries = queries.clone()
    k = self.person_joints
    queries[:, :k] = queries[:, :k] + self.lead_embed
    queries[:, k:] = queries[:, k:] + self.follow_embed
    return queries


HEAD_REGISTRY = {
    'depthwise_simcc': DepthwiseSimCCHead,
    'joint_query_simcc': JointQuerySimCCHead,
    'joint_query_self_attn_simcc': JointQuerySelfAttnSimCCHead,
    'couple_joint_query_self_attn_simcc': CoupleJointQuerySelfAttnSimCCHead,
}


class DinoCC(nn.Module):
    def __init__(self, num_joints: int, img_size: IVec2, freeze_backbone: bool = True,
                 split_ratio: float = 2.0, neck_dim: int = 256,
                 backbone_name: str = 'facebook/dinov2-base',
                 head_name: str = 'depthwise_simcc',
                 head_kwargs: dict | None = None):
        super().__init__()
        self.num_joints = num_joints
        self.img_size = img_size
        self.backbone_name = backbone_name
        self.head_name = head_name

        # Load ViT backbone (Hub id, local weights dir, or config-only dir)
        self.backbone = load_backbone(backbone_name)
        self.patch_size = self.backbone.config.patch_size
        self.num_register_tokens = getattr(self.backbone.config, 'num_register_tokens', 0) or 0

        if self.img_size.x % self.patch_size != 0 or self.img_size.y % self.patch_size != 0:
            raise ValueError(
                f'Image size {self.img_size.x}x{self.img_size.y} must be divisible by '
                f'backbone patch size {self.patch_size}'
            )

        self.grid_size = IVec2(self.img_size.x // self.patch_size, self.img_size.y // self.patch_size)

        # Freeze backbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if head_name not in HEAD_REGISTRY:
            raise ValueError(
                f'Unknown head_name={head_name!r}; expected one of {sorted(HEAD_REGISTRY)}'
            )
        head_cls = HEAD_REGISTRY[head_name]
        kwargs = {
            'in_channels': self.backbone.config.hidden_size,
            'num_joints': self.num_joints,
            'grid_size': self.grid_size,
            'out_size': IVec2(self.img_size.x, self.img_size.y),
            'split_ratio': split_ratio,
            'neck_dim': neck_dim,
        }
        if head_kwargs:
            kwargs.update(head_kwargs)
        self.head = head_cls(**kwargs)

    def load_weights(self, checkpoint_path, map_location='cpu', strict=True):
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.load_state_dict(state_dict, strict=strict)
        return checkpoint


    def forward(self, pixel_values, role=None, spatial=None):
        # DINOv2 needs pos-encoding interpolation for non-native resolutions; DINOv3 uses RoPE
        if getattr(self.backbone.config, 'model_type', None) == 'dinov2':
            outputs = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        # Token 0 is [CLS]; DINOv3 (and some DINOv2 variants) insert register tokens before patches
        patch_start = 1 + self.num_register_tokens
        patch_tokens = outputs.last_hidden_state[:, patch_start:, :]  # [B, N, C]

        head_out = self.head(patch_tokens, role=role, spatial=spatial)
        if isinstance(head_out, tuple) and len(head_out) == 3:
            return head_out
        pred_x, pred_y = head_out
        return pred_x, pred_y


def generate_simcc_labels(
    keypoints,
    img_size: IVec2,
    split_ratio=2.0,
    sigma=6.0,
    sigma_occluded: float | None = None,
    occluded_weight: float = 0.35,
):
    """
    Args:
        keypoints: Array-like of shape [K, 3] with rows: [x, y, visibility]
                   Coordinates must be relative to the cropped 336x448 image.
        img_size: IVec2(x=336, y=448)
        split_ratio: Sub-pixel scaling factor (2.0 = 0.5px bins)
        sigma: Gaussian std (bin units) for visible joints (v==2)
        sigma_occluded: Gaussian std for occluded joints (v==1). None → same as sigma.
            Wider values soften the target so near-misses cost less while wild
            misses still get KL pressure via the tails.
        occluded_weight: Loss weight for occluded joints (v==1); visible use 1.0.
    Returns:
        target_x: [K, out_bins_x]
        target_y: [K, out_bins_y]
        weights:  [K] Loss weight per joint based on visibility
    """
    num_joints = len(keypoints)
    bins_x = int(img_size.x * split_ratio)
    bins_y = int(img_size.y * split_ratio)
    sigma = float(sigma)
    sigma_occ = float(sigma_occluded) if sigma_occluded is not None else sigma
    occ_w = float(occluded_weight)
    if sigma <= 0 or sigma_occ <= 0:
        raise ValueError(
            f'sigma and sigma_occluded must be > 0, got sigma={sigma}, '
            f'sigma_occluded={sigma_occ}'
        )
    if not 0.0 < occ_w <= 1.0:
        raise ValueError(f'occluded_weight must be in (0, 1], got {occ_w}')

    target_x = np.zeros((num_joints, bins_x), dtype=np.float32)
    target_y = np.zeros((num_joints, bins_y), dtype=np.float32)
    weights = np.zeros((num_joints,), dtype=np.float32)

    x_bins = np.arange(bins_x)
    y_bins = np.arange(bins_y)

    for k, (x, y, v) in enumerate(keypoints):
        # Skip joints marked outside frame or invalid
        if v == 0 or x < 0 or y < 0 or x >= img_size.x or y >= img_size.y:
            continue

        # Scale continuous pixels to sub-pixel bin centers
        mu_x = x * split_ratio
        mu_y = y * split_ratio
        joint_sigma = sigma if v == 2 else sigma_occ

        # Continuous 1D Gaussian distributions
        gx = np.exp(-((x_bins - mu_x) ** 2) / (2.0 * joint_sigma ** 2))
        gy = np.exp(-((y_bins - mu_y) ** 2) / (2.0 * joint_sigma ** 2))

        # Normalize to valid probability density
        target_x[k] = gx / (gx.sum() + 1e-8)
        target_y[k] = gy / (gy.sum() + 1e-8)

        # Visibility weighting: full weight if visible, reduced if occluded
        weights[k] = 1.0 if v == 2 else occ_w

    return (
        torch.from_numpy(target_x),
        torch.from_numpy(target_y),
        torch.from_numpy(weights)
    )


class SimCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # batchmean computes the mathematically rigorous KL-divergence over batches
        self.kl = nn.KLDivLoss(reduction='none')

    def forward(self, pred_x, pred_y, target_x, target_y, weights):
        """
        pred_x:   [B, K, bins_x] (raw logits)
        pred_y:   [B, K, bins_y] (raw logits)
        target_x: [B, K, bins_x] (normalized probability density)
        target_y: [B, K, bins_y] (normalized probability density)
        weights:  [B, K]
        """
        # Convert logits to log-probabilities
        log_pred_x = F.log_softmax(pred_x, dim=-1)
        log_pred_y = F.log_softmax(pred_y, dim=-1)

        # Calculate KL divergence per joint: sum across the bin dimension
        loss_x = self.kl(log_pred_x, target_x).sum(dim=-1) # [B, K]
        loss_y = self.kl(log_pred_y, target_y).sum(dim=-1) # [B, K]

        # Weight by joint visibility
        loss = (loss_x + loss_y) * weights # [B, K]

        # Normalize across non-zero weighted joints
        valid_joints = weights.sum() + 1e-6
        return loss.sum() / valid_joints


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on: {device}")

    num_joints = 19
    img_size = IVec2(336, 448) # width=336, height=448

    processor = AutoImageProcessor.from_pretrained(
        'facebook/dinov2-base', 
        size={"height": 448, "width": 336},
        crop_size={"height": 448, "width": 336},
        do_center_crop=False
    )
    
    # 1. Instantiate model and move to GPU
    model = DinoCC(num_joints=num_joints, img_size=img_size, freeze_backbone=True).to(device)
    criterion = SimCCLoss().to(device)

    # 2. Setup AdamW optimizer only over the head & neck parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    # 3. Load sample image and prepare input batch
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device) # Shape: [1, 3, 448, 336]

    # 4. Generate dummy target keypoints [K, 3] -> (x, y, visibility)
    # Simulating 19 keypoints within the 336x448 crop
    dummy_kpts = np.zeros((num_joints, 3), dtype=np.float32)
    for i in range(num_joints):
        dummy_kpts[i] = [50.0 + i * 12.0, 100.0 + i * 15.0, 2 if i % 4 != 0 else 1]

    # Convert coordinates to SimCC 1D Gaussian targets
    target_x, target_y, weights = generate_simcc_labels(dummy_kpts, img_size=img_size, split_ratio=2.0)
    
    # Add batch dimension and send to device: [1, K, bins]
    target_x = target_x.unsqueeze(0).to(device)
    target_y = target_y.unsqueeze(0).to(device)
    weights = weights.unsqueeze(0).to(device)

    # 5. Run Training Loop & Backprop
    model.train()
    print("\nStarting verification backpropagation loop...")
    for step in range(1, 26):
        optimizer.zero_grad()

        # Forward pass
        pred_x, pred_y = model(pixel_values)

        # Compute KL Divergence loss
        loss = criterion(pred_x, pred_y, target_x, target_y, weights)

        # Backpropagate and update gradients
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d} | Loss: {loss.item():.5f}")

    # 6. Verify predicted coordinates after gradient descent
    model.eval()
    with torch.no_grad():
        p_x, p_y = model(pixel_values)
        pred_coord_x = torch.argmax(p_x, dim=-1).float() / 2.0
        pred_coord_y = torch.argmax(p_y, dim=-1).float() / 2.0

    print("\n--- Target vs Learned Prediction Check (Joint 0) ---")
    print(f"Ground Truth : ({dummy_kpts[0][0]:.1f}, {dummy_kpts[0][1]:.1f})")
    print(f"Model Learned: ({pred_coord_x[0, 0].item():.1f}, {pred_coord_y[0, 0].item():.1f})")








"""
Input image
    [B, 3, 448, 336]  (height, width)
            |
            v
DINOv2 ViT (14 x 14 patches)
    patch grid: 32 x 24 (grid_y x grid_x)
    last_hidden_state: [B, 769, 768]
            |
            v
Remove CLS token
    patch_tokens: [B, 768, 768]
            |
            v
Reshape tokens to spatial feature map
    [B, 768, 32, 24]  (channels, grid_y, grid_x)
            |
            v
SimCCPoseHead
    Neck: 1x1 projection -> GroupNorm/GELU
          -> depthwise 3x3 -> 1x1 -> GroupNorm/GELU
    features: [B, 256, 32, 24]
            |
       +----+----+
       |         |
       v         v
    X branch  Y branch
    pool H    pool W
    Linear    Linear
       |         |
       v         v
    pred_x    pred_y
 [B, K, 672] [B, K, 896]
   (x bins)    (y bins)
"""