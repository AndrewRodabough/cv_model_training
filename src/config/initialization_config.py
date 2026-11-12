from dataclasses import dataclass, field


@dataclass
class PairBiasEdgeConfig:
    """One undirected self-attn pair-bias edge (or L/R stem that expands to both sides)."""

    joints: list[str]
    # Applied only when constructing from noise (ignored if a checkpoint is loaded).
    init: float | None = None


@dataclass
class PairBiasInitConfig:
    """Configurable pair-bias graph + per-edge init (noise init only)."""

    # Used when an edge omits ``init``.
    default_init: float = 2.0
    edges: list[PairBiasEdgeConfig] = field(default_factory=list)


@dataclass
class InitializationConfig:
    """Weight source: checkpoint XOR noise inits (pair-bias edges + init values).

    If ``checkpoint`` is set (or ``--init-checkpoint`` is passed), model weights
    come from that file and pair-bias ``init`` values are ignored. Otherwise the
    model is built from noise and pair-bias parameters use the configured inits.
    """

    checkpoint: str | None = None
    pair_bias: PairBiasInitConfig | None = None

    @classmethod
    def from_raw(cls, raw: dict | None) -> 'InitializationConfig':
        if not raw:
            return cls()
        data = dict(raw)
        pair_raw = data.pop('pair_bias', None)
        pair_cfg = None
        if pair_raw is not None:
            edges_raw = list(pair_raw.get('edges') or [])
            edges = []
            for edge in edges_raw:
                if not isinstance(edge, dict):
                    raise ValueError(
                        f'initialization.pair_bias.edges entries must be mappings, got {edge!r}'
                    )
                joints = edge.get('joints')
                if not isinstance(joints, (list, tuple)) or len(joints) != 2:
                    raise ValueError(
                        'initialization.pair_bias.edges[].joints must be a length-2 list '
                        f'(got {joints!r})'
                    )
                edges.append(
                    PairBiasEdgeConfig(
                        joints=[str(joints[0]), str(joints[1])],
                        init=None if edge.get('init') is None else float(edge['init']),
                    )
                )
            pair_cfg = PairBiasInitConfig(
                default_init=float(pair_raw.get('default_init', 2.0)),
                edges=edges,
            )
        checkpoint = data.get('checkpoint')
        if checkpoint is not None:
            checkpoint = str(checkpoint)
            if not checkpoint.strip():
                checkpoint = None
        unknown = set(data) - {'checkpoint'}
        if unknown:
            raise ValueError(f'Unknown initialization keys: {sorted(unknown)}')
        return cls(checkpoint=checkpoint, pair_bias=pair_cfg)
