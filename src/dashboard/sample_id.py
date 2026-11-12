"""Person-aware sample labels for the dashboard (stem + lead/follow)."""

from __future__ import annotations


def sample_choice_label(stem: str, role: str, index: int) -> str:
    """Unique dropdown id: same image can have lead + follow samples."""
    stem = str(stem)
    role = (role or '').strip()
    if role:
        return f'{stem} [{role}]'
    return f'{stem} #{index}'


def parse_sample_choice(label: str) -> tuple[str, str | None, int | None]:
    """Return ``(stem, role|None, index|None)`` from a dropdown or attn-cache label.

    Accepts:
      - ``task_53_frame_000385 [lead]``
      - ``task_53_frame_000385 #12``
      - ``task_53_frame_000385__lead`` (attn cache key)
      - ``task_53_frame_000385`` (plain stem)
    """
    text = str(label).strip()
    if ' [' in text and text.endswith(']'):
        stem, _, rest = text.partition(' [')
        return stem, rest[:-1].strip(), None
    if ' #' in text:
        stem, _, rest = text.rpartition(' #')
        try:
            return stem, None, int(rest)
        except ValueError:
            return text, None, None
    # Attn cache key: stem__lead / stem__follow / stem__p3
    if '__' in text:
        stem, _, suffix = text.rpartition('__')
        if suffix in {'lead', 'follow'} or (
            suffix.startswith('p') and suffix[1:].isdigit()
        ):
            if suffix.startswith('p'):
                return stem, None, int(suffix[1:])
            return stem, suffix, None
    return text, None, None


def attn_cache_key(label: str) -> str:
    """Filesystem-safe key for attn npz / ondemand jpg (includes role)."""
    stem, role, index = parse_sample_choice(label)
    if role:
        return f'{stem}__{role}'
    if index is not None:
        return f'{stem}__p{index}'
    return stem


def find_sample_index(data: dict, label: str) -> int | None:
    """Resolve dropdown/cache label to a row in val_preds (person-aware)."""
    stem, role, index_hint = parse_sample_choice(label)
    stems = [str(s) for s in data['image_stem']]
    roles = data.get('role')

    def _role_at(i: int) -> str:
        if roles is None:
            return ''
        return str(roles[i]).strip()

    if index_hint is not None:
        for i, s in enumerate(stems):
            if s == stem and i == index_hint:
                return i
        matches = [i for i, s in enumerate(stems) if s == stem]
        if 0 <= index_hint < len(matches):
            return matches[index_hint]
        return matches[0] if matches else None
    if role is not None:
        for i, s in enumerate(stems):
            if s == stem and _role_at(i) == role:
                return i
        return None
    try:
        return stems.index(stem)
    except ValueError:
        return None
