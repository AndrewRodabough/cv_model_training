"""Structured run-folder artifacts for live dashboards and post-hoc inspection."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

HISTORY_NAME = 'history.jsonl'
STATUS_NAME = 'status.json'
CONTROL_NAME = 'control.json'
VAL_PREDS_DIR = 'val_preds'
VAL_OVERLAYS_DIR = 'val_overlays'
VAL_SAMPLES_NAME = 'val_samples.json'
# Cross-attn maps for joint-query heads (under best/attn and final/attn).
ATTN_DIR_NAME = 'attn'

VALID_STATES = frozenset({'running', 'paused', 'stopping', 'done', 'error'})
VALID_COMMANDS = frozenset({'pause', 'resume', 'stop', 'none'})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + '\n', encoding='utf-8')
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def read_json(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None


def write_status(
    run_dir: Path,
    state: str,
    *,
    epoch: int | None = None,
    message: str = '',
    pid: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if state not in VALID_STATES:
        raise ValueError(f'invalid status state {state!r}')
    payload: dict[str, Any] = {
        'state': state,
        'epoch': epoch,
        'pid': os.getpid() if pid is None else pid,
        'message': message,
        'updated_at': _utc_now(),
    }
    if extra:
        payload.update(extra)
    write_json(Path(run_dir) / STATUS_NAME, payload)


def read_status(run_dir: Path) -> dict[str, Any] | None:
    return read_json(Path(run_dir) / STATUS_NAME)


def write_control(run_dir: Path, command: str, *, note: str = '') -> None:
    command = (command or 'none').strip().lower()
    if command not in VALID_COMMANDS:
        raise ValueError(f'invalid control command {command!r}')
    write_json(
        Path(run_dir) / CONTROL_NAME,
        {
            'command': command,
            'note': note,
            'updated_at': _utc_now(),
        },
    )


def read_control(run_dir: Path) -> str:
    payload = read_json(Path(run_dir) / CONTROL_NAME)
    if not payload:
        return 'none'
    command = str(payload.get('command', 'none')).strip().lower()
    return command if command in VALID_COMMANDS else 'none'


def clear_control(run_dir: Path) -> None:
    write_control(run_dir, 'none')


def append_history(run_dir: Path, record: dict[str, Any]) -> None:
    path = Path(run_dir) / HISTORY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(record)
    row.setdefault('timestamp', _utc_now())
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, default=_json_default) + '\n')


def read_history(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / HISTORY_NAME
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def val_preds_path(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / VAL_PREDS_DIR / f'epoch_{epoch:03d}.npz'


def val_overlays_dir(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / VAL_OVERLAYS_DIR / f'epoch_{epoch:03d}'


def attn_dir(run_dir: Path, checkpoint_name: str) -> Path:
    """``best/attn`` or ``final/attn`` (also used for on-demand cache)."""
    return Path(run_dir) / checkpoint_name / ATTN_DIR_NAME


def attn_npz_path(run_dir: Path, checkpoint_name: str, stem: str) -> Path:
    safe = str(stem).replace('/', '_')
    return attn_dir(run_dir, checkpoint_name) / f'{safe}.npz'


def save_attn_npz(
    run_dir: Path,
    checkpoint_name: str,
    *,
    stem: str,
    attn_khw: np.ndarray,
    joint_names: list[str],
    image_path: str,
    bbox: list[float] | np.ndarray,
    pred_xy: np.ndarray | None = None,
    grid_wh: tuple[int, int] | None = None,
) -> Path:
    """Save float16 cross-attn [K, H, W] for one sample (H=grid_h, W=grid_w)."""
    out = attn_npz_path(run_dir, checkpoint_name, stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    attn = np.asarray(attn_khw, dtype=np.float16)
    payload: dict[str, Any] = {
        'attn': attn,
        'joint_names': np.asarray(joint_names),
        'image_path': np.asarray(str(image_path)),
        'bbox': np.asarray(bbox, dtype=np.float32).reshape(4),
        'stem': np.asarray(str(stem)),
    }
    if pred_xy is not None:
        payload['pred_xy'] = np.asarray(pred_xy, dtype=np.float32)
    if grid_wh is not None:
        payload['grid_w'] = np.int32(grid_wh[0])
        payload['grid_h'] = np.int32(grid_wh[1])
    else:
        payload['grid_w'] = np.int32(attn.shape[-1])
        payload['grid_h'] = np.int32(attn.shape[-2])
    np.savez_compressed(out, **payload)
    return out


def load_attn_npz(
    run_dir: Path, checkpoint_name: str, stem: str
) -> dict[str, Any] | None:
    path = attn_npz_path(run_dir, checkpoint_name, stem)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def list_attn_stems(run_dir: Path, checkpoint_name: str) -> list[str]:
    root = attn_dir(run_dir, checkpoint_name)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob('*.npz'))


def save_val_preds_npz(run_dir: Path, epoch: int, samples: list[dict[str, Any]]) -> Path:
    """Persist compact per-sample val predictions for one epoch.

    Expected sample keys: image_stem, image_path, bbox, role, loss, oks,
    pred_xy [K,2], gt_xy [K,2], visibility [K], joint_oks [K] (nan if unlabeled).
    """
    if not samples:
        raise ValueError('samples must be non-empty')

    out = val_preds_path(run_dir, epoch)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    k = int(np.asarray(samples[0]['pred_xy']).shape[0])
    pred_xy = np.zeros((n, k, 2), dtype=np.float32)
    gt_xy = np.zeros((n, k, 2), dtype=np.float32)
    visibility = np.zeros((n, k), dtype=np.int16)
    joint_oks = np.full((n, k), np.nan, dtype=np.float32)
    losses = np.zeros((n,), dtype=np.float32)
    oks = np.full((n,), np.nan, dtype=np.float32)
    bboxes = np.zeros((n, 4), dtype=np.float32)
    stems = []
    paths = []
    roles = []

    for i, sample in enumerate(samples):
        pred = np.asarray(sample['pred_xy'], dtype=np.float32)
        gt = np.asarray(sample['gt_xy'], dtype=np.float32)
        vis = np.asarray(sample['visibility'], dtype=np.int16)
        joks = np.asarray(sample.get('joint_oks', np.full(k, np.nan)), dtype=np.float32)
        if pred.shape != (k, 2) or gt.shape[0] != k:
            raise ValueError(f'sample {i}: unexpected pred/gt shape')
        pred_xy[i] = pred
        gt_xy[i] = gt[:, :2] if gt.ndim == 2 and gt.shape[1] >= 2 else gt.reshape(k, 2)
        visibility[i] = vis.reshape(k)
        joint_oks[i] = joks.reshape(k)
        losses[i] = float(sample['loss'])
        oks_val = sample.get('oks')
        oks[i] = float(oks_val) if oks_val is not None else np.nan
        bbox = sample['bbox']
        bboxes[i] = np.asarray(bbox, dtype=np.float32).reshape(4)
        stem = sample.get('image_stem') or Path(sample['image_path']).stem
        stems.append(str(stem))
        paths.append(str(sample['image_path']))
        role = sample.get('role')
        roles.append('' if role is None else str(role))

    np.savez_compressed(
        out,
        epoch=np.int32(epoch),
        pred_xy=pred_xy,
        gt_xy=gt_xy,
        visibility=visibility,
        joint_oks=joint_oks,
        loss=losses,
        oks=oks,
        bbox=bboxes,
        image_stem=np.asarray(stems),
        image_path=np.asarray(paths),
        role=np.asarray(roles),
    )
    return out


def load_val_preds_npz(run_dir: Path, epoch: int) -> dict[str, Any] | None:
    path = val_preds_path(run_dir, epoch)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def list_val_pred_epochs(run_dir: Path) -> list[int]:
    root = Path(run_dir) / VAL_PREDS_DIR
    if not root.is_dir():
        return []
    epochs: list[int] = []
    for path in root.glob('epoch_*.npz'):
        try:
            epochs.append(int(path.stem.split('_')[1]))
        except (IndexError, ValueError):
            continue
    return sorted(epochs)


def save_val_samples_json(run_dir: Path, samples: list[dict[str, Any]]) -> Path:
    path = Path(run_dir) / VAL_SAMPLES_NAME
    write_json(
        path,
        {
            'num_samples': len(samples),
            'updated_at': _utc_now(),
            'samples': samples,
        },
    )
    return path


def wait_while_paused(
    run_dir: Path,
    *,
    epoch: int,
    poll_seconds: float = 1.0,
    stop_requested: Callable[[], bool] | None = None,
) -> str:
    """Block while control command is pause. Returns the command that ended the wait.

    Returns ``resume``, ``stop``, or ``none``.
    """
    while True:
        if stop_requested is not None and stop_requested():
            clear_control(run_dir)
            write_status(run_dir, 'stopping', epoch=epoch, message='stop requested')
            return 'stop'
        command = read_control(run_dir)
        if command == 'stop':
            clear_control(run_dir)
            write_status(run_dir, 'stopping', epoch=epoch, message='stop requested')
            return 'stop'
        if command == 'resume':
            clear_control(run_dir)
            write_status(run_dir, 'running', epoch=epoch, message='resumed')
            return 'resume'
        if command != 'pause':
            # Spurious / cleared while paused → treat as resume.
            write_status(run_dir, 'running', epoch=epoch, message='resumed')
            return 'none'
        write_status(run_dir, 'paused', epoch=epoch, message='paused between epochs')
        time.sleep(poll_seconds)


def poll_control_between_epochs(
    run_dir: Path,
    *,
    epoch: int,
    stop_requested: Callable[[], bool] | None = None,
) -> str:
    """Handle pause/stop between epochs. Returns ``stop`` or ``continue``."""
    if stop_requested is not None and stop_requested():
        write_status(run_dir, 'stopping', epoch=epoch, message='stop requested')
        return 'stop'
    command = read_control(run_dir)
    if command == 'stop':
        clear_control(run_dir)
        write_status(run_dir, 'stopping', epoch=epoch, message='stop requested')
        return 'stop'
    if command == 'pause':
        result = wait_while_paused(run_dir, epoch=epoch, stop_requested=stop_requested)
        return 'stop' if result == 'stop' else 'continue'
    if command == 'resume':
        clear_control(run_dir)
    return 'continue'
