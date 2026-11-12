"""Gradio training observability dashboard.

Launch from repo root (dino env recommended)::

    PYTHONPATH=. python -m src.dashboard.app

Does not import the CUDA training stack; starts ``train.py`` as a subprocess.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import gradio as gr
import numpy as np
import yaml

from src.dashboard.attn_overlay import attn_joint_index, render_attn_on_crop
from src.dashboard.overlay import (
    compose_pose_viewer_html,
    empty_pose_viewer_html,
    image_to_data_url,
    keypoint_ids_sorted,
    load_keypoint_format,
    make_crop_view,
    model_coords_to_display,
    pose_overlay_svg,
    skeleton_connections,
)
from src.dashboard.sample_id import (
    attn_cache_key,
    find_sample_index,
    parse_sample_choice,
    sample_choice_label,
)
from src.dashboard.plots import (
    compare_detector_results_figure,
    compare_history_figure,
    detector_results_figure,
    detector_union_breakdown_figure,
    history_figure,
    history_oks_percentile_figure,
    history_part_oks_figure,
)
from src.util.oks_breakdown import (
    BODY_PART_JOINT_NAMES,
    oks_percentile_band_means,
    part_mean_oks_from_joint_arrays,
)
from src.training import run_artifacts
from src.util.dir import (
    DEFAULT_DETECTOR_CONFIG_PATH,
    DEFAULT_POSE_ESTIMATOR_CONFIG_PATH,
    DETECTOR_RUN_DIR,
    POSE_ESTIMATOR_RUN_DIR,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = (REPO_ROOT / POSE_ESTIMATOR_RUN_DIR).resolve()
DETECTOR_RUNS_ROOT = (REPO_ROOT / DETECTOR_RUN_DIR).resolve()
MAPPING_DIR = REPO_ROOT / 'mapping'
DEFAULT_CONFIG = REPO_ROOT / DEFAULT_POSE_ESTIMATOR_CONFIG_PATH
DEFAULT_DETECTOR_CONFIG = REPO_ROOT / DEFAULT_DETECTOR_CONFIG_PATH
TRAIN_SCRIPT = REPO_ROOT / 'src' / 'training' / 'train.py'
DETECTOR_TRAIN_SCRIPT = REPO_ROOT / 'src' / 'training' / 'train_detector.py'

# Active subprocess started from this dashboard session.
_ACTIVE_PROC: subprocess.Popen | None = None
_ACTIVE_RUN_DIR: Path | None = None
_ACTIVE_DETECTOR_PROC: subprocess.Popen | None = None
_ACTIVE_DETECTOR_RUN_DIR: Path | None = None

# Sticky-frame scrub cache: static photo layer + per-epoch SVG pose overlays.
# Keyed by (run_dir, stem); rebuilt when the stem changes or new epochs appear.
_FRAME_SERIES_CACHE: dict[tuple[str, str], dict] = {}

def list_run_dirs() -> list[str]:
    if not RUNS_ROOT.is_dir():
        return []
    dirs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.name, reverse=True)
    return [str(p) for p in dirs]


def _resolve_run_dir(run_dir: str | None) -> Path | None:
    if not run_dir:
        return None
    path = Path(run_dir)
    if not path.is_absolute():
        path = RUNS_ROOT / run_dir
    return path if path.is_dir() else None


def load_config_text(path: str | None) -> str:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_file():
        return f'# Config not found: {cfg_path}\n'
    return cfg_path.read_text(encoding='utf-8')


def list_config_choices() -> list[str]:
    choices = []
    for path in sorted(REPO_ROOT.glob('config_pose*.yaml')):
        choices.append(str(path))
    return choices or [str(DEFAULT_CONFIG)]


def list_detector_config_choices() -> list[str]:
    choices = []
    for path in sorted(REPO_ROOT.glob('config_detector*.yaml')):
        if path.name.endswith('_template.yaml'):
            continue
        choices.append(str(path))
    if DEFAULT_DETECTOR_CONFIG.is_file() and str(DEFAULT_DETECTOR_CONFIG) not in choices:
        choices.insert(0, str(DEFAULT_DETECTOR_CONFIG))
    return choices or [str(DEFAULT_DETECTOR_CONFIG)]


def list_detector_run_dirs() -> list[str]:
    if not DETECTOR_RUNS_ROOT.is_dir():
        return []
    dirs = [
        p
        for p in DETECTOR_RUNS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    ]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in dirs]


def _resolve_detector_run_dir(run_dir: str | None) -> Path | None:
    if not run_dir:
        return None
    path = Path(run_dir)
    if not path.is_absolute():
        path = DETECTOR_RUNS_ROOT / run_dir
    return path if path.is_dir() else None


def _detector_run_dropdown(selected: str | None = None) -> gr.Dropdown:
    choices = list_detector_run_dirs()
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def read_detector_results(run_dir: Path) -> list[dict]:
    """Parse Ultralytics results.csv into row dicts."""
    import csv

    path = run_dir / 'results.csv'
    if not path.is_file():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {((k or '').strip()): ((v or '').strip()) for k, v in raw.items()}
            if not row.get('epoch'):
                continue
            rows.append(row)
        return rows


def _detector_preview_paths(run_dir: Path) -> list[str]:
    names = [
        'results.png',
        'confusion_matrix.png',
        'confusion_matrix_normalized.png',
        'BoxPR_curve.png',
        'BoxF1_curve.png',
        'BoxP_curve.png',
        'BoxR_curve.png',
        'labels.jpg',
        'train_batch0.jpg',
        'train_batch1.jpg',
        'train_batch2.jpg',
        'val_batch0_labels.jpg',
        'val_batch0_pred.jpg',
        'val_batch1_labels.jpg',
        'val_batch1_pred.jpg',
    ]
    found = [str(run_dir / name) for name in names if (run_dir / name).is_file()]
    # Also pick up any extra val_batch* that appear mid-run.
    for path in sorted(run_dir.glob('val_batch*.jpg')):
        text = str(path)
        if text not in found:
            found.append(text)
    return found[:16]


def _detector_status_markdown(run_dir: Path, rows: list[dict]) -> str:
    args_path = run_dir / 'args.yaml'
    args: dict = {}
    if args_path.is_file():
        try:
            loaded = yaml.safe_load(args_path.read_text(encoding='utf-8')) or {}
            if isinstance(loaded, dict):
                args = loaded
        except Exception:
            args = {}

    best = run_dir / 'weights' / 'best.pt'
    last = run_dir / 'weights' / 'last.pt'
    last_row = rows[-1] if rows else {}
    best_map = None
    best_epoch = None
    for row in rows:
        try:
            value = float(row.get('metrics/mAP50-95(B)', '') or 'nan')
        except ValueError:
            continue
        if value != value:
            continue
        if best_map is None or value > best_map:
            best_map = value
            best_epoch = int(float(row.get('epoch', 0)))

    export_bits = ''
    data_yaml = args.get('data')
    if data_yaml:
        meta_path = Path(str(data_yaml)).parent / 'export_meta.json'
        if meta_path.is_file():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                export_bits = (
                    f"**policy:** `{meta.get('policy')}`  \n"
                    f"**union τ:** `{meta.get('union_iou_threshold')}`  \n"
                    f"**split:** `{meta.get('split_mode')}`  \n"
                    f"**val tasks:** `{', '.join(meta.get('val_tasks') or [])}`  \n"
                    f"**frames:** train=`{meta.get('frames', {}).get('train')}` "
                    f"val=`{meta.get('frames', {}).get('val')}`  \n"
                    f"**unions / separate pairs:** "
                    f"`{(meta.get('label_stats') or {}).get('unions')}` / "
                    f"`{(meta.get('label_stats') or {}).get('separate_pairs')}`  \n"
                )
            except Exception:
                export_bits = ''

    return (
        f'**run:** `{run_dir}`  \n'
        f'**model:** `{args.get("model", "")}`  \n'
        f'**imgsz / batch / epochs:** '
        f'`{args.get("imgsz")}` / `{args.get("batch")}` / `{args.get("epochs")}`  \n'
        f'**data:** `{args.get("data", "")}`  \n'
        f'{export_bits}'
        f'**epochs logged:** `{len(rows)}`  \n'
        f'**latest epoch:** `{last_row.get("epoch", "—")}`  \n'
        f'**latest mAP50 / mAP50-95:** '
        f'`{last_row.get("metrics/mAP50(B)", "—")}` / '
        f'`{last_row.get("metrics/mAP50-95(B)", "—")}`  \n'
        f'**latest P / R:** '
        f'`{last_row.get("metrics/precision(B)", "—")}` / '
        f'`{last_row.get("metrics/recall(B)", "—")}`  \n'
        f'**best mAP50-95:** '
        f'`{best_map if best_map is not None else "—"}` '
        f'(epoch `{best_epoch if best_epoch is not None else "—"}`)  \n'
        f'**weights:** best=`{best.is_file()}` last=`{last.is_file()}`'
    )


def _detector_log_tail(run_dir: Path) -> str:
    candidates = [
        DETECTOR_RUNS_ROOT / f'{run_dir.name}_train.log',
        run_dir / 'train.log',
    ]
    for path in candidates:
        if path.is_file():
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            # Ultralytics progress lines are huge; keep last non-empty-ish chunk.
            return '\n'.join(lines[-60:])
    # Fall back to a compact results.csv tail.
    results = run_dir / 'results.csv'
    if results.is_file():
        lines = results.read_text(encoding='utf-8', errors='replace').splitlines()
        return '\n'.join(lines[-40:])
    return ''


def _detector_union_breakdown_image(run_dir: Path, checkpoint: str = 'best'):
    try:
        from src.dashboard.detector_union_breakdown import compute_union_breakdown

        summary = compute_union_breakdown(run_dir, checkpoint=checkpoint)
        return detector_union_breakdown_figure(summary), summary
    except Exception as exc:
        return detector_union_breakdown_figure(None), {'error': str(exc)}


def refresh_detector_live(run_dir: str):
    path = _resolve_detector_run_dir(run_dir)
    if path is None:
        empty = detector_results_figure([])
        empty_u = detector_union_breakdown_figure(None)
        return (
            empty,
            empty_u,
            'No detector run selected',
            '',
            [],
            _detector_run_dropdown(run_dir),
        )
    rows = read_detector_results(path)
    fig = detector_results_figure(rows)
    fig_u, breakdown = _detector_union_breakdown_image(path, 'best')
    status = _detector_status_markdown(path, rows)
    if isinstance(breakdown, dict) and breakdown.get('by_box_kind'):
        bits = []
        for kind, row in (breakdown.get('by_box_kind') or {}).items():
            if not row.get('n_gt'):
                continue
            bits.append(
                f"{kind}: n={row['n_gt']} recall="
                f"{row['recall']:.3f} meanIoU={row['mean_iou']:.3f}"
            )
        if bits:
            status += '  \n**val union breakdown (best):** ' + ' · '.join(bits)
    elif isinstance(breakdown, dict) and breakdown.get('error'):
        status += f"  \n**val union breakdown:** _{breakdown['error']}_"
    log_tail = _detector_log_tail(path)
    previews = _detector_preview_paths(path)
    return fig, fig_u, status, log_tail, previews, _detector_run_dropdown(str(path))


def compare_detector_runs(run_a: str, run_b: str):
    path_a = _resolve_detector_run_dir(run_a)
    path_b = _resolve_detector_run_dir(run_b)
    rows_a = read_detector_results(path_a) if path_a else []
    rows_b = read_detector_results(path_b) if path_b else []
    return compare_detector_results_figure(
        rows_a,
        rows_b,
        path_a.name if path_a else 'A',
        path_b.name if path_b else 'B',
    )


def start_detector_training(config_path: str, device: str = '0') -> str:
    global _ACTIVE_DETECTOR_PROC, _ACTIVE_DETECTOR_RUN_DIR
    if _ACTIVE_DETECTOR_PROC is not None and _ACTIVE_DETECTOR_PROC.poll() is None:
        raise gr.Error(f'Detector training already running (pid={_ACTIVE_DETECTOR_PROC.pid})')
    cfg = Path(config_path)
    if not cfg.is_file():
        raise gr.Error(f'Config not found: {cfg}')
    if not DETECTOR_TRAIN_SCRIPT.is_file():
        raise gr.Error(f'Train script missing: {DETECTOR_TRAIN_SCRIPT}')

    before = set(list_detector_run_dirs())
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{REPO_ROOT}{os.pathsep}{env["PYTHONPATH"]}'
        if env.get('PYTHONPATH')
        else str(REPO_ROOT)
    )
    python = _training_python()
    DETECTOR_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = DETECTOR_RUNS_ROOT / '_dashboard_detector_train.log'
    log_handle = log_path.open('w', encoding='utf-8')
    cmd = [
        python,
        '-m',
        'src.training.train_detector',
        '--config',
        str(cfg),
        '--device',
        str(device or '0'),
    ]
    _ACTIVE_DETECTOR_PROC = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    run_dir = None
    for _ in range(60):
        time.sleep(0.5)
        after = set(list_detector_run_dirs())
        new = sorted(after - before, key=lambda p: Path(p).stat().st_mtime, reverse=True)
        if new:
            run_dir = Path(new[0])
            break
        if _ACTIVE_DETECTOR_PROC.poll() is not None:
            break
    _ACTIVE_DETECTOR_RUN_DIR = run_dir
    if run_dir is not None:
        (DETECTOR_RUNS_ROOT / '.active_run').write_text(
            str(run_dir.resolve()) + '\n', encoding='utf-8'
        )
        # Prefer a named log next to the run for the Live tab.
        named = DETECTOR_RUNS_ROOT / f'{run_dir.name}_train.log'
        try:
            if not named.is_file() and log_path.is_file():
                named.write_text(log_path.read_text(encoding='utf-8', errors='replace'))
        except Exception:
            pass
    if run_dir is None:
        return (
            f'Started pid={_ACTIVE_DETECTOR_PROC.pid} but run dir not detected yet. '
            f'Check {log_path}'
        )
    return f'Started pid={_ACTIVE_DETECTOR_PROC.pid} run={run_dir}'


def force_kill_active_detector() -> str:
    global _ACTIVE_DETECTOR_PROC
    if _ACTIVE_DETECTOR_PROC is None or _ACTIVE_DETECTOR_PROC.poll() is not None:
        return 'No active dashboard-started detector process'
    _ACTIVE_DETECTOR_PROC.send_signal(signal.SIGTERM)
    try:
        _ACTIVE_DETECTOR_PROC.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _ACTIVE_DETECTOR_PROC.kill()
    pid = _ACTIVE_DETECTOR_PROC.pid
    _ACTIVE_DETECTOR_PROC = None
    return f'Sent SIGTERM/KILL to detector pid={pid}'


def _detector_ckpt_choices(run_dir: str | None) -> list[str]:
    from src.dashboard.detector_overlay import list_detector_checkpoints

    path = _resolve_detector_run_dir(run_dir)
    if path is None:
        return ['best', 'last']
    names = list_detector_checkpoints(path)
    return names or ['best', 'last']


def _preferred_detector_browser_run(run_dir: str | None) -> str | None:
    """Prefer explicit selection; else a run that already has a pred cache."""
    from src.dashboard.detector_overlay import list_cached_checkpoints

    choices = list_detector_run_dirs()
    if not choices:
        return None
    selected = _resolve_detector_run_dir(run_dir) if run_dir else None
    if selected is not None and str(selected) in choices:
        return str(selected)
    for choice in choices:
        path = _resolve_detector_run_dir(choice)
        if path is not None and list_cached_checkpoints(path):
            return choice
    return choices[0]


def load_detector_browser(run_dir: str, checkpoint: str, sort_by: str, current_stem: str | None):
    from src.dashboard.detector_overlay import (
        list_cached_checkpoints,
        list_val_stems,
        load_pred_cache,
        normalize_checkpoint_name,
        sorted_stems,
    )

    run_dir = _preferred_detector_browser_run(run_dir)
    path = _resolve_detector_run_dir(run_dir)
    empty_frames = gr.Dropdown(choices=[], value=None)
    empty_ckpt = gr.Dropdown(choices=['best'], value='best')
    if path is None:
        return (
            _detector_run_dropdown(run_dir),
            empty_ckpt,
            empty_frames,
            'Select a detector run',
            None,
            None,
            'No run selected',
        )

    ckpt_choices = _detector_ckpt_choices(str(path))
    cached = list_cached_checkpoints(path)
    ckpt = normalize_checkpoint_name(checkpoint)
    if ckpt not in ckpt_choices:
        ckpt = ckpt_choices[0]
    # If selected ckpt has no cache but another does, jump to a cached one.
    if ckpt not in cached and cached:
        for name in ('best', 'last', *cached):
            if name in cached and name in ckpt_choices:
                ckpt = name
                break
    try:
        stems_all = list_val_stems(path)
    except Exception as exc:
        return (
            _detector_run_dropdown(str(path)),
            gr.Dropdown(choices=ckpt_choices, value=ckpt),
            empty_frames,
            f'Failed to list val frames: {exc}',
            None,
            None,
            str(exc),
        )
    cache = load_pred_cache(path, ckpt)
    stems = sorted_stems(stems_all, cache, sort_by)
    preferred = current_stem if current_stem in stems else (stems[0] if stems else None)
    if cache:
        cache_note = (
            f'pred cache OK ({len(cache.get("frames") or {})} frames) — '
            f'orange=P, green=GT'
        )
    else:
        cache_note = (
            f'pred cache MISSING for ckpt={ckpt} '
            f'(cached: {cached or "none"}) — green GT only. '
            f'Click Cache preds for checkpoint'
        )
    status = (
        f'{path.name}  ckpt={ckpt}  val frames={len(stems)}  {cache_note}'
    )
    image, info = (None, status)
    if preferred:
        from src.dashboard.detector_overlay import render_frame

        image, info = render_frame(path, preferred, ckpt)
    return (
        _detector_run_dropdown(str(path)),
        gr.Dropdown(choices=ckpt_choices, value=ckpt),
        gr.Dropdown(choices=stems, value=preferred),
        status,
        preferred,
        image,
        info,
    )


def render_detector_frame(run_dir: str, checkpoint: str, stem: str):
    from src.dashboard.detector_overlay import render_frame

    path = _resolve_detector_run_dir(run_dir)
    if path is None or not stem:
        return None, 'Select a run and frame'
    return render_frame(path, stem, checkpoint or 'best')


def cache_detector_browser_preds(run_dir: str, checkpoint: str, device: str):
    """Subprocess into TRAIN_PYTHON + ultralytics to build val_browser cache."""
    path = _resolve_detector_run_dir(run_dir)
    if path is None:
        raise gr.Error('Select a valid detector run')
    ckpt = (checkpoint or 'best').replace('.pt', '')
    weight = path / 'weights' / f'{ckpt}.pt'
    if not weight.is_file():
        raise gr.Error(f'Missing weights: {weight}')
    python = _training_python()
    cmd = [
        python,
        '-m',
        'src.dashboard.compute_detector_preds',
        '--run-dir',
        str(path),
        '--checkpoint',
        ckpt,
        '--device',
        str(device or 'cpu'),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{REPO_ROOT}{os.pathsep}{env["PYTHONPATH"]}'
        if env.get('PYTHONPATH')
        else str(REPO_ROOT)
    )
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60 * 45,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '')[-1000:]
        raise gr.Error(f'Cache failed (exit {proc.returncode}):\n{tail}')
    n_line = (proc.stdout or '').strip().splitlines()[-1:] or ['done']
    return f'Cached {ckpt}: {n_line[0]}'


def save_config_text(path: str, text: str) -> str:
    out = Path(path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Validate YAML before writing.
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise gr.Error('Config must be a YAML mapping')
    out.write_text(text if text.endswith('\n') else text + '\n', encoding='utf-8')
    return f'Saved {out}'


def start_training(config_path: str, workers: int, init_checkpoint: str = '') -> str:
    global _ACTIVE_PROC, _ACTIVE_RUN_DIR
    if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
        raise gr.Error(f'Training already running (pid={_ACTIVE_PROC.pid})')
    cfg = Path(config_path)
    if not cfg.is_file():
        raise gr.Error(f'Config not found: {cfg}')
    if not TRAIN_SCRIPT.is_file():
        raise gr.Error(f'Train script missing: {TRAIN_SCRIPT}')
    init_path = (init_checkpoint or '').strip()
    if init_path:
        init_file = Path(init_path)
        if not init_file.is_file():
            raise gr.Error(f'Init checkpoint not found: {init_file}')

    before = set(list_run_dirs())
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{REPO_ROOT}{os.pathsep}{env["PYTHONPATH"]}'
        if env.get('PYTHONPATH')
        else str(REPO_ROOT)
    )
    python = _training_python()
    cmd = [
        python,
        str(TRAIN_SCRIPT),
        '--config',
        str(cfg),
        '--workers',
        str(int(workers)),
    ]
    if init_path:
        cmd.extend(['--init-checkpoint', init_path])
    _ACTIVE_PROC = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    # Wait briefly for create_run_dir.
    run_dir = None
    for _ in range(40):
        time.sleep(0.25)
        after = set(list_run_dirs())
        new = sorted(after - before)
        if new:
            run_dir = Path(new[0])
            break
        if _ACTIVE_PROC.poll() is not None:
            break
    _ACTIVE_RUN_DIR = run_dir
    if run_dir is not None:
        (RUNS_ROOT / '.active_run').write_text(str(run_dir.resolve()) + '\n', encoding='utf-8')
    if run_dir is None:
        return (
            f'Started pid={_ACTIVE_PROC.pid} but run dir not detected yet. '
            f'Refresh the run list.'
        )
    return f'Started pid={_ACTIVE_PROC.pid} run={run_dir}'


def send_control(run_dir: str, command: str) -> str:
    path = _resolve_run_dir(run_dir)
    if path is None:
        raise gr.Error('Select a valid run directory')
    run_artifacts.write_control(path, command)
    return f'Wrote control={command} to {path / run_artifacts.CONTROL_NAME}'


def force_kill_active() -> str:
    global _ACTIVE_PROC
    if _ACTIVE_PROC is None or _ACTIVE_PROC.poll() is not None:
        return 'No active dashboard-started process'
    _ACTIVE_PROC.send_signal(signal.SIGTERM)
    try:
        _ACTIVE_PROC.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _ACTIVE_PROC.kill()
    pid = _ACTIVE_PROC.pid
    _ACTIVE_PROC = None
    return f'Sent SIGTERM/KILL to pid={pid}'


def _run_dropdown(selected: str | None = None) -> gr.Dropdown:
    choices = list_run_dirs()
    value = selected if selected in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def _preferred_browser_run(run_dir: str | None) -> str | None:
    """Refresh-aware run pick: keep an existing selection, else active, else newest with preds."""
    choices = list_run_dirs()
    if not choices:
        return None
    selected = _resolve_run_dir(run_dir) if run_dir else None
    if selected is not None and str(selected) in choices:
        return str(selected)
    active_file = RUNS_ROOT / '.active_run'
    if active_file.is_file():
        active = _resolve_run_dir(active_file.read_text(encoding='utf-8').strip())
        if active is not None and str(active) in choices:
            return str(active)
    for choice in choices:
        path = _resolve_run_dir(choice)
        if path is not None and run_artifacts.list_val_pred_epochs(path):
            return choice
    return choices[0]


def _browser_run_dropdown(selected: str | None = None) -> gr.Dropdown:
    choices = list_run_dirs()
    value = _preferred_browser_run(selected)
    if value is not None and value not in choices and choices:
        value = choices[0]
    return gr.Dropdown(choices=choices, value=value)


def _preferred_live_run(run_dir: str | None) -> Path | None:
    """Prefer explicit selection; else active_run pointer; else newest run dir."""
    path = _resolve_run_dir(run_dir) if run_dir else None
    if path is not None and (
        (path / run_artifacts.STATUS_NAME).is_file()
        or (path / run_artifacts.HISTORY_NAME).is_file()
        or (path / 'log.csv').is_file()
    ):
        # If Gradio dropdown is stale on an old run while a new one is active, prefer active.
        active_file = RUNS_ROOT / '.active_run'
        if active_file.is_file():
            active = _resolve_run_dir(active_file.read_text(encoding='utf-8').strip())
            if active is not None and active != path:
                active_status = run_artifacts.read_status(active) or {}
                if active_status.get('state') in ('running', 'paused', 'stopping'):
                    return active
        return path
    active_file = RUNS_ROOT / '.active_run'
    if active_file.is_file():
        active = _resolve_run_dir(active_file.read_text(encoding='utf-8').strip())
        if active is not None:
            return active
    choices = list_run_dirs()
    return _resolve_run_dir(choices[0]) if choices else None


def _dance23_part_indices() -> dict[str, list[int]]:
    """Local joint ids for dance_23 body parts (dashboard backfill only)."""
    mapping = load_keypoint_format(MAPPING_DIR / 'dance_23.yaml')
    keypoints = mapping.get('keypoints') or {}
    return {
        part: [int(keypoints[name]['id']) for name in names if name in keypoints]
        for part, names in BODY_PART_JOINT_NAMES.items()
    }


def enrich_history_from_val_preds(
    run_dir: Path, history: list[dict]
) -> list[dict]:
    """Fill missing part / percentile OKS from per-epoch val_preds npz when present."""
    preds_dir = run_dir / 'val_preds'
    if not history or not preds_dir.is_dir():
        return history
    part_indices = None
    enriched: list[dict] = []
    for row in history:
        row = dict(row)
        needs_parts = not row.get('val_part_oks')
        needs_bands = not row.get('val_oks_percentile_bands')
        if not needs_parts and not needs_bands:
            enriched.append(row)
            continue
        epoch = int(row.get('epoch', -1))
        npz_path = preds_dir / f'epoch_{epoch:03d}.npz'
        if not npz_path.is_file():
            enriched.append(row)
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            enriched.append(row)
            continue
        if needs_bands and 'oks' in data.files:
            row['val_oks_percentile_bands'] = oks_percentile_band_means(data['oks'])
        if needs_parts and 'joint_oks' in data.files and 'visibility' in data.files:
            if part_indices is None:
                part_indices = _dance23_part_indices()
            row['val_part_oks'] = part_mean_oks_from_joint_arrays(
                data['joint_oks'], data['visibility'], part_indices
            )
        enriched.append(row)
    return enriched


def refresh_live(run_dir: str):
    path = _preferred_live_run(run_dir)
    if path is None:
        empty = history_figure([])
        empty_part = history_part_oks_figure([])
        empty_pct = history_oks_percentile_figure([])
        return empty, empty_part, empty_pct, 'No run selected', '', _run_dropdown(run_dir)

    status = run_artifacts.read_status(path) or {}
    history = enrich_history_from_val_preds(path, run_artifacts.read_history(path))
    fig = history_figure(history)
    fig_parts = history_part_oks_figure(history)
    fig_pct = history_oks_percentile_figure(history)
    state = status.get('state', 'unknown')
    epoch = status.get('epoch')
    message = status.get('message', '')
    status_md = (
        f'**state:** `{state}`  \n'
        f'**epoch:** `{epoch}`  \n'
        f'**pid:** `{status.get("pid")}`  \n'
        f'**message:** {message}  \n'
        f'**updated:** `{status.get("updated_at", "")}`  \n'
        f'**run:** `{path}`'
    )
    log_path = path / 'log.csv'
    log_tail = ''
    if log_path.is_file():
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        log_tail = '\n'.join(lines[-80:])
    return fig, fig_parts, fig_pct, status_md, log_tail, _run_dropdown(str(path))


def _load_run_image_settings(run_dir: Path) -> tuple[dict, tuple[int, int], float, str]:
    cfg_path = run_dir / 'config.yaml'
    raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) if cfg_path.is_file() else {}
    image = raw.get('image') or {}
    head = raw.get('head') or {}
    fmt_name = head.get('keypoint_format_name', 'dance_23')
    mapping = load_keypoint_format(MAPPING_DIR / f'{fmt_name}.yaml')
    keypoints = mapping['keypoints']
    size = (int(image.get('width', 384)), int(image.get('height', 512)))
    padding = float(image.get('bbox_padding', 0.08))
    return keypoints, size, padding, fmt_name


def _training_python() -> str:
    """Interpreter that has torch + training deps (not necessarily the dashboard venv)."""
    override = os.environ.get('TRAIN_PYTHON')
    if override and Path(override).is_file():
        return override
    conda = Path(os.environ.get('CONDA_PREFIX', ''))
    # Prefer sibling dino env when dashboard runs from a dedicated venv.
    dino = Path.home() / 'anaconda3' / 'envs' / 'dino' / 'bin' / 'python'
    if dino.is_file():
        return str(dino)
    if (conda / 'bin' / 'python').is_file():
        return str(conda / 'bin' / 'python')
    return sys.executable


def _epoch_list(run_dir: str) -> list[int]:
    path = _resolve_run_dir(run_dir)
    if path is None:
        return []
    return run_artifacts.list_val_pred_epochs(path)


def _watchlist_stems(run_dir: Path) -> list[str]:
    cfg_path = run_dir / 'config.yaml'
    if not cfg_path.is_file():
        return []
    raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    training = raw.get('training') or {}
    return [str(s) for s in (training.get('watchlist_frames') or [])]


def _sample_role(data: dict, index: int) -> str:
    roles = data.get('role')
    if roles is None:
        return ''
    return str(roles[index]).strip()


def _stem_task_frame(stem: str) -> tuple[int, int, str]:
    """Parse ``task_N_frame_XXXXXX`` → ``(task_id, frame, stem)`` for sorting."""
    text = str(stem)
    task_id = 10**9
    frame = 10**9
    if '_frame_' in text:
        head, _, frame_s = text.partition('_frame_')
        try:
            frame = int(frame_s)
        except ValueError:
            frame = 10**9
        if head.startswith('task_'):
            try:
                task_id = int(head.split('_', 1)[1])
            except ValueError:
                task_id = 10**9
    return task_id, frame, text


def _role_sort_key(role: str) -> int:
    role = (role or '').strip().lower()
    if role == 'lead':
        return 0
    if role == 'follow':
        return 1
    return 2


def load_epoch_browser(run_dir: str, sort_by: str, current_stem: str | None = None):
    """Load sticky frame list + epoch slider bounds for a run."""
    path = _resolve_run_dir(run_dir)
    epochs = _epoch_list(run_dir)
    empty_slider = gr.Slider(minimum=0, maximum=0, step=1, value=0, interactive=False)
    empty_frames = gr.Dropdown(choices=[], value=None)
    if path is None or not epochs:
        return empty_frames, empty_slider, 'No val_preds yet for this run', None

    # Sample identity is stem+role (stable across epochs). Rank using latest epoch.
    latest = epochs[-1]
    data = run_artifacts.load_val_preds_npz(path, latest)
    if data is None:
        return empty_frames, empty_slider, 'Failed to load latest val_preds', None

    n = len(data['loss'])
    order = list(range(n))
    sort_key = (sort_by or 'loss').strip().lower()
    if sort_key == '1-oks':
        order.sort(
            key=lambda i: (
                -1.0
                if not np.isfinite(data['oks'][i])
                else 1.0 - float(data['oks'][i])
            ),
            reverse=True,
        )
    elif sort_key == 'task':
        order.sort(
            key=lambda i: (
                *_stem_task_frame(str(data['image_stem'][i])),
                _role_sort_key(_sample_role(data, i)),
                i,
            )
        )
    else:
        order.sort(key=lambda i: float(data['loss'][i]), reverse=True)

    watch = set(_watchlist_stems(path))
    choices: list[str] = []
    seen: set[str] = set()

    def _add_choice(label: str) -> None:
        if label and label not in seen:
            seen.add(label)
            choices.append(label)

    # Watchlist first for metric sorts; task order keeps pure task/frame sequence.
    # Only list stems that exist in val_preds (watchlist can name train-only clips).
    missing_watch: list[str] = []
    if sort_key != 'task':
        for stem in _watchlist_stems(path):
            if not stem:
                continue
            matched = False
            for i in range(n):
                if str(data['image_stem'][i]) == stem:
                    _add_choice(
                        sample_choice_label(stem, _sample_role(data, i), i)
                    )
                    matched = True
            if not matched:
                missing_watch.append(stem)
    for i in order:
        stem = str(data['image_stem'][i])
        _add_choice(sample_choice_label(stem, _sample_role(data, i), i))

    preferred = current_stem if current_stem in choices else None
    if preferred is None and current_stem:
        # Old sticky plain stem → first matching person choice.
        for label in choices:
            if label == current_stem or label.startswith(f'{current_stem} ['):
                preferred = label
                break
    if preferred is None and watch and sort_key != 'task':
        for label in choices:
            stem, _, _ = parse_sample_choice(label)
            if stem in watch:
                preferred = label
                break
    if preferred is None:
        preferred = choices[0] if choices else None

    slider = gr.Slider(
        minimum=int(epochs[0]),
        maximum=int(epochs[-1]),
        step=1,
        value=int(epochs[-1]),
        label='Epoch',
        interactive=True,
    )
    frames = gr.Dropdown(
        choices=choices,
        value=preferred,
        label='Sample (sticky across epochs; stem [role])',
    )
    status = (
        f'{len(epochs)} epochs with preds ({epochs[0]}–{epochs[-1]}); '
        f'{len(choices)} samples (person-aware). Pick one, then scrub epoch.'
    )
    if missing_watch:
        preview = ', '.join(missing_watch[:4])
        more = f' (+{len(missing_watch) - 4} more)' if len(missing_watch) > 4 else ''
        status += (
            f'  \nWatchlist not in this val split (omitted from dropdown): '
            f'{preview}{more}'
        )
    return frames, slider, status, preferred


def _build_frame_series(path: Path, stem: str) -> dict | None:
    """Cache static photo data-URL + per-epoch SVG overlays (same box)."""
    epochs = run_artifacts.list_val_pred_epochs(path)
    if not epochs:
        return None
    keypoints, image_size, padding, fmt = _load_run_image_settings(path)
    kp_ids = keypoint_ids_sorted(keypoints)
    connections = skeleton_connections(keypoints)
    view = None
    data_url = ''
    overlays: dict[int, str] = {}
    infos: dict[int, str] = {}
    for epoch_i in epochs:
        data = run_artifacts.load_val_preds_npz(path, epoch_i)
        if data is None:
            continue
        index = find_sample_index(data, stem)
        if index is None:
            continue
        role = _sample_role(data, index)
        if view is None:
            view = make_crop_view(
                str(data['image_path'][index]),
                data['bbox'][index].tolist(),
                image_size=image_size,
                bbox_padding=padding,
            )
            data_url = image_to_data_url(view.base)
        loss = float(data['loss'][index])
        oks_v = data['oks'][index]
        oks = float(oks_v) if np.isfinite(oks_v) else None
        points = model_coords_to_display(view, data['pred_xy'][index])
        w, h = view.base.size
        overlays[epoch_i] = pose_overlay_svg(
            w,
            h,
            points,
            keypoint_ids=kp_ids,
            connections=connections,
            loss=loss,
            oks=oks,
            title=f'epoch {epoch_i}',
        )
        role_part = f'  role={role}' if role else ''
        infos[epoch_i] = (
            f'epoch={epoch_i}/{epochs[-1]}  sample={stem}{role_part}  index={index}  '
            f'loss={loss:.4f}  oks={oks}  format={fmt}  '
            f'layers=photo+svg  crop={w}x{h}'
        )
    if view is None or not overlays:
        return None
    w, h = view.base.size
    return {
        'epochs': sorted(overlays),
        'data_url': data_url,
        'overlays': overlays,
        'infos': infos,
        'fmt': fmt,
        'size': (w, h),
    }


def _get_frame_series(run_dir: str, stem: str) -> dict | None:
    path = _resolve_run_dir(run_dir)
    if path is None or not stem:
        return None
    key = (str(path), str(stem))
    epochs_now = tuple(run_artifacts.list_val_pred_epochs(path))
    cached = _FRAME_SERIES_CACHE.get(key)
    if cached is not None and cached.get('epoch_sig') == epochs_now:
        return cached
    built = _build_frame_series(path, str(stem))
    if built is None:
        _FRAME_SERIES_CACHE.pop(key, None)
        return None
    built['epoch_sig'] = epochs_now
    _FRAME_SERIES_CACHE.clear()  # keep only the active sticky frame in memory
    _FRAME_SERIES_CACHE[key] = built
    return built


def _snap_epoch(series: dict, epoch: float | int | str) -> int | None:
    try:
        epoch_i = int(epoch)
    except (TypeError, ValueError):
        return None
    available = series['epochs']
    if not available:
        return None
    if epoch_i not in series['overlays']:
        return min(available, key=lambda e: abs(e - epoch_i))
    return epoch_i


def _missing_val_pred_message(run_dir: str, stem: str) -> str:
    """Explain why a dropdown/stem has no scrubbable val_preds."""
    path = _resolve_run_dir(run_dir)
    if path is None:
        return 'Select a run'
    plain, _, _ = parse_sample_choice(stem)
    watch = set(_watchlist_stems(path))
    if plain in watch or stem in watch:
        return (
            f'No val_preds for `{plain}` — it is on the watchlist but not in this '
            f"run's val/test split (check `training.test_clips`)."
        )
    return f'No val_preds for `{plain}` yet'


def render_frame_viewer(run_dir: str, epoch: float | int | str, stem: str):
    """Return (viewer_html, info) with photo + SVG sharing one aligned box."""
    empty = empty_pose_viewer_html()
    if stem is None or str(stem).strip() == '':
        return empty, 'Select a frame'
    series = _get_frame_series(run_dir, str(stem))
    if series is None:
        return empty, _missing_val_pred_message(run_dir, str(stem))
    epoch_i = _snap_epoch(series, epoch)
    if epoch_i is None:
        return empty, 'Select an epoch'
    w, h = series['size']
    html = compose_pose_viewer_html(
        series['data_url'],
        w,
        h,
        series['overlays'][epoch_i],
    )
    return html, series['infos'][epoch_i]


def render_overlay_payload(run_dir: str, epoch: float | int | str, stem: str):
    """SVG + info for in-place overlay swap (photo DOM node stays put)."""
    if stem is None or str(stem).strip() == '':
        return '', 'Select a frame'
    series = _get_frame_series(run_dir, str(stem))
    if series is None:
        return '', _missing_val_pred_message(run_dir, str(stem))
    epoch_i = _snap_epoch(series, epoch)
    if epoch_i is None:
        return '', 'Select an epoch'
    return series['overlays'][epoch_i], series['infos'][epoch_i]


_PATCH_OVERLAY_JS = """
(svg) => {
  const slot = document.getElementById('pose-overlay-slot');
  if (slot && typeof svg === 'string') {
    slot.innerHTML = svg;
  }
}
"""


def step_epoch(run_dir: str, epoch: float | int | str, stem: str, delta: int):
    series = _get_frame_series(run_dir, str(stem) if stem else '')
    available = series['epochs'] if series else _epoch_list(run_dir)
    if not available:
        return 0, '', 'No val_preds yet'
    try:
        cur = int(epoch)
    except (TypeError, ValueError):
        cur = available[-1]
    if cur not in available:
        cur = min(available, key=lambda e: abs(e - cur))
    idx = available.index(cur)
    idx = max(0, min(len(available) - 1, idx + delta))
    new_epoch = available[idx]
    overlay, info = render_overlay_payload(run_dir, new_epoch, stem)
    return new_epoch, overlay, info


def compare_runs(run_a: str, run_b: str):
    path_a = _resolve_run_dir(run_a)
    path_b = _resolve_run_dir(run_b)
    rows_a = run_artifacts.read_history(path_a) if path_a else []
    rows_b = run_artifacts.read_history(path_b) if path_b else []
    label_a = path_a.name if path_a else 'A'
    label_b = path_b.name if path_b else 'B'
    return compare_history_figure(rows_a, rows_b, label_a, label_b)


def _joint_name_choices() -> list[str]:
    mapping = load_keypoint_format(MAPPING_DIR / 'dance_23.yaml')
    return [
        name
        for name, _ in sorted(
            mapping['keypoints'].items(), key=lambda item: item[1]['id']
        )
    ]


def _run_image_size_padding(run_dir: Path) -> tuple[tuple[int, int], float]:
    cfg_path = run_dir / 'config.yaml'
    raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) if cfg_path.is_file() else {}
    image = raw.get('image') or {}
    return (
        (int(image.get('width', 384)), int(image.get('height', 512))),
        float(image.get('bbox_padding', 0.08)),
    )


def render_attn_heatmap(
    run_dir: str,
    checkpoint_name: str,
    stem: str,
    joint: str,
):
    """Load saved attn npz and render one joint heatmap (no torch)."""
    path = _resolve_run_dir(run_dir)
    if path is None:
        return None, 'Select a run'
    if not stem:
        return None, 'Select a frame'
    if not joint:
        return None, 'Select a joint'
    cache_stem = attn_cache_key(stem)
    plain_stem, role, _ = parse_sample_choice(stem)
    # Prefer requested ckpt; fall back to ondemand cache. Also try legacy plain stem.
    candidates = [cache_stem]
    if cache_stem != plain_stem:
        candidates.append(plain_stem)
    data = None
    source = checkpoint_name
    used_key = cache_stem
    for key in candidates:
        data = run_artifacts.load_attn_npz(path, checkpoint_name, key)
        if data is not None:
            used_key = key
            break
    if data is None and checkpoint_name != 'ondemand':
        for key in candidates:
            data = run_artifacts.load_attn_npz(path, 'ondemand', key)
            if data is not None:
                source = 'ondemand'
                used_key = key
                break
    if data is None:
        available = []
        for name in ('best', 'final', 'ondemand'):
            stems = run_artifacts.list_attn_stems(path, name)
            if any(k in stems for k in candidates):
                available.append(name)
        hint = (
            f' Saved under: {", ".join(available)}.'
            if available
            else ' None saved yet — use Compute attn (needs TRAIN_PYTHON / dino env).'
        )
        return None, f'No attn npz for {stem!r} @ {checkpoint_name}.{hint}'

    names = [str(n) for n in data['joint_names']]
    try:
        jid = attn_joint_index(names, joint)
    except KeyError as exc:
        return None, str(exc)
    attn = np.asarray(data['attn'][jid], dtype=np.float32)
    image_size, padding = _run_image_size_padding(path)
    image = render_attn_on_crop(
        str(data['image_path']),
        data['bbox'].tolist(),
        attn,
        image_size=image_size,
        bbox_padding=padding,
        title=f'{joint}  attn@{source}',
    )
    role_part = f'  role={role}' if role else ''
    info = (
        f'source={source}  sample={stem}{role_part}  cache={used_key}  joint={joint}  '
        f'grid={int(data["grid_w"])}x{int(data["grid_h"])}  '
        f'attn_peak={float(attn.max()):.4f}'
    )
    return image, info


def compute_attn_heatmap(
    run_dir: str,
    checkpoint_name: str,
    stem: str,
    joint: str,
):
    """Subprocess into TRAIN_PYTHON to compute + cache attn, then render."""
    path = _resolve_run_dir(run_dir)
    if path is None:
        return None, 'Select a run'
    if not stem or not joint:
        return None, 'Select a frame and joint'
    ckpt = checkpoint_name if checkpoint_name in ('best', 'final') else 'best'
    if not (path / f'{ckpt}.pt').is_file():
        return None, f'Missing {ckpt}.pt under {path.name}'

    python = _training_python()
    cache_stem = attn_cache_key(stem)
    out_jpg = path / 'ondemand_attn' / f'{cache_stem}_{joint}.jpg'
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        '-m',
        'src.dashboard.compute_attn',
        '--run-dir',
        str(path),
        '--checkpoint',
        ckpt,
        '--stem',
        stem,
        '--joint',
        joint,
        '--out',
        str(out_jpg),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f'{REPO_ROOT}{os.pathsep}{env["PYTHONPATH"]}'
        if env.get('PYTHONPATH')
        else str(REPO_ROOT)
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, 'Attn compute timed out (300s)'
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '')[-800:]
        return None, f'Attn compute failed (exit {proc.returncode}):\n{tail}'

    # Prefer rendering from the npz we just cached under best/final.
    image, info = render_attn_heatmap(str(path), ckpt, stem, joint)
    if image is None and out_jpg.is_file():
        from PIL import Image as PILImage

        return PILImage.open(out_jpg).convert('RGB'), f'ondemand jpeg  joint={joint}'
    if image is not None:
        return image, f'computed via {python}\n{info}'
    return None, f'Compute finished but render failed.\n{proc.stdout[-400:]}'


def build_app() -> gr.Blocks:
    with gr.Blocks(title='Training dashboard') as demo:
        gr.Markdown(
            '# Training dashboard\n'
            'Pose tabs: configure runs, live metrics, pause/stop, scrub val preds.  \n'
            'Detector tab: Ultralytics bbox runs under `training/detector_runs`.'
        )

        with gr.Tab('Config'):
            config_picker = gr.Dropdown(
                choices=list_config_choices(),
                value=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else None,
                label='Config file',
                allow_custom_value=True,
            )
            config_editor = gr.Textbox(
                value=load_config_text(str(DEFAULT_CONFIG)),
                label='YAML',
                lines=28,
                max_lines=40,
            )
            save_path = gr.Textbox(
                value=str(DEFAULT_CONFIG),
                label='Save path',
            )
            with gr.Row():
                reload_btn = gr.Button('Reload')
                save_btn = gr.Button('Save')
                save_as_btn = gr.Button('Save as path')
            workers = gr.Number(value=4, precision=0, label='DataLoader workers')
            init_checkpoint = gr.Textbox(
                value='',
                label='Init checkpoint (.pt, optional)',
                placeholder='training/pose_estimator_runs/.../final.pt',
            )
            start_btn = gr.Button('Start training', variant='primary')
            start_status = gr.Textbox(label='Start status', interactive=False)

            def _reload(path):
                return load_config_text(path), path

            # api_name=False: Gradio 4.44 + Image/Gallery event schemas crash get_api_info.
            _no_api = dict(api_name=False)

            reload_btn.click(
                _reload, inputs=[config_picker], outputs=[config_editor, save_path], **_no_api
            )
            config_picker.change(
                _reload, inputs=[config_picker], outputs=[config_editor, save_path], **_no_api
            )
            save_btn.click(
                lambda path, text: save_config_text(path, text),
                inputs=[save_path, config_editor],
                outputs=[start_status],
                **_no_api,
            )
            save_as_btn.click(
                lambda path, text: save_config_text(path, text),
                inputs=[save_path, config_editor],
                outputs=[start_status],
                **_no_api,
            )
            start_btn.click(
                start_training,
                inputs=[save_path, workers, init_checkpoint],
                outputs=[start_status],
                **_no_api,
            )

        with gr.Tab('Live run'):
            run_dd = gr.Dropdown(
                choices=list_run_dirs(),
                label='Run directory',
                allow_custom_value=True,
            )
            with gr.Row():
                refresh_btn = gr.Button('Refresh')
                pause_btn = gr.Button('Pause (after epoch)')
                resume_btn = gr.Button('Resume')
                stop_btn = gr.Button('Stop (after epoch)', variant='primary')
                kill_btn = gr.Button('Force kill dashboard job')
            status_md = gr.Markdown('Select a run')
            plot = gr.Image(type='pil', label='Metrics')
            plot_parts = gr.Image(type='pil', label='OKS by body part')
            plot_pct = gr.Image(type='pil', label='OKS by percentile band')
            control_status = gr.Textbox(label='Control', interactive=False)
            log_view = gr.Textbox(label='log.csv tail', lines=18, max_lines=18)
            _no_api = dict(api_name=False)
            live_outs = [plot, plot_parts, plot_pct, status_md, log_view, run_dd]

            refresh_btn.click(
                refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api
            )
            pause_btn.click(
                lambda d: send_control(d, 'pause'),
                inputs=[run_dd],
                outputs=[control_status],
                **_no_api,
            ).then(refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api)
            resume_btn.click(
                lambda d: send_control(d, 'resume'),
                inputs=[run_dd],
                outputs=[control_status],
                **_no_api,
            ).then(refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api)
            stop_btn.click(
                lambda d: send_control(d, 'stop'),
                inputs=[run_dd],
                outputs=[control_status],
                **_no_api,
            ).then(refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api)
            kill_btn.click(force_kill_active, outputs=[control_status], **_no_api)
            demo.load(refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api)
            if hasattr(gr, 'Timer'):
                timer = gr.Timer(value=5)
                timer.tick(refresh_live, inputs=[run_dd], outputs=live_outs, **_no_api)

        with gr.Tab('Epoch browser'):
            gr.Markdown(
                'Pick a **frame** once, then scrub **epochs** — keypoints are an SVG '
                'layer on top of the same photo crop.'
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    run_dd2 = gr.Dropdown(
                        choices=list_run_dirs(),
                        label='Run directory',
                        allow_custom_value=True,
                    )
                    sort_by = gr.Radio(
                        ['loss', '1-oks', 'task'],
                        value='loss',
                        label='Order samples by (latest epoch)',
                    )
                    with gr.Row():
                        refresh_runs_btn = gr.Button('Refresh runs')
                        load_browser_btn = gr.Button('Load frames / epochs', variant='primary')
                    frame_dd = gr.Dropdown(
                        choices=[],
                        label='Frame (sticky across epochs)',
                        allow_custom_value=False,
                    )
                    epoch_slider = gr.Slider(
                        minimum=0,
                        maximum=0,
                        step=1,
                        value=0,
                        label='Epoch',
                        interactive=False,
                    )
                    with gr.Row():
                        prev_epoch_btn = gr.Button('◀ Prev epoch')
                        next_epoch_btn = gr.Button('Next epoch ▶')
                    browser_status = gr.Textbox(
                        label='Browser status',
                        interactive=False,
                        value='Select a run, then Load frames / epochs.',
                    )
                with gr.Column(scale=2, min_width=420):
                    sample_viewer = gr.HTML(
                        value=empty_pose_viewer_html(),
                        show_label=False,
                    )
                    # Carries SVG to the browser; JS patches #pose-overlay-slot in place.
                    overlay_payload = gr.Textbox(visible=False)
                    sample_info = gr.Textbox(label='Info', interactive=False)
                    gr.Markdown(
                        '### Query cross-attn\n'
                        'Saved at **best/final** (watchlist + worst). '
                        '**Compute** runs one forward via `TRAIN_PYTHON` and caches the npz.'
                    )
                    with gr.Row():
                        attn_ckpt = gr.Radio(
                            ['best', 'final'],
                            value='best',
                            label='Attn checkpoint',
                        )
                        attn_joint = gr.Dropdown(
                            choices=_joint_name_choices(),
                            value='ankle_l',
                            label='Joint query',
                            allow_custom_value=False,
                        )
                    with gr.Row():
                        load_attn_btn = gr.Button('Load saved attn')
                        compute_attn_btn = gr.Button(
                            'Compute attn', variant='primary'
                        )
                    attn_image = gr.Image(
                        type='pil',
                        format='jpeg',
                        label='Cross-attn heatmap',
                        height=480,
                    )
                    attn_info = gr.Textbox(label='Attn info', interactive=False)
            sticky_stem = gr.State(None)
            _no_api = dict(api_name=False)

            def _load_browser(run, sort, stem):
                # Dropdown choices are snapshotted at page load — refresh + prefer active run.
                run = _preferred_browser_run(run)
                run_update = _browser_run_dropdown(run)
                frames, slider, status, preferred = load_epoch_browser(run, sort, stem)
                epochs = _epoch_list(run) if run else []
                epoch_val = epochs[-1] if epochs else 0
                html, info = render_frame_viewer(run, epoch_val, preferred)
                return run_update, frames, slider, status, preferred, html, info

            def _on_frame(run, epoch, stem):
                html, info = render_frame_viewer(run, epoch, stem)
                return stem, html, info

            def _on_epoch(run, epoch, stem):
                return render_overlay_payload(run, epoch, stem)

            def _wire_overlay_patch(event):
                # After Python returns SVG, patch the slot without replacing the <img>.
                return event.then(
                    fn=None,
                    inputs=[overlay_payload],
                    js=_PATCH_OVERLAY_JS,
                    **_no_api,
                )

            _browser_load_outs = [
                run_dd2,
                frame_dd,
                epoch_slider,
                browser_status,
                sticky_stem,
                sample_viewer,
                sample_info,
            ]
            refresh_runs_btn.click(
                _load_browser,
                inputs=[run_dd2, sort_by, sticky_stem],
                outputs=_browser_load_outs,
                **_no_api,
            )
            load_browser_btn.click(
                _load_browser,
                inputs=[run_dd2, sort_by, sticky_stem],
                outputs=_browser_load_outs,
                **_no_api,
            )
            demo.load(
                _load_browser,
                inputs=[run_dd2, sort_by, sticky_stem],
                outputs=_browser_load_outs,
                **_no_api,
            )
            run_dd2.change(
                _load_browser,
                inputs=[run_dd2, sort_by, sticky_stem],
                outputs=_browser_load_outs,
                **_no_api,
            )
            sort_by.change(
                _load_browser,
                inputs=[run_dd2, sort_by, sticky_stem],
                outputs=_browser_load_outs,
                **_no_api,
            )
            frame_dd.change(
                _on_frame,
                inputs=[run_dd2, epoch_slider, frame_dd],
                outputs=[sticky_stem, sample_viewer, sample_info],
                **_no_api,
            )
            _wire_overlay_patch(
                epoch_slider.release(
                    _on_epoch,
                    inputs=[run_dd2, epoch_slider, frame_dd],
                    outputs=[overlay_payload, sample_info],
                    **_no_api,
                )
            )
            _wire_overlay_patch(
                epoch_slider.change(
                    _on_epoch,
                    inputs=[run_dd2, epoch_slider, frame_dd],
                    outputs=[overlay_payload, sample_info],
                    **_no_api,
                )
            )
            _wire_overlay_patch(
                prev_epoch_btn.click(
                    lambda run, epoch, stem: step_epoch(run, epoch, stem, -1),
                    inputs=[run_dd2, epoch_slider, frame_dd],
                    outputs=[epoch_slider, overlay_payload, sample_info],
                    **_no_api,
                )
            )
            _wire_overlay_patch(
                next_epoch_btn.click(
                    lambda run, epoch, stem: step_epoch(run, epoch, stem, 1),
                    inputs=[run_dd2, epoch_slider, frame_dd],
                    outputs=[epoch_slider, overlay_payload, sample_info],
                    **_no_api,
                )
            )
            load_attn_btn.click(
                render_attn_heatmap,
                inputs=[run_dd2, attn_ckpt, frame_dd, attn_joint],
                outputs=[attn_image, attn_info],
                **_no_api,
            )
            compute_attn_btn.click(
                compute_attn_heatmap,
                inputs=[run_dd2, attn_ckpt, frame_dd, attn_joint],
                outputs=[attn_image, attn_info],
                **_no_api,
            )
            attn_joint.change(
                render_attn_heatmap,
                inputs=[run_dd2, attn_ckpt, frame_dd, attn_joint],
                outputs=[attn_image, attn_info],
                **_no_api,
            )

        with gr.Tab('Compare'):
            run_a = gr.Dropdown(choices=list_run_dirs(), label='Run A', allow_custom_value=True)
            run_b = gr.Dropdown(choices=list_run_dirs(), label='Run B', allow_custom_value=True)
            compare_btn = gr.Button('Compare')
            compare_plot = gr.Image(type='pil', label='Compare')
            compare_btn.click(
                compare_runs,
                inputs=[run_a, run_b],
                outputs=[compare_plot],
                api_name=False,
            )

        with gr.Tab('Detector'):
            gr.Markdown(
                'YOLO detector runs (`training/detector_runs`). Metrics come from '
                '`results.csv`; previews from Ultralytics batch / curve plots.'
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    det_config_picker = gr.Dropdown(
                        choices=list_detector_config_choices(),
                        value=(
                            str(DEFAULT_DETECTOR_CONFIG)
                            if DEFAULT_DETECTOR_CONFIG.is_file()
                            else None
                        ),
                        label='Detector config',
                        allow_custom_value=True,
                    )
                    det_config_editor = gr.Textbox(
                        value=load_config_text(
                            str(DEFAULT_DETECTOR_CONFIG)
                            if DEFAULT_DETECTOR_CONFIG.is_file()
                            else None
                        ),
                        label='YAML',
                        lines=18,
                        max_lines=28,
                    )
                    det_save_path = gr.Textbox(
                        value=str(DEFAULT_DETECTOR_CONFIG),
                        label='Save path',
                    )
                    with gr.Row():
                        det_reload_btn = gr.Button('Reload')
                        det_save_btn = gr.Button('Save')
                    det_device = gr.Textbox(value='0', label='Device')
                    det_start_btn = gr.Button('Start detector training', variant='primary')
                    det_kill_btn = gr.Button('Force kill detector job')
                    det_start_status = gr.Textbox(label='Start / kill status', interactive=False)
                with gr.Column(scale=2, min_width=420):
                    det_run_dd = gr.Dropdown(
                        choices=list_detector_run_dirs(),
                        label='Detector run',
                        allow_custom_value=True,
                    )
                    with gr.Row():
                        det_refresh_btn = gr.Button('Refresh', variant='primary')
                    det_status_md = gr.Markdown('Select a detector run')
                    det_plot = gr.Image(type='pil', label='Detector metrics')
                    det_union_plot = gr.Image(
                        type='pil',
                        label='Val union vs separate breakdown (best ckpt)',
                    )
                    det_log = gr.Textbox(label='Train log / results tail', lines=14, max_lines=14)
                    det_gallery = gr.Gallery(
                        label='Run previews (labels / batches / curves)',
                        columns=3,
                        height=420,
                        object_fit='contain',
                    )
                    with gr.Row():
                        det_run_a = gr.Dropdown(
                            choices=list_detector_run_dirs(),
                            label='Compare A',
                            allow_custom_value=True,
                        )
                        det_run_b = gr.Dropdown(
                            choices=list_detector_run_dirs(),
                            label='Compare B',
                            allow_custom_value=True,
                        )
                    det_compare_btn = gr.Button('Compare detector runs')
                    det_compare_plot = gr.Image(type='pil', label='Detector compare')

            _no_api = dict(api_name=False)

            def _det_reload(path):
                return load_config_text(path), path

            det_reload_btn.click(
                _det_reload,
                inputs=[det_config_picker],
                outputs=[det_config_editor, det_save_path],
                **_no_api,
            )
            det_config_picker.change(
                _det_reload,
                inputs=[det_config_picker],
                outputs=[det_config_editor, det_save_path],
                **_no_api,
            )
            det_save_btn.click(
                lambda path, text: save_config_text(path, text),
                inputs=[det_save_path, det_config_editor],
                outputs=[det_start_status],
                **_no_api,
            )
            det_start_btn.click(
                start_detector_training,
                inputs=[det_save_path, det_device],
                outputs=[det_start_status],
                **_no_api,
            ).then(
                refresh_detector_live,
                inputs=[det_run_dd],
                outputs=[
                    det_plot,
                    det_union_plot,
                    det_status_md,
                    det_log,
                    det_gallery,
                    det_run_dd,
                ],
                **_no_api,
            )
            det_kill_btn.click(
                force_kill_active_detector,
                outputs=[det_start_status],
                **_no_api,
            )
            det_live_outs = [
                det_plot,
                det_union_plot,
                det_status_md,
                det_log,
                det_gallery,
                det_run_dd,
            ]
            det_refresh_btn.click(
                refresh_detector_live,
                inputs=[det_run_dd],
                outputs=det_live_outs,
                **_no_api,
            )
            det_run_dd.change(
                refresh_detector_live,
                inputs=[det_run_dd],
                outputs=det_live_outs,
                **_no_api,
            )
            demo.load(
                refresh_detector_live,
                inputs=[det_run_dd],
                outputs=det_live_outs,
                **_no_api,
            )
            if hasattr(gr, 'Timer'):
                det_timer = gr.Timer(value=10)
                det_timer.tick(
                    refresh_detector_live,
                    inputs=[det_run_dd],
                    outputs=det_live_outs,
                    **_no_api,
                )
            det_compare_btn.click(
                compare_detector_runs,
                inputs=[det_run_a, det_run_b],
                outputs=[det_compare_plot],
                **_no_api,
            )

        with gr.Tab('Detector browser'):
            gr.Markdown(
                'Val-frame browser for detector runs. **Green = GT**, **orange = preds (P)**.  \n'
                'Orange boxes appear only after **Cache preds** for that run/checkpoint. '
                '`bbox_union_iou0.5_102` / `best` is already cached; newer runs (e.g. 0.75) '
                'need a cache click. Prefer CPU while a train job holds the GPU.'
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    db_run = gr.Dropdown(
                        choices=list_detector_run_dirs(),
                        label='Detector run',
                        allow_custom_value=True,
                    )
                    db_ckpt = gr.Dropdown(
                        choices=['best', 'last'],
                        value='best',
                        label='Checkpoint (best / last / epochN)',
                        allow_custom_value=True,
                    )
                    db_sort = gr.Radio(
                        ['task', '1-iou', 'fn'],
                        value='task',
                        label='Order frames',
                    )
                    with gr.Row():
                        db_load_btn = gr.Button('Load frames', variant='primary')
                        db_refresh_btn = gr.Button('Refresh ckpts')
                    db_frame = gr.Dropdown(
                        choices=[],
                        label='Val frame (sticky across checkpoints)',
                        allow_custom_value=False,
                    )
                    db_cache_device = gr.Textbox(
                        value='cpu',
                        label='Cache device (cpu recommended while training)',
                    )
                    db_cache_btn = gr.Button('Cache preds for checkpoint')
                    db_status = gr.Textbox(
                        label='Browser status',
                        interactive=False,
                        value='Select a run, then Load frames.',
                    )
                    db_cache_status = gr.Textbox(label='Cache status', interactive=False)
                with gr.Column(scale=2, min_width=420):
                    db_image = gr.Image(
                        type='pil',
                        format='jpeg',
                        label='GT (green) vs pred (orange)',
                        height=640,
                    )
                    db_info = gr.Textbox(label='Frame info', interactive=False)
            db_stem = gr.State(None)
            _no_api = dict(api_name=False)

            def _db_load(run, ckpt, sort, stem):
                return load_detector_browser(run, ckpt, sort, stem)

            db_outs = [
                db_run,
                db_ckpt,
                db_frame,
                db_status,
                db_stem,
                db_image,
                db_info,
            ]
            db_load_btn.click(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            db_refresh_btn.click(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            demo.load(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            db_run.change(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            db_sort.change(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            db_ckpt.change(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )
            db_frame.change(
                lambda run, ckpt, stem: (
                    stem,
                    *render_detector_frame(run, ckpt, stem),
                ),
                inputs=[db_run, db_ckpt, db_frame],
                outputs=[db_stem, db_image, db_info],
                **_no_api,
            )
            db_cache_btn.click(
                cache_detector_browser_preds,
                inputs=[db_run, db_ckpt, db_cache_device],
                outputs=[db_cache_status],
                **_no_api,
            ).then(
                _db_load,
                inputs=[db_run, db_ckpt, db_sort, db_stem],
                outputs=db_outs,
                **_no_api,
            )

    return demo


def main():
    # Proxy env often makes Gradio's localhost self-check fail; keep local loopback direct.
    for key in ('NO_PROXY', 'no_proxy'):
        existing = os.environ.get(key, '')
        extras = '127.0.0.1,localhost'
        if existing:
            os.environ[key] = f'{existing},{extras}'
        else:
            os.environ[key] = extras
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

    demo = build_app()
    demo.queue()
    demo.launch(
        server_name='127.0.0.1',
        server_port=int(os.environ.get('DASHBOARD_PORT', '7860')),
        show_error=True,
        show_api=False,
        inbrowser=False,
    )


if __name__ == '__main__':
    main()
