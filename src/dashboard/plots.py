"""Chart helpers for the dashboard (PIL images; avoids Gradio Plot schema bugs)."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.util.oks_breakdown import BODY_PART_HISTORY_ORDER, OKS_PERCENTILE_BANDS


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _empty_axes_message(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('epoch')


def history_figure(rows: list[dict[str, Any]]) -> Image.Image:
    """Build a multi-panel chart from history.jsonl rows."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    if not rows:
        for ax in axes.flat:
            ax.set_title('No history yet')
            ax.set_xlabel('epoch')
        return _fig_to_pil(fig)

    epochs = [int(r['epoch']) for r in rows]
    train = [float(r.get('train_loss', float('nan'))) for r in rows]
    val = [float(r.get('val_loss', float('nan'))) for r in rows]
    dance = [
        float('nan') if r.get('train_loss_dance') is None else float(r['train_loss_dance'])
        for r in rows
    ]
    coco = [
        float('nan') if r.get('train_loss_coco') is None else float(r['train_loss_coco'])
        for r in rows
    ]
    ax = axes[0, 0]
    ax.plot(epochs, train, label='train', marker='o', markersize=3)
    ax.plot(epochs, val, label='val', marker='o', markersize=3)
    if any(v == v for v in dance):
        ax.plot(
            epochs,
            dance,
            label='train dance (simcc)',
            marker='o',
            markersize=2,
            linestyle='--',
        )
    if any(v == v for v in coco):
        ax.plot(
            epochs,
            coco,
            label='train coco (simcc)',
            marker='o',
            markersize=2,
            linestyle='--',
        )
    ax.set_title('Loss')
    ax.set_xlabel('epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    components = ['simcc_kl', 'deep_supervision', 'bone', 'lr_swap', 'oks', 'chirality']
    for name in components:
        series = []
        for row in rows:
            comps = row.get('val_components') or {}
            series.append(float(comps.get(name, float('nan'))))
        if any(v == v for v in series):
            ax.plot(epochs, series, label=name, marker='o', markersize=2)
    ax.set_title('Val loss components')
    ax.set_xlabel('epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ap = [r.get('val_ap') for r in rows]
    pri = [r.get('val_priority_oks') for r in rows]
    if any(v is not None for v in ap):
        ax.plot(
            epochs,
            [float('nan') if v is None else float(v) for v in ap],
            label='AP',
            marker='o',
            markersize=3,
        )
    if any(v is not None for v in pri):
        ax.plot(
            epochs,
            [float('nan') if v is None else float(v) for v in pri],
            label='priority_OKS',
            marker='o',
            markersize=3,
        )
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    else:
        _empty_axes_message(
            ax, 'AP / priority_OKS not logged\n(checkpoint_metric=val_loss?)'
        )
    ax.set_title('Target metrics')
    ax.set_xlabel('epoch')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lrs = [float(r.get('lr', float('nan'))) for r in rows]
    ax.plot(epochs, lrs, marker='o', markersize=3, color='tab:purple')
    ax.set_title('Learning rate')
    ax.set_xlabel('epoch')
    ax.grid(True, alpha=0.3)
    return _fig_to_pil(fig)


def history_part_oks_figure(rows: list[dict[str, Any]]) -> Image.Image:
    """Separate chart: mean OKS by body part over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    if not rows:
        ax.set_title('OKS by body part')
        _empty_axes_message(ax, 'No history yet')
        return _fig_to_pil(fig)

    epochs = [int(r['epoch']) for r in rows]
    plotted = False
    for part in BODY_PART_HISTORY_ORDER:
        series = []
        for row in rows:
            parts = row.get('val_part_oks') or {}
            value = parts.get(part)
            series.append(float('nan') if value is None else float(value))
        if any(v == v for v in series):
            ax.plot(epochs, series, label=part, marker='o', markersize=3)
            plotted = True
    ax.set_title('Val mean OKS by body part')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mean OKS')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        _empty_axes_message(
            ax, 'No part OKS in history yet\n(backfill from val_preds if present)'
        )
    return _fig_to_pil(fig)


def history_oks_percentile_figure(rows: list[dict[str, Any]]) -> Image.Image:
    """Separate chart: mean OKS inside percentile bands over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    if not rows:
        ax.set_title('OKS by percentile band')
        _empty_axes_message(ax, 'No history yet')
        return _fig_to_pil(fig)

    epochs = [int(r['epoch']) for r in rows]
    plotted = False
    for name, lo, hi in OKS_PERCENTILE_BANDS:
        series = []
        for row in rows:
            bands = row.get('val_oks_percentile_bands') or {}
            value = bands.get(name)
            series.append(float('nan') if value is None else float(value))
        if any(v == v for v in series):
            ax.plot(
                epochs,
                series,
                label=f'{lo}–{hi}%',
                marker='o',
                markersize=3,
            )
            plotted = True
    ax.set_title('Val mean OKS by percentile band (low % = hardest people)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('mean OKS in band')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        _empty_axes_message(
            ax, 'No percentile OKS in history yet\n(backfill from val_preds if present)'
        )
    return _fig_to_pil(fig)


def compare_history_figure(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in (
        (axes[0], 'val_loss', 'Val loss'),
        (axes[1], 'train_loss', 'Train loss'),
    ):
        for rows, label in ((rows_a, label_a), (rows_b, label_b)):
            if not rows:
                continue
            epochs = [int(r['epoch']) for r in rows]
            vals = [float(r.get(key, float('nan'))) for r in rows]
            ax.plot(epochs, vals, label=label, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    return _fig_to_pil(fig)


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None or value == '':
            out.append(float('nan'))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float('nan'))
    return out


def detector_results_figure(rows: list[dict[str, Any]]) -> Image.Image:
    """Ultralytics results.csv → loss / mAP / P-R / LR panels."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    if not rows:
        for ax in axes.flat:
            ax.set_title('No detector results yet')
            ax.set_xlabel('epoch')
        return _fig_to_pil(fig)

    epochs = [int(float(r.get('epoch', i + 1))) for i, r in enumerate(rows)]

    ax = axes[0, 0]
    for key, label in (
        ('train/box_loss', 'train box'),
        ('train/cls_loss', 'train cls'),
        ('train/dfl_loss', 'train dfl'),
        ('val/box_loss', 'val box'),
        ('val/cls_loss', 'val cls'),
        ('val/dfl_loss', 'val dfl'),
    ):
        series = _series(rows, key)
        if any(v == v for v in series):
            ax.plot(epochs, series, label=label, marker='o', markersize=2)
    ax.set_title('Box / cls / DFL loss')
    ax.set_xlabel('epoch')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for key, label in (
        ('metrics/mAP50(B)', 'mAP50'),
        ('metrics/mAP50-95(B)', 'mAP50-95'),
    ):
        series = _series(rows, key)
        if any(v == v for v in series):
            ax.plot(epochs, series, label=label, marker='o', markersize=3)
    ax.set_title('mAP')
    ax.set_xlabel('epoch')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, label in (
        ('metrics/precision(B)', 'precision'),
        ('metrics/recall(B)', 'recall'),
    ):
        series = _series(rows, key)
        if any(v == v for v in series):
            ax.plot(epochs, series, label=label, marker='o', markersize=3)
    ax.set_title('Precision / recall')
    ax.set_xlabel('epoch')
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    plotted = False
    for key, label in (
        ('lr/pg0', 'lr pg0'),
        ('lr/pg1', 'lr pg1'),
        ('lr/pg2', 'lr pg2'),
    ):
        series = _series(rows, key)
        if any(v == v for v in series):
            ax.plot(epochs, series, label=label, marker='o', markersize=2)
            plotted = True
    ax.set_title('Learning rate')
    ax.set_xlabel('epoch')
    if plotted:
        ax.legend(fontsize=8)
    else:
        _empty_axes_message(ax, 'No lr columns in results.csv')
    ax.grid(True, alpha=0.3)
    return _fig_to_pil(fig)


def detector_union_breakdown_figure(summary: dict[str, Any] | None) -> Image.Image:
    """Bar charts: union vs separate vs unpaired on val (from pred cache)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    if not summary:
        for ax in axes.flat:
            ax.set_title('No union breakdown')
            _empty_axes_message(ax, 'Cache preds + box kinds first')
        return _fig_to_pil(fig)

    by_kind = summary.get('by_box_kind') or {}
    kinds = ['union', 'separate', 'unpaired']
    labels = []
    recalls = []
    mean_ious = []
    counts = []
    for kind in kinds:
        row = by_kind.get(kind) or {}
        n = int(row.get('n_gt') or 0)
        if n <= 0 and row.get('recall') is None:
            continue
        labels.append(f'{kind}\n(n={n})')
        recalls.append(
            float('nan') if row.get('recall') is None else float(row['recall'])
        )
        mean_ious.append(
            float('nan') if row.get('mean_iou') is None else float(row['mean_iou'])
        )
        counts.append(n)

    ax = axes[0]
    if labels:
        ax.bar(labels, recalls, color=['#2a9d8f', '#e9c46a', '#8d99ae'][: len(labels)])
        ax.set_ylim(0.0, 1.05)
    else:
        _empty_axes_message(ax, 'No GT boxes')
    ax.set_title('Recall@0.5 by GT box kind')
    ax.set_ylabel('recall')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    if labels:
        ax.bar(labels, mean_ious, color=['#2a9d8f', '#e9c46a', '#8d99ae'][: len(labels)])
        ax.set_ylim(0.0, 1.05)
    else:
        _empty_axes_message(ax, 'No GT boxes')
    ax.set_title('Mean best-IoU by GT box kind')
    ax.set_ylabel('mean IoU')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[2]
    cover = summary.get('union_cover') or {}
    cover_keys = ['one_cover', 'two_cover', 'under_cover', 'miss']
    cover_vals = [int(cover.get(k) or 0) for k in cover_keys]
    if sum(cover_vals) > 0:
        ax.bar(cover_keys, cover_vals, color='#457b9d')
        ax.set_title(
            f"Union GT coverage (τ={summary.get('union_iou_threshold')}, "
            f"n={sum(cover_vals)})"
        )
        ax.tick_params(axis='x', rotation=20)
    else:
        _empty_axes_message(ax, 'No union GT boxes in val')
        ax.set_title('Union GT coverage')
    ax.set_ylabel('count')
    ax.grid(True, axis='y', alpha=0.3)

    tau = summary.get('union_iou_threshold')
    ckpt = summary.get('checkpoint')
    n_frames = summary.get('n_val_frames_used')
    fig.suptitle(
        f"Val union vs separate — ckpt={ckpt}  frames={n_frames}  policy τ={tau}",
        fontsize=11,
    )
    return _fig_to_pil(fig)


def compare_detector_results_figure(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    label_a: str,
    label_b: str,
) -> Image.Image:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        (axes[0], 'metrics/mAP50-95(B)', 'mAP50-95'),
        (axes[1], 'metrics/mAP50(B)', 'mAP50'),
        (axes[2], 'val/box_loss', 'Val box loss'),
    )
    for ax, key, title in panels:
        for rows, label in ((rows_a, label_a), (rows_b, label_b)):
            if not rows:
                continue
            epochs = [int(float(r.get('epoch', i + 1))) for i, r in enumerate(rows)]
            ax.plot(epochs, _series(rows, key), label=label, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel('epoch')
        if 'mAP' in title:
            ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    return _fig_to_pil(fig)
