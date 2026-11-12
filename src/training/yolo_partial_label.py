"""Partial-label helpers for Ultralytics YOLO detection.

Sparse dance boxes only cover couples used for keypoint labeling. Stock YOLO
treats the rest of the frame as background and punishes detecting other people.

Supported masks (can combine):

- ``mask_outside_gt``: zero **cls** loss on anchors whose centers fall outside
  any (optionally dilated) GT box. Crowd outside your boxes is ignored, not
  “must be empty.” Box/DFL stay assigner-only (already GT-matched).
- ``full_cls_task_ids``: task ids where labels are complete enough that outside
  GT really is background (or safe crowd-as-bg). Those images skip the outside
  mask and use stock cls. Parsed from ``im_file`` names ``task_<id>_frame_...``.
- ``positive_only_cls``: also drop background cls *inside* GT regions (only
  supervise matched positives). Stricter; usually not needed if outside is masked.

``neg_weight`` optionally keeps a little background cls (global or inside-GT).

Note: ``YOLO.train()`` rebuilds a fresh ``DetectionModel`` and only loads weights,
so the patch is re-applied via an Ultralytics callback after trainer setup.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

_TASK_FROM_NAME = re.compile(r'task_(\d+)_frame_', re.IGNORECASE)


def load_full_cls_task_ids(completeness_yaml: Path | str) -> set[str]:
    """Union of tagged task ids where full background cls is OK."""
    import yaml

    path = Path(completeness_yaml)
    with path.open(encoding='utf-8') as handle:
        doc = yaml.safe_load(handle) or {}
    ids: set[str] = set()
    for key in (
        'perfectly_safe',
        'almost_safe',
        'crowd_bg_ok',
        'background_labeled',
    ):
        for raw in doc.get(key) or []:
            text = str(raw).strip()
            if text.startswith('task_'):
                text = text[len('task_') :]
            elif text.startswith('task-'):
                text = text[len('task-') :]
            if text.isdigit():
                ids.add(str(int(text)))
            else:
                ids.add(text)
    return ids


def task_id_from_path(path: str | Path) -> str | None:
    match = _TASK_FROM_NAME.search(Path(path).name)
    return match.group(1) if match else None


def _batch_allow_full_cls(
    batch: dict[str, Any],
    full_cls_task_ids: set[str],
    *,
    batch_size: int,
    device: Any,
    dtype: Any,
) -> Any:
    """(bs, 1, 1) float mask — 1 where this image should use full cls."""
    import torch

    files = batch.get('im_file') or batch.get('im_files') or ()
    allow = []
    for i in range(batch_size):
        path = files[i] if i < len(files) else None
        tid = task_id_from_path(path) if path is not None else None
        allow.append(1.0 if tid is not None and tid in full_cls_task_ids else 0.0)
    return torch.tensor(allow, device=device, dtype=dtype).view(batch_size, 1, 1)


def _anchors_inside_gt(
    anchor_xy: Any,
    gt_bboxes: Any,
    mask_gt: Any,
    *,
    dilate: float,
) -> Any:
    """Return (bs, n_anchors, 1) bool — anchor center inside any dilated GT."""
    # anchor_xy: (n_anchors, 2) pixel xy; gt_bboxes: (bs, n_max, 4) xyxy
    x = anchor_xy[:, 0].view(1, -1, 1)  # (1, A, 1)
    y = anchor_xy[:, 1].view(1, -1, 1)
    x1, y1, x2, y2 = gt_bboxes.unbind(-1)  # (bs, n_max)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = (x2 - x1) * 0.5 * float(dilate)
    half_h = (y2 - y1) * 0.5 * float(dilate)
    x1d = (cx - half_w).unsqueeze(1)  # (bs, 1, n_max)
    y1d = (cy - half_h).unsqueeze(1)
    x2d = (cx + half_w).unsqueeze(1)
    y2d = (cy + half_h).unsqueeze(1)
    inside = (x >= x1d) & (x <= x2d) & (y >= y1d) & (y <= y2d)  # (bs, A, n_max)
    valid = mask_gt.view(gt_bboxes.shape[0], 1, gt_bboxes.shape[1]).bool()
    inside = inside & valid
    return inside.any(dim=-1, keepdim=True)  # (bs, A, 1)


def patch_detection_criterion_partial_labels(
    criterion: Any,
    *,
    mask_outside_gt: bool = True,
    gt_mask_dilate: float = 1.15,
    positive_only_cls: bool = False,
    neg_weight: float = 0.0,
    full_cls_task_ids: Iterable[str] | None = None,
    side_cls_weight: float = 0.5,
    side_box_weight: float = 0.2,
    side_class_id: int = 1,
) -> Any:
    """Patch cls/box loss for sparse labels and main vs side tiers.

    Side GTs (class ``side_class_id``) get lower box weight than cls weight so
    rough background boxes push recall more than localization precision, and
    both weights stay below main (1.0).
    """
    allow_ids = {str(t) for t in (full_cls_task_ids or ())}
    new_opts = {
        'mask_outside_gt': bool(mask_outside_gt),
        'gt_mask_dilate': float(gt_mask_dilate),
        'positive_only_cls': bool(positive_only_cls),
        'neg_weight': float(neg_weight),
        'full_cls_task_ids': sorted(allow_ids),
        'side_cls_weight': float(side_cls_weight),
        'side_box_weight': float(side_box_weight),
        'side_class_id': int(side_class_id),
    }
    prev = getattr(criterion, '_partial_label_opts', None)
    if getattr(criterion, '_partial_label_patched', False) and prev == new_opts:
        return criterion

    import torch
    from ultralytics.utils.tal import make_anchors

    def get_assigned_targets_and_loss_masked(
        preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ):
        loss = torch.zeros(3, device=criterion.device)
        pred_distri, pred_scores = (
            preds['boxes'].permute(0, 2, 1).contiguous(),
            preds['scores'].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds['feats'], criterion.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds['feats'][0].shape[2:], device=criterion.device, dtype=dtype)
            * criterion.stride[0]
        )

        targets = torch.cat(
            (batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']),
            1,
        )
        targets = criterion.preprocess(
            targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # Per-anchor side vs main weights from matched GT class.
        side_cls_w = float(side_cls_weight)
        side_box_w = float(side_box_weight)
        gt_cls = gt_labels.squeeze(-1)  # (bs, n_max)
        # target_gt_idx: (bs, anchors); invalid when ~fg_mask
        matched_cls = gt_cls.gather(1, target_gt_idx.clamp(min=0))
        anchor_side = (matched_cls == int(side_class_id)) & fg_mask
        cls_scale = torch.where(
            anchor_side,
            torch.full_like(matched_cls, side_cls_w, dtype=dtype),
            torch.ones_like(matched_cls, dtype=dtype),
        )
        box_scale = torch.where(
            anchor_side,
            torch.full_like(matched_cls, side_box_w, dtype=dtype),
            torch.ones_like(matched_cls, dtype=dtype),
        )
        # Non-fg anchors keep scale 1 for mask/neg paths.
        cls_scale = torch.where(fg_mask, cls_scale, torch.ones_like(cls_scale))
        box_scale = torch.where(fg_mask, box_scale, torch.ones_like(box_scale))

        target_scores_cls = target_scores * cls_scale.unsqueeze(-1)
        target_scores_box = target_scores * box_scale.unsqueeze(-1)
        target_scores_sum = target_scores_cls.sum().clamp_(min=1)
        target_scores_box_sum = target_scores_box.sum().clamp_(min=1)

        bce_loss = criterion.bce(pred_scores, target_scores_cls.to(dtype))
        if criterion.class_weights is not None:
            bce_loss *= criterion.class_weights

        weight = torch.ones_like(bce_loss)
        pos = target_scores_cls.sum(dim=-1, keepdim=True) > 0

        if mask_outside_gt:
            anchor_xy = anchor_points * stride_tensor
            inside = _anchors_inside_gt(
                anchor_xy, gt_bboxes, mask_gt, dilate=gt_mask_dilate
            )
            weight = weight * inside.to(dtype=weight.dtype)
            if allow_ids:
                allow = _batch_allow_full_cls(
                    batch,
                    allow_ids,
                    batch_size=batch_size,
                    device=criterion.device,
                    dtype=weight.dtype,
                )
                weight = torch.where(allow > 0, torch.ones_like(weight), weight)

        if positive_only_cls:
            if neg_weight <= 0:
                weight = weight * pos.to(dtype=weight.dtype)
            else:
                bg = (~pos) & (weight > 0)
                weight = torch.where(
                    pos,
                    weight,
                    torch.where(bg, weight * float(neg_weight), weight),
                )

        bce_loss = bce_loss * weight
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = criterion.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores_box,
            target_scores_box_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        loss[0] *= criterion.hyp.box
        loss[1] *= criterion.hyp.cls
        loss[2] *= criterion.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(
                zip(
                    criterion.loss_names,
                    [loss[0].detach(), loss[1].detach(), loss[2].detach()],
                )
            ),
        )

    criterion.get_assigned_targets_and_loss = get_assigned_targets_and_loss_masked
    criterion._partial_label_patched = True
    criterion._partial_label_opts = new_opts
    return criterion


def _unwrap_detection_model(model: Any) -> Any:
    inner = model
    while hasattr(inner, 'module') and not hasattr(inner, 'init_criterion'):
        inner = inner.module
    return inner


def _apply_partial_label_to_model(
    model: Any,
    *,
    mask_outside_gt: bool,
    gt_mask_dilate: float,
    positive_only_cls: bool,
    neg_weight: float,
    full_cls_task_ids: set[str],
    side_cls_weight: float,
    side_box_weight: float,
    side_class_id: int,
) -> None:
    inner = _unwrap_detection_model(model)
    if not hasattr(inner, 'init_criterion'):
        raise RuntimeError('Could not find DetectionModel.init_criterion to patch')

    if not getattr(inner, '_partial_label_init_wrapped', False):
        inner._partial_label_original_init = inner.init_criterion
        inner._partial_label_init_wrapped = True
    original_init = inner._partial_label_original_init

    def init_criterion_masked():
        criterion = original_init()
        return patch_detection_criterion_partial_labels(
            criterion,
            mask_outside_gt=mask_outside_gt,
            gt_mask_dilate=gt_mask_dilate,
            positive_only_cls=positive_only_cls,
            neg_weight=neg_weight,
            full_cls_task_ids=full_cls_task_ids,
            side_cls_weight=side_cls_weight,
            side_box_weight=side_box_weight,
            side_class_id=side_class_id,
        )

    inner.init_criterion = init_criterion_masked
    if getattr(inner, 'criterion', None) is not None:
        patch_detection_criterion_partial_labels(
            inner.criterion,
            mask_outside_gt=mask_outside_gt,
            gt_mask_dilate=gt_mask_dilate,
            positive_only_cls=positive_only_cls,
            neg_weight=neg_weight,
            full_cls_task_ids=full_cls_task_ids,
            side_cls_weight=side_cls_weight,
            side_box_weight=side_box_weight,
            side_class_id=side_class_id,
        )


def enable_partial_label_loss(
    yolo_model: Any,
    *,
    mask_outside_gt: bool = True,
    gt_mask_dilate: float = 1.15,
    positive_only_cls: bool = False,
    neg_weight: float = 0.0,
    full_cls_task_ids: Iterable[str] | None = None,
    side_cls_weight: float = 0.5,
    side_box_weight: float = 0.2,
    side_class_id: int = 1,
) -> None:
    """Patch DetectionModel loss; re-apply after YOLO.train() rebuilds the model."""
    allow_ids = {str(t) for t in (full_cls_task_ids or ())}
    opts = dict(
        mask_outside_gt=mask_outside_gt,
        gt_mask_dilate=gt_mask_dilate,
        positive_only_cls=positive_only_cls,
        neg_weight=neg_weight,
        full_cls_task_ids=allow_ids,
        side_cls_weight=side_cls_weight,
        side_box_weight=side_box_weight,
        side_class_id=side_class_id,
    )

    if getattr(yolo_model, 'model', None) is not None:
        try:
            _apply_partial_label_to_model(yolo_model.model, **opts)
        except RuntimeError:
            pass

    def _on_trainer_model_ready(trainer: Any) -> None:
        _apply_partial_label_to_model(trainer.model, **opts)
        model = _unwrap_detection_model(trainer.model)
        model.criterion = model.init_criterion()
        print(
            'Partial-label mask applied to trainer model: '
            f'mask_outside_gt={mask_outside_gt} dilate={gt_mask_dilate} '
            f'side_cls={side_cls_weight} side_box={side_box_weight} '
            f'full_cls_tasks={sorted(allow_ids, key=lambda x: int(x) if x.isdigit() else x)}'
        )

    yolo_model.add_callback('on_train_start', _on_trainer_model_ready)
    print(
        'Partial-label mask scheduled (callback on_train_start): '
        f'mask_outside_gt={mask_outside_gt} dilate={gt_mask_dilate} '
        f'side_cls={side_cls_weight} side_box={side_box_weight} '
        f'full_cls_tasks={sorted(allow_ids, key=lambda x: int(x) if x.isdigit() else x)}'
    )


def enable_positive_only_cls(yolo_model: Any, *, neg_weight: float = 0.0) -> None:
    enable_partial_label_loss(
        yolo_model,
        mask_outside_gt=False,
        positive_only_cls=True,
        neg_weight=neg_weight,
    )
