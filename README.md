# Dance Model Training

Code for training top-down human pose estimation on dance videos, plus person/couple bbox detection fine-tuning.

## Current State

- CVAT annotations are processed into cleaned, task-specific training annotations.
- Bounding boxes are paired with skeletons using task-local `person_id` values.
- The shared image store uses task/frame image names.
- Pose training uses DINOv3 with a SimCC keypoint head and 23 selected keypoints.
- Detection fine-tuning (this branch) exports cleaned bbox-only labels and trains Ultralytics YOLO.

## Pose training

Pose crops still come from GT boxes in `cleaned_annotations.xml` (skeleton-gated). See `config_pose.yaml` / `config_pose_template.yaml` and `src/training/train.py`.

### Training observability dashboard

Local Gradio UI to edit configs, start/pause/stop runs, watch live loss curves, and scrub val predictions by epoch.

```bash
# One-time: dedicated lightweight venv (avoids Gradio vs training-env conflicts)
python -m venv .venv-dashboard
.venv-dashboard/bin/pip install -r requirements-dashboard.txt

# Launch UI (http://127.0.0.1:7860)
cd /path/to/cv_model_training
PYTHONPATH=. TRAIN_PYTHON=/home/arodabough/anaconda3/envs/dino/bin/python \
  .venv-dashboard/bin/python -m src.dashboard.app
```

`TRAIN_PYTHON` should point at the env that has torch / DINOv3 (e.g. `dino`). The dashboard process itself stays CPU-only.

Each run folder under `runs/pose_estimator/YYYY_MM_DD_NNN/` now also writes:
- `history.jsonl` — per-epoch train/val loss, components, optional AP
- `status.json` / `control.json` — live state + pause/resume/stop
- `val_preds/epoch_XXX.npz` — all val pred/gt coords + loss/OKS
- `val_overlays/epoch_XXX/` — worst-N (+ watchlist) JPG overlays
- `val_samples.json` — full per-sample records at end of run

Config knobs (under `training:`): `save_val_preds_every_n_epochs`, `epoch_overlay_top_n`, `watchlist_frames`.

## Bbox detection training

Goal: teach a detector your cleaned box style — limb-inclusive, occlusion-aware extension — and **union partner boxes when close** so closed-hold yields one couple-covering region (or two accurate boxes when far apart). Either outcome is compatible with a later couple pose head on a union crop.

### Label policy

Single class `person`:

1. Keep only effectively cleaned bbox shapes (`frame_cleaned` or `cleaned` + `cleaned_completion` stride), same rules as `dataset_parser.py`. Skeletons are **not** required.
2. Pair partners by `couple_id` when present; else lead/follow + greedy max IoU.
3. If pair IoU ≥ threshold (default `0.3`) → emit one **union** box; else emit both boxes.
4. Ablation: `--always-separate` skips unions.

Pose `cleaned_annotations.xml` stays skeleton-gated; detection uses this bbox-only export.

### Commands

```bash
# Export COCO + Ultralytics labels (union-when-close)
python export_bbox_detection.py --annotation-version 1.7.0 --union-iou-threshold 0.3

# Ablation export (always separate)
python export_bbox_detection.py --annotation-version 1.7.0 --always-separate

# Train (auto-exports if data.yaml is missing)
python train_detector.py --config config_bbox_union.yaml
python train_detector.py --config config_bbox_separate.yaml

# Evaluate held-out split (AP + limb coverage + close-pair stats)
python eval_detector.py \
  --weights training/detector_runs/bbox_union_iou0.3/weights/best.pt \
  --data dataset/detection/1.7.0/union_iou0.3/data.yaml
```

Requires `ultralytics` (`pip install ultralytics`). Outputs land under `dataset/detection/` (gitignored) and `training/detector_runs/`.

COCO JSONs under `dataset/detection/<version>/<policy>/annotations/` can later feed an RTMDet fine-tune for production drop-in; this branch uses YOLO for fast label-policy experiments.

## Nuclio pose deployment

The CVAT Nuclio package is under `deploy/nuclio/`. It contains the standalone
handler, model implementation, `dance_23` mapping, DINOv3 backbone configuration,
and staged weights. To stage a pose-estimator checkpoint and deploy it locally:

```bash
PYTHON=/home/arodabough/anaconda3/envs/dino/bin/python \
  deploy/nuclio/prepare_assets.sh \
  runs/pose_estimator/2026_10_16_005/final.pt

nuctl deploy pth-dinocc-dance-23 \
  --project-name cvat \
  --path "$PWD/deploy/nuclio" \
  --file "$PWD/deploy/nuclio/function_gpu.yaml" \
  --platform local \
  --resource-limit nvidia.com/gpu=1
```

If no checkpoint argument is supplied, `prepare_assets.sh` uses
`runs/pose_estimator/2026_10_16_005/final.pt`, or the newest
`runs/pose_estimator/*/final.pt` when that file is unavailable. The script also
syncs the package to the sibling `cvat_annotation_tools/nuclio/dinocc` directory
when that checkout exists.
