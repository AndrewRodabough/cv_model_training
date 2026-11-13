# Dance Model Training

Code for training a top-down human pose estimator tuned for dancing.

## Current State

- CVAT annotation exports are processed into cleaned, training-ready annotations
- Dataset review of exports
- 2D top-down pose estimation training
- Detector training
- Dashboard for monitoring and analyzing training runs

## In Development

- 2D → 3D pose estimation lifter

## Detector Training

### Goals

- Teach the detector strategic bbox unions so limbs stay inside the box through occlusions. When dance partners overlap heavily, emit one union box to avoid missed detections of close couples.
- Reduce detector bias in crowds and wide-angle shots that miss smaller people or fail to detect everyone in frame.

### Usage

Requires `ultralytics` (`pip install ultralytics`). Run from the repo root with `PYTHONPATH=.`.

**Label policy** (single primary class `person`; optional `person_side` tier for bbox-only background people):

1. Keep only effectively cleaned bbox shapes (`frame_cleaned` or `cleaned` + `cleaned_completion` stride) — same rules as `dataset_parser.py`. Skeletons are **not** required.
2. Pair partners by `couple_id` when present; else lead/follow + greedy max IoU.
3. If pair IoU ≥ threshold (default `0.3`, configs often use `0.5`–`0.65`) → emit one **union** box; else emit both.
4. Ablation: `--always-separate` / `always_separate: true` skips unions.

Pose `cleaned_annotations.xml` stays skeleton-gated; detection uses this bbox-only export.

```bash
# Export COCO + Ultralytics labels (union-when-close)
PYTHONPATH=. python -m src.dataset_review.export_bbox_detection \
  --annotation-version 1.0.3 --union-iou-threshold 0.65

# Ablation export (always separate)
PYTHONPATH=. python -m src.dataset_review.export_bbox_detection \
  --annotation-version 1.0.3 --always-separate

# Train (auto-exports if data.yaml is missing; --export forces re-export)
PYTHONPATH=. python -m src.training.train_detector --config config/config_detector.yaml
PYTHONPATH=. python -m src.training.train_detector --config config/config_bbox_separate.yaml

# Sparse-label / side-tier experiment (mask outside GT + main/side weights)
PYTHONPATH=. python -m src.training.train_detector \
  --config config/config_detector_union065_side.yaml

# Evaluate held-out split (AP + limb coverage + close-pair stats)
PYTHONPATH=. python -m src.model_review.eval_detector \
  --weights training/detector_runs/bbox_union_iou0.65_side_103/weights/best.pt \
  --data dataset/detection/1.0.3/union_iou0.65/data.yaml
```

Outputs land under `dataset/detection/` (gitignored) and `training/detector_runs/`.

Useful config knobs: `union_iou_threshold`, `mask_outside_gt`, `side_tier_weights`, `pseudo_fill_unlabeled`, `test_clips` (fixed clip holdout shared with pose). COCO JSONs under `dataset/detection/<version>/<policy>/annotations/` can later feed an RTMDet fine-tune; YOLO is used here for fast label-policy experiments.

### Progress

- Union-when-close export + train/eval loop is in place; separate-box ablation configs exist for comparison.
- Active experiments on detector versions `1.0.2` / `1.0.3`: IoU thresholds (`0.5`, `0.65`, `0.75`), outside-GT class masking, and main/side tier weights so unlabeled crowd people are not punished as hard background.
- Eval reports standard box AP plus limb coverage (from pose skeletons) and close-pair behavior (`one_cover` / `two_cover` / `under_cover` / `miss`).
- Raw mAP favors separate boxes on sparse labels; union policy is judged more by couple coverage and limb inclusion than by AP alone.
- Crowd / wide-angle recall remains an open problem; pseudo-fill and completeness allowlists are the current levers.

## 2D Pose Estimation Training

### Goals

- Role-conditioned estimator (lead/follow) so the model knows who to resolve inside a crop — works with the detector’s union policy when both partners share one box.
- Higher accuracy on dance-specific poses underrepresented in standard datasets.
- Better estimates when limbs are in close proximity to a partner.
- Better estimates under heavy partner occlusion.
- Better estimates under heavy clothing occlusion.
- Temporal consistency so hard-to-read frames can borrow context from neighbors.
- Custom keypoint format for dance and biomechanical analysis.

### Usage

Train with the DINOv3 + joint-query SimCC stack (`DinoCC`). Prefer the `dino` conda env (torch + backbone). From repo root:

```bash
PYTHONPATH=. python -m src.training.train --config config/config_pose.yaml

# Continue from a checkpoint with a fresh optimizer
PYTHONPATH=. python -m src.training.train \
  --config config/config_pose_stage5_unfreeze4_dance100_181.yaml \
  --init-checkpoint training/pose_estimator_runs/<prior_run>/best.pt
```

Or use the [Dashboard](#dashboard) to edit configs and start/pause/stop runs.

**Model / data notes**

- Crops still come from GT boxes in `cleaned_annotations.xml` (skeleton-gated), not yet from the trained detector.
- Keypoint format: `dance_23` (`mapping/dance_23.yaml`) — body + detailed feet (`toe_b` / `toe_s` / `heel` / `heel_spike`) + ears + glabella.
- Head: `joint_query_self_attn_simcc` with optional `use_role`, `use_pair_bias`, L/R swap loss, bone/chirality terms.
- Temporal head: `temporal_joint_query_self_attn_simcc` (see `config/config_pose_temporal_*.yaml`); needs densely labeled contiguous clips.
- Optional COCO-WholeBody mix via `dataset.coco_wholebody` for staged pretrain → dance-only fine-tune.

Runs write to `training/pose_estimator_runs/YYYY_MM_DD_NNN/` (`history.jsonl`, `status.json` / `control.json`, `val_preds/`, `val_overlays/`, checkpoints).

Config knobs (under `training:`): `save_val_preds_every_n_epochs`, `epoch_overlay_top_n`, `watchlist_frames`, `lr_swap_loss_weight`, `joint_loss_weights`.

### Progress

- Single-frame top-down trainer is the mainline: DINOv3 backbone, joint-query SimCC, `dance_23` format, role conditioning, pair bias, and L/R-swap / bone losses.
- Staged recipes exist (COCO mix → unfreeze more blocks → dance-only), e.g. `config_pose_stage*_181.yaml` on annotation `1.8.1`.
- Feet remain the weakest part OKS; wrist/ankle/foot joints are up-weighted in config.
- Temporal T=5 training is prototyped but limited by how many clips have every-frame cleaned labels.
- Detector→pose coupling (union crops + role disambiguation at inference) is designed against but not the default train path yet — train still uses GT boxes.

## 2D → 3D Pose Estimation Lifter

### Goals

- Reduce excessive motion smoothing that traditional lifters create, so sharp strikes and accents are preserved.
- Grounded estimation aware of floor contact and partnering connections.
- Avoid movement bias that pretrained lifters show (e.g. leg direction heavily biased by hip facing).

### Usage

Not started — no training entrypoint yet.

### Progress

Scaffolding only; goals above define the target behavior.

## Dashboard

Local Gradio UI to edit configs, start/pause/stop runs, watch live loss curves, and scrub val predictions by epoch (pose + detector).

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

## Nuclio pose deployment

The CVAT Nuclio package is under `deploy/nuclio/`. It contains the standalone handler, model implementation, `dance_23` mapping, DINOv3 backbone configuration, and staged weights. To stage a pose-estimator checkpoint and deploy it locally:

```bash
PYTHON=/home/arodabough/anaconda3/envs/dino/bin/python \
  deploy/nuclio/prepare_assets.sh \
  training/pose_estimator_runs/<run_id>/final.pt

nuctl deploy pth-dinocc-dance-23 \
  --project-name cvat \
  --path "$PWD/deploy/nuclio" \
  --file "$PWD/deploy/nuclio/function_gpu.yaml" \
  --platform local \
  --resource-limit nvidia.com/gpu=1
```

If no checkpoint argument is supplied, `prepare_assets.sh` uses a default under `runs/pose_estimator/` when present, otherwise the newest `training/pose_estimator_runs/*/final.pt` / `best.pt` available. The script also syncs the package to the sibling `cvat_annotation_tools/nuclio/dinocc` directory when that checkout exists.
