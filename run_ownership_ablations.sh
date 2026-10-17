#!/usr/bin/env bash
# Sequential ownership ablations: train each one-flag config, then eval ownership.
set -euo pipefail
cd "$(dirname "$0")"
LOG_DIR=training/runs/_ablation_logs
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/master.log"
SUMMARY="$LOG_DIR/comparison.txt"

{
  echo "=== ownership ablation suite started $(date -Is) ==="
  echo "baseline: training/runs/2026_10_04_003"
} | tee -a "$MASTER"

run_one() {
  local name="$1"
  local cfg="$2"
  local train_log="$LOG_DIR/${name}_train.log"
  echo "" | tee -a "$MASTER"
  echo ">>> START $name  config=$cfg  $(date -Is)" | tee -a "$MASTER"

  # Snapshot runs dir before training so we can find the new run folder.
  local before
  before=$(ls -1d training/runs/2026_* 2>/dev/null | sort || true)

  conda run --no-capture-output -n dino python train.py --config "$cfg" 2>&1 | tee "$train_log"
  local after
  after=$(ls -1d training/runs/2026_* 2>/dev/null | sort || true)
  local run_dir
  run_dir=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tail -1)
  if [[ -z "$run_dir" ]]; then
    # Fallback: newest run dir
    run_dir=$(ls -1dt training/runs/2026_* | head -1)
  fi
  echo ">>> RUN_DIR $name -> $run_dir" | tee -a "$MASTER"

  conda run --no-capture-output -n dino python eval_ownership.py --run "$run_dir" --checkpoint best.pt 2>&1 | tee "$LOG_DIR/${name}_ownership.log"
  echo ">>> DONE $name  $(date -Is)" | tee -a "$MASTER"
  echo "$name $run_dir" >> "$LOG_DIR/run_map.txt"
}

run_one reinject config_ablate_reinject.yaml
run_one spatial  config_ablate_spatial.yaml
run_one deepsup  config_ablate_deepsup.yaml
run_one xy       config_ablate_xy.yaml

{
  echo ""
  echo "=== comparison $(date -Is) ==="
  echo "--- baseline 003 ---"
  cat training/runs/2026_10_04_003/ownership_metrics/ownership_summary.txt 2>/dev/null || true
  while read -r name run_dir; do
    echo ""
    echo "--- $name ($run_dir) ---"
    cat "$run_dir/ownership_metrics/ownership_summary.txt" 2>/dev/null || true
  done < "$LOG_DIR/run_map.txt"
  echo "=== suite finished $(date -Is) ==="
} | tee "$SUMMARY" | tee -a "$MASTER"
