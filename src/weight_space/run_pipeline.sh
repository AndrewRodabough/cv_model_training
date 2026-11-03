#!/bin/bash
# Full weight-space programming pipeline
# Usage: ./run_pipeline.sh <run_dir> <checkpoint> [forward_pass_budget]

set -e

RUN_DIR="${1:-training/runs/2026_10_02_003}"
CHECKPOINT="${2:-best.pt}"
BUDGET="${3:-50}"

echo "========================================" echo "Weight-Space Programming Pipeline"
echo "========================================"
echo "Run: $RUN_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Forward pass budget: $BUDGET"
echo ""

# Step 1: Extract L/R swap cases
echo "Step 1: Extracting L/R swap cases..."
python -m src.weight_space.extract_lr_swaps \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output weight_space_data/lr_swaps.json \
    --confidence-threshold 0.15 \
    --max-cases 200

echo ""
echo "Step 2: Analyzing head weights..."
python -m src.weight_space.analyze_head \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --output weight_space_data/head_analysis \
    --swap-data weight_space_data/lr_swaps.json

echo ""
echo "Step 3: Testing harness (baseline performance)..."
python -m src.weight_space.weight_edit_harness \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --swap-data weight_space_data/lr_swaps.json \
    --forward-pass-budget "$BUDGET" \
    --test-only

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review analysis: cat weight_space_data/head_analysis/summary.txt"
echo "2. Review swap cases: head weight_space_data/lr_swaps.json"
echo "3. Run interactive harness or provide to LLM for editing"
echo ""
echo "Interactive mode:"
echo "  python -m src.weight_space.weight_edit_harness \\"
echo "    --run-dir $RUN_DIR \\"
echo "    --checkpoint $CHECKPOINT \\"
echo "    --swap-data weight_space_data/lr_swaps.json \\"
echo "    --forward-pass-budget $BUDGET \\"
echo "    --interactive"
