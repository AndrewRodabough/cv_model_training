#!/usr/bin/env bash
# Stage the Nuclio package and optionally sync it to cvat_annotation_tools.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANNOTATION_TOOLS_DEST="${ANNOTATION_TOOLS_DEST:-${ROOT}/../cvat_annotation_tools/nuclio/dinocc}"
PYTHON="${PYTHON:-python3}"
DEFAULT_CHECKPOINT="${ROOT}/training/pose_estimator_runs/2026_10_16_005/final.pt"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -f "${DEFAULT_CHECKPOINT}" ]]; then
    CHECKPOINT="${DEFAULT_CHECKPOINT}"
  else
    CHECKPOINT="$("${PYTHON}" - "${ROOT}/training/pose_estimator_runs" <<'PY'
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
checkpoints = sorted(
    run_root.glob("*/final.pt"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
print(checkpoints[0] if checkpoints else "")
PY
)"
  fi
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Usage: $0 [path/to/checkpoint.pt]" >&2
  echo "No checkpoint found under training/pose_estimator_runs/*/final.pt" >&2
  exit 1
fi

mkdir -p "${DEST}/mapping" "${DEST}/weights" "${DEST}/backbone"
for required_file in "${DEST}/main.py" "${DEST}/dino_cc.py" "${DEST}/vec.py" \
  "${DEST}/function_gpu.yaml" "${DEST}/mapping/dance_23.yaml"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing Nuclio package file: ${required_file}" >&2
    exit 1
  fi
done

echo "Copying checkpoint: ${CHECKPOINT}"
CHECKPOINT_ABS="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
DEST_CKPT="${DEST}/weights/best.pt"
if [[ "${CHECKPOINT_ABS}" != "${DEST_CKPT}" ]]; then
  cp -f "${CHECKPOINT}" "${DEST_CKPT}"
fi

BACKBONE_ID="$(
  "${PYTHON}" - "${CHECKPOINT}" <<'PY'
import sys
import torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(ck.get('backbone_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m'))
PY
)"

echo "Refreshing backbone config for ${BACKBONE_ID} (config-only, no weight download)..."
"${PYTHON}" - "${BACKBONE_ID}" "${DEST}/backbone" <<'PY'
import sys
from pathlib import Path

repo_id, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
dest = out_dir / 'config.json'

# Prefer local HF cache / already-synced config to stay offline-friendly.
candidates = []
hub = Path.home() / '.cache/huggingface/hub'
slug = 'models--' + repo_id.replace('/', '--')
snap_root = hub / slug / 'snapshots'
if snap_root.is_dir():
    for snap in sorted(snap_root.iterdir(), reverse=True):
        cfg = snap / 'config.json'
        if cfg.exists():
            candidates.append(cfg)
if dest.exists():
    candidates.append(dest)

if candidates:
    src = candidates[0]
    dest.write_bytes(src.read_bytes() if src.resolve() != dest.resolve() else dest.read_bytes())
    print(f'Wrote {dest} from {src}')
else:
    from huggingface_hub import hf_hub_download
    src = Path(hf_hub_download(repo_id=repo_id, filename='config.json'))
    dest.write_bytes(src.read_bytes())
    print(f'Wrote {dest} via hub download')
PY

# Drop any accidental full backbone weight files so the image stays lean.
rm -f "${DEST}/backbone"/model.safetensors \
      "${DEST}/backbone"/pytorch_model.bin \
      "${DEST}/backbone"/model.bin \
      "${DEST}/backbone"/model.pt

# Sync into cvat_annotation_tools for deploy_gpu.sh (expects function-gpu.yaml).
if [[ -d "$(dirname "${ANNOTATION_TOOLS_DEST}")" ]]; then
  echo "Syncing to annotation-tools: ${ANNOTATION_TOOLS_DEST}"
  mkdir -p "${ANNOTATION_TOOLS_DEST}/mapping" \
           "${ANNOTATION_TOOLS_DEST}/weights" \
           "${ANNOTATION_TOOLS_DEST}/backbone"
  cp -f "${DEST}/main.py" "${ANNOTATION_TOOLS_DEST}/main.py"
  cp -f "${DEST}/dino_cc.py" "${ANNOTATION_TOOLS_DEST}/dino_cc.py"
  cp -f "${DEST}/vec.py" "${ANNOTATION_TOOLS_DEST}/vec.py"
  cp -f "${DEST}/mapping/dance_23.yaml" "${ANNOTATION_TOOLS_DEST}/mapping/dance_23.yaml"
  cp -f "${DEST}/function_gpu.yaml" "${ANNOTATION_TOOLS_DEST}/function-gpu.yaml"
  cp -f "${DEST}/weights/best.pt" "${ANNOTATION_TOOLS_DEST}/weights/best.pt"
  cp -f "${DEST}/backbone/config.json" "${ANNOTATION_TOOLS_DEST}/backbone/config.json"
else
  echo "WARN: annotation-tools parent missing; skipped sync to ${ANNOTATION_TOOLS_DEST}" >&2
fi

du -h "${DEST}/weights/best.pt" "${DEST}/backbone/config.json"
cat <<EOF
Ready. Deploy with:
  nuctl deploy pth-dinocc-dance-23 \\
    --project-name cvat \\
    --path ${DEST} \\
    --file ${DEST}/function_gpu.yaml \\
    --platform local \\
    --resource-limit nvidia.com/gpu=1
EOF
