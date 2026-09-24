#!/usr/bin/env bash
# Stage code + checkpoint into deploy/nuclio before `nuctl deploy`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECKPOINT="${1:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="$(ls -1dt "${ROOT}"/training/runs/*/best.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Usage: $0 [path/to/best.pt]" >&2
  echo "No checkpoint found under training/runs/*/best.pt" >&2
  exit 1
fi

echo "Syncing model code from repo root..."
cp -f "${ROOT}/dino_cc.py" "${DEST}/dino_cc.py"
cp -f "${ROOT}/vec.py" "${DEST}/vec.py"
cp -f "${ROOT}/mapping/dance_20.yaml" "${DEST}/mapping/dance_20.yaml"

mkdir -p "${DEST}/weights" "${DEST}/backbone"
echo "Copying checkpoint: ${CHECKPOINT}"
cp -f "${CHECKPOINT}" "${DEST}/weights/best.pt"

BACKBONE_ID="$(
  python3 - "${CHECKPOINT}" <<'PY'
import sys
import torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(ck.get('backbone_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m'))
PY
)"

echo "Refreshing backbone config for ${BACKBONE_ID} (config-only, no weight download)..."
python3 - "${BACKBONE_ID}" "${DEST}/backbone" <<'PY'
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

du -h "${DEST}/weights/best.pt" "${DEST}/backbone/config.json"
cat <<EOF
Ready. Deploy with:
  nuctl deploy --project-name cvat \\
    --path ${DEST} \\
    --file ${DEST}/function_gpu.yaml \\
    --platform local \\
    --resource-limit nvidia.com/gpu=1
EOF
