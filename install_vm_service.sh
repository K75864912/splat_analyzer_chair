#!/usr/bin/env bash
# Install the EXISTING V5 upload API as an unprivileged, boot-enabled service.
set -euo pipefail
repo="$(pwd)"
python_bin="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
bind="0.0.0.0"
port=8080
dry_run=0
while (($#)); do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --bind) bind="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) echo 'bash install_vm_service.sh --repo REPO --python /absolute/venv/bin/python [--bind 0.0.0.0] [--port 8080] [--dry-run]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
repo="$(cd "$repo" && pwd -P)"
python_bin="${python_bin:-${repo}/venv/bin/python}"
# Do not resolve the Python symlink: doing so would bypass the virtualenv.
[[ "$python_bin" = /* ]] || python_bin="$(pwd)/$python_bin"
for value in "$repo" "$python_bin"; do
  [[ "$value" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Use absolute paths without spaces or special characters.' >&2; exit 2; }
done
[[ "$bind" = '0.0.0.0' || "$bind" = '127.0.0.1' ]] || { echo '--bind must be 0.0.0.0 or 127.0.0.1' >&2; exit 2; }
[[ "$port" =~ ^[0-9]{1,5}$ ]] && ((10#$port >= 1024 && 10#$port <= 65535)) || { echo 'Invalid port (1024-65535).' >&2; exit 2; }
port=$((10#$port))
[[ -x "$python_bin" && -f "$repo/server.py" && -f "$repo/pipeline.py" ]] || { echo 'Missing Python environment, server.py, or pipeline.py.' >&2; exit 1; }
service_user="$(id -un)"
[[ "$service_user" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
if [[ "$service_user" = root && "$dry_run" = 0 ]]; then
  echo 'Run as your normal VM user with venv activated, not sudo bash. Only systemd installation uses sudo.' >&2
  exit 1
fi
service_group="$(id -gn)"
unit_file="$(mktemp)"
env_file="$(mktemp)"
trap 'rm -f "$unit_file" "$env_file"' EXIT
cat > "$unit_file" <<EOF
[Unit]
Description=Splat Analyzer V5 upload API
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=${service_user}
Group=${service_group}
WorkingDirectory=${repo}
Environment="PATH=$(dirname "$python_bin"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="WMD_DEVICE=cuda"
Environment="SPLAT_SAVE_FRAMES=0"
Environment="SPLAT_VERBOSE_DETECTIONS=0"
EnvironmentFile=-/etc/splat-analyzer-v5.env
ExecStart=${python_bin} -m uvicorn server:app --host ${bind} --port ${port} --workers 1
Restart=on-failure
RestartSec=10
TimeoutStopSec=600
UMask=0077
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
# Capture only CUDA/toolchain and pipeline tuning variables; never copy credentials.
"$python_bin" - "$env_file" <<'PY'
import os, sys
names = ('CUDA_HOME', 'LD_LIBRARY_PATH', 'TORCH_EXTENSIONS_DIR', 'MAX_JOBS',
         'SPLAT_RENDER_BATCH', 'SPLAT_IO_WORKERS', 'SPLAT_POSITION_MODE',
         'SPLAT_POST_NMS_IOU', 'SPLAT_CLUSTER_EPS_FRAC', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
with open(sys.argv[1], 'w') as f:
    f.write('# Optional overrides for the VM service. Restart after editing.\n')
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            if any(c in value for c in '\r\n\x00'):
                raise SystemExit('Unsupported newline in ' + name)
            value = value.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'{name}="{value}"\n')
PY
if ((dry_run)); then
  cat "$unit_file"
  echo '# Environment file preview:'
  cat "$env_file"
  exit 0
fi
command -v systemctl >/dev/null
command -v curl >/dev/null
cd "$repo"
echo 'Checking the existing V5 environment (no package upgrades)...'
"$python_bin" - <<'PY'
import importlib, importlib.metadata, pathlib
for module in ('torch', 'fastapi', 'uvicorn', 'multipart', 'ultralytics', 'gsplat'):
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {module}. Use this venv to install the matching requirement, then retry.\n{exc}')
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this Python environment. Check nvidia-smi and the venv.')
if importlib.metadata.version('ultralytics') != '8.4.160':
    raise SystemExit('This V5 requires ultralytics==8.4.160. Do not change PyTorch/CUDA to install the listener.')
if not pathlib.Path('yoloe-26x-seg.pt').is_file():
    raise SystemExit('Put the tested yoloe-26x-seg.pt in the repository root before installing.')
import inspect, pipeline, server
if 'frame_cache' not in inspect.signature(pipeline._run_yoloe_26).parameters:
    raise SystemExit('The checkout is not the expected in-memory V5 pipeline.')
if 'post' not in server.app.openapi().get('paths', {}).get('/analyze', {}):
    raise SystemExit('server:app does not expose POST /analyze.')
print('Preflight OK:', torch.cuda.get_device_name(0))
print('This is an import/CUDA check, not a complete inference warmup.')
PY
# Do not silently replace another process already listening on this port.
if ! systemctl is-active --quiet splat-analyzer-v5.service; then
  "$python_bin" - "$bind" "$port" <<'PY'
import socket, sys
with socket.socket() as s:
    try:
        s.bind((sys.argv[1], int(sys.argv[2])))
    except OSError as exc:
        raise SystemExit(f'Port is already in use or unavailable: {exc}. Stop the old listener explicitly, then retry.')
PY
fi
sudo -v
if sudo test -f /etc/systemd/system/splat-analyzer-v5.service; then
  sudo cp -a /etc/systemd/system/splat-analyzer-v5.service "/etc/systemd/system/splat-analyzer-v5.service.bak.$(date +%Y%m%d%H%M%S)"
fi
sudo install -m 0644 "$unit_file" /etc/systemd/system/splat-analyzer-v5.service
if ! sudo test -f /etc/splat-analyzer-v5.env; then
  sudo install -m 0600 "$env_file" /etc/splat-analyzer-v5.env
fi
sudo systemctl daemon-reload
sudo systemctl enable splat-analyzer-v5.service
sudo systemctl restart splat-analyzer-v5.service
echo 'Waiting for the API schema (up to 90 seconds)...'
for ((i=0; i<45; i++)); do
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${port}/openapi.json" | "$python_bin" -c 'import json,sys; assert "post" in json.load(sys.stdin)["paths"]["/analyze"]' >/dev/null 2>&1; then
    echo "API is listening. Local check: http://127.0.0.1:${port}/docs"
    echo 'Boot startup enabled. Logs: sudo journalctl -u splat-analyzer-v5 -f'
    echo 'Run one real upload before declaring inference ready.'
    exit 0
  fi
  sleep 2
done
sudo journalctl -u splat-analyzer-v5 -n 60 --no-pager
echo 'The API did not become ready. See the log above.' >&2
exit 1
