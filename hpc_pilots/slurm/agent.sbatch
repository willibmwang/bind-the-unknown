#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PILOT_LLM_URL=http://127.0.0.1:18573/v1
export PILOT_LLM_MODEL=emergency-pilot
export PILOT_LLM_TOKEN_FILE="$PWD/runs/backend/token"
mkdir -p runs/backend
.venv-hpc/bin/python - <<'PY'
import os,secrets
from pathlib import Path
p=Path('runs/backend/token')
fd=os.open(p, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
with os.fdopen(fd,'w') as f: f.write(secrets.token_hex(32))
PY
export VLLM_API_KEY
VLLM_API_KEY=$(cat "$PILOT_LLM_TOKEN_FILE")
.venv-llm/bin/vllm serve "$PWD/models/qwen3-4b" \
  --served-model-name emergency-pilot --host 127.0.0.1 --port 18573 \
  --max-model-len 4096 --max-num-seqs 4 --gpu-memory-utilization 0.5 \
  --generation-config vllm --seed 20261012 --disable-log-requests \
  > runs/backend/server.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
export PILOT_SERVER_PID="$server_pid"
.venv-hpc/bin/python - <<'PY'
import os,time,requests
deadline=time.monotonic()+600
while time.monotonic()<deadline:
    try:
        os.kill(int(os.environ['PILOT_SERVER_PID']),0)
        if requests.get('http://127.0.0.1:18573/health',timeout=3).status_code==200: break
    except requests.RequestException: pass
    time.sleep(2)
else: raise SystemExit('LLM server did not become healthy within ten minutes')
PY
.venv-hpc/bin/python -m hpc_pilots binding --pool 0 --policy agent --smoke
.venv-hpc/bin/python -m hpc_pilots structure --policy agent --smoke
export PILOT_RUN_PYTHON="$PWD/.venv-hpc/bin/python"
seq 0 39 | xargs -P 2 -n 1 bash -c \
  'SLURM_ARRAY_TASK_ID="$1" "$PILOT_RUN_PYTHON" -m hpc_pilots binding-array --agent-only' _
.venv-hpc/bin/python -m hpc_pilots structure --policy agent
.venv-hpc/bin/python -m hpc_pilots summarize
