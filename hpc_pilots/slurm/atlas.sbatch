#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec .venv-hpc/bin/python -m hpc_pilots preprocess-atlas --index "${SLURM_ARRAY_TASK_ID:?}"
