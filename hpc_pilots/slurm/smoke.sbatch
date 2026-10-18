#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
.venv-hpc/bin/python -m pytest hpc_pilots/tests -q
for policy in random diverse greedy ucb; do
  .venv-hpc/bin/python -m hpc_pilots binding --pool 0 --policy "$policy" --smoke
done
.venv-hpc/bin/python -m hpc_pilots binding --pool 0 --policy ucb --delay 3 --smoke
.venv-hpc/bin/python -m hpc_pilots preprocess-atlas --index 1 --smoke
for policy in static uniform uncertainty; do
  .venv-hpc/bin/python -m hpc_pilots structure --policy "$policy" --smoke
done
printf 'CPU smoke suite passed\n'
