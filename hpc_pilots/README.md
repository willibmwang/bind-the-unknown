# Emergency co-scientist: first HPC pilots

Prepared 12 October 2026. Two isolated decision-policy experiments support the emergency scientist project: binding-assay acquisition under limited/delayed feedback, and structural-evidence allocation. These are pilot components; they do not establish therapeutic effectiveness or full-system performance.

## Cluster and launch

Target: MATS cluster, `w.wang@185.141.218.207`, Slurm `compute` partition. Project directory: `/mnt/nw/home/w.wang/mats-emergency-20261012`. The cluster's default Python was 3.10; this project uses an isolated Python 3.12 environment and the tested `requirements.lock.txt`. GPU serving uses a separate vLLM 0.10.2 environment.

From the remote project directory:

```bash
.venv-hpc/bin/python -m hpc_pilots.submit --partition compute --with-agent
squeue -u w.wang
```

Submission is idempotent: an existing `runs/submissions.json` prevents duplicate sweeps. Inspect the recorded IDs before any restart. Only this project's jobs may be canceled or replaced during repairs. Scripts use ordinary Slurm scheduling and do not preempt other users.

When model staging is still in progress, submit the CPU pipeline without `--with-agent`, then add the GPU phase with `.venv-hpc/bin/python -m hpc_pilots.submit --partition compute --add-agent`. This adds only the recorded pipeline's GPU job and preserves its dependencies.

Pipeline:

1. CPU smoke suite: data/inference-interface tests, all four binding baselines, delayed-feedback smoke, ATLAS analysis on development entry `1dk8_A`, and three structural baselines.
2. Binding array: 20 disjoint pools × 4 baseline policies × 2 feedback conditions = 160 tasks, at most four active tasks.
3. ATLAS preprocessing: 12 entries, at most two active tasks, four CPUs each.
4. Structural baseline replay on the eight evaluation entries.
5. Baseline summary and charts.
6. One queued L40 GPU allocation: start a local Qwen server, smoke-test both agent interfaces, run 40 binding-agent episodes and structural-agent replay, then summarize all results and release the GPU.

Full jobs depend on successful smoke tests. The GPU phase depends on successful baseline and preprocessing jobs. Inference and analysis failures fail their task; they never silently become a baseline result. Accepted Slurm IDs are recorded immediately, including if submission later fails.

## Dataset provenance

Binding: [ProteinGym v1.3](https://github.com/OATML-Markslab/ProteinGym), assay `SPG1_STRSG_Wu_2016`, 149,360 measured variants. Only the four variable residue identities are provided as candidate features. All unqueried assay scores remain in the evaluator; the LLM sees observed-score summaries, predictor estimates, uncertainty/diversity heuristics, and the remaining resource state. Candidate identities in prompts are opaque pool indices.

Twenty 4,096-candidate pools and a separate 4,096-candidate development pool are disjoint and selected without outcome filtering. The predictor is a fixed 64-tree random forest shared by greedy, UCB, and LLM-mediated acquisition. Tree dispersion is an uncertainty heuristic, not a calibrated posterior. Eight initial measurements are identical across policies within a pool. The total measurement budget is 128, acquired in batches of eight. Delay conditions of zero and three decision rounds are synthetic stress tests; they are not estimates of actual assay turnaround.

The evaluator reports the best selected outcome and top-1% recovery after eventual results are known, as well as the best actually returned outcome and outstanding measurements at each checkpoint. Top-1% membership is never supplied to a policy. Ties at the evaluator threshold are included. Pool variability is not biological replication across independent proteins.

Structural data: [ATLAS](https://www.dsimb.inserm.fr/ATLAS/download.html), pinned metadata `2024_11_18_ATLAS_info.tsv`, three 100 ns replicates per entry in the analysis download. Selection uses human organism, 50-250 residues, nonredundant flag, and one entry per ECOD family sorted by PDB identifier. Trajectory-derived outcome columns are excluded from selection. Four entries are for development, eight for evaluation. Entries are structural controls, not designated therapeutic targets.

ATLAS dataset terms are **CC BY-NC 4.0**. This pilot is scoped to noncommercial research; the API/software licence is separate from the data licence. Commercial use requires terms from the dataset owner.

SASA is computed with MDTraj in nm² per residue, 240 sphere points and trajectory stride five. Each acquisition block contains 20 saved frames. Two initial blocks per protein expose one block from R1 and one from R2; subsequent requests interleave chronological blocks. R3 is the held-out simulation reference and is never provided to a policy. Temporal correlation is preserved; block dispersion is a heuristic, not a frame-independent confidence interval. Surface exposure and simulation agreement do not establish binding or inhibition.

A single starting structure supplies the static baseline. Initial ATLAS structures have already undergone equilibration. Structural replay costs are reconstructed from measured per-block preprocessing CPU times; they are not new MD simulation costs or live request latency. LLM inference latency/tokens are logged separately. Aggregate plots also retain per-protein results for inspection.

All downloaded inputs and frozen splits have SHA-256 hashes in `data/processed/*/manifest.json`. The model snapshot is pinned by Hugging Face commit and hashed in `models/manifest.json`. Scientific results and task traces are JSON under `runs/`; summary CSVs and PNGs are under `runs/summary/`. Smoke outputs are marked and excluded from scientific summaries.

## Agent baseline

Model: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), ungated, Apache-2.0. Use a pinned local safetensors snapshot; custom remote model code is not enabled. Temperature zero, maximum 192 output tokens, 4,096-token server context, and a constrained JSON action schema.

The binding agent selects the acquisition strategy (greedy, explore, or diverse) from the same underlying tools as the baseline policies. The structural agent selects an entry for the next evidence block or stops. These test a limited orchestration policy, not free-form biological discovery. Prompts and rationales are retained. Invalid actions/inference errors fail explicitly.

The GPU server binds to localhost within its Slurm allocation and uses a private token file. Agent runs execute inside that same allocation; no externally hosted model or public inference endpoint receives the dataset. The server is terminated automatically when the job exits.

## Local verification

```bash
.venv-hpc/bin/python -m pytest hpc_pilots/tests -q
.venv-hpc/bin/python -m hpc_pilots binding --pool 0 --policy ucb --delay 3 --smoke
OMP_NUM_THREADS=2 .venv-hpc/bin/python -m hpc_pilots preprocess-atlas --index 1 --smoke
.venv-hpc/bin/python -m hpc_pilots structure --policy uncertainty --smoke
```

Tests check disjoint pools, unique acquisitions, delayed-outcome visibility, policy isolation from hidden labels, structural reference isolation, resource accounting, and the LLM API contract. Smoke tests validate real dataset parsing and the MDTraj analysis path. Initial smoke tests are development checks, not scientific findings.
