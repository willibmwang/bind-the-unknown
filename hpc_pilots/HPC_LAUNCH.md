# MATS launch record: 12 October 2026

Cluster: `w.wang@185.141.218.207`, Slurm partition `compute`.

Remote project: `/mnt/nw/home/w.wang/mats-emergency-20261012`.

| Job ID | Work | Submission/verification |
| --- | --- | --- |
| 19168 | CPU smoke suite | Completed successfully; seven correctness checks and both real-data runners passed |
| 19169 | Binding policy replay | 64 of 160 tasks complete at the latest launch checkpoint; four concurrent tasks maximum |
| 19170 | ATLAS preprocessing | All 12 entries processed successfully |
| 19171 | Structural baseline replay | All three baseline policies completed successfully |
| 19172 | Baseline summaries and plots | Depends on binding and structural baseline jobs |
| 19202 | LLM policy comparison | Queued GPU phase; depends on successful baseline summary; requests one L40 GPU |

The GPU phase runs its own binding and structural smoke checks before the 40 binding-agent episodes and structural-agent replay. All eight GPUs were allocated when the pipeline was submitted, so inference start time depends on scheduler availability. GPU inference has not yet been validated on an allocation at this initial submission checkpoint.

The latest checkpoint found no tracebacks in the pilot task logs. This is a dated launch-progress snapshot; the remaining binding tasks, baseline summary, and GPU phase continue under Slurm.

The pinned model snapshot is `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Safetensors and configuration files are hashed in `model.manifest.json`. Scientific packages are frozen in `requirements.lock.txt`; the separate inference environment uses vLLM 0.10.2 and Torch 2.8.0, with its own lock file.

The source of truth for job IDs and submission arguments is the cluster's `runs/submissions.json`. Its copy in the local project records accepted submissions, not continuously refreshed scheduler states. Scientific outputs are under the cluster's `runs/binding`, `runs/structure`, and `runs/summary`; logs are under `runs/logs`. Smoke results are marked and excluded from summaries.

Monitor from the Mac:

```bash
ssh w.wang@185.141.218.207 'squeue -u w.wang'
```

Retrieve results from the project directory, excluding the private inference-server token:

```bash
rsync -az --exclude='backend/token' \
  w.wang@185.141.218.207:/mnt/nw/home/w.wang/mats-emergency-20261012/runs/ runs/
```

An incomplete summary explicitly reports missing episodes. Treat these as controlled resource-allocation pilot results; no non-canonical, antiviral, or full emergency-scientist performance claim is established by launching them.
