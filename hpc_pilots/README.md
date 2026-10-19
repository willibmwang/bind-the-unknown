# Emergency co-scientist: pilots and domain layer

Prepared 12 October 2026, extended 19 October 2026. Two isolated decision-policy experiments support the emergency scientist project: binding-assay acquisition under limited/delayed feedback, and structural-evidence allocation. These are pilot components; they do not establish therapeutic effectiveness or full-system performance.

## Domain layer, added 19 October

The pilots above measure resource allocation over opaque integer candidates. That is deliberate — it is what keeps the acquisition comparison clean — but it is also domain-free: nothing in it is specific to an emergency, to a pathogen, or to macrocyclic chemistry. These four modules add that specificity, and each one states what it does not establish.

| Module | What it adds | What it is not |
| --- | --- | --- |
| `dossier.py` | The seven-axis target dossier as the decision state, with per-axis citations, timestamps, retained conflicts and open uncertainties, plus the six-action space: retrieve evidence, request structure, rerank, invoke design tool, recommend experiment, stop. Evidence dated after the decision clock is refused and the refusal is recorded. | Readiness is evidence coverage of a rubric. It is not binding, inhibition, antiviral activity or efficacy. Named targets reintroduce pretrained-hindsight risk that the date boundary does **not** control. |
| `targets.py` | Evidence packets for three families (Coronaviridae, Orthomyxoviridae, Henipavirus): 11 targets, 59 sourced entries, 8 synthetic update probes. | Most assessments are the project's own write-up pending expert review, not independent citations. Three claims are flagged `verified=false` and must not be used as positive controls. |
| `emergency.py` | Timed episodes with a deadline and an action budget, scoring whether a policy updates on relevant evidence, holds its recommendation under irrelevant evidence, and retains contradictory evidence. The recommendation's endpoint claim is gated by the assay-availability axis. | Deterministic baselines are stable by construction; the metric earns its value on the LLM policy. No structure backend is wired in, so `request_structure` records a charged request and returns no structural claim. |
| `chemistry.py` | Building-block registry over canonical and non-canonical classes with initiator/closure roles, a thioether library specification, and an encoding that extends past the canonical twenty. | Literature block classes, **not** the SynX catalog. Chemical identifiers, CCD codes, codons, Rosetta parameters and rotamer libraries are explicitly null pending the platform export. Library enumeration predicts nothing. |

Pathogen decision environments are configured separately in `configs/targets.json`, which leaves `configs/today.json` byte-identical so the recorded GB1 and calibration runs still verify against their embedded config hash.

```bash
.venv-hpc/bin/python -m hpc_pilots chemistry                     # build the block registry
.venv-hpc/bin/python -m hpc_pilots targets                       # build the family packets
.venv-hpc/bin/python -m hpc_pilots prepare-target --landscape mpro
.venv-hpc/bin/python -m hpc_pilots target --landscape mpro --pool 0 --policy ucb --delay 3
.venv-hpc/bin/python -m hpc_pilots emergency --family coronaviridae --policy rubric
```

**Endpoint discipline across landscapes.** GB1 is an IgG binding selection, Mpro is FRET cleavage plus growth, and both haemagglutinin scans are viral replication. These are different endpoints and their metrics are never pooled. Pathogen pools hold 512 candidates and use a 5% top fraction, because 1% of 512 is about five members; those figures must not be compared with the GB1 1% figures. Pool geometry shrinks rather than reusing candidates across pools described as disjoint: Mpro supports 10 disjoint pools, HA H1 19, HA H3 20.

**Enrichment.** Acquisition checkpoints now also report an enrichment ratio against the pool baseline, which is the readout shape a display round produces. An enrichment ratio is not an affinity, a functional inhibition or an antiviral activity, and no macrocycle has been measured anywhere in this repository.

## Method-development landscapes, added 19 October

Every pathogen landscape above is scored by enzymatic function or replication fitness, so none of them measures binding. Four landscapes in `configs/method.json` do, and they exist to develop and stress the acquisition machinery rather than to propose targets. They are staged under `data/processed/method/` and run with `prepare-method` and `method`.

| Landscape | Interface | Variants | Pool geometry |
| --- | --- | --- | --- |
| `spike_rbd_ace2` | SARS-CoV-2 RBD against human ACE2 | 3,802 singles | 6 × 512, top 5% |
| `dlg4_pdz3_peptide` | PSD-95 PDZ3 against its CRIPT peptide ligand | 1,576 singles | 5 × 256, top 5% |
| `yap1_ww_peptide` | YAP1 WW domain against a proline-rich peptide | 10,075, of which 9,713 combined | 18 × 512, top 5% |
| `spg1_olson_igg` | GB1 against IgG Fc | 536,962, of which 535,917 combined | 20 × 4,096, top 1% |

Three reasons these were added. The two peptide landscapes measure a peptide–protein interface, which is the kind of interaction the project is ultimately designing against, where GB1's IgG selection is not. `spg1_olson_igg` is large enough to keep the original pilot's 4,096-candidate pools and 1% threshold, which restores the 32× headroom that the small pathogen pools cannot offer — the same protein as the GB1 pilot, so the geometry is matched and only the variant complexity changes. And `spike_rbd_ace2` is the only measured binding-affinity landscape on disk for a viral protein.

**Two encodings.** Single-substitution landscapes keep the compact (position, residue) pair. Landscapes containing combined substitutions cannot: a variant is no longer described by one change, so they switch to residue identity at every variable position. `encode_variants` picks between them and records which was used in the manifest.

**These are substrates, not proposals.** None is a pathogen target recommendation. Spike RBD appears here only for method development and is explicitly rejected in the target write-up as a basis for any conservation or family-transfer claim. All four are canonical residues on folded protein domains, so none supports a claim about macrocyclic or non-canonical chemistry. The peptide landscapes mutate the protein side of the interface rather than the peptide ligand.

**Read the recall multiples with care.** Random top-fraction recall equals the labelled fraction of the pool, so a multiple over random is bounded by the reciprocal of that fraction: 32× for GB1 at 128 labels from 4,096 candidates, but only 4× for a pathogen pool at 128 from 512. The two are not comparable, and the GB1 figure must not be quoted without its budget ratio. Compare recall at matched budget *fractions*, or use the normalised gain `(policy − random) / (1 − random)`. The measured consequence is in §2.5 of the Week 4 debrief.

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
