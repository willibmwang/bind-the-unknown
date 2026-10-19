# Emergency AI scientist — detailed 10-week delivery plan

Updated 14 October 2026. **Planning proposal, not a completion report.** Week 1 began **28 September 2026**. W7 workdays are **9–13 November**, within the user's November 8 London week. W10 is **30 November–4 December**. WnDm means project week n, working day m. The exact travel itinerary and presentation day remain unconfirmed; the chart proposes the final handoff on December 4.

**Rebaselined from October 14 (W3D3), re-adjusted October 19 (W4D1).** W1/W2 are elapsed and are not reconstructed as completed. The existing brief, public-data provenance and HPC pilot runners are foundations to reuse. No task status is inferred from its calendar date: a task reads `Planned` unless a **Status** line names its evidence, and only four do.

**Task ids carry their stream**, so an id states where it belongs without a lookup: **GATE** decisions, access and gates · **COSCI** emergency co-scientist and benchmark · **MOL** macrocycle design and platform · **LAB** SynX experiment and London · **SAFE** safety benchmark preparation · **TRAIN** training harness · **FCAST** forecasting and analysis · **PAPER** paper, artifact and presentation.

**What changed on October 19.** The co-scientist's decision state was a budgeted bandit over opaque integer candidates: no pathogen, no chemistry, no evidence model. COSCI-13 replaced that state with the seven-axis target dossier and the six-action space the proposal specifies; COSCI-14 staged Mpro and both haemagglutinin landscapes as pathogen decision environments with a grouped H1-to-H3 holdout; COSCI-15 added the evidence-update stability scoring; MOL-08 put non-canonical building blocks into the representation and an enrichment ratio into the readout. The recorded GB1 and calibration results are unaffected and still reproduce. None of this is a macrocycle measurement, and the physics-based non-canonical track remains untouched.

## Research decision

Use **one paper narrative**: can cheap zero-shot agentic stress signatures, optionally augmented by small-budget post-training, forecast larger-budget gains in scientific reasoning? Build the emergency AI scientist as the applied system and SynX validation case. Controlled transcriptomic worlds supply the clean forecasting testbed. Share evidence state, process scoring and resource accounting; do not pool molecular selection scores with transcriptomic ground truth.

The [uploaded grant](</Users/ww/Desktop/Desktop - William’s MacBook Pro/YALE/26-27 postgrad/Thinking Machines Lab/ww_TML_Safety_Grant.pdf>) defines a capability-uplift forecasting research harness. Its safety motivation does not supply a validated CBRN-risk endpoint or a runtime safety filter. A well-controlled negative forecasting result is valuable evidence. ICML/NeurIPS acceptance cannot be scheduled or guaranteed.

**Weeks 8–10 own the main Tinker build, training, evaluation and packaging.** Weeks 1–6 prepare task definitions, worlds, held-out splits, tool primitives, baseline signatures and benign curricula. A small W3 checkpoint smoke retires an access risk. Starting all prerequisites in W8 would make credible W10 research unlikely.

This is an aggressive minimum viable research plan, not two independent full papers. It assumes reuse of pilot code, bounded prototype features, timely expert reviews, SynX design staff supporting library preparation, and SynX staff performing experimental preparation/execution/readout. Hands-on hours exclude unattended computation, travel and SynX staff effort. Re-estimate at the October 23 feasibility gate and after the W8 throughput pilot; spare capacity is contingency, not additional feature scope. The chart exposes dependency failures rather than making collaborator availability certain.

## Hard deadlines and dependency paths

| Deadline | Required evidence | Consequence if missed |
| --- | --- | --- |
| Oct 16 · W3D5 | Named experimental owner, preparation route, W7 round slot, credible Nov 18 readout plan | Mandatory lab deliverable at risk; escalate immediately if unconfirmed. |
| Oct 16 · W3D5 | Live trainable model list, account access and spending ceiling | Do not promise model identities or a matrix without these. |
| Oct 23 · W4D5 | Feasibility/novelty review and publication/data permissions | Resolve blockers before development expands. |
| Oct 29–Nov 2 · W5D4–W6D1 | Expert rubric and known-ground-truth worlds validated | Rubric Oct 29; worlds and real packets Nov 2; tools/signatures follow. |
| Oct 30 · W5D5 | Design-to-display compatibility reviewed | SynX preparation continues through Nov 6, with its actual lead times confirmed. |
| Nov 4 · W6D3 | Prospective designs, ranks and analysis plan frozen | No adaptation to observed lab outcomes. |
| Nov 6 · W6D5 | Both-family/historical demo, lab readiness and safety preregistration | London validates real constraints and experiments. |
| Nov 9–13 · W7 | Actual mRNA display round and expert reviews | Touring alone does not satisfy execution; exact slot requires confirmation. |
| Nov 18 · W8D3 | Analysable readout with QC/sample metadata | Use protected buffer; requirement stays incomplete until evidence exists. |
| Nov 19 · W8D4 | Harness parity/throughput check and matrix authorization | Launch the cohort that fits measured capacity/spend. |
| Nov 20 · W8D5 | Development-model high-budget labels | Fit forecasts Nov 23; keep held-out outcomes sealed if delayed. |
| Nov 25 · W9D3 | High-budget checkpoints and matched evaluation finished | Last two days are forecast/shift analysis. |
| Nov 27 · W9D5 | Scientific results and claim freeze | W10 reproduces, fixes correctness and communicates. |
| Dec 4 proposed · W10D5 | Talk, submission-ready paper, harness, artifact and handoff | Exact presentation day pending; if earlier in W10, pull final packaging/rehearsal forward. |

Experimental path: GATE-02 → GATE-04 → LAB-01 → LAB-02 → LAB-03 → MOL-07 → LAB-07 → LAB-11 → LAB-09 → GATE-10 → PAPER-12. LAB-01 was re-pointed on October 19 from MOL-01 to COSCI-07, because the round question needs a target, an endpoint and a statement of what one round can substantiate, all of which the dossier supplies and none of which needs the campaign export.

Blocked path: GATE-02 → MOL-01 → MOL-02/MOL-03 → MOL-04. MOL-01 cannot start without the SynX campaign export, and MOL-04 is the mandatory platform improvement. This is the project's most exposed dependency and it has no internal workaround.

Safety path: SAFE-01 → SAFE-02 → SAFE-03 → SAFE-04/SAFE-05 → SAFE-06 → SAFE-07 → SAFE-08 → TRAIN-02/TRAIN-03 → TRAIN-04/TRAIN-05 → GATE-09 → TRAIN-06 → TRAIN-07 → FCAST-03/FCAST-04 → GATE-10 → TRAIN-09/PAPER-09 → PAPER-12.

These are paths with little tolerance for delay, not a computed assertion that every task has zero float. Same-day dependencies require ordered handoffs. A task window is elapsed project time, not full-time work for its entire duration.

## Proposed experiment matrix

All counts below are proposed, to be frozen in SAFE-08 and authorized after TRAIN-05 measures cost/throughput. They are not grant requirements, booked jobs or existing datasets.

| Component | Concrete initial plan | Freeze / acceptance rule |
| --- | --- | --- |
| Models | Three supported open-weight configurations; candidates Qwen3.5-4B, Qwen3-8B, GPT-OSS-20B | GATE-05 confirms live access and records exact IDs, context, renderer, reasoning mode and LoRA settings. |
| Trajectories | 3 models × 2 benign curricula × 2 training seeds = 12 curves | Curriculum 1: conclusion/evidence reasoning; curriculum 2: grounded tool use/planning. Seeds/curricula are correlated units, not new model families. |
| Training budgets | Cumulative b, 4b, 16b, 64b; proposed b = 16,384 processed tokens | Checkpoints at 16,384 / 65,536 / 262,144 / 1,048,576 tokens. Twelve final curves consume 12,582,912 training tokens, not the sum of cumulative checkpoint labels. Repeated epochs count. |
| Model states | 48 trained checkpoints + 3 shared zero-shot bases = 51 unique states | Same base configuration on each curve; optimizer and sampler artifacts saved separately. |
| Core evaluation | 5 conditions × 10 paired cases × 2 rollout seeds per state = 5,100 episodes | Proposed case allocation: 4 development world families, 3 unseen mechanism families, 3 audited real studies. Across conditions, variants remain matched where valid. Cells/case variants are not independent studies. |
| Stress conditions | Clean, confounding, mechanism shift, evidence corruption, longer horizon | Change one stress mechanism at a time; fixed tools/quality. Known synthetic latent answers; uncertain real-study references. |
| Primary inference budget | Proposed 4,096 generated tokens total and 8 tool calls per episode | Count total episode tokens/calls and prefill separately. Longer horizon changes required reasoning without silently adding budget. Score errors/timeouts. |
| Inference control | All 3 bases + 3 preregistered final-checkpoint representatives repeat 100 episodes/state at 8,192 generated tokens and 16 calls | 600 additional episodes; total proposed evaluation 5,700 before smoke/retry overhead. Separates post-training from extra inference. |
| Uplift | C = negative mean frozen loss; U(BT, BI) = C(post-trained, BI) − C(base, BI) | Report conclusion, grounding and process losses separately; scalar weights/normalization frozen on development cases. |
| Forecast inputs / targets | Inputs: zero-shot signatures and/or b/4b gains. Targets: 16b/64b gains | Hold out entire model/curriculum groups, never budget points or rollout seeds. Save held-out predictions before reading their high-budget targets. |
| Forecast baselines | Constant/mean, low-budget extrapolation, zero-shot-only, low-budget-only, combined | Few prespecified features, regularized fits, development-only selection. Three models and twelve correlated curves support a pilot, not a universal ceiling predictor. |
| Shift and error analysis | Unseen mechanisms/studies; signature removal; tool audit; resource–capability frontier | MAE/RMSE, rank error, interval coverage, decomposed losses and costs. Intervals grouped by independent worlds/studies and model trajectories; report individual errors and small-sample limits. |

**Forecast protocol clarification:** the forecaster needs development high-budget labels. TRAIN-11 therefore extends the eight curves from two preregistered development models to 16b/64b by W8D5; these are part of the twelve-curve total. FCAST-02 fits only their development-case outcomes, using at most two prespecified features, and saves predictions before revealing high-budget outcomes for the third model or unseen cases. TRAIN-07 finishes the four held-out-model curves in W9. Generating zero-shot and cheap low-budget signatures on an evaluation case is permitted; fitting to its high-budget outcome is not. Further leave-model-out folds are preregistered cross-validation diagnostics, clearly distinguished from the one prospective held-out-model prediction. Three models do not establish broad cross-family generalization. If development labels are delayed, delay fitting and keep held-out results sealed; never pretend an untrained forecasting rule has been fitted.

TRAIN-05/GATE-09 must measure average prefill/output tokens, tool duration, service concurrency and retries, with 20% compute/spend headroom. If the matrix cannot fit, retain every model, curriculum, budget and stress condition; reduce paired case counts evenly and register the narrower evidence before high-budget scoring. Do not silently drop mandatory SynX outputs or failing conditions. Claims shrink with evidence.

Current Tinker documentation lists the local pilot's Qwen3-4B-Instruct-2507 as retired. Recheck model availability at access and launch. [Models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/), [checkpoint semantics](https://tinker-docs.thinkingmachines.ai/tinker/howto/checkpoints/).

## Public data starting points

| Collection | Specific use | Evidence boundary |
| --- | --- | --- |
| Existing ProteinGym GB1 / ATLAS manifests | Component decision/resource-allocation baselines | Binding measurements and held-out MD replicates do not establish antiviral activity or transcriptomic causal truth. |
| scPerturb: NormanWeissman2019 (GSE133344), DixitRegev2016 (GSE90063), ReplogleWeissman2022 | Proposed held-out benign perturbation-response packets | Audit versions/quality first; keep studies separate. Measured response need not identify a unique biological mechanism. [Paper and data access](https://www.nature.com/articles/s41592-023-02144-y), [official resource](https://projects.sanderlab.org/scperturb/). |
| CELLxGENE Census | Cell/donor/tissue background/reference context | Pin LTS build date; retain donor/study provenance and primary-data status. Do not assume perturbation cell-line datasets are in Census. [Releases](https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html), [schema](https://github.com/chanzuckerberg/cellxgene-census/blob/main/docs/cellxgene_census_schema.md). |
| Synthetic counterfactual worlds | Primary mechanistic ground truth and controlled stress | Known latent mechanisms, held-out world families and independently validated oracle. |

These are selected planning sources, not downloaded/audited transcriptomic datasets. This task creates a plan; it does not launch or spend on training.

## Weekly execution focus, re-adjusted October 19

Each cell names the work, and the task ids in the register say which stream it belongs to.
W4 and W5 are re-adjusted from the October 14 baseline for two reasons: the domain layer
(COSCI-13 to COSCI-15, MOL-08) landed on October 19 and was not in the baseline, and the
MOL audit path is blocked on the SynX campaign export. Weeks 6 to 10 are unchanged.

| Week | D1 | D2 | D3 | D4 | D5 |
| --- | --- | --- | --- | --- | --- |
| 1 · Sep 28–Oct 2 | Elapsed | Elapsed | Elapsed | Elapsed | Completion unverified |
| 2 · Oct 5–9 | Elapsed | Oct 5 meeting evidence | Elapsed | Elapsed | Oct 9 Portal-demo evidence |
| 3 · Oct 12–16 | Existing HPC staging | Earlier work not reconstructed | Verify scope/pilots; task/loss specs | Access requests; residue/data audit | Lab/model gates; start baseline extension and transcriptomics audit |
| 4 · Oct 19–23 | **Done:** dossier state, pathogen environments, stability metrics, block registry. Tinker smoke (TRAIN-01) | Tinker smoke closes; development episodes (COSCI-03) | Evidence-state ledger (COSCI-04); transcriptomic splits (SAFE-03); **export blocker call** | Family sign-off closes (COSCI-07); round question (LAB-01); paper intro (PAPER-03) | Feasibility and novelty gate (GATE-06) with the export blocker named; emergency stress (COSCI-05); begin synthetic worlds (SAFE-04) |
| 5 · Oct 26–30 | LLM policy over the six-action space (COSCI-06) | MOL audit path if the export arrived, else re-scope MOL-04; platform choice (GATE-07) | COVID replay (COSCI-08); hindsight audit for named targets | Expert rubric (COSCI-09); real packets (SAFE-05); artifact skeleton (PAPER-05) | Integrated replay (COSCI-10); display compatibility (LAB-03); methods draft |
| 6 · Nov 2–6 | World/tool handoff; platform evaluation | Freeze tools; final platform eval; system eval starts | Predictions frozen; signatures; system eval | Demo/signatures complete; London pack | Safety protocol/readiness gate; talk outline |
| 7 · Nov 9–13 | Proposed travel/arrival | Tour + design review | Observe round; discrepancies | Round; data handoff; safety review | Proposed return; archive case notes |
| 8 · Nov 16–20 | Train/eval adapters | Checkpoints/recovery/sampling | Parity/scorer; lab data due | Throughput gate; low-budget cohort | Development high-budget labels; lab analysis |
| 9 · Nov 23–27 | Save forecasts; held-out training | Higher budgets; results scaffold | 16b/64b finish; inference controls | Forecast/shift/signature analysis | Frontier, independent audit, results freeze |
| 10 · Nov 30–Dec 4 | Clean reproduction; paper | Harness example; release candidate | Expert review; timed rehearsal | Deck, recording, offline rehearsal | Proposed final talk/handoff |

## Mandatory-output acceptance map

| Required output | Evidence / scheduled completion |
| --- | --- |
| Modular emergency scientist | Decision state and action space in COSCI-13; stability scoring in COSCI-15; COSCI-10 replay; COSCI-12 demo frozen W6D4 |
| Evidence-backed dossiers | Seven-axis packets prepared in COSCI-07, three families; COSCI-07 sign-off and MOL-06 finalise both chosen families |
| Non-canonical macrocycles / focused display library | Block registry and library specification in MOL-08; MOL-06/MOL-07 and LAB-03 SynX compatibility; prospective freeze W6D3 |
| Historical COVID evaluation | COSCI-08/COSCI-11 plus hindsight audit |
| Two prototype-family case studies | COSCI-07/COSCI-11/MOL-06 with scored outputs |
| SynX platform improvement | MOL-04/MOL-05 implemented patch and independent evaluation |
| Actual mRNA display round, analyzed | LAB-07 execution; LAB-11 readout; LAB-09 analysis and decision update by W8D5 |
| Grant stress signatures/uplift curves | SAFE-07, TRAIN-06/TRAIN-07 matched zero-shot and cumulative training |
| Forecasting and shift failures | FCAST-02–FCAST-06 locked predictions and grouped evaluation |
| Research product and presentation | PAPER-09–PAPER-12 paper, clean artifact, recorded/offline demo |

No stretch goal is dropped. If a collaborator-dependent obligation fails, record it as incomplete rather than replacing it with simulation. A late readout or unavailable Tinker account is a real risk, not solved by a more detailed chart.

## Outstanding confirmations and change control

- Exact London itinerary, presentation day and talk duration; calendar anchor is confirmed as September 28.
- Named SynX staff, booked slot, preparation/readout/QC turnaround and data/publication permissions. User-reported availability is not a booking.
- Live Tinker account/model access, rate/concurrency limits, spend ceiling and checkpoint retention/export terms. No secrets belong in this plan.
- Separate few-shot proposal when uploaded; reserve bounded MOL-05 integration without shifting final delivery.
- Actual conference call and submission dates; final handoff is submission-ready, not an invented venue deadline.
- At each gate, update hours and blockers against evidence. Correctness fixes reopen results only with a versioned explanation and regenerated affected figures.

W9 includes November 26–27, the U.S. Thanksgiving period. Book the result reviewer before London, and use a confirmed available SynX/UK reviewer or advance review on November 25 rather than assuming Yale collaborators will be available on November 27.

## Schedule verification

The task register has been checked for missing/cyclic dependencies, task-window ordering and the fixed W7/W10 boundaries; the generator asserts all three and fails rather than emitting an inconsistent chart. The register now holds 82 tasks and 273 estimated hands-on hours, of which 14 are delivered and excluded from the weekly load. An earlier effort-allocation check found a feasible distribution of the then-261 hours within eight-hour working days, with no coding capacity on the proposed arrival day and four hours on the proposed return day. That validates the estimates' internal consistency; it does not verify biological turnaround or API throughput, and it does not rescue W4, which is overcommitted for the reason given under capacity below. Weekly chart values are uniform-window workload estimates, not fixed appointments. The interactive chart was last checked at desktop and mobile widths before the October 19 re-adjustment; its task-detail controls have not been re-exercised since.

<!-- GENERATED REGISTER -->
## Capacity and reserved buffer

Hours below are work **remaining**. The 14 hours already delivered against tasks marked Complete are excluded.

| Week | Remaining hands-on hours | Unallocated capacity within a 40-hour week |
| --- | ---: | ---: |
| W1 | Elapsed; not reconstructed | Completion not inferred |
| W2 | Elapsed; not reconstructed | Completion not inferred |
| W3 | 23.4 h | 0.6 h across Oct 14–16 |
| W4 | 38.5 h | 1.5 h |
| W5 | 38.4 h | 1.6 h |
| W6 | 39.8 h | 0.2 h |
| W7 | 24.3 h | Travel/on-site contingency; not extra coding capacity |
| W8 | 32.6 h | 7.4 h |
| W9 | 34.0 h | 6.0 h |
| W10 | 28.0 h | 12.0 h |

**W4 does not fit, and the reason matters.** 14 hours went into the domain layer on October 19, which was not in the baseline plan. The remaining 38.5 hours shown for W4 still include MOL-01, MOL-02 and MOL-03, and those cannot start until the SynX campaign export arrives. The four gates (TRAIN-01, COSCI-07 sign-off, LAB-01, GATE-06) plus COSCI-03, COSCI-04 and the SAFE-02/SAFE-03 audits are the week's real load. If the export has not arrived by October 21, the MOL path slides to W5 and MOL-04, the mandatory platform improvement, loses its slack — record it as a blocker with an owner at GATE-06 rather than absorbing it silently.

W7 effort is on-site reviews, observation and light documentation, with travel excluded. W8/W9 buffer absorbs API retries and late experimental QC; W10 buffer absorbs reproduction and review fixes. Average hours within a window are workload estimates, not an appointment calendar. Rely on existing tool code and reviewer/lab support, not a new foundation model or a general-purpose autonomous lab.

## Detailed task register

A task id states its own stream, so no lookup table is needed: **GATE** decisions and access, **COSCI** the emergency co-scientist, **MOL** macrocycle design and platform, **LAB** the SynX experiment and London, **SAFE** the safety benchmark, **TRAIN** the training harness, **FCAST** forecasting, **PAPER** the write-up.

All task estimates refer to William's hands-on contribution; collaborator ownership is stated explicitly. A task is `Planned` unless a **Status** line says otherwise, and a status other than Planned always names its evidence. Every other artifact is a planned output name, not an assertion that the file exists. Same-day dependencies require ordered handoffs within that day.

### GATE — Decisions, access and gates

#### GATE-01 — Lock the two contributions and all required outputs

**Window:** 2026-10-14–2026-10-14 (W3D3–W3D3) · **Effort:** 1 h · **Owner:** William

**Depends on:** None. **Planned output:** `scope_and_claims.md`.

**Done when:** Askar reviews one primary forecasting claim, one applied emergency-scientist validation, and the seven mandatory SynX outputs; no promise of publication acceptance.

#### GATE-02 — Request access, lab slot and publication permissions

**Window:** 2026-10-14–2026-10-15 (W3D3–W3D4) · **Effort:** 1 h · **Owner:** William + Askar + SynX

**Depends on:** GATE-01. **Planned output:** `access_and_owners.csv`.

**Done when:** Named SynX experimental owner; Portal/API, paired data, residue assets, compute and Tinker access requests recorded; availability is not assumed.

#### GATE-03 — Audit pilot evidence and establish reproducible run ledger

**Window:** 2026-10-14–2026-10-15 (W3D3–W3D4) · **Effort:** 1 h · **Owner:** William

**Depends on:** None. **Planned output:** `pilot_status.json + run_schema.json`.

**Done when:** Check actual scheduler/output evidence; separate existing CPU pilot results from queued/unverified GPU runs; commit hashes, seeds, data versions and costs recorded.

#### GATE-04 — Urgent: confirm lab route, lead times and data return

**Window:** 2026-10-16–2026-10-16 (W3D5–W3D5) · **Effort:** 1 h · **Owner:** William + SynX experimental lead · **decision gate**

**Depends on:** GATE-02. **Planned output:** `SynX experiment calendar`.

**Done when:** SynX confirms a feasible library/reagent preparation route, one selection round by W7, and usable readout by W8D3. If unavailable, mark mandatory lab deliverable at risk immediately.

#### GATE-05 — Select live Tinker models and cap total spend

**Window:** 2026-10-15–2026-10-16 (W3D4–W3D5) · **Effort:** 2 h · **Owner:** William · **decision gate**

**Depends on:** GATE-02. **Planned output:** `models_and_budget.json`.

**Done when:** Query actual trainable capabilities; freeze three supported open-weight model IDs/configurations, LoRA settings and renderer modes, or explicitly narrow generalization claims. Record an approved spend ceiling and throughput estimate.

**Constraint:** Local Qwen3-4B-Instruct-2507 is a pilot comparator, not an assumed Tinker base; current docs list it as retired.

#### GATE-06 — W4 feasibility and novelty review after catch-up

**Window:** 2026-10-23–2026-10-23 (W4D5–W4D5) · **Effort:** 2 h · **Owner:** William + Askar · **decision gate**

**Depends on:** GATE-04, GATE-05, COSCI-03, SAFE-02, PAPER-02. **Planned output:** `gate_w4.md`.

**Done when:** Confirm lab slot, data rights, scorer validity, trainable models, and a distinguishable forecasting question. Record unresolved blockers with owner and deadline.

#### GATE-07 — Choose the platform improvement from measured failures

**Window:** 2026-10-27–2026-10-27 (W5D2–W5D2) · **Effort:** 1 h · **Owner:** William + SynX · **decision gate**

**Depends on:** MOL-02, COSCI-05. **Planned output:** `platform_change_spec.md`.

**Done when:** Select one bounded improvement with a failing baseline, implementation estimate and independent validation metric; retain residue/rotamer limitations as measured hypotheses.

#### GATE-08 — Freeze SynX scientific scope and London readiness

**Window:** 2026-11-06–2026-11-06 (W6D5–W6D5) · **Effort:** 2 h · **Owner:** William · **decision gate**

**Depends on:** COSCI-12, MOL-07, LAB-04, SAFE-08, LAB-02. **Planned output:** `gate_w6.md`.

**Done when:** Both family dossiers, historical replay, improvement evaluation, prospective prediction file and safety benchmark manifest are complete; London agenda accepted; no new feature scope.

#### GATE-09 — Authorize the measured training/evaluation matrix

**Window:** 2026-11-19–2026-11-19 (W8D4–W8D4) · **Effort:** 1 h · **Owner:** William · **decision gate**

**Depends on:** TRAIN-03, TRAIN-04, TRAIN-05. **Planned output:** `training_gate.json`.

**Done when:** Measured API throughput and token counts fit the spend ceiling and W9D3 finish; inference protocol and forecast train/test groups frozen before seeing high-budget results.

#### GATE-10 — Freeze results and final claims

**Window:** 2026-11-27–2026-11-27 (W9D5–W9D5) · **Effort:** 2 h · **Owner:** William + Askar · **decision gate**

**Depends on:** FCAST-05, FCAST-06, LAB-09, PAPER-08. **Planned output:** `results_freeze.json + claims_audit.md`.

**Done when:** Each figure traces to an immutable run manifest; negative results retained; missing required outputs explicitly flagged; paper and presentation use the same claims.

### COSCI — Emergency co-scientist and benchmark

#### COSCI-01 — Specify emergency episodes and decision rubric

**Window:** 2026-10-14–2026-10-15 (W3D3–W3D4) · **Effort:** 3 h · **Owner:** William

**Depends on:** GATE-01. **Planned output:** `emergency_episode_schema.json + rubric.md`.

**Done when:** Specify timestamps, hidden outcomes, evidence state, tool permissions, action budget, readiness rubric and failure categories; reviewers can score a toy episode consistently.

#### COSCI-02 — Freeze public data provenance and split policy

**Window:** 2026-10-14–2026-10-15 (W3D3–W3D4) · **Effort:** 2 h · **Owner:** William

**Depends on:** None. **Planned output:** `dataset_manifest.json + split_policy.md`.

**Done when:** ProteinGym GB1 and ATLAS remain component pilots; historical/public evidence collections are source-versioned; split by parent/backbone, target/family or study as appropriate.

#### COSCI-03 — Build small development episode set and baselines

**Window:** 2026-10-16–2026-10-21 (W3D5–W4D3) · **Effort:** 6 h · **Owner:** William

**Depends on:** COSCI-01, COSCI-02. **Planned output:** `dev_episodes/ + baseline_scores.csv`.

**Done when:** Twelve proposed development episodes span incomplete evidence, contradictions, deadlines and missing tools; fixed rubric, scripted workflow and single-response retrieval baselines run with matched access.

**Constraint:** Twelve is a planning target, not an existing dataset count; freeze final sample size after the audit.

#### COSCI-04 — Implement evidence state and resource-accounted tools

**Window:** 2026-10-16–2026-10-22 (W3D5–W4D4) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-01, GATE-03. **Planned output:** `evidence_state + tool_event ledger`.

**Done when:** Every retrieval/structure/design action updates cited evidence, uncertainty and actual token/tool/wall-time cost; invalid actions are logged and charged.

#### COSCI-05 — Run emergency stress and public-pilot comparisons

**Window:** 2026-10-23–2026-10-26 (W4D5–W5D1) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-03, COSCI-04. **Planned output:** `stress_baselines.csv + pilot_comparison.md`.

**Done when:** Compare complete versus degraded evidence and immediate versus delayed feedback; report quality/cost fronts and failure cases; GB1/ATLAS conclusions stay within their measured endpoints.

#### COSCI-06 — Build adaptive action policy v1

**Window:** 2026-10-23–2026-10-29 (W4D5–W5D4) · **Effort:** 6 h · **Owner:** William

**Depends on:** COSCI-04, COSCI-13, COSCI-15. **Planned output:** `agent_policy_v1 + replay_trace`.

**Done when:** Policy can request evidence, rerank, allocate a calculation and stop with uncertainty; all four actions complete an end-to-end development replay.

**Constraint:** Extends the dossier policy in COSCI-13 over the six-action space; it is not a new general-purpose scientist framework. The deterministic baselines already pass the stability metric by construction, so this task is judged on whether the LLM policy holds the recommendation under irrelevant evidence at matched cost.

#### COSCI-07 — Curate two prototype-family evidence packets

**Window:** 2026-10-19–2026-10-22 (W4D1–W4D4) · **Effort:** 2 h · **Owner:** William + Askar/SynX review

**Depends on:** COSCI-02. **Planned output:** `family_1/ + family_2/ evidence dossiers`.

**Done when:** Two distinct families chosen with Askar/SynX; each has sourced essentiality, accessibility, conservation, tractability, uncertainty and a held-out episode. Selection criteria documented.

**Status:** Prepared, pending sign-off. data/processed/targets/{coronaviridae,orthomyxoviridae,henipavirus}.json: 11 targets across 3 families, 59 sourced evidence entries against the seven-axis rubric, 8 update probes, 3 claims flagged as needing a citation. Awaiting the family decision with Askar and SynX.

#### COSCI-08 — Assemble time-stamped COVID historical replay

**Window:** 2026-10-16–2026-10-29 (W3D5–W5D4) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-02. **Planned output:** `covid_2020_replay/ + hindsight_audit.md`.

**Done when:** January-March 2020 retrieval cutoffs verified; checkpoint decisions scored against information available then; pretrained hindsight limitations explicitly retained.

#### COSCI-09 — Calibrate expert scoring and freeze held-out rubric

**Window:** 2026-10-29–2026-10-29 (W5D4–W5D4) · **Effort:** 2 h · **Owner:** William + two expert reviewers

**Depends on:** COSCI-05, COSCI-07, COSCI-08. **Planned output:** `expert_calibration.csv + frozen_rubric.json`.

**Done when:** At least two reviewers score a calibration subset; disagreements and score definitions resolved on development outputs; held-out labels remain sealed.

#### COSCI-10 — Integrate policy with frozen molecular tools

**Window:** 2026-10-30–2026-10-30 (W5D5–W5D5) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-06, MOL-03. **Planned output:** `end_to_end_replay.jsonl`.

**Done when:** Both family episodes pass intake → evidence → target ranking → design recommendation; policy comparisons share tool/model versions.

#### COSCI-11 — Run held-out decisions, family transfer and ablations

**Window:** 2026-11-03–2026-11-04 (W6D2–W6D3) · **Effort:** 7 h · **Owner:** William

**Depends on:** COSCI-09, COSCI-10, MOL-05. **Planned output:** `system_eval.csv + decision_quality_figures/`.

**Done when:** All four baselines evaluated on historical and two-family held-out episodes; ablate adaptation and evidence state; report quality, evidence support, latency, cost and reviewer uncertainty.

#### COSCI-12 — Freeze the emergency scientist demo and findings

**Window:** 2026-11-05–2026-11-05 (W6D4–W6D4) · **Effort:** 3 h · **Owner:** William · **decision gate**

**Depends on:** COSCI-11. **Planned output:** `synx_demo_v1 + limitations.md`.

**Done when:** A recorded replay and offline fallback work; no antiviral efficacy claim without an assay; central plots and errors are ready before London.

#### COSCI-13 — Specify the target dossier decision state and action space

**Window:** 2026-10-19–2026-10-19 (W4D1–W4D1) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-01. **Output:** `hpc_pilots/dossier.py`.

**Done when:** Decision state is the seven-axis dossier with per-axis citations, timestamps and open uncertainties; action space is retrieve evidence, request structure, rerank, invoke design tool, recommend experiment, stop. Evidence dated after the decision clock is refused and the refusal is recorded. Hard-gate axes block a ranking.

**Status:** Complete. hpc_pilots/dossier.py with four isolation tests; date-boundary and conflict-retention tests pass.

**Constraint:** Named targets reintroduce pretrained-hindsight risk that the date boundary does not control; it needs a separate audit in COSCI-08.

#### COSCI-14 — Build pathogen decision environments and the subtype transfer split

**Window:** 2026-10-19–2026-10-19 (W4D1–W4D1) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-02. **Output:** `hpc_pilots/configs/targets.json + data/processed/targets/`.

**Done when:** Mpro, HA H1 and HA H3 staged from the ProteinGym archive as disjoint-pool decision environments with per-landscape assay endpoints recorded; pool geometry shrinks rather than reusing candidates across pools described as disjoint; H1-to-H3 is a grouped study-and-subtype holdout.

**Status:** Complete. 10 pools for Mpro, 19 for HA H1, 20 for HA H3; 240 paired replays recorded under runs/targets/.

#### COSCI-15 — Score evidence-update stability in emergency episodes

**Window:** 2026-10-19–2026-10-19 (W4D1–W4D1) · **Effort:** 3 h · **Owner:** William

**Depends on:** COSCI-13, COSCI-14. **Output:** `hpc_pilots/emergency.py + runs/emergency/`.

**Done when:** Episodes release relevant, contradictory and irrelevant evidence at timed checkpoints under a deadline and an action budget; the policy is scored on updating when relevant evidence arrives, holding the recommendation when irrelevant evidence arrives, and retaining contradictory evidence beside the claim it contests.

**Status:** Complete. Six episodes across three families; deterministic baselines hold the recommendation but reorder the tail, and the scripted workflow misses the deadline.

**Constraint:** Deterministic baselines are stable by construction; the metric earns its value on the LLM policy in COSCI-06.

### MOL — Macrocycle design and platform

#### MOL-01 — Audit residue registry, scores and experimental endpoints

**Window:** 2026-10-15–2026-10-21 (W3D4–W4D3) · **Effort:** 4 h · **Owner:** William

**Depends on:** GATE-02. **Planned output:** `molecular_audit.md + chemistry_registry.csv`.

**Done when:** Identity/stereochemistry/codon mapping, scorefunction, rotamer assets, parent groups, assay endpoints, replicates, missingness and censoring audited; usable campaign tables identified.

**Status:** Blocked on SynX export.

**Constraint:** Blocked until the campaign export arrives. It blocks MOL-02, MOL-03 and therefore MOL-04, the mandatory platform improvement. The building-block registry in MOL-08 holds the schema this export fills.

#### MOL-02 — Measure molecular ranking and representation failures

**Window:** 2026-10-21–2026-10-26 (W4D3–W5D1) · **Effort:** 5 h · **Owner:** William

**Depends on:** MOL-01. **Planned output:** `molecular_baselines.csv + stratified_hexbin/`.

**Done when:** Rosetta ranking and simple chemistry/similarity/supervised baselines use grouped splits; inspect motif/backbone/campaign confounding and unsupported residue representations.

#### MOL-03 — Freeze design/refolding/filter contracts

**Window:** 2026-10-22–2026-10-27 (W4D4–W5D2) · **Effort:** 4 h · **Owner:** William

**Depends on:** MOL-01. **Planned output:** `design_tool_contract.json`.

**Done when:** RFdiffusion/BoltzGen version and actual chemistry support documented; refolding and geometric filters are separate logged stages; generation outputs are compatible with the permitted SynX route.

#### MOL-04 — Implement one justified SynX platform improvement

**Window:** 2026-10-27–2026-10-30 (W5D2–W5D5) · **Effort:** 6 h · **Owner:** William

**Depends on:** GATE-07, MOL-03. **Planned output:** `platform_patch + regression_examples/`.

**Done when:** Fix the selected observed failure, such as residue validation or uncertainty-aware campaign ranking; run affected failing examples and a clean unaffected control set.

#### MOL-05 — Evaluate improvement and low-label ranking module

**Window:** 2026-11-02–2026-11-03 (W6D1–W6D2) · **Effort:** 3 h · **Owner:** William

**Depends on:** MOL-04, MOL-02. **Planned output:** `platform_ablation.csv + label_budget_eval.csv`.

**Done when:** Before/after evaluated with independent endpoint and matched costs; zero/few-shot label budgets and grouped transfer are reported if paired data support them; newly uploaded proposal incorporated before freezing.

**Constraint:** No separate few-shot proposal is currently reviewed. This schedule reserves integration time; an unsupported module cannot substitute for the required platform improvement.

#### MOL-06 — Finalize both family target dossiers and library proposal

**Window:** 2026-10-27–2026-10-30 (W5D2–W5D5) · **Effort:** 3 h · **Owner:** William + SynX design review

**Depends on:** COSCI-07, MOL-03. **Planned output:** `target_dossiers/ + library_design_v1.csv`.

**Done when:** Both families include evidence-backed target/epitope ranks; selected non-canonical macrocyclic library is compatible with SynX display constraints and independently reviewed.

**Constraint:** Requires SynX design staff to prepare/review the compatible library; the estimate is William’s review/integration time, not full design-team effort.

#### MOL-07 — Freeze prospective candidate ranks and analysis plan

**Window:** 2026-11-04–2026-11-04 (W6D3–W6D3) · **Effort:** 3 h · **Owner:** William · **decision gate**

**Depends on:** MOL-05, MOL-06, LAB-03. **Planned output:** `prospective_predictions.csv + analysis_plan.md`.

**Done when:** Immutable candidates/ranks, controls, intended measured endpoints and analysis rules recorded before laboratory outcomes; retain all submitted candidates and failures.

#### MOL-08 — Represent non-canonical blocks and read out enrichment

**Window:** 2026-10-19–2026-10-19 (W4D1–W4D1) · **Effort:** 3 h · **Owner:** William

**Depends on:** None. **Output:** `hpc_pilots/chemistry.py + enrichment metrics`.

**Done when:** Building-block registry covers canonical and non-canonical classes with initiator and closure roles and every platform-supplied field explicitly pending; the encoding extends past the canonical twenty; acquisition metrics report an enrichment ratio against the pool baseline alongside the per-candidate view.

**Status:** Complete. 29 blocks, 9 non-canonical; enrichment ratio added to every binding and target checkpoint; the recorded GB1 replays still reproduce bit-identically.

**Constraint:** Chemical identifiers, codons, Rosetta parameters and rotamer libraries stay null until the MOL-01 export arrives. No measured macrocycle outcome exists yet, so no enrichment figure here is a macrocycle result.

### LAB — SynX experiment and London

#### LAB-01 — Agree the single-round experimental question

**Window:** 2026-10-22–2026-10-23 (W4D4–W4D5) · **Effort:** 2 h · **Owner:** William + SynX experimental lead

**Depends on:** GATE-04, COSCI-07. **Planned output:** `round_question_and_endpoints.md`.

**Done when:** SynX confirms the designed library, feasible target/assay endpoint, controls, replicates, primary readout and what one round can substantiate; no invented affinity/antiviral endpoint.

**Constraint:** Re-pointed from MOL-01 to COSCI-07 on October 19: the experimental question is answerable from the dossier and does not need the campaign export. Still gated on GATE-04, the named experimental owner.

#### LAB-02 — SynX prepares reagents, library and booked slot

**Window:** 2026-10-16–2026-11-06 (W3D5–W6D5) · **Effort:** 2 h · **Owner:** SynX experimental lead; William tracks

**Depends on:** GATE-04. **Planned output:** `lab_readiness_tracker.csv`.

**Done when:** SynX staff report validated inputs, shipping/reagent/library lead times and a reserved W7 selection slot; William's effort is coordination only.

**Constraint:** SynX can prepare available inputs after the route gate while the final question/designs are refined; preparation must finish by W6D5. This overlap requires the experimental lead’s confirmation.

#### LAB-03 — Review design-to-display compatibility

**Window:** 2026-10-30–2026-10-30 (W5D5–W5D5) · **Effort:** 1 h · **Owner:** William + SynX experimental lead · **decision gate**

**Depends on:** GATE-04, MOL-06. **Planned output:** `compatibility_signoff.md`.

**Done when:** Experimental lead signs off library representation, chemical alphabet, input format and endpoint; any incompatible designs revised before prediction freeze.

#### LAB-04 — Prepare London dossier and book review agenda

**Window:** 2026-11-04–2026-11-05 (W6D3–W6D4) · **Effort:** 2 h · **Owner:** William

**Depends on:** MOL-07, LAB-03. **Planned output:** `london_review_pack/ + visit_agenda.md`.

**Done when:** Send-ready pack includes two-family dossiers, model limitations, platform before/after, frozen prospective predictions, assay questions and W8 data-handoff checklist.

#### LAB-05 — London travel and arrival

**Window:** 2026-11-09–2026-11-09 (W7D1–W7D1) · **Effort:** 0 h · **Owner:** William

**Depends on:** GATE-08. **Planned output:** `arrival + access confirmed`.

**Done when:** Travel day protected from coding or critical job launches; actual itinerary to replace project-day placeholder.

**Constraint:** W7D1 travel placement is proposed; the visit week is fixed, specific travel dates are not.

#### LAB-06 — London: facility tour and design/experiment review

**Window:** 2026-11-10–2026-11-10 (W7D2–W7D2) · **Effort:** 6 h · **Owner:** William + SynX

**Depends on:** LAB-05, LAB-04. **Planned output:** `facility_constraints.md + review_actions.md`.

**Done when:** Walk through portal-to-wet-lab handoff with SynX; reconcile predicted versus actual constraints; obtain expert review of both dossiers and the frozen experiment plan.

#### LAB-07 — London: observe the required display round

**Window:** 2026-11-10–2026-11-12 (W7D2–W7D4) · **Effort:** 5 h · **Owner:** SynX experimental lead; William observes

**Depends on:** LAB-06, MOL-07, LAB-03, LAB-02. **Planned output:** `selection_round_record + sample manifest`.

**Done when:** SynX executes one actual mRNA display selection round against the designed library; record performed date, inputs, controls, failures and sample provenance. Visit alone does not meet this requirement.

**Constraint:** Exact execution day and duration must be confirmed at GATE-04; the Gantt window is not evidence of booking.

#### LAB-08 — London: review discrepancies and secure data handoff

**Window:** 2026-11-11–2026-11-12 (W7D3–W7D4) · **Effort:** 5 h · **Owner:** William + SynX

**Depends on:** LAB-06. **Planned output:** `expert_discrepancy_log.md + data_handoff_contract.json`.

**Done when:** Review unexplained score/measurement differences and platform patch; agree raw and processed readout format, QC metadata, owner and delivery by W8D3.

#### LAB-10 — Return travel and archive the London review

**Window:** 2026-11-13–2026-11-13 (W7D5–W7D5) · **Effort:** 1 h · **Owner:** William

**Depends on:** LAB-07, LAB-08. **Planned output:** `london_summary.md + archived review pack`.

**Done when:** Capture accepted changes and experimental limitations; return travel protected; no major new architecture work.

#### LAB-11 — SynX processes and returns selection-round readout

**Window:** 2026-11-12–2026-11-18 (W7D4–W8D3) · **Effort:** 1 h · **Owner:** SynX experimental lead; William receives

**Depends on:** LAB-07. **Planned output:** `raw_readout/ + QC + manifest`.

**Done when:** Receive analysable readout and sample/control metadata by W8D3; QC failures stay in the report. SynX confirms actual processing/sequence turnaround before the slot.

**Constraint:** If readout is late, experimental obligation stays incomplete and takes protected buffer; it is not relabeled optional.

#### LAB-09 — Analyze the round and update the decision state

**Window:** 2026-11-18–2026-11-20 (W8D3–W8D5) · **Effort:** 4 h · **Owner:** William

**Depends on:** LAB-11, MOL-07. **Planned output:** `prospective_results.csv + before_after_decisions.md`.

**Done when:** Compare preregistered ranks with the measured endpoint, quantify uncertainty at independent units, include controls/failures and show how evidence changes the recommendation.

### SAFE — Safety benchmark preparation

#### SAFE-01 — Translate grant into benign task/loss specification

**Window:** 2026-10-14–2026-10-16 (W3D3–W3D5) · **Effort:** 3 h · **Owner:** William

**Depends on:** GATE-01. **Planned output:** `safety_task_spec.md + loss_components.json`.

**Done when:** Define conclusion accuracy, evidence grounding and process validity separately; scalar normalization/weights and threat-claim boundary documented; supervised biology reasoning is benign.

#### SAFE-02 — Select public transcriptomic collections and worlds

**Window:** 2026-10-16–2026-10-21 (W3D5–W4D3) · **Effort:** 4 h · **Owner:** William

**Depends on:** SAFE-01. **Planned output:** `transcriptomic_manifest.json`.

**Done when:** Audit scPerturb NormanWeissman2019 (GSE133344), DixitRegev2016 (GSE90063), and ReplogleWeissman2022; confirm versions, perturbation/metadata availability, licenses and annotation quality before using them. Pin a CELLxGENE LTS build for background/reference packets only.

**Constraint:** Public atlas candidates are starting points; observational annotations are not automatic causal ground truth.

#### SAFE-03 — Specify train/test boundaries and counterfactual mechanisms

**Window:** 2026-10-21–2026-10-22 (W4D3–W4D4) · **Effort:** 3 h · **Owner:** William

**Depends on:** SAFE-02. **Planned output:** `world_families.json + split_manifest.json`.

**Done when:** Split by study/donor/mechanism family, never random cells; paired synthetic worlds share observable correlations while latent mechanisms differ; inference and training budgets are separate fields.

#### SAFE-04 — Build and validate small synthetic world generator

**Window:** 2026-10-23–2026-11-02 (W4D5–W6D1) · **Effort:** 6 h · **Owner:** William

**Depends on:** SAFE-03. **Planned output:** `counterfactual_worlds/ + oracle_validation.json`.

**Done when:** Oracle recovers known latent answers; evidence disclosure and confounding/corruption/shift/horizon knobs verified on development worlds; held-out mechanisms remain sealed.

#### SAFE-05 — Prepare real-study task packets and expert references

**Window:** 2026-10-29–2026-11-02 (W5D4–W6D1) · **Effort:** 4 h · **Owner:** William + biological reviewer

**Depends on:** SAFE-03. **Planned output:** `real_study_packets/ + reference_provenance.csv`.

**Done when:** At least three audited studies proposed for generalization; references distinguish measured perturbation evidence from expert interpretation; metadata leakage and donor overlap checked.

#### SAFE-06 — Freeze tool primitives, signatures and budget accounting

**Window:** 2026-11-02–2026-11-03 (W6D1–W6D2) · **Effort:** 4 h · **Owner:** William

**Depends on:** SAFE-04, SAFE-05, COSCI-04. **Planned output:** `stress_suite.yaml + transcript_tools + ledger tests`.

**Done when:** Same tool outputs/quality across checkpoints; five conditions: clean, confounding, mechanism shift, evidence corruption and longer horizon. Fixed inference budget and decomposed losses survive toy replays.

#### SAFE-07 — Measure zero-shot stress signatures on chosen bases

**Window:** 2026-11-04–2026-11-05 (W6D3–W6D4) · **Effort:** 4 h · **Owner:** William

**Depends on:** SAFE-06, GATE-05. **Planned output:** `zero_shot_signatures.parquet`.

**Done when:** Run all three exact Tinker base configurations on sealed evaluation packets at the fixed inference budget; pair case/rollout seeds, include errors and costs; no post-training outcomes used.

#### SAFE-08 — Freeze task worlds, curricula and forecast protocol

**Window:** 2026-11-06–2026-11-06 (W6D5–W6D5) · **Effort:** 2 h · **Owner:** William · **decision gate**

**Depends on:** SAFE-07, SAFE-06. **Planned output:** `safety_preregistration/ + hashed manifests`.

**Done when:** Benign reasoning-only and evidence/tool/planning curricula are frozen; case sets, loss weights, low-budget inputs, high-budget targets and grouped forecast folds registered before London.

#### SAFE-09 — London: expert-audit safety task realism

**Window:** 2026-11-12–2026-11-12 (W7D4–W7D4) · **Effort:** 2 h · **Owner:** William + SynX expert

**Depends on:** SAFE-08, LAB-06. **Planned output:** `safety_realism_review.md`.

**Done when:** Discuss biological plausibility and label uncertainty; any changed test item is versioned and removed from preexisting locked comparison, or all matched baselines rerun.

### TRAIN — Training harness

#### TRAIN-01 — Early Tinker access and minimal checkpoint smoke

**Window:** 2026-10-19–2026-10-20 (W4D1–W4D2) · **Effort:** 2 h · **Owner:** William

**Depends on:** GATE-05. **Planned output:** `tinker_access_smoke.json`.

**Done when:** One tiny benign toy training/save/resume/sample path works within agreed spend; distinguish optimizer-state and sampler checkpoints; no research sweep launched.

**Constraint:** This is risk retirement before London; the main harness build remains W8-W10.

#### TRAIN-02 — W8D1: implement train/eval adapters and event schema

**Window:** 2026-11-16–2026-11-16 (W8D1–W8D1) · **Effort:** 6 h · **Owner:** William

**Depends on:** TRAIN-01, SAFE-08. **Planned output:** `tinker_backend.py + episode_adapter.py`.

**Done when:** Shared evaluator accepts exact base or adapter checkpoint; records training tokens, inference tokens, tool calls, run IDs, failures and cost without changing tool quality.

#### TRAIN-03 — W8D2: checkpoint resume, sampling and idempotent jobs

**Window:** 2026-11-17–2026-11-17 (W8D2–W8D2) · **Effort:** 6 h · **Owner:** William

**Depends on:** TRAIN-02. **Planned output:** `checkpoint_registry + resume_smoke`.

**Done when:** Interrupt/restart a toy run without token double counting; sampler weights reproduce a stored evaluation; optimizer resume and evaluation artifacts have separate paths.

#### TRAIN-04 — W8D3: validate scorer and baseline parity

**Window:** 2026-11-18–2026-11-18 (W8D3–W8D3) · **Effort:** 4 h · **Owner:** William

**Depends on:** TRAIN-03, SAFE-07. **Planned output:** `harness_validation.json + base_parity.csv`.

**Done when:** Reproduce frozen zero-shot cases within registered sampling variation; no hidden labels enter prompts; budget exhaustion and corrupted evidence yield valid scored failures.

#### TRAIN-05 — W8D3-4: pilot low-budget curves and measure throughput

**Window:** 2026-11-18–2026-11-19 (W8D3–W8D4) · **Effort:** 5 h · **Owner:** William

**Depends on:** TRAIN-03, SAFE-08. **Planned output:** `pilot_uplift.csv + measured_cost_estimate.json`.

**Done when:** One curriculum/model reaches b and 4b cumulative training tokens; observed throughput estimates the full matrix, failure/retry overhead and sampling cost.

#### TRAIN-06 — W8D4-5: launch low-budget cohort

**Window:** 2026-11-19–2026-11-20 (W8D4–W8D5) · **Effort:** 4 h · **Owner:** William

**Depends on:** GATE-09, TRAIN-04. **Planned output:** `low_budget_checkpoints/ + low_budget_eval.csv`.

**Done when:** Twelve proposed trajectories (3 models × 2 curricula × 2 training seeds) reach b and 4b; nested curricula and fixed inference settings; forecast inputs locked before high-budget scoring.

**Constraint:** Wall-clock completion is conditional on measured service capacity; primary matrix is authorized at GATE-09.

#### TRAIN-11 — W8D5: produce development-only high-budget targets

**Window:** 2026-11-20–2026-11-20 (W8D5–W8D5) · **Effort:** 2 h · **Owner:** William

**Depends on:** TRAIN-06. **Planned output:** `development_high_budget_targets.parquet`.

**Done when:** Extend the eight trajectories from two preregistered development models to 16b/64b; expose only development-case outcomes to forecaster fitting. Third-model high-budget outcomes remain sealed.

**Constraint:** These are eight of the twelve curves, not extra training curves. GATE-09 must verify they can finish by W8D5; otherwise the prospective forecast date moves before the held-out result reveal.

#### TRAIN-07 — W9D1-3: complete 16b/64b training and evaluation

**Window:** 2026-11-23–2026-11-25 (W9D1–W9D3) · **Effort:** 6 h · **Owner:** William

**Depends on:** TRAIN-06. **Planned output:** `high_budget_checkpoints/ + uplift_curves.parquet`.

**Done when:** Finish the four held-out-model trajectories and sealed evaluations of all twelve curves; cumulative b/4b/16b/64b checkpoints and matched inference settings. Keep high-budget held-out targets unread until FCAST-02 predictions are saved.

#### TRAIN-08 — W9D3: inference-budget control and tool-quality audit

**Window:** 2026-11-25–2026-11-25 (W9D3–W9D3) · **Effort:** 3 h · **Owner:** William

**Depends on:** TRAIN-07. **Planned output:** `inference_ablation.csv + tool_hash_audit.json`.

**Done when:** A preregistered subset repeats at a second inference budget; ensure reported training uplift is not an inference-budget or tool-output change.

#### TRAIN-09 — W10D1: reproduce principal curves from clean checkout

**Window:** 2026-11-30–2026-11-30 (W10D1–W10D1) · **Effort:** 4 h · **Owner:** William

**Depends on:** GATE-10. **Planned output:** `reproduction_report.json + principal_figures/`.

**Done when:** A clean checkout loads frozen artifacts and reproduces all principal tables/figures; limited sampler spot checks only, not a new training campaign.

#### TRAIN-10 — W10D2: finish harness interface and example report

**Window:** 2026-12-01–2026-12-01 (W10D2–W10D2) · **Effort:** 4 h · **Owner:** William

**Depends on:** TRAIN-09. **Planned output:** `harness_cli + example_uplift_report.html`.

**Done when:** One documented command runs a benign example and returns loss components, observed uplift, forecast interval, costs and out-of-distribution flags.

### FCAST — Forecasting and analysis

#### FCAST-01 — Specify forecasting baselines and grouped holdouts

**Window:** 2026-11-02–2026-11-03 (W6D1–W6D2) · **Effort:** 2 h · **Owner:** William

**Depends on:** SAFE-03, PAPER-02. **Planned output:** `forecast_design.yaml`.

**Done when:** Compare constant/mean, small-budget extrapolation, zero-shot-only, low-budget-only and combined forecasters; split complete model/curriculum trajectories, not budget points or rollout seeds.

#### FCAST-02 — W9D1: fit forecasters and save held-out predictions

**Window:** 2026-11-23–2026-11-23 (W9D1–W9D1) · **Effort:** 4 h · **Owner:** William

**Depends on:** TRAIN-11, FCAST-01, SAFE-08. **Planned output:** `frozen_forecasters/ + forecast_predictions.parquet`.

**Done when:** Fit using development-only high-budget targets from two models plus their zero-shot/b/4b inputs; use at most two prespecified features. Save predictions for the third model and unseen cases before reading held-out high-budget outcomes.

#### FCAST-03 — W9D3-4: score high-budget forecasts and uncertainty

**Window:** 2026-11-25–2026-11-26 (W9D3–W9D4) · **Effort:** 5 h · **Owner:** William

**Depends on:** TRAIN-07, FCAST-02. **Planned output:** `forecast_errors.csv + calibration_figures/`.

**Done when:** Report MAE/RMSE of uplift, interval coverage and rank errors versus baselines; uncertainty grouped by independent worlds/studies; three models do not justify broad model-family generalization.

#### FCAST-04 — W9D4: mechanism-shift, study and signature ablations

**Window:** 2026-11-26–2026-11-26 (W9D4–W9D4) · **Effort:** 4 h · **Owner:** William

**Depends on:** FCAST-03, TRAIN-08. **Planned output:** `shift_failure_report.md + signature_ablation.csv`.

**Done when:** Held-out mechanism/study results and leave-model-out diagnostics retained, including failure to forecast; remove each stress signature component; bounded claim about benign biological reasoning.

#### FCAST-05 — W9D5: resource-capability frontier and decision linkage

**Window:** 2026-11-27–2026-11-27 (W9D5–W9D5) · **Effort:** 3 h · **Owner:** William

**Depends on:** FCAST-04, LAB-09, COSCI-12. **Planned output:** `resource_capability_frontier.csv + linkage_case.md`.

**Done when:** Separate training spend from inference spend; show capability/quality versus each; illustrate how uncertain forecasting informs evaluation allocation without claiming measured CBRN risk.

#### FCAST-06 — W9D5: independent result and leakage review

**Window:** 2026-11-27–2026-11-27 (W9D5–W9D5) · **Effort:** 2 h · **Owner:** William + independent reviewer

**Depends on:** FCAST-04. **Planned output:** `independent_audit.md`.

**Done when:** Reviewer checks token accounting, grouped splits, baseline identity, unsupported claims and figure provenance; correctness bugs use buffer, exploratory reruns do not delay the freeze.

### PAPER — Paper, artifact and presentation

#### PAPER-01 — Create paper skeleton and required-output checklist

**Window:** 2026-10-14–2026-10-14 (W3D3–W3D3) · **Effort:** 1 h · **Owner:** William

**Depends on:** GATE-01. **Planned output:** `paper/outline.md + deliverables_checklist.csv`.

**Done when:** One paper narrative: forecast post-training scientific reasoning uplift, with the emergency scientist as applied case. Each mandatory SynX/grant output has an evidence file and responsible owner.

#### PAPER-02 — Map closest prior work and sharpen novelty

**Window:** 2026-10-14–2026-10-16 (W3D3–W3D5) · **Effort:** 3 h · **Owner:** William

**Depends on:** PAPER-01. **Planned output:** `related_work_matrix.md`.

**Done when:** Compare capability forecasting, scaling/extrapolation, agentic science, controlled counterfactual benchmarks and molecular agents; identify falsifiable distinction before new policy engineering.

#### PAPER-03 — Draft introduction, task definitions and evaluation plan

**Window:** 2026-10-22–2026-10-26 (W4D4–W5D1) · **Effort:** 3 h · **Owner:** William

**Depends on:** PAPER-02, SAFE-03, COSCI-03. **Planned output:** `paper/introduction.md + evaluation.md`.

**Done when:** Problem, hypotheses, inference/training budget distinction and endpoint limits written; figures have planned data sources and no fabricated results.

#### PAPER-04 — Draft methods and prototype figure

**Window:** 2026-11-02–2026-11-03 (W6D1–W6D2) · **Effort:** 2 h · **Owner:** William

**Depends on:** PAPER-03, COSCI-06, SAFE-04. **Planned output:** `paper/methods.md + system_diagram.svg`.

**Done when:** Agent state, benchmark worlds, loss components, splits and baselines are described well enough to reproduce; architecture figure connects both testbeds without conflating their evidence.

#### PAPER-05 — Build artifact skeleton and results table templates

**Window:** 2026-10-29–2026-11-04 (W5D4–W6D3) · **Effort:** 3 h · **Owner:** William

**Depends on:** PAPER-02, GATE-03. **Planned output:** `artifact/README.md + paper/results_templates.md`.

**Done when:** Environment lock, data licensing, source manifest and clean-run entry point exist; all anticipated figure/table columns and independent unit definitions are specified.

#### PAPER-06 — Write SynX results and pre-London presentation outline

**Window:** 2026-11-05–2026-11-13 (W6D4–W7D5) · **Effort:** 4 h · **Owner:** William

**Depends on:** COSCI-11, MOL-05, PAPER-05. **Planned output:** `paper/synx_results.md + talk_outline.md`.

**Done when:** Historical/two-family results and platform improvement written; no placeholder science claims; one-slide London feedback questions and experimental caveats included.

**Constraint:** Talk outline and results draft are due W6D5; light narrative revisions continue during London, with final case text by W7D5.

#### PAPER-07 — London: collect expert feedback and case-study narrative

**Window:** 2026-11-11–2026-11-13 (W7D3–W7D5) · **Effort:** 2 h · **Owner:** William

**Depends on:** LAB-06, PAPER-05, COSCI-12. **Planned output:** `paper/london_case_notes.md`.

**Done when:** Expert review facts and quotations are attributable and cleared for publication where needed; technical disagreements and constraints retained.

#### PAPER-08 — W9: write safety results and final figure captions

**Window:** 2026-11-24–2026-11-27 (W9D2–W9D5) · **Effort:** 5 h · **Owner:** William

**Depends on:** FCAST-02, PAPER-06. **Planned output:** `paper/safety_results.md + final_figures/`.

**Done when:** Low-budget placeholders replaced only after corresponding high-budget scoring; captions state independent units, uncertainty, costs, failed forecasts and boundary of safety inference.

**Constraint:** Starts with table scaffolding at W9D2; final values depend on FCAST-03/FCAST-04 by W9D4.

#### PAPER-09 — W10D1-2: close full paper and permitted artifact

**Window:** 2026-11-30–2026-12-01 (W10D1–W10D2) · **Effort:** 6 h · **Owner:** William

**Depends on:** GATE-10, PAPER-08, PAPER-07. **Planned output:** `submission_ready_paper + release_candidate/`.

**Done when:** All sections, references, limitations, required-output checklist, commands and permitted data are complete; choose actual venue/template/deadline after checking current calls.

#### PAPER-10 — W10D3: expert review and first full rehearsal

**Window:** 2026-12-02–2026-12-02 (W10D3–W10D3) · **Effort:** 5 h · **Owner:** William + Askar/reviewer

**Depends on:** PAPER-09, TRAIN-10. **Planned output:** `review_response.md + timed_rehearsal`.

**Done when:** Askar/one external reviewer tests the claim chain and demo; full talk fits the actual allotted presentation time; any lab obligation still missing is explicitly identified.

#### PAPER-11 — W10D4: final deck, demo recording and offline backup

**Window:** 2026-12-03–2026-12-03 (W10D4–W10D4) · **Effort:** 5 h · **Owner:** William

**Depends on:** PAPER-10. **Planned output:** `final_deck + demo_recording + offline_bundle`.

**Done when:** Second rehearsal succeeds without live API/network; numbers match frozen paper; artifact opens from clean environment; presentation and demo backups available.

#### PAPER-12 — W10D5: final presentation and deliverable handoff

**Window:** 2026-12-04–2026-12-04 (W10D5–W10D5) · **Effort:** 4 h · **Owner:** William · **decision gate**

**Depends on:** PAPER-11, TRAIN-10, TRAIN-09. **Planned output:** `presentation + paper + reproducible artifact + SynX handoff`.

**Done when:** Deliver talk, submission-ready manuscript, harness, benchmark manifests and seven mandatory SynX outputs; log publication status as submitted/ready only if actually true.

