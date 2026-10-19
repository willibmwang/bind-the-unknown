"""Authoritative planning data; generates the Markdown task register and embeds data.

Windows are project working days, not verified calendar dates. Effort is William's
hands-on hours, not elapsed time, API time, or SynX staff effort.
"""
import json
import re
from pathlib import Path
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parent
# Stream prefixes are the readable names; task ids are PREFIX-NN so an id states its own stream.
streams = [
    ("GATE", "Decisions, access and gates", 1),
    ("COSCI", "Emergency co-scientist and benchmark", 2),
    ("MOL", "Macrocycle design and platform", 2),
    ("LAB", "SynX experiment and London", 3),
    ("SAFE", "Safety benchmark preparation", 4),
    ("TRAIN", "Training harness", 4),
    ("FCAST", "Forecasting and analysis", 5),
    ("PAPER", "Paper, artifact and presentation", 6),
]
tasks = []

def add(id, name, start, end, hours, deps, artifact, acceptance,
        owner="William", gate=False, note="", status="Planned", evidence=""):
    """`status` is only ever 'Complete' where `evidence` names a committed artifact."""
    tasks.append(dict(id=id, stream=id.split("-")[0], name=name, start=start, end=end,
                      hours=hours, deps=deps.split() if deps else [],
                      artifact=artifact, acceptance=acceptance, owner=owner,
                      gate=gate, note=note, status=status, evidence=evidence))

# Workdays 1-5 = W1; 31-35 = London W7; 46-50 = final W10.
add("GATE-01", "Lock the two contributions and all required outputs", 1, 1, 2, "",
    "scope_and_claims.md", "Askar reviews one primary forecasting claim, one applied emergency-scientist validation, and the seven mandatory SynX outputs; no promise of publication acceptance.")
add("GATE-02", "Request access, lab slot and publication permissions", 1, 3, 2, "GATE-01",
    "access_and_owners.csv", "Named SynX experimental owner; Portal/API, paired data, residue assets, compute and Tinker access requests recorded; availability is not assumed.", owner="William + Askar + SynX")
add("GATE-03", "Audit pilot evidence and establish reproducible run ledger", 1, 4, 3, "",
    "pilot_status.json + run_schema.json", "Check actual scheduler/output evidence; separate existing CPU pilot results from queued/unverified GPU runs; commit hashes, seeds, data versions and costs recorded.")
add("GATE-04", "Confirm wet-lab route, lead times and data return", 4, 5, 2, "GATE-02",
    "SynX experiment calendar", "SynX confirms a feasible library/reagent preparation route, one selection round by W7, and usable readout by W8D3. If unavailable, mark mandatory lab deliverable at risk immediately.", owner="William + SynX experimental lead", gate=True)
add("GATE-05", "Select live Tinker models and cap total spend", 6, 7, 3, "GATE-02",
    "models_and_budget.json", "Query actual trainable capabilities; freeze three supported open-weight model IDs/configurations, LoRA settings and renderer modes, or explicitly narrow generalization claims. Record an approved spend ceiling and throughput estimate.", gate=True,
    note="Local Qwen3-4B-Instruct-2507 is a pilot comparator, not an assumed Tinker base; current docs list it as retired.")
add("GATE-06", "Week-2 feasibility and novelty review", 10, 10, 2, "GATE-04 GATE-05 COSCI-03 SAFE-02 PAPER-02",
    "gate_w2.md", "Confirm lab slot, data rights, scorer validity, trainable models, and a distinguishable forecasting question. Record unresolved blockers with owner and deadline.", owner="William + Askar", gate=True)
add("GATE-07", "Choose the platform improvement from measured failures", 15, 15, 1, "MOL-02 COSCI-05",
    "platform_change_spec.md", "Select one bounded improvement with a failing baseline, implementation estimate and independent validation metric; retain residue/rotamer limitations as measured hypotheses.", owner="William + SynX", gate=True)
add("GATE-08", "Freeze SynX scientific scope and London readiness", 29, 30, 2, "COSCI-12 MOL-07 LAB-04 SAFE-08",
    "gate_w6.md", "Both family dossiers, historical replay, improvement evaluation, prospective prediction file and safety benchmark manifest are complete; London agenda accepted; no new feature scope.", gate=True)
add("GATE-09", "Authorize the measured training/evaluation matrix", 39, 39, 1, "TRAIN-03 TRAIN-04 TRAIN-05",
    "training_gate.json", "Measured API throughput and token counts fit the spend ceiling and W9D3 finish; inference protocol and forecast train/test groups frozen before seeing high-budget results.", gate=True)
add("GATE-10", "Freeze results and final claims", 45, 45, 2, "FCAST-05 FCAST-06 LAB-09 PAPER-08",
    "results_freeze.json + claims_audit.md", "Each figure traces to an immutable run manifest; negative results retained; missing required outputs explicitly flagged; paper and presentation use the same claims.", owner="William + Askar", gate=True)

add("COSCI-01", "Specify emergency episodes and decision rubric", 1, 3, 6, "GATE-01",
    "emergency_episode_schema.json + rubric.md", "Specify timestamps, hidden outcomes, evidence state, tool permissions, action budget, readiness rubric and failure categories; reviewers can score a toy episode consistently.")
add("COSCI-02", "Freeze public data provenance and split policy", 2, 5, 4, "",
    "dataset_manifest.json + split_policy.md", "ProteinGym GB1 and ATLAS remain component pilots; historical/public evidence collections are source-versioned; split by parent/backbone, target/family or study as appropriate.")
add("COSCI-03", "Build small development episode set and baselines", 5, 10, 10, "COSCI-01 COSCI-02",
    "dev_episodes/ + baseline_scores.csv", "Twelve proposed development episodes span incomplete evidence, contradictions, deadlines and missing tools; fixed rubric, scripted workflow and single-response retrieval baselines run with matched access.",
    note="Twelve is a planning target, not an existing dataset count; freeze final sample size after the audit.")
add("COSCI-04", "Implement evidence state and resource-accounted tools", 6, 12, 8, "COSCI-01 GATE-03",
    "evidence_state + tool_event ledger", "Every retrieval/structure/design action updates cited evidence, uncertainty and actual token/tool/wall-time cost; invalid actions are logged and charged.")
add("COSCI-05", "Run emergency stress and public-pilot comparisons", 12, 15, 6, "COSCI-03 COSCI-04",
    "stress_baselines.csv + pilot_comparison.md", "Compare complete versus degraded evidence and immediate versus delayed feedback; report quality/cost fronts and failure cases; GB1/ATLAS conclusions stay within their measured endpoints.")
add("COSCI-06", "Build adaptive action policy v1", 13, 18, 10, "COSCI-04",
    "agent_policy_v1 + replay_trace", "Policy can request evidence, rerank, allocate a calculation and stop with uncertainty; all four actions complete an end-to-end development replay.")
add("COSCI-07", "Curate two prototype-family evidence packets", 8, 17, 6, "COSCI-02",
    "family_1/ + family_2/ evidence dossiers", "Two distinct families chosen with Askar/SynX; each has sourced essentiality, accessibility, conservation, tractability, uncertainty and a held-out episode. Selection criteria documented.", owner="William + Askar/SynX review")
add("COSCI-08", "Assemble time-stamped COVID historical replay", 13, 20, 6, "COSCI-02",
    "covid_2020_replay/ + hindsight_audit.md", "January-March 2020 retrieval cutoffs verified; checkpoint decisions scored against information available then; pretrained hindsight limitations explicitly retained.")
add("COSCI-09", "Calibrate expert scoring and freeze held-out rubric", 20, 20, 3, "COSCI-05 COSCI-07 COSCI-08",
    "expert_calibration.csv + frozen_rubric.json", "At least two reviewers score a calibration subset; disagreements and score definitions resolved on development outputs; held-out labels remain sealed.", owner="William + two expert reviewers")
add("COSCI-10", "Integrate policy with frozen molecular tools", 19, 23, 8, "COSCI-06 MOL-03",
    "end_to_end_replay.jsonl", "Both family episodes pass intake → evidence → target ranking → design recommendation; policy comparisons share tool/model versions.")
add("COSCI-11", "Run held-out decisions, family transfer and ablations", 25, 28, 9, "COSCI-09 COSCI-10 MOL-05",
    "system_eval.csv + decision_quality_figures/", "All four baselines evaluated on historical and two-family held-out episodes; ablate adaptation and evidence state; report quality, evidence support, latency, cost and reviewer uncertainty.")
add("COSCI-12", "Freeze the emergency scientist demo and findings", 29, 29, 3, "COSCI-11",
    "synx_demo_v1 + limitations.md", "A recorded replay and offline fallback work; no antiviral efficacy claim without an assay; central plots and errors are ready before London.", gate=True)

# COSCI-13 to COSCI-15 close the domain-specificity gap: before them the agent was a
# budgeted bandit over opaque integer candidates, with no pathogen and no chemistry in it.
add("COSCI-13", "Specify the target dossier decision state and action space", 16, 16, 4, "COSCI-01",
    "hpc_pilots/dossier.py", "Decision state is the seven-axis dossier with per-axis citations, timestamps and open uncertainties; action space is retrieve evidence, request structure, rerank, invoke design tool, recommend experiment, stop. Evidence dated after the decision clock is refused and the refusal is recorded. Hard-gate axes block a ranking.",
    status="Complete", evidence="hpc_pilots/dossier.py with four isolation tests; date-boundary and conflict-retention tests pass.",
    note="Named targets reintroduce pretrained-hindsight risk that the date boundary does not control; it needs a separate audit in COSCI-08.")
add("COSCI-14", "Build pathogen decision environments and the subtype transfer split", 16, 16, 4, "COSCI-02",
    "hpc_pilots/configs/targets.json + data/processed/targets/", "Mpro, HA H1 and HA H3 staged from the ProteinGym archive as disjoint-pool decision environments with per-landscape assay endpoints recorded; pool geometry shrinks rather than reusing candidates across pools described as disjoint; H1-to-H3 is a grouped study-and-subtype holdout.",
    status="Complete", evidence="10 pools for Mpro, 19 for HA H1, 20 for HA H3; 240 paired replays recorded under runs/targets/.")
add("COSCI-15", "Score evidence-update stability in emergency episodes", 16, 16, 3, "COSCI-13 COSCI-14",
    "hpc_pilots/emergency.py + runs/emergency/", "Episodes release relevant, contradictory and irrelevant evidence at timed checkpoints under a deadline and an action budget; the policy is scored on updating when relevant evidence arrives, holding the recommendation when irrelevant evidence arrives, and retaining contradictory evidence beside the claim it contests.",
    status="Complete", evidence="Six episodes across three families; deterministic baselines hold the recommendation but reorder the tail, and the scripted workflow misses the deadline.",
    note="Deterministic baselines are stable by construction; the metric earns its value on the LLM policy in COSCI-06.")

add("MOL-01", "Audit residue registry, scores and experimental endpoints", 3, 9, 6, "GATE-02",
    "molecular_audit.md + chemistry_registry.csv", "Identity/stereochemistry/codon mapping, scorefunction, rotamer assets, parent groups, assay endpoints, replicates, missingness and censoring audited; usable campaign tables identified.")
add("MOL-02", "Measure molecular ranking and representation failures", 10, 14, 7, "MOL-01",
    "molecular_baselines.csv + stratified_hexbin/", "Rosetta ranking and simple chemistry/similarity/supervised baselines use grouped splits; inspect motif/backbone/campaign confounding and unsupported residue representations.")
add("MOL-03", "Freeze design/refolding/filter contracts", 11, 18, 4, "MOL-01",
    "design_tool_contract.json", "RFdiffusion/BoltzGen version and actual chemistry support documented; refolding and geometric filters are separate logged stages; generation outputs are compatible with the permitted SynX route.")
add("MOL-04", "Implement one justified SynX platform improvement", 18, 22, 10, "GATE-07 MOL-03",
    "platform_patch + regression_examples/", "Fix the selected observed failure, such as residue validation or uncertainty-aware campaign ranking; run affected failing examples and a clean unaffected control set.")
add("MOL-05", "Evaluate improvement and low-label ranking module", 23, 25, 5, "MOL-04 MOL-02",
    "platform_ablation.csv + label_budget_eval.csv", "Before/after evaluated with independent endpoint and matched costs; zero/few-shot label budgets and grouped transfer are reported if paired data support them; newly uploaded proposal incorporated before freezing.",
    note="No separate few-shot proposal is currently reviewed. This schedule reserves integration time; an unsupported module cannot substitute for the required platform improvement.")
add("MOL-06", "Finalize both family target dossiers and library proposal", 19, 23, 6, "COSCI-07 MOL-03",
    "target_dossiers/ + library_design_v1.csv", "Both families include evidence-backed target/epitope ranks; selected non-canonical macrocyclic library is compatible with SynX display constraints and independently reviewed.", owner="William + SynX design review")
add("MOL-07", "Freeze prospective candidate ranks and analysis plan", 26, 28, 4, "MOL-05 MOL-06 LAB-03",
    "prospective_predictions.csv + analysis_plan.md", "Immutable candidates/ranks, controls, intended measured endpoints and analysis rules recorded before laboratory outcomes; retain all submitted candidates and failures.", gate=True)

# MOL-08 puts the two features that define the SynX platform into the system: non-canonical
# building blocks in the representation, and an enrichment ratio as the readout.
add("MOL-08", "Represent non-canonical blocks and read out enrichment", 16, 16, 3, "",
    "hpc_pilots/chemistry.py + enrichment metrics", "Building-block registry covers canonical and non-canonical classes with initiator and closure roles and every platform-supplied field explicitly pending; the encoding extends past the canonical twenty; acquisition metrics report an enrichment ratio against the pool baseline alongside the per-candidate view.",
    status="Complete", evidence="29 blocks, 9 non-canonical; enrichment ratio added to every binding and target checkpoint; the recorded GB1 replays still reproduce bit-identically.",
    note="Chemical identifiers, codons, Rosetta parameters and rotamer libraries stay null until the MOL-01 export arrives. No measured macrocycle outcome exists yet, so no enrichment figure here is a macrocycle result.")

add("LAB-01", "Agree the single-round experimental question", 9, 10, 3, "GATE-04 MOL-01",
    "round_question_and_endpoints.md", "SynX confirms the designed library, feasible target/assay endpoint, controls, replicates, primary readout and what one round can substantiate; no invented affinity/antiviral endpoint.", owner="William + SynX experimental lead")
add("LAB-02", "SynX prepares reagents, library and booked slot", 11, 25, 2, "LAB-01",
    "lab_readiness_tracker.csv", "SynX staff report validated inputs, shipping/reagent/library lead times and a reserved W7 selection slot; William's effort is coordination only.", owner="SynX experimental lead; William tracks",
    note="Elapsed preparation is a scheduling placeholder, not a laboratory protocol or confirmed turnaround.")
add("LAB-03", "Review design-to-display compatibility", 25, 25, 2, "LAB-02 MOL-06",
    "compatibility_signoff.md", "Experimental lead signs off library representation, chemical alphabet, input format and endpoint; any incompatible designs revised before prediction freeze.", owner="William + SynX experimental lead", gate=True)
add("LAB-04", "Prepare London dossier and book review agenda", 28, 29, 2, "MOL-07 LAB-03",
    "london_review_pack/ + visit_agenda.md", "Send-ready pack includes two-family dossiers, model limitations, platform before/after, frozen prospective predictions, assay questions and W8 data-handoff checklist.")
add("LAB-05", "London travel and arrival", 31, 31, 0, "GATE-08",
    "arrival + access confirmed", "Travel day protected from coding or critical job launches; actual itinerary to replace project-day placeholder.", owner="William", note="W7D1 travel placement is proposed; the visit week is fixed, specific travel dates are not.")
add("LAB-06", "London: facility tour and design/experiment review", 32, 32, 6, "LAB-05 LAB-04",
    "facility_constraints.md + review_actions.md", "Walk through portal-to-wet-lab handoff with SynX; reconcile predicted versus actual constraints; obtain expert review of both dossiers and the frozen experiment plan.", owner="William + SynX")
add("LAB-07", "London: observe the required display round", 32, 34, 5, "LAB-06 MOL-07 LAB-03",
    "selection_round_record + sample manifest", "SynX executes one actual mRNA display selection round against the designed library; record performed date, inputs, controls, failures and sample provenance. Visit alone does not meet this requirement.", owner="SynX experimental lead; William observes",
    note="Exact execution day and duration must be confirmed at GATE-04; the Gantt window is not evidence of booking.")
add("LAB-08", "London: review discrepancies and secure data handoff", 33, 34, 5, "LAB-06",
    "expert_discrepancy_log.md + data_handoff_contract.json", "Review unexplained score/measurement differences and platform patch; agree raw and processed readout format, QC metadata, owner and delivery by W8D3.", owner="William + SynX")
add("LAB-10", "Return travel and archive the London review", 35, 35, 1, "LAB-07 LAB-08",
    "london_summary.md + archived review pack", "Capture accepted changes and experimental limitations; return travel protected; no major new architecture work.")
add("LAB-11", "SynX processes and returns selection-round readout", 34, 38, 1, "LAB-07",
    "raw_readout/ + QC + manifest", "Receive analysable readout and sample/control metadata by W8D3; QC failures stay in the report. SynX confirms actual processing/sequence turnaround before the slot.", owner="SynX experimental lead; William receives",
    note="If readout is late, experimental obligation stays incomplete and takes protected buffer; it is not relabeled optional.")
add("LAB-09", "Analyze the round and update the decision state", 38, 40, 4, "LAB-11 MOL-07",
    "prospective_results.csv + before_after_decisions.md", "Compare preregistered ranks with the measured endpoint, quantify uncertainty at independent units, include controls/failures and show how evidence changes the recommendation.")

add("SAFE-01", "Translate grant into benign task/loss specification", 2, 5, 4, "GATE-01",
    "safety_task_spec.md + loss_components.json", "Define conclusion accuracy, evidence grounding and process validity separately; scalar normalization/weights and threat-claim boundary documented; supervised biology reasoning is benign.")
add("SAFE-02", "Select public transcriptomic collections and worlds", 6, 10, 4, "SAFE-01",
    "transcriptomic_manifest.json", "Audit scPerturb NormanWeissman2019 (GSE133344), DixitRegev2016 (GSE90063), and ReplogleWeissman2022; confirm versions, perturbation/metadata availability, licenses and annotation quality before using them. Pin a CELLxGENE LTS build for background/reference packets only.",
    note="Public atlas candidates are starting points; observational annotations are not automatic causal ground truth.")
add("SAFE-03", "Specify train/test boundaries and counterfactual mechanisms", 11, 14, 4, "SAFE-02",
    "world_families.json + split_manifest.json", "Split by study/donor/mechanism family, never random cells; paired synthetic worlds share observable correlations while latent mechanisms differ; inference and training budgets are separate fields.")
add("SAFE-04", "Build and validate small synthetic world generator", 15, 20, 6, "SAFE-03",
    "counterfactual_worlds/ + oracle_validation.json", "Oracle recovers known latent answers; evidence disclosure and confounding/corruption/shift/horizon knobs verified on development worlds; held-out mechanisms remain sealed.")
add("SAFE-05", "Prepare real-study task packets and expert references", 16, 22, 5, "SAFE-03",
    "real_study_packets/ + reference_provenance.csv", "At least three audited studies proposed for generalization; references distinguish measured perturbation evidence from expert interpretation; metadata leakage and donor overlap checked.", owner="William + biological reviewer")
add("SAFE-06", "Freeze tool primitives, signatures and budget accounting", 22, 24, 5, "SAFE-04 SAFE-05 COSCI-04",
    "stress_suite.yaml + transcript_tools + ledger tests", "Same tool outputs/quality across checkpoints; five conditions: clean, confounding, mechanism shift, evidence corruption and longer horizon. Fixed inference budget and decomposed losses survive toy replays.")
add("SAFE-07", "Measure zero-shot stress signatures on chosen bases", 25, 28, 5, "SAFE-06 GATE-05",
    "zero_shot_signatures.parquet", "Run all three exact Tinker base configurations on sealed evaluation packets at the fixed inference budget; pair case/rollout seeds, include errors and costs; no post-training outcomes used.")
add("SAFE-08", "Freeze task worlds, curricula and forecast protocol", 29, 29, 3, "SAFE-07 SAFE-06",
    "safety_preregistration/ + hashed manifests", "Benign reasoning-only and evidence/tool/planning curricula are frozen; case sets, loss weights, low-budget inputs, high-budget targets and grouped forecast folds registered before London.", gate=True)
add("SAFE-09", "London: expert-audit safety task realism", 34, 34, 2, "SAFE-08 LAB-06",
    "safety_realism_review.md", "Discuss biological plausibility and label uncertainty; any changed test item is versioned and removed from preexisting locked comparison, or all matched baselines rerun.", owner="William + SynX expert")

add("TRAIN-01", "Early Tinker access and minimal checkpoint smoke", 11, 12, 2, "GATE-05",
    "tinker_access_smoke.json", "One tiny benign toy training/save/resume/sample path works within agreed spend; distinguish optimizer-state and sampler checkpoints; no research sweep launched.",
    note="This is risk retirement before London; the main harness build remains W8-W10.")
add("TRAIN-02", "W8D1: implement train/eval adapters and event schema", 36, 36, 6, "TRAIN-01 SAFE-08",
    "tinker_backend.py + episode_adapter.py", "Shared evaluator accepts exact base or adapter checkpoint; records training tokens, inference tokens, tool calls, run IDs, failures and cost without changing tool quality.")
add("TRAIN-03", "W8D2: checkpoint resume, sampling and idempotent jobs", 37, 37, 6, "TRAIN-02",
    "checkpoint_registry + resume_smoke", "Interrupt/restart a toy run without token double counting; sampler weights reproduce a stored evaluation; optimizer resume and evaluation artifacts have separate paths.")
add("TRAIN-04", "W8D3: validate scorer and baseline parity", 38, 38, 4, "TRAIN-03 SAFE-07",
    "harness_validation.json + base_parity.csv", "Reproduce frozen zero-shot cases within registered sampling variation; no hidden labels enter prompts; budget exhaustion and corrupted evidence yield valid scored failures.")
add("TRAIN-05", "W8D3-4: pilot low-budget curves and measure throughput", 38, 39, 5, "TRAIN-03 SAFE-08",
    "pilot_uplift.csv + measured_cost_estimate.json", "One curriculum/model reaches b and 4b cumulative training tokens; observed throughput estimates the full matrix, failure/retry overhead and sampling cost.")
add("TRAIN-06", "W8D4-5: launch low-budget cohort", 39, 40, 4, "GATE-09 TRAIN-04",
    "low_budget_checkpoints/ + low_budget_eval.csv", "Twelve proposed trajectories (3 models × 2 curricula × 2 training seeds) reach b and 4b; nested curricula and fixed inference settings; forecast inputs locked before high-budget scoring.",
    note="Wall-clock completion is conditional on measured service capacity; primary matrix is authorized at GATE-09.")
add("TRAIN-11", "W8D5: produce development-only high-budget targets", 40, 40, 2, "TRAIN-06",
    "development_high_budget_targets.parquet", "Extend the eight trajectories from two preregistered development models to 16b/64b; expose only development-case outcomes to forecaster fitting. Third-model high-budget outcomes remain sealed.",
    note="These are eight of the twelve curves, not extra training curves. GATE-09 must verify they can finish by W8D5; otherwise the prospective forecast date moves before the held-out result reveal.")
add("TRAIN-07", "W9D1-3: complete 16b/64b training and evaluation", 41, 43, 6, "TRAIN-06",
    "high_budget_checkpoints/ + uplift_curves.parquet", "Finish the four held-out-model trajectories and sealed evaluations of all twelve curves; cumulative b/4b/16b/64b checkpoints and matched inference settings. Keep high-budget held-out targets unread until FCAST-02 predictions are saved.")
add("TRAIN-08", "W9D3: inference-budget control and tool-quality audit", 43, 43, 3, "TRAIN-07",
    "inference_ablation.csv + tool_hash_audit.json", "A preregistered subset repeats at a second inference budget; ensure reported training uplift is not an inference-budget or tool-output change.")
add("TRAIN-09", "W10D1: reproduce principal curves from clean checkout", 46, 46, 4, "GATE-10",
    "reproduction_report.json + principal_figures/", "A clean checkout loads frozen artifacts and reproduces all principal tables/figures; limited sampler spot checks only, not a new training campaign.")
add("TRAIN-10", "W10D2: finish harness interface and example report", 47, 47, 4, "TRAIN-09",
    "harness_cli + example_uplift_report.html", "One documented command runs a benign example and returns loss components, observed uplift, forecast interval, costs and out-of-distribution flags.")

add("FCAST-01", "Specify forecasting baselines and grouped holdouts", 23, 25, 3, "SAFE-03 PAPER-02",
    "forecast_design.yaml", "Compare constant/mean, small-budget extrapolation, zero-shot-only, low-budget-only and combined forecasters; split complete model/curriculum trajectories, not budget points or rollout seeds.")
add("FCAST-02", "W9D1: fit forecasters and save held-out predictions", 41, 41, 4, "TRAIN-11 FCAST-01 SAFE-08",
    "frozen_forecasters/ + forecast_predictions.parquet", "Fit using development-only high-budget targets from two models plus their zero-shot/b/4b inputs; use at most two prespecified features. Save predictions for the third model and unseen cases before reading held-out high-budget outcomes.")
add("FCAST-03", "W9D3-4: score high-budget forecasts and uncertainty", 43, 44, 5, "TRAIN-07 FCAST-02",
    "forecast_errors.csv + calibration_figures/", "Report MAE/RMSE of uplift, interval coverage and rank errors versus baselines; uncertainty grouped by independent worlds/studies; three models do not justify broad model-family generalization.")
add("FCAST-04", "W9D4: mechanism-shift, study and signature ablations", 44, 44, 4, "FCAST-03 TRAIN-08",
    "shift_failure_report.md + signature_ablation.csv", "Held-out mechanism/study results and leave-model-out diagnostics retained, including failure to forecast; remove each stress signature component; bounded claim about benign biological reasoning.")
add("FCAST-05", "W9D5: resource-capability frontier and decision linkage", 45, 45, 3, "FCAST-04 LAB-09 COSCI-12",
    "resource_capability_frontier.csv + linkage_case.md", "Separate training spend from inference spend; show capability/quality versus each; illustrate how uncertain forecasting informs evaluation allocation without claiming measured CBRN risk.")
add("FCAST-06", "W9D5: independent result and leakage review", 45, 45, 2, "FCAST-04",
    "independent_audit.md", "Reviewer checks token accounting, grouped splits, baseline identity, unsupported claims and figure provenance; correctness bugs use buffer, exploratory reruns do not delay the freeze.", owner="William + independent reviewer")

add("PAPER-01", "Create paper skeleton and required-output checklist", 2, 5, 3, "GATE-01",
    "paper/outline.md + deliverables_checklist.csv", "One paper narrative: forecast post-training scientific reasoning uplift, with the emergency scientist as applied case. Each mandatory SynX/grant output has an evidence file and responsible owner.")
add("PAPER-02", "Map closest prior work and sharpen novelty", 6, 10, 4, "PAPER-01",
    "related_work_matrix.md", "Compare capability forecasting, scaling/extrapolation, agentic science, controlled counterfactual benchmarks and molecular agents; identify falsifiable distinction before new policy engineering.")
add("PAPER-03", "Draft introduction, task definitions and evaluation plan", 14, 15, 4, "PAPER-02 SAFE-03 COSCI-03",
    "paper/introduction.md + evaluation.md", "Problem, hypotheses, inference/training budget distinction and endpoint limits written; figures have planned data sources and no fabricated results.")
add("PAPER-04", "Draft methods and prototype figure", 20, 20, 3, "PAPER-03 COSCI-06 SAFE-04",
    "paper/methods.md + system_diagram.svg", "Agent state, benchmark worlds, loss components, splits and baselines are described well enough to reproduce; architecture figure connects both testbeds without conflating their evidence.")
add("PAPER-05", "Build artifact skeleton and results table templates", 21, 25, 3, "PAPER-04 GATE-03",
    "artifact/README.md + paper/results_templates.md", "Environment lock, data licensing, source manifest and clean-run entry point exist; all anticipated figure/table columns and independent unit definitions are specified.")
add("PAPER-06", "Write SynX results and pre-London presentation outline", 28, 30, 4, "COSCI-11 MOL-05 PAPER-05",
    "paper/synx_results.md + talk_outline.md", "Historical/two-family results and platform improvement written; no placeholder science claims; one-slide London feedback questions and experimental caveats included.")
add("PAPER-07", "London: collect expert feedback and case-study narrative", 33, 35, 2, "LAB-06 PAPER-06",
    "paper/london_case_notes.md", "Expert review facts and quotations are attributable and cleared for publication where needed; technical disagreements and constraints retained.")
add("PAPER-08", "W9: write safety results and final figure captions", 42, 45, 5, "FCAST-02 PAPER-06",
    "paper/safety_results.md + final_figures/", "Low-budget placeholders replaced only after corresponding high-budget scoring; captions state independent units, uncertainty, costs, failed forecasts and boundary of safety inference.",
    note="Starts with table scaffolding at W9D2; final values depend on FCAST-03/FCAST-04 by W9D4.")
add("PAPER-09", "W10D1-2: close full paper and permitted artifact", 46, 47, 6, "GATE-10 PAPER-08 PAPER-07",
    "submission_ready_paper + release_candidate/", "All sections, references, limitations, required-output checklist, commands and permitted data are complete; choose actual venue/template/deadline after checking current calls.")
add("PAPER-10", "W10D3: expert review and first full rehearsal", 48, 48, 5, "PAPER-09 TRAIN-10",
    "review_response.md + timed_rehearsal", "Askar/one external reviewer tests the claim chain and demo; full talk fits the actual allotted presentation time; any lab obligation still missing is explicitly identified.", owner="William + Askar/reviewer")
add("PAPER-11", "W10D4: final deck, demo recording and offline backup", 49, 49, 5, "PAPER-10",
    "final_deck + demo_recording + offline_bundle", "Second rehearsal succeeds without live API/network; numbers match frozen paper; artifact opens from clean environment; presentation and demo backups available.")
add("PAPER-12", "W10D5: final presentation and deliverable handoff", 50, 50, 4, "PAPER-11 TRAIN-10 TRAIN-09",
    "presentation + paper + reproducible artifact + SynX handoff", "Deliver talk, submission-ready manuscript, harness, benchmark manifests and seven mandatory SynX outputs; log publication status as submitted/ready only if actually true.", gate=True)

milestones = [
    dict(day=5, label="Lab route agreed", task="GATE-04"),
    dict(day=10, label="Access / feasibility", task="GATE-06"),
    dict(day=20, label="Rubric frozen", task="COSCI-09"),
    dict(day=28, label="Predictions frozen", task="MOL-07"),
    dict(day=30, label="London readiness", task="GATE-08"),
    dict(day=35, label="London complete", task="LAB-10"),
    dict(day=38, label="Lab data due", task="LAB-11"),
    dict(day=39, label="Sweep gate", task="GATE-09"),
    dict(day=43, label="Training complete", task="TRAIN-07"),
    dict(day=45, label="Results frozen", task="GATE-10"),
    dict(day=50, label="Final handoff", task="PAPER-12"),
]

# Rebaseline after the confirmed calendar anchor: today is W3D3, not W1.
# Existing brief/runners reduce verification effort; none of these updates mark
# a historical task as completed. New safety work is scheduled from today.
rebaseline = {
    'GATE-01':(13,13,1), 'GATE-02':(13,14,1), 'GATE-03':(13,14,1), 'GATE-04':(15,15,1),
    'GATE-05':(14,15,2), 'GATE-06':(20,20,2), 'GATE-07':(22,22,1), 'GATE-08':(30,30,2),
    'COSCI-01':(13,14,3), 'COSCI-02':(13,14,2), 'COSCI-03':(15,18,6), 'COSCI-04':(15,19,4),
    'COSCI-05':(20,21,4), 'COSCI-06':(20,24,6), 'COSCI-07':(16,22,4), 'COSCI-08':(15,24,4),
    'COSCI-09':(24,24,2), 'COSCI-10':(25,25,4), 'COSCI-11':(27,28,7),
    'MOL-01':(14,18,4), 'MOL-02':(18,21,5), 'MOL-03':(19,22,4), 'MOL-04':(22,25,6),
    'MOL-05':(26,27,3), 'MOL-06':(22,25,3), 'MOL-07':(28,28,3),
    'LAB-01':(19,20,2), 'LAB-02':(15,30,2), 'LAB-03':(25,25,1),
    'SAFE-01':(13,15,3), 'SAFE-02':(15,18,4), 'SAFE-03':(18,19,3),
    'SAFE-04':(20,26,6), 'SAFE-05':(24,26,4), 'SAFE-06':(26,27,4),
    'SAFE-07':(28,29,4), 'SAFE-08':(30,30,2), 'TRAIN-01':(16,17,2),
    'FCAST-01':(26,27,2), 'PAPER-01':(13,13,1), 'PAPER-02':(13,15,3),
    'PAPER-03':(19,21,3), 'PAPER-04':(26,27,2), 'PAPER-05':(24,28,3), 'PAPER-06':(29,35,4)
}
for t in tasks:
    if t['id'] in rebaseline:
        t['start'], t['end'], t['hours'] = rebaseline[t['id']]
    if t['id']=='LAB-02':
        t['deps']=['GATE-04']
        t['note']='SynX can prepare available inputs after the route gate while the final question/designs are refined; preparation must finish by W6D5. This overlap requires the experimental lead’s confirmation.'
    if t['id']=='LAB-03': t['deps']=['GATE-04','MOL-06']
    if t['id']=='PAPER-05': t['deps']=['PAPER-02','GATE-03']
    if t['id']=='PAPER-06': t['note']='Talk outline and results draft are due W6D5; light narrative revisions continue during London, with final case text by W7D5.'
    if t['id']=='PAPER-07': t['deps']=['LAB-06','PAPER-05','COSCI-12']
    if t['id']=='GATE-08': t['deps'].append('LAB-02')
    if t['id']=='LAB-07': t['deps'].append('LAB-02')
    if t['id']=='COSCI-06':
        # The schema has to exist before the policy that consumes it, or this task
        # rebuilds a strategy selector with biology bolted on afterwards.
        t['deps']=['COSCI-04','COSCI-13','COSCI-15']
        t['note']='Extends the dossier policy in COSCI-13 over the six-action space; it is not a new general-purpose scientist framework. The deterministic baselines already pass the stability metric by construction, so this task is judged on whether the LLM policy holds the recommendation under irrelevant evidence at matched cost.'
    if t['id']=='MOL-06': t['note']='Requires SynX design staff to prepare/review the compatible library; the estimate is William’s review/integration time, not full design-team effort.'
    if t['id']=='LAB-01':
        # The round question needs a target, an endpoint and a statement of what one round
        # can substantiate. The dossier supplies all three; the historical campaign export
        # does not. Keeping the old MOL-01 dependency would block an Oct 23 gate on a file
        # that has not arrived, for no reason.
        t['deps']=['GATE-04','COSCI-07']
        t['note']='Re-pointed from MOL-01 to COSCI-07 on October 19: the experimental question is answerable from the dossier and does not need the campaign export. Still gated on GATE-04, the named experimental owner.'
    if t['id']=='COSCI-07':
        # The packets exist; the choice of families is not William's to make alone. The window
        # closes on the family decision (Oct 22) so LAB-01 can depend on it, not on the export.
        t['start'], t['end'], t['hours'] = 16, 19, 2
        t['status']='Prepared, pending sign-off'
        t['evidence']='data/processed/targets/{coronaviridae,orthomyxoviridae,henipavirus}.json: 11 targets across 3 families, 59 sourced evidence entries against the seven-axis rubric, 8 update probes, 3 claims flagged as needing a citation. Awaiting the family decision with Askar and SynX.'
    if t['id']=='MOL-01':
        t['status']='Blocked on SynX export'
        t['note']='Blocked until the campaign export arrives. It blocks MOL-02, MOL-03 and therefore MOL-04, the mandatory platform improvement. The building-block registry in MOL-08 holds the schema this export fills.'
    if t['id']=='GATE-04': t['name']='Urgent: confirm lab route, lead times and data return'
    if t['id']=='GATE-06':
        t['name']='W4 feasibility and novelty review after catch-up'
        t['artifact']='gate_w4.md'

anchor=date(2026,9,28)
def calendar_day(n):
    return anchor+timedelta(days=((n-1)//5)*7+(n-1)%5)
for t in tasks:
    t['start_date']=calendar_day(t['start']).isoformat()
    t['end_date']=calendar_day(t['end']).isoformat()
milestones=[dict(day=15,label='Lab route / access',task='GATE-04'),
            dict(day=20,label='Feasibility gate',task='GATE-06'),
            dict(day=24,label='Rubric frozen',task='COSCI-09'),
            dict(day=28,label='Predictions frozen',task='MOL-07'),
            dict(day=30,label='London readiness',task='GATE-08'),
            dict(day=35,label='London complete',task='LAB-10'),
            dict(day=38,label='Lab data due',task='LAB-11'),
            dict(day=39,label='Sweep gate',task='GATE-09'),
            dict(day=43,label='Training complete',task='TRAIN-07'),
            dict(day=45,label='Results frozen',task='GATE-10'),
            dict(day=50,label='Final handoff',task='PAPER-12')]

plan = dict(title="Emergency AI scientist: 10-week delivery plan",
            updated="2026-10-19", calendar_anchor=anchor.isoformat(), today_day=16, workdays=50,
            streams=[dict(id=i, label=l, color=c) for i,l,c in streams],
            tasks=tasks, milestones=milestones,
            assumptions=["Week 1 began September 28. London is the November 8 week; W7 workdays are November 9-13. Exact itinerary and final talk day are unconfirmed.",
                         "Rebaselined from October 14 (W3D3) and updated October 19 (W4D1); elapsed weeks are not marked complete without evidence.",
                         "Task ids carry their stream: GATE decisions and access, COSCI the emergency co-scientist, MOL macrocycle design and platform, LAB the SynX experiment and London, SAFE the safety benchmark, TRAIN the training harness, FCAST forecasting, PAPER the write-up.",
                         "Four tasks are marked Complete and each names a committed artifact; COSCI-07 is Prepared pending sign-off and MOL-01 is Blocked on the SynX export. Everything else is planned, including the lab slot and model access.",
                         "One principal researcher; SynX staff execute the laboratory work.",
                         "Main Tinker implementation is W8-W10; prerequisites are prepared W1-W6."])

def wd(day):
    return f"W{(day-1)//5+1}D{(day-1)%5+1}"

by_id = {t['id']:t for t in tasks}
assert len(by_id)==len(tasks)
for t in tasks:
    assert 1 <= t['start'] <= t['end'] <= 50
    for dep in t['deps']:
        assert dep in by_id, (t['id'],dep)
        # Same-day handoffs are deliberately allowed, with hourly ordering in execution.
        assert by_id[dep]['end'] <= t['start'], (t['id'],dep,'predecessor completes too late')

def visit(task, stack=()):
    assert task not in stack, stack
    for dep in by_id[task]['deps']:
        visit(dep, (*stack,task))
for t in tasks:
    visit(t['id'])

weekly_hours = []
for week in range(1,11):
    lo,hi=5*week-4,5*week
    # Completed work is not remaining capacity, so it is excluded from the weekly load.
    total=sum(t['hours']*max(0,min(hi,t['end'])-max(lo,t['start'])+1)/(t['end']-t['start']+1)
              for t in tasks if t['status']!='Complete')
    weekly_hours.append(round(total,1))
plan['weekly_effort_hours']=weekly_hours
plan['completed_hours']=sum(t['hours'] for t in tasks if t['status']=='Complete')
plan['effort_definition']=('Remaining hands-on hours, spread uniformly across each task window; '
                           'tasks marked Complete are excluded. Replace with real estimates after '
                           'access/throughput review. Travel and SynX staff hours are excluded.')
assert all(h<=40 for h in weekly_hours) and weekly_hours[2]<=24, weekly_hours

(ROOT/'ten-week-detailed-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
fragment = ROOT/'emergency-scientist-detailed-gantt.html'
if fragment.exists():
    html = fragment.read_text()
    html = re.sub(r'(<script id="emergency-plan-data" type="application/json">)[\s\S]*?(</script>)',
                  lambda m:m.group(1)+json.dumps(plan).replace('</','<\\/')+m.group(2),html,count=1)
    fragment.write_text(html)

register = ROOT/'TEN_WEEK_DELIVERY_PLAN.md'
if register.exists():
    intro = register.read_text().split('<!-- GENERATED REGISTER -->')[0]
    lines = [intro, '<!-- GENERATED REGISTER -->', '## Capacity and reserved buffer\n',
             f"Hours below are work **remaining**. The {plan['completed_hours']} hours already "
             "delivered against tasks marked Complete are excluded.\n",
             '| Week | Remaining hands-on hours | Unallocated capacity within a 40-hour week |',
             '| --- | ---: | ---: |']
    for i,h in enumerate(weekly_hours,1):
        if i<3:
            lines.append(f'| W{i} | Elapsed; not reconstructed | Completion not inferred |')
            continue
        capacity=24 if i==3 else 40
        reserve = 'Travel/on-site contingency; not extra coding capacity' if i==7 else f'{capacity-h:.1f} h'+(' across Oct 14–16' if i==3 else '')
        lines.append(f'| W{i} | {h:.1f} h | {reserve} |')
    lines.append(f'''
**W4 does not fit, and the reason matters.** {plan['completed_hours']} hours went into the domain layer on October 19, which was not in the baseline plan. The remaining {weekly_hours[3]:.1f} hours shown for W4 still include MOL-01, MOL-02 and MOL-03, and those cannot start until the SynX campaign export arrives. The four gates (TRAIN-01, COSCI-07 sign-off, LAB-01, GATE-06) plus COSCI-03, COSCI-04 and the SAFE-02/SAFE-03 audits are the week's real load. If the export has not arrived by October 21, the MOL path slides to W5 and MOL-04, the mandatory platform improvement, loses its slack — record it as a blocker with an owner at GATE-06 rather than absorbing it silently.

W7 effort is on-site reviews, observation and light documentation, with travel excluded. W8/W9 buffer absorbs API retries and late experimental QC; W10 buffer absorbs reproduction and review fixes. Average hours within a window are workload estimates, not an appointment calendar. Rely on existing tool code and reviewer/lab support, not a new foundation model or a general-purpose autonomous lab.

## Detailed task register

A task id states its own stream, so no lookup table is needed: **GATE** decisions and access, **COSCI** the emergency co-scientist, **MOL** macrocycle design and platform, **LAB** the SynX experiment and London, **SAFE** the safety benchmark, **TRAIN** the training harness, **FCAST** forecasting, **PAPER** the write-up.

All task estimates refer to William's hands-on contribution; collaborator ownership is stated explicitly. A task is `Planned` unless a **Status** line says otherwise, and a status other than Planned always names its evidence. Every other artifact is a planned output name, not an assertion that the file exists. Same-day dependencies require ordered handoffs within that day.
''')
    for sid,label,_ in streams:
        lines.append(f'### {sid} — {label}\n')
        for t in [x for x in tasks if x['stream']==sid]:
            gate=' · **decision gate**' if t['gate'] else ''
            deps=', '.join(t['deps']) or 'None'
            noun='Output' if t['status']=='Complete' else 'Planned output'
            lines.append(f"#### {t['id']} — {t['name']}\n\n**Window:** {t['start_date']}–{t['end_date']} ({wd(t['start'])}–{wd(t['end'])}) · **Effort:** {t['hours']} h · **Owner:** {t['owner']}{gate}\n\n**Depends on:** {deps}. **{noun}:** `{t['artifact']}`.\n\n**Done when:** {t['acceptance']}\n")
            if t['status']!='Planned':
                detail = f" {t['evidence']}" if t['evidence'] else ''
                lines.append(f"**Status:** {t['status']}.{detail}\n")
            if t['note']:
                lines.append(f"**Constraint:** {t['note']}\n")
    register.write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(tasks=len(tasks), planned_hours=sum(t['hours'] for t in tasks), weekly_hours=weekly_hours),indent=2))
