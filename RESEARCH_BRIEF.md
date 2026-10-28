# SynX MATS research brief: emergency AI co-scientist

Updated: 14 October 2026. Status: working proposal with preliminary component pilots; no end-to-end or prospective scientific result established.

## Objective and required scope

Deliver an **emergency AI co-scientist** that helps experts decide what to pursue, what to test next, and how to adapt as evidence changes. Macrocyclic peptide therapeutic discovery is the concrete application and experimental testbed. The final deadline is now **week 10**, with a working system, submission-ready ICML/NeurIPS paper, and reproducible research artifact. Week 1 began September 28, 2026; the final week is November 30–December 4. Publication acceptance is an external outcome.

Emergency conditions must be represented explicitly: incomplete pathogen knowledge, evolving or conflicting evidence, deadlines, tool and compute costs, and limited experimental capacity. The system should maintain an evidence-backed decision state, recommend the next useful action, and explain its rationale and uncertainties for expert review.

Sources: `SynX_MATS_11.0_Project_WW.pdf`; my expanded scope; meeting notes dated 5 and 9 October 2026. All original stretch goals are mandatory:

- Modular prototype covering pathogen intake, evidence retrieval, structure tools, target and epitope ranking, and library design.
- Evidence-backed target dossiers, including confidence, conservation, accessibility, essentiality evidence, reagent tractability, and uncertainties.
- Non-canonical macrocyclic designs and focused libraries compatible with SynX's display platform.
- Historical COVID-19 evaluation using January-March 2020 evidence.
- Case studies covering two prototype pathogen families.
- Implemented and evaluated SynX design-platform improvements.
- One completed mRNA display selection round against a designed library, with analysis and feedback into the system.

My scope expansion includes experimental execution. Scheduling, access, and laboratory work must be coordinated with SynX.

## Meeting evidence and its status

The 5 October discussion motivates pandemic preparedness, transfer across related viruses, extracellular therapeutic opportunities, and an evidence-backed pathogen/target shortlist. The reported display-library scale is approximately 10^12 molecules; this is a physical library scale, not a proposed count of individually simulated structures. Family-wide efficacy is an objective requiring evidence, not an established property.

The 9 October portal demo describes synthetic-codon mappings, chemical identifiers, molecular weights, structures, rotamer distributions, non-canonical mutation campaigns, energy-score distributions, design/refolding stages, and interaction filters. The team reports a correlation between single-mutant experimental measurements and Rosetta scores. Dataset, assay, score definition, replicates, and generality remain unverified.

The phrase "extracellular peptides, anti inhibitors" needs clarification before becoming a target-selection rule. No pathogen, target, assay, or chemical alphabet has been selected by this brief.

## Central research question and candidate contribution

Working question: **can an AI co-scientist make better, faster, evidence-backed scientific decisions under emergency constraints, and adapt appropriately as evidence and experimental results arrive?**

The candidate methodological contribution is an evidence-aware, uncertainty-aware policy for selecting scientific actions under time and resource constraints. Possible actions include retrieving missing evidence, requesting a structure calculation, revisiting a target ranking, using a design tool, or recommending an experiment. A decision state should record the evidence available at that moment, unresolved uncertainties, available resources, and the reason for each recommendation. The exact policy and its novelty remain to be established through baseline experiments and related-work review.

Evaluate the system at three connected levels:

1. **Primary: scientific decisions.** Target prioritization, choice of next action, uncertainty handling, adaptation to evidence, and decision quality versus time and cost.
2. **Supporting: molecular tools.** Reliability of structure/design/scoring tools, molecular ranking, chemical and structural transfer, and any gain from experimental adaptation.
3. **End-to-end validation.** Historical COVID-19 reconstruction, two prototype pathogen families, and the required prospective display round, with conclusions limited to the actual measurements.

Few-shot peptide ranking is a supporting research proposal that may improve the design/selection module. It does not define the overall project. A separate few-shot evaluation proposal is planned for this folder; it is not yet in place or reviewed. Reconcile its assumptions and experiments after it arrives, and measure any module improvement separately from the agent's decision policy.

The first experiments should establish whether emergency constraints create a measurable decision-making challenge and whether the proposed policy improves over simpler workflows. Molecular data audits determine the feasible supporting evaluations. Preserve the overall framing if one proposed molecular method fails.

Assess RFdiffusion and BoltzGen as candidate generation tools, recording the exact variant/version and supported chemical representation. Higher-resolution non-canonical rotamer libraries and expanded Rosetta residue support are candidate platform improvements; select the improvement whose measured failure mode and implementation cost justify it within the project timeline.

## Technical corrections and implications

- CCD means **Chemical Component Dictionary**. A registry should distinguish chemical identity and stereochemistry, component identifiers, coordinates, and the platform's codon-to-building-block mapping. See [wwPDB CCD](https://west.wwpdb.org/data/ccd).
- Rosetta supports non-canonical residues. Valid parameterization, rotamer coverage, and supported scoring terms determine applicability; canonical structure is not a universal requirement. Rosetta uses both physical and statistical terms. Treat natural-residue bias as a hypothesis to measure. See [NCAA documentation](https://docs.rosettacommons.org/docs/latest/rosetta_basics/non_protein_residues/Noncanonical-Amino-Acids), [score-term support](https://docs.rosettacommons.org/docs/latest/rosetta_basics/scoring/score-types), and [rotamer libraries](https://docs.rosettacommons.org/demos/latest/public/make_rot_lib/README).
- Lower scores are interpretable only with a defined metric and comparable protocol. Rosetta energy units are not measured binding free energies. Record total score, interface score, and mutation effects separately, together with scorefunction and sampling settings. See [units](https://docs.rosettacommons.org/docs/latest/rosetta_basics/Units-in-Rosetta) and [InterfaceAnalyzer](https://docs.rosettacommons.org/docs/latest/application_documentation/analysis/interface-analyzer).
- BoltzGen documents complex refolding and a separate binder-only refolding stage; the latter is disabled by default for its peptide protocol. Freeze versions and retain stage-level outputs so generation, refolding, and filtering effects can be evaluated separately. See [official repository](https://github.com/HannesStark/boltzgen).
- A contact-distance filter is a geometric criterion. Binding, functional inhibition, and antiviral activity are separate endpoints. Likewise, display enrichment alone does not establish affinity or antiviral efficacy. State exactly which assay supports each claim.
- Existing [Agent Rosetta research](https://arxiv.org/abs/2603.15952) already couples an LLM to Rosetta and evaluates non-canonical design. A tool-using agent alone is not a sufficient novelty argument. This is an initial related-work finding, not a completed literature review.
- WHO's preparedness framework includes family-based research and representative prototype pathogens. Record the authority, version, and purpose of each prioritization framework rather than treating WHO and CDC terminology as interchangeable. See [WHO framework](https://www.who.int/teams/blueprint/who-r-and-d-blueprint-for-epidemics).

## Evaluation commitments

- Define emergency benchmark episodes with time-stamped evidence, decision checkpoints, resource budgets, and expert review criteria fixed before system outputs are evaluated.
- Compare a transparent fixed rubric, a scripted tool workflow, retrieval with a single LLM response, and an adaptive agent. Match document access, tool capabilities, and underlying molecular models when isolating the decision policy.
- Evaluate evidence support, expert-assessed recommendation quality, appropriate uncertainty, justified evidence updates, and time/cost to a dossier satisfying a predefined review rubric. Dossier readiness is not proof of therapeutic efficacy.
- Vary evidence completeness, contradictions, deadlines, and tool availability separately. Track actual tool calls, latency, costs, errors, and unsuccessful actions rather than comparing nominal budgets alone.
- Audit chemical identity, structures, score provenance, assay endpoints, replicate information, missingness, and censored measurements before fitting models.
- Include unmodified Rosetta ranking, chemistry/similarity heuristics, and simple supervised models as baselines; select additional baselines after reviewing related work and data.
- For any zero-shot or few-shot molecular module, evaluate explicit label budgets. Split by parent peptide/backbone and chemical similarity groups; examine target/family transfer separately where data support it. Keep related single mutants together when evaluating transfer to new parents.
- Use nested development/test separation. Select transformations, filters, and contact criteria on development data, then freeze them for held-out evaluation.
- Report ranking metrics, top-budget selection quality, uncertainty calibration, failures, and compute costs. Match the metric to the assay; quantify uncertainty using independent biological units rather than counting correlated mutants as independent examples.
- Ablate structural information, physics scores, experimental adaptation, and agent policy where applicable. Do not evaluate a Rosetta-optimized method solely by Rosetta energy.
- Use hexbin plots to inspect score/measurement relationships, supplemented by group-stratified analyses for motifs, backbones, chemical classes, and campaign effects.
- Audit historical evidence availability and pretrained-model hindsight leakage. Date-limited retrieval does not by itself remove knowledge already present in model weights. State the limits of historical claims.
- Freeze prospective predictions before experimental results are available. Record all submitted candidates, selection rules, failures, controls, and measured endpoints. Determine what the single required display round can substantiate with the experimental team.
- Release code, configurations, evaluation definitions, and permitted data or a usable public evaluation subset. Resolve publication and data permissions early.

## First experiments: emergency scientific decision-making

These are proposed experiments, not completed results. Start with fixed evidence packets and archived cases so the system-level benchmark can progress while SynX data and laboratory access are arranged. The first two weeks should establish the decision task, its baselines, and the effect of emergency constraints.

### S0. Define and validate emergency decision episodes

**Hypothesis:** deadlines, incomplete evidence, and limited tool/experimental capacity materially affect which scientific actions are useful.

**Design:** construct a small panel of episodes with the evidence initially available, later evidence releases, candidate scientific actions, and declared budgets. Experts define the decision rubric and source-supported uncertainties before reviewing system outputs. Include a complete-evidence condition and a constrained condition for the same cases. Keep later evidence hidden until its release checkpoint.

**Readouts:** expert agreement on the rubric, changes in recommended priorities/actions across conditions, and whether the budget constrains feasible actions. Expert preferences are a reference assessment, not unique biological ground truth.

**Decision:** if constraints do not affect the task, revise the benchmark before claiming an emergency-specific contribution. Begin with public or team-provided cases; include both prototype families as evidence materials become available.

### S1. Establish evidence-grounded decision baselines

**Hypothesis:** structured evidence tracking improves support, uncertainty reporting, and dossier quality over simpler workflows.

**Design:** compare a fixed prioritization rubric, a scripted workflow, retrieval with a single LLM response, and the agent on the same initial evidence and tool capabilities. Start with target dossiers and next-action recommendations; hold the molecular tools fixed. Blind expert ratings to system identity where feasible.

**Readouts:** factual/citation support, unsupported claims, omitted uncertainties, expert-assessed decision rationale, and elapsed time/cost to a dossier meeting the frozen rubric.

**Deliverable/decision:** a baseline scorecard and failure taxonomy. If the agent offers no gain, identify the missing capability before adding orchestration complexity.

### S2. Test adaptation to new and conflicting evidence

**Hypothesis:** the system revises recommendations when relevant evidence warrants it while remaining stable under irrelevant updates.

**Design:** release a relevant document, a contradictory document, or an irrelevant document at a predefined checkpoint. Give all methods identical updates; compare a frozen initial recommendation, a full rerun of the simple workflow, and the adaptive policy. Experts assess warranted changes and unresolved disagreement using only evidence available at that checkpoint.

**Readouts:** appropriateness of changes, support for the revised rationale, uncertainty handling, update latency/cost, and retention of still-valid evidence. A changed ranking is not automatically a better ranking.

**Deliverable/decision:** recommendation trajectories and scored update episodes. Add the Jan-March 2020 COVID-19 setting after the controlled pilot; document pretrained-model hindsight leakage limits. Synthetic or anonymized updates test evidence-following, not the absence of historical knowledge.

### S3. Test next-action selection under deadlines and budgets

**Hypothesis:** an adaptive policy uses constrained resources more effectively than a fixed action order or an unconstrained agent stopped at the same budget.

**Design:** reuse the episodes while varying evidence completeness, tool availability, and resource/deadline constraints. Compare fixed action order, a simple uncertainty-reduction-per-cost heuristic, and the proposed policy using the same action set and executor. Fix heuristic settings on development episodes. Use matched recorded tool outputs for controlled replay, then validate selected episodes with real tools; label replay timing as estimated rather than measured live latency.

**Readouts:** expert-assessed decision quality versus consumed budget and elapsed time; time to an adequately supported dossier; unsuccessful calls and unaddressed uncertainties. Evaluate feasibility and rationale of proposed experimental actions separately when their outcomes are unobserved. Do not assign invented rewards to unperformed experiments.

**Deliverable/decision:** budget-quality curves and policy ablations. Pursue the policy contribution if gains survive held-out episodes and a simple cost-aware baseline. Measure the agent policy with identical molecular models; evaluate improved molecular models in a separate comparison.

### S4. Validate the full decision chain prospectively

**Hypothesis:** the system produces traceable recommendations that remain useful when taken through SynX's real design and experimental workflow.

**Design:** retain the required two-family case studies and display round. Freeze the rationale, evidence state, selected designs/library, model versions, prediction ranks, and analysis before new measurements arrive. Record expert interventions and practical reasons for changes. Use experimental results to evaluate pre-round predictions and document subsequent recommendation updates.

**Readouts:** evidence-to-decision traceability, workflow feasibility, the actual experimental endpoint, and the quality of the resulting update. Distinguish dossier quality, binding/enrichment, functional inhibition, and antiviral efficacy. A single display round supplies an experimental checkpoint; it does not by itself prove repeated adaptive discovery or general therapeutic effectiveness.

**Dependency:** coordinate the experimental slot immediately. The detailed E7 evaluation sketch in the appendix supports this experiment; laboratory execution remains a SynX collaboration.

### First-week working plan

This is a planning estimate. Begin the evidence-based pilot without waiting for a complete molecular campaign export.

| Day | Core emergency co-scientist work | Supporting molecular work |
| --- | --- | --- |
| 1 | S0: define episodes, evidence checkpoints, budgets, and expert rubric | Request campaign export; confirm experimental schedule |
| 2 | S0-S1: prepare source-supported reference dossiers and fixed/scripted baselines | E0 data audit if export is available |
| 3 | S1: compare retrieval baseline and agent under matched access | E1 score/measurement diagnostics if data permit |
| 4 | S2: relevant, contradictory, and irrelevant evidence updates | E2 grouped transfer diagnostics if data permit |
| 5 | Review S0-S2; scope S3 budget replay and identify the policy's main failure mode | Review molecular feasibility; reconcile uploaded few-shot proposal when available |

End-of-week outputs: a benchmark episode specification, baseline dossier scorecard, evidence-update trajectories, an initial resource-accounting report, and a focused policy-development hypothesis. Molecular audit/results are additional outputs as data become available. Experimental coordination proceeds from the start.

## Milestones and decision gates

The project runs to ten weeks, with the final week 30 November – 4 December. The dated gates and the week-by-week shape are in [PROJECT_OVERVIEW_AND_PLAN.md](PROJECT_OVERVIEW_AND_PLAN.md) §7. Task ids carry their stream (GATE, COSCI, MOL, LAB, SAFE, TRAIN, FCAST, PAPER). No elapsed week is marked complete without named evidence. The London visit is the November 8 week, corresponding to W7 workdays November 9–13. Exact itinerary and final presentation day remain unconfirmed.

The newly supplied [Thinking Machines grant](</Users/ww/Desktop/Desktop - William’s MacBook Pro/YALE/26-27 postgrad/Thinking Machines Lab/ww_TML_Safety_Grant.pdf>) adds a proposed ML contribution: forecast larger-budget post-training biological reasoning uplift from zero-shot agentic stress signatures and cheap low-budget measurements. The delivery plan recommends one paper narrative around this forecasting question, with the emergency scientist and required SynX work as applied validation. This recommendation avoids committing to two independent full research papers in the remaining time; the emergency system and every original mandatory output remain required.

The grant's controlled transcriptomic worlds and held-out real studies must remain distinct from peptide binding/MD component benchmarks. Known synthetic mechanisms support ground-truth loss; real perturbation studies support more limited biological conclusions. Neither molecular ranking nor benign transcriptomic reasoning directly measures CBRN threat. The proposed harness measures uplift, forecasting error, uncertainty and resource–capability tradeoffs; it is not an already validated runtime safety filter.

| Weeks / dates | Deliverable and decision |
| --- | --- |
| W1–2 · Sep 28–Oct 9 | Elapsed; meeting notes are available. Do not infer completion of access, benchmark or laboratory milestones. |
| W3 remainder · Oct 14–16 | Reuse existing brief/pilots; verify scope/run evidence; confirm SynX route, slot and turnaround; query live Tinker capabilities; begin safety task/data audit and baseline extension. |
| W4 · Oct 19–23 | Minimal Tinker checkpoint smoke; reproducible development baselines; dataset/split audits; family evidence; emergency stress and feasibility/novelty gate. |
| W5 · Oct 26–30 | Bounded adaptive policy and platform patch; both-family dossiers; historical replay; expert rubric; SynX compatibility review; synthetic/real safety packets. |
| W6 · Nov 2–6 | Complete held-out system evaluation and platform comparison; prospective predictions frozen Nov 4; zero-shot safety signatures, curricula, forecast protocol and London readiness frozen Nov 6. |
| W7 · Nov 9–13 | London facility/design review, actual SynX selection round, expert feedback and readout-handoff agreement; protect travel from critical coding work. |
| W8 · Nov 16–20 | Main Tinker backend/checkpoint/scorer build; throughput/spend gate; b/4b curves and development-only higher-budget targets; lab readout due Nov 18 and analysis by Nov 20. |
| W9 · Nov 23–27 | Save forecasts before held-out outcome reveal; complete high-budget curves by Nov 25; forecast/shift/budget analysis; independent review and results freeze Nov 27. |
| W10 · Nov 30–Dec 4 | Clean reproduction, harness example/report, complete manuscript and artifact, expert review, rehearsals, recorded/offline demo, final presentation/handoff. |

Main harness construction belongs in weeks 8–10; task worlds, splits, signatures, logging and access checks must precede it. Forecaster fitting requires development high-budget labels, scheduled separately from held-out result reveal. Tinker model availability must be verified: the local pilot model is not automatically a supported training model. The exact proposed 12-curve matrix and 5,700-episode evaluation are planning targets subject to measured throughput and a spend ceiling, not submitted jobs.

Laboratory timing is planned backward from usable results by November 18. A booked slot, named SynX experimental owner, compatible preparation route and actual readout turnaround are dependencies, not confirmed facts. SynX staff must execute the laboratory work and support library preparation; William's estimates cover review/integration. Missing a selection/readout leaves the mandatory requirement incomplete. If system or molecular development results are negative, retain them and narrow scientific claims rather than dropping required deliverables. Freeze expert decision metrics, molecular analysis and prospective predictions before final evaluation or experimental outcome reveal.

## HPC pilots staged for 12 October 2026

The first implementation is in [hpc_pilots/README.md](hpc_pilots/README.md), with frozen configuration, dataset provenance, tested runners, and dependency-gated Slurm scripts. The target is the MATS cluster (`w.wang@185.141.218.207`), remote project directory `/mnt/nw/home/w.wang/mats-emergency-20261012`.

Two controlled components of the emergency scientist are implemented: (1) budgeted next-batch selection using ProteinGym's measured GB1 binding landscape, including a synthetic delayed-feedback condition; and (2) allocation of structural analysis using saved ATLAS trajectories, with a held-out replicate as a simulation reference. Strong heuristic policies are compared with a constrained LLM-mediated action policy. These evaluate resource allocation and evidence gathering; they do not establish non-canonical design quality, antiviral activity, or full emergency-scientist performance.

Actual submission IDs and states are recorded in `runs/submissions.json` on the cluster, with local copies after synchronization. Smoke outputs are excluded from scientific summaries. GPU queueing and scientific results must be reported from scheduler/task evidence rather than inferred from staging.

The CPU smoke suite passed on MATS. Submitted jobs are 19169 (160 binding replays), 19170 (12 ATLAS preprocessing tasks), 19171 (structural baselines), 19172 (baseline summary), and 19202 (queued GPU policy comparisons). See [the launch record](hpc_pilots/HPC_LAUNCH.md) for dependencies, model provenance, and monitoring commands. These are submission-checkpoint records, not live scheduler status.

## Appendix: supporting molecular experiment proposals

These are proposed experiments, not completed results. Start with archived measurements and existing structures; confirm resources before scheduling new computation or laboratory work. These molecular experiments support the core decision-making evaluation. The separate few-shot proposal is pending upload and review; E3 remains provisional.

### Recommended order

| ID | Experiment | Priority | Main decision |
| --- | --- | --- | --- |
| E0 | Data and assay reproducibility audit | First | Can the available labels support the proposed evaluation? |
| E1 | Reproduce and dissect Rosetta/experiment correlation | First | Does physics scoring carry useful signal within campaigns and parents? |
| E2 | Transfer and shortcut stress test | First | Does that signal survive new backbones and chemical classes? |
| E3 | Few-shot correction of physics scores | Provisional; review uploaded proposal | Do a few labels improve ranking beyond simple calibration? |
| E4 | Rotamer and refolding sensitivity | Focused diagnostic | Is a practical platform improvement supported by a measured failure mode? |
| E5 | Fixed-budget retrospective selection | After a credible predictor | Does better prediction improve experimental choices? |
| E7 | Prospective, frozen-prediction validation | Plan immediately; execute on laboratory schedule | Does the improvement survive new experimental measurements? |

E0-E2 establish the supporting molecular baselines as data become available. E3 awaits reconciliation with the uploaded proposal. E4-E5 are conditional module diagnostics. S0-S3 define the core system evaluation; S4 and E7 organize prospective validation. The required display round remains mandatory.

### E0. Establish what the experimental labels actually mean

**Hypothesis:** the campaign contains reproducible measurements and enough independent chemical/structural groups to test more than within-parent interpolation.

**Minimum input:** one complete campaign export with parent/mutant identities, chemical definitions, score provenance, assay endpoints, replicates, measurement limits, and candidate-selection history. Request more campaigns for later transfer evaluation.

**First analysis:** build a data dictionary and eligibility table. Count molecules, parent backbones, chemical classes, campaigns, targets, and independent replicate units. Plot replicate agreement and within-condition measurement spread. Keep affinity, enrichment, and inhibition endpoints separate; preserve assay limits as censored observations rather than inventing exact values. Record why each candidate was measured, if known.

**Deliverable:** an audit report, replicate plots, and a frozen initial split manifest. If replicates are unavailable, explicitly mark assay reproducibility as unassessed.

**Decision:** a large row count from one parent does not establish cross-backbone transfer. If labels or independent groups are insufficient, obtain additional exports or narrow the first claim. Historically measured candidates may be a selected subset of the design population; restrict conclusions accordingly.

### E1. Reproduce the reported Rosetta/experiment correlation

**Hypothesis:** a defined Rosetta metric ranks experimental outcomes within matched assay conditions, rather than only separating different campaigns or parent peptides.

**Comparison:** compare total score, available interface metrics, and matched mutant-minus-parent score changes. Include simple chemistry-only descriptors and a random-ranking reference. Examine molecular size and parent/campaign effects as possible explanations for an apparent score relationship.

**Readouts:** pooled and within-parent/campaign rank correlations, top-budget retrieval, and parent-relative experimental changes where compatible parent measurements exist. Use hexbin plots supplemented by small-group scatterplots. For an affinity endpoint such as Kd, define the direction explicitly; do not call arbitrary assay units binding energy. Choose one primary ranking metric before model comparisons.

**Controls:** do not pool incompatible assays. Quantify uncertainty at the relevant independent-group level. Use within-group label permutations as a sanity check where exchangeability is defensible. Fit any score calibration using training data only.

**Deliverable:** one figure showing the pooled relationship beside group-specific relationships, plus a table of score definitions, sample counts, and uncertainty.

**Decision:** if the association survives grouping, use the score as a candidate prior. If it disappears, investigate confounding and assay/protocol mismatch before claiming a useful physics predictor. Failure here is informative and can justify a different contribution.

### E2. Test transfer and expose hidden shortcuts

**Hypothesis:** useful predictions transfer to new peptide parents/backbones and chemical groups, with identifiable degradation under larger distribution shifts.

**Design:** use random row splits as a diagnostic comparison only. Separately evaluate held-out parent/backbone groups, held-out chemical classes, and held-out campaigns/targets when available. Define chemical grouping without experimental labels; record the grouping method. A crossed backbone-and-chemistry holdout is a stronger later test if the dataset supports it.

**Baselines:** raw physics ranking, training-only score calibration, chemical nearest-neighbor prediction, and regularized regression using inexpensive descriptors. Add sequence/motif or backbone-similarity baselines where representations are available. Any label-derived parent statistics must come only from permitted training/support observations.

**Readouts:** ranking quality and top-budget retrieval for each split, with independent-group counts and uncertainty. Compare chemistry-only, structure-only, physics-only, and combined features. Ask whether motif/backbone similarity explains the performance attributed to the richer model.

**Deliverable:** a transfer-performance matrix and an error analysis stratified by chemistry, backbone, and campaign.

**Decision:** a strong random-split result with weak grouped transfer supports an interpolation claim. It does not support zero-shot generalization. Prioritize the shift with adequate data and the clearest reproducible failure; all pathogen-family case studies remain required, even if they cannot yet provide labeled molecular-transfer tests.

### E3. Test whether a handful of labels improves ranking

**Hypothesis:** experimental adaptation corrects systematic score errors more efficiently than chemistry-only learning or simple score rescaling.

**Starting methods:** training-only affine score calibration; a small regularized descriptor model; and a residual model predicting experimental outcome as a calibrated physics prediction plus a learned correction. Begin with ridge regression and, if justified by data size, a Gaussian-process baseline. These are diagnostic baselines, not a novelty claim by themselves.

**Label-budget experiment:** try illustrative budgets of 0, 4, 8, 16, and 32 unique labeled molecules where group sizes permit. Count all support labels and repeated-measurement costs explicitly. Repeat support sampling with paired seeds across methods. Keep hyperparameter selection on development groups; do not tune it on the evaluation queries.

Distinguish two settings: (1) strict transfer, with no labels from the held-out group; and (2) adaptation, where a declared support subset from a new parent/campaign is labeled and the remaining query subset stays hidden. Report these separately. Zero-shot means zero labels from the new evaluation group, not necessarily zero historical training data.

**Ablations/readouts:** physics-only, chemistry-only, structure-only, and combined models; learning curves for ranking and top-budget retrieval; uncertainty coverage where predictive intervals are meaningful. Compare against simple calibration before attributing gains to structural reasoning.

**Deliverable:** a label-efficiency figure and a short list of systematic, reproducible failure modes the eventual method should address.

**Decision:** continue the few-shot direction if gains persist across independent held-out groups and exceed uncertainty in the comparison. If score rescaling explains all gains, use the simpler correction and retain scientific decision-making as the primary contribution. A null result is a valid research decision, not a reason to manufacture complexity.

### E4. Identify a useful platform improvement through sensitivity tests

**Hypothesis:** some ranking errors reflect inadequate conformational sampling or pipeline-stage changes, and a targeted improvement can reduce those errors.

**Rotamer diagnostic:** choose an archived, development-only subset spanning chemical classes and observed prediction errors. Using supported residues and existing platform protocols, compare the current rotamer/sampling setup with one bounded alternative. Match compute when attributing gains to library quality; separately report an increased-compute condition if needed. Repeat stochastic runs to estimate rank stability. Select settings on development data and confirm on untouched examples.

**Refolding diagnostic:** for designs with saved stage outputs, measure changes in structural metrics, contact summaries, and candidate ranks between generation and complex refolding. Evaluate binder-only refolding only when the actual protocol enables and supports it. Compare the current filter against a development-tuned alternative using historical measurements, if those exist for both pass and fail candidates.

**Readouts:** rank stability, failure rate, compute cost, and agreement with experimental endpoints. A better Rosetta score alone is not independent confirmation of a scoring improvement. Unmeasured filtered-out candidates cannot be assumed to be nonbinders.

**Deliverable/decision:** a stage/sensitivity report identifying one supported platform change. Improve rotamer coverage if that is the demonstrated bottleneck; improve stage tracking or supported filtering if those are more consequential. A complete force-field redesign is not the default first experiment.

### E5. Replay experimental choices under a fixed budget

**Hypothesis:** uncertainty-aware or diversity-aware selection discovers useful measured candidates faster than static score ranking.

**Design:** use an archived candidate pool with outcomes observed for every eligible member. Hide outcomes, reveal only selected observations, and let each policy update. Compare random selection, fixed Rosetta ranking, greedy prediction, diversity-aware selection, and one uncertainty-aware acquisition policy. Use identical starting observations, paired random seeds, and a common total budget. If the archive contains only preferentially assayed candidates, state that the replay concerns that restricted pool; do not claim population-wide policy performance.

**Readouts:** cumulative independently defined hits, best observed endpoint, chemical/backbone diversity, prediction uncertainty, and compute. Define hits from the assay before comparing policies. Fit and tune policies on separate development pools. Treat fully informed oracle ranking only as an unattainable reference.

**Decision:** add sophisticated acquisition only if it improves on greedy and diversity baselines. If measured pools are incomplete, defer causal claims about selection benefit and design prospective comparisons with SynX instead.

### E7. Freeze predictions before the required prospective validation

**Hypothesis:** an improvement selected on historical data persists on new measurements.

**Planning:** coordinate the required mRNA display round immediately. With the experimental team, specify the compatible designed library, baseline comparison, feasible controls, measured endpoint, turnaround, and possible orthogonal validation. This is an evaluation brief, not a laboratory protocol.

**Design:** freeze eligible candidates, model version, prediction ranks, selection logic, assay interpretation, and analysis before outcomes arrive. Include model/physics disagreement cases and representative matched comparisons where the display format and assay capacity allow them. Where feasible, compare selection strategies with equal capacity and account for overlap. A pooled library can test variant-level predictions if identities and outcomes are resolvable; it does not automatically identify a causal policy benefit.

**Readouts:** prospective ranking agreement and the actual assay endpoint, with all submitted candidates and failures reported. Use the first round's results as held-out validation of pre-round predictions; post-round model updates require a separate future test. Report enrichment as enrichment unless an additional assay establishes affinity, inhibition, or antiviral activity.

**Decision:** define the strongest supported claim jointly with the experimental team. One successful example may illustrate feasibility; the extent of generalization depends on independent groups and controls.

### Literature anchors for these proposals

Grouped evaluation should preserve independence between training and held-out groups; see the official [scikit-learn grouped cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html). Related experimental work evaluates extrapolation explicitly in [Neural network extrapolation to distant regions of the protein fitness landscape](https://www.nature.com/articles/s41467-024-50712-3) and model-guided selection in [Machine learning-assisted directed protein evolution with combinatorial libraries](https://arxiv.org/abs/1902.07231). These are methodological precedents, not evidence that the proposed SynX experiments will succeed or that the candidate method is novel. The Rosetta, BoltzGen, and Agent Rosetta sources above anchor the platform-specific diagnostics.

