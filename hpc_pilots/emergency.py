"""Emergency decision episode: time-stamped evidence, a deadline, and the stability
requirement that distinguishes emergency decision support from ordinary research help.

A general research agent is scored on whether its final answer is right. An emergency
co-scientist has to be scored on something harder: whether it revises its recommendation
when relevant evidence arrives and *holds* it when irrelevant evidence arrives, while
contradictory evidence is retained alongside the claim it contradicts rather than
overwriting it. That pair of readouts is the research brief's S2 experiment, and it is
what this module measures.

Two honest limits. The deterministic baselines rank from a fixed readiness function, so
they are stable under irrelevant evidence by construction; the metric is informative for
the LLM policy and for any policy that reranks from free-form rationale. And no structure
backend is wired in, so `request_structure` records a charged request and returns no
structural claim rather than inventing one.
"""
import json
import time

from .chemistry import library_spec
from .common import ROOT, Agent, ContentFiltered, digest, provenance, write_json
from .dossier import ACTIONS, AXES, leaders, rank
from .targets import PACKETS, packet

POLICIES = ("rubric", "scripted", "agent")

SYSTEM = (
    "You are an emergency countermeasure co-scientist. A poorly characterised pathogen is "
    "spreading and you must decide which target to pursue and what to test next, from an "
    "incomplete and time-stamped evidence base, under a deadline and a fixed action budget. "
    "Use only the supplied state; never assert biology the state does not cite. Prefer "
    "resolving an axis that gates the decision over confirming one already substantiated. "
    "A hard-gate axis left unresolved blocks the campaign. Return one JSON object.")


def recommendation(d):
    """What the single display round can substantiate, gated by the assay-availability axis.

    The gate reads the *cited and uncontested* assessment. An assay believed available on
    the strength of an uncited claim, or one another observed claim contradicts, does not
    license a functional endpoint in the write-up.
    """
    r = d.readiness()
    assessed = d.assess("assay_availability")
    assay = "strong" if (assessed["verified_assessment"] == "strong"
                         and not assessed["conflict"]) else assessed["assessment"]
    return {"target": d.target, "pathogen": d.pathogen, "family": d.family,
            "compartment": d.compartment,
            "permeability_story_required": d.compartment == "cytosolic",
            "one_round_substantiates":
                "enrichment, plus a functional endpoint if the orthogonal assay is run"
                if assay == "strong" else "enrichment only",
            "endpoint_separation": "Enrichment is not affinity, functional inhibition or "
                                   "antiviral activity. Each requires its own assay.",
            "readiness": r, "open_uncertainties": d.open_uncertainties(),
            "unverified_claims": [e["id"] for e in d.observed if not e["verified"]]}


def _retrievable(pool, retrieved, now):
    """Pairs a policy may ask for at `now`: not yet retrieved, and already in existence.

    Future-dated entries are withheld rather than offered and then refused. Offering one
    tells the policy that this axis is a thing somebody later thought worth writing about,
    which is information from after the decision clock arriving through a channel the date
    boundary does not close -- and it charges a call for the privilege. An axis with no
    evidence and an axis whose only evidence is still in the future must look the same.
    """
    return [key for key, entry in pool.items()
            if key not in retrieved and entry["available_from"] <= now]


def _rubric_action(by_target, available, ranking, axes=None):
    """Fixed rubric: resolve the leading target's unknown axes in rubric order, then recommend.

    `axes` supplies the order the axes were presented in. It defaults to the declaration order,
    which is what every recorded episode used, and exists so an axis-order nuisance reaches the
    policy that is most obviously sensitive to it: this walks the list and takes the first
    unresolved entry, so permuting the list changes what it buys.
    """
    for row in ranking:
        d = by_target[row["target"]]
        for axis in (axes or AXES):
            if d.assess(axis)["assessment"] == "unknown" and (d.target, axis) in available:
                return "retrieve_evidence", d.target, axis
        if d.readiness()["hard_gates_met"]:
            return "recommend_experiment", d.target, None
    return "stop", ranking[0]["target"], None


def decision_relevant(dossiers, pool, retrieved, now):
    """The (target, axis) pairs whose purchase would actually move the decision at `now`.

    Computed by the evaluator, which can see the whole evidence base: each retrievable
    claim is applied to a forked state and the ranking recomputed. A pair counts as
    relevant if observing it changes the leading target, the leader's hard-gate verdict,
    or how much of the leader is substantiated.

    This exists because the readout the decision arm had been scored on is saturated. A
    policy that has memorised which target is best scores 1.0 on target agreement having
    bought no evidence at all, so that metric cannot separate a policy using evidence from
    one reciting a prior. Which axis is worth buying *now* is a different question: it
    depends on what this episode has already observed and on what this packet happens to
    contain, neither of which is recoverable from a target's name.

    It also comes with its own floor. The share of retrievable pairs that are relevant is
    the rate a policy choosing uniformly at random would achieve, and the evaluator can
    compute it exactly, so the metric needs no baseline arm to be interpretable.

    The comparison is against `leaders` rather than `rank(...)[0]`. The first version read
    the top row of the ranking, which under a tie is whichever of the tied targets the
    caller's list happened to put first, so permuting the dossier list changed the size of
    the relevant set -- between 26 and 27 pairs on the Coronaviridae packet -- without any
    policy doing anything. A quantity that moves when the state is rearranged cannot be the
    oracle a policy is scored against, so the tie is now carried as a tie.
    """
    by_target = {d.target: d for d in dossiers}
    unchanged = leaders(dossiers, now)
    relevant = []
    for target, axis in _retrievable(pool, retrieved, now):
        forks = {name: d.fork() for name, d in by_target.items()}
        forks[target].observe(pool[(target, axis)], now)
        if leaders(list(forks.values()), now) != unchanged:
            relevant.append((target, axis))
    return relevant


def _addressed_conflict(later, target, axis):
    """Did the policy do anything about a contradiction once it had landed?

    `conflict_retained` scores the dossier, not the policy: the state object keeps
    contradictions side by side by construction, so that metric reads 1.0 for every
    policy including one that ignores the contradiction entirely, and cannot discriminate.
    This looks at what the policy did next instead. Retrieving further evidence on the
    contested axis, revisiting the ranking, or declining to recommend all count as a
    response; recommending the contested target anyway does not.
    """
    for record in later:
        if record["action"] == "retrieve_evidence" and record["target"] == target \
                and record["axis"] == axis:
            return True
        if record["action"] in ("rerank_targets", "stop"):
            return True
        if record["action"] == "recommend_experiment":
            return record["target"] != target
    return False


def _scripted_action(step, order, ranking):
    """Fixed workflow regardless of state: retrieve down a fixed list, then rank, design, recommend."""
    if step < len(order):
        target, axis = order[step]
        return "retrieve_evidence", target, axis
    tail = ["rerank_targets", "invoke_design_tool", "recommend_experiment"]
    return tail[min(step - len(order), len(tail) - 1)], ranking[0]["target"], None


def reference_ranking(name, deadline, build=None):
    """The ranking the whole evidence base supports by `deadline`.

    Built from a fresh packet so an episode's own dossiers are untouched, and used only
    in scoring -- it is never exposed to a policy. Its purpose is to give the
    zero-retrieval control something to be compared against: a policy that names the
    reference-best target while having paid for no evidence at all did not get there
    from evidence, and the gap between that and its scored performance is the size of
    the hindsight problem this project has so far only been able to name.

    `build` substitutes a different constructor for the same packet, which is how a
    relabelled instance is scored against its own reference rather than against the
    original's. It defaults to `packet`, so every recorded episode is unaffected.
    """
    _, dossiers, entries, probes = (build or packet)(name)
    by_target = {d.target: d for d in dossiers}
    for target, entry in entries:
        by_target[target].observe(entry, deadline)
    for probe in probes:
        by_target[probe["target"]].observe(probe["entry"], deadline)
    return rank(dossiers, deadline)


def run_episode(name, policy, deadline=8, call_budget=8, agent=None, control=None, build=None):
    """Run one decision episode. `control="zero_retrieval"` is the memorisation probe.

    Under that control the policy may not retrieve, and no evidence is pushed to it: it
    must rank and recommend from an empty dossier. Targets in this system are named, so
    pretrained knowledge can substitute for retrieval and the date boundary does nothing
    about it -- the architecture notes say exactly this and leave it unaudited. This is
    the audit. Whatever the control scores is a floor the evidence-using arm has to beat
    before any of its performance can be attributed to the evidence.

    `build` replaces the packet constructor with one returning the same four-tuple, which
    is how a nuisance-relabelled or twinned instance reaches this loop. It is a parameter
    rather than a second loop because an orbit is only comparable with its nominal run if
    both were scored by identical code, and two loops drift. It defaults to `packet`, so
    every episode recorded before it existed is unaffected.
    """
    if control not in (None, "zero_retrieval"):
        raise ValueError(f"Unknown control arm {control!r}")
    spec, dossiers, entries, probes = (build or packet)(name)
    blind = control == "zero_retrieval"
    if blind:
        probes = []
    allowed = [a for a in ACTIONS if not (blind and a == "retrieve_evidence")]
    by_target = {d.target: d for d in dossiers}
    pool = {(t, e["axis"]): e for t, e in entries}
    order = list(pool)
    retrieved, probe_scores, log = set(), [], []
    requested_structures, libraries = [], []
    started = time.perf_counter()
    result = None
    calls = 0
    rounds = 0

    for round_ in range(1, deadline + 1):
        # Checked before the round opens. Pushing evidence into a round the policy cannot
        # act in charged the episode for information it could not use, and the probes due
        # in that round were then skipped by the break below and never scored at all.
        if calls >= call_budget:
            log.append({"round": round_, "action": None, "result": "action budget exhausted"})
            break
        rounds = round_
        due = [p for p in probes if p["entry"]["available_from"] == round_]
        before_rank = rank(dossiers, round_)
        before_axes = {(d.target, a): d.assess(a)["assessment"] for d in dossiers for a in AXES}
        for p in due:
            by_target[p["target"]].observe(p["entry"], round_)

        ranking = rank(dossiers, round_)
        # One point of control for every policy. Gating only the agent's action list left
        # the rubric reading the retrievable set directly and buying evidence anyway.
        available = set() if blind else set(_retrievable(pool, retrieved, round_))
        # Scored against this round's state, before the policy acts, so the oracle cannot
        # be contaminated by the choice it is about to grade.
        relevant = set() if blind else set(decision_relevant(dossiers, pool, retrieved, round_))
        if policy == "rubric":
            action, target, axis = _rubric_action(by_target, available, ranking)
        elif policy == "scripted":
            action, target, axis = _scripted_action(len(retrieved), [] if blind else order, ranking)
        elif policy == "agent":
            if agent is None:
                raise ValueError("Missing agent")
            state = {"decision_round": round_, "rounds_remaining": deadline - round_,
                     "remaining_actions": call_budget - calls,
                     "ranking": ranking,
                     "dossiers": [d.state(round_) for d in dossiers],
                     "retrievable_axes": [{"target": t, "axis": a} for t, a in sorted(available)],
                     "available_actions": {a: ACTIONS[a] for a in allowed}}
            answer = agent.act("Choose the next scientific action for this emergency target "
                               "decision.", state, allowed, fields={
                                   "target": [d.target for d in dossiers],
                                   "axis": list(AXES)}, system=SYSTEM)
            action, target, axis = answer["action"], answer["target"], answer["axis"]
        else:
            raise ValueError(f"Unknown policy {policy!r}")
        calls += 1

        record = {"round": round_, "action": action, "target": target, "axis": axis,
                  "pushed_evidence": [p["entry"]["id"] for p in due]}
        if action == "retrieve_evidence" and not blind:
            # Recorded whether or not the pair exists, so a policy cannot improve its score
            # by asking for claims that were never on offer.
            record["decision_relevant"] = (target, axis) in relevant
            record["relevant_offered"] = len(relevant)
            record["retrievable_offered"] = len(available)
        if action == "retrieve_evidence":
            if blind:
                raise ValueError("retrieve_evidence reached the zero-retrieval control")
            entry = pool.get((target, axis))
            if entry is None:
                record["result"] = "no evidence exists for this target and axis; action charged"
            else:
                observed = by_target[target].observe(entry, round_)
                retrieved.add((target, axis))
                record["result"] = "observed" if observed else "refused: dated after decision clock"
        elif action == "request_structure":
            requested_structures.append({"target": target, "round": round_})
            record["result"] = ("no structure backend configured; request recorded and charged, "
                                "no structural claim produced")
        elif action == "rerank_targets":
            record["result"] = "reranked"
        elif action == "invoke_design_tool":
            spec_out = library_spec()
            libraries.append({"target": target, "round": round_, "library": spec_out})
            record["result"] = "focused thioether macrocycle library proposed; no property predicted"
        elif action == "recommend_experiment":
            result = recommendation(by_target[target])
            record["result"] = "recommendation issued"
        elif action == "stop":
            record["result"] = "stopped"
        log.append(record)

        after_rank = rank(dossiers, round_)
        for p in due:
            t, ax = p["target"], p["entry"]["axis"]
            assessment = by_target[t].assess(ax)
            score = {"kind": p["kind"], "target": t, "axis": ax, "round": round_,
                     "top_before": before_rank[0]["target"], "top_after": after_rank[0]["target"]}
            if p["kind"] == "relevant":
                score["appropriate_update"] = bool(
                    before_axes[(t, ax)] != assessment["assessment"]
                    or before_rank[0]["target"] != after_rank[0]["target"])
            elif p["kind"] == "irrelevant":
                # Two strengths of stability. The recommendation holding is the requirement;
                # the full order holding is stricter, and irrelevant evidence can legitimately
                # reorder targets below the leader without changing what to do next.
                score["held_recommendation"] = before_rank[0]["target"] == after_rank[0]["target"]
                score["held_full_ranking"] = ([r["target"] for r in before_rank]
                                              == [r["target"] for r in after_rank])
            elif p["kind"] == "contradictory":
                score["conflict_retained"] = bool(assessment["conflict"])
            probe_scores.append(score)

        if action in ("recommend_experiment", "stop"):
            break

    ranking = rank(dossiers, rounds)
    # Scored after the loop, because it depends on what the policy did in later rounds.
    # A probe landing in the final round leaves no opportunity to respond, and is left
    # unscored rather than counted as a failure to respond.
    for score in probe_scores:
        if score["kind"] == "contradictory":
            later = [r for r in log if r["round"] > score["round"]]
            score["addressed_conflict"] = (_addressed_conflict(later, score["target"],
                                                               score["axis"])
                                           if later else None)

    def rate(kind, key):
        rows = [s[key] for s in probe_scores if s["kind"] == kind and s.get(key) is not None]
        return None if not rows else sum(rows) / len(rows)

    if result is not None:
        terminal = "recommended"
    elif any(r["action"] == "stop" for r in log):
        terminal = "stopped"
    elif any(r["action"] is None for r in log):
        terminal = "action_budget_exhausted"
    else:
        terminal = "deadline_reached"

    # Chance is the share of retrievable pairs that were decision-relevant, averaged over
    # the rounds where the policy actually bought something, so the floor is matched to the
    # decisions taken rather than to rounds the policy never spent on.
    scored_picks = [r for r in log if r.get("decision_relevant") is not None]
    relevant_picks = sum(1 for r in scored_picks if r["decision_relevant"])
    # The statistic is observed over expected. Expected is the per-round probability of
    # picking a relevant pair by chance, summed over the rounds the policy actually spent
    # on. Reporting it as a count rather than a rate is what makes it readable: a round
    # where nothing relevant remained contributes nothing to either side, so two policies
    # that buy a different number of times can legitimately share a lift.
    offers = [r["relevant_offered"] / r["retrievable_offered"] for r in scored_picks
              if r["retrievable_offered"]]
    expected_relevant = sum(offers) if offers else None
    chance_rate = expected_relevant / len(offers) if offers else None

    reference = reference_ranking(name, deadline, build)
    chosen = result["target"] if result else None
    metrics = {"policy": policy, "control": control, "family": spec["family"],
               # The comparison the control exists for. Under zero retrieval this says
               # whether the policy reached the reference answer having bought nothing.
               "evidence_retrieved": len(retrieved),
               # Were the purchases the ones that moved the decision? Reported beside the
               # rate a uniform chooser would have achieved on the same offers, which is
               # the comparator -- the zero-retrieval control makes no purchases and so
               # cannot saturate this the way it saturates target agreement.
               "retrievals_scored": len(scored_picks),
               "decision_relevant_retrievals": relevant_picks,
               "decision_relevant_rate": round(relevant_picks / len(scored_picks), 4)
               if scored_picks else None,
               "chance_relevant_rate": round(chance_rate, 4) if chance_rate is not None else None,
               "expected_relevant_retrievals": round(expected_relevant, 3)
               if expected_relevant is not None else None,
               "decision_relevant_lift": round(relevant_picks / expected_relevant, 3)
               if expected_relevant else None,
               "reference_top_target": reference[0]["target"],
               "chosen_target": chosen,
               "chosen_matches_reference": None if chosen is None
               else chosen == reference[0]["target"],
               "rounds_used": rounds,
               "deadline_rounds": deadline, "actions_used": calls, "action_budget": call_budget,
               "recommendation_issued": result is not None,
               "deadline_met": result is not None and rounds <= deadline,
               "appropriate_update_rate_on_relevant": rate("relevant", "appropriate_update"),
               "held_rate_on_irrelevant": rate("irrelevant", "held_recommendation"),
               "held_full_ranking_on_irrelevant": rate("irrelevant", "held_full_ranking"),
               # A harness invariant, kept as a regression check on the state object: it
               # reads 1.0 for every policy and must never be reported as policy skill.
               "conflict_retained_rate": rate("contradictory", "conflict_retained"),
               "addressed_conflict_rate": rate("contradictory", "addressed_conflict"),
               "terminal_condition": terminal,
               "actions_unspent_at_exit": call_budget - calls,
               "axes_substantiated_top_target": ranking[0]["substantiated_axes"],
               "unknown_axes_top_target": len(ranking[0]["unknown_axes"]),
               "contested_axes_top_target": len(ranking[0]["contested_axes"]),
               "uncited_axes_top_target": len(ranking[0]["uncited_axes"]),
               "unsupported_assessments": sum(len(d.unsupported()) for d in dossiers),
               "structure_requests_unserviced": len(requested_structures),
               "libraries_proposed": len(libraries),
               "refused_future_evidence": sum(len(d.rejected) for d in dossiers),
               # An episode that needed several attempts to obtain a parseable action is
               # not the same observation as one that needed one, so the count travels
               # with the result rather than only with the agent report.
               "agent_filtered_attempts": agent.filtered if agent else None,
               "agent_transport_retries": agent.transport_retries if agent else None,
               "policy_wall_seconds": time.perf_counter() - started}
    return {"metrics": metrics, "ranking": ranking, "recommendation": result, "trace": log,
            "probe_scores": probe_scores, "requested_structures": requested_structures,
            "libraries": libraries,
            "units": "deadline and release times are synthetic decision rounds, not calendar time",
            "interpretation": "Decision-process measurement. No binding, enrichment, inhibition "
                              "or antiviral claim is made or implied."}


def run_emergency(name, policy, smoke=False, control=None, repeat=1):
    """Run `repeat` independent episodes, each to its own artifact.

    The output path previously carried only the packet, the policy and the control, so a
    second episode of the same cell overwrote the first and the arm could not hold more
    than one observation per cell. Every recorded emergency result was therefore n = 1,
    which is not enough to report anything about a policy whose output varies between
    calls. A replicate index in the path is the whole fix.
    """
    if name not in PACKETS:
        raise ValueError(f"Unknown family {name!r}")
    deadline = 4 if smoke else 8
    prefix = "smoke_" if smoke else ""
    outputs = []
    for replicate in range(1, repeat + 1):
        agent = Agent() if policy == "agent" else None
        try:
            result = run_episode(name, policy, deadline=deadline, agent=agent, control=control)
        except ContentFiltered as filtered:
            # One packet's generations are filtered intermittently. Losing the rest of the
            # batch to it would make the recorded replicate count a function of the
            # provider's filter rather than of the design, and would quietly bias the
            # surviving sample toward whatever the filter happens to permit. The episode
            # is recorded as missing data with a stated cause and the batch continues;
            # `status` is not "complete", so no aggregate will read it as a result.
            label_policy = agent.label if agent else policy
            suffix = f"{f'_{control}' if control else ''}_rep{replicate:02d}"
            output = ROOT / "runs/emergency" / f"{prefix}{name}_{label_policy}{suffix}_filtered.json"
            write_json(output, {
                "experiment": "emergency", "family_packet": name, "policy": label_policy,
                "control": control, "smoke": smoke, "status": "filtered",
                "replicate": replicate, "replicates_requested": repeat,
                "reason": str(filtered), "agent": agent.report() if agent else None,
                "provenance": provenance()})
            outputs.append(str(output))
            print(json.dumps({"output": str(output), "replicate": replicate,
                              "status": "filtered"}))
            continue
        label_policy = agent.label if agent else policy
        result.update({"experiment": "emergency", "family_packet": name, "policy": label_policy,
                       "control": control, "smoke": smoke, "status": "complete",
                       "replicate": replicate, "replicates_requested": repeat,
                       "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
                       "provenance": provenance()})
        if agent:
            result["agent"] = agent.report()
        # The control writes its own file, so it can never be pooled with the evidence arm.
        suffix = f"{f'_{control}' if control else ''}{f'_rep{replicate:02d}' if repeat > 1 else ''}"
        output = ROOT / "runs/emergency" / f"{prefix}{name}_{label_policy}{suffix}.json"
        write_json(output, result)
        outputs.append(str(output))
        print(json.dumps({"output": str(output), "replicate": replicate,
                          "metrics": result["metrics"]}, indent=2))
    return outputs
