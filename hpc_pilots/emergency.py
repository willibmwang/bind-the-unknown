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
from .common import ROOT, Agent, digest, write_json
from .dossier import ACTIONS, AXES, rank
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
    """What the single display round can substantiate, gated by the assay-availability axis."""
    r = d.readiness()
    assay = d.assess("assay_availability")["assessment"]
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


def _retrievable(pool, retrieved):
    """Pull-retrievable (target, axis) pairs not yet observed."""
    return [key for key in pool if key not in retrieved]


def _rubric_action(by_target, pool, retrieved, ranking):
    """Fixed rubric: resolve the leading target's unknown axes in rubric order, then recommend."""
    for row in ranking:
        d = by_target[row["target"]]
        for axis in AXES:
            if d.assess(axis)["assessment"] == "unknown" and (d.target, axis) in pool \
                    and (d.target, axis) not in retrieved:
                return "retrieve_evidence", d.target, axis
        if d.readiness()["hard_gates_met"]:
            return "recommend_experiment", d.target, None
    return "stop", ranking[0]["target"], None


def _scripted_action(step, order, ranking):
    """Fixed workflow regardless of state: retrieve down a fixed list, then rank, design, recommend."""
    if step < len(order):
        target, axis = order[step]
        return "retrieve_evidence", target, axis
    tail = ["rerank_targets", "invoke_design_tool", "recommend_experiment"]
    return tail[min(step - len(order), len(tail) - 1)], ranking[0]["target"], None


def run_episode(name, policy, deadline=8, call_budget=8, agent=None):
    spec, dossiers, entries, probes = packet(name)
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
        rounds = round_
        due = [p for p in probes if p["entry"]["available_from"] == round_]
        before_rank = rank(dossiers, round_)
        before_axes = {(d.target, a): d.assess(a)["assessment"] for d in dossiers for a in AXES}
        for p in due:
            by_target[p["target"]].observe(p["entry"], round_)

        if calls >= call_budget:
            log.append({"round": round_, "action": None, "result": "action budget exhausted"})
            break

        ranking = rank(dossiers, round_)
        if policy == "rubric":
            action, target, axis = _rubric_action(by_target, pool, retrieved, ranking)
        elif policy == "scripted":
            action, target, axis = _scripted_action(len(retrieved), order, ranking)
        elif policy == "agent":
            if agent is None:
                raise ValueError("Missing agent")
            state = {"decision_round": round_, "rounds_remaining": deadline - round_,
                     "remaining_actions": call_budget - calls,
                     "ranking": ranking,
                     "dossiers": [d.state(round_) for d in dossiers],
                     "retrievable_axes": [{"target": t, "axis": a}
                                          for t, a in _retrievable(pool, retrieved)],
                     "available_actions": ACTIONS}
            answer = agent.act("Choose the next scientific action for this emergency target "
                               "decision.", state, list(ACTIONS), fields={
                                   "target": [d.target for d in dossiers],
                                   "axis": list(AXES)}, system=SYSTEM)
            action, target, axis = answer["action"], answer["target"], answer["axis"]
        else:
            raise ValueError(f"Unknown policy {policy!r}")
        calls += 1

        record = {"round": round_, "action": action, "target": target, "axis": axis,
                  "pushed_evidence": [p["entry"]["id"] for p in due]}
        if action == "retrieve_evidence":
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
    def rate(kind, key):
        rows = [s[key] for s in probe_scores if s["kind"] == kind and key in s]
        return None if not rows else sum(rows) / len(rows)

    metrics = {"policy": policy, "family": spec["family"], "rounds_used": rounds,
               "deadline_rounds": deadline, "actions_used": calls, "action_budget": call_budget,
               "recommendation_issued": result is not None,
               "deadline_met": result is not None and rounds <= deadline,
               "appropriate_update_rate_on_relevant": rate("relevant", "appropriate_update"),
               "held_rate_on_irrelevant": rate("irrelevant", "held_recommendation"),
               "held_full_ranking_on_irrelevant": rate("irrelevant", "held_full_ranking"),
               "conflict_retained_rate": rate("contradictory", "conflict_retained"),
               "axes_substantiated_top_target": ranking[0]["substantiated_axes"],
               "unknown_axes_top_target": len(ranking[0]["unknown_axes"]),
               "unsupported_assessments": sum(len(d.unsupported()) for d in dossiers),
               "structure_requests_unserviced": len(requested_structures),
               "libraries_proposed": len(libraries),
               "refused_future_evidence": sum(len(d.rejected) for d in dossiers),
               "policy_wall_seconds": time.perf_counter() - started}
    return {"metrics": metrics, "ranking": ranking, "recommendation": result, "trace": log,
            "probe_scores": probe_scores, "requested_structures": requested_structures,
            "libraries": libraries,
            "units": "deadline and release times are synthetic decision rounds, not calendar time",
            "interpretation": "Decision-process measurement. No binding, enrichment, inhibition "
                              "or antiviral claim is made or implied."}


def run_emergency(name, policy, smoke=False):
    if name not in PACKETS:
        raise ValueError(f"Unknown family {name!r}")
    agent = Agent() if policy == "agent" else None
    deadline = 4 if smoke else 8
    result = run_episode(name, policy, deadline=deadline, agent=agent)
    label_policy = agent.label if agent else policy
    result.update({"experiment": "emergency", "family_packet": name, "policy": label_policy,
                   "smoke": smoke, "status": "complete",
                   "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json")})
    if agent:
        result["agent"] = agent.report()
    prefix = "smoke_" if smoke else ""
    output = ROOT / "runs/emergency" / f"{prefix}{name}_{label_policy}.json"
    write_json(output, result)
    print(json.dumps({"output": str(output), "metrics": result["metrics"]}, indent=2))
