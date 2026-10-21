"""Audits on the readouts themselves: which of them can still tell two policies apart.

Three readouts in this project have died, and none of them died loudly. Each kept producing
a number, the number kept going into a summary, and in two of the three cases it was quoted
before anybody noticed it could not move.

* `conflict_retained_rate` reads 1.0 for every policy. The state object retains
  contradictions by construction, so the metric scores the harness and not the decision. It
  is now labelled as such in the code, which is the only reason it is harmless.
* `chosen_matches_reference` is saturated twice over. The fixed rubric reaches 1.0 with
  evidence, so nothing can be observed above it; and separately the hosted agent named the
  reference-best target in sixteen zero-retrieval episodes of sixteen having bought nothing.
  The two findings are distinct and are often run together: under the zero-retrieval control
  the deterministic policies *decline to recommend* and report `None`, so they contribute
  nothing to that match rate and cannot be what saturated it.
* `decision_relevant_rate` saturated against the fixed rubric. The oracle marks a purchase
  relevant when it moves the lead verdict; the rubric resolves the leading target's unknown
  axes; those are the same set, so the trivial policy scored 6 of 6 and a lift of 4.48 --
  above the hosted model's 3.81 -- by definitional collision.

The common shape is that a dead readout looks exactly like a live one from the artifact. It
has a plausible value, a plausible spread across seeds, and nothing in the file says that no
policy could have scored differently. So this module asks of every readout the questions that
would have caught all three, and writes the answers beside the numbers.

Four audits, each tied to a failure that actually happened:

* **Degeneracy.** Does the readout take the same value for every policy? If so it is
  measuring the harness. This is what `conflict_retained_rate` fails.
* **Triviality.** Does a policy with no access to the state, or one that walks a fixed order,
  reach the readout's ceiling? If so the readout is defined on something the trivial policy
  already does. This is what `decision_relevant_rate` fails.
* **Floor.** Is the value reported beside the rate a chance chooser would obtain on the same
  offers? Without one a rate is unreadable, and with one the readout needs no baseline arm.
* **Order.** Is the readout invariant to rearranging the state? A quantity that moves when
  the dossiers are listed in a different order is reporting the list order. This is what
  `decision_relevant` failed until `leaders` carried the tie as a tie.

The audits run the deterministic policies themselves rather than reading recorded artifacts,
so a verdict describes the code as it stands now. A ledger assembled from files on disk would
describe whichever runs happened to be lying around, and the staleness would be invisible --
which is the same failure one level up.
"""
import json

from .common import ROOT, provenance, write_json
from .dossier import leaders
from .emergency import decision_relevant, run_episode
from .targets import PACKETS, packet
from .twins import FAMILIES, generate, run_twins

# Policies with no access to the decision state, or with a fixed traversal of it. A readout a
# member of this set can max out is defined on something that requires no reasoning.
TRIVIAL = ("rubric", "scripted")

# Readouts carried in an episode's metrics, with the floor each must be read against. `None`
# means the readout has no computable floor, which is itself an audit finding rather than an
# omission in this table.
EPISODE_READOUTS = {
    "decision_relevant_rate": "chance_relevant_rate",
    "chosen_matches_reference": None,
    "appropriate_update_rate_on_relevant": None,
    "held_rate_on_irrelevant": None,
    "held_full_ranking_on_irrelevant": None,
    "conflict_retained_rate": None,
    "addressed_conflict_rate": None,
    "axes_substantiated_top_target": None,
}

VERDICTS = ("discriminating", "degenerate", "saturated_by_a_trivial_policy",
            "floor_not_computable", "order_dependent", "not_exercised")


# Ordered from most to least usable, so a readout's headline verdict is the best its
# packets achieved. `min` over this order is what picks it.
PRECEDENCE = ("discriminating", "floor_not_computable", "saturated_by_a_trivial_policy",
              "degenerate", "not_exercised")


def _spread(values):
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return None
    return round(max(present) - min(present), 6)


def packet_verdict(observed, floor_metric):
    """The verdict for one readout on one packet, from the values the policies scored.

    Separated from the run loop so it can be exercised on a constructed table. The defect it
    exists to prevent was a pooling error inside the loop, which no test could reach while
    the arithmetic was tangled with running episodes.

    Order matters. A trivial policy at the ceiling is reported as saturation even when the
    spread is non-zero, because the discrimination that spread represents is all below a
    ceiling a fixed traversal already reaches, and there is no headroom above it for a better
    policy to be observed in. A readout is only called degenerate when the policies actually
    scored the same.
    """
    values = list(observed.values())
    present = [v for v in values if v is not None]
    spread = _spread(values)
    at_ceiling = sorted(p for p, v in observed.items() if p in TRIVIAL and v == 1.0)
    if not present:
        verdict = "not_exercised"
    elif at_ceiling:
        verdict = "saturated_by_a_trivial_policy"
    elif len(present) < 2:
        verdict = "not_exercised"
    elif spread == 0:
        verdict = "degenerate"
    elif floor_metric is None:
        verdict = "floor_not_computable"
    else:
        verdict = "discriminating"
    return {"verdict": verdict, "spread_across_policies": spread, "observed": observed,
            "trivial_policies_at_ceiling": at_ceiling}


def episode_table(packets=None, policies=TRIVIAL, deadline=8):
    """Every readout, and its floor, for every deterministic policy on every packet.

    Run fresh and in one pass. Boolean readouts are carried as floats so a spread is defined
    on them. `None` is kept as `None` rather than coerced to zero: a readout no episode
    exercised and a readout that scored zero are different findings, and folding them
    together is how an unexercised metric comes to look like a measured one.
    """
    names = sorted(PACKETS) if packets is None else list(packets)
    table, floors = {}, {}
    for name in names:
        for policy in policies:
            metrics = run_episode(name, policy, deadline=deadline)["metrics"]
            for readout, floor in EPISODE_READOUTS.items():
                value = metrics.get(readout)
                table.setdefault(readout, {})[(name, policy)] = (
                    None if value is None else float(value))
                if floor:
                    floors.setdefault(readout, {})[(name, policy)] = metrics.get(floor)
    return table, floors, names


def audit_episode_readouts(packets=None, policies=TRIVIAL):
    """Degeneracy, triviality and floor, for each readout an episode reports.

    A readout is called saturated when a trivial policy reaches 1.0, which for every rate
    here is the attainable ceiling. That is deliberately strict: a trivial policy scoring
    0.95 is a warning and a trivial policy scoring 1.0 means no policy can be observed to do
    better, so the readout has no headroom left to measure anything in.
    """
    table, floors, names = episode_table(packets, policies)
    rows = {}
    for readout, cells in table.items():
        # Spread is computed *within* a packet and never across them. Pooling made
        # `held_full_ranking_on_irrelevant` look like it discriminated with a spread of 1.0,
        # when the two policies scored identically inside every packet and the whole range
        # was between-packet variation -- 0.0 on Coronaviridae against 1.0 on Henipavirus.
        # That is the error this codebase forbids everywhere else, committed by the audit
        # that exists to catch errors of exactly that kind.
        per_packet = {}
        for name in names:
            observed = {p: cells[(name, p)] for p in policies}
            per_packet[name] = {
                **packet_verdict(observed, EPISODE_READOUTS[readout]),
                "floor_observed": None if readout not in floors else
                {p: floors[readout][(name, p)] for p in policies}}
        # A readout discriminating on one packet and not another is packet-dependent, not
        # dead, so the headline verdict is the most favourable observed and the per-packet
        # detail is kept beside it. Reporting the worst case would retire a usable readout.
        best = min((r["verdict"] for r in per_packet.values()), key=PRECEDENCE.index)
        rows[readout] = {
            "verdict": best, "floor_metric": EPISODE_READOUTS[readout],
            "verdict_is_the_most_favourable_packet": True,
            "per_packet": per_packet,
            "packets_discriminating": sorted(n for n, r in per_packet.items()
                                             if r["verdict"] == "discriminating")}
    return rows


def audit_order_invariance(packets=None, rounds=(1, 2, 3)):
    """Does the oracle move when the dossiers are listed in a different order?

    Run against the live oracle rather than a recorded number, because the defect this audit
    exists for was invisible in every artifact: the relevant set moved between 26 and 28
    pairs under a rotation of the dossier list, and no field in the file recorded which order
    produced it.
    """
    names = sorted(PACKETS) if packets is None else list(packets)
    rows = {}
    for name in names:
        _, dossiers, entries, _ = packet(name)
        pool = {(t, e["axis"]): e for t, e in entries}
        sizes, leads = set(), set()
        for now in rounds:
            base = set(decision_relevant(dossiers, pool, set(), now))
            for shift in range(len(dossiers)):
                rotated = dossiers[shift:] + dossiers[:shift]
                sizes.add(len(set(decision_relevant(rotated, pool, set(), now))))
                leads.add(tuple(sorted(leaders(rotated, now)["tied_for_lead"])))
            sizes.add(len(base))
        rows[name] = {
            "verdict": "discriminating" if len(sizes) == 1 else "order_dependent",
            "relevant_set_sizes_observed": sorted(sizes),
            "lead_sets_observed": len(leads),
            "rotations_tried": len(dossiers) * len(rounds),
            "tied_at_the_opening_round": sorted(leaders(dossiers, 1)["tied_for_lead"])}
    return rows


def audit_twin_readout(packets=None, attempts=300):
    """The replacement readout, audited by the same questions that condemned its predecessor.

    The positive control is part of the audit and not a separate check. A probe on which the
    rubric does *not* reach the ceiling in the leader stratum is broken rather than hard, so
    the audit reports `probe_unanswerable` for that case -- which is a different finding from
    a readout no policy can score, and calls for fixing the generator rather than the policy.
    """
    names = sorted(PACKETS) if packets is None else list(packets)
    rows = {}
    for name in names:
        states, counts = generate(name, 20261021, attempts=attempts)
        by_policy = {policy: run_twins(name, policy, attempts=attempts)["summary"]
                     for policy in TRIVIAL}
        strata = {}
        for family in FAMILIES:
            values = [by_policy[p][family]["attainment"] for p in TRIVIAL]
            strata[family] = {"observed": {p: by_policy[p][family]["attainment"]
                                           for p in TRIVIAL},
                              "spread_across_policies": _spread(values)}
        answerable = by_policy["rubric"]["leader_best"]["attainment"] == 1.0
        challenger = by_policy["rubric"]["challenger_best"]["attainment"]
        if not counts["challenger_best"]:
            verdict = "not_exercised"
        elif not answerable:
            verdict = "probe_unanswerable"
        elif challenger is not None and challenger >= 1.0:
            verdict = "saturated_by_a_trivial_policy"
        else:
            verdict = "discriminating"
        rows[name] = {
            "verdict": verdict, "state_yield": counts, "strata": strata,
            "positive_control_attained": answerable,
            "trivial_drop": {p: by_policy[p]["contrast"]["leader_minus_challenger"]
                             for p in TRIVIAL},
            # The headroom a model has to be observed in. A readout whose trivial policy
            # already sits at the ceiling of the hard stratum leaves none.
            "challenger_headroom": None if challenger is None else round(1.0 - challenger, 4)}
    return rows


def run_monitor(packets=None, attempts=300, smoke=False):
    """Audit every readout and write the ledger.

    The ledger is the artifact to read before quoting any number from the decision arm. It
    reports a verdict per readout and, for the ones that fail, which of the three recorded
    failures they are repeating.
    """
    names = sorted(PACKETS) if packets is None else list(packets)
    if smoke:
        names, attempts = names[:1], 40
    episodes = audit_episode_readouts(names)
    order = audit_order_invariance(names)
    twins = audit_twin_readout(names, attempts)

    counts = {}
    for row in list(episodes.values()) + list(order.values()) + list(twins.values()):
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    usable = [r for r, row in episodes.items() if row["verdict"] == "discriminating"]
    usable += [f"twins/{n}" for n, row in twins.items() if row["verdict"] == "discriminating"]

    ledger = {"experiment": "monitor", "status": "complete", "smoke": smoke,
              "packets": names, "verdict_counts": counts,
              "discriminating_readouts": sorted(usable),
              "episode_readouts": episodes, "order_invariance": order,
              "twin_readout": twins, "provenance": provenance(),
              "interpretation": "A verdict describes the code as it stands, not a recorded "
                                "run. `degenerate` means the readout scores the harness; "
                                "`saturated_by_a_trivial_policy` means it is defined on "
                                "something a fixed traversal already does; "
                                "`floor_not_computable` means the rate has no stated chance "
                                "level and cannot be read on its own; `order_dependent` "
                                "means it reports the order the state was listed in."}
    output = ROOT / "runs/monitor" / f"{'smoke_' if smoke else ''}ledger.json"
    write_json(output, ledger)
    print(json.dumps({"output": str(output), "verdict_counts": counts,
                      "discriminating_readouts": sorted(usable),
                      "verdicts": {**{k: v["verdict"] for k, v in episodes.items()},
                                   **{f"order/{k}": v["verdict"] for k, v in order.items()},
                                   **{f"twins/{k}": v["verdict"] for k, v in twins.items()}}},
                     indent=2))
    return ledger
