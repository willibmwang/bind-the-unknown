"""One-step decision probes over matched states, and why the readout had to move here.

The decision arm has now saturated twice, for two different reasons, and the second one is
the argument for this module.

Target agreement saturated because the targets are named: forbidden from retrieving
anything, the policy named the reference-best target in sixteen episodes of sixteen. The
replacement readout was decision-relevance -- score each purchase against whether observing
it would move the lead verdict, which depends on the evidence state and not on the target's
name, so a prior cannot supply it.

That readout saturates too, and measurably. The fixed rubric scores 6 of 6 on Coronaviridae
and 6 of 6 on Orthomyxoviridae, a lift of 4.48 and 2.70 over chance, which is *higher* than
the hosted model's recorded 3.81. The reason is a definitional collision rather than skill.
`_rubric_action` resolves the leading target's unknown axes in declaration order.
`decision_relevant` marks a purchase relevant when it moves the lead verdict, and once any
target is partly substantiated the only purchases that move that verdict are on the leader.
The two sets are the same set. The relevant pool collapses 28, 5, 4, 3, 2, 1 as the episode
runs, until it contains exactly the leader's remaining unknown axes -- which is the rubric's
definition. A policy scoring 1.0 against an oracle that was independently defined to select
its own output has not been measured.

The general lesson, and the reason this is not a third patch: **a one-step oracle defined on
the decision structure is gameable by any policy that shares the structure.** No amount of
rewriting the oracle fixes that while the question stays "was this purchase relevant",
because relevance is computed from the same ranking rule the policy is built around.

What a prior cannot supply, and a fixed order cannot stumble into, is *which target* the
decision currently turns on. So the probe here is built from pairs of states that differ in
exactly that:

* **Leader-gated.** Every decision-relevant purchase is on the current leader. A policy that
  only ever examines the leader scores perfectly. This is the positive control: it proves
  the probe is answerable at all, and an instrument on which the rubric does *not* score
  near 1.0 is broken rather than discriminating.
* **Challenger-gated.** The decision-relevant purchases are on a target that is not leading,
  because that target is one cited claim away from taking or tying the lead. Resolving the
  leader further changes nothing. A policy that examines only the leader scores at chance
  here, and so does a policy reciting which target it remembers being best, because the
  prior points at the leader and the answer is elsewhere.

The pair is matched on packet, on the number of claims pre-observed and on the decision
round, so the only difference between the two halves is which claims were observed. The
headline is the challenger-gated rate against its exact chance floor; the leader-gated rate
beside it is what licenses reading the first one as a measurement.

A probe is one decision, not an episode. That is deliberate: eight rounds cost eight
inference calls to produce one scored action in most of them, and the question here needs
many independent states rather than long trajectories. The state handed to the policy is
assembled by the same code path an episode uses, so the task is the episode's task.
"""
import json
import time

from .common import ROOT, Agent, ContentFiltered, digest, provenance, write_json
from .dossier import ACTIONS, AXES, leaders, rank
from .emergency import SYSTEM, _retrievable, _rubric_action, _scripted_action, decision_relevant
from .streams import stream
from .targets import PACKETS, packet

FAMILIES = ("leader_best", "challenger_best")

# Sampling bounds for the pre-observed subset. Below the floor every target is still tied on
# an empty dossier and every purchase breaks the tie, so there is nothing to discriminate;
# above the ceiling the decision is settled and no purchase moves it. Both ends are reported
# by `generate` as yield rather than quietly skipped.
OBSERVED_FRACTION = (0.15, 0.65)

# How far one observation moves the decision, ordered by what it settles. Binary relevance --
# did the lead verdict change at all -- cannot separate these, and that is why it collapsed:
# resolving the leader's next axis and handing the lead to a challenger both scored 1, so a
# policy walking the leader's axes in declaration order was indistinguishable from one
# choosing the purchase that decides the campaign.
GRADES = {0: "the lead verdict does not move",
          1: "the leading target is better substantiated; the contest is unchanged",
          2: "which targets are still contending changes, and more than one remains",
          3: "the contest resolves to a single target"}


def grade_move(before, after):
    """Grade one observation by how much of the decision it settles.

    Ordered by decision value rather than by magnitude of any single field. Narrowing six
    tied targets to one ends the contest and is the most valuable thing a purchase can do;
    changing which targets contend is next; improving the leader's own substantiation while
    the same set still contends is real but least valuable, because the campaign is no closer
    to being decided than it was.

    A contradiction can demote a leader, since a contested axis stops counting as
    substantiated, so a purchase can move the lead backwards. That is still a move of the
    contest and grades as one; the grade measures how much the decision changed, not whether
    the change was welcome.
    """
    if after == before:
        return 0
    if after["tied_for_lead"] != before["tied_for_lead"]:
        return 3 if len(after["tied_for_lead"]) == 1 < len(before["tied_for_lead"]) else 2
    return 1


def graded_moves(dossiers, pool, retrieved, now):
    """The grade of every purchase on offer, computed by the evaluator on forked states.

    One fork per offered pair, as `decision_relevant` does, but keeping the magnitude instead
    of thresholding it at non-zero. The evaluator can do this because it holds the whole
    evidence base; a policy cannot, which is the point.
    """
    by_target = {d.target: d for d in dossiers}
    before = leaders(dossiers, now)
    out = {}
    for target, axis in _retrievable(pool, retrieved, now):
        forks = {name: d.fork() for name, d in by_target.items()}
        forks[target].observe(pool[(target, axis)], now)
        out[(target, axis)] = grade_move(before, leaders(list(forks.values()), now))
    return out, before


def generate(name, seed, attempts=400, now=3):
    """Sample pre-observed evidence subsets and keep every state with something to get right.

    A state is kept when the best purchase on offer grades above the average purchase on
    offer, because that is exactly the condition under which choosing well is distinguishable
    from choosing at random. States failing it are not discarded quietly: the counts record
    how many were settled already and how many were still wide open, and those two numbers
    are the first thing to read, since a packet that cannot produce discriminating states
    cannot support the readout at all.

    Each kept state is labelled by where its best purchase sits. `challenger_best` means the
    highest-grading purchase is on a target that is not leading -- the stratum a policy
    examining only the leader, or reciting which target it remembers being best, must fail.
    `leader_best` is the positive control: both kinds of policy can score there, so a probe
    on which they do not is broken rather than discriminating.

    States are always sampled from the unrelabelled packet and translated afterwards by
    `probe`. Sampling under each relabelling instead looked equivalent and was not: the
    subset is drawn over `sorted(pool)`, whose order is the target names, so every element of
    a nuisance orbit saw a *different* set of states and the orbit confounded renaming the
    target with resampling the problem. The confound was visible -- the name-blind rubric
    reported a 0.11 identity effect it cannot have -- which is what sampling once and
    translating fixes.
    """
    if name not in PACKETS:
        raise ValueError(f"Unknown family {name!r}")
    _, _, entries, _ = packet(name)
    pool = {(t, e["axis"]): e for t, e in entries}
    keys = sorted(pool)
    low, high = OBSERVED_FRACTION
    counts = {"attempts": attempts, "leader_best": 0, "challenger_best": 0,
              "no_purchase_moves_the_decision": 0, "every_purchase_grades_alike": 0}
    states = []
    for index in range(attempts):
        rng = stream(seed, "twin-state", name, index)
        size = int(rng.integers(max(1, int(low * len(keys))), max(2, int(high * len(keys)))))
        observed = [keys[int(i)] for i in rng.choice(len(keys), size=size, replace=False)]
        _, dossiers, _, _ = packet(name)
        by_target = {d.target: d for d in dossiers}
        for key in observed:
            by_target[key[0]].observe(pool[key], now)

        grades, lead = graded_moves(dossiers, pool, set(observed), now)
        if not grades:
            counts["no_purchase_moves_the_decision"] += 1
            continue
        best = max(grades.values())
        chance = sum(grades.values()) / len(grades)
        if best == 0:
            counts["no_purchase_moves_the_decision"] += 1
            continue
        if best <= chance:
            counts["every_purchase_grades_alike"] += 1
            continue

        optimal = sorted(k for k, g in grades.items() if g == best)
        family = ("leader_best" if all(t in lead["tied_for_lead"] for t, _ in optimal)
                  else "challenger_best")
        counts[family] += 1
        states.append({"index": index, "family_packet": name, "state_family": family,
                       "decision_round": now, "observed": [list(k) for k in sorted(observed)],
                       "observed_count": len(observed),
                       "retrievable": [list(k) for k in sorted(grades)],
                       "grades": {f"{t}|{a}": g for (t, a), g in sorted(grades.items())},
                       "optimal": [list(k) for k in optimal],
                       "best_grade": best, "chance_grade": round(chance, 4),
                       "tied_for_lead": sorted(lead["tied_for_lead"]),
                       "optimal_on_leader": sum(1 for t, _ in optimal
                                                if t in lead["tied_for_lead"])})
    return states, counts


def pair(states, seed):
    """Match each challenger-gated state to a leader-gated one with the same observed count.

    Matched on the number of claims observed because the relevant set shrinks as the dossier
    fills, so the chance floor moves with it. Comparing a challenger-gated state at twenty
    observed claims against a leader-gated one at five would compare two different floors and
    attribute the difference to the policy. Unmatched states are returned separately rather
    than paired to the nearest available count, which would reintroduce the same confound in
    a smaller and less visible form.
    """
    by_count = {}
    for state in states:
        by_count.setdefault((state["observed_count"], state["state_family"]), []).append(state)
    pairs, unmatched = [], []
    for (count, family), rows in sorted(by_count.items()):
        if family != "challenger_best":
            continue
        partners = list(by_count.get((count, "leader_best"), []))
        rng = stream(seed, "twin-pair", count)
        order = rng.permutation(len(rows))
        for slot, which in enumerate(order):
            if slot < len(partners):
                pairs.append({"observed_count": count, "challenger_best": rows[int(which)],
                              "leader_best": partners[slot]})
            else:
                unmatched.append(rows[int(which)])
    return pairs, unmatched


def _state_for_policy(dossiers, available, now, deadline, remaining, allowed):
    """The decision state, in the shape an episode presents it.

    Assembled from the same fields `run_episode` assembles, so a probe measures the episode's
    task rather than a simplified relative of it. Divergence here would be invisible and would
    make every probe number incomparable with every episode number.
    """
    return {"decision_round": now, "rounds_remaining": deadline - now,
            "remaining_actions": remaining,
            "ranking": rank(dossiers, now),
            "dossiers": [d.state(now) for d in dossiers],
            "retrievable_axes": [{"target": t, "axis": a} for t, a in sorted(available)],
            "available_actions": {a: ACTIONS[a] for a in allowed}}


def translate(state, relabelling):
    """Carry a state in base coordinates over to a relabelled packet.

    The state's target names are mapped through the substitution table and nothing else
    moves: the same claims are pre-observed, at the same round, so the two halves of an
    orbit are the same problem under two names rather than two problems. `probe` recomputes
    the grades on the relabelled packet and checks them against the translated record, so
    the claim that the oracle is the image is verified per probe and not inherited from the
    group certificate.
    """
    if relabelling is None:
        return state
    table = relabelling.table()

    def name(target):
        return table.get(target, target)

    return {**state,
            "observed": [[name(t), a] for t, a in state["observed"]],
            "retrievable": [[name(t), a] for t, a in state["retrievable"]],
            "optimal": [[name(t), a] for t, a in state["optimal"]],
            "grades": {f"{name(k.split('|')[0])}|{k.split('|')[1]}": g
                       for k, g in state["grades"].items()},
            "tied_for_lead": sorted(name(t) for t in state["tied_for_lead"])}


def probe(state, policy, agent=None, relabelling=None, deadline=8):
    """Put one state to one policy and score the single action it returns.

    Scores three things, and the third is the diagnostic that explains the other two.
    `chosen_relevant` is the readout. `chance` is the exact rate a uniform choice over the
    offered pairs would achieve, computed by the evaluator, so the number needs no baseline
    arm. `chose_leader` records whether the policy bought on a target tied for the lead,
    which is what separates a policy that failed because it reasoned badly from one that
    failed because it never looked past the leader.
    """
    from .invariance import relabel_packet

    name = state["family_packet"]
    state = translate(state, relabelling)
    now = state["decision_round"]
    build = packet if relabelling is None else (lambda n: relabel_packet(n, relabelling))
    _, dossiers, entries, _ = build(name)
    pool = {(t, e["axis"]): e for t, e in entries}
    by_target = {d.target: d for d in dossiers}
    observed = [tuple(k) for k in state["observed"]]
    for key in observed:
        by_target[key[0]].observe(pool[key], now)

    available = set(_retrievable(pool, set(observed), now))
    grades = {tuple(k.split("|")): g for k, g in state["grades"].items()}
    if available != set(grades):
        raise ValueError("The state's offered set does not reproduce; the packet moved under "
                         "it and the recorded floor and ceiling no longer apply")
    # Recomputed rather than trusted. The group certificate shows the oracle is the image on
    # the fully observed packet; this checks it on the partially observed state the policy is
    # actually being asked about, which is a different state and the one the score comes from.
    recomputed, _ = graded_moves(dossiers, pool, set(observed), now)
    if recomputed != grades:
        raise ValueError("The graded moves are not the image of the relabelled state; the "
                         "relabelling changed the decision rather than renaming it")
    optimal = {tuple(k) for k in state["optimal"]}
    lead = leaders(dossiers, now)
    allowed = list(ACTIONS)
    ranking = rank(dossiers, now)

    started = time.perf_counter()
    if policy == "rubric":
        action, target, axis = _rubric_action(by_target, available, ranking)
    elif policy == "scripted":
        action, target, axis = _scripted_action(len(observed), sorted(pool), ranking)
    elif policy == "agent":
        if agent is None:
            raise ValueError("Missing agent")
        answer = agent.act("Choose the next scientific action for this emergency target "
                           "decision.",
                           _state_for_policy(dossiers, available, now, deadline,
                                             deadline - now, allowed),
                           allowed, fields={"target": [d.target for d in dossiers],
                                            "axis": list(AXES)}, system=SYSTEM)
        action, target, axis = answer["action"], answer["target"], answer["axis"]
    else:
        raise ValueError(f"Unknown policy {policy!r}")

    bought = action == "retrieve_evidence"
    chosen = (target, axis) if bought else None
    best, chance = state["best_grade"], state["chance_grade"]
    chosen_grade = None if not bought else grades.get(chosen, 0)
    # Attainment against the floor and the ceiling the evaluator computed for this state, the
    # same normalisation the acquisition arm reports: 1.0 is the best purchase available, 0.0
    # is the average purchase available, and a negative value is worse than choosing blindly.
    # A raw grade cannot be compared across states, because the ceiling moves with how much
    # of the decision is still open.
    attainment = (None if chosen_grade is None or best <= chance
                  else round((chosen_grade - chance) / (best - chance), 4))
    return {"family_packet": name, "state_family": state["state_family"],
            "state_index": state["index"], "observed_count": state["observed_count"],
            "decision_round": now, "policy": policy, "action": action,
            "target": target, "axis": axis,
            "retrieved": bought,
            # A policy declining to buy is not scored as a bad purchase. It made a different
            # decision, and folding a refusal into the attainment rate would let a policy
            # that never buys outscore one that buys imperfectly.
            "chosen_grade": chosen_grade, "best_grade": best, "chance_grade": chance,
            "attainment": attainment,
            "chose_optimally": None if not bought else chosen in optimal,
            "chosen_offered": None if not bought else chosen in available,
            "chose_leader": None if not bought else target in lead["tied_for_lead"],
            "retrievable_offered": len(available), "optimal_offered": len(optimal),
            "tied_for_lead": sorted(lead["tied_for_lead"]),
            "policy_wall_seconds": round(time.perf_counter() - started, 4)}


def _rate(rows, key):
    kept = [r for r in rows if r.get(key) is not None]
    return None if not kept else round(sum(bool(r[key]) for r in kept) / len(kept), 4)


def _mean(rows, key):
    kept = [r[key] for r in rows if r.get(key) is not None]
    return None if not kept else round(sum(kept) / len(kept), 4)


def summarise(probes):
    """Per-stratum attainment, and the drop between the two strata.

    The drop is the readout. A policy that computes which purchase settles the decision
    scores alike whether or not the best purchase happens to sit on the leading target. A
    policy that walks the leader's axes in order, and equally a policy reciting which target
    it remembers being best, scores near 1.0 when the best purchase is on the leader and near
    0.0 when it is not -- because both of them answer from something other than the state.

    Both strata are reported rather than only the drop. A policy failing both has a different
    problem from one failing only the challenger stratum, and a single difference cannot tell
    those apart: the leader stratum is the positive control that the probe was answerable.
    """
    out = {}
    for family in FAMILIES:
        rows = [p for p in probes if p["state_family"] == family]
        scored = [p for p in rows if p["attainment"] is not None]
        out[family] = {
            "probes": len(rows), "purchases": len(scored),
            "declined_to_buy": len(rows) - len(scored),
            "attainment": _mean(scored, "attainment"),
            "optimal_rate": _rate(rows, "chose_optimally"),
            "mean_best_grade": _mean(rows, "best_grade"),
            "mean_chance_grade": _mean(rows, "chance_grade"),
            "bought_on_leader_rate": _rate(rows, "chose_leader"),
            "offered_rate": _rate(rows, "chosen_offered")}
    lead = out["leader_best"]["attainment"]
    challenge = out["challenger_best"]["attainment"]
    out["contrast"] = {
        "leader_minus_challenger": None if lead is None or challenge is None
        else round(lead - challenge, 4),
        "interpretation": "Attainment is scaled per state between the average purchase on "
                          "offer (0.0) and the best one (1.0), so it is comparable across "
                          "states whose decisions are open to different degrees. The drop "
                          "from the leader stratum to the challenger stratum is the "
                          "measurement. The leader stratum alone is the positive control: a "
                          "probe no policy can score there is broken, not hard."}
    return out


def run_twins(name, policy, seed=None, attempts=400, limit=None, smoke=False,
              relabelling=None, now=3):
    """Generate matched states for one packet and probe one policy against all of them.

    `relabelling` composes this instrument with the nuisance group: the same matched states
    are rebuilt from a relabelled packet, so the challenger-gated rate can be read with the
    target names present and again with them replaced. The two instruments answer different
    questions and composing them answers a third -- whether what survives the rename is the
    part that tracks the decision.
    """
    from .invariance import certify, residual_candidates

    if name not in PACKETS:
        raise ValueError(f"Unknown family {name!r}")
    seed = PACKETS[name].get("seed", 20261021) if seed is None else seed
    certificate = None if relabelling is None else certify(name, relabelling)
    if certificate is not None and not certificate["certified"]:
        raise ValueError(f"Relabelling {relabelling.code()} is not certified: "
                         f"{certificate['checks']}")

    states, counts = generate(name, seed, attempts=16 if smoke else attempts, now=now)
    pairs, unmatched = pair(states, seed)
    paired = [s for p in pairs for s in (p["challenger_best"], p["leader_best"])]
    if limit:
        paired = paired[:limit]

    agent = Agent() if policy == "agent" else None
    probes, filtered = [], 0
    for state in paired:
        try:
            probes.append(probe(state, policy, agent, relabelling))
        except ContentFiltered:
            # Recorded as missing data rather than dropped. Losing a probe to a provider
            # filter and not saying so makes the scored sample a function of the filter, and
            # biases it toward whatever the filter happens to permit.
            filtered += 1

    label = agent.label if agent else policy
    result = {"experiment": "twins", "family_packet": name, "policy": label,
              "seed": seed, "decision_round": now, "smoke": smoke,
              "status": "complete" if probes else "no probes generated",
              "state_yield": counts, "pairs": len(pairs),
              "unmatched_challenger_states": len(unmatched),
              "probes_scored": len(probes), "probes_filtered": filtered,
              "summary": summarise(probes), "probes": probes,
              "relabelling": None if relabelling is None else relabelling.record(),
              "certificate": certificate,
              "residual_identifier_candidates":
                  None if relabelling is None else residual_candidates(
                      [translate(s, relabelling) for s in paired]),
              "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
              "provenance": provenance(),
              "units": "decision rounds are synthetic, not calendar time",
              "interpretation": "Decision-process measurement on matched states. No binding, "
                                "enrichment, inhibition or antiviral claim is made or implied."}
    if agent:
        result["agent"] = agent.report()

    suffix = f"_{relabelling.kind}-{relabelling.code()}" if relabelling else ""
    output = ROOT / "runs/twins" / f"{'smoke_' if smoke else ''}{name}_{label}{suffix}.json"
    write_json(output, result)
    print(json.dumps({"output": str(output), "state_yield": counts,
                      "summary": result["summary"]}, indent=2))
    return result
