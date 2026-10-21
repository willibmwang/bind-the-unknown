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
from .lattice import lattice_action
from .streams import stream
from .targets import PACKETS, packet

FAMILIES = ("leader_best", "mixed_best", "challenger_best")

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
    counts = {"attempts": attempts, "leader_best": 0, "mixed_best": 0, "challenger_best": 0,
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
        # The best any policy can do while only ever buying on a target tied for the lead.
        # This is recorded because the hard stratum's headline number turned out to *be* this
        # ceiling: an independent check found the fixed rubric attaining it in every probe,
        # 84 of 84 and 81 of 81, so reporting the rubric's score as a measurement of its
        # reasoning described a structural bound as a behaviour. Recorded per state, the
        # bound becomes the thing to read a real policy against: a policy above it has
        # demonstrably looked past the leader, and one at it has not, whatever its rationale.
        on_lead = [g for (t, _), g in grades.items() if t in lead["tied_for_lead"]]
        leader_ceiling = max(on_lead) if on_lead else 0
        on_leader = sum(1 for t, _ in optimal if t in lead["tied_for_lead"])
        # Three strata, not two. The boundary was `all optimal purchases on the leader` versus
        # anything else, which put the *mixed* states -- some optimal purchases on the leader,
        # some off it -- into the hard stratum. A policy that only examines the leader can
        # still land an optimal purchase there, so the hard stratum was not purely hard and
        # its rate was correspondingly flattered. Mixed states are 31, 29 and 0 of about 394
        # across the three packets, so separating them costs almost nothing and the hard
        # stratum becomes what the docstring always claimed it was.
        family = ("leader_best" if on_leader == len(optimal)
                  else "challenger_best" if on_leader == 0 else "mixed_best")
        counts[family] += 1
        # The label and the count it is derived from are both recorded, and they must agree.
        # Moving this boundary is how an adversary raises headroom by redefinition while every
        # yield, optimal rate and the positive control look ordinary, so the two are checked
        # against each other rather than one being derived and trusted.
        if (family == "leader_best") != (on_leader == len(optimal)) \
                or (family == "challenger_best") != (on_leader == 0):
            raise ValueError(f"stratum {family!r} disagrees with {on_leader} of "
                             f"{len(optimal)} optimal purchases on the leader")
        states.append({"index": index, "family_packet": name, "state_family": family,
                       "decision_round": now, "observed": [list(k) for k in sorted(observed)],
                       "observed_count": len(observed),
                       # The denominator scarcity has to be read against. Packets hold 33, 17
                       # and 9 claims, so twelve observed claims is a third of one packet and
                       # all but nothing of another; a curve plotted on the raw count would
                       # compare three different quantities on one axis.
                       "pool_claims": len(keys),
                       "observed_share": round(len(observed) / len(keys), 4),
                       "retrievable": [list(k) for k in sorted(grades)],
                       "grades": {f"{t}|{a}": g for (t, a), g in sorted(grades.items())},
                       "optimal": [list(k) for k in optimal],
                       "best_grade": best, "chance_grade": round(chance, 4),
                       "leader_ceiling_grade": leader_ceiling,
                       "tied_for_lead": sorted(lead["tied_for_lead"]),
                       "optimal_on_leader": on_leader})
    return states, counts


def sample_audit(states, scored):
    """Is the scored sample distributionally like the pool it was drawn from?

    Only about a fifth of challenger states find a leader partner at the same observed count,
    so `pair` discards most of them, and whoever chooses the draw chooses the sample. An
    independent adversary exploited exactly that: replacing the random draw with an ordering
    on each state's own margin -- the gap between the best purchase on offer and the average
    one -- keeps the states where the reference policy is least able to clear the floor and
    raised headroom from 0.66 to 0.79, and from 0.82 to 1.08 on another packet, while the
    yield, the pair count, the probe count, the optimal rate and the positive control all
    matched the baseline exactly. Nothing in the artifact distinguished it from a random draw.

    This puts the comparison in the artifact. Under a random draw the scored states' mean
    margin agrees with the pool's; under a targeted one it does not, and the selection rule
    has to be correlated with the margin to move the score at all.
    """
    def margin(rows):
        gaps = [r["best_grade"] - r["chance_grade"] for r in rows]
        return None if not gaps else round(sum(gaps) / len(gaps), 4)

    out = {}
    for family in FAMILIES:
        pool = [s for s in states if s["state_family"] == family]
        drawn = [s for s in scored if s["state_family"] == family]
        pooled, sampled = margin(pool), margin(drawn)
        out[family] = {
            "pool_states": len(pool), "scored_states": len(drawn),
            "pool_mean_margin": pooled, "scored_mean_margin": sampled,
            "difference": None if pooled is None or sampled is None
            else round(sampled - pooled, 4)}
    return out


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
        # The paired design is over the two extremes. Mixed states belong to neither and are
        # excluded here rather than folded into one of them; `generate` counts them, so the
        # exclusion is a number in the artifact and not a silent filter.
        if state["state_family"] == "mixed_best":
            continue
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


def answer_key(name, seed, attempts, now=3):
    """The optimal purchase for each state, keyed by the *names* in that state.

    This is a planted contamination, and it exists because the nuisance orbit was until now
    only half validated. Run against the fixed rubric the orbit returns exactly zero with
    zero spread, which shows it does not invent an identity effect where none exists. Nothing
    showed it could detect one. A detector verified only on negatives is not verified.

    So this is the positive control: a policy that has memorised the answer key, which is
    what a model that has seen the benchmark has. The table is built on the unrelabelled
    packet and keyed by the observed claims under their real names, so a renamed instance
    misses every lookup -- exactly as a recalled answer stops being retrievable once the
    entity it was filed under is gone. A policy consulting it should score at the ceiling
    nominally, hold that score when the state is merely rearranged, and fall to the rubric's
    level when the targets are renamed.
    """
    states, _ = generate(name, seed, attempts=attempts, now=now)
    return {(name, frozenset(tuple(k) for k in s["observed"])): tuple(s["optimal"][0])
            for s in states}


def probe(state, policy, agent=None, relabelling=None, deadline=8, key=None):
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
    for claim in observed:
        by_target[claim[0]].observe(pool[claim], now)

    available = set(_retrievable(pool, set(observed), now))
    grades = {tuple(k.split("|")): g for k, g in state["grades"].items()}
    if available != set(grades):
        raise ValueError("The state's offered set does not reproduce; the packet moved under "
                         "it and the recorded floor and ceiling no longer apply")
    # Recomputed rather than trusted. The group certificate shows the oracle is the image on
    # the fully observed packet; this checks it on the partially observed state the policy is
    # actually being asked about, which is a different state and the one the score comes from.
    recomputed, _ = graded_moves(dossiers, pool, set(observed), now)
    # The scale is checked before the image, because a grading function returning levels
    # outside the declared scale is a broken grading function and should be reported as one
    # rather than as a relabelling that failed to be a rename.
    if not set(recomputed.values()) <= set(GRADES):
        raise ValueError(f"Grades {sorted(set(recomputed.values()) - set(GRADES))} are outside "
                         f"the declared scale {sorted(GRADES)}. The grade is ordinal and "
                         f"attainment divides by its range, so rescaling the levels moves "
                         f"every attainment figure while leaving every yield, stratum, "
                         f"optimal rate and the positive control untouched")
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
    elif policy == "lattice":
        # The cost-aware comparator the brief pre-registered, over a state with no slot for a
        # target's name. It reads the lattice and nothing else -- no ranking is passed, so it
        # cannot inherit the caller's tie order the way a policy walking `ranking` does.
        action, target, axis = lattice_action(dossiers, pool, set(observed), now)
    elif policy == "memorised":
        if key is None:
            raise ValueError("The memorised arm is a planted contamination and needs the "
                             "answer key it is meant to have memorised")
        # Looked up by the names in the state, so the lookup survives a rearrangement and
        # misses a rename. On a miss it does what any policy does when recall fails: falls
        # back to the fixed traversal.
        remembered = key.get((name, frozenset(observed)))
        if remembered is not None and remembered in available:
            action, target, axis = "retrieve_evidence", remembered[0], remembered[1]
        else:
            action, target, axis = _rubric_action(by_target, available, ranking)
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
    # Recomputed from the grades, not read from the record. The grades were already verified
    # and the floor and ceiling were not, and an independent adversary used exactly that
    # asymmetry: setting the recorded floor equal to the ceiling on the states where the
    # reference policy would beat the average sends those probes down the `best <= chance`
    # branch, `summarise` drops them as unscored, and the surviving mean is taken over the
    # states where the policy does worse than chance. Headroom went from 0.66 to 1.47 on one
    # packet and 0.82 to 2.03 on another, with the yield, the strata and the positive control
    # all unchanged. A floor that is a function of the policy being scored is circular.
    best = max(recomputed.values())
    chance = sum(recomputed.values()) / len(recomputed)
    if (best, round(chance, 4)) != (state["best_grade"], state["chance_grade"]):
        raise ValueError(f"The state's floor and ceiling do not reproduce: recorded "
                         f"{state['best_grade']}/{state['chance_grade']}, recomputed "
                         f"{best}/{round(chance, 4)}")
    chosen_grade = None if not bought else grades.get(chosen, 0)
    # Attainment against the floor and the ceiling the evaluator computed for this state, the
    # same normalisation the acquisition arm reports: 1.0 is the best purchase available, 0.0
    # is the average purchase available, and a negative value is worse than choosing blindly.
    # A raw grade cannot be compared across states, because the ceiling moves with how much
    # of the decision is still open.
    def scaled(grade):
        return None if grade is None or best <= chance \
            else round((grade - chance) / (best - chance), 4)

    attainment = scaled(chosen_grade)
    ceiling = state["leader_ceiling_grade"]
    recomputed_ceiling = max([g for (t, _), g in recomputed.items()
                              if t in lead["tied_for_lead"]] or [0])
    if ceiling != recomputed_ceiling:
        raise ValueError(f"The state's leader-first ceiling does not reproduce: recorded "
                         f"{ceiling}, recomputed {recomputed_ceiling}")
    return {"family_packet": name, "state_family": state["state_family"],
            "state_index": state["index"], "observed_count": state["observed_count"],
            # The scarcity axis travels with every probe. The raw count is not comparable
            # across packets -- they hold 33, 17 and 9 claims -- so the share is what a curve
            # is plotted against and the count is kept beside it for readability.
            "observed_share": state["observed_share"], "pool_claims": state["pool_claims"],
            "decision_round": now, "policy": policy, "action": action,
            "target": target, "axis": axis,
            "retrieved": bought,
            # A policy declining to buy is not scored as a bad purchase. It made a different
            # decision, and folding a refusal into the attainment rate would let a policy
            # that never buys outscore one that buys imperfectly.
            "chosen_grade": chosen_grade, "best_grade": best, "chance_grade": chance,
            "attainment": attainment,
            # The bound and whether the policy cleared it. A policy pinned at the ceiling has
            # not looked past the leading target, however it explains itself; one above it has.
            "leader_ceiling_grade": ceiling,
            "leader_ceiling_attainment": scaled(ceiling),
            "exceeded_leader_ceiling": None if chosen_grade is None
            else chosen_grade > ceiling,
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


def curve(rows, bins=4):
    """Attainment as a function of how much evidence exists, rather than averaged over it.

    The brief committed to this before any result existed -- S3 asks for "budget-quality
    curves" and for evidence completeness to be varied "separately" -- and it was never
    built. Averaging over the sampling band answers a question nobody asked: the operating
    condition is that there is almost no evidence for the target that matters, and what
    matters is whether decisions degrade gracefully as that bites.

    Two trends are reported and the second is not optional. Attainment is already scaled per
    state between the average purchase on offer and the best one, but the **leader-first
    ceiling itself moves with scarcity**: a nearly-empty dossier leaves every target tied, so
    buying on the leader is buying on everything. If attainment and the ceiling move together
    the policy is not getting better or worse -- the problem is -- and reporting only the
    first would read a change in the instrument as a change in the agent.

    The trend is a rank correlation over individual probes, not over bins, so no binning
    choice can tune it. The bins exist to be read, and their count is reported with each.
    """
    from scipy.stats import spearmanr

    scored = [r for r in rows if r.get("attainment") is not None
              and r.get("observed_share") is not None]
    if len(scored) < 8:
        return {"probes": len(scored), "trend": None, "bins": [],
                "note": "too few scored probes to fit a trend"}

    def trend(key):
        xs = [r["observed_share"] for r in scored]
        ys = [r[key] for r in scored if r.get(key) is not None]
        if len(ys) != len(xs) or len(set(ys)) < 2:
            return None
        result = spearmanr(xs, ys)
        return {"rho": round(float(result.statistic), 4),
                "p": round(float(result.pvalue), 5), "n": len(xs)}

    ordered = sorted(scored, key=lambda r: r["observed_share"])
    width = max(1, len(ordered) // bins)
    grouped = [ordered[i:i + width] for i in range(0, len(ordered), width)][:bins]
    return {"probes": len(scored),
            "trend": trend("attainment"),
            # The control. A ceiling that moves with scarcity means the problem changed, not
            # the policy, and the two trends have to be read against each other.
            "ceiling_trend": trend("leader_ceiling_attainment"),
            "bins": [{"observed_share": [round(g[0]["observed_share"], 4),
                                         round(g[-1]["observed_share"], 4)],
                      "observed_count": [g[0]["observed_count"], g[-1]["observed_count"]],
                      "probes": len(g),
                      "attainment": _mean(g, "attainment"),
                      "leader_first_ceiling": _mean(g, "leader_ceiling_attainment"),
                      "exceeded_ceiling_rate": _rate(g, "exceeded_leader_ceiling")}
                     for g in grouped],
            "interpretation": "Attainment against the share of the packet's claims already "
                              "observed. Read the trend against `ceiling_trend`: if both move "
                              "together the decision problem got easier or harder, and only a "
                              "divergence between them is about the policy."}


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
        # The reported figures must be recomputable from the probe rows the artifact carries,
        # and the counts must add up. Three of an independent adversary's hacks lived in this
        # gap: a one-sided trimmed mean that dropped the reference policy's successes only, a
        # refusal scored as zero rather than excluded, and a thinned sample. Each moved the
        # headline while leaving `purchases` and `declined_to_buy` printing their old values,
        # so the file disagreed with the number computed from it.
        declined = [p for p in rows if p["attainment"] is None]
        if len(scored) + len(declined) != len(rows):
            raise ValueError("Probe rows are neither scored nor declined")
        mistaken = [p for p in declined if p["retrieved"] and p["best_grade"] > p["chance_grade"]]
        if mistaken:
            raise ValueError(f"{len(mistaken)} probes bought something and were still left "
                             f"unscored on a state with headroom; a purchase is not a refusal")
        out[family] = {
            "probes": len(rows), "purchases": len(scored),
            "declined_to_buy": len(rows) - len(scored),
            "attainment": _mean(scored, "attainment"),
            # Read the attainment against this, not against zero. It is the most a policy
            # buying only on the leading target can score, and the fixed rubric sits exactly
            # on it in every probe, so the gap between a policy and this number is the part
            # that is about the policy.
            "leader_first_ceiling": _mean(scored, "leader_ceiling_attainment"),
            # A diagnostic, not a skill measure, and the distinction cost a withdrawn claim.
            # It answers only "did the policy ever buy off the leading target", which a policy
            # working down a fixed list does by accident: measured on the same states in the
            # same run, the state-blind scripted policy clears the ceiling on 0.18 to 0.22 of
            # probes and the hosted model on 0.21. The rate therefore has a chance level near
            # a fifth and does not separate them. What separates them is the attainment above
            # -- 0.369 for the model against -0.099 for the state-blind policy on the same
            # packet -- so a non-zero rate here licenses nothing on its own.
            "exceeded_leader_ceiling_rate": _rate(rows, "exceeded_leader_ceiling"),
            "optimal_rate": _rate(rows, "chose_optimally"),
            "mean_best_grade": _mean(rows, "best_grade"),
            "mean_chance_grade": _mean(rows, "chance_grade"),
            "bought_on_leader_rate": _rate(rows, "chose_leader"),
            "offered_rate": _rate(rows, "chosen_offered")}
    # Per stratum, because the hard stratum is where scarcity is supposed to bite and the
    # easy one is the control that the probe stayed answerable as evidence thinned.
    out["by_scarcity"] = {family: curve([p for p in probes
                                         if p["state_family"] == family])
                          for family in FAMILIES}
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
    # Whole pairs, never individual states. The limit exists to bound inference spend on an
    # agent policy, and drawing states independently would leave the two strata unbalanced by
    # chance -- which is the matched design, discarded to save money. Sampled rather than
    # truncated, because `pair` emits in ascending observed-count order, so a prefix would
    # keep only the states with the fewest claims observed, where the decision is most open
    # and the floor highest, and would do so identically for every element of an orbit,
    # biasing every arm the same way and hiding it. The draw is keyed on the packet and the
    # limit alone, so each orbit element keeps the same pairs.
    if limit and limit // 2 < len(pairs):
        keep = stream(seed, "twin-limit", limit).choice(len(pairs), size=max(1, limit // 2),
                                                        replace=False)
        pairs = [pairs[int(i)] for i in sorted(keep)]
    paired = [s for p in pairs for s in (p["challenger_best"], p["leader_best"])]

    agent = Agent() if policy == "agent" else None
    key = answer_key(name, seed, 16 if smoke else attempts, now) \
        if policy == "memorised" else None
    probes, filtered = [], 0
    for state in paired:
        try:
            probes.append(probe(state, policy, agent, relabelling, key=key))
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
              "sample_audit": sample_audit(states, paired),
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
