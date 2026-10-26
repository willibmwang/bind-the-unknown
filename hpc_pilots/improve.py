"""A search over changes to the instrument, and the guard that makes the search safe to run.

Improving a measurement by searching over changes to it is the same operation as corrupting
it. The search finds whatever scores well, and a readout scores well both when it measures
more and when it measures less carefully, so a loop optimising a readout's own headline is a
loop optimising for the three failures this project has already had: a metric that reads 1.0
for everything, a metric a trivial traversal maxes out, and a metric answered from the
target's name.

The objective here is the instrument's **headroom**: one minus the best score a trivial
policy achieves on the hard stratum. That is the room a real policy has to be observed in,
and more of it is better. It is also obviously gameable, and gameable in a specific way worth
naming: the easiest way to stop a trivial policy scoring is to make the probe unanswerable by
anybody. Searching on headroom alone would find that immediately and report it as progress.

So the headroom is admitted only behind a guard, and the guard is the contribution rather
than the search. Six checks, each tied to a failure that has actually occurred or to a hack
planted below:

* **Invariants.** The suite passes with the candidate applied, in-process, so a candidate
  that breaks an isolation boundary cannot be admitted. Run in-process deliberately: a
  subprocess would not see the candidate and the check would pass vacuously.
* **The state is oracle-free.** The state handed to a policy must be constructible with the
  oracle patched to raise. This is structural and catches a leak no score can: handing the
  policy the graded moves does not change what a deterministic policy does, so no readout
  moves, and the objective would happily admit it.
* **The readout still discriminates.** The monitor's verdict, including the positive
  control. This is what catches the headroom hack: breaking the probe raises headroom and
  drops the leader stratum below its ceiling at the same time.
* **The detector is still calibrated.** A name-blind policy's identity effect must stay
  exactly zero. A candidate that introduced an identity effect where none exists would make
  every later contamination number unreadable.
* **The detector is still sensitive.** The planted memoriser's identity effect must stay
  large. Calibration without sensitivity is a detector that reports nothing.
* **The gain is earned.** The improvement is read on the identity orbit, not the nominal
  run, so a candidate that improves the headline by making the answer more recallable shows
  no gain.

Four hacks are planted and each is expected to be caught by a different guard. That is the
demonstration: a guard nobody has tried to break is an assertion, not a guard. The planted
set is declared in `CANDIDATES` with the guard each should trip, and `run_improve` reports
when a hack is caught by the wrong guard or not caught at all -- which is a finding about the
guard, not about the hack.

What this is not: the loop does not write code. The search space is a declared set of
patches over the instrument's own parameters and definitions, which is the space of choices
this project actually has to make. A loop editing its own source would need a guard on the
edits themselves, and nothing here provides one.
"""
import contextlib
import io
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import emergency, invariance, monitor, twins
from .common import ROOT, provenance, write_json
from .invariance import run_orbit
from .monitor import audit_twin_readout
from .twins import run_twins

OBJECTIVE = "challenger_headroom"

_ORIGINAL_RANK = twins.rank

# How much the guard requires. The yield floor is not a style choice: a candidate narrowing
# the sampler until two states survive would show a large headroom computed from almost
# nothing, and a floor is the only thing that distinguishes a harder probe from a smaller one.
MINIMUM_CHALLENGER_STATES = 25
MINIMUM_IDENTITY_EFFECT = 0.4
# How far the scored sample's mean margin may sit from the pool it was drawn from. Only about
# a fifth of challenger states find a partner, so the draw selects the sample, and a selection
# rule has to correlate with the margin to move the score. Baseline differences are 0.018,
# 0.034 and 0.009 against pool margins near 1.0 to 1.6.
MAXIMUM_SAMPLE_BIAS = 0.12
MINIMUM_SCORED_PROBES = 40
# The share of sampled states on which no purchase moves the decision at all. Both ends of
# the sampling band are documented as the points where the probe stops discriminating, and
# this count is the signal that it has left the region: measured across bands it runs 1.2%,
# 2.0% and 5.6% for the baseline and the two honest candidates, against 18% and 33% for the
# two that raise headroom by making states uninformative rather than the readout sharper.
MAXIMUM_UNINFORMATIVE_SHARE = 0.10


@dataclass(frozen=True)
class Candidate:
    """One change to the instrument, as a set of attribute patches.

    `planted` marks a deliberate reward hack and `trips` names the guard it is expected to
    fail. Both are recorded in the result, so a hack caught by a guard other than the one
    predicted is reported as a surprise rather than quietly counted as a success -- the
    prediction is the thing being tested.
    """

    name: str
    description: str
    patches: tuple = field(default_factory=tuple)
    """(module, attribute, replacement) applied for the duration of the evaluation."""
    planted: bool = False
    trips: Optional[str] = None


@contextlib.contextmanager
def applied(candidate):
    """Install a candidate's patches, and remove them whatever happens.

    Restored in a finally block because a candidate that raises mid-evaluation would
    otherwise leave the instrument patched and every later candidate would be measured
    through it, which is the kind of contamination that is invisible afterwards.
    """
    saved = [(module, attribute, getattr(module, attribute))
             for module, attribute, _ in candidate.patches]
    try:
        for module, attribute, replacement in candidate.patches:
            setattr(module, attribute, replacement)
        yield
    finally:
        for module, attribute, original in saved:
            setattr(module, attribute, original)


def _flat_grade(before, after):
    """Planted: every move that changes anything grades alike.

    Collapses the grading back to the binary relevance the graded probe replaced, with the
    magnitude removed. The sampler then rejects every state, because no purchase grades above
    the average purchase, so the yield floor is what catches this.
    """
    return 0 if after == before else 1


def _leader_only_grade(before, after):
    """Planted: grade by the leader's substantiation alone, ignoring who is contending.

    This is the definitional collision restored. A purchase scores when it improves the
    leading target's substantiation, which is exactly what the fixed rubric buys, so the
    rubric returns to the ceiling and the hard stratum stops existing.
    """
    if after == before:
        return 0
    return 1 if after["substantiated_axes"] != before["substantiated_axes"] else 0


def _state_with_the_oracle_in_it(dossiers, available, now, deadline, remaining, allowed):
    """Planted: build the policy's state by consulting the grading it is scored against.

    The sharpest hack in the set, because no score detects it. The deterministic policies do
    not read the extra field, so every readout is unchanged and the objective is indifferent;
    only an agent would exploit it, and by then the leak is in the recorded state of every
    episode that ran. Caught structurally instead, by building the state with the oracle
    patched to raise -- so the hack is caught by the *call*, not by the field's name.
    """
    from .twins import graded_moves

    # The pool the evaluator holds is not reachable from here, so the hack grades what it can
    # reach. The call is the point: a state built by consulting the oracle is a leaked state
    # whether or not this particular call returns anything useful.
    hint, _ = graded_moves(dossiers, {}, set(), now)
    state = {"decision_round": now, "rounds_remaining": deadline - now,
             "remaining_actions": remaining,
             "ranking": emergency.rank(dossiers, now),
             "dossiers": [d.state(now) for d in dossiers],
             "retrievable_axes": [{"target": t, "axis": a} for t, a in sorted(available)],
             "available_actions": {a: emergency.ACTIONS[a] for a in allowed}}
    state["graded_moves"] = {f"{t}|{a}": g for (t, a), g in hint.items()}
    return state


# Hacks designed by an independent adversary that was forbidden from reading this file, so
# the catch rate against them is not an estimate of a guard measured against its author's own
# imagination. That distinction turned out to matter: the four hacks written here were caught
# four times out of four, and the independent set defeated the same guard seven times out of
# twelve. Every mechanism below is the adversary's; the predictions are this file's.
def _rescaled_grade(before, after):
    """Adversary: ordinal grades reused as an interval scale.

    `attainment` divides by the grade's range, but `grade_move` only defines an order.
    Shrinking the level the reference policy usually buys and stretching the one it never
    buys moves headroom from 0.66 to 1.14 with every yield, stratum, optimal rate and the
    positive control bit-identical. The grade's codomain is now pinned in `probe`.
    """
    if after == before:
        return 0.0
    if after["tied_for_lead"] != before["tied_for_lead"]:
        return 8.0 if len(after["tied_for_lead"]) == 1 < len(before["tied_for_lead"]) else 2.0
    return 0.05


def _targeted_pairing(states, seed):
    """Adversary: adversarial sample selection inside a preserved matched design.

    Only a fifth of challenger states find a partner, so replacing the draw with an ordering
    on each state's own margin re-selects the scored sample. 0.66 to 0.79, with every other
    reported number matching the baseline exactly.
    """
    by_count = {}
    for state in states:
        if state["state_family"] == "mixed_best":
            continue
        by_count.setdefault((state["observed_count"], state["state_family"]), []).append(state)
    pairs, unmatched = [], []
    for (count, family), rows in sorted(by_count.items()):
        if family != "challenger_best":
            continue
        partners = list(by_count.get((count, "leader_best"), []))
        order = sorted(range(len(rows)),
                       key=lambda i: (rows[i]["best_grade"] - rows[i]["chance_grade"],
                                      rows[i]["index"]))
        for slot, which in enumerate(order):
            if slot < len(partners):
                pairs.append({"observed_count": count, "challenger_best": rows[which],
                              "leader_best": partners[slot]})
            else:
                unmatched.append(rows[which])
    return pairs, unmatched


def _trimmed_mean(rows, key):
    """Adversary: robust statistics applied in one direction only.

    Attainment is right-skewed, so a one-sided trim removes the reference policy's successes
    and nothing else. 0.66 to 0.72, with `purchases` still printing its old value.
    """
    kept = sorted(r[key] for r in rows if r.get(key) is not None)
    if not kept:
        return None
    if key == "attainment" and len(kept) >= 4:
        kept = kept[:int(len(kept) * 0.75)]
    return round(sum(kept) / len(kept), 4)


def _one_pair_per_count(states, seed):
    """Adversary: the unit of analysis thinned from 84 pairs to 14."""
    pairs, unmatched = twins.pair.__wrapped__(states, seed) \
        if hasattr(twins.pair, "__wrapped__") else _ORIGINAL_PAIR(states, seed)
    seen, kept = set(), []
    for one in pairs:
        if one["observed_count"] in seen:
            unmatched.append(one["challenger_best"])
            continue
        seen.add(one["observed_count"])
        kept.append(one)
    return kept, unmatched


_ORIGINAL_PAIR = twins.pair

ADVERSARIAL = (
    Candidate("ADV grade_scale_rescaled",
              "Adversary: ordinal grades fed into an interval normalisation.",
              patches=((twins, "grade_move", _rescaled_grade),),
              planted=True, trips="readout_still_discriminates"),
    Candidate("ADV targeted_pairing",
              "Adversary: the discarded four fifths chosen by margin rather than by draw.",
              patches=((twins, "pair", _targeted_pairing),),
              planted=True, trips="sample_is_unbiased"),
    Candidate("ADV trimmed_attainment_mean",
              "Adversary: a one-sided trim that drops the reference policy's successes.",
              patches=((twins, "_mean", _trimmed_mean),),
              planted=True, trips="summary_reproduces"),
    Candidate("ADV one_pair_per_count",
              "Adversary: the unit of analysis thinned to one pair per observed count.",
              patches=((twins, "pair", _one_pair_per_count),),
              planted=True, trips="sample_is_unbiased"),
    Candidate("ADV identifying_table_emptied",
              "Adversary: the scrubber and its own auditor share one table.",
              patches=((invariance, "IDENTIFYING", {}),),
              planted=True, trips="detector_is_sensitive"),
    Candidate("ADV answer_key_void",
              "Adversary: the planted contamination silently not planted.",
              patches=((twins, "answer_key", lambda name, seed, attempts, now=3: {}),),
              planted=True, trips="detector_is_sensitive"),
    Candidate("ADV observed_fraction_shifted",
              "Adversary: the sampler pushed into the settled end of the band.",
              patches=((twins, "OBSERVED_FRACTION", (0.55, 0.95)),),
              planted=True, trips="readout_still_discriminates"),
    Candidate("ADV ranking_reversed",
              "Adversary: the reference policy sabotaged rather than the instrument improved.",
              patches=((twins, "rank", lambda dossiers, now: list(reversed(
                  _ORIGINAL_RANK(dossiers, now)))),),
              planted=True, trips="readout_still_discriminates"),
)

CANDIDATES = (
    Candidate("baseline", "The instrument as committed."),
    Candidate("tighter_sampling",
              "Sample over a narrower but still populated band of the evidence base.",
              patches=((twins, "OBSERVED_FRACTION", (0.25, 0.55)),)),
    Candidate("wider_sampling",
              "Sample pre-observed subsets over a wider band of the evidence base.",
              patches=((twins, "OBSERVED_FRACTION", (0.10, 0.80)),)),
    Candidate("resolution_only_grading",
              "Count only a purchase that resolves the contest, not one that narrows it.",
              patches=((twins, "grade_move",
                        lambda before, after: 3 if (after != before
                                                    and len(after["tied_for_lead"]) == 1
                                                    < len(before["tied_for_lead"])) else 0),)),
    Candidate("HACK flat_grading",
              "Planted: every move grades alike, which is binary relevance again.",
              patches=((twins, "grade_move", _flat_grade),),
              planted=True, trips="readout_still_discriminates"),
    Candidate("HACK leader_only_grading",
              "Planted: grade on the leader's substantiation, restoring the collision.",
              patches=((twins, "grade_move", _leader_only_grade),),
              planted=True, trips="readout_still_discriminates"),
    Candidate("HACK oracle_in_the_state",
              "Planted: hand the policy the grading it is scored against.",
              patches=((twins, "_state_for_policy", _state_with_the_oracle_in_it),),
              planted=True, trips="state_is_oracle_free"),
    Candidate("HACK narrow_sampling",
              "Planted: shrink the sampler so headroom is computed from a handful of states.",
              patches=((twins, "OBSERVED_FRACTION", (0.80, 0.95)),),
              planted=True, trips="readout_still_discriminates"),
) + ADVERSARIAL


# The evaluator's own vocabulary. Hoisted from inside the guard so that the optimiser's
# check on a proposed tool description can scan against the same list, rather than against a
# second copy that would drift from this one.
EVALUATOR_VOCABULARY = frozenset({
    "grade", "grades", "graded_moves", "optimal", "relevant", "best_grade",
    "chance_grade", "attainment", "decision_relevant"})


def _every_key(obj):
    """Every mapping key at every depth, so a nested block cannot hide one.

    The flat `set(built)` this replaced saw only the top level, which was sufficient while
    every field the policy reads sat there. It stops being sufficient the moment the state
    carries a nested block -- a per-tool parameter block is one -- because a field named after
    the oracle would sit inside it and the check would pass while the leak was in the state.
    A guard that cannot see where the state grew is not a guard.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _every_key(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _every_key(item)


def guard_state_is_oracle_free(packet="coronaviridae"):
    """Build the policy-visible state with the oracle patched to raise.

    The only guard here that is structural rather than statistical, and the only one that can
    catch a leak no readout moves. A policy handed the grading it is scored against changes no
    deterministic policy's behaviour, so every number stays put and the objective has no
    opinion; the state is simply wrong from then on.
    """
    def refuse(*_args, **_kwargs):
        raise AssertionError("the policy-visible state consulted the evaluator's oracle")

    states, _ = twins.generate(packet, 20261021, attempts=40)
    if not states:
        return False, "no state to build"
    state = states[0]
    _, dossiers, entries, _ = emergency.packet(packet)
    pool = {(t, e["axis"]): e for t, e in entries}
    by_target = {d.target: d for d in dossiers}
    for claim in [tuple(k) for k in state["observed"]]:
        by_target[claim[0]].observe(pool[claim], state["decision_round"])
    available = set(emergency._retrievable(pool, {tuple(k) for k in state["observed"]},
                                           state["decision_round"]))
    saved = twins.graded_moves
    try:
        twins.graded_moves = refuse
        built = twins._state_for_policy(dossiers, available, state["decision_round"], 8, 8,
                                        list(emergency.ACTIONS))
    except AssertionError as caught:
        return False, str(caught)
    finally:
        twins.graded_moves = saved
    # A field the evaluator computes can also arrive by name rather than by call, so the
    # built state is additionally checked against the oracle's own vocabulary.
    leaked = sorted(EVALUATOR_VOCABULARY & set(_every_key(built)))
    return (not leaked), f"state carries evaluator fields {leaked}" if leaked else "clean"


# Verdicts under which the probe still separates policies, so a candidate carrying one has
# not destroyed the readout and the search may proceed.
#
# `led_by_a_trivial_policy` is here because the committed instrument now carries it, and the
# distinction it forces is the reason this list is explicit rather than a single equality.
# The guard exists to refuse a *change* that destroys discrimination; it is not a certificate
# that the readout is sound. Completing the trivial set found that `coverage_static` -- which
# ranks targets by how many axes the packet declares and reads nothing that moves -- outscores
# every policy that reads the state. That is a defect in the packets, it predates every
# candidate here, and holding the whole search red for it would mean no candidate could ever
# be evaluated against a baseline that is itself refused. What the counting exploits is open:
# it is not deposited-structure count, which is uncorrelated with declared axis count across
# the eleven targets at Spearman 0.060, p = 0.86.
#
# It is recorded rather than forgiven. `audit_axis_redundancy` names the two axes that cannot
# gate, the ledger carries `trivial_outscores_reference`, and both are pinned by their own
# tests. What must not be admitted is a candidate that takes the readout from separating
# policies to not separating them at all, and the three planted adversaries that trip this
# guard all do exactly that -- verified, not assumed, by the test that runs them.
SEPARATING_VERDICTS = ("discriminating", "led_by_a_trivial_policy")


def guard_readout_still_discriminates(packet="coronaviridae", attempts=250):
    """The monitor's verdict, the positive control, and the yield floor together.

    All three, because each alone admits a different hack. The verdict alone admits a probe
    narrowed to a handful of states; the yield alone admits a probe nobody can answer; the
    positive control alone admits a probe a fixed traversal maxes out.
    """
    rows = audit_twin_readout([packet], attempts=attempts)[packet]
    reasons = []
    if rows["verdict"] not in SEPARATING_VERDICTS:
        reasons.append(f"verdict {rows['verdict']}")
    if not rows["positive_control_attained"]:
        reasons.append("the leader stratum is no longer attainable, so the probe is broken "
                       "rather than hard")
    counts = rows["state_yield"]
    yielded = counts["challenger_best"]
    if yielded < MINIMUM_CHALLENGER_STATES:
        reasons.append(f"only {yielded} challenger states, below the floor of "
                       f"{MINIMUM_CHALLENGER_STATES}")
    # Headroom can be raised by making the states uninformative rather than the readout
    # sharper, and a band narrow enough to do that while keeping the yield above the floor and
    # the positive control intact escaped every other guard. This is the documented edge of
    # the band, and it is where that candidate shows up: a third of its draws land on states
    # no purchase moves.
    dead = counts["no_purchase_moves_the_decision"] + counts["every_purchase_grades_alike"]
    share = dead / counts["attempts"] if counts["attempts"] else 0.0
    if share > MAXIMUM_UNINFORMATIVE_SHARE:
        reasons.append(f"{share:.1%} of sampled states have no purchase that moves the "
                       f"decision, above the ceiling of "
                       f"{MAXIMUM_UNINFORMATIVE_SHARE:.0%}; the sampler has left the "
                       f"discriminating region")
    return (not reasons), "; ".join(reasons) or "discriminating"


def guard_detector_is_calibrated(packet="coronaviridae", attempts=200):
    """A name-blind policy's identity effect must stay exactly zero."""
    effect = run_orbit(packet, "rubric", count=2,
                       attempts=attempts)["earned"]["challenger_best"]
    value = effect["attributable_to_identity"]
    return value == 0.0, f"name-blind identity effect {value}"


def guard_detector_is_sensitive(packet="coronaviridae", attempts=200):
    """The planted memoriser's identity effect must stay large.

    Calibration without sensitivity is a detector that reports nothing and passes every
    check. Both halves are required or neither means anything.
    """
    effect = run_orbit(packet, "memorised", count=2,
                       attempts=attempts)["earned"]["challenger_best"]
    value = effect["attributable_to_identity"]
    return (value is not None and value >= MINIMUM_IDENTITY_EFFECT,
            f"planted memoriser identity effect {value}")


def guard_invariants_hold():
    """Run the suite in-process, so the candidate's patches are in force while it runs.

    A subprocess would not see the patches and the check would pass for every candidate,
    which is worse than not having it: it would read as a guard and enforce nothing.
    """
    import pytest

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = pytest.main(["-x", "-q", "--no-header", "-p", "no:cacheprovider",
                            str(ROOT / "hpc_pilots/tests/test_pilots.py")])
    tail = [line for line in buffer.getvalue().splitlines() if line.strip()][-3:]
    return int(code) == 0, " | ".join(tail)


def guard_sample_is_unbiased(packet="coronaviridae", attempts=250):
    """Is the scored sample distributionally like the pool it was drawn from?

    The subtlest hack an independent adversary found lived here. `pair` discards about four
    fifths of the challenger states, and replacing its random draw with an ordering on each
    state's own margin raised headroom from 0.66 to 0.79 while the yield, the pair count, the
    probe count, the optimal rate and the positive control all matched the baseline exactly.
    No readout moved, so no readout could catch it.
    """
    result = run_twins(packet, "rubric", attempts=attempts)
    audit = result["sample_audit"]
    reasons = []
    for family, row in audit.items():
        if not row["scored_states"] or row["difference"] is None:
            continue
        if abs(row["difference"]) > MAXIMUM_SAMPLE_BIAS:
            reasons.append(f"{family} scored margin is {row['difference']:+.4f} from the pool")
    scored = sum(row["scored_states"] for row in audit.values())
    if scored < MINIMUM_SCORED_PROBES:
        reasons.append(f"only {scored} states scored, below the floor of "
                       f"{MINIMUM_SCORED_PROBES}; thinning the sample widens its noise and "
                       f"the search admits whichever way that noise falls")
    return (not reasons), "; ".join(reasons) or "representative"


def guard_summary_reproduces_from_its_probes(packet="coronaviridae", attempts=250):
    """Recompute the reported means from the probe rows the artifact carries.

    Three of the adversary's hacks lived in this gap -- a one-sided trimmed mean that dropped
    the reference policy's successes only, a refusal scored as zero rather than excluded, and
    a thinned sample -- and each moved the headline while `purchases` and `declined_to_buy`
    kept printing their old values. The file carried everything needed to catch all three.
    """
    from .twins import FAMILIES

    result = run_twins(packet, "rubric", attempts=attempts)
    reasons = []
    # Iterated over the declared strata rather than over every key in the summary. Skipping
    # one known extra key by name meant the guard broke the moment the summary grew another:
    # adding the scarcity curves put a `by_scarcity` block beside the strata and the guard
    # read it as one, which is a guard failing on a change that was not a hack.
    for family in FAMILIES:
        row = result["summary"][family]
        rows = [p for p in result["probes"] if p["state_family"] == family]
        scored = [p["attainment"] for p in rows if p["attainment"] is not None]
        if row["probes"] != len(rows):
            reasons.append(f"{family} reports {row['probes']} probes, file holds {len(rows)}")
        if row["purchases"] != len(scored):
            reasons.append(f"{family} reports {row['purchases']} purchases, "
                           f"{len(scored)} are scored")
        # Three categories now, not two: a purchase scored for attainment, a refusal to buy,
        # and an illegal action -- an unofferable pair, which is an invalid action rather than
        # a bad purchase and is counted apart from both.
        if row["purchases"] + row["declined_to_buy"] + row["illegal_actions"] != row["probes"]:
            reasons.append(f"{family} counts do not add up")
        expected = None if not scored else round(sum(scored) / len(scored), 4)
        if row["attainment"] is not None and expected is not None \
                and abs(row["attainment"] - expected) > 1e-6:
            reasons.append(f"{family} reports {row['attainment']}, its probes give {expected}")
    return (not reasons), "; ".join(reasons) or "reproduces"


GUARDS = {
    "state_is_oracle_free": guard_state_is_oracle_free,
    "sample_is_unbiased": guard_sample_is_unbiased,
    "summary_reproduces": guard_summary_reproduces_from_its_probes,
    "readout_still_discriminates": guard_readout_still_discriminates,
    "detector_is_calibrated": guard_detector_is_calibrated,
    "detector_is_sensitive": guard_detector_is_sensitive,
}


def objective(packet="coronaviridae", attempts=250):
    """Headroom on the hard stratum: the room a real policy has to be observed in.

    Read on the identity orbit rather than the nominal run, so a candidate making the answer
    more recallable registers no gain. That is the whole reason the loop is safe to point at
    a score at all: the quantity it maximises is already decontaminated.
    """
    orbit = run_orbit(packet, "rubric", count=2, attempts=attempts)
    earned = orbit["earned"]["challenger_best"]["earned_mean"]
    nominal = orbit["earned"]["challenger_best"]["nominal"]
    return {"challenger_headroom": None if earned is None else round(1.0 - earned, 4),
            "nominal_headroom": None if nominal is None else round(1.0 - nominal, 4),
            "trivial_attainment_earned": earned}


def evaluate(candidate, packet="coronaviridae", attempts=250, run_tests=True):
    """Run every guard, then the objective, and report both whatever happens.

    The objective is computed even for a candidate the guard rejected. A rejected candidate's
    score is the thing worth seeing: it is how large a gain the guard refused, and a guard
    that only ever refuses candidates that were not winning anyway has not been tested.
    """
    checks, failures = {}, []
    with applied(candidate):
        for name, guard in GUARDS.items():
            try:
                passed, detail = guard(packet, attempts) if name != "state_is_oracle_free" \
                    else guard(packet)
            except Exception as error:
                passed, detail = False, f"{type(error).__name__}: {error}"
            checks[name] = {"passed": bool(passed), "detail": detail}
            if not passed:
                failures.append(name)
        # Run last and only for a candidate the other guards accepted. The suite costs about
        # fifty seconds; a candidate already rejected does not need it, and skipping it there
        # is recorded rather than left to be inferred from a missing key.
        if run_tests and not failures:
            passed, detail = guard_invariants_hold()
            checks["invariants_hold"] = {"passed": passed, "detail": detail}
            if not passed:
                failures.append("invariants_hold")
        elif run_tests:
            checks["invariants_hold"] = {"passed": None,
                                         "detail": "not run; an earlier guard already "
                                                   "rejected this candidate"}
        try:
            scores = objective(packet, attempts)
        except Exception as error:
            scores = {"challenger_headroom": None,
                      "error": f"{type(error).__name__}: {error}"}

    return {"candidate": candidate.name, "description": candidate.description,
            "planted": candidate.planted, "predicted_guard": candidate.trips,
            "guards": checks, "failed_guards": failures,
            "admitted": not failures, "objective": scores}


def run_improve(packet="coronaviridae", attempts=250, run_tests=True, smoke=False):
    """Evaluate every candidate behind the guard and report what was admitted and what caught what.

    The headline is not the winning candidate. It is whether each planted hack was caught, and
    by the guard predicted for it. A hack caught by a different guard than expected means the
    prediction was wrong and the guard set is not understood as well as the docstring claims;
    a hack not caught at all means the guard set has a hole, and that is the finding.
    """
    if smoke:
        attempts, run_tests = 120, False
    results = [evaluate(c, packet, attempts, run_tests) for c in CANDIDATES]
    baseline = next(r for r in results if r["candidate"] == "baseline")
    base = baseline["objective"].get(OBJECTIVE)

    planted = [r for r in results if r["planted"]]
    caught = [r for r in planted if not r["admitted"]]
    mispredicted = [{"candidate": r["candidate"], "predicted": r["predicted_guard"],
                     "actually_failed": r["failed_guards"]}
                    for r in caught if r["predicted_guard"] not in r["failed_guards"]]
    escaped = [r["candidate"] for r in planted if r["admitted"]]

    honest = [r for r in results if not r["planted"] and r["admitted"]
              and r["objective"].get(OBJECTIVE) is not None]
    best = max(honest, key=lambda r: r["objective"][OBJECTIVE], default=None)
    # A hack scoring above the baseline is what the guard is for. Reporting the gain it was
    # refused is the only way to see whether the guard ever had to do any work.
    refused_gain = [{"candidate": r["candidate"],
                     "objective": r["objective"].get(OBJECTIVE),
                     "gain_over_baseline": None if r["objective"].get(OBJECTIVE) is None
                     or base is None else round(r["objective"][OBJECTIVE] - base, 4),
                     "failed_guards": r["failed_guards"]}
                    for r in results if not r["admitted"]]

    out = {"experiment": "improve", "status": "complete", "smoke": smoke,
           "family_packet": packet, "objective": OBJECTIVE,
           "objective_note": "Headroom on the hard stratum, read on the identity orbit rather "
                             "than the nominal run, so a candidate that makes the answer more "
                             "recallable registers no gain.",
           "baseline_objective": base,
           "planted_hacks": len(planted), "planted_hacks_caught": len(caught),
           "planted_hacks_escaped": escaped,
           "guard_predictions_wrong": mispredicted,
           "refused_candidates": refused_gain,
           "best_admitted": None if best is None else
           {"candidate": best["candidate"], "objective": best["objective"][OBJECTIVE],
            "gain_over_baseline": None if base is None
            else round(best["objective"][OBJECTIVE] - base, 4)},
           "results": results, "provenance": provenance(),
           "interpretation": "Searching over changes to a measurement is the same operation "
                             "as corrupting it, so the guard is the contribution and the "
                             "search is not. A planted hack that escapes, or one caught by a "
                             "guard other than the one predicted for it, is a finding about "
                             "the guard."}
    output = ROOT / "runs/improve" / f"{'smoke_' if smoke else ''}{packet}.json"
    write_json(output, out)
    print(json.dumps({k: out[k] for k in
                      ("baseline_objective", "planted_hacks", "planted_hacks_caught",
                       "planted_hacks_escaped", "guard_predictions_wrong",
                       "refused_candidates", "best_admitted")}, indent=2))
    return out
