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

from . import emergency, monitor, twins
from .common import ROOT, provenance, write_json
from .invariance import run_orbit
from .monitor import audit_twin_readout
from .twins import run_twins

OBJECTIVE = "challenger_headroom"

# How much the guard requires. The yield floor is not a style choice: a candidate narrowing
# the sampler until two states survive would show a large headroom computed from almost
# nothing, and a floor is the only thing that distinguishes a harder probe from a smaller one.
MINIMUM_CHALLENGER_STATES = 25
MINIMUM_IDENTITY_EFFECT = 0.4


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
)


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
    forbidden = {"grade", "grades", "graded_moves", "optimal", "relevant", "best_grade",
                 "chance_grade", "attainment", "decision_relevant"}
    leaked = sorted(forbidden & set(built))
    return (not leaked), f"state carries evaluator fields {leaked}" if leaked else "clean"


def guard_readout_still_discriminates(packet="coronaviridae", attempts=250):
    """The monitor's verdict, the positive control, and the yield floor together.

    All three, because each alone admits a different hack. The verdict alone admits a probe
    narrowed to a handful of states; the yield alone admits a probe nobody can answer; the
    positive control alone admits a probe a fixed traversal maxes out.
    """
    rows = audit_twin_readout([packet], attempts=attempts)[packet]
    reasons = []
    if rows["verdict"] != "discriminating":
        reasons.append(f"verdict {rows['verdict']}")
    if not rows["positive_control_attained"]:
        reasons.append("the leader stratum is no longer attainable, so the probe is broken "
                       "rather than hard")
    yielded = rows["state_yield"]["challenger_best"]
    if yielded < MINIMUM_CHALLENGER_STATES:
        reasons.append(f"only {yielded} challenger states, below the floor of "
                       f"{MINIMUM_CHALLENGER_STATES}")
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


GUARDS = {
    "state_is_oracle_free": guard_state_is_oracle_free,
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
