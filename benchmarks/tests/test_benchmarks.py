"""Tests for both benchmarks, all offline.

Task 1's harness reaches NCBI and UniProt for its candidate sets, so nothing here fetches one.
The scoring path is tested instead against a synthetic candidate set built to put the labelled
targets at the ranks `runs/target_ranks.csv` already recorded -- which checks the thing worth
checking: that this harness and the existing experiment compute the same number from the same
ranks. A harness that quietly disagreed with the recorded result would make every comparison in
the paper unverifiable.
"""

from __future__ import annotations

import csv
import math
import os

import pytest

from benchmarks import task1, task1_labels, task2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Task 2: the environment and the oracle
# ---------------------------------------------------------------------------

def test_every_requirement_is_a_tool_name():
    from hpc_pilots import selector
    assert set(task2.TOOL_NAMES) == set(selector.REQUIREMENTS)


def test_gateable_axes_are_the_ones_some_tool_gates():
    from hpc_pilots import selector
    declared = {axis for r in selector.REQUIREMENTS.values() for axis in r.gates}
    assert set(task2.GATEABLE) == declared
    # The benchmark must not score policies against axes nothing can reach, or every arm fails
    # identically and the metric measures the registry rather than the policy.
    assert set(task2.GATEABLE) < set(selector.AXES)


@pytest.mark.parametrize("goal", task2.GOALS)
@pytest.mark.parametrize("seed", sorted(task2.SEEDS))
def test_optimal_moves_are_always_legal(goal, seed):
    dist, seen = task2.distances(goal, task2.State(facts=task2.SEEDS[seed]))
    for state in seen.values():
        for move in task2.optimal_moves(goal, state, dist):
            assert task2.legal(move, state), f"{move} proposed as optimal where it is blocked"


@pytest.mark.parametrize("goal", task2.GOALS)
@pytest.mark.parametrize("seed", sorted(task2.SEEDS))
def test_oracle_achieves_the_optimum(goal, seed):
    """The oracle defines the metric, so an oracle that misses it invalidates every other row."""
    run = task2.episode(task2.Oracle(), goal, seed)
    if run.optimal == math.inf:
        assert not run.reached
        return
    assert run.reached
    assert len(run.calls) == run.optimal
    assert not run.illegal and not run.repeats


@pytest.mark.parametrize("goal", task2.GOALS)
@pytest.mark.parametrize("seed", sorted(task2.SEEDS))
def test_no_arm_beats_the_oracle(goal, seed):
    for arm in task2.arms():
        run = task2.episode(arm, goal, seed)
        if run.reached and run.optimal != math.inf:
            assert len(run.calls) >= run.optimal, f"{arm.name} beat the shortest path"


def test_every_arm_honours_the_mask():
    """A masked tool must never be proposed, or the no-repeat rule is unevenly applied.

    This is the test the first version of this benchmark lacked, and its absence produced a
    wrong ranking: the similarity arms kept their own no-repeat bookkeeping while the repo's
    selector was called statelessly, so the selector repeated one tool to the budget and placed
    last. The asymmetry was in the harness, not in the policy.
    """
    state = task2.State(facts=task2.SEEDS["onset"])
    for arm in task2.arms():
        arm.reset("call_all", task2.SEEDS["onset"])
        mask = frozenset(task2.TOOL_NAMES) - {"literature.search"}
        pick = arm.choose(state, "call_all", mask)
        assert pick in (None, "literature.search"), f"{arm.name} ignored the mask and said {pick}"


def test_unreachable_from_a_bare_sequence():
    """The registry's real gap, pinned so a later change has to notice it.

    No registered tool turns an amino-acid sequence into a reviewed protein accession, so from a
    protein nothing has catalogued the accession-keyed lanes are permanently shut. That is the
    case this project is named for. If a tool is ever added that provides `protein_accession`
    from a sequence, this test fails and should be deleted with a note saying so.
    """
    report = {(r["goal"], r["seed"]): r for r in task2.reachability()}
    assert report[("unlock_all", "novel_sequence")]["reachable"] is False
    never = report[("unlock_all", "novel_sequence")]["tools_never_callable"]
    assert "geometry.surface_markers" in never
    assert "structures.folded" not in never  # folding is the one lane a novel sequence keeps


def test_the_ordering_rules_now_cut_both_ways():
    """The benchmark's main result, and it reverses sign between goals.

    **This test replaces one that pinned the opposite.** Until the structural chain was
    registered, `gap_ordering` was 0.000 on every instance: every tool took an identifier,
    identification produced every identifier, so after one call everything was callable and no
    ordering remained to get right. A lexical retriever handed the legality filter tied an
    optimal planner everywhere, and the honest reading was that the precondition model carried
    the whole result while the ranking rules rode along unmeasured.

    `structures.picks` -> `geometry.markers_for_chain` -> `geometry.macrocycle_score` gives the
    graph a lane three deep, and the rules become measurable in both directions at once:

    * On `unlock_all`, where the goal is to widen the action set, the selector's
      unblocking-first rule is worth about +0.7 in optimal-move rate over the same retriever.
      That is the claim `selector.rank` makes for `score[1]`, measured for the first time.
    * On `gate_all`, where the goal is axis coverage, the identical rule *costs* it roughly
      half. More unlocking calls exist to be distracted by, and every one of them is a detour
      from the two calls that actually gate the axes.

    So the rule is not right or wrong, it is a bet on option value, and this is its price. The
    docstring of `selector.rank` already names `score[1]` as "the assumption most likely to be
    wrong"; what it lacked was a number. Pinned loosely, because the exact values move with the
    registry and the sign is the finding.
    """
    rows = {(r["goal"], r["seed"]): r for r in task2.run_states()["discrimination"]}
    assert rows, "no instances scored"

    unlock = rows[("unlock_all", "onset")]
    assert unlock["selector"] == 1.0, "the selector should still play unlock_all perfectly"
    assert unlock["gap_ordering"] > 0.5, (
        f"unblocking-first should beat lexical retrieval on unlock_all; "
        f"gap_ordering={unlock['gap_ordering']}, similarity_legal={unlock['similarity_legal']}")

    for seed in ("onset", "known_protein"):
        gate = rows[("gate_all", seed)]
        assert gate["selector"] < gate["similarity_legal"], (
            f"gate_all/{seed}: the cost of preferring option value over axis coverage should "
            f"show up here; selector={gate['selector']}, "
            f"similarity_legal={gate['similarity_legal']}")

    # `call_all` still measures legality and not ordering: once no-repeat and legality are
    # given, every legal order is optimal, so a uniform choice among legal calls scores 1.000.
    # Kept as a reported instance and never quoted as a ranking result.
    assert rows[("call_all", "onset")]["random_legal"] == 1.0
    assert rows[("call_all", "onset")]["saturated"] is True

    # Legality remains worth more than ordering on three of the four instances.
    assert sum(1 for r in rows.values() if (r["gap_legality"] or 0) > 0.5) >= 3


def test_the_selector_pays_three_calls_for_preferring_option_value():
    """What the documented judgement costs, as a sequence rather than a rate.

    From a known protein with both axes open, the shortest path is two calls: measure the
    surface, look up the structural precedent. The selector instead spends three calls on
    unblocking first -- `families.entries`, then `structures.picks`, then
    `geometry.markers_for_chain` -- and only then makes the two that gate the axes. Five calls
    where two would do.

    **The price grew with the graph, which is the point.** On the flat registry this cost
    exactly one call, because only one unblocking move existed to be distracted by. Adding a
    three-deep structural lane added two more, and the selector took both. So `score[1]` is not
    a fixed overhead -- it scales with how much of the graph is reachable, which is precisely
    when option value is supposed to be worth paying for and precisely when it is not.
    """
    run = task2.episode(task2.RepoSelector(), "gate_all", "known_protein")
    assert run.reached
    assert run.optimal == 2
    assert len(run.calls) == 5
    # The order is the explanation: every unblocking call comes before either gating call.
    assert run.calls[:3] == ["families.entries", "structures.picks",
                             "geometry.markers_for_chain"]
    assert set(run.calls[3:]) == {"geometry.surface_markers", "structures.evidence"}
    # And all three detours were legal, so this is a ranking choice and not a precondition bug.
    assert not run.illegal
    assert run.suboptimal == ["families.entries", "structures.picks",
                              "geometry.markers_for_chain"]


# ---------------------------------------------------------------------------
# Task 1: labels and scoring
# ---------------------------------------------------------------------------

def test_verified_labels_match_the_recorded_experiment():
    """The eleven verified rows are the ones in `runs/target_ranks.csv`, name for name."""
    path = os.path.join(ROOT, "runs", "target_ranks.csv")
    with open(path) as handle:
        recorded = {(r["pathogen"], r["target_protein"]) for r in csv.DictReader(handle)}
    declared = {(e["pathogen"], t["name"])
                for e in task1_labels.pathogens(include_draft=False)
                for t in e["targets"]}
    assert declared == recorded


def test_draft_labels_are_excluded_by_default():
    """A drafted label must never reach a headline number without being asked for."""
    default = task1_labels.pathogens()
    assert all(t.get("verified") for e in default for t in e["targets"])
    widened = task1_labels.pathogens(include_draft=True)
    assert sum(len(e["targets"]) for e in widened) > sum(len(e["targets"]) for e in default)
    # And every draft row has to say what would promote it, so the check is never a guess.
    assert all(t.get("source") for e in widened for t in e["targets"] if not t.get("verified"))


def test_every_label_has_a_status_the_weights_know():
    for entry in task1_labels.PATHOGENS:
        for target in entry["targets"]:
            assert target["status"] in task1_labels.STATUS_WEIGHT, target


def test_matcher_is_exact_and_refuses_a_substring():
    """"f protein" sits inside "transmembrane protein" -- a fuzzy matcher credits the wrong chain.

    Pinned because the failure is silent and flattering: a loose matcher finds every label
    somewhere in a candidate list and inflates every rank in the benchmark.
    """
    label = {"name": "F fusion glycoprotein", "aliases": ("f protein", "fusion protein")}
    assert task1.matches(label, {"name": "F protein"})
    assert task1.matches(label, {"name": "Fusion protein"})
    assert not task1.matches(label, {"name": "Transmembrane protein gp41"})
    assert not task1.matches(label, {"name": "Nucleoprotein"})


def _synthetic(ranks, total, label_names):
    """A candidate set with the labelled targets planted at the given ranks."""
    at = dict(zip(ranks, label_names))
    return {"candidates": [{"rank": i, "name": at.get(i, f"filler protein {i}"),
                            "parent": None, "length_aa": 100}
                           for i in range(1, total + 1)]}


@pytest.mark.parametrize("pathogen,ranks,total,expected_p", [
    ("SARS-CoV-2", [1, 2, 5, 6, 7], 23, 8.62e-04),
    ("Nipah virus", [1, 3, 4], 8, 7.14e-02),
    ("Influenza A PR8", [1, 2, 4], 11, 1.21e-02),
])
def test_scoring_reproduces_the_recorded_p_values(pathogen, ranks, total, expected_p):
    """This harness and the recorded experiment must agree on the same ranks.

    The ranks and the expected p-values are read off `runs/target_ranks.csv`. The point is not
    that `rank_enrichment` works -- it has its own tests -- but that this benchmark feeds it the
    same ranks and the same denominator, so a number produced here is comparable with one
    already published in the repository.
    """
    entry = next(e for e in task1_labels.pathogens() if e["pathogen"] == pathogen)
    names = [t["name"] for t in entry["targets"]]
    assert len(names) == len(ranks), "label count drifted from the recorded experiment"
    payload = _synthetic(ranks, total, names)

    row = task1.score_pathogen(task1.Pipeline(), entry, payload)
    assert row["matched"] == len(ranks)
    assert row["ranks"] == sorted(ranks)
    assert row["p_vs_random"] == pytest.approx(expected_p, rel=1e-2)


def test_underpowered_is_not_reported_as_a_failure():
    """Nipah's p of 0.071 must read as underpowered, never as "does not beat chance"."""
    entry = next(e for e in task1_labels.pathogens() if e["pathogen"] == "Nipah virus")
    names = [t["name"] for t in entry["targets"]]
    row = task1.score_pathogen(task1.Pipeline(), entry, _synthetic([1, 3, 4], 8, names))
    assert row["verdict"] == "not established (underpowered)"
    assert row["adequately_powered"] is False


def test_an_unmatched_label_is_reported_not_dropped():
    """A label the candidate stage never proposed is a miss on the record, not a silent absence."""
    entry = next(e for e in task1_labels.pathogens() if e["pathogen"] == "Nipah virus")
    payload = _synthetic([1], 8, [entry["targets"][0]["name"]])
    row = task1.score_pathogen(task1.Pipeline(), entry, payload)
    assert row["matched"] == 1
    assert len(row["unmatched"]) == len(entry["targets"]) - 1
    assert all(t["rank"] is None for t in row["targets"] if t["target"] in row["unmatched"])


def test_blinded_prompt_names_neither_virus_nor_protein():
    """The control the LLM arms rest on. If the pathogen leaks, the blinded arm is not blinded."""
    entry = next(e for e in task1_labels.pathogens() if e["pathogen"] == "SARS-CoV-2")
    payload = _synthetic([1, 2, 5, 6, 7], 23, [t["name"] for t in entry["targets"]])
    for row in payload["candidates"]:
        row["sequence"] = "MFVFLVLLPLVSSQ"
    text = task1.prompt(entry, payload["candidates"], blinded=True)
    assert "SARS" not in text and "NC_045512" not in text
    assert "Spike" not in text and "nsp5" not in text
    named = task1.prompt(entry, payload["candidates"], blinded=False)
    assert "SARS-CoV-2" in named and "Spike glycoprotein" in named


def test_stouffer_combines_and_reports_what_it_dropped():
    assert task1.stouffer([0.5, 0.5, 0.5])["p"] == pytest.approx(0.5, abs=1e-3)
    small = task1.stouffer([0.01, 0.01, 0.01])["p"]
    assert small < 0.01, "combining three independent small p-values must strengthen them"
    dropped = task1.stouffer([0.05, None, 1.0])
    assert dropped["k"] == 1 and dropped["dropped"] == 2
    assert task1.stouffer([])["p"] is None
