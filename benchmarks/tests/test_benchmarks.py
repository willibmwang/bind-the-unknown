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
# Task 1: labels, matching, and scoring the panel
# ---------------------------------------------------------------------------

def test_every_panel_pattern_is_graded_or_dropped():
    """No panel row may default to a status, because a default is a judgement nobody made.

    `data/benchmark_panel.json` is curated elsewhere and may gain a row. If an ungraded row fell
    through to some default, the row least likely to deserve a status would silently receive one
    -- and it would be a *new* row, which is exactly the one nobody has read yet. So the loader
    raises, and this pins that the overlay currently covers the panel exactly.
    """
    panel = task1_labels._panel()
    patterns = {(e["pathogen"], p) for e in panel for p in e["targets"]}
    covered = set(task1_labels.OVERLAY) | set(task1_labels.DROPPED)
    assert patterns - covered == set(), f"ungraded panel rows: {sorted(patterns - covered)}"
    # And nothing in the overlay names a row the panel does not have, or the two have drifted.
    assert covered - patterns == set(), f"overlay names absent rows: {sorted(covered - patterns)}"
    # 157 since the panel gained its `Influenza A PR8` row -- three patterns on one genome. The
    # pin moved with the curation rather than being relaxed: this assertion is the only thing
    # that would have told anyone the panel had grown, and for a while it was the only thing
    # that did, because `pathogens()` raised on the ungraded rows long before a reader got here.
    assert len(patterns) == 157


def test_an_ungraded_pattern_raises_rather_than_defaulting():
    panel = [{"pathogen": "Testvirus", "accession": "NC_000000.1", "holdout": "Testvirus",
              "targets": ["Some protein nobody graded"], "validated_by": "n/a"}]
    with pytest.raises(ValueError, match="no status"):
        task1_labels.pathogens(tier="all", panel=panel)


def test_the_strong_tier_is_what_a_headline_may_quote():
    """Only an approved drug binding this protein in this virus reaches the default tier."""
    counts = task1_labels.counts()
    # 47 genomes and 27 strong labels since the panel's `Influenza A PR8` row was graded. The
    # pathogen count is unchanged at 14: PR8 was already reaching the strong tier through
    # `EXTRA`, so what the grading added is one more approved label (M2) on a pathogen that was
    # already there, not a new pathogen.
    assert counts["panel_genomes"] == 47
    assert counts["strong_labels"] == 27
    assert counts["strong_pathogens"] == 14
    strong = task1_labels.pathogens(tier="strong")
    assert all(t["status"] in task1_labels.STRONG for e in strong for t in e["targets"])
    # The weak tier is far larger, which is the fact that makes the split matter: scoring the
    # panel as one set would let a class effect carry an approved drug's claim.
    weak = sum(v for k, v in counts["graded_by_status"].items()
               if k not in task1_labels.STRONG)
    assert weak > 3 * counts["strong_labels"] - len(task1_labels.EXTRA[0]["targets"])


def test_every_graded_status_is_one_the_weights_know():
    for key, (status, why) in task1_labels.OVERLAY.items():
        assert status in task1_labels.STATUS_WEIGHT, (key, status)
        assert why, f"{key} has no justification"
    for entry in task1_labels.EXTRA:
        for target in entry["targets"]:
            assert target["status"] in task1_labels.STATUS_WEIGHT


def test_mislabels_are_dropped_and_not_merely_downgraded():
    """The distinction grading cannot make, so it has to be made here.

    A class effect is real evidence of the wrong kind and `status` handles it. A row that is
    simply wrong scores at whatever tier it is given and keeps scoring, so it has to leave the
    set -- and leave a record, or a later re-key reinstates it.
    """
    kinds = {k: v[0] for k, v in task1_labels.DROPPED.items()}
    assert kinds[("Herpes simplex virus 1", "terminase")] == "mislabel"
    assert kinds[("Hepatitis B virus", "surface antigen")] == "mislabel"
    assert kinds[("Hepatitis B virus", "Large envelope")] == "mislabel"
    # Every drop carries a reason, and none of them is in the graded set.
    assert all(reason for _, reason in task1_labels.DROPPED.values())
    assert not set(task1_labels.DROPPED) & set(task1_labels.OVERLAY)
    # A duplicate is not a mislabel: HBV's bare "polymerase" names the same protein P that
    # "Reverse transcriptase" names, and would also match any RdRp in any virus.
    assert kinds[("Hepatitis B virus", "polymerase")] == "duplicate"
    assert kinds[("HIV-1", "Gag-Pol")] == "parent"


def test_a_label_on_a_segment_the_genome_does_not_carry_is_dropped_as_absent():
    """The ranker must not answer for the panel's accession choice.

    Three segmented viruses are supplied as one segment, so some labelled proteins are not in
    the genome handed to the ranker at all. Verified against the records: Hantaan's NC_005218.1
    yields only a nucleoprotein, and CCHF's NC_005300.2 yields only polymerase-complex
    products. Counting those labels as misses would mark the ranking wrong for a gene that was
    never in its input.
    """
    kinds = {k: v[0] for k, v in task1_labels.DROPPED.items()}
    assert kinds[("Hantaan virus", "RNA-directed RNA polymerase")] == "absent_from_genome"
    assert kinds[("Hantaan virus", "Glycoprotein")] == "absent_from_genome"
    assert kinds[("Lassa virus", "RNA-directed RNA polymerase")] == "absent_from_genome"
    assert set(task1_labels.SEGMENTED) >= {"Hantaan virus", "Lassa virus",
                                           "Crimean-Congo hemorrhagic fever virus"}


def test_an_instance_that_cannot_discriminate_is_excluded_rather_than_scored():
    """Zero candidates has no score; one candidate against one label is a guaranteed hit."""
    scored = {e["pathogen"] for e in task1_labels.pathogens(tier="all")}
    assert "Human parvovirus B19" not in scored
    assert "Hantaan virus" not in scored
    assert set(task1_labels.UNSCOREABLE) == {"Human parvovirus B19", "Hantaan virus"}
    assert all(task1_labels.UNSCOREABLE.values())
    widened = {e["pathogen"] for e in task1_labels.pathogens(tier="all",
                                                             include_unscoreable=True)}
    assert "Human parvovirus B19" in widened


# ---------------------------------------------------------------------------
# Matching

def test_token_run_matching_refuses_the_substring_traps():
    """Both traps, in both directions, because loosening the matcher only ever inflates.

    `NS5` inside `NS5A` is the one substring matching gets wrong in the flattering direction: a
    bare NS5 label would claim HCV's NS5A and NS5B as two separate hits. `f protein` inside
    `transmembrane protein gp41` is the trap that retired the first version.
    """
    ns5 = {"name": "NS5", "aliases": ("ns5",)}
    assert not task1.matches(ns5, {"name": "NS5A"})
    assert not task1.matches(ns5, {"name": "NS5B"})
    assert task1.matches(ns5, {"name": "Serine protease NS5"})
    assert task1.matches({"name": "NS5A", "aliases": ("ns5a",)}, {"name": "NS5A"})

    fusion = {"name": "F fusion glycoprotein", "aliases": ("f protein", "fusion protein")}
    assert task1.matches(fusion, {"name": "F protein"})
    assert not task1.matches(fusion, {"name": "Transmembrane protein gp41"})
    assert not task1.matches(fusion, {"name": "Nucleoprotein"})

    # HBV's bare "polymerase" is why that pattern is dropped rather than aliased: as a token run
    # it matches every RdRp in the panel.
    bare = {"name": "polymerase", "aliases": ("polymerase",)}
    assert task1.matches(bare, {"name": "RNA-directed RNA polymerase nsp12"})


def test_matching_survives_the_affixes_the_ranker_actually_emits():
    """Strict equality missed nearly every real chain, which is why the rule is a token run."""
    aliases = task1_labels.aliases_for
    assert task1.matches({"name": "3C-like proteinase",
                          "aliases": aliases("SARS-CoV-2", "3C-like proteinase")},
                         {"name": "3C-like proteinase nsp5"})
    assert task1.matches({"name": "Nucleoprotein",
                          "aliases": aliases("Rabies virus", "Nucleoprotein")},
                         {"name": "NCAP_SEOUS Nucleoprotein"})
    assert task1.matches({"name": "DNA polymerase",
                          "aliases": aliases("Human cytomegalovirus", "DNA polymerase")},
                         {"name": "DPOL_RHCM6 DNA polymerase catalytic subunit"})


def test_a_per_pathogen_alias_is_scoped_to_that_pathogen():
    """`gp120` must not become a name every bare `Envelope` pattern can claim."""
    assert "gp120" in task1_labels.aliases_for("HIV-1", "Envelope")
    assert "gp120" not in task1_labels.aliases_for("Zika virus", "Envelope")


# ---------------------------------------------------------------------------
# The null, and scoring the panel

@pytest.mark.parametrize("k,expected", [(1, 25.0), (3, 60.1), (5, 80.6), (10, 98.4)])
def test_the_hypergeometric_null_is_high_enough_to_change_the_reading(k, expected):
    """The number without which a recall@k is unreadable.

    With 20 candidates and 5 matching chains, random ranking already scores 80.6% at k=5. A
    harness reporting 100% there is reporting about nineteen points, not a hundred. These four
    values were recomputed independently by the panel's author before being accepted.
    """
    assert task1.hypergeometric_at_k(20, 5, k) == pytest.approx(expected, abs=0.1)


def test_the_null_is_computed_per_genome_and_not_quoted_once():
    """Candidate sets run from 0 to 22, so one headline null would be wrong for nearly all."""
    assert task1.hypergeometric_at_k(4, 1, 5) == pytest.approx(100.0, abs=0.1)
    assert task1.hypergeometric_at_k(22, 1, 5) < task1.hypergeometric_at_k(11, 1, 5)
    assert task1.hypergeometric_at_k(0, 0, 5) is None


def _record(chains, pathogen="SARS-CoV-2"):
    return {"pathogen": pathogen,
            "chains": [{"rank": i, "target": name} for i, name in enumerate(chains, 1)]}


def test_one_label_contributes_one_rank_at_its_best():
    """The whole fix for chain-level counting, and it needs no collapse step.

    `decompose` splits spike into S1, S2 and S2', and counting matching *chains* turned one
    protein into three hits -- inflating the numerator and the null together. Scoring per label
    and taking the first match counts spike once, at rank 1.
    """
    entry = next(e for e in task1_labels.pathogens(tier="strong")
                 if e["pathogen"] == "SARS-CoV-2")
    record = _record(["Spike protein S2'", "3C-like proteinase nsp5", "Spike protein S2",
                      "Spike protein S1", "RNA-directed RNA polymerase nsp12"])
    row = task1.score_panel_row(entry, record)
    assert row["labels"] == 3, "the panel gives SARS-CoV-2 three targets, not five chains"
    assert row["matched"] == 3
    assert row["ranks"] == [1, 2, 5]


def test_the_recorded_sars_cov_2_result_is_reproduced_from_the_committed_record():
    """Pins the live number, so a change to matching or grading cannot move it unnoticed."""
    result = task1.run_panel(tier="strong")
    row = next((r for r in result["rows"] if r["pathogen"] == "SARS-CoV-2"), None)
    if row is None:
        pytest.skip("the panel runner has not written a SARS-CoV-2 record yet")
    assert row["candidates"] == 20
    assert row["ranks"] == [1, 2, 5]
    assert row["p_vs_random"] == pytest.approx(0.00351, rel=1e-2)
    assert row["verdict"] == "beats chance"


def test_underpowered_is_never_reported_as_a_failure():
    """Most genomes carry one strong label, so most cannot beat chance however good the rank.

    Hepatitis B has four identified candidates and one approved target. The best attainable p
    at that size is 0.25, so even a perfect ranking is weak evidence and the verdict has to say
    so rather than reading as a miss. This is the distinction `rank_enrichment` exists for and
    the reason a per-genome p-value cannot be the headline here.
    """
    entry = next(e for e in task1_labels.pathogens(tier="strong")
                 if e["pathogen"] == "Hepatitis B virus")
    record = _record(["HBSAG_GSHV Large envelope protein", "DPOL_GSHV Protein P",
                      "External core antigen", "X_GSHV Protein X"],
                     pathogen="Hepatitis B virus")
    row = task1.score_panel_row(entry, record)
    assert row["ranks"] == [2]
    assert row["adequately_powered"] is False
    assert row["verdict"] == "not established (underpowered)"


def test_an_unmatched_label_is_reported_and_not_dropped():
    """A protein the candidate stage never proposed is a finding, not an absence.

    Four strong labels go unmatched against the committed records, and all four were checked
    against the chain names rather than assumed: HCMV's terminase and UL97 kinase, and the
    poxvirus p37 in both monkeypox and vaccinia. The ranker proposed none of them. Widening an
    alias until they matched would convert a real candidate-stage gap into a label fix, which
    is the one edit this test exists to make visible.
    """
    result = task1.run_panel(tier="strong")
    if not result["rows"]:
        pytest.skip("the panel runner has written no records yet")
    unmatched = set(result["unmatched"])
    assert ("Human cytomegalovirus", "terminase") in unmatched
    assert ("Monkeypox virus", "p37") in unmatched
    assert result["labels_matched"] < result["labels_total"]


def test_the_aggregate_combines_families_and_not_genomes():
    """26 of 47 genomes share a pattern list with a sibling, so genomes are not instances.

    Five coronaviruses carry one target list between them and five unrelated viruses share
    `RdRp / Glycoprotein / Nucleoprotein`. Stouffer needs independence, and the family step is
    what supplies it -- the instance count has to be the families, never the genome count.
    """
    result = task1.run_panel(tier="strong")
    if not result["rows"]:
        pytest.skip("the panel runner has written no records yet")
    assert result["independent_instances"] == len(result["per_family"])
    assert result["independent_instances"] < result["genomes_scored"]
    assert result["combined_p_across_families"]["k"] == result["independent_instances"]
    assert result["combined_p_across_families"]["method"] == "Stouffer, equal weights"
    # Herpesviridae is the family that makes the point: four genomes, one combined p.
    herpes = next(f for f in result["per_family"] if f["family"] == "Herpesviridae")
    assert herpes["genomes"] == 4


def test_recall_at_k_never_travels_without_its_chance_value():
    result = task1.run_panel(tier="strong")
    if not result["rows"]:
        pytest.skip("the panel runner has written no records yet")
    for key in result["recall_at_k"]:
        assert result["chance_at_k"][key] is not None, f"{key} has no null beside it"
    for row in result["rows"]:
        assert set(row["recall_at_k"]) == set(row["chance_at_k"])


def test_influenza_carries_the_recorded_experiments_labels():
    """Every protein the recorded PR8 run scored is still covered by exactly one label.

    **PR8 used to arrive twice and this test could not see it.** The panel gained an
    `Influenza A PR8` row while PR8 was also supplied through `EXTRA`, and `pathogens()`
    appended rather than merged, so the pathogen came back as two rows -- with `Neuraminidase`
    and `Polymerase acidic` from one and `Neuraminidase (NA)` and `PA endonuclease` from the
    other naming the same two proteins. Asserting against `EXTRA` alone, as this test did, was
    blind to the half of the problem that lived in the panel.

    So the assertion is now the invariant rather than a name list: whatever the two curations
    call them, each protein the recorded experiment scored must be matched by one label and not
    by two. It is checked through `task1.matches`, the matcher the benchmark itself scores with,
    because the names deliberately differ between the panel and the recorded run and a string
    comparison is what hid the duplicate in the first place.
    """
    rows = [e for e in task1_labels.pathogens(tier="all")
            if e["pathogen"] == "Influenza A PR8"]
    assert len(rows) == 1, f"PR8 appears {len(rows)} times; the panel and EXTRA have diverged"
    entry = rows[0]
    assert entry["accession"].count(",") == 7, "all eight segments, comma-joined"
    assert entry["holdout"] == "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)"

    path = os.path.join(ROOT, "runs", "target_ranks.csv")
    with open(path) as handle:
        recorded = {r["target_protein"] for r in csv.DictReader(handle)
                    if r["pathogen"] == "Influenza A PR8"}
    assert recorded, "the recorded run has no PR8 rows to check against"
    for protein in sorted(recorded):
        hits = [t["name"] for t in entry["targets"]
                if task1.matches(t, {"name": protein})]
        assert len(hits) == 1, f"{protein!r} is matched by {hits}, expected exactly one label"


def test_blinded_prompt_names_neither_virus_nor_protein():
    """The control the external arms rest on. If the pathogen leaks, the arm is not blinded."""
    entry = next(e for e in task1_labels.pathogens(tier="strong")
                 if e["pathogen"] == "SARS-CoV-2")
    candidates = [{"rank": i, "name": n, "parent": None, "length_aa": 300,
                   "sequence": "MFVFLVLLPLVSSQ"}
                  for i, n in enumerate(["Spike protein S1", "3C-like proteinase nsp5"], 1)]
    text = task1.prompt(entry, candidates, blinded=True)
    assert "SARS" not in text and "NC_045512" not in text
    assert "Spike" not in text and "nsp5" not in text
    named = task1.prompt(entry, candidates, blinded=False)
    assert "SARS-CoV-2" in named and "Spike protein S1" in named


def test_stouffer_combines_and_reports_what_it_dropped():
    assert task1.stouffer([0.5, 0.5, 0.5])["p"] == pytest.approx(0.5, abs=1e-3)
    assert task1.stouffer([0.01, 0.01, 0.01])["p"] < 0.01
    dropped = task1.stouffer([0.05, None, 1.0])
    assert dropped["k"] == 1 and dropped["dropped"] == 2
    assert task1.stouffer([])["p"] is None
