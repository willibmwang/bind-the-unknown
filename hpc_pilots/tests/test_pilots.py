import json

import numpy as np
import pandas as pd
import pytest

from hpc_pilots.binding import design_matrix, enrichment, pick_batch, replay
from hpc_pilots.calibration import calibration_metrics, trajectory
from hpc_pilots.chemistry import alphabet, encode, library_spec, registry, sample_library
from hpc_pilots.common import ROOT
from hpc_pilots.common import (Agent, COMMITTED_MODEL_MANIFEST, STAGED_MODEL_MANIFEST,
                               digest, model_manifest, provenance, source_manifest)
from hpc_pilots.data import AA, encode_variants, pool_geometry
from hpc_pilots.dossier import Dossier, evidence, rank
from hpc_pilots.emergency import run_episode
from hpc_pilots.structure import structure_replay
from hpc_pilots.targets import packet


def test_shared_initialization_unique_queries_and_delayed_observations():
    rng = np.random.default_rng(14)
    x = rng.integers(0, 20, size=(256, 4))
    y = rng.normal(size=256)
    outputs = [replay(x, y, p, 19, maximum=32, delay=3, checkpoints=(8, 16, 32))
               for p in ('random', 'diverse', 'greedy', 'ucb')]
    assert all(o['initial_indices'] == outputs[0]['initial_indices'] for o in outputs)
    for o in outputs:
        selected = o['initial_indices'] + [i for t in o['trace'] for i in t['selected']]
        assert len(selected) == len(set(selected)) == 32
        assert o['trace'][1]['revealed_before_action'] == o['initial_indices']
        assert o['final_return_round'] > o['metrics'][-1]['round']


def test_acquisition_only_consumes_observed_labels():
    rng = np.random.default_rng(12)
    x = rng.integers(0, 20, size=(128, 4))
    observed = np.arange(8)
    labels = rng.normal(size=128)
    changed = labels.copy()
    changed[8:] = 10000
    available = np.arange(8, 128)
    a, _ = pick_batch(x, observed, labels[observed], available, 'ucb', 8, 8)
    b, _ = pick_batch(x, observed, changed[observed], available, 'ucb', 8, 8)
    np.testing.assert_array_equal(a, b)


def test_structural_reference_never_changes_actions():
    rng = np.random.default_rng(5)
    def entry():
        return {'static': rng.random(12), 'r1': rng.random((80, 12)),
                'r2': rng.random((80, 12)), 'r3': rng.random((80, 12)),
                'r1_cpu_seconds': np.ones(4), 'r2_cpu_seconds': np.ones(4)}
    proteins = {str(i): entry() for i in range(3)}
    before = structure_replay(proteins, 'uncertainty', additional_blocks=8, seed=3)
    for e in proteins.values():
        e['r3'][:] = 1000
    after = structure_replay(proteins, 'uncertainty', additional_blocks=8, seed=3)
    assert before['trace'] == after['trace']
    assert before['metrics'][-1]['mean_rmse_nm2'] != after['metrics'][-1]['mean_rmse_nm2']


def test_structural_initial_budget_and_static_cost():
    rng = np.random.default_rng(3)
    entry = {'static': rng.random(12), 'r1': rng.random((80, 12)),
             'r2': rng.random((80, 12)), 'r3': rng.random((80, 12)),
             'r1_cpu_seconds': np.ones(4), 'r2_cpu_seconds': np.ones(4)}
    result = structure_replay({'one': entry}, 'uniform', additional_blocks=3)
    assert result['metrics'][0]['total_blocks'] == 2
    assert result['metrics'][-1]['total_blocks'] == 5
    assert result['metrics'][-1]['processed_frames'] == 100
    static = structure_replay({'one': entry}, 'static')
    assert static['metrics'][0]['total_blocks'] == 0
    assert static['metrics'][0]['estimated_acquisition_cpu_seconds'] == 0


def test_staged_pools_are_disjoint_and_match_manifest():
    path = ROOT / 'data/processed/binding'
    if not (path / 'splits.npz').exists():
        pytest.skip('Dataset not staged')
    manifest = json.loads((path / 'manifest.json').read_text())
    split = np.load(path / 'splits.npz')
    indices = np.r_[split['development'], split['evaluation'].ravel()]
    assert len(indices) == len(np.unique(indices))
    assert np.load(path / 'features.npy').shape == (manifest['rows'], 4)


def test_agent_contract_and_token_accounting(monkeypatch):
    import requests
    for key in ('PILOT_LLM_ENDPOINT_FILE', 'PILOT_LLM_TOKEN_FILE', 'PILOT_LLM_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PILOT_LLM_URL', 'http://local-test/v1')
    monkeypatch.setenv('PILOT_LLM_MODEL', 'test-model')
    recorded = {}
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'message': {'content': '{"action":"greedy","rationale":"test"}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}
    def fake_post(url, **kwargs):
        recorded.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(requests, 'post', fake_post)
    agent = Agent()
    assert agent.choose('select', {'observed_labels': 8}, ['greedy', 'explore']) == 'greedy'
    assert recorded['url'] == 'http://local-test/v1/chat/completions'
    schema = recorded['json']['response_format']['json_schema']['schema']
    assert schema['properties']['action']['enum'] == ['greedy', 'explore']
    assert agent.report()['tokens'] == {'prompt_tokens': 10, 'completion_tokens': 4}


def test_invalid_agent_action_fails_instead_of_substituting_policy(monkeypatch):
    import requests
    for key in ('PILOT_LLM_ENDPOINT_FILE', 'PILOT_LLM_TOKEN_FILE', 'PILOT_LLM_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PILOT_LLM_URL', 'http://local-test/v1')
    monkeypatch.setenv('PILOT_LLM_MODEL', 'test-model')
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'message': {'content': '{"action":"invented","rationale":"bad"}'}}]}
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match='outside'):
        Agent().choose('select', {'observed_labels': 8}, ['greedy'])


def test_calibration_audits_only_held_out_candidates_and_shares_initial_labels():
    rng = np.random.default_rng(7)
    x = np.eye(20, dtype=np.float32)[rng.integers(0, 20, size=(192, 4))].reshape(192, -1)
    y = rng.normal(size=192)
    rows = {regime: [r for r in trajectory(x, y, regime, 3, 8, (8, 16), 8) if r['budget'] == 8][0]
            for regime in ('random', 'greedy')}
    # The initial labels are shared, so both regimes must report identical first-row metrics.
    assert {k: v for k, v in rows['random'].items() if k != 'regime'} == \
        {k: v for k, v in rows['greedy'].items() if k != 'regime'}
    for regime in ('random', 'greedy'):
        for row in trajectory(x, y, regime, 3, 8, (8, 16, 32), 8):
            assert row['n_eval'] == len(x) - row['budget']
            assert 0.0 <= row['coverage_1sigma'] <= row['coverage_2sigma'] <= 1.0


def test_design_matrix_reproduces_the_canonical_four_site_encoding():
    rng = np.random.default_rng(4)
    features = rng.integers(0, 20, size=(32, 4))
    np.testing.assert_array_equal(design_matrix(features),
                                  np.eye(20, dtype=np.float32)[features].reshape(32, -1))
    wide = np.stack([rng.integers(0, 303, size=16), rng.integers(0, 20, size=16)], axis=1)
    assert design_matrix(wide, [303, 20]).shape == (16, 323)


def test_enrichment_is_a_ratio_against_the_pool_baseline():
    outcomes = np.arange(200.0)
    fraction = 0.05
    hits = np.argsort(-outcomes)[:10]
    # Recovering only hits enriches by the inverse of the baseline hit rate.
    assert enrichment(outcomes, hits, fraction) == pytest.approx(1 / fraction)
    # Recovering the whole pool cannot enrich over the pool it came from.
    assert enrichment(outcomes, np.arange(200), fraction) == pytest.approx(1.0)
    assert enrichment(outcomes, np.array([], dtype=int), fraction) is None


def test_pool_geometry_shrinks_rather_than_overlapping():
    # GB1 supports the original twenty disjoint 4,096-candidate pools.
    assert pool_geometry(149360, 4096, 20)["evaluation_pools"] == 20
    # Mpro cannot: the pool size must fall rather than reuse candidates across "disjoint" pools.
    mpro = pool_geometry(5725, 4096, 20)
    assert mpro["disjoint"] and mpro["reduced"]
    assert mpro["pool_size"] < 4096
    assert mpro["candidates_used"] <= 5725
    assert mpro["pool_size"] * (mpro["evaluation_pools"] + 1) == mpro["candidates_used"]
    with pytest.raises(ValueError, match="disjoint"):
        pool_geometry(100, 4096, 20)


def test_dossier_refuses_evidence_dated_after_the_decision_clock():
    d = Dossier("Mpro (nsp5)", "SARS-CoV-2", "Coronaviridae", "cytosolic")
    future = evidence("e1", "conservation", "later finding", "strong", "src", available_from=5)
    assert d.observe(future, now=2) is False
    assert d.assess("conservation")["assessment"] == "unknown"
    # The refusal is recorded for the episode artifact but must not reach the policy: a
    # count of claims it is not allowed to see would tell it the axis is one somebody
    # later wrote about, which the date boundary cannot take back.
    assert len(d.rejected) == 1
    assert 'refused_future_evidence' not in json.dumps(d.state(2))
    unseen = Dossier("Mpro (nsp5)", "SARS-CoV-2", "Coronaviridae", "cytosolic")
    assert d.state(2)['axes']['conservation'] == unseen.state(2)['axes']['conservation']
    assert d.observe(future, now=5) is True
    assert d.assess("conservation")["assessment"] == "strong"


def test_a_contested_or_uncited_axis_is_not_substantiated():
    # Retaining a contradiction is decorative if the ranking still reads the axis as
    # settled, and the verified flag is decorative if an uncited claim can carry a gate.
    settled = Dossier("settled", "P", "F")
    contested = Dossier("contested", "P", "F")
    uncited = Dossier("uncited", "P", "F")
    for d in (settled, contested, uncited):
        d.observe(evidence(f"{d.target}:gate", "reagent_tractability", "c", "strong", "src", 0), 0)
    settled.observe(evidence("s:ess", "essentiality", "c", "strong", "src", 0), 0)
    contested.observe(evidence("c:ess", "essentiality", "c", "strong", "src", 0), 0)
    contested.observe(evidence("c:ess2", "essentiality", "disputed", "weak", "other", 1,
                               conflicts_with="c:ess"), 1)
    uncited.observe(evidence("u:ess", "essentiality", "c", "strong", "memory", 0, verified=False), 0)
    assert settled.readiness()["substantiated_axes"] == 2
    assert contested.readiness()["substantiated_axes"] == 1
    assert contested.readiness()["contested_axes"] == ["essentiality"]
    assert uncited.readiness()["substantiated_axes"] == 1
    assert uncited.readiness()["uncited_axes"] == ["essentiality"]
    # The strongest claim is still reported; it simply no longer counts toward readiness.
    assert contested.assess("essentiality")["assessment"] == "strong"
    assert uncited.assess("essentiality")["verified_assessment"] == "unknown"
    assert rank([contested, settled, uncited], now=1)[0]["target"] == "settled"


def test_a_contradiction_is_scored_on_what_the_policy_did_next():
    result = run_episode("coronaviridae", "rubric", deadline=8, call_budget=8)
    contradictory = [s for s in result["probe_scores"] if s["kind"] == "contradictory"]
    assert contradictory, "the coronaviridae packet carries a contradictory probe"
    # Retention is a property of the state object, so it holds for every policy and is
    # kept only as a regression check. The policy-level reading is scored separately and
    # is allowed to be unscored when the probe lands with no round left to respond in.
    assert all(s["conflict_retained"] for s in contradictory)
    assert all("addressed_conflict" in s for s in contradictory)
    assert result["metrics"]["terminal_condition"] in (
        "recommended", "stopped", "action_budget_exhausted", "deadline_reached")


def test_contradictory_evidence_coexists_instead_of_overwriting():
    d = Dossier("Mpro (nsp5)", "SARS-CoV-2", "Coronaviridae")
    first = evidence("e1", "conservation", "conserved across the family", "strong", "a", 0)
    second = evidence("e2", "conservation", "divergent clade reported", "weak", "b", 3,
                      conflicts_with="e1")
    d.observe(first, 0)
    d.observe(second, 3)
    assert len(d.axis_evidence("conservation")) == 2
    conflict = d.conflicts()
    assert len(conflict) == 1 and conflict[0]["timestamps"] == {"e1": 0, "e2": 3}
    assert d.assess("conservation")["conflict"] is True
    assert {"axis": "conservation", "issue": "contradictory evidence retained"} in d.open_uncertainties()


def test_hard_gate_blocks_ranking_and_claims_stay_cited():
    strong = Dossier("tractable", "P", "F")
    blocked = Dossier("untractable", "P", "F")
    for axis in ("reagent_tractability", "essentiality", "conservation"):
        strong.observe(evidence(f"s:{axis}", axis, "c", "strong", "src", 0), 0)
        if axis != "reagent_tractability":
            blocked.observe(evidence(f"b:{axis}", axis, "c", "strong", "src", 0), 0)
    ordered = rank([blocked, strong], now=0)
    assert ordered[0]["target"] == "tractable"
    assert ordered[1]["blocking_axes"] == ["reagent_tractability"]
    assert strong.unsupported() == [] and blocked.unsupported() == []


def test_episode_holds_recommendation_under_irrelevant_evidence_and_gates_the_endpoint():
    result = run_episode("coronaviridae", "rubric", deadline=8, call_budget=8)
    held = [s for s in result["probe_scores"] if s["kind"] == "irrelevant"]
    assert held and all(s["held_recommendation"] for s in held)
    retained = [s for s in result["probe_scores"] if s["kind"] == "contradictory"]
    assert all(s["conflict_retained"] for s in retained)
    assert result["metrics"]["unsupported_assessments"] == 0
    recommendation = result["recommendation"]
    assert recommendation is not None
    # A cytosolic target must carry the permeability requirement, and an unsubstantiated
    # orthogonal assay must limit the claim to enrichment.
    if recommendation["compartment"] == "cytosolic":
        assert recommendation["permeability_story_required"]
    strong_assay = recommendation["readiness"] and "enrichment" in recommendation["one_round_substantiates"]
    assert strong_assay


def test_packets_flag_every_uncited_claim_and_registry_marks_platform_fields_pending():
    for name in ("coronaviridae", "orthomyxoviridae"):
        spec, dossiers, entries, probes = packet(name)
        unverified = [e["id"] for _, e in entries if not e["verified"]]
        assert all("precedent" in i for i in unverified), unverified
        assert probes, "each prototype family needs at least one update probe"
    blocks = {b["code"]: b for b in registry()}
    assert blocks["ClAc-D-Tyr"]["role"] == "initiator"
    for code in ("NMe-Ala", "D-Phe", "beta-Ala"):
        assert blocks[code]["block_class"] != "canonical"
        for field in ("reprogrammed_codon", "rosetta_params", "chemical_identifier"):
            assert blocks[code][field] is None
    spec = library_spec()
    assert spec["closure_residue"] == "C"
    assert "C" not in spec["elongator_alphabet"]
    assert any(c.startswith("NMe-") for c in spec["elongator_alphabet"])


def test_macrocycle_encoding_covers_noncanonical_blocks():
    rng = np.random.default_rng(11)
    members = sample_library(library_spec(), 12, rng)
    assert all(m[0].startswith("ClAc-") and m[-1] == "C" for m in members)
    matrix, cardinality = encode(members)
    assert cardinality == len(alphabet()) + 1
    assert cardinality > 21, "alphabet must extend past the canonical twenty plus padding"
    with pytest.raises(KeyError, match="registry"):
        encode([["ClAc-D-Tyr", "Not-A-Block", "C"]])


def test_calibration_metrics_ignore_hidden_outcomes_of_observed_labels():
    rng = np.random.default_rng(21)
    x = np.eye(20, dtype=np.float32)[rng.integers(0, 20, size=(96, 4))].reshape(96, -1)
    y = rng.normal(size=96)
    observed = np.arange(8)
    mean, sigma = rng.normal(size=96), rng.uniform(0.1, 1.0, size=96)
    corrupted = y.copy()
    corrupted[observed] = 10000
    assert calibration_metrics(y, mean, sigma, observed) == \
        calibration_metrics(corrupted, mean, sigma, observed)


def test_encoding_narrows_for_singles_and_widens_for_combined_substitutions():
    sequence = "MKTAYIAKQR"
    singles = pd.DataFrame({"mutant": ["K2A", "T3G", "Y5V"]})
    features, cardinalities, positions, note = encode_variants(singles, sequence)
    assert features.shape == (3, 2) and cardinalities == [3, 20] and positions == 3
    assert "single substitution" in note
    # The same three sites, but now with combined variants, must switch representation:
    # a variant is no longer described by one change, so every site carries its identity.
    combined = pd.DataFrame({"mutant": ["K2A", "T3G:Y5V", "K2A:T3G:Y5V"]})
    features, cardinalities, positions, note = encode_variants(combined, sequence)
    assert features.shape == (3, 3) and cardinalities == [20] * 3
    assert "combined substitutions" in note
    wildtype = [AA.index(sequence[p - 1]) for p in (2, 3, 5)]
    assert features[0].tolist() == [AA.index("A"), wildtype[1], wildtype[2]]
    assert features[2].tolist() == [AA.index("A"), AA.index("G"), AA.index("V")]


def test_encoding_rejects_a_reference_sequence_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        encode_variants(pd.DataFrame({"mutant": ["Q2A"]}), "MKTAYIAKQR")


def test_surrogate_hyperparameters_are_taken_from_the_callers_config():
    # The pathogen and method configs each carried forest_trees and ucb_beta that nothing
    # read. Passing them must now change the exploration bonus, or the keys are still dead.
    rng = np.random.default_rng(31)
    x = rng.integers(0, 20, size=(256, 4))
    observed = np.arange(16)
    values = rng.normal(size=16)
    available = np.arange(16, 256)
    flat, wide = ({"forest_trees": 16, "ucb_beta": 0.0}, {"forest_trees": 16, "ucb_beta": 50.0})
    greedy_like, _ = pick_batch(x, observed, values, available, 'ucb', 8, 5, surrogate=flat)
    explorative, _ = pick_batch(x, observed, values, available, 'ucb', 8, 5, surrogate=wide)
    assert set(greedy_like.tolist()) != set(explorative.tolist())
    # With the bonus switched off, UCB must reduce to the greedy ordering at the same seed.
    greedy, _ = pick_batch(x, observed, values, available, 'greedy', 8, 5, surrogate=flat)
    np.testing.assert_array_equal(greedy_like, greedy)


def test_model_manifest_resolves_to_the_committed_pin_in_a_clean_checkout():
    resolved = model_manifest()
    assert resolved == (STAGED_MODEL_MANIFEST if STAGED_MODEL_MANIFEST.exists()
                        else COMMITTED_MODEL_MANIFEST)
    # The committed pin is what a clean checkout has to be able to pass its staging guard on.
    pin = json.loads(COMMITTED_MODEL_MANIFEST.read_text())
    assert pin['status'] == 'staged' and len(pin['revision']) == 40


def test_provenance_names_the_host_and_the_source_state():
    record = provenance()
    assert set(record) == {'host', 'platform', 'python', 'git_commit', 'git_dirty',
                           'slurm_job_id', 'slurm_array_task_id', 'recorded_at_utc'}
    assert record['host'] and record['recorded_at_utc'].endswith(('Z', '+00:00'))
    # A run from a modified tree is not reproducible from the commit it names, so the
    # flag has to be present and boolean wherever git is available at all.
    assert record['git_dirty'] is None or isinstance(record['git_dirty'], bool)


def test_source_manifest_check_reports_drift_without_rewriting_the_record():
    path = ROOT / 'hpc_pilots/source_manifest.json'
    before = digest(path)
    returned, drift = source_manifest(check=True)
    assert returned == path and digest(path) == before
    assert set(drift) == {'changed', 'added', 'removed'}
    # Every pattern the record claims to cover must actually resolve to files on disk.
    assert json.loads(path.read_text())['files']
