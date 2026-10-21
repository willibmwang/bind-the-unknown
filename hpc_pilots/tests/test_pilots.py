import json
import time

import numpy as np
import pandas as pd
import pytest

from hpc_pilots.__main__ import main as cli_main
from hpc_pilots.arms import ARMS
from hpc_pilots.binding import design_matrix, enrichment, pick_batch, replay
from hpc_pilots.calibration import calibration_metrics, trajectory
from hpc_pilots.chemistry import alphabet, encode, library_spec, registry, sample_library
from hpc_pilots.common import ROOT
from hpc_pilots.common import (Agent, COMMITTED_MODEL_MANIFEST, STAGED_MODEL_MANIFEST,
                               content_digest, digest, model_manifest, provenance,
                               source_manifest)
from hpc_pilots.data import AA, encode_variants, pool_geometry
from hpc_pilots.dossier import Dossier, evidence, leaders, rank
from hpc_pilots.emergency import decision_relevant, run_episode
from hpc_pilots.monitor import packet_verdict, run_monitor
from hpc_pilots.monomers import (MODELS, _standardise, absence_flags, encode as encode_monomers, holdout,
                                 load, transfer)
from hpc_pilots.invariance import (KINDS, Relabelling, certify, earned, group, leaks, lexicon,
                                   run_orbit,
                                   relabel_packet)
from hpc_pilots.twins import generate, grade_move, probe, run_twins
from hpc_pilots.streams import stream, stream_int
from hpc_pilots.structure import structure_replay
from hpc_pilots.summarize import collect, headroom
from hpc_pilots.targets import PACKETS, packet


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


def _mock_agent(monkeypatch, responses):
    """Wire a fake transport returning `responses` in order. Each is (content, finish_reason)."""
    import requests
    for key in ('PILOT_LLM_ENDPOINT_FILE', 'PILOT_LLM_TOKEN_FILE', 'PILOT_LLM_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PILOT_LLM_URL', 'http://local-test/v1')
    monkeypatch.setenv('PILOT_LLM_MODEL', 'test-model')
    sent = []
    class Response:
        def __init__(self, content, reason):
            self.content, self.reason = content, reason
        def raise_for_status(self): pass
        def json(self):
            return {'choices': [{'message': {'content': self.content},
                                 'finish_reason': self.reason}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}
    def fake_post(url, **kwargs):
        sent.append(kwargs)
        return Response(*responses[min(len(sent) - 1, len(responses) - 1)])
    monkeypatch.setattr(requests, 'post', fake_post)
    monkeypatch.setattr(time, 'sleep', lambda *a: None)
    return sent


def test_a_content_filter_is_retried_and_the_recovery_is_recorded(monkeypatch):
    # Measured on a real packet: two of four attempts were stopped at 666 and 795 chars
    # while rationales elsewhere complete at 919, so the same request can succeed
    # unchanged. Retrying it is recovery from a provider failure, not resampling a policy.
    truncated = ('{"action":"greedy","rationale":"All three targets have zero substantiated',
                 'content_filter')
    good = ('{"action":"greedy","rationale":"ok"}', 'stop')
    _mock_agent(monkeypatch, [truncated, truncated, good])
    agent = Agent()
    assert agent.choose('select', {'observed_labels': 8}, ['greedy', 'explore']) == 'greedy'
    assert agent.filtered == 2
    assert agent.calls == 3, 'every attempt costs a call and must be counted'
    assert [i['finish_reason'] for i in agent.incidents] == ['content_filter'] * 2
    assert agent.report()['filtered_attempts'] == 2


def test_a_persistent_content_filter_fails_rather_than_substituting(monkeypatch):
    truncated = ('{"action":"greedy","rationale":"unterminated', 'content_filter')
    _mock_agent(monkeypatch, [truncated])
    monkeypatch.setenv('PILOT_LLM_RETRIES', '3')
    with pytest.raises(ValueError, match='content filter'):
        Agent().choose('select', {'observed_labels': 8}, ['greedy'])


def test_a_truncated_response_is_not_retried_because_it_cannot_clear(monkeypatch):
    # A request that hit the output ceiling will hit it again; retrying spends money to
    # reproduce the same failure. The ceiling has to be raised instead, and the message says so.
    _mock_agent(monkeypatch, [('{"action":"greedy","rationale":"cut', 'length')])
    agent = Agent()
    with pytest.raises(ValueError, match='output ceiling'):
        agent.choose('select', {'observed_labels': 8}, ['greedy'])
    assert agent.calls == 1 and agent.filtered == 0


def test_an_invalid_action_is_never_retried_away(monkeypatch):
    # Resampling until the policy returns something legal would filter its behaviour out
    # of the record. One call, then raise.
    _mock_agent(monkeypatch, [('{"action":"invented","rationale":"bad"}', 'stop')])
    agent = Agent()
    with pytest.raises(ValueError, match='outside'):
        agent.choose('select', {'observed_labels': 8}, ['greedy'])
    assert agent.calls == 1


def test_a_client_error_is_not_retried_but_a_rate_limit_is(monkeypatch):
    import requests
    for key in ('PILOT_LLM_ENDPOINT_FILE', 'PILOT_LLM_TOKEN_FILE', 'PILOT_LLM_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PILOT_LLM_URL', 'http://local-test/v1')
    monkeypatch.setenv('PILOT_LLM_MODEL', 'test-model')
    monkeypatch.setattr(time, 'sleep', lambda *a: None)
    attempts = []
    class Response:
        def __init__(self, status): self.status_code = status
        def raise_for_status(self):
            if self.status_code >= 400:
                error = requests.HTTPError(f'{self.status_code}')
                error.response = self
                raise error
        def json(self):
            return {'choices': [{'message': {'content': '{"action":"greedy","rationale":"ok"}'},
                                 'finish_reason': 'stop'}], 'usage': {}}
    def post_sequence(statuses):
        def fake(url, **kwargs):
            attempts.append(1)
            return Response(statuses[min(len(attempts) - 1, len(statuses) - 1)])
        return fake
    # A 429 clears on a later attempt, so the sweep survives it.
    monkeypatch.setattr(requests, 'post', post_sequence([429, 429, 200]))
    agent = Agent()
    assert agent.choose('select', {}, ['greedy']) == 'greedy'
    assert agent.transport_retries == 2
    # A 400 is a malformed request and will fail identically however often it is sent.
    attempts.clear()
    monkeypatch.setattr(requests, 'post', post_sequence([400]))
    with pytest.raises(requests.HTTPError):
        Agent().choose('select', {}, ['greedy'])
    assert len(attempts) == 1


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


def test_streams_do_not_collide_along_the_pool_round_diagonal():
    # The defect this replaces: streams were derived by ADDING a coordinate to the base
    # seed, so pool p at round r and pool p+r at initialisation got the same integer and
    # the same draws. Verified before the fix: pool 0 round 1 matched pool 1 init exactly.
    base = 20261012
    for pool in range(4):
        for round_number in range(1, 4):
            acquire = stream(base + pool, 'acquire', round_number).integers(0, 1000, size=12)
            collided = stream(base + pool + round_number, 'initial').integers(0, 1000, size=12)
            assert not np.array_equal(acquire, collided), (pool, round_number)
    # The surrogate's bootstrap must not share a stream with the tie-break that exists to
    # decorrelate its tied plateaus from index order.
    assert stream_int(base, 'surrogate', 2) != stream_int(base, 'acquire', 2)
    # Same coordinate, same stream: the scheme is reproducible, not merely scattered.
    assert np.array_equal(stream(base, 'acquire', 3).integers(0, 1000, size=8),
                          stream(base, 'acquire', 3).integers(0, 1000, size=8))
    # A string coordinate must survive a process boundary, so it cannot use salted hash().
    # Recorded from the implementation: this fails if the derivation changes, which is the
    # point -- a silent change would re-randomise every run without anyone noticing.
    assert stream_int(base, 'surrogate', 1) == 817098475


def test_all_policies_still_open_from_identical_observations():
    # The invariant the whole matched comparison rests on, re-pinned after the seed
    # rework: initialisation must not be a function of the policy.
    rng = np.random.default_rng(3)
    x = rng.integers(0, 20, size=(256, 4))
    y = rng.normal(size=256)
    runs = [replay(x, y, policy, 11, maximum=32, checkpoints=(32,))
            for policy in ('random', 'diverse', 'greedy', 'ucb')]
    assert all(r['initial_indices'] == runs[0]['initial_indices'] for r in runs)
    # And the permuted control must not perturb which labels the episode opens from.
    control = replay(x, y, 'greedy', 11, maximum=32, checkpoints=(32,), control='permuted')
    assert control['initial_indices'] == runs[0]['initial_indices']


def test_permuted_control_destroys_a_learnable_signal():
    # The falsification arm. On a landscape where the outcome is a clean function of the
    # features, a learning policy must do well; once the outcomes are permuted there is
    # nothing left to learn and it must not. A policy that scored the same either way
    # would be scoring on index order or tie-breaking, not on the evidence it bought.
    rng = np.random.default_rng(17)
    x = rng.integers(0, 20, size=(512, 4))
    y = x[:, 0].astype(float) + 0.01 * rng.normal(size=512)
    shared = dict(maximum=128, initial=8, batch=8, checkpoints=(128,), fraction=0.05)
    # Averaged over several episode coordinates rather than asserted at one. A single
    # coordinate is a knife edge: on this synthetic landscape greedy occasionally stalls
    # on a tied plateau and scores near zero, so a one-seed assertion tests the draw
    # rather than the control.
    def recall(policy, seed, **extra):
        return replay(x, y, policy, seed, **shared, **extra)['metrics'][-1]['top_recall']
    seeds = range(5, 17)
    real = [recall('greedy', s) for s in seeds]
    permuted = [recall('greedy', s, control='permuted') for s in seeds]
    baseline = [recall('random', s) for s in seeds]
    # The claim is the separation, not an absolute level. Greedy cold-starts from eight
    # labels, so on a minority of coordinates the forest sees no high-outcome example,
    # predicts a flat surface and the whole batch resolves on the tie-break -- measured
    # at 4 of 20 coordinates. An absolute threshold would be testing the draw.
    assert np.mean(real) > 2 * np.mean(permuted), (np.mean(real), np.mean(permuted))
    # Under permutation the learning policy must fall back onto the random band, which is
    # the whole point: what is left is the budget, not the policy. Measured gap over 20
    # coordinates was 0.002.
    assert abs(np.mean(permuted) - np.mean(baseline)) < 0.06
    assert replay(x, y, 'greedy', 5, **shared)['control'] is None
    with pytest.raises(ValueError, match='control'):
        replay(x, y, 'greedy', 5, maximum=16, control='nonsense')


def test_zero_retrieval_control_reaches_a_decision_having_bought_nothing():
    # The memorisation probe. Targets are named, so pretrained knowledge can stand in for
    # retrieval; whatever this arm scores is a floor the evidence-using arm must clear
    # before its performance can be credited to the evidence.
    blind = run_episode("coronaviridae", "rubric", deadline=8, call_budget=8,
                        control="zero_retrieval")
    assert blind["metrics"]["evidence_retrieved"] == 0
    assert not any(r["action"] == "retrieve_evidence" for r in blind["trace"])
    assert blind["probe_scores"] == [], "no evidence may be pushed under the control"
    assert blind["metrics"]["reference_top_target"]
    # The evidence-using arm buys evidence and is scored against the same reference.
    seeing = run_episode("coronaviridae", "rubric", deadline=8, call_budget=8)
    assert seeing["metrics"]["evidence_retrieved"] > 0
    assert seeing["metrics"]["reference_top_target"] == blind["metrics"]["reference_top_target"]
    with pytest.raises(ValueError, match="control"):
        run_episode("coronaviridae", "rubric", control="nonsense")


def test_headroom_is_capped_by_whichever_ceiling_binds():
    # 128 labels in a 512 pool at the 5 percent threshold: the budget binds, not the
    # threshold, and the attainable multiple is 4 rather than 20.
    assert headroom(0.25, 0.05) == {"recall_ceiling": 1.0, "headroom_multiple": 4.0}
    # 8 labels in the same pool: the threshold binds. Reporting 1/labelled here would
    # claim 64x of headroom where 20x exists, at the checkpoint where raw multiples
    # look most impressive.
    early = headroom(8 / 512, 0.05)
    assert early["headroom_multiple"] == 20.0
    assert early["recall_ceiling"] == pytest.approx(0.3125)


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


def test_content_digest_ignores_when_but_not_what():
    """A manifest's identity must survive restaging and must not survive editing."""
    base = {'rows': 10, 'sources': [{'url': 'u', 'sha256': 'abc',
                                     'verified_at_utc': '2026-10-20T20:05:35+00:00'}]}
    later = json.loads(json.dumps(base))
    later['sources'][0]['verified_at_utc'] = '2026-10-21T09:00:00+00:00'
    assert content_digest(base) == content_digest(later)
    # Key order is not content either, so a reordered manifest keeps its identity.
    assert content_digest({'b': 1, 'a': 2}) == content_digest({'a': 2, 'b': 1})
    # But anything that describes the data itself must change it.
    changed = json.loads(json.dumps(base))
    changed['sources'][0]['sha256'] = 'def'
    assert content_digest(base) != content_digest(changed)
    assert content_digest(base) != content_digest({**base, 'rows': 11})
    # The field is excluded from its own input, so stamping it in is idempotent.
    stamped = {**base, 'content_sha256': content_digest(base)}
    assert content_digest(stamped) == content_digest(base)


def test_every_arm_round_trips_its_result_filename():
    """The registry is the only place an arm's layout is written down, so it must be total."""
    for name, a in ARMS.items():
        assert a.name == name
        assert a.config_path().startswith('hpc_pilots/configs/')
        assert a.runs.startswith('runs/') and a.processed.startswith('data/processed/')
        # A control arm writes to its own file and can never collide with the real run.
        real = a.filename('ls', 'greedy', 3, 0)
        control = a.filename('ls', 'greedy', 3, 0, control='permuted')
        smoke = a.filename('ls', 'greedy', 3, 0, smoke=True)
        assert len({real, control, smoke}) == 3
        assert real.endswith('_pool03_delay0.json') and smoke.startswith('smoke_')
        # Only the pilot has no landscape to name; every other arm must carry one.
        assert ('ls_' in real) == a.per_landscape


def test_staged_landscapes_carry_a_stable_content_hash():
    """Every staged landscape must be identifiable by content, not only by file bytes."""
    staged = sorted((ROOT / 'data/processed').glob('*/manifest.json')) + \
             sorted((ROOT / 'data/processed').glob('*/*/manifest.json'))
    if not staged:
        pytest.skip('no landscapes staged')
    for path in staged:
        manifest = json.loads(path.read_text())
        assert manifest.get('content_sha256'), f'{path.parent.name} has no content hash'
        # Recomputing from the file on disk must reproduce the value the file records.
        assert content_digest(manifest) == manifest['content_sha256']


def test_collect_never_pools_a_falsification_run_into_a_result():
    """A permuted-control replay exists to be compared against a result, never averaged in.

    The exclusion used to be written once per summariser and two of them had lost it, so
    196 control replays were pooled into the pathogen metrics and 436 into the method
    metrics. One collector means one test can hold the line for every arm.
    """
    for name in ARMS:
        frame = collect(name)
        if frame.empty:
            continue
        assert not frame.file.str.contains('_permuted').any(), f'{name} pooled a control run'
        assert not frame.file.str.contains('smoke_').any(), f'{name} pooled a smoke run'
        # Identity columns must survive into the rows, or a reader cannot tell which
        # substrate and endpoint produced a number without going back to the run files.
        for column in ARMS[name].identity:
            assert column in frame.columns, f'{name} dropped {column}'
        if ARMS[name].per_landscape:
            assert 'landscape' in frame.columns and frame.landscape.notna().all()


def test_the_cli_exposes_every_registered_arm():
    """Adding an arm to the registry must not require remembering to add it to the CLI."""
    import contextlib
    import io

    for name, spec in ARMS.items():
        if not spec.per_landscape:
            continue
        for command in (f'prepare-{name}', name):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exit:
                cli_main([command, '--help'])
            assert exit.value.code == 0, f'{command} is not a CLI subcommand'
            assert '--landscape' in out.getvalue()
        # The falsification arm must be reachable for every staged arm, not just the ones
        # somebody remembered to copy the flag into.
        out = io.StringIO()
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
            cli_main([name, '--help'])
        assert '--control' in out.getvalue(), f'{name} cannot run its control arm'


def test_decision_relevance_does_not_depend_on_the_order_the_dossiers_arrived_in():
    # The oracle compared rank(...)[0], and `rank` sorts stably, so under a tie at the top
    # the leader was whichever target the caller's list happened to put first. Permuting the
    # list then changed the size of the relevant set -- 26 pairs against 28 on Coronaviridae
    # -- with no policy doing anything. A quantity that moves when the state is rearranged
    # cannot be the oracle a policy is scored against.
    _, dossiers, entries, _ = packet("coronaviridae")
    pool = {(t, e["axis"]): e for t, e in entries}
    base = set(decision_relevant(dossiers, pool, set(), 1))
    for shift in range(1, len(dossiers)):
        rotated = dossiers[shift:] + dossiers[:shift]
        assert set(decision_relevant(rotated, pool, set(), 1)) == base
    # And the tie it was silently resolving is real: at the opening round every target sits
    # on an empty dossier, so all of them are tied for the lead.
    assert len(leaders(dossiers, 1)["tied_for_lead"]) == len(dossiers)
    assert not leaders(dossiers, 1)["decided"]


def test_a_nuisance_relabelling_is_certified_against_the_oracle_before_it_is_used():
    # A transformation is a nuisance only if the oracle's answer is its image. That is
    # checkable here rather than assumed, because the evaluator computes the oracle exactly.
    for name in PACKETS:
        for kind in KINDS:
            for element in group(name, kind, 20261021, 2):
                certificate = certify(name, element)
                assert certificate["certified"], (name, kind, certificate["checks"])


def test_an_identity_relabelling_leaves_no_declared_identifier_anywhere_in_the_record():
    # Scrubbing the fields a policy reads is not enough. The packet specification is a
    # nested dict whose keys are target names and whose claims cross-reference other
    # targets -- "conserved but less so than Mpro across genera" is filed under PLpro --
    # so a field-by-field scrub left every name in the artifact while removing it from the
    # state. That leaks to every later reader of the record and to anything trained on it.
    element = group("coronaviridae", "identity", 20261021, 1)[0]
    spec, dossiers, entries, probes = relabel_packet("coronaviridae", element)
    record = [spec, [d.state(8) for d in dossiers], [e for _, e in entries], probes]
    assert leaks(record, lexicon("coronaviridae")) == {}
    # The conflict pointer has to survive the rename, or a retained contradiction stops
    # resolving to the claim it contradicts and the state quietly loses the thing it exists
    # to preserve.
    identifiers = {e["id"] for _, e in entries} | {p["entry"]["id"] for p in probes}
    pointers = {p["entry"]["conflicts_with"] for p in probes
                if p["entry"]["conflicts_with"] is not None}
    assert pointers and pointers <= identifiers


def test_an_uncertified_relabelling_is_refused_rather_than_measured():
    # An element that fails certification is evidence about the harness, not about any
    # policy, and averaging it into an orbit would report the harness as a result.
    broken = Relabelling("identity", 0, names=(("Mpro (nsp5)", "TGT-01"),))
    assert not certify("coronaviridae", broken)["certified"]
    with pytest.raises(ValueError, match="not certified"):
        run_twins("coronaviridae", "rubric", attempts=8, relabelling=broken)


def test_a_move_is_graded_by_how_much_of_the_decision_it_settles():
    # Binary relevance could not separate resolving the leader's next axis from handing the
    # lead to a challenger: both scored 1, so a policy walking the leader's axes in
    # declaration order was indistinguishable from one choosing the purchase that decides
    # the campaign. The grade is ordered by what the purchase settles.
    tied = {"tied_for_lead": frozenset({"a", "b"}), "substantiated_axes": 1}
    assert grade_move(tied, tied) == 0
    assert grade_move(tied, {**tied, "substantiated_axes": 2}) == 1
    assert grade_move(tied, {**tied, "tied_for_lead": frozenset({"a", "b", "c"})}) == 2
    assert grade_move(tied, {**tied, "tied_for_lead": frozenset({"a"})}) == 3
    # Resolving a contest that was already resolved is a change of leader, not a decision.
    decided = {"tied_for_lead": frozenset({"a"}), "substantiated_axes": 1}
    assert grade_move(decided, {**decided, "tied_for_lead": frozenset({"b"})}) == 2


def test_the_fixed_rubric_attains_the_ceiling_only_when_the_best_purchase_is_on_the_leader():
    # The instrument's two controls, and the reason it replaced the relevance readout.
    # `_rubric_action` resolves the leading target's unknown axes in declaration order, and
    # the previous oracle marked a purchase relevant when it moved the lead verdict -- which
    # once any target leads is the same set. The rubric scored 6 of 6 against it, a lift of
    # 4.48, higher than the hosted model's 3.81, by definitional collision rather than skill.
    result = run_twins("orthomyxoviridae", "rubric", attempts=200)
    summary = result["summary"]
    # Positive control: where the best purchase is on the leader the probe is answerable and
    # the rubric answers it perfectly. A probe no policy can score here is broken, not hard.
    assert summary["leader_best"]["attainment"] == 1.0
    assert summary["leader_best"]["optimal_rate"] == 1.0
    # Discrimination: where the best purchase is on a challenger the rubric falls to chance
    # or below, because it never examines a target that is not leading.
    assert summary["challenger_best"]["attainment"] < 0.5
    assert summary["challenger_best"]["optimal_rate"] < 0.2
    assert summary["leader_best"]["bought_on_leader_rate"] == 1.0
    assert summary["challenger_best"]["bought_on_leader_rate"] == 1.0
    assert summary["contrast"]["leader_minus_challenger"] > 0.5


def test_a_state_blind_policy_scores_at_chance_in_both_strata():
    # The normalisation's own check. Attainment is scaled so that 0.0 is the average purchase
    # on offer, so a policy that ignores the state must land near zero in both strata. If it
    # did not, the floor would be mis-stated and every attainment figure would inherit the
    # error. It also separates the two failures: the rubric's large drop is leader fixation,
    # not general incompetence, because an incompetent policy drops very little.
    result = run_twins("coronaviridae", "scripted", attempts=200)
    summary = result["summary"]
    for family in ("leader_best", "challenger_best"):
        assert abs(summary[family]["attainment"]) < 0.3, family
    assert abs(summary["contrast"]["leader_minus_challenger"]) < 0.3


def test_earned_performance_is_read_against_the_presentation_orbit_not_the_nominal_run():
    # A policy that is merely unstable must not be reported as one reciting a name. The
    # identity orbit is compared with the presentation orbit -- the same content rearranged
    # at no cost -- rather than with the nominal run, so the policy's own variance and its
    # sensitivity to the arrangement of the state are not charged to memorisation.
    split = earned(1.0, {"null": [0.9, 0.9, 0.9], "presentation": [0.9, 0.9],
                         "identity": [0.4, 0.4]})
    assert split["attributable_to_identity"] == pytest.approx(0.5)
    assert split["earned_mean"] == pytest.approx(0.4)
    assert split["policy_variance_floor"] == 0.0
    assert split["exceeds_variance_floor"] is True
    # A policy whose identity orbit matches its presentation orbit earned its score, however
    # unstable it is across repeats of the identical instance.
    unstable = earned(1.0, {"null": [0.2, 1.0], "presentation": [0.6, 0.6],
                            "identity": [0.6, 0.6]})
    assert unstable["attributable_to_identity"] == 0.0
    assert unstable["exceeds_variance_floor"] is False
    # A missing arm reports a gap rather than substituting the nominal run for it.
    assert earned(1.0, {"null": [0.5]})["attributable_to_identity"] is None


def test_a_name_blind_policy_has_exactly_no_identity_contribution():
    # The calibration that licenses every number this instrument reports. The rubric reads
    # the evidence state and never a target's name, so renaming the targets cannot move its
    # score, and the orbit must return exactly zero rather than approximately zero.
    #
    # It did not, before the states were matched. `generate` draws its subset over
    # sorted(pool), whose order is the target names, so each orbit element saw a different
    # sample of states and the orbit confounded renaming with resampling -- reporting a 0.11
    # identity effect for a policy that cannot have one. States are now sampled once in base
    # coordinates and translated, so the two halves are the same problem under two names.
    orbit = run_orbit("orthomyxoviridae", "rubric", count=2, attempts=120)
    for score, split in orbit["earned"].items():
        assert split["attributable_to_identity"] == 0.0, score
        for kind in KINDS:
            assert split[f"{kind}_orbit"]["sd"] == 0.0, (score, kind)
            assert split[f"{kind}_orbit"]["mean"] == split["nominal"], (score, kind)


def test_a_relabelling_that_merges_two_targets_is_caught_rather_than_measured():
    # The substitution table has to be a bijection on targets. One that is not merges two
    # dossiers onto a single name, so the packet silently loses a target and every purchase
    # on offer is re-graded against a decision with one fewer contender. That is a different
    # problem, not a renamed one, and it would otherwise be averaged into an orbit as though
    # it were the same instance.
    states, _ = generate("coronaviridae", 20261021, attempts=60)
    assert states, "the generator produced no discriminating state to probe"
    element = group("coronaviridae", "identity", 20261021, 1)[0]
    table = dict(element.names)
    targets = sorted(PACKETS["coronaviridae"]["targets"])
    table[targets[1]] = table[targets[0]]
    merged = Relabelling("identity", 0, names=tuple(sorted(table.items())))
    assert not certify("coronaviridae", merged)["certified"]
    with pytest.raises(ValueError, match="does not reproduce|not the image"):
        probe(states[0], "rubric", relabelling=merged)


def test_a_vocabulary_encoding_cannot_tell_two_unseen_monomers_apart():
    # Structural, not statistical. Holding a monomer out of training leaves its one-hot column
    # with zero variance over the training rows, so no tree splits on it and no coefficient is
    # fitted to it. Two members differing only in which held-out monomer they carry therefore
    # traverse identically and receive the identical prediction -- 1599 of 1599 pairs on the
    # thrombin library. No quantity of further data of the same kind moves that number, which
    # is why a surrogate over token indices cannot rank a library built from a new block.
    result = transfer("thrombin_cyclic", held_count=6)
    by_cell = {(r["encoder"], r["model"]): r for r in result["results"]}
    for model in MODELS:
        paired = by_cell[("vocabulary", model)]["paired"]
        assert paired["pairs"] > 500, "too few discriminating pairs to pin the claim"
        assert paired["tied_rate"] == 1.0, model
        assert paired["tied_predictions"] == paired["pairs"]
        assert paired["concordance"] is None, "an all-tied arm has nothing to be concordant on"
    # Grounding the position in measured structure removes the blindness: every descriptor
    # arm separates the pairs it could not separate before.
    for encoder in ("descriptor", "both", "descriptor_shuffled"):
        for model in MODELS:
            assert by_cell[(encoder, model)]["paired"]["tied_rate"] == 0.0, (encoder, model)


def test_descriptor_grounding_removes_the_blindness_without_adding_ranking_skill():
    # The negative half, and the control that licenses calling it one. The descriptor arm can
    # order two unseen monomers, but it orders them at chance: concordance 0.49 against a
    # shuffled control's 0.49, where shuffling permutes the ten descriptors across monomers
    # and so destroys the correspondence between a block's identity and its structure while
    # leaving dimensionality, column scales, the absence indicator and the model class alone.
    #
    # Being able to represent a new building block is necessary and not sufficient. On this
    # landscape the per-monomer descriptors carry no transferable signal about which new
    # monomer is better, which is consistent with the staged manifest's own caveat that they
    # describe the building blocks and nothing about the assembled macrocycle.
    result = transfer("thrombin_cyclic", held_count=6)
    by_cell = {(r["encoder"], r["model"]): r for r in result["results"]}
    for model in MODELS:
        real = by_cell[("descriptor", model)]["paired"]["concordance"]
        shuffled = by_cell[("descriptor_shuffled", model)]["paired"]["concordance"]
        assert abs(real - 0.5) < 0.1, (model, real)
        # The claim is that chemistry adds nothing here, so the real descriptors must not
        # beat their own shuffle by a margin worth reporting. A reversal of this assertion
        # would be the positive result and must be read as one, not patched away.
        assert real - shuffled < 0.1, (model, real, shuffled)


def test_a_monomer_split_holds_the_block_out_everywhere_it_occurs():
    # A member-wise split leaks: every monomer appears in hundreds of members, so a random
    # split leaves each one well represented in training and measures interpolation inside a
    # known alphabet. The monomers here are shared across two positions, so a block held out
    # at one position and left at another is not held out at all.
    features, outcomes, descriptors, vocabularies, _ = load("thrombin_cyclic")
    train_rows, test_rows, held, positions = holdout(features, vocabularies, 6, 20261021)
    assert set(train_rows) & set(test_rows) == set()
    assert len(train_rows) + len(test_rows) == len(features)
    assert len(positions) > 1, "the claim needs a monomer shared across positions"
    for position in positions:
        held_indices = {i for i, s in enumerate(vocabularies[position]) if s in held}
        if held_indices:
            assert not set(features[train_rows, position]) & held_indices, position
    # Standardisation uses training rows only. Scaling on the whole matrix would let the test
    # members' chemistry set the mean, a leak that flatters the descriptor arm specifically --
    # the arm whose advantage is the claim under test.
    flags = absence_flags(vocabularies, "thrombin_cyclic")
    matrix, _ = encode_monomers(features, descriptors, flags, vocabularies, train_rows, "descriptor",
                       20261021)
    informative = matrix[train_rows].std(axis=0) > 0
    # Tolerance is float32's, not the standardisation's: the matrix is float32 and the mean is
    # accumulated over five thousand rows, so a residual around 1e-4 is the dtype rather than
    # a centring error.
    assert np.allclose(matrix[train_rows][:, informative].mean(axis=0), 0, atol=1e-3)
    assert np.allclose(matrix[train_rows][:, informative].std(axis=0), 1, atol=1e-3)


def test_a_column_with_one_distinct_value_is_dropped_however_wide_the_matrix_is():
    # Constancy is tested on the extent, not on the standard deviation, and the difference is
    # a real defect rather than a style preference. A column holding one distinct value has a
    # deviation of zero mathematically, but numpy accumulates the mean in float32 and orders
    # that sum by the matrix's stride, so the returned deviation is around 1e-7 and *changes
    # with how many other columns sit beside it*. Dividing by it turned three constant
    # thrombin columns into features with mean +/-1 whose sign depends on the split.
    column = np.full((400, 1), 0.667, dtype=np.float32)
    for width in (1, 7, 66, 150):
        padded = np.concatenate(
            [column] + [np.arange(400, dtype=np.float32).reshape(-1, 1) + j
                        for j in range(width - 1)], axis=1)
        scaled, constant = _standardise(padded, np.arange(400))
        assert constant == 1, width
        assert np.all(scaled[:, 0] == 0), width
    # And the quantity the discarded test compared is not zero and not reliably small: the
    # same constant column reduces to 1.2e-07 as a contiguous slice and 3.2e-06 column-wise,
    # so a threshold on the deviation catches it or misses it depending on the reduction.
    deviation = float(np.concatenate(
        [column] + [np.arange(400, dtype=np.float32).reshape(-1, 1) + j
                    for j in range(65)], axis=1).std(axis=0)[0])
    assert deviation > 1e-6 * max(abs(0.667), 1.0), "the deviation test would have passed"
    assert float(column.max() - column.min()) == 0.0, "the extent is exact"


def test_the_readout_audit_never_pools_across_packets():
    # The audit's own regression test. Pooling the spread over packets made
    # `held_full_ranking_on_irrelevant` look discriminating with a range of 1.0, when both
    # policies scored identically inside every packet and the entire range was between-packet
    # variation: 0.0 on Coronaviridae against 1.0 on Henipavirus. That is the error this
    # codebase forbids everywhere else, committed by the audit that exists to catch it.
    identical_within_packet = packet_verdict({"rubric": 0.0, "scripted": 0.0}, "floor")
    assert identical_within_packet["verdict"] == "degenerate"
    assert identical_within_packet["spread_across_policies"] == 0.0
    other_packet = packet_verdict({"rubric": 1.0, "scripted": 1.0}, "floor")
    assert other_packet["verdict"] == "saturated_by_a_trivial_policy"
    # Neither packet discriminates, so no combination of them may be reported as doing so.
    assert "discriminating" not in {identical_within_packet["verdict"], other_packet["verdict"]}


def test_a_trivial_policy_at_the_ceiling_outranks_a_non_zero_spread():
    # A readout can separate two trivial policies and still be unusable, because the
    # separation is all below a ceiling a fixed traversal already reaches and there is no
    # headroom above it for a better policy to be observed in. This is
    # `decision_relevant_rate` on Coronaviridae: rubric 1.0, scripted 0.75, spread 0.25.
    saturated = packet_verdict({"rubric": 1.0, "scripted": 0.75}, "chance_relevant_rate")
    assert saturated["verdict"] == "saturated_by_a_trivial_policy"
    assert saturated["spread_across_policies"] == 0.25
    # Below the ceiling with a computable floor, the same spread is usable. This is the same
    # readout on Henipavirus, which is why the headline verdict is the best packet and not
    # the worst: a packet-dependent readout is not a dead one.
    usable = packet_verdict({"rubric": 0.8, "scripted": 0.5}, "chance_relevant_rate")
    assert usable["verdict"] == "discriminating"
    # The same spread with no stated chance level cannot be read on its own.
    assert packet_verdict({"rubric": 0.8, "scripted": 0.5}, None)["verdict"] \
        == "floor_not_computable"
    # One policy declining to produce the readout is missing data, not agreement.
    assert packet_verdict({"rubric": 0.4, "scripted": None}, "floor")["verdict"] \
        == "not_exercised"


def test_the_audit_condemns_every_readout_already_known_to_be_dead():
    # The three failures this module was written from, caught without being told about them.
    ledger = run_monitor(packets=["coronaviridae"], attempts=120)
    verdicts = {k: v["verdict"] for k, v in ledger["episode_readouts"].items()}
    # Scores the state object, which retains contradictions by construction.
    assert verdicts["conflict_retained_rate"] != "discriminating"
    # Saturated: the zero-retrieval control reached it having bought nothing.
    assert verdicts["chosen_matches_reference"] == "saturated_by_a_trivial_policy"
    # The definitional collision with the rubric's own traversal.
    assert ledger["episode_readouts"]["decision_relevant_rate"]["per_packet"][
        "coronaviridae"]["verdict"] == "saturated_by_a_trivial_policy"
    # And the replacement passes the audit that condemned its predecessor, with its positive
    # control attained -- a probe no policy can score in the easy stratum is broken, not hard.
    twins_row = ledger["twin_readout"]["coronaviridae"]
    assert twins_row["verdict"] == "discriminating"
    assert twins_row["positive_control_attained"] is True
    assert twins_row["challenger_headroom"] > 0.3


def test_the_orbit_detects_a_planted_memoriser_and_not_a_name_blind_policy():
    # Both halves of the detector, because one without the other proves nothing. Run against
    # the fixed rubric the orbit returns exactly zero with zero spread: it does not invent an
    # identity effect where none can exist, since the rubric reads the evidence state and
    # never a target's name.
    #
    # The planted memoriser is the positive control -- a policy holding the answer key, which
    # is what a model that has seen the benchmark has. It is invisible to every other readout
    # in this project: it attains the ceiling on the graded probe, which is the best score
    # available, and looks like the strongest policy here. The orbit separates it, and
    # separates it exactly, to the name-blind baseline.
    #
    # The presentation orbit is what makes that reading sound. The memoriser *holds* its
    # ceiling when the state is merely rearranged, so the collapse under renaming is not
    # prompt sensitivity or sampling noise, which is the first objection to any such result.
    for name in ("coronaviridae", "orthomyxoviridae", "henipavirus"):
        blind = run_orbit(name, "rubric", count=2, attempts=200)["earned"]["challenger_best"]
        assert blind["attributable_to_identity"] == 0.0, name
        assert blind["identity_orbit"]["sd"] == 0.0, name

        planted = run_orbit(name, "memorised", count=2,
                            attempts=200)["earned"]["challenger_best"]
        assert planted["nominal"] == 1.0, name
        assert planted["presentation_orbit"]["mean"] == 1.0, name
        # Falls back to the fixed traversal once recall misses, so it lands on exactly the
        # name-blind policy's score rather than somewhere merely lower.
        assert planted["identity_orbit"]["mean"] == pytest.approx(blind["nominal"], abs=0.02), name
        assert planted["attributable_to_identity"] > 0.5, name
        assert planted["exceeds_variance_floor"] is True, name


def test_the_planted_memoriser_refuses_to_run_without_the_key_it_memorised():
    # It is a planted contamination, not a policy. Running it without the answer key would
    # silently make it the rubric, and the positive control would then report zero and be
    # read as the detector failing.
    states, _ = generate("henipavirus", 20261021, attempts=60)
    with pytest.raises(ValueError, match="answer key"):
        probe(states[0], "memorised")
