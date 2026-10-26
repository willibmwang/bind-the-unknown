import ast
import dataclasses
import json
import os
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from hpc_pilots.__main__ import main as cli_main
from hpc_pilots.arms import ARMS
from hpc_pilots import chief
from hpc_pilots import common
from hpc_pilots import emergency
from hpc_pilots import invariance
from hpc_pilots import twins
from hpc_pilots import finder
from hpc_pilots import optimizer
from hpc_pilots import surrogates
from hpc_pilots import tools
from hpc_pilots.binding import (design_matrix, enrichment, pick_batch, replay, run_arm,
                                run_binding)
from hpc_pilots.calibration import calibration_metrics, trajectory
from hpc_pilots.chemistry import alphabet, encode, library_spec, registry, sample_library
from hpc_pilots.common import ROOT
from hpc_pilots.common import (Agent, COMMITTED_MODEL_MANIFEST, STAGED_MODEL_MANIFEST,
                               content_digest, digest, model_manifest, provenance,
                               source_manifest)
from hpc_pilots.data import AA, encode_variants, pool_geometry
from hpc_pilots.dossier import ACTIONS, AXES, Dossier, evidence, leaders, rank
from hpc_pilots.emergency import decision_relevant, run_episode
from hpc_pilots.lattice import lattice, lattice_action
from hpc_pilots.monitor import audit_rule_robustness, packet_verdict, run_monitor
from hpc_pilots.monomers import (MODELS, _standardise, absence_flags, encode as encode_monomers, holdout,
                                 load, transfer)
from hpc_pilots.invariance import (IDENTIFYING, KINDS, LEXICON_FLOOR, Relabelling, certify,
                                   lexicon_floor,
                                   earned, group, leaks, lexicon, relabel_packet,
                                   residual_candidates,
                                   run_orbit,
                                   relabel_packet)
from hpc_pilots.twins import generate, grade_move, pair, probe, run_twins
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


def test_an_identity_effect_is_read_against_the_null_orbit_and_never_against_presentation():
    # The estimator was first defined as the identity orbit's shortfall against the
    # *presentation* orbit, reasoning that a policy which is merely unstable would otherwise
    # look like one reciting a name. That was wrong, and the guard in `improve` caught it: the
    # fixed rubric, which reads no names, reported a 0.18 identity effect under a wider
    # sampling band. The two transformations are orthogonal, not nested -- renaming holds the
    # order fixed, reordering holds the names fixed -- so their difference mixes two
    # sensitivities. The rubric walks the ranking and takes the first target, and `rank`
    # resolves ties in the caller's list order, so permuting the list genuinely changes what it
    # buys; differencing the orbits charged that to memorisation.
    #
    # Instability is controlled by the null orbit, which re-runs the identical instance.
    order_sensitive = earned(1.0, {"null": [1.0, 1.0], "presentation": [0.5, 0.5],
                                   "identity": [1.0, 1.0]})
    assert order_sensitive["attributable_to_identity"] == 0.0
    assert order_sensitive["attributable_to_presentation"] == pytest.approx(0.5)

    memorising = earned(1.0, {"null": [1.0, 1.0], "presentation": [1.0, 1.0],
                              "identity": [0.4, 0.4]})
    assert memorising["attributable_to_identity"] == pytest.approx(0.6)
    assert memorising["attributable_to_presentation"] == 0.0
    assert memorising["earned_mean"] == pytest.approx(0.4)
    assert memorising["exceeds_variance_floor"] is True

    # An effect inside the policy's own variance does not clear the floor.
    noisy = earned(1.0, {"null": [0.2, 1.0], "presentation": [0.6, 0.6],
                         "identity": [0.9, 0.9]})
    assert noisy["policy_variance_floor"] > 0.1
    assert noisy["exceeds_variance_floor"] is False

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
    # The tree ensemble is blind structurally, at any precision.
    paired = by_cell[("vocabulary", "forest")]["paired"]
    assert paired["pairs"] > 500, "too few discriminating pairs to pin the claim"
    assert paired["tied_rate"] == 1.0
    assert paired["tied_predictions"] == paired["pairs"]
    assert paired["concordance"] is None, "an all-tied arm has nothing to be concordant on"
    # The linear model on the same features is NOT structurally blind, and the first version
    # of this test asserted it was. Regularisation alone fixes the unseen column's coefficient
    # at something small and non-zero, so the predictions differ -- and at the float32 the
    # design matrix used to be built in, that difference rounded to zero and read as the tree
    # ensemble's blindness. At float64 the linear arm separates nearly every pair, by about
    # 1e-15: real arithmetic, no usable signal, and its concordance sits at chance.
    linear = by_cell[("vocabulary", "ridge")]["paired"]
    assert linear["tied_rate"] < 0.01, linear["tied_rate"]
    assert linear["smallest_separation"] < 1e-9, linear["smallest_separation"]
    assert abs(linear["concordance"] - 0.5) < 0.12, linear["concordance"]
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
    # Saturated because the fixed rubric reaches 1.0 with evidence, so nothing can be observed
    # above it. That is a separate finding from the zero-retrieval result, which is the *agent*
    # policy's: under that control the rubric and scripted policies decline to recommend and
    # report None, so they contribute nothing to the match rate and cannot saturate it. Both
    # say the readout is unusable; only one of them is what this audit measured.
    assert verdicts["chosen_matches_reference"] == "saturated_by_a_trivial_policy"
    # The definitional collision with the rubric's own traversal.
    assert ledger["episode_readouts"]["decision_relevant_rate"]["per_packet"][
        "coronaviridae"]["verdict"] == "saturated_by_a_trivial_policy"
    # The replacement attains its positive control, and does *not* pass the audit that
    # condemned its predecessor. This assertion read `== "discriminating"` until the trivial
    # set was completed, and the claim it encoded is withdrawn.
    #
    # The audit asked only whether the fixed rubric reached 1.0, so a trivial policy that
    # outscores every policy reading the state without touching the ceiling was invisible to
    # it -- which is the exact shape of the three failures this module was written from, one
    # level up. Completing the set and comparing against a declared reference rather than
    # against 1.0 moves the verdict to `led_by_a_trivial_policy` and empties
    # `discriminating_readouts`.
    #
    # What does the outscoring is `coverage_static`, which ranks targets by how many axes the
    # packet declares for each and reads no claim, no strength, no gate verdict, no ranking
    # and nothing that moves as evidence arrives. Two readings are available and the artifact
    # carries both: resolving a target with more open axes genuinely settles more of the
    # decision, so it is partly a sound heuristic; and the quantity it keys on is a record of
    # how much the Week 4 write-up had to say, which tracks published volume and is precisely
    # what a pretrained policy recalls for free. Only the second makes it a defect, and the
    # readout cannot tell them apart, which is why the verdict is not `discriminating`.
    twins_row = ledger["twin_readout"]["coronaviridae"]
    assert twins_row["verdict"] == "led_by_a_trivial_policy"
    assert twins_row["positive_control_attained"] is True
    assert twins_row["challenger_headroom"] > 0.3
    assert twins_row["trivial_outscores_reference"] is True
    assert ledger["discriminating_readouts"] == []


def test_surface_scoring_prefers_an_interface_to_a_pocket():
    """The inversion the module exists for, pinned from its artifact.

    The keyword rubric scores zero on six of the project's own eleven targets, and the first
    reading of that was that non-enzymatic targets fall outside scope. Seven of the eleven are
    interfaces or non-enzymatic surfaces, and the theory of change says macrocycles exist
    precisely to reach what small molecules cannot -- so a rubric biased against them is
    biased against the molecule class.

    The obvious repair, a druggability score, reproduces the bias through geometry: fpocket
    and its relatives are calibrated on deep enclosed small-molecule sites, and on pocket
    depth the main protease beats the spike receptor-binding domain. What is pinned here is
    that scoring the largest contiguous exposed hydrophobic patch inverts that, and that the
    inversion survives the one free parameter.
    """
    import json

    from hpc_pilots.common import ROOT

    path = ROOT / "runs/surface_markers.json"
    if not path.exists():
        pytest.skip("surface marker artifact absent")
    payload = json.loads(path.read_text())

    sensitivity = {row["contact_angstrom"]: row for row in payload["contact_cutoff_sensitivity"]}
    assert set(sensitivity) == {8.0, 10.0, 12.0}, sorted(sensitivity)

    # The direction must hold at every cutoff, or the result is an artefact of the cutoff and
    # the module is measuring its own parameter.
    for cutoff, row in sensitivity.items():
        assert row["gap"] > 0, (cutoff, row)
        assert row["mean_non_enzyme"] > row["mean_enzyme"], (cutoff, row)

    # And the specific comparison the argument rests on, at the chosen cutoff.
    chosen = sensitivity[10.0]
    by_target = {row["target"]: row for row in chosen["rows"]}
    assert by_target["Spike RBD"]["score"] > by_target["Mpro (nsp5)"]["score"], chosen
    assert by_target["Spike RBD"]["patch_area"] > by_target["Mpro (nsp5)"]["patch_area"]


def test_a_holdout_that_removes_nothing_raises():
    """The control that silently is not one.

    A name holdout removing nothing produces output identical to one that worked: the
    pathogen is still in the reference, every protein identifies at full confidence, and the
    run reads as a passing test of something never tested. Both failure modes here are real
    and were found while assembling benchmark families rather than imagined.

    NCBI titles the dengue reference "Dengue virus 2" and UniProt writes "Dengue virus type
    2", so the obvious string matches nothing. That one is caught. The other is not catchable
    by any string: holding out "West Nile virus" leaves Kunjin virus, West Nile lineage 1b at
    97.6% identity, whose name shares no substring with it. That is why clade holdout exists
    and why this test asserts the guard is a floor and not a guarantee.
    """
    from hpc_pilots.common import ROOT
    from hpc_pilots.intake import build_index, fetch_reference, read_fasta

    path = ROOT / "data/raw/reference/viral_reviewed.fasta"
    if not path.exists():
        pytest.skip("viral reference not staged")
    entries = read_fasta(fetch_reference(scope="viral"))

    # A string UniProt actually uses removes entries and is allowed through.
    _index, kept = build_index(entries, "Nipah virus")
    assert 0 < len(entries) - len(kept) < len(entries)

    # A plausible string UniProt does not use must raise rather than pass quietly.
    for absent in ("Dengue virus 2", "Severe acute respiratory syndrome coronavirus 3"):
        with pytest.raises(ValueError, match="removed no entries"):
            build_index(entries, absent)

    # The escape hatch stays available for callers that mean it, so the guard cannot become a
    # reason to stop holding anything out.
    _index, kept = build_index(entries, "Dengue virus 2", require_removal=False)
    assert len(kept) == len(entries)


def test_the_cascade_recovers_enzymes_by_homology_and_antigens_by_topology():
    """What survives past the cliff, and the split that matters more than the recovery.

    The holdout ladder found identification falling to zero one rank out. Two mechanisms get
    past it and they reach different proteins, which is the point. Conserved enzymes have
    genuine distant relatives: with all of Coronaviridae removed, both replicase halves are
    found through Tobaniviridae. Family-specific surface proteins have none, so no homology
    method reaches them at any sensitivity -- and those are the antibody targets. They are
    recovered from their own sequence instead, as a class rather than a name.

    The statistics are pinned alongside, because the threshold is what makes the two tiers
    separable. A raw alignment score cannot be compared between a 4,400-residue replicase and
    a 1,282-residue spike; the Karlin-Altschul expectation can, and at 1e-3 it admits the
    replicase at 1e-68 while rejecting the spike at 20 and the nucleoprotein at 0.35.
    """
    import json

    from hpc_pilots.common import ROOT
    from hpc_pilots.intake import bit_score, expectation, membrane_topology

    # Pure arithmetic, so this half runs with no artifact and no network.
    assert round(bit_score(655)) == 257
    assert expectation(655, 2595, 6_000_000) < 1e-50
    assert expectation(62, 1282, 6_000_000) > 1.0        # spike against a non-relative
    assert 0.1 < expectation(73, 419, 6_000_000) < 1.0   # nucleoprotein, also rejected

    # A polyprotein must not read as a surface antigen however hydrophobic it is. The
    # SARS-CoV-2 replicase 1a carries twenty-four membrane segments and reads as surface on
    # hydropathy alone; the length guard is what stops it.
    assert membrane_topology("M" + "LVILVILVIL" * 250)["plausible_antigen_length"] is False

    path = ROOT / "runs/identification_cascade.json"
    if not path.exists():
        pytest.skip("cascade artifact absent")
    payload = json.loads(path.read_text())
    rows = payload["coronaviridae_removed"]["rows"]
    by_tier = {}
    for row in rows:
        by_tier.setdefault(row["tier"], []).append(row)

    # Both replicase halves, through remote homology, at genuine significance.
    remote = by_tier.get("remote") or []
    assert len(remote) == 2, remote
    assert all(row["ev"] < 1e-9 for row in remote), remote
    assert all("eplicase" in row["name"] for row in remote), remote

    # The spike, by topology, with no homolog at all.
    topology = by_tier.get("topology") or []
    assert len(topology) == 1, topology
    assert topology[0]["len"] == 1282, topology
    assert "surface antigen" in topology[0]["name"]
    # Scored below every homology hit, so a class never outranks a name.
    assert topology[0]["score"] < min(row["score"] for row in remote)


def test_identification_falls_off_a_cliff_one_rank_out():
    """The operating envelope, pinned from its artifact.

    The validation beside this one holds the pathogen out by name and recovers every target.
    That reads as a strong control and is a weaker one than it looks: ICTV places SARS-CoV,
    SARS-CoV-2 and three bat coronaviruses in a single species, so a name holdout leaves four
    conspecific relatives in the reference.

    Taking the holdout up the ranks finds the limit, and the limit is not where taxonomy would
    put it. Nipah survives a species holdout because Hendra is a separate species in the same
    genus; SARS-CoV-2 does not, because its species already contains its closest relatives.
    The pipeline's constraint is sequence identity and rank is only a proxy for it.

    Two properties are asserted because a later change could quietly lose either. The fall is
    a cliff -- one rung out, at most one ORF is identified and no expected target is found,
    rather than a ranking that degrades. And the failure is empty rather than wrong: nothing
    identified means no targets returned, not a confident list of incorrect ones, which for an
    emergency tool is the failure worth having.
    """
    import json

    from hpc_pilots.common import ROOT

    path = ROOT / "runs/holdout_ladder.json"
    if not path.exists():
        pytest.skip("holdout ladder artifact absent")
    ladders = json.loads(path.read_text())["ladders"]

    cov = {rung["rank"]: rung for rung in ladders["SARS-CoV-2"]["rungs"]}
    nipah = {rung["rank"]: rung for rung in ladders["Nipah virus"]["rungs"]}

    # Succeeds with conspecific relatives present, fails as soon as they are removed.
    assert cov["isolate (name match)"]["expected_targets_found"] == 5
    assert cov["species"]["expected_targets_found"] == 0, cov["species"]
    assert cov["species"]["orfs_identified"] <= 1, cov["species"]

    # Nipah's congeneric relative is a different species, so it survives one rung further.
    assert nipah["species"]["expected_targets_found"] == 3, nipah["species"]
    assert nipah["genus"]["expected_targets_found"] == 0, nipah["genus"]

    # The break is at a different rank in each family, which is the finding: taxonomy is a
    # proxy for identity and not the thing itself. If this ever stops holding, the envelope
    # has changed and the claim above needs rewriting.
    assert cov["species"]["orfs_identified"] < nipah["species"]["orfs_identified"]

    # A cliff, not a slope, and empty rather than wrong.
    for rung in list(cov.values()) + list(nipah.values()):
        if rung["expected_targets_found"] == 0:
            assert rung["orfs_identified"] <= 1, rung


def test_intake_recovers_the_real_targets_of_three_virus_families():
    """The validation, read from its artifact rather than re-run.

    Re-running would need NCBI and UniProt and about a minute, which does not belong in a
    suite that has to pass offline. What is pinned is the recorded result and the conditions
    that make it mean anything: the pathogen was held out of the reference in every case, so
    identification proceeded through a relative rather than through the organism itself.

    The targets recovered are the ones with drugs or antibodies against them -- the main
    protease and spike of SARS-CoV-2, the fusion and attachment glycoproteins of Nipah,
    haemagglutinin and neuraminidase of influenza. A rubric that found them on coronaviruses
    and not on the other two would be tuned to coronaviruses, which is the failure this exists
    to detect.
    """
    import json

    from hpc_pilots.common import ROOT

    path = ROOT / "runs/intake_validation.json"
    if not path.exists():
        pytest.skip("validation artifact absent; run the intake validation to produce it")
    payload = json.loads(path.read_text())

    cases = {case["pathogen"]: case for case in payload["cases"]}
    assert set(cases) == {"SARS-CoV-2", "Nipah virus", "Influenza A PR8"}, sorted(cases)

    for pathogen, case in cases.items():
        # The control. Without it the run is a database lookup with a known answer.
        assert case["held_out"], pathogen
        for target, rank in case["expected_target_ranks"].items():
            assert rank is not None, (pathogen, target)
        # Identified through relatives, never through the pathogen itself.
        for organism in case["nearest_organisms"]:
            assert case["held_out"].lower() not in organism.lower(), (pathogen, organism)

    # `all_expected_in_top_8` is kept in the artifact for continuity and is NOT asserted,
    # because it cannot fail where it matters most: Nipah produced exactly eight candidates,
    # so the claim says only that its targets appear in the list at all. The rank-sum
    # probability replaces it and can fail -- and for Nipah it does.
    enrichment = {name: case["rank_enrichment"] for name, case in cases.items()}

    assert enrichment["SARS-CoV-2"]["beats_chance_at_05"] is True, enrichment["SARS-CoV-2"]
    assert enrichment["Influenza A PR8"]["beats_chance_at_05"] is True, enrichment["Influenza A PR8"]

    # Nipah is neither a pass nor a failure and is pinned as the third thing. Three targets
    # among eight candidates at ranks 1, 3 and 4 gives p = 0.071, but the best attainable p at
    # that size is 0.018, so even a perfect ranking would be weak evidence and the observed
    # value says almost nothing about the ranking. It was reported as a success twice and then
    # as a failure once; both overstate what eight candidates can show.
    nipah = enrichment["Nipah virus"]
    assert nipah["verdict"] == "not established (underpowered)", nipah
    assert nipah["adequately_powered"] is False, nipah
    assert nipah["best_attainable_p"] > 0.01, nipah

    # The two that are adequately powered must say so, or the verdict above is just a label
    # the underpowered case wears to avoid being a failure.
    for name in ("SARS-CoV-2", "Influenza A PR8"):
        assert enrichment[name]["adequately_powered"] is True, enrichment[name]
        assert enrichment[name]["verdict"] == "beats chance", enrichment[name]

    # The denominator decides the answer, so the rule must be recorded with it. Nipah's ranks
    # give p = 0.071 against eight candidates and p = 0.006 against seventeen with nothing
    # about the ranking changed, and counting unidentified reading frames as candidates would
    # rescue it by choosing the denominator that gives the wanted result.
    assert nipah["denominator_rule"] == "candidates identified and ranked by the pipeline"

    # The denominator must travel with the rank. SARS-CoV-2 produced 23 candidates and the
    # artifact stored only the top 8 rows, which is how the denominator went missing.
    assert cases["SARS-CoV-2"]["candidates_ranked"] == 23, cases["SARS-CoV-2"]
    assert cases["Nipah virus"]["candidates_ranked"] == 8

    # Influenza is segmented and must be read as eight records, not concatenated into one.
    # Joining them would manufacture reading frames across junctions that exist in no molecule.
    assert cases["Influenza A PR8"]["segments"] == 8
    assert cases["SARS-CoV-2"]["segments"] == 1


def test_six_frame_calling_recovers_the_coronavirus_proteome():
    """ORF calling, checked against products whose lengths are known exactly.

    Offline and deterministic: the genome is read from the staged reference if present and the
    test skips otherwise, so the suite never depends on NCBI being reachable.

    What is pinned is the part that must not drift. Nucleocapsid is 419 residues, membrane is
    222, ORF3a is 275, and a caller returning anything else has a frame or a stop-codon bug.
    ORF1ab is the interesting case and is pinned as a *failure*: the biological product is one
    7,096-residue polyprotein made by a -1 ribosomal frameshift, and six-frame translation
    returns it as 4,405 and 2,595 because no reader of stop codons can see a frameshift. That
    is a limitation of the method rather than of the implementation, and it is recorded here
    so a later reader does not mistake the two fragments for a bug or, worse, patch them
    together with coronavirus-specific special-casing in a step meant to be family-agnostic.
    """
    from hpc_pilots.common import ROOT
    from hpc_pilots.intake import orfs, translate

    # The genetic code, spot-checked at the three stops and one ordinary codon, because an
    # off-by-one in the table builder would shift every translated protein silently.
    assert translate("TAATAGTGA") == "***"
    assert translate("ATGGCC") == "MA"

    staged = ROOT / "data/processed/intake/NC_045512.2_heldout.json"
    cached = ROOT / "data/raw/reference/NC_045512.2.fasta"
    if not cached.exists():
        pytest.skip("SARS-CoV-2 genome not staged; run `python -m hpc_pilots intake`")
    seq = "".join(line.strip() for line in cached.read_text().splitlines()
                  if not line.startswith(">"))

    found = orfs(seq, minimum_aa=60)
    lengths = [o["length_aa"] for o in found]

    # Exact, because these are the products' real lengths.
    for expected in (419, 275, 222):
        assert expected in lengths, (expected, lengths[:12])

    # The frameshift, pinned as the known failure it is.
    assert 4405 in lengths and 2595 in lengths, lengths[:6]
    assert 7096 not in lengths, "a six-frame caller cannot see a -1 frameshift"

    # Every ORF starts at a methionine and contains no stop.
    for orf in found:
        assert orf["protein"].startswith("M")
        assert "*" not in orf["protein"]

    del staged


def test_the_count_channel_is_measured_rather_than_certified_away():
    """The decomposition that covers the channel the orbit cannot.

    `coverage_static` earns 0.75 on the hard stratum while reading nothing but how many claims
    each target carries, and the orbit reads exactly 0.0 for it on every nuisance kind --
    identity, presentation, axis order -- so the instrument built to find supplied advantage
    certifies it clean. Nothing in the group permutes claim counts, and nothing can: a
    nuisance has to leave the oracle's answer as the image of the original, and removing a
    claim changes what is substantiated and so changes the answer. The count channel is not
    certifiable away, so it is measured instead.

    Both ends are pinned, because one without the other proves nothing. The count-only policy
    must agree with itself on every probe, or the diagnostic is not measuring what it says.
    And a policy that reads the state must agree with it far less often, or the diagnostic
    cannot separate reasoning from counting and is just another saturated number.
    """
    import io
    from contextlib import redirect_stdout

    summaries = {}
    for policy in ("coverage_static", "lattice", "scripted"):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summaries[policy] = run_twins("coronaviridae", policy,
                                          attempts=200)["summary"]["challenger_best"]

    # Exactly one: the diagnostic reconstructs the count-only choice independently of the
    # policy branch that produced it, so anything below 1.0 means the two have drifted apart.
    assert summaries["coverage_static"]["count_only_agreement"] == 1.0, summaries

    # And it separates. The lattice reads the evidence state and lands far from the count-only
    # choice while scoring comparably, which is the whole point of reporting the two together.
    assert summaries["lattice"]["count_only_agreement"] < 0.4, summaries["lattice"]
    assert summaries["lattice"]["attainment"] > 0.3, summaries["lattice"]

    # A fixed traversal keyed on the target name has no reason to track counts at all.
    assert summaries["scripted"]["count_only_agreement"] < 0.2, summaries["scripted"]


def test_four_of_five_public_label_claims_are_not_binding_measurements():
    """The only independently verified axis, checked for the endpoint it actually cites.

    `public_labels` is the one axis whose claims were checked against something outside the
    project -- the ProteinGym archive on disk -- so it is the axis most likely to be quoted as
    settled. The check was of existence. Under the endpoint rule in force since 22 October,
    which admits measured binding affinity alone, four of the five populated claims cite a
    scan that measures something else: FRET and growth for Mpro, viral replication for the HA
    stem and the PA-PB1 interface, growth for the neuraminidase. Nine of the archive's 217
    assays are `selection_type: Binding` and only Starr's RBD scan is cited here.

    Pinned exactly rather than as a bound. If a claim is repaired the test should fail and the
    repair be recorded; if a sixth claim is populated it should fail and the endpoint be
    stated. Skipped without the archive, because "could not check" and "nothing is wrong" are
    different findings and only one of them is true here.
    """
    from hpc_pilots.common import ROOT
    from hpc_pilots.monitor import audit_public_label_endpoints

    if not (ROOT / "data/raw/proteingym_metadata.csv").exists():
        pytest.skip("ProteinGym metadata not staged; run `python -m hpc_pilots prepare`")

    report = audit_public_label_endpoints()
    assert report["checked"] is True
    assert report["populated_claims"] == 5, report["populated_claims"]
    assert report["claims_on_a_binding_endpoint"] == 1, report
    assert report["claims_on_another_endpoint"] == [
        "HA stem (HA2 helix A)", "Mpro (nsp5)", "NA catalytic site", "PA-PB1 interface"], report

    by_target = {row["target"]: row for row in report["claims"]}
    # The one that does measure binding, and the regex that finds it. Starr's scan ships with
    # a `_binding`/`_expression` suffix after the year, so a pattern stopping at the year
    # matches neither half and the audit would report the claim as citing nothing resolvable.
    spike = by_target["Spike RBD"]
    assert spike["all_cited_are_binding"] is True, spike
    assert "SPIKE_SARS2_Starr_2020_binding" in spike["cited"], spike["cited"]

    # ProteinGym records no selection_type at all for the Mpro scan, which is a third state
    # distinct from binding and from fitness, and it must not read as binding by default.
    mpro = by_target["Mpro (nsp5)"]
    assert mpro["endpoints"][0]["selection_type"] is None, mpro
    assert mpro["all_cited_are_binding"] is False, mpro

    # Every claim states its own endpoint now, so a reader who never runs the audit still sees
    # it. The strengths are untouched on purpose: demoting them would move the ranking on a
    # scope decision rather than on a measurement.
    for target in report["claims_on_another_endpoint"]:
        claim = by_target[target]
        assert claim["strength"] in ("strong", "moderate"), claim


def test_padding_absent_axes_moves_the_leak_rather_than_closing_it():
    """The obvious repair, and the reason it is not adopted.

    Equalising axis coverage is the repair that suggests itself once `coverage_static` is
    seen to outscore every policy that reads the state, and it is structurally sound: `assess`
    already answers `unknown` for an axis with no entries and `unknown` never substantiates,
    so padding leaves `readiness` and the ranking untouched and changes only what is offered.

    It does not work, and the measurement is in `runs/monitor/coverage_channel.json`. Padded,
    every target carries seven entries, the per-target total is constant, and the policy falls
    through to the first-listed dossier -- which is the reference-best target in all three
    packets as authored. Challenger attainment on Coronaviridae moves 0.7652 to 0.7344, and on
    Henipavirus it *rises*, 0.6095 to 0.6761. The orbit shows where it went: unpadded,
    `attributable_to_presentation` is 0.0; padded, it is 0.4279.

    What is pinned here is the cheap structural half -- that padding equalises the counts and
    leaves the ranking alone -- because that is what makes the expensive half interpretable.
    If padding ever changed a ranking, the comparison in that artifact would be between two
    different decision problems and the numbers would not mean what they are reported to mean.
    """
    from hpc_pilots import targets as targets_module

    before = targets_module.PAD_ABSENT_AXES
    try:
        shapes, rankings = {}, {}
        for pad in (False, True):
            targets_module.PAD_ABSENT_AXES = pad
            _spec, dossiers, entries, _probes = targets_module.packet("coronaviridae")
            counts = {}
            for target, _entry in entries:
                counts[target] = counts.get(target, 0) + 1
            for dossier in dossiers:
                for target, entry in entries:
                    if target == dossier.target:
                        dossier.observe(entry, 8)
            shapes[pad] = sorted(counts.values())
            rankings[pad] = [row["target"] for row in rank(dossiers, 8)]
    finally:
        targets_module.PAD_ABSENT_AXES = before

    # Uneven as authored, constant when padded: the count channel is closed by construction.
    assert shapes[False] == [5, 5, 5, 5, 6, 7], shapes[False]
    assert shapes[True] == [7] * 6, shapes[True]
    # And nothing else moved, which is what makes the attainment comparison a fair one.
    assert rankings[False] == rankings[True], (rankings[False], rankings[True])
    # The first-declared target is the one padding hands a fixed-order policy, and it is the
    # reference-best. That is the channel the advantage relocates into.
    assert rankings[False][0] == "Mpro (nsp5)", rankings[False]

    # The default is unpadded, so every recorded artifact means what it says.
    assert targets_module.PAD_ABSENT_AXES is False


def test_axis_coverage_is_not_a_structure_count_proxy():
    """The withdrawal, pinned so it cannot be quietly reasserted.

    `coverage_static` outscores every policy that reads the state, and that measurement is
    pinned elsewhere. It was introduced as a *research-attention* control, on the reasoning
    that a target with more declared axes is one more had been published about. Checked
    against the per-target deposited-structure counts, which is a standard attention proxy
    and the one this project now fetches with dates attached, the two are uncorrelated.

    Spike RBD carries 2,099 structures and five axes; the neuraminidase accession the packet
    cites carries none and six. Whatever the coverage policy exploits, this proxy is not it,
    and a docstring claiming otherwise should fail here rather than be believed.

    The test is skipped rather than failed when the structural table has not been built, so a
    clean checkout without a network does not report a withdrawal it could not check.
    """
    from scipy.stats import spearmanr

    from hpc_pilots.common import ROOT

    path = ROOT / "data/processed/structures/manifest.json"
    if not path.exists():
        pytest.skip("structural table not built; run `python -m hpc_pilots structures`")
    structures = json.loads(path.read_text())["targets"]

    axes, counts = [], []
    for spec in PACKETS.values():
        for name, (_pathogen, _compartment, claims) in spec["targets"].items():
            if name not in structures:
                continue
            axes.append(len({c[0] for c in claims}))
            counts.append(structures[name]["structures"])

    assert len(axes) == 11, len(axes)
    rho = spearmanr(axes, counts)
    # Deliberately a bound on the magnitude rather than on the p-value. With eleven targets
    # the test is underpowered and a null p-value would pass for a correlation of 0.5; what
    # the withdrawal rests on is that the estimate itself is near zero.
    assert abs(rho.statistic) < 0.35, (rho.statistic, rho.pvalue, list(zip(axes, counts)))


def test_two_of_the_seven_axes_cannot_gate_a_decision():
    """The state audit, pinned on the two axes that fail it today.

    `essentiality` is `strong` on all eleven targets, so it separates nothing. `accessibility`
    is recoverable from `compartment` exactly -- every cytosolic target weak, every
    extracellular one strong -- and `compartment` is already in the state the policy reads,
    so the axis can be resolved without being bought.

    Both matter for the readout the six-week plan puts first. Scoring which axis a policy
    chooses to resolve next, against which axis actually gates the decision, is scored over
    an action space of seven; two of those seven cannot gate anything, and a policy that
    picks them is not making the mistake the readout would record.

    The assertion is deliberately exact rather than a bound. If a third axis joins them the
    test should fail and the finding be rewritten, and if one is repaired it should fail and
    the repair be recorded -- which is the point of pinning it.
    """
    from hpc_pilots.monitor import audit_axis_redundancy

    rows = audit_axis_redundancy()
    uninformative = sorted(a for a, r in rows.items() if r["constant"] or r["recoverable_from"])
    assert uninformative == ["accessibility", "essentiality"], rows

    assert rows["essentiality"]["constant"] is True
    assert rows["essentiality"]["distinct_assessments"] == ["strong"]
    # Constancy short-circuits the predictor search; a constant axis must not be blamed on a
    # field that is not doing the work.
    assert rows["essentiality"]["recoverable_from"] is None

    assert rows["accessibility"]["constant"] is False
    assert rows["accessibility"]["recoverable_from"] == "compartment"
    assert rows["accessibility"]["mapping"] == {"cytosolic": "weak", "extracellular": "strong"}

    # The axes that do carry information are not swept up by the same rule.
    assert rows["reagent_tractability"]["recoverable_from"] is None
    assert rows["conservation"]["recoverable_from"] is None


def test_counting_declared_axes_outscores_the_policies_that_read_the_state():
    """The dossier's shape is a research-attention proxy, and the twin probe pays for it.

    Axis coverage is not uniform across the packets: Mpro carries claims on all seven axes
    and Nipah's F glycoprotein on four, with the rest between. That spread records how much
    was known when the evidence was assembled rather than how good the target is, and
    `retrievable_axes` hands the shape to the policy directly, so counting it needs no claim.

    Two things are pinned here and the second is what makes the first a defect rather than a
    heuristic. The policy outscores the lattice on the hard stratum; and it does so with
    nothing attributable to identity or to presentation, so the score is not a name leak and
    not an order leak -- it is earned, by counting, which is the problem.
    """
    from hpc_pilots.invariance import run_orbit

    by_policy = {policy: run_twins("coronaviridae", policy, attempts=200)["summary"]
                 for policy in ("coverage_static", "lattice", "rubric")}
    hard = {p: v["challenger_best"]["attainment"] for p, v in by_policy.items()}
    assert hard["coverage_static"] > hard["lattice"], hard
    assert hard["coverage_static"] > hard["rubric"], hard

    earned = run_orbit("coronaviridae", "coverage_static", count=2, attempts=200,
                       kinds=("null", "identity", "presentation"))["earned"]["challenger_best"]
    assert earned["attributable_to_identity"] == 0.0, earned
    assert earned["attributable_to_presentation"] == 0.0, earned


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


def test_the_finding_survives_a_decision_rule_this_project_did_not_author():
    # The strongest objection available to the whole decision arm, raised independently by
    # three of six architecture proposals: the seven axes, the choice of hard gate and the
    # lexicographic ranking key were all written here, so "the best purchase" means best with
    # respect to a rule of our own and the readout may measure conformity to it.
    #
    # The answer is to vary the rule. Each variant is a different decision problem and the
    # oracle moves with it by design, so the variants are never averaged; the question is
    # whether the ordering survives, not whether the numbers do.
    audit = audit_rule_robustness("coronaviridae", attempts=200)
    assert audit["variants_evaluated"] >= 6
    low, high = audit["drop_range"]
    assert low > 0.4, audit["drop_range"]
    assert audit["drop_holds_everywhere"]
    # And the detector's sensitivity is rule-independent: the planted memoriser attains the
    # ceiling under every rule, so the orbit has something to detect in all of them.
    assert audit["planted_memoriser_range"] == [1.0, 1.0]
    # What is rule-dependent is only whether the rubric reaches *exactly* 1.0 in the easy
    # stratum, which is a coincidence of the authored gate -- its fixed axis order happens to
    # match the oracle's optimum there. The measurement does not rest on it, and asserting it
    # would be asserting the coincidence.
    assert len(audit["positive_control_exact_in"]) < audit["variants_evaluated"]
    assert audit["verdict"] == "robust_to_the_rule"


def test_the_lead_verdict_does_not_depend_on_the_ranking_key_having_three_terms():
    # `leaders` unpacked the ranking key positionally, which bound it to the key's arity and
    # raised for a two-term rule. That matters because varying the key is how the project
    # shows a finding is not an artifact of the rule it authored, so the coupling made the
    # robustness check impossible to run on the variant that mattered most.
    from hpc_pilots import dossier as dossier_module

    _, dossiers, entries, _ = packet("coronaviridae")
    by_target = {d.target: d for d in dossiers}
    for target, entry in entries:
        by_target[target].observe(entry, 8)
    authored = leaders(dossiers, 8)
    saved = dossier_module._rank_key
    try:
        dossier_module._rank_key = lambda d: (d.readiness()["hard_gates_met"],
                                              d.readiness()["substantiated_axes"])
        shorter = leaders(dossiers, 8)
    finally:
        dossier_module._rank_key = saved
    for verdict in (authored, shorter):
        assert set(verdict) == {"tied_for_lead", "hard_gates_met", "substantiated_axes",
                                "unknown_axes", "decided"}
        assert isinstance(verdict["unknown_axes"], int) and verdict["unknown_axes"] >= 0


def test_the_unseen_monomer_result_is_pinned_on_resampled_holdout_sets():
    # The unit of analysis is the held-out *set*: removing a monomer removes every member
    # containing it, so a split is one observation however many thousand rows it has. A single
    # draw per size reported a monotone decline in held-out Spearman and an ordering in which
    # the shuffled control beat the real descriptors which beat the vocabulary. Five draws
    # overturned all of it as draw noise. So both halves of the claim are pinned across draws,
    # not within one.
    draws = [transfer("thrombin_cyclic", 6, stream_int(20261021, "holdout-replicate", r),
                      record=False) for r in range(3)]
    sets = {tuple(d["held_out_monomers"]) for d in draws}
    assert len(sets) == 3, "the draws must differ or they are one observation repeated"

    for run in draws:
        by_cell = {(r["encoder"], r["model"]): r for r in run["results"]}
        # Structural for the tree ensemble, and so exact in every draw rather than on
        # average: no split can use a column with zero training variance.
        assert by_cell[("vocabulary", "forest")]["paired"]["tied_rate"] == 1.0
        # Empirical, and so a null: the real descriptors must not beat their own shuffle,
        # and both sit at chance. A reversal here is the positive result and must be read as
        # one rather than patched away.
        for model in MODELS:
            real = by_cell[("descriptor", model)]["paired"]["concordance"]
            shuffled = by_cell[("descriptor_shuffled", model)]["paired"]["concordance"]
            assert abs(real - 0.5) < 0.12, (model, real)
            assert real - shuffled < 0.12, (model, real, shuffled)


def test_the_floor_and_ceiling_are_recomputed_rather_than_read_from_the_state():
    # The largest hole an independent adversary found. `probe` verified the grades and trusted
    # the floor and the ceiling, so setting the recorded floor equal to the ceiling on the
    # states where the reference policy would beat the average sent those probes down the
    # `best <= chance` branch, `summarise` dropped them as unscored, and the surviving mean was
    # taken over the states where the policy does worse than chance. Headroom went from 0.66 to
    # 1.47 on one packet and 0.82 to 2.03 on another, with the yield, the strata and the
    # positive control all unchanged. A floor that is a function of the policy being scored is
    # circular.
    states, _ = generate("coronaviridae", 20261021, attempts=80)
    assert states
    tampered = {**states[0], "chance_grade": states[0]["best_grade"]}
    with pytest.raises(ValueError, match="floor and ceiling do not reproduce"):
        probe(tampered, "rubric")
    raised_ceiling = {**states[0], "best_grade": states[0]["best_grade"] + 1}
    with pytest.raises(ValueError, match="floor and ceiling do not reproduce"):
        probe(raised_ceiling, "rubric")


def test_the_grade_scale_is_pinned_because_attainment_divides_by_its_range():
    # The grade is ordinal; attainment divides by its range. An adversary rescaled the levels
    # -- shrinking the one the reference policy usually buys, stretching the one it never buys
    # -- and moved headroom from 0.66 to 1.14 with every yield, stratum, optimal rate and the
    # positive control bit-identical. No readout moved, so no readout could have caught it.
    from hpc_pilots import twins as twins_module

    states, _ = generate("coronaviridae", 20261021, attempts=80)
    saved = twins_module.grade_move
    try:
        twins_module.grade_move = lambda before, after: (
            0.0 if after == before
            else (8.0 if len(after["tied_for_lead"]) == 1 < len(before["tied_for_lead"]) else 2.0)
            if after["tied_for_lead"] != before["tied_for_lead"] else 0.05)
        with pytest.raises(ValueError, match="outside the declared scale"):
            probe(states[0], "rubric")
    finally:
        twins_module.grade_move = saved


def test_the_hard_stratum_holds_no_optimal_purchase_on_the_leader():
    # The boundary was `all optimal purchases on the leader` against anything else, which put
    # the mixed states -- some optimal purchases on the leader, some off it -- into the hard
    # stratum, where a policy examining only the leader can still land an optimal purchase. The
    # stratum was not purely hard and its rate was flattered: separating the mixed states moved
    # the rubric from 0.362 to 0.278 on one packet and 0.112 to 0.054 on another.
    states, counts = generate("coronaviridae", 20261021, attempts=300)
    assert counts["mixed_best"] > 0, "this packet must produce mixed states or the test is vacuous"
    for state in states:
        on_leader = state["optimal_on_leader"]
        total = len(state["optimal"])
        if state["state_family"] == "leader_best":
            assert on_leader == total, state["index"]
        elif state["state_family"] == "challenger_best":
            assert on_leader == 0, state["index"]
        else:
            assert 0 < on_leader < total, state["index"]
    # Mixed states are excluded from the paired design by count, not silently filtered.
    pairs, _ = pair(states, 20261021)
    assert all(p[f]["state_family"] == f for p in pairs for f in ("leader_best",
                                                                  "challenger_best"))


def test_the_identity_lexicon_cannot_shrink_and_cannot_name_a_packet_that_does_not_exist():
    # Two failures, one found by an adversary and one by the check it motivated.
    #
    # `leaks` audits with `lexicon(name)`, built from the same table `_codes` substitutes
    # from, so emptying the table makes the scrub and its own auditor vanish together and
    # certification passes with no surviving tokens while eleven names reach the state.
    # Checked against a committed floor instead.
    for name in PACKETS:
        floor, source = lexicon_floor(name)
        assert len(lexicon(name)) >= floor, (name, floor)
        assert source == "committed", (name, source)
        assert certify(name, group(name, "identity", 20261021, 1)[0])["certified"], name
    # And the table was first written with an entry for `filoviridae`, which is not a packet,
    # while `henipavirus` had none -- so that packet scrubbed five tokens and left `Nipah`,
    # `Hendra`, `Langya`, `Cedar` and the ephrin receptors in the state it called anonymised.
    # A table keyed by name does nothing when the name is wrong, and reports nothing.
    assert set(IDENTIFYING) <= set(PACKETS)
    assert "henipavirus" in IDENTIFYING


def test_the_residual_detector_does_not_report_its_own_opaque_codes():
    # `SUSPICIOUS` matches runs of capitals on word boundaries, and `-` is not a word
    # character, so it found the bare `TGT` inside every `TGT-01` while the filter tested the
    # whole `TGT-01` form and never fired. A clean run reported {"TGT": 9415}, so a real leak
    # would have arrived inside that noise -- which is the failure this function prevents.
    # With the codes stripped first, it found four genuine leaks: ProteinGym dataset
    # identifiers quoted in claim text, whose `9INFA` names the organism outright.
    for name in PACKETS:
        element = group(name, "identity", 20261021, 1)[0]
        spec, dossiers, entries, probes = relabel_packet(name, element)
        record = [spec, [d.state(8) for d in dossiers], [e for _, e in entries], probes]
        residual = residual_candidates(record)
        assert residual == {}, (name, list(residual)[:6])
        assert not any(token.startswith(("TGT", "ORG", "SRC", "TOK")) for token in residual)


def test_the_hard_stratum_reports_the_bound_a_leader_first_policy_cannot_beat():
    # An independent check found the fixed rubric attaining, in every probe, the maximum
    # available to any policy that only buys on a target tied for the lead -- 84 of 84 and 81
    # of 81. So the hard stratum's headline number was a structural ceiling reported as a
    # measurement of the rubric's reasoning. The ceiling is now recorded per state, which
    # turns the objection into the readout: the quantity to read is whether a policy clears
    # the bound, and a policy pinned on it has not looked past the leading target whatever its
    # rationale says.
    for name in ("coronaviridae", "orthomyxoviridae", "henipavirus"):
        blind = run_twins(name, "rubric", attempts=250)["summary"]["challenger_best"]
        assert blind["attainment"] == blind["leader_first_ceiling"], name
        assert blind["exceeded_leader_ceiling_rate"] == 0.0, name
        planted = run_twins(name, "memorised", attempts=250)["summary"]["challenger_best"]
        assert planted["exceeded_leader_ceiling_rate"] == 1.0, name
        assert planted["attainment"] > planted["leader_first_ceiling"], name


def test_each_guard_catches_the_hack_it_exists_for():
    # The guard set's history is the point of this test. Measured against four hacks written
    # by the same author as the guard it caught four of four, which is what a guard measured
    # against its author's imagination always reports. An independent adversary, forbidden
    # from reading the guard, defeated it seven times out of eight -- the largest hole being
    # that the probe verified the grades and trusted the floor and the ceiling. After the
    # fixes those motivated, the full set of twelve is caught, and eight of the twelve were
    # ahead of the baseline on the objective when they were refused.
    #
    # Three representative hacks are exercised here rather than all twelve, because the full
    # sweep runs every guard against every candidate and takes minutes. Each of these is
    # caught by a different guard, and each is one no readout could have caught.
    from hpc_pilots import improve as improve_module

    wanted = {"HACK oracle_in_the_state": "state_is_oracle_free",
              "ADV targeted_pairing": "sample_is_unbiased",
              "ADV trimmed_attainment_mean": "summary_reproduces"}
    by_name = {c.name: c for c in improve_module.CANDIDATES}
    for name, guard in wanted.items():
        candidate = by_name[name]
        assert candidate.planted, name
        result = improve_module.evaluate(candidate, "coronaviridae", attempts=150,
                                         run_tests=False)
        assert not result["admitted"], name
        assert guard in result["failed_guards"], (name, result["failed_guards"])
    # And the unpatched instrument passes every guard, or the catch rate above would be a
    # guard that refuses everything.
    baseline = improve_module.evaluate(by_name["baseline"], "coronaviridae", attempts=150,
                                       run_tests=False)
    assert baseline["admitted"], baseline["failed_guards"]


def test_naming_the_surrogate_leaves_the_recorded_replay_bit_identical():
    # The forest was constructed inside the function every policy calls, so trying a different
    # surrogate meant editing that function. Making it a registry lookup is only safe if the
    # default path is unchanged, because otherwise a new surrogate's effect cannot be told from
    # the refactor's. Checked against a recorded artifact rather than against itself.
    recorded = json.loads((ROOT / "runs/binding/greedy_pool00_delay0.json").read_text())
    rerun = run_binding(0, "greedy", 0)
    assert [m["top_recall"] for m in recorded["metrics"]] \
        == [m["top_recall"] for m in rerun["metrics"]]
    assert [m["best_score"] for m in recorded["metrics"]] \
        == [m["best_score"] for m in rerun["metrics"]]
    assert [t["selected"] for t in recorded["trace"]] == [t["selected"] for t in rerun["trace"]]
    # Every config on disk omits the key, so every one of them resolves to the forest.
    for path in sorted((ROOT / "hpc_pilots/configs").glob("*.json")):
        spec = json.loads(path.read_text())
        blocks = [spec] + [v for v in spec.values() if isinstance(v, dict)]
        for block in blocks:
            assert surrogates.surrogate(block.get("surrogate")).name == "forest", path.name


def test_a_surrogate_declares_whether_its_spread_is_a_posterior():
    # The acquisition rule is mean + ucb_beta * sigma, which reads as a confidence multiplier
    # and is not one here: the forest's sigma is the dispersion of its per-tree predictions.
    # That is a reasonable thing to acquire on and a different thing from what the formula
    # appears to say, so which it is travels with the model rather than with whoever reads the
    # number.
    forest = surrogates.surrogate("forest")
    assert forest.spread_is_posterior is False
    assert "per-tree" in forest.spread_note
    assert forest.takes_design_matrix is True
    # An unregistered name raises rather than falling back to the default, because a typo in a
    # config silently selecting the forest would report a surrogate comparison that never ran.
    with pytest.raises(ValueError, match="Unknown surrogate"):
        surrogates.surrogate("tabpfn-v2")
    # And a surrogate whose dependency is absent names the dependency. The scientific
    # environment is deliberately free of torch, so this is the expected path there, and an
    # ImportError surfacing from inside round fifteen of a replay is not a diagnosis.
    needs_torch = surrogates.Surrogate(
        name="needs-torch", fit_predict=lambda *a: None, takes_design_matrix=False,
        spread_is_posterior=True, spread_note="posterior", requires=("torch",))
    surrogates.register(needs_torch)
    try:
        if needs_torch.available():
            with pytest.raises(RuntimeError, match="torch"):
                surrogates.surrogate("needs-torch")
        with pytest.raises(ValueError, match="already registered"):
            surrogates.register(surrogates.Surrogate(
                name="needs-torch", fit_predict=lambda *a: None, takes_design_matrix=False,
                spread_is_posterior=True, spread_note="other"))
    finally:
        del surrogates.SURROGATES["needs-torch"]


def test_clearing_the_leader_first_ceiling_is_a_diagnostic_and_not_a_skill_measure():
    # The rate at which a policy buys off the leading target has a chance level near a fifth,
    # because a policy working down a fixed list lands off the leader by accident. Measured on
    # the same states in the same run the state-blind scripted policy clears the ceiling about
    # as often as the hosted model did, so a non-zero rate licenses nothing on its own. This is
    # pinned because the rate was briefly read as evidence that the model reasons past the
    # leader, and it is not.
    blind = run_twins("orthomyxoviridae", "scripted", attempts=300)["summary"]["challenger_best"]
    assert blind["exceeded_leader_ceiling_rate"] > 0.1, blind["exceeded_leader_ceiling_rate"]
    # What separates them is the attainment, and the assertion is now a SEPARATION rather than
    # a position against zero. Bounding attainment below at -1 removes large negatives and
    # nothing else, so it raises a policy that chooses badly often: this arm moved from -0.063
    # to +0.104 when the bound was added. Zero stops meaning "the average purchase" once
    # clipping is common, so the honest claim is the gap between the policies.
    built = run_twins("orthomyxoviridae", "lattice", attempts=300)["summary"]["challenger_best"]
    assert built["attainment"] - blind["attainment"] > 0.4, (built, blind)
    assert blind["attainment"] < built["leader_first_ceiling"] + 0.2, blind["attainment"]
    fixed = run_twins("orthomyxoviridae", "rubric", attempts=300)["summary"]["challenger_best"]
    assert fixed["exceeded_leader_ceiling_rate"] == 0.0


def test_capping_the_probe_count_keeps_whole_pairs():
    # The cap exists to bound inference spend on an agent policy, which is what held the first
    # live orbit to two group elements with no interval on anything. Drawing states
    # independently to save that money would leave the two strata unbalanced by chance, which
    # is the matched design discarded for cost. Pairs are drawn instead, so the strata stay
    # balanced at every budget.
    for cap in (6, 12, 20):
        result = run_twins("henipavirus", "rubric", attempts=300, limit=cap)
        leader = result["summary"]["leader_best"]["probes"]
        challenger = result["summary"]["challenger_best"]["probes"]
        assert leader == challenger, (cap, leader, challenger)
        assert leader + challenger <= cap + 1, (cap, leader, challenger)
        assert leader > 0, cap


def test_a_packet_cannot_carry_two_base_entries_on_one_axis():
    # Every consumer builds the retrievable pool as {(target, axis): entry}, so a second entry
    # on the same pair silently replaces the first: the dossier observes both and `assess` sees
    # both, but only one is ever purchasable and the count of offered claims quietly disagrees
    # with the count authored. All eleven hand-written targets happen to satisfy this, which is
    # why it never surfaced -- measured, a duplicated axis turns five entries into four pool
    # keys. A generated packet has no such luck, so the invariant is enforced rather than
    # assumed.
    from hpc_pilots import targets as targets_module

    spec = targets_module.PACKETS["henipavirus"]
    target, (pathogen, compartment, axes) = next(iter(spec["targets"].items()))
    doubled = {**spec, "targets": {target: (pathogen, compartment,
                                            list(axes) + [axes[0]])}}
    targets_module.PACKETS["doubled-axis"] = doubled
    try:
        with pytest.raises(ValueError, match="two base entries"):
            packet("doubled-axis")
    finally:
        del targets_module.PACKETS["doubled-axis"]
    # The real packets satisfy it, so enforcing it changes nothing recorded.
    for name in PACKETS:
        _, _, entries, _ = packet(name)
        keys = [(t, e["axis"]) for t, e in entries]
        assert len(keys) == len(set(keys)), name


def test_the_identity_audit_has_a_floor_for_a_packet_nobody_curated():
    # `LEXICON_FLOOR.get(name, 0)` made the check `len(declared) >= 0` -- true for every
    # input -- so the guard was vacuous for every packet the table does not name, which is the
    # regression the floor exists to prevent. The three curated packets have zero slack against
    # their committed floors, so the omission was invisible while only they ran.
    from hpc_pilots import targets as targets_module

    for name in PACKETS:
        floor, source = lexicon_floor(name)
        assert source == "committed", name
        assert floor > 0

    targets_module.PACKETS["uncurated"] = {
        "family": "Fam", "pathogen": "Org", "prototype_rationale": "x",
        "targets": {"A": ("Org", "extracellular",
                          [("essentiality", "strong", "a claim", "src-one", True, 0)])}}
    try:
        floor, source = lexicon_floor("uncurated")
        assert floor > 0, "an uncurated packet must not fall back to a floor of zero"
        assert source.startswith("derived")
        # And the certificate says which guard ran, because a derived floor shrinks with the
        # table it audits and so cannot catch an emptied IDENTIFYING the way a committed one can.
        certificate = certify("uncurated", group("uncurated", "identity", 20261021, 1)[0])
        assert certificate["declared_tokens_floor_source"].startswith("derived")
        assert certificate["checks"]["lexicon_is_not_smaller_than_its_floor"]
    finally:
        del targets_module.PACKETS["uncurated"]


def test_the_scarcity_curve_is_read_against_a_ceiling_that_moves_with_it():
    # The brief committed to "budget-quality curves" and to varying evidence completeness
    # "separately", and it was never built: every figure averaged over the sampling band.
    #
    # The control is the point. Attainment is already scaled per state between the average
    # purchase on offer and the best one, but the leader-first ceiling moves with scarcity
    # too, so reporting only the policy's trend would read a change in the problem as a change
    # in the agent. For the fixed rubric the two trends are *identical*, because the rubric is
    # the ceiling -- which is the internal check that the control is wired to the right thing.
    curves = run_twins("orthomyxoviridae", "rubric", attempts=400)["summary"]["by_scarcity"]
    hard = curves["challenger_best"]
    assert hard["probes"] > 50
    assert hard["trend"]["rho"] == hard["ceiling_trend"]["rho"], "the rubric is the ceiling"
    # Not flat: scarcity is an operative variable rather than a framing device.
    assert abs(hard["trend"]["rho"]) > 0.1 and hard["trend"]["p"] < 0.05

    # A state-blind policy diverges from the ceiling, so there is something for the contrast
    # to measure rather than two restatements of the same quantity.
    blind = run_twins("orthomyxoviridae", "scripted",
                      attempts=400)["summary"]["by_scarcity"]["challenger_best"]
    assert blind["trend"]["rho"] != blind["ceiling_trend"]["rho"]

    # The bins span a real range of evidence availability and carry their own n.
    shares = [b["observed_share"] for b in hard["bins"]]
    assert len(hard["bins"]) >= 3
    assert shares[0][0] < shares[-1][1]
    assert all(b["probes"] > 5 for b in hard["bins"])


def test_the_leader_first_ceiling_decays_as_evidence_accumulates():
    # The direction is the opposite of the intuition the plan was written with, so it is
    # pinned rather than left to be re-derived. More observed evidence lowers the leader-first
    # ceiling: as the leading target's axes fill in, a policy that only buys on the leader
    # runs out of purchases that move anything. Measured, the headroom between the best and
    # the average purchase stays roughly flat across the band (1.27 -> 1.14 on this packet),
    # so the decay is in what the leader has left to offer, not in the problem's difficulty.
    #
    # The consequence for the framing: the data-poor regime is where leader-fixation is
    # cheapest, and the cost of a bad decision policy shows up as evidence arrives.
    states, _ = generate("orthomyxoviridae", 20261021, attempts=400)
    hard = [s for s in states if s["state_family"] == "challenger_best"]
    sparse = [s for s in hard if s["observed_share"] < 0.3]
    rich = [s for s in hard if s["observed_share"] > 0.5]
    assert len(sparse) > 20 and len(rich) > 20

    def mean(rows, key):
        return sum(key(r) for r in rows) / len(rows)

    headroom = (mean(sparse, lambda s: s["best_grade"] - s["chance_grade"]),
                mean(rich, lambda s: s["best_grade"] - s["chance_grade"]))
    assert abs(headroom[0] - headroom[1]) < 0.5, headroom
    # And the lead is already decided in this stratum at both ends, so the decay is not a
    # tie-breaking artifact.
    assert mean(sparse, lambda s: len(s["tied_for_lead"])) == 1.0
    assert mean(rich, lambda s: len(s["tied_for_lead"])) == 1.0


def test_the_smallest_checkpoint_is_recorded_before_any_policy_acts():
    # A fourth claim of the same genus as the three already withdrawn, and the reason this
    # test exists rather than a comment. The overview reports "the surrogate has no
    # discriminative power at the smallest budget -- at eight labels every policy returns
    # identical recall, 21 of 105 rows in the gain table". The identity is real and the
    # attribution is wrong: `snapshot()` is called before the acquisition loop opens, so the
    # eight-label row is the shared random initialisation recorded before any `pick_batch`
    # call has executed. Every policy agrees there by construction at every seed, which says
    # nothing whatever about the surrogate.
    #
    # The genuine smallest decision budget is 16 -- one batch of eight chosen by a policy,
    # with the surrogate fitted on eight labels.
    rng = np.random.default_rng(7)
    x = rng.integers(0, 20, size=(512, 4))
    y = rng.normal(size=512)
    shared = dict(maximum=32, checkpoints=(8, 16, 32), fraction=0.05)
    runs = {p: replay(x, y, p, 11, **shared) for p in ("random", "diverse", "greedy", "ucb")}

    # The first record predates the loop: round 0, while the first acquisition is round 1.
    for policy, out in runs.items():
        assert out["metrics"][0]["round"] == 0, policy
        assert out["metrics"][0]["labels"] == 8, policy
        assert out["trace"][0]["round"] == 1, policy
    # So it is identical across policies, and that is arithmetic rather than a measurement.
    first = {round(out["metrics"][0]["top_recall"], 10) for out in runs.values()}
    assert len(first) == 1, "the pre-loop snapshot must agree across policies"
    # And the acquisitions themselves do differ once a policy has acted, so the agreement
    # above is not evidence that the policies are the same.
    selections = {p: tuple(out["trace"][0]["selected"]) for p, out in runs.items()}
    assert len(set(selections.values())) > 1, selections


def test_a_landscape_staged_without_pools_says_so_instead_of_indexing_an_empty_split():
    # `pool_geometry` writes `disjoint` when it built pools and `pooled: False` when it could
    # not. The guard tested only the first, so the one landscape that genuinely cannot be
    # replayed fell through to indexing a (0, 0) split and died with a bare IndexError --
    # while its geometry carried a `reason` string written to explain precisely that.
    with pytest.raises(ValueError, match="without pools|does not exist"):
        run_arm("display", "il17c_bioactivity", 0, "greedy")
    # A pool beyond what was staged is named rather than traced back from an array error.
    with pytest.raises(ValueError, match="does not exist"):
        run_arm("display", "thrombin_cyclic", 999, "greedy")


def test_the_lattice_has_nowhere_to_put_a_target_name():
    # The architectural claim, and the only one here that needs no experiment. A target is
    # exactly its row of cell features; there is no target embedding, no index, no slot a name
    # could occupy. So renaming every target cannot change the state, and therefore cannot
    # change any function of it.
    element = group("coronaviridae", "identity", 20261021, 1)[0]
    table = element.table()
    # The observed set is chosen in base coordinates and translated, never re-sorted under the
    # new names. Sorting pool keys would sort by target name, so renaming would change which
    # claims are observed -- the same confound that made an earlier orbit report a 0.11
    # identity effect for a policy that reads no names.
    _, _, base_entries, _ = packet("coronaviridae")
    base_observed = sorted({(t, e["axis"]) for t, e in base_entries})[:11]
    built = {}
    for name, build in (("named", packet), ("renamed", lambda n: relabel_packet(n, element))):
        _, dossiers, entries, _ = build("coronaviridae")
        pool = {(t, e["axis"]): e for t, e in entries}
        observed = [(table.get(t, t), a) if name == "renamed" else (t, a)
                    for t, a in base_observed]
        by_target = {d.target: d for d in dossiers}
        for key in observed:
            by_target[key[0]].observe(pool[key], 3)
        built[name] = lattice(dossiers, pool, set(observed), 3)

    np.testing.assert_array_equal(built["named"][0], built["renamed"][0])
    np.testing.assert_array_equal(built["named"][1], built["renamed"][1])
    np.testing.assert_array_equal(built["named"][2], built["renamed"][2])
    # The order list is the one place a name appears, and it is used only to name the chosen
    # action -- never read by the score.
    assert built["named"][3] != built["renamed"][3]
    # And nothing in the state is a string at all.
    assert built["named"][0].dtype.kind == "f"


def test_the_lattice_policy_is_equivariant_to_the_order_the_targets_arrived_in():
    # `rank` resolves ties by the caller's list order, which is how the oracle acquired an
    # order-dependence that took 26-versus-28 relevant pairs to find. A representation with no
    # row order must not be able to reacquire it.
    #
    # The claim is equivariance **up to indistinguishability**, and that qualifier is the
    # honest version rather than a weakening. Measured on Coronaviridae, `nsp13 helicase`,
    # `Spike RBD` and `Spike S2 stem-helix` have byte-identical rows: the lattice cannot tell
    # them apart, so naming a different member of that class under rotation is correct
    # behaviour, and asserting a particular name would be asserting something the
    # representation deliberately cannot see. What must be invariant is the chosen cell's
    # content and axis.
    #
    # Tested on a state the generator kept, so the score actually discriminates. The first
    # version of this test used an arbitrary prefix of the pool, where every offered cell
    # scored 0.0 and all 22 were tied -- an uninformative state proves nothing about a
    # tie-break.
    states, _ = generate("coronaviridae", 20261021, attempts=200)
    discriminating = [s for s in states if s["state_family"] == "challenger_best"]
    assert discriminating, "no discriminating state to test equivariance on"
    state = discriminating[0]

    _, dossiers, entries, _ = packet("coronaviridae")
    pool = {(t, e["axis"]): e for t, e in entries}
    observed = [tuple(k) for k in state["observed"]]
    by_target = {d.target: d for d in dossiers}
    for key in observed:
        by_target[key[0]].observe(pool[key], state["decision_round"])

    def chosen(rows):
        action, target, axis = lattice_action(rows, pool, set(observed),
                                              state["decision_round"])
        cells, _, _, order = lattice(rows, pool, set(observed), state["decision_round"])
        content = tuple(cells[order.index(target), :, :].ravel())
        return action, axis, content

    baseline = chosen(dossiers)
    for shift in range(1, len(dossiers)):
        rotated = dossiers[shift:] + dossiers[:shift]
        assert chosen(rotated) == baseline, shift

    # And where the winning row's content is unique, the name is invariant too -- so the
    # qualifier above is not hiding an order-dependence.
    cells, _, _, order = lattice(dossiers, pool, set(observed), state["decision_round"])
    contents = [tuple(cells[r, :, :].ravel()) for r in range(len(order))]
    _, winner, _ = lattice_action(dossiers, pool, set(observed), state["decision_round"])
    if contents.count(contents[order.index(winner)]) == 1:
        for shift in range(1, len(dossiers)):
            rotated = dossiers[shift:] + dossiers[:shift]
            assert lattice_action(rotated, pool, set(observed),
                                  state["decision_round"])[1] == winner, shift


def test_the_lattice_score_does_not_depend_on_the_value_of_its_prior():
    # The prior is a constant multiplying every cell, so it scales the score and cannot move
    # the argmax. It is carried because the magnitude is reported, not because the choice turns
    # on it -- which is what stops it becoming a tuned knob, and is why it can be a measured
    # base rate (40 of 59 entries) rather than a fitted parameter.
    _, dossiers, entries, _ = packet("orthomyxoviridae")
    pool = {(t, e["axis"]): e for t, e in entries}
    observed = sorted(pool)[:6]
    by_target = {d.target: d for d in dossiers}
    for key in observed:
        by_target[key[0]].observe(pool[key], 3)
    chosen = {p: lattice_action(dossiers, pool, set(observed), 3, prior=p)
              for p in (0.1, 0.5, 0.678, 0.9, 1.0)}
    assert len(set(chosen.values())) == 1, chosen


def test_the_lattice_policy_clears_the_bound_on_the_larger_packets_and_not_the_smallest():
    # The measurement the architecture exists for, with the number it actually earns.
    #
    # Reported first as 0.899 / 0.933 / 0.365 against ceilings of 0.258 / 0.054 / -0.229. That
    # was wrong, and the axis-order nuisance is what found it: ties were broken on the column
    # index, so nearly a third of the attainment was the authored axis declaration order
    # rather than anything in the lattice -- a 0.291 axis-order effect against the fixed
    # rubric's 0.116. With ties drawn instead, the earned figures are 0.492 / 0.725 / -0.245.
    #
    # The advantage holds on the two larger packets and vanishes on the smallest, where two
    # targets leave barely a challenger to find. Reported per packet and never pooled.
    for name, floor in (("coronaviridae", 0.4), ("orthomyxoviridae", 0.6)):
        fixed = run_twins(name, "rubric", attempts=400)["summary"]["challenger_best"]
        built = run_twins(name, "lattice", attempts=400)["summary"]["challenger_best"]
        assert fixed["exceeded_leader_ceiling_rate"] == 0.0, name
        assert fixed["bought_on_leader_rate"] == 1.0, name
        assert built["attainment"] > floor, (name, built["attainment"])
        assert built["attainment"] > built["leader_first_ceiling"], name
        assert built["exceeded_leader_ceiling_rate"] > 0.5, (name, built)

    # Henipavirus, two targets: the advantage is gone and the test says so rather than
    # quietly excluding the packet that disagrees.
    small = run_twins("henipavirus", "lattice", attempts=400)["summary"]["challenger_best"]
    assert small["attainment"] < small["leader_first_ceiling"] + 0.1, small["attainment"]


def test_the_axis_order_nuisance_catches_a_policy_reading_a_list_order():
    # Added because the local model chose the first-declared axis in 39 probes of 40, and the
    # group as first written permuted the dossier order and not the axis order -- so it had no
    # way to see the failure that mattered most. The oracle counts substantiated axes and does
    # not read their order, so it is invariant, and `certify` checks that rather than assuming.
    for name in PACKETS:
        for element in group(name, "axis_order", 20261021, 2):
            certificate = certify(name, element)
            assert certificate["certified"], (name, certificate["checks"])
            assert len(certificate["element"]["axis_order"]) == len(AXES)

    # The fixed rubric walks the list and takes the first unresolved entry, so permuting the
    # list must cost it. Measured at +0.116 on Coronaviridae.
    walker = run_orbit("coronaviridae", "rubric", count=4, attempts=300,
                       kinds=("null", "axis_order"))["earned"]["challenger_best"]
    assert walker["attributable_to_axis_order"] > 0.05, walker

    # A policy that ignores the state entirely cannot be sensitive to how it is ordered.
    blind = run_orbit("coronaviridae", "scripted", count=2, attempts=300,
                      kinds=("null", "axis_order"))["earned"]["challenger_best"]
    assert blind["attributable_to_axis_order"] == 0.0, blind

def test_the_state_blind_policy_is_not_the_orbits_name_blind_control():
    # Two different controls that were briefly conflated, and the full orbit separated them.
    #
    # `scripted` ignores the decision state, so it is a sound control for the readout's
    # normalisation: a policy reading nothing must land near zero attainment, and it does.
    # But its traversal is `sorted(pool)`, which orders by target NAME -- so renaming the
    # targets reshuffles what it buys. Measured identity effects +0.046, -0.328 and +0.270
    # across the three packets, sign varying, which is reshuffling noise and not memorisation.
    # It is therefore NOT a control for the orbit.
    _, _, entries, _ = packet("coronaviridae")
    named = sorted({(t, e["axis"]) for t, e in entries})
    element = group("coronaviridae", "identity", 20261021, 1)[0]
    _, _, renamed_entries, _ = relabel_packet("coronaviridae", element)
    renamed = sorted({(t, e["axis"]) for t, e in renamed_entries})
    table = element.table()
    # The traversal is not the image of itself under the rename: position i holds a different
    # claim, which is exactly how a name reaches a policy that never reads one.
    translated = [(table.get(t, t), a) for t, a in named]
    assert translated != renamed, "the rename did not reorder the traversal"

    # The rubric is the orbit's name-blind control, and it is exact rather than approximate.
    for name in PACKETS:
        blind = run_orbit(name, "rubric", count=2, attempts=200,
                          kinds=("null", "identity"))["earned"]["challenger_best"]
        assert blind["attributable_to_identity"] == 0.0, name


def test_a_tool_spec_describes_the_function_it_names():
    # A spec is a claim about a function and nothing enforces it. The declared list can drift
    # from the signature it describes and every call still works, right up to the one that
    # does not -- and by then the caller has been trusting the spec for weeks. So the
    # conformance check is here rather than in a docstring.
    import inspect

    for name in sorted(tools.TOOLS):
        spec = tools.TOOLS[name]
        signature = inspect.signature(spec.resolve())
        for parameter in spec.parameters:
            assert parameter.name in signature.parameters, \
                f"{name} declares {parameter.name!r}, which is not in its signature"
            actual = signature.parameters[parameter.name]
            has_default = actual.default is not inspect.Parameter.empty
            # `required` is not a judgement about importance. It is whether the signature has
            # a default, and conflating the two is how a reader learns the spec was written
            # from the docstring rather than from the function.
            assert parameter.required is not has_default, \
                f"{name}.{parameter.name} declares required={parameter.required}"
            if has_default:
                assert parameter.default == actual.default, \
                    f"{name}.{parameter.name} declares a default the signature does not have"
        if spec.check:
            spec.resolve("check")


def test_an_unregistered_tool_raises_instead_of_resolving():
    # A typo in a spec name, or an artifact naming a tool that has since been renamed, must
    # stop rather than resolve to something. There is no sensible default tool, and a silent
    # substitution would be indistinguishable from the caller having asked for it.
    with pytest.raises(ValueError, match="Unknown tool"):
        tools.tool("intake.rank_target")
    with pytest.raises(ValueError, match="already registered"):
        tools.register(tools.Tool(
            name="intake.rank_targets", version=99, summary="s", description="d",
            entry="hpc_pilots.intake:run_intake"))
    with pytest.raises(TypeError, match="Tool record"):
        tools.register({"name": "not-a-record"})
    # An entry point naming an attribute that does not exist is a spec describing a function
    # nobody can call, which is worse than no spec because the caller trusts it.
    with pytest.raises(ValueError, match="no attribute"):
        tools.Tool(name="x", version=1, summary="s", description="d",
                   entry="hpc_pilots.intake:no_such_function").resolve()


def test_a_tool_whose_dependency_is_absent_names_the_dependency():
    # The scientific environment is deliberately free of torch. Checking declared imports
    # before dispatch means a missing backend names its cause, instead of surfacing as an
    # ImportError from inside the fifth round of an episode, where it is not a diagnosis.
    needs_torch = tools.Tool(
        name="needs-torch", version=1, summary="s", description="d",
        entry="hpc_pilots.chemistry:library_spec", requires=("torch",))
    tools.register(needs_torch)
    try:
        if needs_torch.available():
            with pytest.raises(RuntimeError, match="torch"):
                tools.tool("needs-torch")
    finally:
        del tools.TOOLS["needs-torch"]


def test_a_tool_that_produces_evidence_declares_where_its_date_comes_from():
    # The date boundary is the leakage boundary this project is built on: `Dossier.observe`
    # compares `available_from` against the decision clock. A tool that produced evidence
    # without saying which returned key dates it would enter a dossier at any clock, so the
    # obligation is checked at registration and not left to whoever writes the next tool.
    structures = tools.tool("structures.evidence")
    assert structures.produces_evidence is True
    assert structures.date_field == "earliest_deposition"
    assert structures.citation
    # The return is a table of eleven dated rows, not one dated object, so the date key lives
    # inside a row. The first version of this spec declared the per-row keys as top-level,
    # which a dispatcher would have read as None and dated every claim as undated.
    assert structures.rows == "targets"
    assert structures.rows in [key for key, _, _ in structures.returns]
    assert structures.date_field == "earliest_deposition"
    with pytest.raises(ValueError, match="produces no evidence"):
        tools.register(tools.Tool(
            name="undated-table", version=1, summary="s", description="d",
            entry="hpc_pilots.chemistry:library_spec", rows="targets",
            returns=(("targets", "dict", "m"),)))
    with pytest.raises(ValueError, match="produces evidence"):
        tools.register(tools.Tool(
            name="undated", version=1, summary="s", description="d",
            entry="hpc_pilots.chemistry:library_spec", produces_evidence=True))
    # A ranked target row is not a cited claim, and the registry says so rather than leaving a
    # caller to infer it from the absence of a date field.
    assert tools.tool("intake.rank_targets").produces_evidence is False
    assert "not a cited claim" in tools.tool("intake.rank_targets").citation


def test_a_parameter_vocabulary_that_varies_per_episode_is_bound_late():
    # A spec that hard-coded target names would survive `relabel_packet`, which rewrites packet
    # content and not source constants. The identity orbit would then read a renamed packet
    # while the model still saw the real names, and `attributable_to_identity` -- pinned at
    # exactly zero by two tests -- would stop meaning anything.
    late = tools.Parameter("target", "str", "Which target.", enum_from="target")
    assert late.values({"target": ["Mpro (nsp5)", "nsp12"]}) == ("Mpro (nsp5)", "nsp12")
    # A declared vocabulary the caller did not supply raises rather than widening to
    # unconstrained, because a missing vocabulary must not read as a permissive schema.
    with pytest.raises(ValueError, match="did not supply"):
        late.values({"axis": ["conservation"]})
    assert tools.Parameter("top", "int", "How many.").values() == ()
    with pytest.raises(ValueError, match="one vocabulary, not two"):
        tools.register(tools.Tool(
            name="two-vocabularies", version=1, summary="s", description="d",
            entry="hpc_pilots.chemistry:library_spec",
            parameters=(tools.Parameter("initiator", "str", "m", enum=("a",),
                                        enum_from="initiator"),)))


def test_the_committed_tool_snapshot_matches_the_registry():
    # The snapshot is a verifier and not a loader: nothing in the package reads it at runtime.
    # That is what makes an optimiser proposal structurally unable to reach a run -- the only
    # thing it can write is JSON, and no code path loads JSON into the registry. What the
    # snapshot buys instead is that editing a spec cannot go unnoticed.
    path = ROOT / "hpc_pilots" / "configs" / f"tools.v{tools.TOOL_SET_VERSION}.json"
    assert path.exists(), "the registry has no committed snapshot"
    committed = json.loads(path.read_text())
    assert committed["tool_set_sha256"] == tools.spec_set_digest()
    assert sorted(committed["tools"]) == sorted(tools.TOOLS)
    # Every earlier snapshot keeps its digest, so the tool_set_sha256 recorded in an already
    # finished run still verifies. A new spec goes in a new file; this is the device
    # configs/today.json already uses, applied to the registry.
    # Every earlier snapshot keeps its digest for ever, which is what lets a
    # tool_set_sha256 recorded in a finished run still be resolved to the specs it named.
    # v1 described the structures table's per-row keys as top-level; correcting that cut v2
    # rather than editing v1, which is the device configs/today.json already uses.
    historical = {1: "b67cdcfb525a0ff9f8beef3178a9dd8b549d7ea333bfeec9acdd372ad7aba509"}
    for version, expected in historical.items():
        earlier = ROOT / "hpc_pilots" / "configs" / f"tools.v{version}.json"
        assert json.loads(earlier.read_text())["tool_set_sha256"] == expected, \
            f"tools.v{version}.json moved; a run recording its digest can no longer resolve it"
    assert tools.TOOL_SET_VERSION not in historical or \
        committed["tool_set_sha256"] == historical[tools.TOOL_SET_VERSION]
    # The digest is over what the specs say and not over the file, because the file carries a
    # generation timestamp and re-cutting a byte-identical registry would otherwise hash
    # differently -- which is the defect content_digest was written for.
    assert tools.spec_set_digest() == tools.spec_set_digest()


def test_the_cli_reports_drift_between_the_registry_and_its_snapshot():
    """Editing a spec must turn the snapshot red rather than passing quietly."""
    import contextlib
    import io

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli_main(["tools", "--check"])
    assert json.loads(out.getvalue())["drift"] is False
    edited = tools.Tool(
        name="chemistry.library_spec", version=2,
        summary=tools.TOOLS["chemistry.library_spec"].summary,
        description="an edited description",
        entry="hpc_pilots.chemistry:library_spec")
    original = tools.TOOLS["chemistry.library_spec"]
    tools.TOOLS["chemistry.library_spec"] = edited
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
            cli_main(["tools", "--check"])
        assert json.loads(out.getvalue())["drift"] is True
    finally:
        tools.TOOLS["chemistry.library_spec"] = original
    # And the registry is listable without importing anything a tool needs, which is why
    # entry points are strings: asking what a tool costs must not cost biotite and pyrodigal.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli_main(["tools"])
    assert sorted(json.loads(out.getvalue())) == sorted(tools.TOOLS)


def test_an_emergency_artifact_names_the_tool_set_it_ran_against():
    # A result that cannot name its tool set cannot be compared with one run under a different
    # set, and a later spec edit would silently reinterpret it. The digest goes at the top
    # level beside config_sha256 and deliberately not into provenance(), whose key set is
    # asserted by exact equality elsewhere in this file and is embedded in every artifact.
    from hpc_pilots.emergency import run_emergency

    outputs = run_emergency("coronaviridae", "rubric", smoke=True)
    recorded = json.loads(Path(outputs[0]).read_text())
    assert recorded["tool_set_sha256"] == tools.spec_set_digest()
    assert "tool_set_sha256" not in recorded["provenance"]
    # Absence is meaningful and must not be read as an empty registry: the episodes recorded
    # before the registry existed carry no such key, and backfilling one would assert
    # something nobody can verify.
    before = json.loads((ROOT / "runs/emergency/coronaviridae_rubric.json").read_text())
    assert "config_sha256" in before and "tool_set_sha256" not in before


def test_the_absent_structure_backend_string_is_byte_identical_to_the_one_on_the_record():
    # Thirteen recorded artifacts carry this string and `structure_requests_unserviced` counts
    # it. It is also the discipline the whole tool layer inherits: a tool whose backend is
    # missing returns an explicit absence that is charged, rather than a plausible value for
    # something it did not compute. Rewording it would break the count and the discipline at
    # once, so it is pinned against the record rather than against a memory of the record.
    import ast

    expected = ("no structure backend configured; request recorded and charged, "
                "no structural claim produced")
    on_the_record = {r["result"]
                     for path in (ROOT / "runs/emergency").glob("*.json")
                     for r in json.loads(path.read_text()).get("trace", [])
                     if isinstance(r.get("result"), str)
                     and "no structure backend" in r["result"]}
    assert on_the_record == {expected}, \
        f"the recorded artifacts carry {on_the_record}, not the string this test pins"
    # And the source still emits exactly that. Adjacent literals are joined by the parser, so
    # reading the constants is robust where reassembling the source text is not.
    literals = {node.value
                for node in ast.walk(ast.parse((ROOT / "hpc_pilots/emergency.py").read_text()))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert expected in literals, "the no-backend string moved away from what the runs carry"


def test_an_argument_of_the_wrong_type_is_rejected_and_never_coerced():
    # Coercion is a silent fallback under another name. A "8" where an integer was declared
    # means the decoder did not honour the schema, and repairing it here would hide the one
    # failure this check exists to surface.
    with pytest.raises(chief.ToolCallRejected, match="must be int"):
        chief.schema_conformance(tools.tool("intake.rank_targets"),
                                 {"accession": "NC_045512.2", "top": "8"})
    with pytest.raises(chief.ToolCallRejected, match="requires 'accession'"):
        chief.schema_conformance(tools.tool("intake.rank_targets"), {"top": 8})
    # A flat schema carries a property for every shown tool, so a property the chosen tool
    # does not declare must arrive null rather than carrying another tool's value.
    with pytest.raises(chief.ToolCallRejected, match="declares no parameter"):
        chief.schema_conformance(tools.tool("chemistry.library_spec"), {"accession": "NC_045512.2"})
    assert chief.schema_conformance(tools.tool("intake.rank_targets"),
                                    {"accession": "NC_045512.2", "top": 8}).name == \
        "intake.rank_targets"


def test_true_is_not_accepted_for_an_integer_parameter_however_much_python_thinks_it_is():
    # isinstance(True, int) is True in Python and that is the wrong answer to this question. A
    # model emitting `true` for a count is not off by a type, it is answering a different
    # question, and accepting it would run the pipeline with top=1.
    with pytest.raises(chief.ToolCallRejected, match="must be int, got bool"):
        chief.schema_conformance(tools.tool("intake.rank_targets"),
                                 {"accession": "NC_045512.2", "top": True})
    assert chief._is_kind(True, "bool") is True
    assert chief._is_kind(True, "int") is False
    assert chief._is_kind(1, "bool") is False


def test_a_bound_the_schema_could_not_carry_is_charged_rather_than_raised():
    # The line between raising and charging sits where the emitted schema's expressive power
    # ends. A strict JSON schema cannot carry a numeric bound portably -- OpenAI's strict mode
    # drops `minimum` and both vLLM grammar backends ignore it -- so the model was never
    # constrained against one and a violation is a wrong guess, which is a policy result.
    spec = tools.tool("structures.evidence")
    assert chief.semantics(spec, {"minimum": 5.0}).startswith("minimum=5.0 is above")
    assert chief.semantics(spec, {"minimum": 0.5}) is None
    out = chief.call(spec, {"minimum": 5.0}, chief.context())
    assert out["status"] == "rejected" and out["citable"] is False
    assert "above the declared maximum" in out["note"]


def test_a_tool_whose_backend_is_absent_returns_the_no_backend_string_and_not_a_value():
    # The discipline `emergency.py` has followed for one action since the structure arm was
    # specified, promoted to a property of the dispatcher so it stays true for tools nobody
    # has written yet. An absent backend is recorded and charged; it does not produce a
    # plausible value for something it did not compute.
    needs_torch = tools.Tool(name="needs-torch", version=1, summary="s", description="d",
                             entry="hpc_pilots.chemistry:library_spec", requires=("torch",))
    out = chief.call(needs_torch, {}, chief.context())
    assert out["status"] == "unavailable"
    assert out["value"] is None and out["citable"] is False
    assert out["note"] == chief.NO_BACKEND.format(backend="needs-torch", claim="artifact")
    # A tool in the registry that nothing has wired to a dispatcher is the same situation and
    # must read the same way, rather than raising a KeyError from inside an episode.
    unwired = tools.Tool(name="unwired", version=1, summary="s", description="d",
                         entry="hpc_pilots.chemistry:library_spec")
    assert chief.call(unwired, {}, chief.context())["status"] == "unavailable"


def test_a_model_asserted_return_is_never_verified_and_never_citable():
    # Anything a model said cannot substantiate an axis. `assess` and `readiness` already
    # enforce that downstream; this makes it structural, so a tool cannot launder an assertion
    # into evidence by supplying a source and a date for it.
    spec = tools.tool("structures.evidence")
    asserted = chief.envelope(spec, {}, "ok", value=7, source_kind="model_assertion",
                              source="the model said so", available_from="2020-01-01")
    assert asserted["citable"] is False and asserted["verified"] is False
    # An in-house judgement may be cited and is still not verified, which is the distinction
    # `targets.INTERNAL` already encodes.
    internal = chief.envelope(spec, {}, "ok", value=7, source_kind="internal_assessment",
                              source="bench judgement", available_from="2020-01-01")
    assert internal["citable"] is True and internal["verified"] is False
    # And a computed return from a tool that declares no evidence is neither, because a
    # library definition is a design choice and not a finding about the world.
    design = chief.call(tools.tool("chemistry.library_spec"), {}, chief.context())
    assert design["status"] == "ok"
    assert design["citable"] is False and design["verified"] is False


def test_an_uncitable_success_must_declare_itself_rather_than_arrive_as_one():
    # A successful return with no declared source kind raises instead of degrading to
    # uncitable. Degrading would mean every tool with a provenance bug contributes quietly to
    # unverified_claims, and the rubric would read the bug as policy behaviour.
    spec = tools.tool("chemistry.library_spec")
    with pytest.raises(ValueError, match="must declare itself"):
        chief.envelope(spec, {}, "ok", value=1)
    with pytest.raises(ValueError, match="Unknown status"):
        chief.envelope(spec, {}, "finished", source_kind="computed")
    with pytest.raises(ValueError, match="Unknown source_kind"):
        chief.envelope(spec, {}, "ok", source_kind="vibes")
    # A claim that is citable but undated cannot enter a dated dossier, because observe()
    # compares available_from against the decision clock and an undated claim would pass at
    # any clock -- which is the one boundary the packet design exists to hold.
    with pytest.raises(ValueError, match="no source or no date"):
        chief.envelope(tools.tool("structures.evidence"), {}, "ok", value=1,
                       source_kind="external_record", source="PDBe")


def test_a_table_row_with_no_date_is_kept_and_marked_uncitable(monkeypatch):
    # One call on the structures table produces eleven claims, of which some are genuinely
    # undated: a target with no deposited structure has no earliest deposition. That is a real
    # zero and not a missing value, so the row is kept and marked uncitable. Dropping it would
    # make the table read as though every target had been structurally characterised.
    spec = tools.tool("structures.evidence")
    monkeypatch.setattr(
        "hpc_pilots.structures_evidence.build",
        lambda **kw: {"coverage_threshold": kw.get("minimum"),
                      "targets": {"dated": {"earliest_deposition": "2020-03-01",
                                            "structures": 4},
                                  "undeposited": {"earliest_deposition": None,
                                                  "structures": 0}}})
    out = chief.call(spec, {"minimum": 0.5}, chief.context())
    assert out["status"] == "ok"
    rows = {r["name"]: r for r in out["rows"]}
    assert set(rows) == {"dated", "undeposited"}
    assert rows["dated"]["citable"] is True
    assert rows["undeposited"]["citable"] is False
    assert rows["undeposited"]["value"]["structures"] == 0
    # The envelope is citable because some row is, and says which.
    assert out["citable"] is True


def test_a_tool_that_raises_is_recorded_as_failed_and_is_not_retried(monkeypatch):
    # Retrying a tool failure is resampling, which is the objection this project already
    # raises against retrying an invalid action. The failure is caught at exactly one place so
    # that no tool can smooth over its own, and the count is what reaches the record.
    calls = []

    def exploding(**kw):
        calls.append(kw)
        raise RuntimeError("the endpoint went away")

    monkeypatch.setattr("hpc_pilots.intake.run_intake", exploding)
    out = chief.call(tools.tool("intake.rank_targets"),
                     {"accession": "NC_045512.2"}, chief.context())
    assert out["status"] == "failed"
    assert "RuntimeError: the endpoint went away" in out["note"]
    assert out["value"] is None and out["citable"] is False
    assert len(calls) == 1, "a tool failure was retried, which is resampling"


def test_a_tool_with_no_policy_is_denied_rather_than_assumed_safe():
    # Default-deny, and both polarities, because a published capability system shipped a
    # policy whose is_trusted test was inverted -- it denied trusted ids and allowed untrusted
    # ones, failing open. A guard nobody has tried to break is an assertion, not a guard, and
    # asserting only the deny direction is how an inverted test passes its own suite.
    unpolicied = tools.Tool(name="writes.something", version=1, summary="s", description="d",
                            entry="hpc_pilots.chemistry:library_spec")
    assert isinstance(chief.check_policy(unpolicied, {}, chief.context()), chief.Denied)
    for name in chief.NO_SIDE_EFFECT:
        assert isinstance(chief.check_policy(tools.tool(name), {}, chief.context()),
                          chief.Allowed), f"{name} is on the no-side-effect list and is denied"
    # A user_only parameter must trace to the task statement, and must be allowed when it
    # does. The reason is redacted on the way back to the model, because a reason that quotes
    # a tool's bytes into the deciding model's context is itself a channel.
    gated = tools.Tool(
        name="gated", version=1, summary="s", description="d",
        entry="hpc_pilots.chemistry:library_spec",
        parameters=(tools.Parameter("initiator", "str", "m", required=False,
                                    default="ClAc-D-Tyr", trust="user_only"),))
    # With no trusted set this raises rather than denying for ever -- see
    # test_a_trusted_field_nobody_can_satisfy_fails_loudly_instead_of_denying_for_ever, which
    # is the adversary's finding. The denial path needs a trusted set that the value misses.
    supplied = chief.context(trusted=("asked-for",))
    denied = chief.check_policy(gated, {"initiator": "injected"}, supplied)
    assert isinstance(denied, chief.Denied) and denied.redact is True
    assert isinstance(
        chief._trusted_fields_policy(gated, {"initiator": "asked-for"}, supplied),
        chief.Allowed)
    # The gates compose, and the trust gate alone is not sufficient: a tool that traces its
    # arguments to the task and that nothing has decided about is still denied. Passing one
    # gate is not passing the set, which is what default-deny is for.
    assert isinstance(chief.check_policy(gated, {"initiator": "asked-for"}, supplied),
                      chief.Denied)


def test_a_tool_context_is_constructible_with_the_oracle_patched_to_raise(monkeypatch):
    # Structural rather than scored, and it catches a leak no readout could: handing a policy
    # the graded moves does not change what a deterministic policy does, so no number moves
    # and an objective would happily admit it. This is the check improve.py already applies to
    # the state, applied to what a tool is allowed to know.
    def refuse(*a, **k):
        raise AssertionError("the oracle must not be reachable from a tool context")

    monkeypatch.setattr("hpc_pilots.twins.graded_moves", refuse)
    monkeypatch.setattr("hpc_pilots.emergency.decision_relevant", refuse)
    built = chief.context(now=3, wall_budget_seconds=60, trusted=("Mpro (nsp5)",))
    assert set(built) == {"now", "wall_budget_seconds", "targets", "trusted"}
    assert built["now"] == 3 and built["trusted"] == frozenset({"Mpro (nsp5)"})


def test_no_dispatched_tool_writes_its_own_artifact_or_issues_a_model_request(monkeypatch):
    # A tool returns; the episode loop writes. The rule is offload's -- nothing writes inside
    # the container -- and it holds here for the same reason: a tool that wrote its own
    # artifact would put a second file on disk that no run names. A tool issuing a model
    # request would be a second LLM client spending budget nobody charged, which
    # ARCHITECTURE.md forbids outright.
    import inspect

    source = (ROOT / "hpc_pilots/chief.py").read_text()
    assert "write_json" not in source
    assert "import Agent" not in source and "from .common import Agent" not in source
    for name in chief.DISPATCH:
        spec = tools.tool(name)
        if "record" not in inspect.signature(spec.resolve()).parameters:
            continue
        seen = {}
        monkeypatch.setattr(spec.entry.replace(":", "."),
                            lambda **kw: seen.update(kw) or {"targets": {}})
        chief.call(spec, {}, chief.context())
        assert seen.get("record") is False, f"{name} was dispatched with record={seen.get('record')}"


def _two_action_tools():
    """Two tools declaring the same two parameters identically, both required.

    This is the configuration the recorded episodes ran under -- every action carried a target
    and an axis -- expressed as tool records rather than as two global fields.
    """
    shared = (tools.Parameter("target", "str", "Which target.", enum_from="target"),
              tools.Parameter("axis", "str", "Which axis.", enum_from="axis"))
    return [tools.Tool(name="retrieve_evidence", version=1, summary="s", description="d",
                       entry="hpc_pilots.chemistry:library_spec", parameters=shared),
            tools.Tool(name="stop", version=1, summary="s", description="d",
                       entry="hpc_pilots.chemistry:library_spec", parameters=shared)]


def test_a_tool_universe_with_every_parameter_required_emits_the_payload_it_emitted_before(
        monkeypatch):
    # The migration story, as a check and not a claim. When every shown tool declares every
    # parameter as required, no null branch is emitted and the payload is byte-identical to
    # the one the `fields` form produces -- which is what the recorded episodes ran under. If
    # this ever fails, the single-call flat encoding has bought nothing over a discriminated
    # union and the choice should be revisited.
    reply = ('{"action": "stop", "rationale": "ok", "target": "Mpro (nsp5)", '
             '"axis": "conservation"}')
    choices = {"target": ["Mpro (nsp5)", "nsp12"], "axis": ["conservation", "essentiality"]}

    sent = _mock_agent(monkeypatch, [(reply, "stop")])
    Agent().act("t", {"s": 1}, ["retrieve_evidence", "stop"], fields=choices)
    legacy = sent[-1]["json"]

    sent2 = _mock_agent(monkeypatch, [(reply, "stop")])
    Agent().act("t", {"s": 1}, ["retrieve_evidence", "stop"],
                tools=_two_action_tools(), choices=choices)
    assert sent2[-1]["json"] == legacy, \
        "an all-required tool set no longer replays the payload the recorded episodes used"


def test_a_nullable_parameter_enum_carries_null_because_a_decoder_cannot_emit_what_it_omits(
        monkeypatch):
    # With type ["string", "null"] and an enum that omits null, null is not a permitted value,
    # so a conforming decoder cannot emit the one thing this encoding needs it to emit. That
    # failure presents as a provider which "does not support schemas" rather than as a schema
    # bug, so it is pinned here.
    optional = tools.Tool(
        name="stop", version=1, summary="s", description="d",
        entry="hpc_pilots.chemistry:library_spec",
        parameters=(tools.Parameter("target", "str", "Which target.", required=False,
                                    default=None, enum_from="target"),))
    required = _two_action_tools()[0]
    sent = _mock_agent(monkeypatch, [('{"action": "stop", "rationale": "r", '
                                      '"target": null, "axis": "conservation"}', "stop")])
    Agent().act("t", {}, ["stop", "retrieve_evidence"], tools=[optional, required],
                choices={"target": ["Mpro (nsp5)"], "axis": ["conservation"]})
    properties = sent[-1]["json"]["response_format"]["json_schema"]["schema"]["properties"]
    assert properties["target"]["type"] == ["string", "null"]
    assert None in properties["target"]["enum"]
    # `axis` is required by one shown tool and absent from the other, so it is nullable too.
    assert properties["axis"]["type"] == ["string", "null"]


def test_two_tools_declaring_one_parameter_differently_raise_instead_of_merging():
    # The flat encoding's limit is not a tool count but the first real collision: whichever
    # property is emitted last would constrain the other tool's decoding, and only post-hoc
    # validation would catch it -- which is what constrained decoding exists to avoid. So it
    # raises, and names the remedy.
    from hpc_pilots.common import _union

    a = tools.Tool(name="a", version=1, summary="s", description="d",
                   entry="hpc_pilots.chemistry:library_spec",
                   parameters=(tools.Parameter("limit", "int", "m", maximum=10),))
    b = tools.Tool(name="b", version=1, summary="s", description="d",
                   entry="hpc_pilots.chemistry:library_spec",
                   parameters=(tools.Parameter("limit", "int", "m", maximum=1000),))
    with pytest.raises(ValueError, match="PILOT_TOOL_SCHEMA_MODE"):
        _union([a, b])
    # Identical declarations are not a collision, and the union keeps declaration order.
    same = tools.Tool(name="c", version=1, summary="s", description="d",
                      entry="hpc_pilots.chemistry:library_spec",
                      parameters=(tools.Parameter("limit", "int", "m", maximum=10),))
    assert [p.name for p in _union([a, same])] == ["limit"]


def test_an_unknown_tool_schema_mode_raises_the_way_an_unknown_provider_does(monkeypatch):
    # The PROVIDERS precedent: an unrecognised mode is a typo in an environment variable, and
    # silently running the default would report one encoding's numbers under another's name.
    _mock_agent(monkeypatch, [('{"action": "stop", "rationale": "r"}', "stop")])
    monkeypatch.setenv("PILOT_TOOL_SCHEMA_MODE", "discriminated")
    with pytest.raises(RuntimeError, match="PILOT_TOOL_SCHEMA_MODE"):
        Agent()
    # And a non-default mode writes its own artifact, for the reason the hosted label exists.
    monkeypatch.setenv("PILOT_TOOL_SCHEMA_MODE", "two_call")
    assert Agent().label.endswith("-two_call")
    monkeypatch.setenv("PILOT_TOOL_SCHEMA_MODE", "flat")
    assert Agent().label == "agent"


class _DesignToolAgent:
    """Chooses invoke_design_tool once, then recommends, so the tool branch actually runs.

    Needed because none of the five deterministic policies ever calls the design tool: only
    two recorded agent episodes did, so comparing the deterministic cells would have reported
    the branch unchanged without executing it once.
    """

    label = "agent-stub"
    filtered = 0
    transport_retries = 0
    incidents = []
    # Declared rather than defaulted by the episode. A `getattr(agent, "shows_tools", False)`
    # would hand a stub that forgot the flag the arm that pools with the record, which is the
    # silent fallback this repository treats as corruption.
    shows_tools = False

    def __init__(self):
        self.n = 0

    def act(self, task, state, allowed, fields=None, system=None, tools=None, choices=None):
        self.n += 1
        return {"action": "invoke_design_tool" if self.n == 1 else "recommend_experiment",
                "rationale": "stub", "target": state["ranking"][0]["target"],
                "axis": "conservation"}

    def report(self):
        return {"calls": self.n}


def test_routing_the_design_tool_through_the_caller_leaves_its_record_identical():
    # The one action that already called a tool now calls it through the caller, so it is
    # validated, policy-checked and carries provenance. Everything a recorded run counted has
    # to be untouched by that: `library` holds exactly what library_spec() returns,
    # `libraries_proposed` still counts entries, and the result string is byte-identical to
    # the one two recorded agent episodes carry.
    out = run_episode("coronaviridae", "agent", agent=_DesignToolAgent())
    entry = out["libraries"][0]
    assert entry["library"] == library_spec()
    assert out["metrics"]["libraries_proposed"] == 1
    design = [r for r in out["trace"] if r["action"] == "invoke_design_tool"][0]
    assert design["result"] == \
        "focused thioether macrocycle library proposed; no property predicted"
    # The recorded trace carried exactly these keys; anything new must be additive, because a
    # reader indexing the old shape must keep working.
    assert {"action", "axis", "pushed_evidence", "result", "round", "target"} <= set(design)
    # And the envelope says what the return is worth: computed honestly, and still not
    # evidence, because a library definition is a design choice and not a finding.
    provenance_ = entry["provenance"]
    assert provenance_["tool"] == "chemistry.library_spec"
    assert provenance_["status"] == "ok"
    assert provenance_["source_kind"] == "computed"
    assert provenance_["citable"] is False and provenance_["verified"] is False


def test_the_recorded_deterministic_cells_still_reproduce_every_number_they_recorded():
    # The guard against a refactor that is indistinguishable from a result. Every recorded
    # deterministic cell is replayed and compared field by field; `runs/` is gitignored, so
    # run_episode is called directly rather than through the CLI and no artifact is rewritten.
    checked = ("actions_used", "action_budget", "rounds_used", "deadline_rounds",
               "chosen_target", "chosen_matches_reference", "recommendation_issued",
               "deadline_met", "evidence_retrieved", "terminal_condition",
               "libraries_proposed", "structure_requests_unserviced",
               "uncited_axes_top_target", "axes_substantiated_top_target",
               "appropriate_update_rate_on_relevant")
    seen = 0
    for packet in PACKETS:
        for policy in ("rubric", "scripted", "coverage", "coverage_static"):
            path = ROOT / "runs/emergency" / f"{packet}_{policy}.json"
            if not path.exists():
                continue
            recorded = json.loads(path.read_text())["metrics"]
            fresh = run_episode(packet, policy)["metrics"]
            for key in checked:
                assert recorded.get(key) == fresh.get(key), \
                    f"{packet}/{policy}.{key}: recorded {recorded.get(key)!r}, " \
                    f"now {fresh.get(key)!r}"
            seen += 1
    assert seen, "no recorded deterministic cell was available to compare against"


def _finder_state(name, element=None):
    """A policy-facing state for the finder, optionally under a relabelling."""
    from hpc_pilots.invariance import relabel_packet

    spec, dossiers, entries, probes = \
        relabel_packet(name, element) if element is not None else packet(name)
    return {"decision_round": 0, "rounds_remaining": 8,
            "dossiers": [d.state(0) for d in dossiers],
            "retrievable_axes": [{"target": t, "axis": e["axis"]}
                                 for t, e in sorted(entries,
                                                    key=lambda p: (p[0], p[1]["axis"]))]}


def test_the_finder_shows_everything_at_this_scale_and_says_so_in_the_record():
    # 511 characters of action block against a 26,198-character state is 2.0%, and it would
    # take on the order of sixty-five tools before the block matched the dossiers beside it.
    # Below that a retrieval stage can only spend reachability to buy nothing, so the default
    # shows everything and writes that down rather than leaving it to be inferred.
    shown, record = finder.find(list(ACTIONS), ACTIONS, _finder_state("coronaviridae"))
    assert record["mode"] == "all"
    assert shown == list(ACTIONS) and record["withheld"] == []
    assert record["offered"] == len(ACTIONS)
    with pytest.raises(ValueError, match="Unknown finder"):
        finder.find(list(ACTIONS), ACTIONS, {}, mode="embeddings")


def test_the_finder_can_never_withhold_the_action_that_ends_an_episode():
    # A finder that hid `stop` would make `terminal_condition`, `deadline_met` and
    # `actions_unspent_at_exit` properties of the retriever, and one that hid
    # `recommend_experiment` would make `recommendation_issued` one. Both would be read as
    # policy behaviour, so this raises rather than being reported.
    def withholds_a_terminal(names, descriptions, state, limit, seed):
        return [n for n in names if n != "stop"], {"query": (), "note": "test stage"}

    finder.FINDERS["withholds"] = withholds_a_terminal
    try:
        with pytest.raises(ValueError, match="pinned"):
            finder.find(list(ACTIONS), ACTIONS, {}, mode="withholds")
    finally:
        del finder.FINDERS["withholds"]
    # Pinning holds through the floor too, which is the one stage that does withhold.
    shown, record = finder.find(list(ACTIONS), ACTIONS, _finder_state("coronaviridae"),
                                mode="shuffle", limit=4, seed=11)
    assert set(finder.PINNED) <= set(shown)
    assert len(shown) == 4 and len(record["withheld"]) == 2


def test_the_shown_set_does_not_move_under_a_relabelling():
    # The constraint that shapes the module. Two tests pin attributable_to_identity at exactly
    # zero for the rubric on all three packets, so a query built from dossier text would make
    # the shown set a function of the target names -- and even the rubric, which reads no
    # state, would have its reachable set move under a rename. The query is therefore drawn
    # from AXES and ASSESSMENTS, which relabel_packet does not touch, and sorted, because the
    # axis-order nuisance is a measured quantity and a query that read a list's order would be
    # reading it.
    from hpc_pilots.invariance import group

    for name in PACKETS:
        baseline = {mode: finder.find(list(ACTIONS), ACTIONS, _finder_state(name), mode=mode)[0]
                    for mode in ("all", "keyword", "tfidf")}
        for kind in ("identity", "presentation", "axis_order"):
            for element in group(name, kind, seed=7, count=2):
                state = _finder_state(name, element)
                for mode, expected in baseline.items():
                    shown, _ = finder.find(list(ACTIONS), ACTIONS, state, mode=mode)
                    assert shown == expected, \
                        f"{name}/{kind}/{mode}: the shown set moved under a relabelling"


def test_a_ranking_that_matched_nothing_shows_everything_rather_than_blacking_out():
    # Measured on the action space this is the common case, not the edge: the query is axis
    # names and the six action descriptions do not contain them, so keyword ranks nothing. The
    # conservative direction is to show everything -- which is what `all` does and what the
    # recorded runs already have -- because showing only the pinned pair would make
    # retrieve_evidence unreachable and change an episode's outcome for a reason that has
    # nothing to do with the policy. What must never be silent is the blackout.
    state = _finder_state("coronaviridae")
    for mode in ("keyword", "tfidf"):
        shown, record = finder.find(list(ACTIONS), ACTIONS, state, mode=mode)
        assert record["empty_ranking"] is True
        assert shown == list(ACTIONS), f"{mode} blacked out instead of failing open"
        assert record["withheld"] == []
    # An arm reporting empty_ranking on every round is a finder doing nothing, which is a
    # finding about the finder and has to be visible in the artifact to be one.
    assert finder.find(list(ACTIONS), ACTIONS, state)[1]["empty_ranking"] is False


def test_the_shuffle_floor_is_replayable_and_refuses_to_run_without_a_seed():
    # The floor exists so that a keyword stage which does not beat it at equal count is known
    # to be doing nothing -- the same move _coverage_action makes as a counting control. A
    # floor that could not be replayed would not be a control.
    state = _finder_state("coronaviridae")
    first, _ = finder.find(list(ACTIONS), ACTIONS, state, mode="shuffle", limit=4, seed=11)
    again, _ = finder.find(list(ACTIONS), ACTIONS, state, mode="shuffle", limit=4, seed=11)
    other, _ = finder.find(list(ACTIONS), ACTIONS, state, mode="shuffle", limit=4, seed=12)
    assert first == again
    assert first != other or len(ACTIONS) - len(finder.PINNED) < 3
    with pytest.raises(ValueError, match="needs a seed"):
        finder.find(list(ACTIONS), ACTIONS, state, mode="shuffle", limit=4)


def test_the_oracle_free_guard_sees_a_field_nested_inside_a_block():
    # The flat `set(built)` this replaced saw only the top level, which was sufficient while
    # every field the policy reads sat there. It stops being sufficient the moment the state
    # carries a nested block -- a per-tool parameter block is one -- because a field named
    # after the oracle would sit inside it and the check would pass while the leak was in the
    # state. A guard that cannot see where the state grew is not a guard.
    from hpc_pilots.improve import EVALUATOR_VOCABULARY, _every_key, guard_state_is_oracle_free

    nested = {"decision_round": 0,
              "available_tools": {"retrieve_evidence":
                                  {"parameters": [{"decision_relevant": 1}]}}}
    assert not EVALUATOR_VOCABULARY & set(nested), "the flat scan would have passed this"
    assert EVALUATOR_VOCABULARY & set(_every_key(nested)) == {"decision_relevant"}
    # Lists and tuples are walked too, since a parameter block is a sequence of records.
    assert "grade" in set(_every_key({"a": [{"b": ({"grade": 1},)}]}))
    # And the committed instrument is still clean under the stricter scan, so the change is a
    # strengthening and not a new finding about the state as it stands.
    clean, detail = guard_state_is_oracle_free()
    assert clean is True, detail


def test_the_certificate_scrubs_the_action_block_the_policy_actually_reads():
    # The thing scanned should be the thing the policy reads. The action block carries only
    # source constants today, so adding it found nothing across all twenty-four certificates
    # -- but it stops being a no-op the moment an action carries a parameter block whose
    # vocabulary is bound per episode, which is exactly when a target name could reach the
    # policy through a channel this check was not looking at.
    from hpc_pilots.invariance import certify, group, leaks, lexicon

    for name in PACKETS:
        for element in group(name, "identity", seed=7, count=1):
            certificate = certify(name, element)
            assert certificate["checks"]["no_declared_token_survives"] is True
    # The scan itself must be able to find a name planted in an action description, or the
    # addition to the scrub list is decorative.
    planted = {"retrieve_evidence": "Retrieve evidence; Mpro (nsp5) usually leads."}
    assert leaks([planted], lexicon("coronaviridae")), \
        "a target name planted in an action description was not detected"
    assert not leaks([{a: ACTIONS[a] for a in ACTIONS}], lexicon("coronaviridae"))


def test_each_proposal_guard_catches_the_edit_it_exists_for():
    # The method improve.py states: a guard nobody has tried to break is an assertion, not a
    # guard. The predictions are recorded before the run and are not tuned to match it, so a
    # hack caught by a different guard than predicted is reported as a finding about the guard.
    result = optimizer.run_optimize(smoke=True)
    assert result["planted_hacks"] == 6
    assert result["planted_hacks_caught"] == 6, result["planted_hacks_escaped"]
    assert result["planted_hacks_escaped"] == []
    assert result["guard_predictions_wrong"] == [], result["guard_predictions_wrong"]
    # And nothing is admitted, because nothing in the record could justify a proposal.
    assert result["proposals"] == [] and result["proposals_admitted"] == 0
    # A full catch rate on hacks written by the guards' own author is weak evidence, and the
    # artifact has to say so rather than leaving the number to speak for itself.
    assert "weak evidence" in result["interpretation"]


def test_the_optimizer_refuses_an_edit_no_recorded_field_supports():
    # The grader's own outputs are absent from SUPPORTED_BY on purpose: a spec edit justified
    # by chosen_matches_reference or decision_relevant_rate is Goodharting by construction.
    design = tools.tool("chemistry.library_spec")
    goodhart = optimizer.Proposal(
        tool="chemistry.library_spec", field="description",
        before=design.description, after=design.description + " Prefer this early.",
        evidence=(("runs/emergency/coronaviridae_rubric.json",
                   "metrics.chosen_matches_reference", True),),
        rationale="justified by the grader")
    passed, detail = optimizer.guard_evidence_is_recorded_not_inferred(goodhart)
    assert passed is False and "no recorded field supports" in detail
    # A summary edit is unsupported until retrieval is instrumented, and the refusal is a
    # statement about what is recorded rather than about the edit.
    bait = optimizer.Proposal(tool="chemistry.library_spec", field="summary",
                              before=design.summary, after="library design macrocycle tool")
    assert optimizer.guard_evidence_is_recorded_not_inferred(bait)[0] is False
    # Deleting a failure can never be supported: runs record occurrences, never absences.
    structures = tools.tool("structures.evidence")
    deletion = optimizer.Proposal(tool="structures.evidence", field="failures",
                                  before=structures.failures, after=())
    passed, detail = optimizer.guard_edit_is_within_the_proposable_set(deletion)
    assert passed is False and "add-only" in detail
    # And a field outside the proposable set is refused by name rather than evaluated.
    frozen = optimizer.Proposal(tool="structures.evidence", field="citation",
                                before=structures.citation, after="anything")
    assert optimizer.guard_edit_is_within_the_proposable_set(frozen)[0] is False


def test_the_optimizer_never_writes_outside_its_proposal_store(monkeypatch):
    # Propose-only, enforced rather than configured. The registry is read from source and this
    # module writes only JSON under runs/tools/, a tree nothing in the package reads -- so the
    # loop *cannot* apply an edit, which is a stronger claim than being set not to.
    written = []
    monkeypatch.setattr("hpc_pilots.optimizer.write_json",
                        lambda path, payload: written.append(str(path)))
    optimizer.run_optimize(smoke=True)
    assert written, "the optimizer wrote nothing at all"
    for path in written:
        assert "/runs/tools/" in path, f"the optimizer wrote outside its store: {path}"
    # The registry is untouched by a run, including by the planted edits it evaluates.
    assert tools.spec_set_digest() == \
        json.loads((ROOT / "hpc_pilots/configs"
                    / f"tools.v{tools.TOOL_SET_VERSION}.json").read_text())["tool_set_sha256"]


def test_a_proposal_authored_against_a_stale_spec_is_refused():
    # Otherwise it would be applied to a spec that has since moved, and the merge would carry
    # a `before` nobody can reconcile with what was registered.
    design = tools.tool("chemistry.library_spec")
    stale = optimizer.Proposal(tool="chemistry.library_spec", field="warning",
                               before="a value this spec has never held",
                               after=design.warning + " Measured.")
    passed, detail = optimizer.guard_proposal_is_current_and_moves_the_digest(stale)
    assert passed is False and "authored against" in detail
    # A proposal that changes nothing cannot move the digest, and a no-op merge would make an
    # already recorded tool_set_sha256 ambiguous about which specs it named.
    noop = optimizer.Proposal(tool="chemistry.library_spec", field="warning",
                              before=design.warning, after=design.warning)
    assert optimizer.guard_proposal_is_current_and_moves_the_digest(noop)[0] is False


def test_a_cost_claim_is_recomputed_from_the_calls_it_cites():
    # Cost deflation is the cheapest available hack in an action-budgeted arm: a tool that
    # claims to be free gets chosen. So the guard recomputes the statistic rather than
    # accepting the number, and refuses a claim resting on too small a sample.
    intake = tools.tool("intake.rank_targets")
    thin = optimizer.Proposal(tool="intake.rank_targets", field="cost_seconds",
                              before=intake.cost_seconds, after=2.0,
                              evidence=tuple(("runs/emergency/coronaviridae_rubric.json",
                                              "trace[].tool_seconds", 9.0) for _ in range(5)))
    passed, detail = optimizer.guard_cost_claim_is_measured(thin)
    assert passed is False and "floor" in detail
    enough = tuple(("runs/emergency/coronaviridae_rubric.json", "trace[].tool_seconds", 9.0)
                   for _ in range(optimizer.MINIMUM_CALLS_FOR_A_COST_CLAIM))
    wrong = optimizer.Proposal(tool="intake.rank_targets", field="cost_seconds",
                               before=intake.cost_seconds, after=2.0, evidence=enough)
    passed, detail = optimizer.guard_cost_claim_is_measured(wrong)
    assert passed is False and "recomputed median" in detail
    right = optimizer.Proposal(tool="intake.rank_targets", field="cost_seconds",
                               before=intake.cost_seconds, after=9.0, evidence=enough)
    assert optimizer.guard_cost_claim_is_measured(right)[0] is True


def test_a_zero_fidelity_says_whether_it_means_complete_or_uninstrumented():
    # A zero meaning "the spec predicts everything observed" and a zero meaning "nothing was
    # observed" are opposite situations, and reporting both as 0 would make an uninstrumented
    # tool look like a fully characterised one.
    for name in tools.TOOLS:
        measured = optimizer.fidelity(name)
        assert measured["fidelity"] == 0 and measured["observed"] == 0
        assert "uninstrumented" in measured["interpretation"]
    gap = optimizer.instrumentation_gap()
    assert gap["pointers_present"] == []
    assert not any(info["proposable"] for info in gap["fields"].values()), \
        "a field is advertised as proposable while the record is silent about it"


def test_an_agent_incident_does_not_license_a_failure_on_a_network_free_tool():
    # Found by the gap report rather than by review. The only incidents on record are
    # content_filter, which is a property of a provider's moderation and which no tool can
    # experience -- so eight recorded filters would otherwise have licensed a failure mode on
    # a local, network-free library-definition tool. The admissible set is an allowlist, not a
    # denylist: the first attempt excluded the LLM-only keys it knew of and missed `chars` and
    # `content_head`, so the field read as licensed.
    design = tools.tool("chemistry.library_spec")
    assert not design.is_remote() and not design.cost_network_calls
    borrowed = optimizer.Proposal(
        tool="chemistry.library_spec", field="failures", before=design.failures,
        after=design.failures + (tools.Failure("filtered", "the provider filtered it", ""),),
        evidence=(("runs/emergency/coronaviridae_agent-anthropic-claude-sonnet-5-5_rep01.json",
                   "agent.incidents[]", {"finish_reason": "content_filter"}),))
    passed, detail = optimizer.guard_evidence_is_recorded_not_inferred(borrowed)
    assert passed is False and "reaches no network" in detail


def test_a_trusted_field_nobody_can_satisfy_fails_loudly_instead_of_denying_for_ever():
    # Found by an adversary, not by review. No caller in this repository populates
    # context["trusted"], so a parameter declaring trust="user_only" would deny every call
    # supplying a value, for ever -- and redact=True means the policy sees only a refusal that
    # reads like its own mistake, so a harness defect presents as policy behaviour while the
    # policy keeps spending charged actions on it. A kill switch wearing the shape of a
    # hardening is exactly the proposal an optimiser would rate as a strict improvement.
    gated = tools.Tool(
        name="gated", version=1, summary="s", description="d",
        entry="hpc_pilots.chemistry:library_spec",
        parameters=(tools.Parameter("initiator", "str", "m", trust="user_only"),))
    with pytest.raises(RuntimeError, match="no trusted set"):
        chief.check_policy(gated, {"initiator": "anything"}, chief.context())
    # With a trusted set supplied it behaves as designed, in both directions.
    supplied = chief.context(trusted=("asked-for",))
    assert isinstance(chief._trusted_fields_policy(gated, {"initiator": "asked-for"}, supplied),
                      chief.Allowed)
    assert isinstance(chief._trusted_fields_policy(gated, {"initiator": "injected"}, supplied),
                      chief.Denied)


def test_proposed_text_may_not_introduce_the_retrieval_query_vocabulary():
    # finder._query is built from AXES and ASSESSMENTS so that no relabelling can move the
    # shown set. The consequence an adversary found is that prose drawn from the same
    # vocabulary moves it instead, and the identity orbit is blind to that by construction. The
    # scan is against AXES itself, because the attack that found this used no suspicious
    # language -- "accessibility" and "precedent" are ordinary English in a chemistry sentence.
    design = tools.tool("chemistry.library_spec")
    bait = optimizer.Proposal(
        tool="chemistry.library_spec", field="summary", before=design.summary,
        after=("Define a focused thioether-macrocycle library over a randomised region, given "
               "a target's accessibility and the precedent available for it."))
    passed, detail = optimizer.guard_text_avoids_the_retrieval_vocabulary(bait)
    assert passed is False
    assert "accessibility" in detail and "precedent" in detail
    # A summary that adds no axis vocabulary is untouched by this guard.
    plain = optimizer.Proposal(tool="chemistry.library_spec", field="summary",
                               before=design.summary,
                               after=design.summary + " Returns a definition only.")
    assert optimizer.guard_text_avoids_the_retrieval_vocabulary(plain)[0] is True


def test_a_failures_edit_may_extend_a_cause_and_may_not_rewrite_it():
    # add-only on the tuple does not stop an entry being replaced by one with the same symptom
    # and a narrower cause. An adversary used exactly that to turn "a holdout that silently
    # removed nothing would report leakage as generalisation" into a note about string
    # matching, which leaves a reader every reason to work around the error.
    intake = tools.tool("intake.rank_targets")
    original = intake.failures[0]
    narrowed = tools.Failure(original.symptom, "A spelling difference leaves entries in place.",
                             original.raises, original.observed_in)
    rewrite = optimizer.Proposal(tool="intake.rank_targets", field="failures",
                                 before=intake.failures,
                                 after=(narrowed,) + intake.failures[1:])
    passed, detail = optimizer.guard_a_replaced_failure_keeps_its_cause(rewrite)
    assert passed is False and "rewritten rather than extended" in detail
    extended = tools.Failure(original.symptom, original.cause + " Also observed for Nipah.",
                             original.raises, original.observed_in)
    widen = optimizer.Proposal(tool="intake.rank_targets", field="failures",
                               before=intake.failures,
                               after=(extended,) + intake.failures[1:])
    assert optimizer.guard_a_replaced_failure_keeps_its_cause(widen)[0] is True


def test_the_optimizer_records_what_an_adversary_got_past_it():
    # A guard set is worth what an adversary could not get past it. The author's own planted
    # set scored six of six; an independent adversary scored seven of twelve on the merits,
    # and one of its attacks defeats every guard here. That escape is recorded rather than
    # patched until the number looks better, and rather than widened into NOT_ADJUDICABLE --
    # which would convert a known hole into a claimed design decision.
    result = optimizer.run_optimize(smoke=True)
    assert result["adversary"]["caught_on_the_merits"] == 7
    assert result["adversary"]["escaping_unguarded"] == 1
    assert len(result["open_escapes"]) == 1
    escape = result["open_escapes"][0]
    assert escape["field"] == "failures" and escape["would_close_it"]
    # And the as-submitted figure must not be reported without the caveat that inflates it.
    assert "inflated" in result["adversary"]["note"]


class _StateCapturingAgent(_DesignToolAgent):
    """Keeps the state it was handed, so a test can assert on what the policy actually read."""

    def __init__(self, shows_tools=False):
        super().__init__()
        self.shows_tools = shows_tools
        self.states = []

    def act(self, task, state, allowed, fields=None, system=None, tools=None, choices=None):
        self.states.append(json.loads(json.dumps(state)))
        return super().act(task, state, allowed, fields=fields, system=system,
                           tools=tools, choices=choices)


def test_the_tool_block_is_absent_by_default_and_additive_when_asked_for():
    # The default is the arm that stays comparable with the roughly forty recorded episodes,
    # so the absence is the thing under test. When the block does arrive it must be purely
    # additive: every other key byte-identical, so the only difference between the two arms
    # is the block itself and not some field that moved while nobody was looking.
    off = _StateCapturingAgent(shows_tools=False)
    on = _StateCapturingAgent(shows_tools=True)
    run_episode("coronaviridae", "agent", agent=off)
    run_episode("coronaviridae", "agent", agent=on)
    assert len(off.states) == len(on.states) >= 2
    for before, after in zip(off.states, on.states):
        assert "available_tools" not in before
        assert "available_tools" in after
        assert {k: v for k, v in after.items() if k != "available_tools"} == before


def test_the_tool_block_shows_only_tools_an_action_can_actually_dispatch():
    # Three tools are registered and the action space reaches one. Showing the other two
    # would hand the policy an accurate description of work no action it can take produces --
    # and `request_structure`, the action that sounds like it would produce it, returns a
    # stated absence. That is a fabrication channel pointed at the policy.
    assert set(emergency.ACTION_TOOLS) <= set(emergency.ACTIONS)
    reachable = {n for names in emergency.ACTION_TOOLS.values() for n in names}
    assert reachable < set(tools.TOOLS)
    # The table tracks the dispatch chain rather than the memory of whoever last edited it:
    # recovered from source, so moving a `chief.call` without the table fails here.
    source = ast.parse(Path(emergency.__file__).read_text())
    named = {a.value for node in ast.walk(source)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "tool"
             for a in node.args if isinstance(a, ast.Constant)}
    assert named == reachable
    agent_ = _StateCapturingAgent(shows_tools=True)
    run_episode("coronaviridae", "agent", agent=agent_)
    assert set(agent_.states[0]["available_tools"]) == reachable


def test_the_policy_view_withholds_the_fields_that_are_not_addressed_to_a_decider():
    # Four kinds of field live on a spec and one is addressed to a policy. A module path is
    # unreachable from any action; `trust` tells a policy which argument to launder; `raises`
    # is a dispatch detail.
    spec = tools.tool("structures.evidence")
    view = tools.policy_view([spec])[spec.name]
    text = json.dumps(view)
    assert "hpc_pilots.structures_evidence" not in text and "entry" not in view
    assert "trust" not in text and "user_only" not in text
    assert "RuntimeError" not in text
    for field in ("endpoint", "requires", "version", "cost_seconds", "worst_case_seconds"):
        assert field not in view
    # What it does keep is the part that bears on a decision: the caveat, the citation
    # obligation, and whether the return can substantiate an axis at all.
    assert view["produces_evidence"] is True and view["citation"]
    assert "attention proxy" in view["warning"]
    assert [p["name"] for p in view["parameters"]] == [p.name for p in spec.parameters]
    # An unmeasured cost reads as unmeasured. Shown 0.0, a decider would be told the call is
    # free, and no artifact in this repository supports that; a string where a number would
    # go is what `request_structure` already returns instead of a plausible claim.
    assert view["cost"]["median_seconds"] == tools.UNMEASURED
    assert view["cost"]["actions"] == 1


def test_a_run_path_on_a_failure_never_reaches_the_policy():
    # The sharp one, and the reason the projection is a function rather than a field tuple.
    # `observed_in` holds paths under `runs/`, and those spell a family as a source literal
    # that `relabel_packet` cannot rewrite. Every registered failure has an empty tuple
    # today, so reading the live registry would pass vacuously and fail silently the first
    # time a proposal is merged -- and the optimiser's evidence guard refuses any proposal
    # that cites no artifact path, so a merged failure arrives carrying exactly this.
    planted = tools.Tool(
        name="planted.observed", version=1, summary="s", description="d",
        entry="hpc_pilots.chemistry:library_spec",
        failures=(tools.Failure("a quoted symptom", "a recorded cause", "ValueError",
                                observed_in=("runs/emergency/coronaviridae_rubric.json",)),))
    shown = json.dumps(tools.policy_view([planted]))
    assert "runs/emergency" not in shown and "coronaviridae" not in shown
    # And the part worth showing survives, so this is a projection and not a deletion.
    assert "a quoted symptom" in shown and "a recorded cause" in shown
    # The auditable record still carries it, which is the whole point of the split.
    assert "coronaviridae" in json.dumps(planted.record())


def test_a_target_name_written_into_a_spec_is_caught_by_the_identity_orbit():
    # The orbit scrub over the tool block finds nothing today, which on its own is an
    # assertion rather than a guard. This plants the leak it exists to catch: an enum whose
    # values are written into source survives `relabel_packet`, which rewrites packet content
    # and not source constants, so the policy would read the real name while the certificate
    # read a renamed packet. `enum_from` exists so a per-episode vocabulary is bound late.
    leaky = tools.Tool(name="planted.leak", version=1, summary="s", description="d",
                       entry="hpc_pilots.chemistry:library_spec",
                       parameters=(tools.Parameter("t", "str", "which target",
                                                   enum=("Mpro (nsp5)",)),))
    saved_tools = dict(tools.TOOLS)
    saved_actions = dict(emergency.ACTION_TOOLS)
    try:
        tools.TOOLS["planted.leak"] = leaky
        emergency.ACTION_TOOLS["invoke_design_tool"] = ("chemistry.library_spec",
                                                        "planted.leak")
        elements = invariance.group("coronaviridae", "identity", 20261021, 4)
        verdicts = [invariance.certify("coronaviridae", e) for e in elements]
        assert all(v["surviving_tokens"] for v in verdicts)
        assert not any(v["certified"] for v in verdicts)
        # An uncertified element stops the orbit rather than being skipped, so the leak
        # cannot be averaged away over whichever elements happened to pass.
        assert all(v["checks"]["no_declared_token_survives"] is False for v in verdicts)
    finally:
        tools.TOOLS.clear(); tools.TOOLS.update(saved_tools)
        emergency.ACTION_TOOLS.clear(); emergency.ACTION_TOOLS.update(saved_actions)
    # With the plant removed the orbit is clean again, so the failure above was the plant.
    assert all(invariance.certify("coronaviridae", e)["certified"]
               for e in invariance.group("coronaviridae", "identity", 20261021, 4))


def test_a_late_bound_vocabulary_reaches_the_block_and_a_missing_one_refuses():
    # The mechanism that makes the previous test's plant unnecessary in real specs. A missing
    # vocabulary must raise rather than widen to unconstrained, because an unconstrained
    # schema is the permissive reading of an absent one.
    late = tools.Tool(name="planted.late", version=1, summary="s", description="d",
                      entry="hpc_pilots.chemistry:library_spec",
                      parameters=(tools.Parameter("t", "str", "which target",
                                                  enum_from="target"),))
    view = tools.policy_view([late], {"target": ["A", "B"]})["planted.late"]
    assert view["parameters"][0]["admissible"] == ["A", "B"]
    with pytest.raises(ValueError, match="the caller did not supply"):
        tools.policy_view([late], {"axis": ["conservation"]})
    # And the positive control for the previous test's detector, because a detector verified
    # only on negatives is not verified: the same parameter bound late resolves to the
    # *relabelled* names, so the block carries the renamed packet's vocabulary and the
    # certificate stays clean where the hard-coded enum failed on all four elements.
    saved_tools = dict(tools.TOOLS)
    saved_actions = dict(emergency.ACTION_TOOLS)
    try:
        tools.TOOLS["planted.late"] = late
        emergency.ACTION_TOOLS["invoke_design_tool"] = ("planted.late",)
        element = invariance.group("coronaviridae", "identity", 20261021, 4)[0]
        _, renamed, _, _ = invariance.relabel_packet("coronaviridae", element)
        state = twins._state_for_policy(renamed, set(), 1, 8, 11, list(ACTIONS),
                                        shows_tools=True)
        admissible = state["available_tools"]["planted.late"]["parameters"][0]["admissible"]
        assert admissible == [d.target for d in renamed]
        assert admissible != [d.target for d in packet("coronaviridae")[1]]
        certificate = invariance.certify("coronaviridae", element)
        assert certificate["certified"] and not certificate["surviving_tokens"]
    finally:
        tools.TOOLS.clear(); tools.TOOLS.update(saved_tools)
        emergency.ACTION_TOOLS.clear(); emergency.ACTION_TOOLS.update(saved_actions)


def test_showing_the_specs_is_a_distinct_policy_label_and_a_typo_is_not_off():
    # Hazard five: the input text changes, so no episode recorded before this is comparable
    # with one after it even at identical payload bytes -- the schema is unmoved and the
    # prompt is not. The label is the device that already keeps hosted runs from pooling with
    # local ones, and it is read off the same flag that builds the block so the two cannot
    # disagree about which arm a file holds.
    environment = {"PILOT_LLM_URL": "http://localhost:8000/v1/chat/completions",
                   "PILOT_LLM_MODEL": "stub-model"}
    with mock.patch.dict(os.environ, {**environment, "PILOT_TOOL_STATE": "0"}, clear=False):
        plain = common.Agent()
    with mock.patch.dict(os.environ, {**environment, "PILOT_TOOL_STATE": "1"}, clear=False):
        shown = common.Agent()
    assert plain.label == "agent" and plain.shows_tools is False
    assert shown.label == "agent-toolstate" and shown.shows_tools is True
    assert plain.report()["tool_state"] == "0" and shown.report()["tool_state"] == "1"
    # A typo must not read as off, because off is the arm that pools with the record.
    with mock.patch.dict(os.environ, {**environment, "PILOT_TOOL_STATE": "true"}, clear=False):
        with pytest.raises(RuntimeError, match="Unknown PILOT_TOOL_STATE"):
            common.Agent()


def test_the_tool_block_costs_more_state_than_the_action_block_it_sits_beside():
    # Measured rather than asserted, because the finder's stated threshold was computed from
    # an estimate that was wrong by a factor of seven and the correction is in its docstring.
    # One tool's policy view is roughly five times the whole six-entry action block, and the
    # reason is that seventy per cent of it is prose: the caveats, the citation obligation and
    # the recorded causes. Trimming the block to fit would mean trimming exactly the
    # disclaimers that make a tool safe to show, which inverts the obvious optimisation.
    spec, dossiers, _, _ = packet("coronaviridae")
    actions = len(json.dumps({a: ACTIONS[a] for a in ACTIONS}))
    one = len(json.dumps(tools.policy_view([tools.tool("chemistry.library_spec")])))
    everything = len(json.dumps(tools.policy_view(
        [tools.tool(n) for n in sorted(tools.TOOLS)])))
    assert one > 4 * actions
    per_tool = everything / len(tools.TOOLS)
    state = len(json.dumps({"dossiers": [d.state(1) for d in dossiers]}))
    # The condition under which `all` stops being the honest default, recomputed from the
    # real per-tool cost instead of the estimate: single digits, not sixty-five.
    assert 5 < state / per_tool < 15


def test_every_spec_field_is_classified_as_shown_or_withheld():
    # `POLICY_VISIBLE` and `WITHHELD_FROM_POLICY` would otherwise be documentation that
    # nothing checks, and `policy_view` builds its output by hand. A field added to `Tool`
    # later would then be withheld by omission -- the safe direction, but unrecorded, so a
    # reader could not tell a deliberate exclusion from a forgotten one. This makes adding a
    # field to the record fail here until somebody says which kind it is.
    classified = set(tools.POLICY_VISIBLE) | set(tools.WITHHELD_FROM_POLICY)
    for record, prefix in ((tools.Tool, ""), (tools.Parameter, "parameters."),
                           (tools.Failure, "failures.")):
        for field in dataclasses.fields(record):
            name = prefix + field.name
            assert name in classified, \
                f"{name} is neither shown to a policy nor recorded as withheld"
    # And nothing is claimed both ways.
    assert not set(tools.POLICY_VISIBLE) & set(tools.WITHHELD_FROM_POLICY)
    # The withheld half is a reason per field, not a bare list: a field excluded without a
    # stated reason is indistinguishable from one excluded by accident.
    assert all(reason for reason in tools.WITHHELD_FROM_POLICY.values())
    # And the shown half is not aspirational: every field it names must actually appear in a
    # view built from a spec that declares one, or the table claims more than the code does.
    view = tools.policy_view([tools.tool("structures.evidence")])["structures.evidence"]
    flat = set(view) | {"name", "cost_actions", "cost_network_calls"}
    flat |= {f"parameters.{k}" for p_ in view["parameters"] for k in p_}
    flat |= {f"failures.{k}" for f_ in view["failures"] for k in f_}
    missing = {f for f in tools.POLICY_VISIBLE if f not in flat}
    # Exactly one, and it is absence rather than withholding: no v1 parameter declares a
    # length ceiling, and a bound shown as null would read to a decider as a bound of null.
    # The numeric bounds do appear, so dropping one from the view fails here.
    assert missing == {"parameters.max_length"}
