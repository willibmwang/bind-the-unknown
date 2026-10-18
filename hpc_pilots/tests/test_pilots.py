import json

import numpy as np
import pytest

from hpc_pilots.binding import pick_batch, replay
from hpc_pilots.calibration import calibration_metrics, trajectory
from hpc_pilots.common import ROOT
from hpc_pilots.common import Agent
from hpc_pilots.structure import structure_replay


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
