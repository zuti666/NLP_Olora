"""Read-only model-free recheck of the user-referenced completed B1/B2 cohort."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
JOBS = ('dense', 'sparse_train', 'sparse_test', 'sparse_train_full',
        'trajectory_fisher', 'path_t01', 'path_t05', 'path_t09', 'bridge_full')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return True


def stats(values):
    return dict(values=list(map(float, values)), mean=float(np.mean(values)),
                sample_sd=float(np.std(values, ddof=1)), n=3)


def paired(a, b):
    return dict(EWCLoRA=stats(a), FS_EWCLoRA=stats(b),
                paired_FS_minus_EWC=stats(np.array(b) - a),
                FS_lower_seeds=sum(y < x for x, y in zip(a, b)))


def main():
    coverage = read(BASE / 'final_report/coverage.json')
    expected = {(s, m, j) for s in range(3) for m in ('sgd', 'gam') for j in JOBS}
    observed = {(r['seed'], r['method'], r['job']) for r in coverage['diagnostics']}
    assert expected == observed and len(coverage['diagnostics']) == 54
    assert not coverage['errors']
    summaries, source_hashes = {}, {}
    for item in coverage['diagnostics']:
        p = Path(item['summary']); digest = sha(p)
        assert digest == item['sha256']
        d = read(p)
        assert d['status'] == 'complete' and finite(d)
        assert d['method'] == item['method'] and d['seed'] == item['seed']
        if item['job'].startswith('path_t'):
            assert d['job'] == 'path' and d['target_task'] == int(item['job'][6:])
        else:
            assert d['job'] == item['job']
        assert d['protocol_sha256'] == item['protocol_sha256']
        summaries[item['seed'], item['method'], item['job']] = d
        source_hashes[str(p)] = digest
    queue = read(BASE / 'formal_queue/pipeline_state.json')
    assert queue['status'] == 'complete' and len(queue['jobs']) == 88
    assert all(j['status'] == 'complete' for j in queue['jobs'].values())
    raw_states_checked = 0
    direction_library = {}
    for seed in range(3):
        for method in ('sgd', 'gam'):
            for job, grid, directions, scope, samples in (
                    ('dense', [i / 4 for i in range(81)], 100, None, None),
                    ('sparse_train', [0, .5, 1, 2, 5, 10, 20], 16, 'subset', 256),
                    ('sparse_test', [0, 1, 10, 20], 16, 'test', 307),
                    ('sparse_train_full', [0, 1, 10, 20], 4, 'train', 1259)):
                directory = BASE / f'formal_results/seed{seed}/{method}/{job}'
                data = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
                summary = summaries[seed, method, job]
                manifest = read(directory / 'manifest.json')
                assert len(data) == len({r['state_id'] for r in data}) == 1 + (len(grid) - 1) * directions
                assert summary['states_completed'] == len(data) == summary['expected_states']
                assert [r['radius'] for r in summary['radius_summary']] == grid
                assert summary['direction_indices'] == list(range(directions))
                origin = next(r for r in data if r['point']['radius'] == 0)
                by_radius = {radius: [] for radius in grid}
                for row in data:
                    assert row['method'] == method and row['seed'] == seed
                    assert row['protocol_sha256'] == manifest['protocol_sha256']
                    assert row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method]
                    assert row['data_manifest_sha256'] == manifest['data_manifest_sha256']
                    assert row['coordinate_sha256'] == manifest['coordinate_sha256']
                    assert row['anchor_restored_exactly'] is True
                    assert row['metrics']['train']['samples'] == 1259 and row['metrics']['test']['samples'] == 307
                    radius = row['point']['radius']; by_radius[radius].append(row)
                    if radius:
                        direction = row['point']['direction']
                        idx = direction['index']
                        if idx in direction_library:
                            assert direction['sha256'] == direction_library[idx]
                        direction_library[idx] = direction['sha256']
                    if scope is None:
                        assert row['fisher_by_scope'] == {}
                    else:
                        fisher = row['fisher_by_scope'][scope]
                        assert fisher['samples'] == samples and fisher['normalized'] is False
                        assert fisher['floor'] is None and fisher['regularizer'] is None
                for aggregate in summary['radius_summary']:
                    radius = aggregate['radius']; states = by_radius[radius]
                    assert len(states) == (1 if radius == 0 else directions)
                    if radius:
                        assert {r['point']['direction']['index'] for r in states} == set(range(directions))
                    for split in ('train', 'test'):
                        losses = np.array([r['metrics'][split]['loss'] for r in states])
                        damage = losses - origin['metrics'][split]['loss']
                        positive = np.maximum(damage, 0)
                        actual = [damage.mean(), positive.mean(), np.quantile(positive, .95), losses.mean(),
                                  np.mean([r['metrics'][split]['accuracy'] for r in states])]
                        stored = [aggregate[split][k] for k in ('D_mean', 'G', 'Q95', 'absolute_loss_mean', 'accuracy_mean')]
                        np.testing.assert_allclose(actual, stored, rtol=1e-10, atol=1e-12)
                    if scope:
                        trace = [r['fisher_by_scope'][scope]['raw_trace'] for r in states]
                        np.testing.assert_allclose(np.mean(trace), aggregate['raw_trace_by_scope'][scope]['mean'], rtol=1e-10)
                for item in summary['tolerance']:
                    prefix = 0
                    for aggregate in summary['radius_summary']:
                        if aggregate[item['split']][item['metric']] > item['epsilon']:
                            break
                        prefix = aggregate['radius']
                    assert prefix == item['prefix_radius']
                raw_states_checked += len(data)
    results = {}
    for split in ('train', 'test'):
        for metric in ('G', 'Q95'):
            values = {m: [summaries[s, m, 'dense']['radius_summary'][-1][split][metric] for s in range(3)] for m in ('sgd', 'gam')}
            results[f'dense_r20_{split}_{metric}'] = paired(values['sgd'], values['gam'])
    for job, scope in (('sparse_train', 'subset'), ('sparse_train_full', 'train'), ('sparse_test', 'test')):
        for radius in (0, 20):
            values = {m: [next(r for r in summaries[s, m, job]['radius_summary'] if r['radius'] == radius)['raw_trace_by_scope'][scope]['mean'] for s in range(3)] for m in ('sgd', 'gam')}
            results[f'{job}_r{radius}_raw_trace'] = paired(values['sgd'], values['gam'])
    probe_comparisons, selected_tolerance = {}, {}
    for split in ('train', 'test'):
        for metric in ('G', 'Q95'):
            differences = [b[split][metric] - a[split][metric] for seed in range(3)
                           for a, b in zip(summaries[seed, 'sgd', 'dense']['radius_summary'][1:],
                                           summaries[seed, 'gam', 'dense']['radius_summary'][1:])]
            probe_comparisons[f'{split}_{metric}'] = dict(FS_lower=sum(x < 0 for x in differences),
                FS_higher=sum(x > 0 for x in differences), ties=sum(x == 0 for x in differences),
                comparisons=240, independent_training_pairs=3)
        selected_tolerance[split] = {method: [next(t['prefix_radius']
            for t in summaries[seed, method, 'dense']['tolerance']
            if t['split'] == split and t['metric'] == 'G' and t['epsilon'] == .01)
            for seed in range(3)] for method in ('sgd', 'gam')}
    output = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
        expected_diagnostics=54, validated_diagnostics=54, missing=0, failed=0,
        formal_queue_jobs_complete=88, B1_raw_states_recomputed=raw_states_checked,
        common_direction_library_size=len(direction_library),
        B1_summary_metrics_recomputed_from_raw=True, all_54_summaries_finite=True,
        stored_receipt_hashes_match=True, statistics_unit='training seed; n=3 paired; sample SD ddof=1',
        results=results, radius_probe_comparisons=probe_comparisons,
        G_tolerance_epsilon_001_seed_order=selected_tolerance, source_summary_sha256=source_hashes,
        limits='CPU log/summary recheck only; no new model evaluation or checkpoint file rehash')
    (HERE / 'REQUESTED_B1_SOURCE_CHECK.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'source_summary_sha256'}, indent=2))


if __name__ == '__main__':
    main()
