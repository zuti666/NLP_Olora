"""CPU-only, identity-checked merge of historical B states and new radii."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parents[1] / 'execution'
LEGACY = EXECUTION / 'results_seed0_recovery1'
NEW_RADII = list(range(11, 20))


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend_verified_data(neighborhood, traces, selected, radii, checks, source_dir):
    """Original seven-point loader has already verified original state identities."""
    old_radii = [0, .5, 1, 2, 5, 10, 20]
    assert list(radii) == old_radii
    added, trace_added, direction_added, sources = [], [], [], []
    paired = {}
    comparisons = {}
    for method in ('sgd', 'gam'):
        directory = HERE / 'results' / method / 'neighborhood'
        legacy_dir = LEGACY / method / 'neighborhood'
        summary = read(directory / 'summary.json')
        assert summary['status'] == 'complete'
        assert summary['radius_extension_guard_passed'] is True
        manifest, legacy = read(directory / 'manifest.json'), read(legacy_dir / 'manifest.json')
        keys = ('method', 'seed', 'source_path', 'source_files', 'diagnostic_code_sha256',
                'common_code_sha256', 'checkpoint_sha256', 'data_manifest_sha256',
                'coordinate_order', 'coordinate_numel', 'coordinate_sha256', 'head_sha256',
                'anchor_effective_sha256', 'production_precision', 'fisher_definition')
        for key in keys:
            assert manifest[key] == legacy[key], (method, key)
        assert read(directory.parent / 'data_manifest.json') == read(legacy_dir.parent / 'data_manifest.json')
        expected_config = dict(legacy['config'])
        expected_config.update(output_dir=str(HERE / 'results'), radii=[0] + NEW_RADII)
        assert manifest['config'] == expected_config, 'Only output path and radii may change'
        records, old_records = rows(directory / 'states.jsonl'), rows(legacy_dir / 'states.jsonl')
        assert len(records) == 145 and len({r['state_id'] for r in records}) == 145
        assert summary['states_completed'] == 145
        origin = next(r for r in old_records if r['point']['kind'] == 'anchor')
        new_origin = next(r for r in records if r['point']['kind'] == 'anchor')
        assert new_origin['point_effective_sha256'] == origin['point_effective_sha256']
        for split in ('train', 'test'):
            np.testing.assert_allclose(new_origin['metrics'][split]['loss'], origin['metrics'][split]['loss'], rtol=1e-5, atol=3e-6)
            np.testing.assert_allclose(new_origin['metrics'][split]['accuracy'], origin['metrics'][split]['accuracy'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(new_origin['fisher']['raw_trace'], origin['fisher']['raw_trace'], rtol=1e-4, atol=1e-6)
        old_directions = {r['point']['direction']['index']: r['point']['direction']['sha256']
                          for r in old_records if r['point']['radius'] == 10}
        assert len(old_directions) == 16
        for row in records:
            assert row['method'] == method
            assert row['protocol_sha256'] == manifest['protocol_sha256']
            assert row['checkpoint_sha256'] == manifest['checkpoint_sha256'][method]
            assert row['data_manifest_sha256'] == manifest['data_manifest_sha256']
            assert row['coordinate_sha256'] == manifest['coordinate_sha256']
            assert row['fisher']['samples'] == 256 and row['fisher']['normalized'] is False
            assert row['fisher']['floor'] is None and row['fisher']['regularizer'] is None
            assert row['metrics']['train']['samples'] == 1259
            assert row['metrics']['test']['samples'] == 307
            if row['point']['kind'] == 'anchor':
                continue
            assert row['point']['kind'] == 'neighborhood'
            radius = row['point']['radius']
            assert radius in NEW_RADII and row['amount'] == radius
            direction = row['point']['direction']
            assert direction['sha256'] == old_directions[direction['index']]
            paired[method, radius, direction['index']] = direction['sha256']
            direction_added.append(dict(method=method, method_label=checks['method_names'][method], seed=0,
                state_id=row['state_id'], radius=radius, direction_index=direction['index'],
                direction_sha256=direction['sha256'], point_effective_sha256=row['point_effective_sha256'],
                raw_trace_train256=row['fisher']['raw_trace'],
                D_train=row['metrics']['train']['loss'] - origin['metrics']['train']['loss'],
                D_test=row['metrics']['test']['loss'] - origin['metrics']['test']['loss'],
                accuracy_train=row['metrics']['train']['accuracy'], reused_origin=False))
        for radius in NEW_RADII:
            subset = [r for r in records if r['point']['radius'] == radius]
            assert len(subset) == 16
            assert {r['point']['direction']['index'] for r in subset} == set(range(16))
            raw_trace = np.array([r['fisher']['raw_trace'] for r in subset])
            assert np.isfinite(raw_trace).all() and (raw_trace >= 0).all()
            trace_added.append(dict(method=method, method_label=checks['method_names'][method], seed=0,
                radius=radius, directions=16, train_samples=256, trace_mean=raw_trace.mean(),
                trace_Q95=np.quantile(raw_trace, .95), trace_sample_max=raw_trace.max(),
                reused_origin=False, accuracy_mean_train=np.mean([r['metrics']['train']['accuracy'] for r in subset]),
                accuracy_train_samples=1259))
            for split in ('train', 'test'):
                damage = np.array([r['metrics'][split]['loss'] - origin['metrics'][split]['loss'] for r in subset])
                positive = np.maximum(damage, 0)
                assert np.isfinite(damage).all()
                added.append(dict(method=method, seed=0, transform='deterministic_test', split=split,
                    radius=float(radius), directions=16, D_mean=damage.mean(), G=positive.mean(),
                    Q95=np.quantile(positive, .95), raw_trace_mean_train256=raw_trace.mean()))
        comparisons[method] = {'identity_passed': True, 'new_states': len(records),
            'anchor_train_loss_difference': new_origin['metrics']['train']['loss'] - origin['metrics']['train']['loss'],
            'new_damage_reference': 'unchanged original historical FP32 anchor',
            'protocol_sha256': manifest['protocol_sha256']}
        sources.extend([directory / name for name in ('summary.json', 'manifest.json', 'states.jsonl')])
        sources.append(directory.parent / 'data_manifest.json')
    for radius in NEW_RADII:
        for index in range(16):
            assert paired['sgd', radius, index] == paired['gam', radius, index]
    neighborhood = pd.concat([neighborhood, pd.DataFrame(added)], ignore_index=True)
    traces = pd.concat([traces, pd.DataFrame(trace_added)], ignore_index=True)
    radii = np.array(sorted(set(old_radii + NEW_RADII)), dtype=float)
    assert len(radii) == 16 and len(neighborhood) == 64 and len(traces) == 32
    assert not neighborhood.duplicated(['method', 'split', 'radius']).any()
    assert not traces.duplicated(['method', 'radius']).any()
    # Retained as a numeric appendix export only; tolerance is not plotted.
    tolerance = []
    for method in ('sgd', 'gam'):
        for split in ('train', 'test'):
            for epsilon in (.001, .005, .01, .02, .05):
                prefix = 0
                for row in neighborhood.query('method == @method and split == @split').sort_values('radius').itertuples():
                    if row.G > epsilon:
                        break
                    prefix = row.radius
                tolerance.append(dict(method=method, seed=0, transform='deterministic_test', epsilon=epsilon,
                    metric='G', prefix_radius=prefix, right_censored=prefix == 20, split=split))
    selected = pd.DataFrame(tolerance)
    neighborhood.assign(method_label=neighborhood.method.map(checks['method_names']), dataset='ImageNet-A').to_csv(source_dir / 'B_damage.csv', index=False)
    old_direction_csv = pd.read_csv(source_dir / 'B_direction_states.csv', float_precision='round_trip')
    pd.concat([old_direction_csv, pd.DataFrame(direction_added)], ignore_index=True).to_csv(source_dir / 'B_direction_states.csv', index=False)
    traces.to_csv(source_dir / 'B_trace_direction_summary.csv', index=False)
    selected.assign(method_label=selected.method.map(checks['method_names']), dataset='ImageNet-A').to_csv(source_dir / 'B_tolerance_Gmean.csv', index=False)
    checks.update(raw_state_count_per_method=241, radii=radii.tolist(), radius_extension_applied=True,
        new_radii=NEW_RADII, radius_extension_identity=comparisons,
        source_states_used=482, radius_extension_new_states_used=288,
        radius_extension_reference_anchor='original historical FP32 anchor; new anchors checked but not duplicated',
        radius_extension_new_direction_hashes_match_original=True,
        Gmean_method_ties_at_all_sampled_thresholds=None,
        appendix_Gmean_tolerance_recomputed_on_extended_grid=True)
    checks['sources'].extend([{'path': str(p), 'sha256': sha(p)} for p in sources])
    (HERE / 'MERGE_CHECKS.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    return neighborhood, traces, selected, radii, checks
