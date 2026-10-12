#!/usr/bin/env python3
"""Audit and publish formal three-seed paths for all nine future endpoints.

CPU only. Existing t1/t5/t9 measurements stay immutable in formal_results;
the six added stages are read from this extension's results directory.
No plot is produced until all 54 paths pass the source and numerical checks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

EXT = Path(__file__).resolve().parent
BASE = EXT.parent
OUT = EXT / 'figures'
REPORT = EXT / 'report'
STYLE_PATH = BASE / 'figures_formal_20260921/plot_common.py'
os.environ.setdefault('MPLCONFIGDIR', str(EXT / '.mplcache'))
spec = importlib.util.spec_from_file_location('formal_stage_plot_style', STYLE_PATH)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
style.HERE = OUT  # The imported module's BASE remains the original formal cohort.
np, pd, plt = style.np, style.pd, style.plt
read, sha, canonical = style.read, style.sha, style.canonical
METHODS = style.METHODS
EXISTING_TASKS = (1, 5, 9)
NEW_TASKS = (2, 3, 4, 6, 7, 8)
IDENTITY_FIELDS = (
    'schema', 'seed', 'method', 'diagnostic_code_sha256', 'common_code_sha256',
    'pilot_runner_code_sha256', 'source_path', 'source_files',
    'data_manifest_sha256', 'counts', 'coordinate_order', 'coordinate_numel',
    'coordinate_sha256', 'head_sha256', 'anchor_effective_sha256',
    'direction_seed', 'production_precision', 'fisher_definition',
    'point_restoration',
)


def close(actual, expected, *, atol=1e-11, rtol=1e-9):
    assert np.isclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, (float, int)):
        assert math.isfinite(value), value


def simpson(values):
    assert len(values) >= 3 and len(values) % 2 == 1
    assert all(math.isfinite(x) for x in values)
    return (values[0] + values[-1] + 4 * sum(values[1:-1:2])
            + 2 * sum(values[2:-1:2])) / (3 * (len(values) - 1))


def scientific_config(config):
    return {k: v for k, v in config.items() if k != 'output_dir'}


def source_record(path):
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def endpoint_check(recovery, meta, method, task):
    assert recovery['passed'] and recovery['task'] == task
    assert recovery['checkpoint'] == meta['checkpoints'][method][str(task)]
    assert recovery['checkpoint_sha256'] == meta['checkpoint_sha256'][method][str(task)]
    assert recovery['max_logit_error'] < 3e-5
    assert all(abs(e) < 3e-6 for e in recovery['local_loss_errors'].values())
    if 'path_endpoint_loss_errors' in recovery:
        assert all(abs(e) < 3e-6 for e in recovery['path_endpoint_loss_errors'].values())


def load_job(directory, seed, method, sources, checkpoints, *, coverage=None,
             job=None, trajectory_meta=None):
    summary_path = directory / 'summary.json'
    meta_path = directory / 'manifest.json'
    raw_path = directory / 'states.jsonl'
    sanity_path = directory / 'sanity.json'
    summary, meta, sanity = map(read, (summary_path, meta_path, sanity_path))
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    sources.extend(source_record(p) for p in (summary_path, meta_path, raw_path, sanity_path))
    assert summary['status'] == 'complete'
    assert summary['seed'] == meta['seed'] == seed
    assert summary['method'] == meta['method'] == method
    assert summary['protocol_sha256'] == meta['protocol_sha256']
    assert canonical({k: v for k, v in meta.items() if k != 'protocol_sha256'}) == meta['protocol_sha256']
    assert meta['config']['training_seed'] == seed
    assert Path(meta['config']['data_root']).name == 'imagenet-a'
    assert meta['coordinate_numel'] == 14155776
    assert meta['counts'] == {'train': 1259, 'test': 307, 'subset': 256}
    config_path = (EXT / f'configs/path_seed{seed}.json' if trajectory_meta is not None
                   and meta['path_task'] in NEW_TASKS else BASE / f'configs/seed{seed}.json')
    assert config_path.is_file(), config_path
    assert read(config_path) == meta['config']
    assert sha(config_path) == meta['config_sha256']
    sources.append(source_record(config_path))
    if coverage is not None:
        matches = [r for r in coverage['diagnostics'] if
                   (r['seed'], r['method'], r['job']) == (seed, method, job)]
        assert len(matches) == 1 and matches[0]['sha256'] == sha(summary_path)
    if trajectory_meta is not None:
        for field in IDENTITY_FIELDS:
            assert meta[field] == trajectory_meta[field], (directory, field)
        assert scientific_config(meta['config']) == scientific_config(trajectory_meta['config'])
        expected_root = (EXT / f'results/seed{seed}' if meta['path_task'] in NEW_TASKS
                         else BASE / f'formal_results/seed{seed}')
        assert Path(meta['config']['output_dir']) == expected_root
        assert meta['fisher_scopes'] == ['train']
        assert meta['job'] == summary['job'] == 'path'
        assert set(meta['checkpoints']) == {method}
        assert set(meta['checkpoints'][method]) == {'0', str(meta['path_task'])}
        for task, p in meta['checkpoints'][method].items():
            assert p == trajectory_meta['checkpoints'][method][task]
            assert meta['checkpoint_sha256'][method][task] == trajectory_meta['checkpoint_sha256'][method][task]
    assert len(rows) == summary['states_completed']
    assert len({r['state_id'] for r in rows}) == len(rows)
    assert sanity['passed'] and sanity['predictions_match']
    assert sanity['protocol_sha256'] == meta['protocol_sha256']
    assert sanity['effective_sha256'] == meta['anchor_effective_sha256']
    assert summary['end_restoration']['predictions_match']
    assert summary['end_restoration']['effective_sha256'] == meta['anchor_effective_sha256']
    assert sanity['max_logit_error'] == summary['end_restoration']['max_logit_error'] == 0
    for task, checkpoint_path in meta['checkpoints'][method].items():
        p = Path(checkpoint_path)
        assert p.is_file()
        record = dict(path=str(p), recorded_sha256=meta['checkpoint_sha256'][method][task],
                      bytes=p.stat().st_size,
                      verification='file exists; digest matches manifests/raw states/endpoint recovery and original formal coverage')
        assert len(record['recorded_sha256']) == 64
        if checkpoint_path in checkpoints:
            assert record == checkpoints[checkpoint_path]
        checkpoints[checkpoint_path] = record
    for row in rows:
        finite_tree(row)
        assert row['seed'] == seed and row['method'] == method
        assert row['anchor_restored_exactly']
        assert row['protocol_sha256'] == meta['protocol_sha256']
        assert row['checkpoint_sha256'] == meta['checkpoint_sha256'][method]
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            assert row[key] == meta[key]
        for scope in ('train', 'test'):
            assert row['metrics'][scope]['samples'] == meta['counts'][scope]
        close(row['effective_weight_drift'] ** 2, row['effective_weight_drift_squared'])
        for scope, fisher in row['fisher_by_scope'].items():
            assert fisher['samples'] == {'train': 1259, 'test': 307}[scope]
            assert not fisher['normalized'] and fisher['floor'] is None and fisher['regularizer'] is None
            assert fisher['raw_trace'] >= 0
    finite_tree(summary)
    receipt = directory / 'extension_receipt.json'
    if trajectory_meta is not None and meta['path_task'] in NEW_TASKS:
        evidence = read(receipt)
        assert evidence['status'] == 'complete' and evidence['schema'] == 'rq2_formal_completion_v1'
        assert evidence['seed'] == seed and evidence['method'] == method
        assert evidence['job'] == 'path' and evidence['target_task'] == meta['path_task']
        assert evidence['samples'] == 1259 and evidence['protocol_sha256'] == meta['protocol_sha256']
        assert evidence['summary'] == str(summary_path)
        for key, p in [('summary_sha256', summary_path), ('manifest_sha256', meta_path),
                       ('states_sha256', raw_path), ('wrapper_sha256', EXT / 'diagnostic_extension.py')]:
            assert evidence[key] == sha(p), (receipt, key)
        for key in ('numerical_gate_passed', 'quadrature_converged', 'bound_passed_with_numeric_tolerance'):
            assert evidence[key] is True and evidence[key] == summary[key]
        assert evidence['preflight']['status'] == 'passed'
        for key in ('identity_check', 'origin_check', 'endpoint_check'):
            assert evidence[key]['passed'] is True, (receipt, key)
        sources.append(source_record(receipt))
    return summary, meta, rows, raw_path


def extract_trajectory(seed, method, summary, meta, rows, raw_path):
    assert summary['job'] == meta['job'] == 'trajectory_fisher'
    assert summary['task_count'] == 10 and len(rows) == 11
    assert summary['train_fisher_samples'] == 1259 and summary['test_fisher_samples'] == 307
    assert sorted(r['task'] for r in summary['tasks']) == list(range(10))
    by_id = {r['state_id']: r for r in rows}
    origin = by_id['r0']
    output = []
    for item in summary['tasks']:
        task = item['task']
        raw = by_id[f'task{task:02d}']
        assert raw['point']['kind'] == 'trajectory' and raw['point']['task'] == task
        assert raw['metrics'] == item['metrics']
        close(raw['effective_weight_drift'], item['effective_weight_drift'])
        endpoint_check(item['endpoint_recovery'], meta, method, task)
        local = item['direct_local_and_seen']['test']['local']
        assert local['classes'] == 20 and local['samples'] == 307
        close(raw['metrics']['test']['accuracy'], local['accuracy'])
        close(raw['metrics']['test']['loss'], local['loss'], atol=3e-6)
        damage = raw['metrics']['test']['loss'] - origin['metrics']['test']['loss']
        close(damage, item['D0']['test'])
        output.append(dict(seed=seed, method=method, task=task, T=task+1,
            weight_drift=raw['effective_weight_drift'], test_CE=raw['metrics']['test']['loss'],
            anchor_test_CE=origin['metrics']['test']['loss'], signed_test_damage=damage,
            test_accuracy_percent=raw['metrics']['test']['accuracy'], test_samples=307,
            fixed_head_classes=20, source=str(raw_path), source_state_id=raw['state_id'],
            protocol_sha256=meta['protocol_sha256'],
            endpoint_effective_delta_sha256=item['endpoint_recovery']['effective_delta_sha256'],
            checkpoint=meta['checkpoints'][method][str(task)],
            checkpoint_sha256=meta['checkpoint_sha256'][method][str(task)]))
    return output


def recompute_gate(coarse, fine, damage, q, cfg):
    tolerance = cfg['quadrature_atol'] + cfg['quadrature_rtol'] * max(abs(fine), abs(coarse))
    bound_tolerance = max(1e-8, cfg['bound_rtol'] * abs(fine))
    peak = any(q[i] > cfg['path_peak_neighbor_factor'] * max(q[i-1], q[i+1], 1e-12)
               for i in range(1, len(q)-1))
    stable = abs(fine - coarse) <= tolerance
    bound = damage ** 2 <= fine + bound_tolerance
    return dict(absolute_difference=abs(fine-coarse),
        relative_difference=abs(fine-coarse)/max(abs(fine), abs(coarse), 1e-30),
        stability_tolerance=tolerance, stable=stable, bound_tolerance=bound_tolerance,
        bound_passed=bound, bound_passed_without_tolerance=damage ** 2 <= fine,
        sampled_peak_needs_resolution_check=peak,
        request_refinement=not stable or not bound or peak, passed=stable and bound)


def extract_path(seed, method, task, summary, meta, rows, raw_path, trajectory):
    count = summary['selected_nodes']
    assert count in (9, 17, 33) and count == len(rows) == len(summary['path'])
    assert summary['samples'] == 1259 and summary['fisher_scope'] == 'full old-task train1259'
    assert summary['target_task'] == meta['path_task'] == task
    assert summary['numerical_gate_passed'] and summary['quadrature_converged']
    assert summary['bound_passed_with_numeric_tolerance']
    endpoint_check(summary['endpoint_recovery'], meta, method, task)
    raw = sorted(rows, key=lambda r: r['amount'])
    assert [r['amount'] for r in raw] == [i/(count-1) for i in range(count)]
    name = f'{method}_t{task:02d}'
    assert set(meta['actual_drift']) == set(meta['actual_direction_sha256']) == {name}
    assert meta['actual_direction_sha256'][name] == summary['endpoint_recovery']['effective_delta_sha256']
    assert meta['actual_direction_sha256'][name] == trajectory['endpoint_effective_delta_sha256']
    q, traces, nodes = [], [], []
    for row, item in zip(raw, summary['path']):
        assert set(row['fisher_by_scope']) == {'train'}
        fisher = row['fisher_by_scope']['train']
        assert set(fisher['q']) == {name} and fisher['q'][name] >= 0
        close(row['amount'], item['s']); close(fisher['q'][name], item['q'])
        close(fisher['raw_trace'], item['raw_trace'])
        close(fisher['subset_loss'], item['train_loss_for_bound'])
        assert row['metrics'] == item['metrics']
        if row['amount']:
            assert row['point']['kind'] == 'path' and row['point']['target_task'] == task
            close(row['point']['path_fraction'], row['amount'])
            assert row['point']['direction_sha256'] == meta['actual_direction_sha256'][name]
        q.append(fisher['q'][name]); traces.append(fisher['raw_trace'])
        nodes.append(dict(seed=seed, method=method, task=task, T=task+1, s=row['amount'],
            q=fisher['q'][name], raw_trace=fisher['raw_trace'],
            train_loss_for_bound=fisher['subset_loss'], train_samples=fisher['samples'],
            test_CE=row['metrics']['test']['loss'], source=str(raw_path),
            source_state_id=row['state_id'], protocol_sha256=meta['protocol_sha256']))
    integral, trace_integral = simpson(q), simpson(traces)
    close(integral, summary['integral']); close(trace_integral, summary['trace_integral'])
    drift = meta['actual_drift'][name]
    assert drift > 0
    close(drift, summary['drift']); close(drift ** 2, summary['drift_squared'])
    for row in raw:
        close(row['effective_weight_drift'], row['amount'] * drift, atol=1e-7)
    sensitivity = integral / drift ** 2
    close(sensitivity, summary['S_path'])
    train_damage = raw[-1]['fisher_by_scope']['train']['subset_loss'] - raw[0]['fisher_by_scope']['train']['subset_loss']
    test_damage = raw[-1]['metrics']['test']['loss'] - raw[0]['metrics']['test']['loss']
    close(train_damage, summary['D_train']); close(test_damage, summary['D_test'])
    close(train_damage ** 2, summary['D_train_squared'])
    close(drift, trajectory['weight_drift'], atol=1e-7)
    close(test_damage, trajectory['signed_test_damage'], atol=3e-6)
    close(raw[-1]['metrics']['test']['loss'], trajectory['test_CE'], atol=3e-6)
    close(raw[0]['metrics']['test']['loss'], trajectory['anchor_test_CE'], atol=3e-6)
    close(raw[-1]['metrics']['test']['accuracy'], trajectory['test_accuracy_percent'])
    nested = [n for n in (5,9,17,33) if n <= count]
    assert set(summary['integrals_simpson']) == {str(n) for n in nested}
    assert [g['nodes'] for g in summary['quadrature_gates']] == nested[1:]
    previous, gates = None, []
    for n in nested:
        stride = (count-1)//(n-1)
        qq, tt = q[::stride], traces[::stride]
        current = simpson(qq)
        close(current, summary['integrals_simpson'][str(n)]['q_integral'])
        close(simpson(tt), summary['integrals_simpson'][str(n)]['trace_integral'])
        if previous is not None:
            gate = recompute_gate(previous, current, train_damage, qq, meta['config'])
            recorded = next(g for g in summary['quadrature_gates'] if g['nodes'] == n)
            for key, value in gate.items():
                if isinstance(value, bool):
                    assert value == recorded[key], (key, value, recorded[key])
                else:
                    close(value, recorded[key])
            if n < count:
                assert not gate['passed'] or gate['sampled_peak_needs_resolution_check']
            else:
                assert gate['passed'] and (not gate['sampled_peak_needs_resolution_check'] or n == 33)
            gates.append(dict(seed=seed, method=method, task=task, nodes=n, **gate))
        previous = current
    close(integral - train_damage ** 2, summary['bound_gap_I_minus_D2'])
    close(drift ** 2 * trace_integral, summary['trace_upper_bound'])
    assert integral <= drift ** 2 * trace_integral + max(1e-8, 1e-5 * abs(drift ** 2 * trace_integral))
    path = dict(seed=seed, method=method, task=task, T=task+1, S_path=sensitivity,
        drift=drift, integral=integral, trace_integral=trace_integral,
        D_train=train_damage, D_test=test_damage, train_samples_per_node=1259,
        measured_nodes=count, source=str(raw_path), protocol_sha256=meta['protocol_sha256'],
        cohort='formal', provenance='reused_original' if task in EXISTING_TASKS else 'new_stage_completion')
    return path, nodes, gates


def collect():
    coverage_path = BASE / 'final_report/coverage.json'
    coverage = read(coverage_path)
    assert coverage['validated_diagnostics'] == coverage['expected_diagnostics'] == 54
    assert len(coverage['diagnostics']) == 54
    sources, checkpoints = [source_record(coverage_path)], {}
    trajectory, paths, nodes, gates, missing = [], [], [], [], []
    for seed in range(3):
        for method in METHODS:
            directory = BASE / f'formal_results/seed{seed}/{method}/trajectory_fisher'
            summary, trajectory_meta, rows, raw_path = load_job(directory, seed, method,
                sources, checkpoints, coverage=coverage, job='trajectory_fisher')
            sequence = extract_trajectory(seed, method, summary, trajectory_meta, rows, raw_path)
            trajectory.extend(sequence)
            for task in range(1,10):
                job = f'path_t{task:02d}'
                root = BASE / 'formal_results' if task in EXISTING_TASKS else EXT / 'results'
                directory = root / f'seed{seed}/{method}/{job}'
                complete_summary = (directory / 'summary.json').is_file()
                complete_receipt = task in EXISTING_TASKS or (directory / 'extension_receipt.json').is_file()
                if not complete_summary or not complete_receipt:
                    missing.append(dict(seed=seed, method=method, task=task, T=task+1,
                        expected_directory=str(directory),
                        raw_states_present=(directory / 'states.jsonl').is_file(),
                        summary_present=complete_summary, completion_receipt_present=complete_receipt))
                    continue
                summary, meta, rows, raw_path = load_job(directory, seed, method,
                    sources, checkpoints, coverage=coverage if task in EXISTING_TASKS else None,
                    job=job, trajectory_meta=trajectory_meta)
                endpoint = next(r for r in sequence if r['task'] == task)
                path, raw_nodes, checks = extract_path(seed, method, task, summary, meta, rows, raw_path, endpoint)
                paths.append(path); nodes.extend(raw_nodes); gates.extend(checks)
    assert len(trajectory) == 60
    assert len(paths) + len(missing) == 54
    assert len({(r['seed'],r['method'],r['task']) for r in paths}) == len(paths)
    unique_sources = {item['path']: item for item in sources}
    evidence = dict(status='ready' if not missing else 'partial',
        expected_paths=54, validated_paths=len(paths), missing_paths=missing,
        expected_new_paths=36, validated_new_paths=sum(p['task'] in NEW_TASKS for p in paths),
        reused_original_paths=sum(p['task'] in EXISTING_TASKS for p in paths),
        trajectory_points=60, measured_path_nodes=len(nodes),
        sources=list(unique_sources.values()), checkpoints=list(checkpoints.values()),
        numerical_gates=gates, errors=[], scientific_identity_verified=True,
        checkpoint_payloads_rehashed_by_this_cpu_collector=False,
        checkpoint_verification='recorded SHA across original trajectory, path manifest, raw states and endpoint recovery; original runner rehashes payloads at job start',
        metric='S_path = Simpson integral of Delta^T F_emp Delta / ||Delta||^2',
        path_scope='1259 old-task training examples; true-label per-example effective Q/V Fisher; fixed 20-class old-task head',
        no_T1_path_value=True,
        rendered_visual_review='not performed by automatic collector')
    return pd.DataFrame(trajectory), pd.DataFrame(paths), pd.DataFrame(nodes), evidence


def captions(paths):
    counts = sorted(paths.measured_nodes.unique())
    node_text = ', '.join(str(int(n)) for n in counts)
    caption = r'''Formal three-seed sequential diagnostics on ImageNet-A. From left to right: effective Q/V weight drift from each method's own Task0 anchor; signed Task0 test cross-entropy change relative to that anchor; Task0 test accuracy; and normalized directional path sensitivity (``Average Fisher'') $\overline S_{\mathrm{path},t}=\int_0^1\Delta_t^\top F_0^{\mathrm{emp}}(W_0+s\Delta_t)\Delta_t\,ds/\|\Delta_t\|_2^2$, with $\Delta_t=W_t-W_0$. The horizontal coordinate is the number of learned tasks $T=t+1$. The first three panels contain all ten measured endpoints $T=1,\ldots,10$. Loss and accuracy use all 307 old-task test examples and the fixed 20-class Task0 head. The last panel contains independently measured anchor-to-endpoint straight chords for every $T=2,\ldots,10$, with no estimate at $T=1$ because the denominator is zero. Each chord's empirical Fisher uses all 1,259 old-task training examples, true-label per-example gradients in the 24 effective Q/V matrices, and no EWC coefficient, trace normalization, floor, or regularization. Adaptive nested Simpson quadrature used NODE_COUNTS measured nodes per chord; all final stability and numerical-bound checks passed. Integration nodes are positions along a chord, not task stages; these chords are not the optimizer's chronological path. Curves show arithmetic means over training seeds 0, 1, and 2, and shaded bands show sample standard deviations ($n=3$, denominator $n-1$). Sensitivity uses each seed's own normalized integral before aggregation and is not an endpoint Fisher trace. The original 18 paths at $t=1,5,9$ are reused; 36 new paths fill $t=2,3,4,6,7,8$.'''.replace('NODE_COUNTS', node_text)
    (OUT / 'B2_all_stages_caption.md').write_text('Figure B2. ' + caption + '\n')
    (OUT / 'B2_all_stages_caption.tex').write_text('\\caption{' + caption + '}\n')
    (OUT / 'B2_all_stages_include.tex').write_text('\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{Figure_B2_sequence_stage_paths_all_stages.pdf}\n    \\input{B2_all_stages_caption.tex}\n    \\label{fig:formal-sequence-all-stage-paths}\n\\end{figure*}\n')


def publish(trajectory, paths, nodes, evidence):
    assert len(paths) == 54 and not evidence['missing_paths']
    assert set(paths['T']) == set(range(2,11))
    OUT.mkdir(parents=True, exist_ok=True)
    for label, frame in (('trajectory_seed_data', trajectory), ('path_seed_data', paths), ('path_node_evidence', nodes)):
        frame.to_csv(OUT / f'B2_all_stages_{label}.csv', index=False)
    font = style.setup_style()
    fig, axes = plt.subplots(1,4,figsize=(7.6,1.85))
    fig.subplots_adjust(left=.064, right=.992, bottom=.26, top=.97, wspace=.49)
    statistics, limits = [], {}
    settings = [(trajectory,'weight_drift','Weight drift','lower right'),
                (trajectory,'signed_test_damage','Loss change of Task 0','upper left'),
                (trajectory,'test_accuracy_percent','Accuracy of Task 0 (%)','lower left'),
                (paths,'S_path','Average Fisher','best')]
    for ax, (frame, value, label, legend) in zip(axes, settings):
        aggregate = style.draw_mean_sd(ax, frame, 'T', value)
        limits[value] = style.limits_from_sd(ax, aggregate, include_zero=value == 'weight_drift')
        aggregate['metric'] = value; statistics.append(aggregate)
        ax.set_ylabel(label,labelpad=3)
        ax.set_xlabel('Number of Tasks',labelpad=3)
        ax.set_xticks([1,2,4,6,8,10]); ax.set_xlim(.7,10.3)
        style.internal_legend(ax,legend)
    pd.concat(statistics,ignore_index=True).to_csv(OUT / 'B2_all_stages_mean_sd.csv',index=False)
    captions(paths)
    evidence.update(font=font, script=source_record(Path(__file__).resolve()),
        shared_style=source_record(STYLE_PATH), y_limits=limits,
        path_x_points=list(range(2,11)), trajectory_x_points=list(range(1,11)),
        negative_damage_retained=bool((trajectory.signed_test_damage < 0).any()),
        last_panel_uses_trace=False, generated_evidence=[source_record(p) for p in sorted(OUT.glob('B2_all_stages_*.csv'))])
    result = style.save_figure(fig,'Figure_B2_sequence_stage_paths_all_stages',evidence)
    result.update(status='complete', completion_scope='CPU source/numerical/layout checks; all 54 measured paths',
                  rendered_visual_review='pending root and independent inspection')
    write_json(REPORT / 'complete.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only',action='store_true',help='Audit available data and report missing paths; never render.')
    mode.add_argument('--execute',action='store_true',help='Require all 54 paths, then export data and render the complete figure.')
    args = parser.parse_args()
    trajectory, paths, nodes, evidence = collect()
    write_json(REPORT / 'coverage.json',evidence)
    if args.execute:
        if evidence['missing_paths']:
            raise SystemExit(f"Incomplete: {evidence['validated_paths']}/54 paths; see {REPORT / 'coverage.json'}")
        evidence = publish(trajectory,paths,nodes,evidence)
    print(json.dumps({key:evidence[key] for key in ('status','expected_paths','validated_paths','validated_new_paths','reused_original_paths','trajectory_points','measured_path_nodes')},indent=2))


if __name__ == '__main__':
    main()
