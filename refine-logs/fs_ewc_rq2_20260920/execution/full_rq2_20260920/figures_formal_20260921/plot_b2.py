"""Rebuild formal three-training-seed B2 from audited raw diagnostic records.

Run with the project's Python environment; CPU only, no model evaluation.
"""
from plot_common import *


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
        assert np.isfinite(value), value


def simpson(values):
    assert len(values) >= 3 and len(values) % 2 == 1
    return (values[0] + values[-1] + 4 * sum(values[1:-1:2])
            + 2 * sum(values[2:-1:2])) / (3 * (len(values) - 1))


def load_job(seed, method, job, sources, checkpoints):
    summary, meta, receipt = audited_summary(seed, method, job)
    sources.extend(receipt)
    directory = BASE / f'formal_results/seed{seed}/{method}/{job}'
    raw_path = directory / 'states.jsonl'
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(rows) == summary['states_completed']
    assert len({r['state_id'] for r in rows}) == len(rows)
    sanity_path = directory / 'sanity.json'
    sanity = read(sanity_path)
    sources.extend(dict(path=str(p), sha256=sha(p)) for p in (raw_path, sanity_path))
    assert sanity['passed'] and sanity['predictions_match']
    assert sanity['protocol_sha256'] == meta['protocol_sha256']
    assert sanity['effective_sha256'] == meta['anchor_effective_sha256']
    assert summary['end_restoration']['predictions_match']
    assert summary['end_restoration']['effective_sha256'] == meta['anchor_effective_sha256']
    assert sanity['max_logit_error'] == summary['end_restoration']['max_logit_error'] == 0
    assert meta['counts']['train'] == 1259 and meta['counts']['test'] == 307
    for task, checkpoint_path in meta['checkpoints'][method].items():
        p = Path(checkpoint_path)
        recorded_sha = meta['checkpoint_sha256'][method][task]
        assert p.is_file() and len(recorded_sha) == 64
        record = dict(path=str(p), recorded_sha256=recorded_sha, bytes=p.stat().st_size,
                      verification='digest agrees across manifest, raw states, and endpoint recovery; file exists; original training audit retained')
        if checkpoint_path in checkpoints:
            assert checkpoints[checkpoint_path] == record
        checkpoints[checkpoint_path] = record
    for row in rows:
        finite_tree(row)
        assert row['seed'] == seed and row['method'] == method
        assert row['anchor_restored_exactly']
        assert row['protocol_sha256'] == meta['protocol_sha256']
        assert row['checkpoint_sha256'] == meta['checkpoint_sha256'][method]
        for key in ('coordinate_sha256', 'data_manifest_sha256'):
            assert row[key] == meta[key]
        assert row['metrics']['test']['samples'] == 307
        assert row['metrics']['train']['samples'] == 1259
        close(row['effective_weight_drift'] ** 2, row['effective_weight_drift_squared'])
        for scope, fisher in row['fisher_by_scope'].items():
            assert fisher['samples'] == {'train': 1259, 'test': 307}[scope]
            assert not fisher['normalized'] and fisher['floor'] is None and fisher['regularizer'] is None
    finite_tree(summary)
    return summary, meta, rows, raw_path


def endpoint_check(recovery, meta, method, task):
    assert recovery['passed'] and recovery['task'] == task
    assert recovery['checkpoint'] == meta['checkpoints'][method][str(task)]
    assert recovery['checkpoint_sha256'] == meta['checkpoint_sha256'][method][str(task)]
    assert recovery['max_logit_error'] < 3e-5
    assert all(abs(e) < 3e-6 for e in recovery['local_loss_errors'].values())


def extract():
    coverage_path = BASE / 'final_report/coverage.json'
    coverage = read(coverage_path)
    assert coverage['validated_diagnostics'] == coverage['expected_diagnostics'] == 54
    assert len(coverage['diagnostics']) == 54
    sources = [dict(path=str(coverage_path), sha256=sha(coverage_path))]
    checkpoints, trajectory, paths, nodes, gates = {}, [], [], [], []
    for seed in range(3):
        for method in METHODS:
            summary, meta, rows, raw_path = load_job(seed, method, 'trajectory_fisher', sources, checkpoints)
            assert summary['task_count'] == 10 and len(rows) == 11
            assert summary['train_fisher_samples'] == 1259
            assert summary['test_fisher_samples'] == 307
            by_id = {r['state_id']: r for r in rows}
            assert sorted(r['task'] for r in summary['tasks']) == list(range(10))
            origin = by_id['r0']
            for item in summary['tasks']:
                t = item['task']
                raw = by_id[f'task{t:02d}']
                assert raw['point']['kind'] == 'trajectory' and raw['point']['task'] == t
                assert raw['metrics'] == item['metrics']
                close(raw['effective_weight_drift'], item['effective_weight_drift'])
                endpoint_check(item['endpoint_recovery'], meta, method, t)
                local = item['direct_local_and_seen']['test']['local']
                assert local['classes'] == 20 and local['samples'] == 307
                close(raw['metrics']['test']['accuracy'], local['accuracy'])
                close(raw['metrics']['test']['loss'], local['loss'], atol=3e-6)
                damage = raw['metrics']['test']['loss'] - origin['metrics']['test']['loss']
                close(damage, item['D0']['test'])
                trajectory.append(dict(seed=seed, method=method, task=t, T=t+1,
                    weight_drift=raw['effective_weight_drift'], test_CE=raw['metrics']['test']['loss'],
                    anchor_test_CE=origin['metrics']['test']['loss'], signed_test_damage=damage,
                    test_accuracy_percent=raw['metrics']['test']['accuracy'], test_samples=307,
                    fixed_head_classes=20, source=str(raw_path), source_state_id=raw['state_id'],
                    protocol_sha256=meta['protocol_sha256'],
                    checkpoint=meta['checkpoints'][method][str(t)],
                    checkpoint_sha256=meta['checkpoint_sha256'][method][str(t)]))
            for t in (1, 5, 9):
                summary, meta, rows, raw_path = load_job(seed, method, f'path_t{t:02d}', sources, checkpoints)
                assert summary['target_task'] == meta['path_task'] == t
                assert summary['samples'] == 1259 and summary['selected_nodes'] == len(rows) == 9
                assert summary['numerical_gate_passed'] and summary['quadrature_converged']
                assert summary['bound_passed_with_numeric_tolerance']
                endpoint_check(summary['endpoint_recovery'], meta, method, t)
                name = f'{method}_t{t:02d}'
                raw = sorted(rows, key=lambda r: r['amount'])
                assert [r['amount'] for r in raw] == [i / 8 for i in range(9)]
                q, trace = [], []
                for row, item in zip(raw, summary['path']):
                    fisher = row['fisher_by_scope']['train']
                    assert set(fisher['q']) == {name}
                    close(row['amount'], item['s'])
                    close(fisher['q'][name], item['q'])
                    close(fisher['raw_trace'], item['raw_trace'])
                    close(fisher['subset_loss'], item['train_loss_for_bound'])
                    assert row['metrics'] == item['metrics']
                    if row['amount']:
                        assert row['point']['direction_sha256'] == meta['actual_direction_sha256'][name]
                    q.append(fisher['q'][name]); trace.append(fisher['raw_trace'])
                    nodes.append(dict(seed=seed, method=method, task=t, T=t+1, s=row['amount'],
                        q=fisher['q'][name], raw_trace=fisher['raw_trace'],
                        train_loss_for_bound=fisher['subset_loss'], train_samples=fisher['samples'],
                        test_CE=row['metrics']['test']['loss'], source=str(raw_path),
                        source_state_id=row['state_id'], protocol_sha256=meta['protocol_sha256']))
                i9, i5, trace9, trace5 = simpson(q), simpson(q[::2]), simpson(trace), simpson(trace[::2])
                close(i9, summary['integral']); close(trace9, summary['trace_integral'])
                for n, value, tr in ((5, i5, trace5), (9, i9, trace9)):
                    close(value, summary['integrals_simpson'][str(n)]['q_integral'])
                    close(tr, summary['integrals_simpson'][str(n)]['trace_integral'])
                drift = meta['actual_drift'][name]
                close(drift, summary['drift']); close(drift ** 2, summary['drift_squared'])
                close(drift, raw[-1]['effective_weight_drift'], atol=1e-7)
                sensitivity = i9 / drift ** 2
                close(sensitivity, summary['S_path'])
                train_damage = raw[-1]['fisher_by_scope']['train']['subset_loss'] - raw[0]['fisher_by_scope']['train']['subset_loss']
                test_damage = raw[-1]['metrics']['test']['loss'] - raw[0]['metrics']['test']['loss']
                close(train_damage, summary['D_train']); close(test_damage, summary['D_test'])
                traj = next(r for r in trajectory if (r['seed'],r['method'],r['task']) == (seed,method,t))
                close(drift, traj['weight_drift'], atol=1e-7)
                close(test_damage, traj['signed_test_damage'], atol=3e-6)
                cfg = meta['config']
                stability_tolerance = cfg['quadrature_atol'] + cfg['quadrature_rtol'] * max(abs(i9), abs(i5))
                bound_tolerance = max(1e-8, cfg['bound_rtol'] * abs(i9))
                unresolved_peak = any(q[i] > cfg['path_peak_neighbor_factor'] * max(q[i-1],q[i+1],1e-12) for i in range(1,8))
                assert abs(i9 - i5) <= stability_tolerance and train_damage ** 2 <= i9 + bound_tolerance
                assert not unresolved_peak and summary['quadrature_gates'][0]['passed']
                close(stability_tolerance, summary['quadrature_gates'][0]['stability_tolerance'])
                close(bound_tolerance, summary['quadrature_gates'][0]['bound_tolerance'])
                close(i9 - train_damage ** 2, summary['bound_gap_I_minus_D2'])
                close(drift ** 2 * trace9, summary['trace_upper_bound'])
                assert i9 <= drift ** 2 * trace9 + max(1e-8,1e-5 * abs(drift ** 2 * trace9))
                paths.append(dict(seed=seed, method=method, task=t, T=t+1, S_path=sensitivity,
                    drift=drift, integral=i9, integral_5_nodes=i5, trace_integral=trace9,
                    D_train=train_damage, D_test=test_damage, train_samples_per_node=1259,
                    measured_nodes=9, source=str(raw_path), protocol_sha256=meta['protocol_sha256']))
                gates.append(dict(seed=seed, method=method, task=t, absolute_5_9_difference=abs(i9-i5),
                    stability_tolerance=stability_tolerance, bound_tolerance=bound_tolerance,
                    bound_gap=i9-train_damage**2, unresolved_peak=unresolved_peak, passed=True))
    trajectory, paths, nodes = map(pd.DataFrame, (trajectory, paths, nodes))
    assert len(trajectory) == 60 and len(paths) == 18 and len(nodes) == 162
    assert set(paths['T']) == {2,6,10} and paths.S_path.notna().all()
    for method in METHODS:
        assert set(trajectory.query('method == @method')['T']) == set(range(1,11))
    for label, frame in (('trajectory_seed_data',trajectory),('path_seed_data',paths),('path_node_evidence',nodes)):
        frame.to_csv(HERE / f'B2_{label}.csv', index=False)
    return trajectory, paths, dict(sources=sources, checkpoints=list(checkpoints.values()),
        path_numerical_gates=gates, raw_states_reloaded=228, raw_trajectory_points=60,
        measured_path_points=18, measured_path_nodes=162, valid_diagnostic_coverage=54,
        checkpoint_payloads_rehashed_by_this_plot=False,
        metric_recomputation='Signed test CE differences and 5-/9-node Simpson q/trace integrals recomputed from states.jsonl; S_path=I/drift^2; no GPU reevaluation')


def caption():
    text = r'''Formal three-seed sequence diagnostics on ImageNet-A. From left to right: effective Q/V weight drift $r_T=\|\Delta_T\|_2$ from each method's own Task0 anchor; signed Task0 test cross-entropy damage $D_T^{\mathrm{test}}=L_0^{\mathrm{test}}(w_T)-L_0^{\mathrm{test}}(w_1)$ (including negative changes); Task0 test accuracy; and chord-normalized directional empirical-Fisher sensitivity (``Average Fisher'') $S_T=I_T/r_T^2$, where $\Delta_T=w_T-w_1$ and $I_T=\int_0^1\Delta_T^\top F_0^{\mathrm{emp}}(w_1+s\Delta_T)\Delta_T\,ds$. Here $T=t+1$ is the number of learned tasks; $T=1$ denotes the Task0 anchor. The first three panels include all ten measured endpoints $T=1,\ldots,10$. Test loss and accuracy use all 307 Task0 test examples and each method's fixed 20-class Task0 head. $F_0^{\mathrm{emp}}$ is the unnormalized, true-label, per-example empirical Fisher over all 1,259 Task0 training examples in the 24 effective Q/V matrices; Average Fisher is the normalized directional integral, not a Fisher trace. Each sensitivity estimate integrates its own straight chord from the anchor using nine measured Simpson nodes, with the five-/nine-node stability and numerical bound checks passed. Path measurements exist only at $t=1,5,9$ ($T=2,6,10$); the connecting segments are visual guides and represent no additional measurements. $S_1$ is undefined because $r_1=0$ and is omitted. All curves show arithmetic means across three training seeds (0, 1, 2); bands in the first three panels and error bars at the three measured path stages show sample standard deviations ($n=3$, denominator $n-1$).'''
    (HERE / 'B2_caption.md').write_text('Figure B2. ' + text + '\n')
    (HERE / 'B2_caption.tex').write_text('\\caption{' + text + '}\n')
    (HERE / 'B2_include.tex').write_text('\\begin{figure*}[t]\n    \\centering\n    \\includegraphics[width=\\textwidth]{Figure_B2_sequence_stage_paths.pdf}\n    \\input{B2_caption.tex}\n    \\label{fig:formal-sequence-paths}\n\\end{figure*}\n')


def main():
    font = setup_style()
    trajectory, paths, provenance = extract()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.064, right=.992, bottom=.26, top=.97, wspace=.49)
    statistics, limits = [], {}
    settings = [('weight_drift', 'Weight drift', 'lower right'),
                ('signed_test_damage', 'Loss change of Task 0', 'upper left'),
                ('test_accuracy_percent', 'Accuracy of Task 0 (%)', 'lower left')]
    for ax, (value, label, legend_loc) in zip(axes[:3], settings):
        aggregate = draw_mean_sd(ax, trajectory, 'T', value)
        limits[value] = limits_from_sd(ax, aggregate, include_zero=value == 'weight_drift')
        aggregate['metric'] = value; statistics.append(aggregate)
        ax.set_ylabel(label, labelpad=3)
        ax.set_xticks([1,2,4,6,8,10]); ax.set_xlim(.7,10.3)
        internal_legend(ax, legend_loc)
    aggregate = draw_mean_sd(axes[3], paths, 'T', 'S_path', bands=False, errorbars=True)
    limits['S_path'] = limits_from_sd(axes[3], aggregate)
    aggregate['metric'] = 'S_path'; statistics.append(aggregate)
    axes[3].set_ylabel('Average Fisher', labelpad=3)
    axes[3].set_xticks([2,6,10]); axes[3].set_xlim(.7,10.3)
    internal_legend(axes[3], 'upper left')
    for ax in axes:
        ax.set_xlabel('Number of Tasks', labelpad=3)
    pd.concat(statistics, ignore_index=True).to_csv(HERE / 'B2_mean_sd.csv', index=False)
    caption()
    provenance.update(font=font, script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        shared_style=dict(path=str(HERE / 'plot_common.py'),sha256=sha(HERE / 'plot_common.py')),
        generated_evidence=[dict(path=str(p),sha256=sha(p)) for p in sorted(HERE.glob('B2_*.csv'))],
        y_limits=limits, path_x_points=[2,6,10], trajectory_x_points=list(range(1,11)),
        first_three_band='mean +/- sample SD', last_panel_errorbar='mean +/- sample SD, no band across missing path stages',
        negative_damage_retained=bool((trajectory.signed_test_damage < 0).any()),
        negative_damage_count=int((trajectory.signed_test_damage < 0).sum()),
        axis_titles_inside=False, last_panel_uses_trace=False)
    (HERE / 'B2_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    result = save_figure(fig, 'Figure_B2_sequence_stage_paths', provenance)
    print(json.dumps({key: result[key] for key in ('status','outputs','embedded_fonts','figure_inches','y_limits')},indent=2))


if __name__ == '__main__':
    main()
