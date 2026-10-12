"""Figure A: realized future learning, displayed with one-based task indexing."""
import plot_style as style


def main():
    pc = style.pc
    font, prior = style.setup(), style.snapshot()
    sources = [style.SOURCE / f'B2_{suffix}.csv' for suffix in
               ['trajectory_seed_data', 'path_seed_data', 'mean_sd']]
    trajectory, paths, old_means = [pc.pd.read_csv(p) for p in sources]
    assert len(trajectory) == 60 and len(paths) == 54
    for frame in [trajectory, paths]:
        assert (frame['T'] == frame.task + 1).all()
        frame['display_task'] = frame['T']
    assert set(trajectory.display_task) == set(range(1, 11))
    assert set(paths.display_task) == set(range(2, 11))
    pc.np.testing.assert_allclose(paths.S_path, paths.integral / paths.drift**2,
                                rtol=1e-12, atol=1e-14)
    specs = [
        (trajectory, 'weight_drift', 'Weight Drift', r'Effective-weight drift $d_t$',
         (0, 1.6), [0, .4, .8, 1.2, 1.6], 'lower right'),
        (trajectory, 'signed_test_damage', 'Task-1 Loss Change', r'$D_{1,t}^{\mathrm{test}}$',
         (-.05, .3), [-.05, 0, .1, .2, .3], 'upper left'),
        (trajectory, 'test_accuracy_percent', 'Task-1 Accuracy', r'$A_{1,t}^{\mathrm{test}}$ (%)',
         (75, 95), [75, 80, 85, 90, 95], 'lower left'),
        (paths, 'S_path', 'Path Fisher', 'Path Fisher sensitivity',
         (0, .3), [0, .1, .2, .3], 'upper left'),
    ]
    fig, axes = style.canvas()
    groups = []
    for ax, (frame, metric, title, ylabel, ylim, ticks, loc) in zip(axes, specs):
        mean = pc.mean_sd(frame, 'T', metric)
        mean['metric'] = metric
        groups.append(mean)
        style.draw(ax, mean, 'T', 'mean')
        style.label(ax, title, ylabel, 'Number of learned tasks', (.7, 10.3),
                    [1, 2, 4, 6, 8, 10], ylim, ticks, loc)
    means = pc.pd.concat(groups, ignore_index=True)
    keys, cols = ['metric', 'method', 'T'], ['metric', 'method', 'T', 'mean', 'sd', 'n']
    pc.pd.testing.assert_frame_equal(means[cols].sort_values(keys).reset_index(drop=True),
                                    old_means[cols].sort_values(keys).reset_index(drop=True),
                                    check_dtype=False, rtol=1e-12, atol=1e-14)
    assert len(means) == 78
    means.rename(columns={'T': 'display_task_t'}).to_csv(style.HERE / 'Figure_A_plot_values.csv', index=False)
    style.finish(fig, 'Figure_A_realized_future', dict(
        source_files=[{'path': str(p), 'sha256': pc.sha(p)} for p in sources],
        previous_figure='B2', display_task_mapping='displayed t = source task + 1 = source T',
        anchor='displayed Task 1 corresponds to original Task0; same samples/head/checkpoint',
        drift_definition='d_t = ||Delta w_t||_2; Delta w_t = w_t - w_1 in effective Q/V coordinates',
        path_definition='Sbar_F,1^path(t) = integral_0^1 v_t^T F_1,train^emp(w_1+s Delta w_t) v_t ds; v_t=Delta w_t/d_t',
        aggregation='mean over three seeds of each seed own integral/drift^2',
        statistics_scope='three independent training seeds; means shown; sample SD retained, no bands',
        all_78_mean_sd_groups_unchanged=True, all_54_path_ratios_checked=True,
        path_undefined_at_displayed_task_1=True, path_legend='upper left'), prior, font, __file__)


if __name__ == '__main__':
    main()
