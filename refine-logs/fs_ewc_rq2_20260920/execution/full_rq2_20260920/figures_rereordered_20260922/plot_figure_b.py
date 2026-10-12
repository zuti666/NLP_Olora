"""Figure B: controlled training neighborhood, d and tau are display notation."""
import plot_style as style


def main():
    pc = style.pc
    font, prior = style.setup(), style.snapshot()
    sources = [style.SOURCE / name for name in
               ['B1_dense_seed0.csv', 'B1_fisher_seed0.csv', 'B1_tolerance_seed0.csv']]
    dense, fisher, tolerance = [pc.pd.read_csv(p) for p in sources]
    assert len(dense) == 214 and len(fisher) == 84 and len(tolerance) == 22
    dense_grid = [i / 4 for i in range(81)] + list(range(25, 51))
    fisher_grid = [0, .5, 1, 2, 5, 10] + list(range(11, 21)) + list(range(25, 51))
    for frame, grid in [(dense, dense_grid), (fisher, fisher_grid)]:
        assert set(frame.seed) == {0}
        for _, rows in frame.groupby('method'):
            assert sorted(rows.radius.tolist()) == grid
        frame['d'] = frame.radius
    tolerance['tau'] = tolerance.epsilon
    # Recompute prefix-safe values only on the measured grid; no interpolated d.
    for row in tolerance.itertuples():
        curve = dense[dense.method == row.method].sort_values('d')
        prefix = pc.np.maximum.accumulate(curve.G_train.to_numpy())
        safe = curve.d.to_numpy()[prefix <= row.tau]
        assert len(safe) and float(safe[-1]) == row.largest_safe_measured_radius == row.mean
    marks = list(range(0, 81, 4)) + list(range(81, 107))
    specs = [
        (dense, 'd', 'train_accuracy_percent', 'Task-1 Train\nAccuracy',
         r'$\overline{A}_1^{\mathrm{nbr}}(d)$ (%)', (70, 100), [70, 80, 90, 100], 'lower left', marks),
        (dense, 'd', 'G_train', 'Mean Positive\nLoss Increase',
         r'$\overline{D}_1^{+}(d)$', (0, .8), [0, .2, .4, .6, .8], 'upper left', marks),
        (fisher, 'd', 'raw_fisher_trace', 'Neighborhood\nFisher Trace',
         r'$\overline{R}_{F,1}^{\mathrm{nbr}}(d)$', (0, 25000), [0, 5000, 10000, 15000, 20000, 25000], 'upper left', 1),
        (tolerance, 'tau', 'mean', 'Measured\nTolerance Radius',
         r'Measured radius $\hat{d}_\tau$', (0, 30), [0, 10, 20, 30], 'lower right', 1),
    ]
    fig, axes = style.canvas()
    plot_values = []
    for ax, (frame, x, metric, title, ylabel, ylim, ticks, loc, marks) in zip(axes, specs):
        style.draw(ax, frame, x, metric, markevery=marks)
        if x == 'd':
            xlabel, xlim, xticks = r'Perturbation radius $d$', (-1.25, 51.25), [0, 25, 50]
        else:
            xlabel, xlim, xticks = r'Allowed loss increase $\tau$', (0, .104), [0, .025, .05, .075, .1]
        style.label(ax, title, ylabel, xlabel, xlim, xticks, ylim, ticks, loc)
        if x == 'tau':
            ax.set_xticklabels(['0', '0.025', '0.05', '0.075', '0.1'])
        values = frame[['method', x, metric]].rename(columns={x: 'x_value', metric: 'value'})
        values['x_symbol'], values['source_metric'] = x, metric
        plot_values.append(values)
    values = pc.pd.concat(plot_values, ignore_index=True)
    assert len(values) == 534
    values.to_csv(style.HERE / 'Figure_B_plot_values.csv', index=False)
    style.finish(fig, 'Figure_B_controlled_neighborhood', dict(
        source_files=[{'path': str(p), 'sha256': pc.sha(p)} for p in sources],
        previous_figure='B1', anchor='displayed Task 1 is original Task0, seed0',
        notation_mapping={'source_radius': 'd', 'source_epsilon': 'tau', 'source_G_train': 'Dbar_1^+(d)'},
        controlled_displacement='d = ||d u_k|| in effective Q/V weight space; full matrix perturbations',
        all_512_direct_curve_values_unchanged=True, all_22_tolerance_values_recomputed=True,
        positive_part_before_direction_mean=True, dense_directions=50, dense_samples=1259,
        dense_measured_displacements=dense_grid, fisher_directions=16, fisher_samples=256,
        fisher_measured_displacements=fisher_grid, unmeasured_d=[21, 22, 23, 24],
        raw_fisher_has_lambda=False, interpolation_used_for_measurements_or_tolerance=False,
        statistics_scope='one training seed (seed0); direction means shown; cross-seed SD is undefined',
        connecting_lines='visual guides between measured points; missing d=21..24 not imputed'),
        prior, font, __file__)


if __name__ == '__main__':
    main()
