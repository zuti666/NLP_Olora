#!/usr/bin/env python3
"""CPU-only redraw of validated seed0 B1 measurements; preserves prior figures."""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "figures/radius50_integer_seed0_dense50_fisher16"
PREVIOUS = ROOT / "figures/replot_20260922_v2"
EPSILONS = [.005, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq2-b1-replot-20260922-mpl")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measured_tolerances(pc, dense):
    """Invert panel 1's exact mean-positive-damage curve on the measured grid."""
    rows = []
    for (method, seed), curve in dense.groupby(["method", "seed"]):
        curve = curve.sort_values("radius")
        radii = curve.radius.to_numpy(dtype=float)
        damage = curve.G_train.to_numpy(dtype=float)
        prefix = pc.np.maximum.accumulate(damage)
        for epsilon in EPSILONS:
            failures = pc.np.flatnonzero(prefix > epsilon)
            first_failed = int(failures[0]) if len(failures) else None
            assert first_failed != 0
            last_safe = len(radii) - 1 if first_failed is None else first_failed - 1
            rows.append(dict(method=method, seed=int(seed), epsilon=epsilon,
                largest_safe_measured_radius=float(radii[last_safe]),
                G_at_last_safe=float(damage[last_safe]),
                prefix_G_at_last_safe=float(prefix[last_safe]),
                first_failed_measured_radius=None if first_failed is None else float(radii[first_failed]),
                G_at_first_failed=None if first_failed is None else float(damage[first_failed]),
                right_censored=first_failed is None,
                plotted=first_failed is not None,
                mean=None if first_failed is None else float(radii[last_safe]),
                n_training_seeds=1, interpolation_used=False))
    return pc.pd.DataFrame(rows)


def main():
    pub = load("validated_b1_publisher", ROOT / "publish_radius50.py")
    pc = load("b1_plot_style", ROOT.parent / "figures_formal_20260921/plot_common.py")
    receipt_path = ROOT / "report/radius50_complete.json"
    receipt = pub.read(receipt_path)
    assert receipt["status"] == "complete"
    assert receipt["validated_jobs"] == receipt["expected_jobs"] == 4
    assert receipt["automatic_numeric_and_layout_checks"] == "passed"
    # Check the raw-measurement and source-code chain as well as the plotted CSVs.
    for item in receipt["source_files"] + receipt["output_files"]:
        assert pub.sha(item["path"]) == item["sha256"], item["path"]
    prior_hashes = {str(p): pub.sha(p) for directory in (SOURCE, PREVIOUS)
                    for p in directory.iterdir() if p.is_file()}
    dense = pc.pd.read_csv(SOURCE / "dense_training_per_seed.csv")
    fisher = pc.pd.read_csv(SOURCE / "fisher_training_per_seed.csv")
    expected_dense = [i / 4 for i in range(81)] + list(range(25, 51))
    expected_fisher = pub.FISHER_RADII + list(range(25, 51))
    for frame, grid, directions in ((dense, expected_dense, 50), (fisher, expected_fisher, 16)):
        assert set(frame.seed) == {0} and set(frame.method) == {"sgd", "gam"}
        assert not frame.duplicated(["method", "seed", "radius"]).any()
        assert (frame.train_loss_samples == 1259).all()
        for _, curve in frame.groupby("method"):
            assert sorted(curve.radius.tolist()) == grid
        assert (frame.loc[frame.radius > 0, "directions"] == directions).all()
        assert (frame.loc[frame.radius == 0, "directions"] == 1).all()
    assert (fisher.fisher_samples == 256).all()
    assert pc.np.isfinite(dense[["G_train", "train_accuracy_percent"]]).all().all()
    assert pc.np.isfinite(fisher.raw_fisher_trace).all()
    old_bounds, _ = pub.tolerance_tables(pc, dense)
    stored_bounds = pc.pd.read_csv(SOURCE / "tolerance_bounds_per_seed.csv")
    pc.pd.testing.assert_frame_equal(old_bounds, stored_bounds, check_dtype=False, rtol=1e-12)
    tolerance = measured_tolerances(pc, dense)
    for row in old_bounds.itertuples():
        match = tolerance[(tolerance.method == row.method) & (tolerance.epsilon == row.epsilon)].iloc[0]
        assert match.largest_safe_measured_radius == row.largest_safe_measured_radius
        assert match.first_failed_measured_radius == row.first_failed_measured_radius
    dense.to_csv(HERE / "dense_training.csv", index=False)
    fisher.to_csv(HERE / "fisher_training.csv", index=False)
    tolerance.to_csv(HERE / "tolerance_bounds.csv", index=False)
    tolerance.to_csv(HERE / "tolerance_plot.csv", index=False)
    pc.HERE = HERE
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize': 8.2, 'xtick.labelsize': 7.2,
                           'ytick.labelsize': 7.2, 'axes.labelpad': 1.5})
    fig, axes = pc.plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.30, top=.966, wspace=.56)
    specifications = [
        (dense, "G_train", r"Loss change $\overline{D_0(\boldsymbol{\theta},\boldsymbol{\Delta})}$", "upper left"),
        (dense, "train_accuracy_percent", "accuracy of task 0 (%)", "lower left"),
        (fisher, "raw_fisher_trace", r"Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$", "upper left"),
    ]
    plotted = []
    for ax, (frame, metric, label, legend_loc) in zip(axes, specifications):
        markers = 1 if metric == "raw_fisher_trace" else list(range(0, 81, 4)) + list(range(81, len(expected_dense)))
        values = pub.draw_single_seed(pc, ax, frame, metric, markers)
        for line in ax.lines:
            line.set_markersize(1.6)
            line.set_markeredgewidth(.35)
        pc.limits_from_sd(ax, values.assign(sd=0.), include_zero=metric == "G_train")
        ax.set(xlabel=r"Weight drift $r$", ylabel=label,
               xlim=(-1.25, 51.25), xticks=[0, 25, 50])
        legend = pc.internal_legend(ax, legend_loc)
        for label in legend.get_texts():
            label.set_fontsize(7.2)
        values["metric"] = metric
        plotted.append(values)
    pc.pd.concat(plotted, ignore_index=True).to_csv(HERE / "plotted_values.csv", index=False)
    ax = axes[3]
    for method in pub.METHODS:
        curve = tolerance[tolerance.method == method].sort_values("epsilon")
        ax.plot(curve.epsilon, curve["mean"], color=pc.COLORS[method],
                linestyle=pc.LINES[method], linewidth=1.25,
                marker=pc.MARKERS[method], markersize=2.7, label=pc.LABELS[method], zorder=3)
    upper = max(float(tolerance.loc[tolerance.plotted, "mean"].max()) * 1.12, 5)
    ax.set(xlabel=r"Allowed damage $\epsilon$", ylabel=r"Tolerance radius $\hat r_\epsilon$",
           xlim=(0, .104), ylim=(-.55, upper), yticks=list(range(0, int(upper) + 1, 5)),
           xticks=EPSILONS, xticklabels=[f'{value:g}' for value in EPSILONS])
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.6)
    # The 0.005/0.01 pair is closer than the other thresholds. Right-align
    # only the first tick's text, keeping every tick and data point in place.
    ax.get_xticklabels()[0].set_ha('right')
    legend = pc.internal_legend(ax, "lower right")
    for label in legend.get_texts():
        label.set_fontsize(7.2)
    # Independent automatic label placement keeps the first three x labels near
    # their axes instead of pushing them down to match the rotated epsilon ticks.
    name = "Figure_B1_neighborhood_seed0_radius50_20260922_v3"
    metadata = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    font=font, source_receipt=pub.file_receipt(receipt_path),
                    source_csvs=[pub.file_receipt(SOURCE / p) for p in
                                 ("dense_training_per_seed.csv", "fisher_training_per_seed.csv", "tolerance_bounds_per_seed.csv")],
                    generation_code=[pub.file_receipt(Path(__file__)), pub.file_receipt(ROOT / "publish_radius50.py"),
                                     pub.file_receipt(ROOT.parent / "figures_formal_20260921/plot_common.py")],
                    training_seeds=[0], dense_directions=50, fisher_directions=16,
                    dense_examples=1259, fisher_examples=256,
                    dense_radii=expected_dense, fisher_radii=expected_fisher,
                    thresholds=EPSILONS, tolerance_recomputed=True,
                    tolerance_source_is_panel1_dataframe=True,
                    axis_label_fontsize=8.2, tick_fontsize=7.2, legend_fontsize=7.2,
                    first_three_marker_size_points=1.6, first_three_marker_edge_width_points=.35,
                    fourth_panel_xtick_fontsize=6.6, first_epsilon_tick_text_alignment='right',
                    x_labelpad_points=1.5, x_labels_aligned_across_panels=False,
                    data_cohort="formal ImageNet-A; ViT-B/16 rank-10 Q/V LoRA",
                    no_new_model_evaluations=True, prior_artifacts_preserved=True)
    result = pub.save_single_seed_figure(pc, fig, name, metadata)
    for path, digest in prior_hashes.items():
        assert pub.sha(path) == digest, f"Prior output changed: {path}"
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
