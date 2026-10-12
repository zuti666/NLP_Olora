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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq2-b1-replot-20260922-mpl")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    prior_hashes = {str(p): pub.sha(p) for p in SOURCE.iterdir() if p.is_file()}
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
    bounds, tolerance = pub.tolerance_tables(pc, dense)
    stored_bounds = pc.pd.read_csv(SOURCE / "tolerance_bounds_per_seed.csv")
    pc.pd.testing.assert_frame_equal(bounds, stored_bounds, check_dtype=False, rtol=1e-12)
    dense.to_csv(HERE / "dense_training.csv", index=False)
    fisher.to_csv(HERE / "fisher_training.csv", index=False)
    bounds.to_csv(HERE / "tolerance_bounds.csv", index=False)
    tolerance.drop(columns="sd").to_csv(HERE / "tolerance_plot.csv", index=False)
    pc.HERE = HERE
    font = pc.setup_style()
    fig, axes = pc.plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.32, top=.966, wspace=.56)
    specifications = [
        (dense, "G_train", r"Loss change $\overline{D_t(\boldsymbol{\theta},\boldsymbol{\Delta})}$", "upper left"),
        (dense, "train_accuracy_percent", "Accuracy (%)", "lower left"),
        (fisher, "raw_fisher_trace", r"Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$", "upper left"),
    ]
    plotted = []
    for ax, (frame, metric, label, legend_loc) in zip(axes, specifications):
        markers = 1 if metric == "raw_fisher_trace" else list(range(0, 81, 4)) + list(range(81, len(expected_dense)))
        values = pub.draw_single_seed(pc, ax, frame, metric, markers)
        pc.limits_from_sd(ax, values.assign(sd=0.), include_zero=metric == "G_train")
        ax.set(xlabel=r"Weight drift $r$", ylabel=label,
               xlim=(-1.25, 51.25), xticks=[0, 25, 50])
        pc.internal_legend(ax, legend_loc)
        values["metric"] = metric
        plotted.append(values)
    pc.pd.concat(plotted, ignore_index=True).to_csv(HERE / "plotted_values.csv", index=False)
    ax = axes[3]
    for method in pub.METHODS:
        curve = tolerance[tolerance.method == method].sort_values("epsilon")
        ax.plot(curve.epsilon, curve["mean"], color=pc.COLORS[method],
                linestyle=pc.LINES[method], linewidth=1.25,
                marker=pc.MARKERS[method], markersize=3.1, label=pc.LABELS[method], zorder=3)
    upper = max(float(tolerance.loc[tolerance.plotted, "mean"].max()) * 1.12, 5)
    ax.set(xlabel=r"Allowed damage $\epsilon$", ylabel=r"Tolerance radius $\hat r_\epsilon$",
           xlim=(0, .0415), ylim=(-.55, upper), yticks=list(range(0, int(upper) + 1, 5)),
           xticks=pub.EPSILONS, xticklabels=["0.005", "0.01", "0.02", "0.04"])
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    pc.internal_legend(ax, "lower right")
    fig.align_xlabels(axes)
    name = "Figure_B1_neighborhood_seed0_radius50_20260922"
    metadata = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    font=font, source_receipt=pub.file_receipt(receipt_path),
                    source_csvs=[pub.file_receipt(SOURCE / p) for p in
                                 ("dense_training_per_seed.csv", "fisher_training_per_seed.csv", "tolerance_bounds_per_seed.csv")],
                    generation_code=[pub.file_receipt(Path(__file__)), pub.file_receipt(ROOT / "publish_radius50.py"),
                                     pub.file_receipt(ROOT.parent / "figures_formal_20260921/plot_common.py")],
                    training_seeds=[0], dense_directions=50, fisher_directions=16,
                    dense_examples=1259, fisher_examples=256,
                    dense_radii=expected_dense, fisher_radii=expected_fisher,
                    thresholds=pub.EPSILONS, tolerance_recomputed=True,
                    data_cohort="formal ImageNet-A; ViT-B/16 rank-10 Q/V LoRA",
                    no_new_model_evaluations=True, prior_artifacts_preserved=True)
    result = pub.save_single_seed_figure(pc, fig, name, metadata)
    for path, digest in prior_hashes.items():
        assert pub.sha(path) == digest, f"Prior output changed: {path}"
    print(json.dumps(result["outputs"], indent=2))


if __name__ == "__main__":
    main()
