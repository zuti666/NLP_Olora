"""CPU-only Figure B redraw, with raw-state checks and no measurement changes.

Run from any directory with Python, NumPy, pandas and Matplotlib installed.
The adjacent paper_plot_style.py provides the shared publication style.
"""
from paper_plot_style import *
from matplotlib import font_manager
from matplotlib.ticker import NullLocator, MaxNLocator

SPLITS = ("train", "test")
SPLIT_STYLES = {"train": "-", "test": (0, (4, 2))}
FIGURE_NAME = "Figure_B_neighborhood_merged"


def verified_data():
    neighborhood_path = REPORT / "B_neighborhood_deterministic.csv"
    tolerance_path = REPORT / "B_tolerance_deterministic.csv"
    config_path = EXECUTION / "diagnostic_config_recovery1.json"
    config = json.loads(config_path.read_text())
    assert Path(config["data_root"]).name == "imagenet-a"
    assert config["training_seed"] == 0 and config["directions"] == 16
    assert config["fisher_samples"] == 256
    neighborhood = pd.read_csv(neighborhood_path, float_precision="round_trip")
    tolerance = pd.read_csv(tolerance_path, float_precision="round_trip")
    for frame in (neighborhood, tolerance):
        assert set(frame.seed) == {0}
        assert set(frame["transform"]) == {"deterministic_test"}
        assert set(frame.method) == set(METHODS)
        assert set(frame.split) == set(SPLITS)
    assert len(neighborhood) == 28 and len(tolerance) == 40
    assert not neighborhood.duplicated(["method", "split", "radius"]).any()
    assert not tolerance.duplicated(["method", "split", "metric", "epsilon"]).any()
    assert set(tolerance.metric) == {"G", "Q95"}
    radii = np.sort(neighborhood.radius.unique())
    np.testing.assert_array_equal(radii, config["radii"])
    np.testing.assert_array_equal(np.sort(tolerance.epsilon.unique()), [.001, .005, .01, .02, .05])
    directions, trace_records, sources, manifests = [], [], [neighborhood_path, tolerance_path, config_path], {}
    direction_hashes = {}
    for method in METHODS:
        path = RAW / method / "neighborhood" / "states.jsonl"
        manifest_path = path.parent / "manifest.json"
        sources.extend([path, manifest_path])
        manifest = json.loads(manifest_path.read_text())
        manifests[method] = manifest
        assert manifest["method"] == method and manifest["seed"] == 0
        assert manifest["config"]["radii"] == config["radii"]
        assert manifest["config"]["directions"] == 16
        assert manifest["config"]["fisher_samples"] == 256
        assert len(manifest["coordinate_order"]) == 24
        assert manifest["production_precision"] == "full FP32; cuDNN TF32 and matmul TF32 disabled; no operand emulation"
        states = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(states) == 97 and len({r["state_id"] for r in states}) == 97
        origins = [r for r in states if r["point"]["kind"] == "anchor"]
        assert len(origins) == 1 and origins[0]["reused_origin"]
        origin = origins[0]
        assert origin["point_effective_sha256"] == manifest["anchor_effective_sha256"]
        for row in states:
            assert row["method"] == method
            assert row["coordinate_sha256"] == manifest["coordinate_sha256"]
            assert row["data_manifest_sha256"] == manifest["data_manifest_sha256"]
            assert row["checkpoint_sha256"] == manifest["checkpoint_sha256"][method]
            assert row["fisher"]["samples"] == 256
            assert row["fisher"]["normalized"] is False
            assert row["fisher"]["floor"] is None and row["fisher"]["regularizer"] is None
            assert row["fisher"]["estimator"] == "mean per-example CE gradient outer product"
            assert "fixed Task0 20-class head" in row["fisher"]["coordinate"]
            assert row["metrics"]["train"]["samples"] == 1259
            assert row["metrics"]["test"]["samples"] == 307
            direction = row["point"].get("direction", {})
            directions.append({
                "method": method, "method_label": LABELS[method], "seed": 0,
                "state_id": row["state_id"], "radius": row["point"]["radius"],
                "direction_index": direction.get("index", -1),
                "direction_sha256": direction.get("sha256", "reused_origin"),
                "point_effective_sha256": row["point_effective_sha256"],
                "raw_trace_train256": row["fisher"]["raw_trace"],
                "D_train": row["metrics"]["train"]["loss"] - origin["metrics"]["train"]["loss"],
                "D_test": row["metrics"]["test"]["loss"] - origin["metrics"]["test"]["loss"],
                "accuracy_train": row["metrics"]["train"]["accuracy"],
                "reused_origin": row.get("reused_origin", False),
            })
        current = pd.DataFrame([r for r in directions if r["method"] == method])
        for radius, subset in current.groupby("radius", sort=True):
            assert len(subset) == (1 if radius == 0 else 16)
            if radius > 0:
                assert set(subset.direction_index) == set(range(16))
                direction_hashes[method, radius] = list(subset.sort_values("direction_index").direction_sha256)
            tr = subset.raw_trace_train256.to_numpy()
            assert np.isfinite(tr).all() and (tr >= 0).all()
            trace_records.append({
                "method": method, "method_label": LABELS[method], "seed": 0,
                "radius": radius, "directions": len(subset), "train_samples": 256,
                "trace_mean": tr.mean(), "trace_Q95": np.quantile(tr, .95, method="linear"),
                "trace_sample_max": tr.max(), "reused_origin": radius == 0,
                "accuracy_mean_train": subset.accuracy_train.mean(),
                "accuracy_train_samples": 1259,
            })
            for split in SPLITS:
                observed = neighborhood.query("method == @method and radius == @radius and split == @split")
                assert len(observed) == 1
                row = observed.iloc[0]
                assert row.directions == len(subset)
                damage = subset[f"D_{split}"].to_numpy()
                positive = np.maximum(damage, 0)
                np.testing.assert_allclose(
                    [row.D_mean, row.G, row.Q95, row.raw_trace_mean_train256],
                    [damage.mean(), positive.mean(), np.quantile(positive, .95, method="linear"), tr.mean()],
                    rtol=1e-10, atol=1e-14)
    assert manifests["sgd"]["data_manifest_sha256"] == manifests["gam"]["data_manifest_sha256"]
    assert manifests["sgd"]["coordinate_sha256"] == manifests["gam"]["coordinate_sha256"]
    for radius in radii[radii > 0]:
        assert direction_hashes["sgd", radius] == direction_hashes["gam", radius]
        for method in METHODS:
            assert direction_hashes[method, radius] == direction_hashes[method, .5]
    for row in tolerance.itertuples(index=False):
        ordered = neighborhood.query("method == @row.method and split == @row.split").sort_values("radius")
        prefix = 0.0
        for observation in ordered.itertuples(index=False):
            if getattr(observation, row.metric) > row.epsilon:
                break
            prefix = observation.radius
        assert prefix == row.prefix_radius
        assert bool(row.right_censored) == (prefix == max(radii))
    traces = pd.DataFrame(trace_records)
    selected = tolerance.query("metric == 'G'").copy()
    for path in (neighborhood_path, tolerance_path):
        shutil.copyfile(path, SOURCE_DIR / path.name)
        assert sha256(path) == sha256(SOURCE_DIR / path.name)
    neighborhood.assign(method_label=neighborhood.method.map(LABELS), dataset="ImageNet-A").to_csv(
        SOURCE_DIR / "B_damage.csv", index=False)
    pd.DataFrame(directions).to_csv(SOURCE_DIR / "B_direction_states.csv", index=False)
    traces.to_csv(SOURCE_DIR / "B_trace_direction_summary.csv", index=False)
    selected.assign(method_label=selected.method.map(LABELS), dataset="ImageNet-A").to_csv(
        SOURCE_DIR / "B_tolerance_Gmean.csv", index=False)
    tied = selected.pivot(index=["split", "epsilon"], columns="method", values="prefix_radius")
    checks = {
        "status": "passed", "dataset": "ImageNet-A", "training_seeds": [0],
        "method_names": LABELS, "raw_state_count_per_method": 97,
        "positive_radius_K": 16, "radius_zero_is_one_reused_origin": True,
        "radii": radii.tolist(), "fixed_Task0_head_classes": 20,
        "train_evaluation_samples": 1259, "test_evaluation_samples": 307,
        "raw_Fisher_train_samples": 256,
        "raw_damage_signed_mean_positive_mean_and_linear_Q95_match_all_report_rows": True,
        "raw_Fisher_mean_matches_all_report_rows": True,
        "raw_Fisher_Q95_and_sample_max_recomputed_from_all_raw_states": True,
        "raw_Fisher_normalization_floor_regularizer": [False, None, None],
        "checkpoint_and_coordinate_hashes_match_manifest": True,
        "checkpoint_weight_files_rehashed": False,
        "paired_direction_hashes_match_methods_and_radii": True,
        "shared_data_manifest_hash_matches_methods_and_all_states": True,
        "all_40_prefix_safe_radii_and_censoring_flags_recomputed": True,
        "Gmean_method_ties_at_all_sampled_thresholds": bool((tied.sgd == tied.gam).all()),
        "sample_max_preserved_in_CSV_but_not_plotted": True,
        "original_report_CSV_bytes_preserved": True,
        "point_jitter_displacement_clipping_smoothing": False,
        "CPU_only_redraw": True,
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "axis_scales": {"all_axes": "linear"},
        "tolerance_interpretation": "Largest sampled radius whose entire sampled prefix meets Gmean <= epsilon; train points at epsilon=.05 are right-censored at rmax=20; connecting lines guide the eye.",
    }
    return neighborhood, traces, selected, radii, checks


def plot_checked(ax, x, y, **kwargs):
    line, = ax.plot(x, y, **kwargs)
    np.testing.assert_array_equal(line.get_xdata(), np.asarray(x))
    np.testing.assert_array_equal(line.get_ydata(), np.asarray(y))
    return line


def radius_axis(ax):
    ax.set_xscale("linear")
    ax.set_xlim(-.7, 20.7)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Weight drift", labelpad=3)
    ax.xaxis.set_minor_locator(NullLocator())


def data_limits(ax, values, include_zero=False):
    minimum, maximum = float(np.min(values)), float(np.max(values))
    span = maximum - minimum
    padding = .04 * span
    lower, upper = minimum - padding, maximum + padding
    if include_zero:
        lower = 0
    ax.set_ylim(lower, upper)
    ticks = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper)
    ax.set_yticks([v for v in ticks if lower <= v <= upper])
    return dict(data_min=minimum, data_max=maximum, axis_min=lower, axis_max=upper)


def main():
    neighborhood, traces, selected, radii, checks = verified_data()
    font_dir = Path("/home-local/liying/.fonts")
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        font_manager.fontManager.addfont(font_dir / name)
    font_path = Path(font_manager.findfont("Times New Roman", fallback_to_default=False))
    assert font_manager.FontProperties(fname=font_path).get_name() == "Times New Roman"
    plt.rcParams.update({"font.family": "Times New Roman", "font.serif": ["Times New Roman"],
                         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                         "mathtext.it": "Times New Roman:italic",
                         "mathtext.bf": "Times New Roman:bold",
                         "mathtext.bfit": "Times New Roman:italic:bold",
                         "mathtext.cal": "Times New Roman:italic",
                         "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    train = neighborhood.query("split == 'train'").merge(
        traces[["method", "radius", "accuracy_mean_train", "trace_mean"]],
        on=["method", "radius"], validate="one_to_one")
    tolerance = selected.query("split == 'train'").copy()
    assert len(train) == 14 and len(tolerance) == 10
    assert train.accuracy_mean_train.between(0, 100).all()
    train.to_csv(SOURCE_DIR / "B_train_plot_data.csv", index=False)
    tolerance.to_csv(SOURCE_DIR / "B_train_tolerance.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.85))
    fig.subplots_adjust(left=.073, right=.993, bottom=.24, top=.96, wspace=.62)
    columns = [("G", r"Loss $\overline{[D_t(\boldsymbol{\theta},\boldsymbol{\Delta})]_+}$"),
               ("accuracy_mean_train", "Accuracy"),
               ("trace_mean", "Raw Fisher trace")]
    ranges, plotted = {}, []
    for index, (ax, (column, ylabel)) in enumerate(zip(axes[:3], columns)):
        for method in METHODS:
            d = train.query("method == @method").sort_values("radius")
            assert np.array_equal(d.radius.to_numpy(), radii)
            plot_checked(ax, d.radius, d[column], color=COLORS[method],
                         ls=LINES[method], marker=MARKERS[method], markersize=3.1,
                         linewidth=1.3, label=LABELS[method])
            plotted.append({"panel": index + 1, "method": method, "split": "train",
                            "column": column, "x": d.radius.tolist(), "y": d[column].tolist()})
        radius_axis(ax)
        ax.set_ylabel(ylabel, labelpad=3)
        ranges[column] = data_limits(ax, train[column], include_zero=column == "trace_mean")
        ax.legend(loc="lower left" if index == 1 else "upper left", frameon=False,
                  fontsize=8, handlelength=1, handletextpad=.3, labelspacing=.25,
                  borderpad=0, borderaxespad=.3)

    ax = axes[3]
    for method in METHODS:
        d = tolerance.query("method == @method").sort_values("epsilon")
        # Preserve exact ties: thick blue underneath narrow dashed orange, and
        # nested markers at the original coordinates. No offsets or jitter.
        plot_checked(ax, d.epsilon, d.prefix_radius, color=COLORS[method],
                     ls=LINES[method], linewidth=2.5 if method == "sgd" else 1.1,
                     label=LABELS[method], zorder=2 if method == "sgd" else 3)
        for censored in (False, True):
            points = d.loc[d.right_censored == censored]
            plot_checked(ax, points.epsilon, points.prefix_radius, ls="none",
                         marker="^" if censored else MARKERS[method],
                         markersize=(6.0 if censored else 5.0) if method == "sgd" else 3.0,
                         markerfacecolor="white" if method == "sgd" else COLORS[method],
                         markeredgecolor=COLORS[method], markeredgewidth=1,
                         zorder=4 if method == "sgd" else 5)
        plotted.append({"panel": 4, "method": method, "split": "train", "column": "prefix_radius",
                        "x": d.epsilon.tolist(), "y": d.prefix_radius.tolist(),
                        "right_censored": d.right_censored.tolist()})
    ax.set_xlim(-.001, .052)
    ax.set_xticks([0, .01, .02, .03, .04, .05], ["0", ".01", ".02", ".03", ".04", ".05"])
    ax.set_ylim(-.7, 20.8)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Loss threshold", labelpad=3)
    ax.set_ylabel("Tolerance radius", labelpad=3)
    ax.legend(loc="lower right", frameon=False, fontsize=8,
              handlelength=1, handletextpad=.3, labelspacing=.25, borderpad=0, borderaxespad=.3)
    for ax in axes:
        ax.tick_params(axis="both", length=3, pad=3)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        legend = ax.get_legend().get_window_extent(renderer)
        bounds = ax.get_window_extent(renderer)
        assert bounds.contains(legend.x0, legend.y0) and bounds.contains(legend.x1, legend.y1)
        for line in ax.lines:
            assert not line.get_path().transformed(line.get_transform()).intersects_bbox(legend, filled=False), ax.get_ylabel()
        labels = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        assert all(not first.overlaps(second) for first, second in zip(labels, labels[1:])), ax.get_xlabel()
    checks.update({"figure_inches": [7.6, 1.85], "layout": "one row, four columns",
                   "font_family": "Times New Roman", "font_path": str(font_path),
                   "font_sha256": sha256(font_path), "minimum_native_font_pt": 8,
                   "subplot_letters": False, "plotted_series_per_panel": [2, 2, 2, 2],
                   "plotted_split": "train", "test_and_Q95_plotted": False,
                   "accuracy_recomputed_from_raw_train_metrics": True,
                   "mean_accuracy_is_percent_not_fraction": True,
                   "tolerance_curves_coincide_without_jitter": True,
                   "radius_positions_proportional_to_numeric_distance": True,
                   "y_ranges": ranges, "legends_inside_without_curve_overlap": True,
                   "axis_limits_expanded_for_legends": False,
                   "x_labels": ["Weight drift", "Weight drift", "Weight drift", "Loss threshold"],
                   "y_labels": [label for _, label in columns] + ["Tolerance radius"],
                   "raw_fisher_multiplied_by_lambda": False,
                   "raw_fisher_coordinate_average": False,
                   "effective_weight_coordinate_count": 14155776,
                   "plotted_data": plotted,
                   "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}})
    save_figure(fig, FIGURE_NAME, checks)
    (FIG_DIR / "B_provenance.json").write_text(json.dumps(checks, indent=2) + "\n")
    caption = (
        "Figure B. Training-set neighborhood diagnostics on historical ImageNet-A seed0 checkpoints. "
        "From left to right: mean positive Task0 cross-entropy increase; mean Task0 training accuracy (%); "
        "mean raw empirical Fisher trace; and the mean-damage empirical tolerance radius. "
        "Weight drift is the perturbation radius: the joint Frobenius distance across 24 effective Q/V weight matrices from each method's own Task0 anchor. "
        "The measured radii are 0, 0.5, 1, 2, 5, 10 and 20. "
        "The same 16 unit directions are shared between methods and all positive radii; radius zero is one reused anchor. "
        "Loss and accuracy use all 1,259 Task0 training images with deterministic transforms and each method's fixed 20-class head; "
        "Fisher uses the same fixed 256 training images, true-label per-example gradients, without trace normalization, flooring or regularization. "
        "Its trace sums squared gradients over all 14,155,776 effective-weight coordinates, averages over samples and directions, and is not multiplied by the EWC coefficient. "
        "The first three panels show means across directions. Absolute accuracy retains the methods' different anchor accuracies. "
        r"The loss-damage notation is $D_t(\boldsymbol{\theta},\boldsymbol{\Delta})=L_t(\boldsymbol{\theta}+\boldsymbol{\Delta})-L_t(\boldsymbol{\theta})$; "
        r"the first panel displays $G(r)=K^{-1}\sum_k[D_0(w_0,ru_k)]_+$, the mean positive damage, not the absolute loss. "
        r"The tolerance radius $\hat r_\epsilon$ is the largest sampled radius such that $G(r_i)\leq\epsilon$ at every sampled radius up to it. "
        "Both methods have identical tolerance values at the five preset thresholds, shown with nested markers. "
        "This panel is a threshold summary of the first panel's measured damage curve, not an independent measurement. "
        "Triangles mark right-censoring at the maximum scanned radius 20: the measured tolerance reaches the scan limit. "
        "All axes are linear; lines guide the eye between measured points. Only training means are plotted, with no test curves or Q95 series. "
        "These finite-direction, single-seed diagnostics are not worst-case guarantees.\n")
    (FIG_DIR / "B_caption.md").write_text(caption)
    escaped = caption.replace("%", r"\%")
    (FIG_DIR / "B_caption.tex").write_text("\\caption{" + escaped.strip() + "}\n")
    (FIG_DIR / "B_train_only_include.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=\\textwidth]{Figure_B_neighborhood_merged.pdf}\n"
        "\\caption{" + escaped.strip() + "}\n\\label{fig:training-neighborhood}\n\\end{figure*}\n")
    # The user supplied both figure paths; publish identical rendered artifacts.
    aliases = FIG_DIR.parent / "figures"
    for suffix in ("pdf", "png"):
        shutil.copyfile(FIG_DIR / f"{FIGURE_NAME}.{suffix}", aliases / f"Figure_B_neighborhood.{suffix}")
    shutil.copyfile(FIG_DIR / f"{FIGURE_NAME}_validation.json", aliases / "Figure_B_neighborhood_validation.json")
    shutil.copyfile(FIG_DIR / "B_provenance.json", aliases / "Figure_B_provenance.json")
    shutil.copyfile(FIG_DIR / "B_caption.md", aliases / "Figure_B_neighborhood_caption.md")
    print(json.dumps({"status": "passed", "training_rows": len(train), "tolerance_rows": len(tolerance),
                      "source_states": 194, "figure": str(FIG_DIR / f"{FIGURE_NAME}.pdf")}))


if __name__ == "__main__":
    main()
