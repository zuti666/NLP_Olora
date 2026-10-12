"""CPU-only Figure B redraw, with raw-state checks and no measurement changes.

Run from any directory with Python, NumPy, pandas and Matplotlib installed.
The adjacent paper_plot_style.py provides the shared publication style.
"""
from paper_plot_style import *
from matplotlib.ticker import NullLocator

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
        "axis_scales": {
            "abc_radius": {"kind": "symlog", "base": 10, "linthresh": .5, "linscale": .65},
            "ab_damage": {"kind": "symlog", "base": 10, "linthresh": 1e-4, "linscale": .65},
            "c_Fisher": "linear", "d_epsilon": "log base10", "d_radius": "linear",
        },
        "tolerance_interpretation": "Largest sampled radius whose entire sampled prefix meets Gmean <= epsilon; train points at epsilon=.05 are right-censored at rmax=20; connecting lines guide the eye.",
    }
    return neighborhood, traces, selected, radii, checks


def plot_checked(ax, x, y, **kwargs):
    line, = ax.plot(x, y, **kwargs)
    np.testing.assert_array_equal(line.get_xdata(), np.asarray(x))
    np.testing.assert_array_equal(line.get_ydata(), np.asarray(y))
    return line


def radius_axis(ax, radii):
    ax.set_xscale("symlog", linthresh=.5, linscale=.65, base=10)
    ax.set_xticks(radii, [f"{r:g}" for r in radii])
    ax.set_xlim(-.055, 23)
    ax.set_xlabel(r"Effective-weight radius $r$")
    ax.xaxis.set_minor_locator(NullLocator())


def damage_axis(ax):
    ax.set_yscale("symlog", linthresh=1e-4, linscale=.65, base=10)
    ax.set_yticks([0, 1e-4, 1e-3, 1e-2, 1e-1],
                 ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    ax.set_ylim(-2e-5, .2)
    ax.yaxis.set_minor_locator(NullLocator())


def main():
    neighborhood, traces, selected, radii, checks = verified_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.05))
    fig.subplots_adjust(left=.108, right=.982, bottom=.087, top=.875, wspace=.31, hspace=.43)
    for letter, ax in zip("abcd", axes.flat):
        ax.text(0, 1.06, f"({letter})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11, clip_on=False)
        ax.tick_params(axis="both", length=3, pad=3)
    for ax, column, ylabel in ((axes[0, 0], "G", r"$G_{\mathrm{mean}}$ (CE damage)"),
                               (axes[0, 1], "Q95", r"$G_{95}$ (CE damage)")):
        for split in SPLITS:
            for method in METHODS:
                d = neighborhood.query("method == @method and split == @split").sort_values("radius")
                plot_checked(ax, d.radius, d[column], color=COLORS[method], ls=SPLIT_STYLES[split],
                             marker=MARKERS[method], markersize=4.8 if method == "sgd" else 3.8,
                             markerfacecolor=COLORS[method] if split == "train" else "white",
                             markeredgewidth=1.0, linewidth=1.8 if method == "sgd" else 1.45)
        # All four zero-damage anchors coincide: retain the exact zero coordinates.
        # Test outlines surround the smaller filled train markers at the origin.
        for split, sizes, order in (("test", {"sgd": 7.8, "gam": 6.5}, 6),
                                    ("train", {"sgd": 4.4, "gam": 2.9}, 8)):
            for j, method in enumerate(METHODS):
                plot_checked(ax, [0], [0], ls="none", marker=MARKERS[method],
                             markersize=sizes[method], markeredgewidth=1.0,
                             color=COLORS[method], markerfacecolor="none" if split == "test" else COLORS[method],
                             zorder=order+j)
        radius_axis(ax, radii)
        damage_axis(ax)
        ax.set_ylabel(ylabel)

    ax = axes[1, 0]
    for method in METHODS:
        d = traces.query("method == @method").sort_values("radius")
        for column, split in (("trace_mean", "train"), ("trace_Q95", "test")):
            plot_checked(ax, d.radius, d[column], color=COLORS[method], ls=SPLIT_STYLES[split],
                         marker=MARKERS[method], markersize=4 if split == "train" else 3,
                         markerfacecolor=COLORS[method] if split == "train" else "white",
                         markeredgewidth=1, linewidth=1.6)
    radius_axis(ax, radii)
    ax.set_ylabel("Raw Fisher trace")
    ax.set_ylim(0, 3700)
    ax.set_yticks([0, 1000, 2000, 3000])
    ax.legend(handles=[Line2D([], [], color=".2", linestyle=SPLIT_STYLES[s],
                              label=label) for s, label in (("train", "Mean"), ("test", r"$Q_{95}$"))],
              loc="upper left", frameon=False, ncol=2, columnspacing=1.1,
              handlelength=2, handletextpad=.5, borderaxespad=.4)

    ax = axes[1, 1]
    # A broad blue line underneath a narrow orange line exposes exact method ties.
    # Both use the original sampled coordinates, without jitter or vertical offsets.
    for split in SPLITS:
        for method in METHODS:
            d = selected.query("method == @method and split == @split").sort_values("epsilon")
            plot_checked(ax, d.epsilon, d.prefix_radius, color=COLORS[method],
                         ls=SPLIT_STYLES[split], linewidth=3.0 if method == "sgd" else 1.25,
                         zorder=2 if method == "sgd" else 3)
    for split, sizes, order in (("test", {"sgd": 8.2, "gam": 6.6}, 6),
                                ("train", {"sgd": 5.2, "gam": 3.4}, 8)):
        for j, method in enumerate(METHODS):
            d = selected.query("method == @method and split == @split").sort_values("epsilon")
            ordinary = d.loc[~d.right_censored]
            plot_checked(ax, ordinary.epsilon, ordinary.prefix_radius, ls="none", marker=MARKERS[method],
                         markersize=sizes[method], color=COLORS[method], markeredgewidth=1.15,
                         markerfacecolor="none" if split == "test" else COLORS[method], zorder=order+j)
            censored = d.loc[d.right_censored]
            if not censored.empty:
                plot_checked(ax, censored.epsilon, censored.prefix_radius, ls="none", marker="^",
                             markersize=8.8 if method == "sgd" else 4.8,
                             markerfacecolor="white" if method == "sgd" else COLORS[method],
                             color=COLORS[method], markeredgewidth=1.3, zorder=order+j)
    epsilons = np.sort(selected.epsilon.unique())
    ax.set_xscale("log")
    ax.set_xticks(epsilons, [f"{e:g}" for e in epsilons])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(epsilons.min()/1.18, epsilons.max()*1.22)
    ax.set_ylim(-.7, 22)
    ax.set_yticks([0, 2, 5, 10, 20], ["0", "2", "5", "10", r"$\geq 20$"])
    ax.set_xlabel(r"CE-damage threshold $\epsilon$")
    ax.set_ylabel(r"Tolerance radius $\hat r_\epsilon$")
    method_handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls="-",
                             markersize=4, label=LABELS[m]) for m in METHODS]
    split_handles = [Line2D([], [], color=".25", ls=SPLIT_STYLES[s], marker="o", markersize=3.8,
                            markerfacecolor=".25" if s == "train" else "white",
                            label=f"{s.capitalize()} (a, b, d)") for s in SPLITS]
    fig.legend(handles=method_handles+split_handles, loc="upper center", bbox_to_anchor=(.535, .997),
               ncol=2, frameon=False, columnspacing=2.6, handlelength=2.4, handletextpad=.65)
    checks["figure_inches"] = [7.6, 6.05]
    checks["plotted_series"] = {"a": 4, "b": 4, "c": 4, "d": 4}
    save_figure(fig, FIGURE_NAME, checks)
    # Explicit B-only provenance accompanies the shared save helper's validation.
    (FIG_DIR / "B_provenance.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    main()
