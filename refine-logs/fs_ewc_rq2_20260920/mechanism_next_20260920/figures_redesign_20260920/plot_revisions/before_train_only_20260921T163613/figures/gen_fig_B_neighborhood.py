"""Regenerate Figure B from reports, verifying direction summaries against JSONL."""
from paper_plot_style import *
from matplotlib.ticker import FixedLocator, NullLocator

def main():
    neighborhood_path = REPORT / "B_neighborhood_deterministic.csv"
    tolerance_path = REPORT / "B_tolerance_deterministic.csv"
    neighborhood = pd.read_csv(neighborhood_path)
    tolerance = pd.read_csv(tolerance_path)
    assert set(neighborhood.seed) == set(tolerance.seed) == {0}
    assert set(neighborhood["transform"]) == {"deterministic_test"}
    direction_records, trace_records, state_paths = [], [], []
    direction_hashes = {}
    for method in METHODS:
        path = RAW / method / "neighborhood" / "states.jsonl"
        state_paths.append(path)
        states = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len({r["state_id"] for r in states}) == len(states)
        origin = [r for r in states if r["point"]["kind"] == "anchor"]
        assert len(origin) == 1 and origin[0]["reused_origin"]
        origin = origin[0]
        for row in states:
            assert row["fisher"]["samples"] == 256
            assert row["fisher"]["normalized"] is False
            assert row["fisher"]["floor"] is None and row["fisher"]["regularizer"] is None
            radius = row["point"]["radius"]
            direction = row["point"].get("direction", {})
            direction_records.append({
                "method": method, "seed": 0, "state_id": row["state_id"],
                "radius": radius, "direction_index": direction.get("index", -1),
                "direction_sha256": direction.get("sha256", "reused_origin"),
                "point_effective_sha256": row["point_effective_sha256"],
                "raw_trace_train256": row["fisher"]["raw_trace"],
                "D_train": row["metrics"]["train"]["loss"] - origin["metrics"]["train"]["loss"],
                "D_test": row["metrics"]["test"]["loss"] - origin["metrics"]["test"]["loss"],
                "reused_origin": row.get("reused_origin", False),
            })
        current = pd.DataFrame([r for r in direction_records if r["method"] == method])
        for radius, subset in current.groupby("radius", sort=True):
            assert len(subset) == (1 if radius == 0 else 16)
            if radius > 0:
                assert set(subset.direction_index) == set(range(16))
                direction_hashes[method, radius] = list(subset.sort_values("direction_index").direction_sha256)
            tr = subset.raw_trace_train256.to_numpy()
            trace_records.append({"method": method, "seed": 0, "radius": radius,
                "directions": len(subset), "train_samples": 256,
                "trace_mean": tr.mean(), "trace_Q95": np.quantile(tr, .95, method="linear"),
                "trace_sample_max": tr.max(), "reused_origin": radius == 0})
            for split in ("train", "test"):
                row = neighborhood.query("method == @method and radius == @radius and split == @split")
                assert len(row) == 1
                row = row.iloc[0]
                damage = subset[f"D_{split}"].to_numpy()
                positive = np.maximum(damage, 0)
                np.testing.assert_allclose([row.D_mean, row.G, row.Q95, row.raw_trace_mean_train256],
                    [damage.mean(), positive.mean(), np.quantile(positive, .95, method="linear"), tr.mean()],
                    rtol=1e-10, atol=1e-14)
    radii = np.sort(neighborhood.radius.unique())
    for radius in radii[radii > 0]:
        assert direction_hashes["sgd", radius] == direction_hashes["gam", radius]
    for row in tolerance.itertuples():
        ordered = neighborhood.query("method == @row.method and split == @row.split").sort_values("radius")
        prefix = 0.0
        for observation in ordered.itertuples():
            if getattr(observation, row.metric) > row.epsilon:
                break
            prefix = observation.radius
        assert prefix == row.prefix_radius
        assert bool(row.right_censored) == (prefix == max(radii))
    traces = pd.DataFrame(trace_records)
    neighborhood.to_csv(SOURCE_DIR / "B_damage.csv", index=False)
    pd.DataFrame(direction_records).to_csv(SOURCE_DIR / "B_direction_states.csv", index=False)
    traces.to_csv(SOURCE_DIR / "B_trace_direction_summary.csv", index=False)
    tolerance.to_csv(SOURCE_DIR / "B_tolerance_all.csv", index=False)
    selected = tolerance.query("metric == 'G'").copy()
    selected.to_csv(SOURCE_DIR / "B_tolerance_Gmean.csv", index=False)
    tied = selected.pivot(index=["split", "epsilon"], columns="method", values="prefix_radius")
    ties = bool((tied["sgd"] == tied["gam"]).all())
    test = neighborhood.query("split == 'test' and radius > 0").pivot(index="radius", columns="method", values="G")
    gam_test_higher = bool((test.gam > test.sgd).all())

    fig, axes = plt.subplots(2, 2, figsize=(8, 6.5))
    fig.subplots_adjust(left=.12, right=.98, bottom=.105, top=.865, wspace=.34, hspace=.52)
    for letter, ax in zip("abcd", axes.flat):
        panel_label(ax, letter)
    for ax, metric, ylabel in ((axes[0, 0], "G", r"$G_{\mathrm{mean}}$ (CE damage)"),
                                (axes[0, 1], "Q95", r"$G_{95}$ (CE damage)")):
        for method in METHODS:
            for split in ("train", "test"):
                d = neighborhood.query("method == @method and split == @split").sort_values("radius")
                ax.plot(d.radius, d[metric], color=COLORS[method], ls=SPLIT_STYLES[split],
                        marker=MARKERS[method], markersize=3.6, markerfacecolor="white" if split == "test" else COLORS[method])
        radius_axis(ax, radii)
        ce_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=-2e-5)
    trace_ax = axes[1, 0]
    trace_styles = {"trace_mean": ("Mean", "-"), "trace_Q95": (r"$Q_{95}$", (0, (4, 2))),
                    "trace_sample_max": ("Sample max.", (0, (1, 1.6)))}
    for method in METHODS:
        d = traces.query("method == @method").sort_values("radius")
        for column, (_, style) in trace_styles.items():
            trace_ax.plot(d.radius, d[column], color=COLORS[method], ls=style,
                          marker=MARKERS[method] if column == "trace_mean" else None, markersize=3.4)
    radius_axis(trace_ax, radii)
    trace_ax.set_ylabel(r"Raw Fisher trace (train256)")
    trace_ax.set_ylim(bottom=0)
    trace_ax.legend(handles=[Line2D([], [], color=".2", ls=sty, label=lab)
                            for lab, sty in trace_styles.values()], loc="upper left",
                    bbox_to_anchor=(-.02, 1.20), ncol=3, frameon=False,
                    columnspacing=.8, handlelength=1.7, handletextpad=.4, fontsize=8.2)

    tolerance_ax = axes[1, 1]
    # Nested markers and a narrow orange line atop a broader blue line preserve
    # exact numeric ties without jittering radii or thresholds.
    for split in ("train", "test"):
        for method in METHODS:
            d = selected.query("method == @method and split == @split").sort_values("epsilon")
            tolerance_ax.plot(d.epsilon, d.prefix_radius, color=COLORS[method], ls=SPLIT_STYLES[split],
                linewidth=2.8 if method == "sgd" else 1.05,
                marker=MARKERS[method], markersize=7 if method == "sgd" else 3.5,
                markerfacecolor="white" if method == "sgd" else COLORS[method],
                markeredgewidth=1.25, zorder=2 if method == "sgd" else 3)
            censored = d[d.right_censored]
            tolerance_ax.scatter(censored.epsilon, censored.prefix_radius,
                s=125 if method == "sgd" else 35, marker="^", edgecolor=COLORS[method],
                facecolor="white" if method == "sgd" else COLORS[method], linewidth=1.25, zorder=4 if method == "sgd" else 5)
    epsilons = np.sort(selected.epsilon.unique())
    tolerance_ax.set_xscale("log")
    tolerance_ax.set_xticks(epsilons, [f"{e:g}" for e in epsilons], rotation=25)
    tolerance_ax.xaxis.set_minor_locator(NullLocator())
    tolerance_ax.set_yticks([0, 2, 5, 10, 20], ["0", "2", "5", "10", r"$\geq 20$"])
    tolerance_ax.set_ylim(-.7, 23)
    tolerance_ax.set_xlim(epsilons.min() / 1.18, epsilons.max() * 1.22)
    tolerance_ax.set_xlabel(r"CE-damage threshold $\epsilon$")
    tolerance_ax.set_ylabel(r"Prefix-safe radius ($G_{\mathrm{mean}}$)")
    if ties:
        tolerance_ax.text(.03, .92, "Methods tie at every threshold", transform=tolerance_ax.transAxes,
                          fontsize=8.4, va="top")
    split_handles = [Line2D([], [], color=".25", ls=SPLIT_STYLES[s], label=f"{s.capitalize()} (a, b, d)") for s in ("train", "test")]
    fig.legend(handles=method_handles() + split_handles, loc="upper center", bbox_to_anchor=(.53, .997),
               ncol=2, frameon=False, columnspacing=3.1, handlelength=2.8)
    save_fig(fig, "Figure_B_neighborhood")
    record_provenance("Figure_B", [neighborhood_path, tolerance_path] + state_paths, {
        "raw_damage_mean_positive_mean_and_linear_Q95_match_report": True,
        "raw_Fisher_mean_matches_report": True,
        "paired_direction_hashes_match": True,
        "positive_radius_K": 16, "radius_zero_is_one_reused_origin": True,
        "prefix_safe_radius_recomputed": True, "Gmean_methods_tie_all_thresholds": ties,
        "Gmean_GAM_exceeds_SGD_at_all_positive_test_radii": gam_test_higher,
        "trace_Q95_and_sample_max_recomputed_from_raw_states": True,
    })

if __name__ == "__main__":
    main()
