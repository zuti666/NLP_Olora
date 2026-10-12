"""Historical-sequence FP32 re-evaluation plus measured t9 endpoint sensitivity."""
from paper_plot_style import *

def main():
    trajectory_path = REPORT / "C_current_FP32_task_trajectory.csv"
    path_summary_path = REPORT / "C_path_summary.csv"
    trajectory = pd.read_csv(trajectory_path)
    paths = pd.read_csv(path_summary_path)
    assert set(trajectory.seed) == set(paths.seed) == {0}
    assert set(trajectory.precision) == {"current full FP32"}
    assert set(trajectory.test_local_classes) == {20}
    assert set(trajectory.test_local_samples) == {307}
    assert paths.numerical_gate_passed.all()
    assert paths.quadrature_converged.all()
    assert paths.bound_passed_with_2pct_numeric_tolerance.all()
    assert set(paths.nodes) == {9}
    for method in METHODS:
        d = trajectory.query("method == @method").sort_values("task")
        assert list(d.task) == list(range(10))
        np.testing.assert_allclose(d.test_local_loss_delta_recomputed,
                                   d.test_local_loss - d.iloc[0].test_local_loss, atol=1e-14)
        p = paths.query("method == @method").iloc[0]
        np.testing.assert_allclose(d.iloc[-1].delta_norm_cpu_recomputed, p.drift, rtol=1e-10)
        np.testing.assert_allclose(p.I / p.drift ** 2, p.S_path, rtol=1e-12)
    trajectory.to_csv(SOURCE_DIR / "C_task_trajectory.csv", index=False)
    paths = paths.assign(endpoint_task=9, path_kind="Task0-to-Task9 effective-weight straight segment")
    paths.to_csv(SOURCE_DIR / "C_endpoint_path.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.1))
    fig.subplots_adjust(left=.12, right=.98, bottom=.105, top=.895, wspace=.37, hspace=.44)
    for letter, ax in zip("abcd", axes.flat):
        panel_label(ax, letter)
    for ax, column, ylabel in [
        (axes[0, 0], "test_local_accuracy", "Task0 test accuracy (%)"),
        (axes[0, 1], "test_local_loss_delta_recomputed", "Task0 test CE damage"),
        (axes[1, 1], "delta_norm_cpu_recomputed", r"Effective displacement $\|\Delta_t\|_2$")]:
        for method in METHODS:
            d = trajectory.query("method == @method").sort_values("task")
            ax.plot(d.task, d[column], color=COLORS[method], marker=MARKERS[method],
                    markersize=4, linestyle="-" if method == "sgd" else (0, (4, 2)))
        ax.set_xticks(range(10))
        ax.set_xlabel(r"Completed task $t$")
        ax.set_ylabel(ylabel)
    axes[0, 1].axhline(0, color=".65", linewidth=.7, zorder=0)
    axes[1, 1].set_ylim(bottom=0)
    ax = axes[1, 0]
    ordered = paths.set_index("method").loc[list(METHODS)]
    bars = ax.bar([0, 1], ordered.S_path, color=[COLORS[m] for m in METHODS], width=.52)
    bars[1].set_hatch("///")
    bars[1].set_edgecolor("white")
    ax.set_xticks([0, 1], ["SGD + EWC", "historical\nGAM + EWC"])
    ax.set_xlabel(r"Task0-to-Task9 endpoint segment")
    ax.set_ylabel(r"Directional sensitivity $S_{\mathrm{path}}(9)$")
    ax.set_ylim(0, ordered.S_path.max() * 1.23)
    for bar, value in zip(bars, ordered.S_path):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    handles = method_handles()
    handles[1].set_linestyle((0, (4, 2)))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .999),
               ncol=2, frameon=False, columnspacing=2.8)
    save_fig(fig, "Figure_C_sequence_and_endpoint")
    record_provenance("Figure_C", [trajectory_path, path_summary_path], {
        "all_10_tasks_in_both_methods": True,
        "local_test_head_20_classes_samples_307": True,
        "test_damage_recomputed_from_base_loss": True,
        "endpoint_displacement_matches_path_summary": True,
        "S_path_equals_I_over_displacement_squared": True,
        "both_path_numerical_gates_pass": True, "Simpson_nodes": 9,
        "S_path_only_plotted_at_measured_t9": True,
    })

if __name__ == "__main__":
    main()
