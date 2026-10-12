"""Reproduce the stage-aware C figure from existing ImageNet-A sources."""
from paper_plot_style import *


def main():
    trajectory, paths, checks = load_verified_data()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7))
    fig.subplots_adjust(left=.105, right=.975, bottom=.16, top=.86, wspace=.36, hspace=.42)
    for letter, ax in zip("abcd", axes.flat):
        ax.text(0, 1.06, f"({letter})", transform=ax.transAxes,
                fontsize=11, va="bottom", ha="left")
    for ax, column, ylabel in [
        (axes[0, 0], "test_local_accuracy", "Task0 test accuracy (%)"),
        (axes[0, 1], "test_local_loss_delta_recomputed", "Signed Task0 test\nCE damage $D_T$"),
        (axes[1, 1], "r_T", r"Actual displacement $r_T$")]:
        draw_trajectory(ax, trajectory, column)
        ax.set_ylabel(ylabel)
    axes[0, 0].set_ylim(78.5, 91)
    axes[0, 1].axhline(0, color=".6", linewidth=.7, zorder=0)
    axes[0, 1].set_ylim(-.035, .29)
    axes[1, 1].set_ylim(-.065, 1.51)
    ax = axes[1, 0]
    ordered = paths.set_index("method").loc[list(METHODS)]
    bars = ax.bar([0, 1], ordered.Sbar, color=[COLORS[m] for m in METHODS], width=.48)
    bars[1].set_hatch("///")
    bars[1].set_edgecolor("white")
    ax.set_xticks([0, 1], ["SGD", "GAM"])
    ax.set_xlabel("$T=10$ only: Task0 → Task9 chord\ntrain $n=256$ · 9 Simpson nodes", fontsize=9.5)
    ax.set_ylabel(r"Chord sensitivity $\bar S_{10}$")
    ax.set_ylim(0, ordered.Sbar.max() * 1.24)
    for bar, value in zip(bars, ordered.Sbar):
        assert bar.get_height() == value
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=2, frameon=False, columnspacing=2.2)
    fig.text(.54, .921, STAGES, ha="center", fontsize=9.5)
    fig.text(.54, .045, PROVENANCE, ha="center", fontsize=10.5, fontweight="bold")
    fig.text(.54, .017, "Current FP32 · fixed 20-class Task0 head · test n=307", ha="center", fontsize=9)
    checks["panels"] = {"a": "Task0 test accuracy", "b": "signed Task0 test loss damage",
                        "c": "two T10-only chord-sensitivity bars", "d": "actual anchor-relative r_T"}
    checks["plotted_source_rows_per_curve_panel"] = 20
    checks["no_unmeasured_Sbar_stages_drawn"] = True
    save_figure(fig, "Figure_C_stage_aware", checks)


if __name__ == "__main__":
    main()
