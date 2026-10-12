"""Actual anchor-relative effective-weight displacement at all 10 learned-task stages."""
from paper_plot_style import *


def main():
    trajectory, _, checks = load_verified_data()
    fig, ax = plt.subplots(figsize=(6.2, 4.45))
    fig.subplots_adjust(left=.135, right=.975, bottom=.265, top=.765)
    draw_trajectory(ax, trajectory, "r_T")
    ax.set_ylabel("Actual displacement\n" + r"$r_T=\|W_{T-1}-W_0\|_2$")
    ax.set_ylim(-.065, 1.52)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.55, .99),
               ncol=2, frameon=False, columnspacing=1.4, fontsize=9)
    fig.text(.55, .867, r"$T=1$: Task0 anchor (open markers)", ha="center", fontsize=9.5)
    fig.text(.55, .816, r"$T=2$–$10$: subsequent learning (Tasks1–9)", ha="center", fontsize=9.5)
    fig.text(.55, .097, PROVENANCE, ha="center", fontsize=10, fontweight="bold")
    fig.text(.55, .047, "Distance from each sequence’s own Task0 anchor", ha="center", fontsize=9)
    checks["plotted_source_rows"] = 20
    checks["coincident_T1_anchors_visible"] = "SGD outer open circle and GAM inner open diamond at unchanged coordinate (1,0)"
    save_figure(fig, "Figure_r_vs_learning_stage", checks)


if __name__ == "__main__":
    main()
