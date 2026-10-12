"""Redraw existing ImageNet-A sequence measurements as a clean four-panel row.

CPU only. Source integrity assertions are shared with the other paper figures.
No source measurement, historical figure, or checkpoint is modified.
"""
from paper_plot_style import (
    FIG_DIR, METHODS, COLORS, MARKERS, LINES, LABELS,
    load_verified_data, save_figure, sha256, plt, np,
)

# Native two-column print dimensions; do not rely on shrinking a 12-inch canvas.
# These overrides are local to this generator and leave the shared style intact.
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
})


def draw_stage_curve(ax, trajectory, column, legend_location):
    for method in METHODS:
        rows = trajectory.query("method == @method").sort_values("T")
        line, = ax.plot(
            rows["T"], rows[column],
            color=COLORS[method], marker=MARKERS[method], linestyle=LINES[method],
            linewidth=1.3, markersize=3.1, label=LABELS[method], zorder=3,
        )
        # Keep both zero anchors visible without displacing either coordinate.
        if column != "test_local_accuracy":
            ax.plot(
                [1], [rows.iloc[0][column]], linestyle="none",
                marker=MARKERS[method], markersize=5.3 if method == "sgd" else 3.1,
                markerfacecolor="white", markeredgecolor=COLORS[method],
                markeredgewidth=1.2, zorder=4,
            )
        np.testing.assert_array_equal(line.get_xdata(), rows["T"].to_numpy())
        np.testing.assert_array_equal(line.get_ydata(), rows[column].to_numpy())
    ax.set_xlim(.65, 10.35)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(r"Learning stage $T$")
    ax.tick_params(axis="both", length=3, pad=3)
    ax.legend(
        loc=legend_location, frameon=False, fontsize=8,
        handlelength=1.3, handletextpad=.35, borderaxespad=.35,
        labelspacing=.35,
    )


def main():
    trajectory, paths, checks = load_verified_data()
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    fig.subplots_adjust(left=.065, right=.993, bottom=.205, top=.89, wspace=.5)
    for letter, ax in zip("abcd", axes):
        ax.text(0, 1.045, f"({letter})", transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left")

    specifications = [
        ("r_T", r"Displacement $r_T$", "upper left", (-.065, 1.85)),
        ("test_local_loss_delta_recomputed", "Signed Task0 CE change",
         "upper left", (-.035, .34)),
        ("test_local_accuracy", "Task0 accuracy (%)", "upper left", (78.5, 94)),
    ]
    for ax, (column, ylabel, location, limits) in zip(axes[:3], specifications):
        draw_stage_curve(ax, trajectory, column, location)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_ylim(*limits)
        assert trajectory[column].min() > limits[0]
        assert trajectory[column].max() < limits[1]
        np.testing.assert_array_equal(ax.get_xticks(), np.arange(1, 11))
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [LABELS[m] for m in METHODS]
    axes[1].axhline(0, color=".7", linewidth=.7, zorder=0)
    axes[0].set_yticks([0, .5, 1, 1.5])
    axes[1].set_yticks([0, .1, .2, .3])
    axes[2].set_yticks([80, 85, 90])

    ax = axes[3]
    ordered = paths.set_index("method").loc[list(METHODS)]
    bars = ax.bar([0, 1.25], ordered.Sbar, color=[COLORS[m] for m in METHODS], width=.55)
    bars[1].set_hatch("///")
    bars[1].set_edgecolor("white")
    ax.set_xticks([0, 1.25], [LABELS[m] for m in METHODS], fontsize=8)
    ax.set_ylabel(r"Chord sensitivity $\bar S_{10}$", labelpad=4)
    ax.set_ylim(0, .44)
    ax.set_yticks([0, .1, .2, .3, .4])
    ax.set_xlim(-.65, 1.95)
    ax.tick_params(axis="both", length=3, pad=3)
    for bar, value in zip(bars, ordered.Sbar):
        assert bar.get_height() == value
        ax.annotate(
            f"{value:.4f}", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
        )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes[:3]:
        legend_box = ax.get_legend().get_window_extent(renderer)
        axes_box = ax.get_window_extent(renderer)
        assert axes_box.contains(legend_box.x0, legend_box.y0)
        assert axes_box.contains(legend_box.x1, legend_box.y1)
        for line in ax.lines:
            display_path = line.get_path().transformed(line.get_transform())
            assert not display_path.intersects_bbox(legend_box, filled=False)
    for ax in axes:
        tick_boxes = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        assert all(not left.overlaps(right) for left, right in zip(tick_boxes, tick_boxes[1:]))
    assert len(fig.legends) == 0
    checks.update({
        "panels": {
            "a": "Actual anchor-relative displacement r_T",
            "b": "Signed old Task0 test CE loss change relative to own anchor",
            "c": "Old Task0 test accuracy using its fixed 20-class head",
            "d": "Two final T10 chord-normalized sensitivity bars",
        },
        "panel_reordering_from_previous_C": ["d", "b", "a", "c"],
        "layout": "one row, four columns",
        "figure_size_inches": [7.6, 2.5],
        "native_print_width_inches": 7.6,
        "font_sizes_points": {"axes_and_panels": 9, "ticks_legends_bar_labels": 8},
        "minimum_native_font_size_points": 8,
        "subplot_width_ratios": [1, 1, 1, 1.3],
        "display_names": LABELS,
        "plotted_source_rows_per_curve_panel": 20,
        "all_source_values_strictly_inside_displayed_y_limits": True,
        "all_stage_ticks_are_actual_integers_1_through_10": True,
        "legend_inside_each_first_three_axes": True,
        "legend_bounding_boxes_do_not_intersect_plotted_lines": True,
        "adjacent_x_tick_label_bounding_boxes_do_not_overlap": True,
        "global_legend_present": False,
        "stage_separator_present": False,
        "footer_or_dataset_annotation_in_image": False,
        "no_unmeasured_Sbar_stages_drawn": True,
        "script": {"path": str(FIG_DIR / "gen_fig_C_sequence_row.py"),
                   "sha256": sha256(FIG_DIR / "gen_fig_C_sequence_row.py")},
    })
    save_figure(fig, "Figure_C_sequence_row", checks)


if __name__ == "__main__":
    main()
