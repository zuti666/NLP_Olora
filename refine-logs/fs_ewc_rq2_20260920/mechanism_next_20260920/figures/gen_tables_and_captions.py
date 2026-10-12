"""Regenerate Table A, two appendix tables, captions, and LaTeX includes."""
from paper_plot_style import *

def tex_table(path, caption, label, columns, headers, rows, environment="table*"):
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    path.write_text(
        "\\begin{" + environment + "}[t]\n\\centering\n\\small\n"
        + "\\caption{" + caption + "}\n\\label{" + label + "}\n"
        + "\\begin{tabular}{" + columns + "}\n\\toprule\n"
        + " & ".join(headers) + r" \\" + "\n\\midrule\n" + body
        + "\n\\bottomrule\n\\end{tabular}\n\\end{" + environment + "}\n")

def main():
    a_path = REPORT / "A_two_task.csv"
    tolerance_path = REPORT / "B_tolerance_deterministic.csv"
    bridge_path = REPORT / "B_actual_direction_bridge.csv"
    a = pd.read_csv(a_path)
    assert set(a.anchor_method) == {"sgd", "gam"} and set(a.future_optimizer) == {"sgd"}
    assert set(a.seed) == {0}
    a.to_csv(FIG_DIR / "TABLE_A_controlled_continuation.csv", index=False)
    a.to_csv(SOURCE_DIR / "A_controlled_continuation.csv", index=False)
    a_display = pd.DataFrame([{
        "Task0 anchor": LABELS[row.anchor_method],
        "a11 (%)": f"{row.a11:.4f}", "a21 (%)": f"{row.a21:.2f}", "a22 (%)": f"{row.a22:.2f}",
        "D0 train (CE)": f"{row.D0_train_local:+.6f}", "D0 test (CE)": f"{row.D0_test_local:+.6f}",
        "Raw trace T0 (train256)": f"{row.raw_trace_t0_train256:.3f}",
        "Effective displacement": f"{row.effective_weight_drift:.6f}",
    } for row in a.itertuples()])
    write_markdown_table(a_display, FIG_DIR / "TABLE_A_controlled_continuation.md")
    by_anchor = a.set_index("anchor_method")
    a22_gap = float(by_anchor.loc["gam", "a22"] - by_anchor.loc["sgd", "a22"])
    a_caption = (
        r"Controlled Task1 continuation from historical SGD+EWC and visual GAM+EWC Task0 anchors, "
        r"with SGD+EWC used for both continuations (training seed 0). "
        r"The class-incremental accuracies $a_{11},a_{21},a_{22}$ (\%) use all classes seen "
        r"at the evaluation stage: old-task accuracy before and after continuation, and new-task "
        r"accuracy after continuation, respectively. $a_{11}$ is the current FP32 anchor evaluation; "
        r"$a_{21}$ and $a_{22}$ are from the continuation accuracy matrix. "
        r"$D_0$ is signed old-task cross-entropy change relative to each anchor, measured separately "
        r"with the fixed Task0 20-class head and deterministic preprocessing on 1,259 train and 307 test examples. "
        r"Negative damage means loss decreased. Raw trace at Task0 averages squared per-example "
        r"unscaled CE gradients over the same 256 old-task training examples in the ambient coordinates "
        r"of 24 effective Q/V $BA$ matrices (scaling 1), without normalization, floor, or regularization. "
        r"Displacement is the Euclidean/Frobenius norm in those effective-weight coordinates. "
        r"The inherited EWC Fisher matrices may differ across anchors; this is not an isolated "
        r"protection-matrix intervention. The new-task plasticity guard is retained: "
        + rf"GAM-anchor $a_{{22}}$ differs by ${a22_gap:+.2f}$ percentage points from the SGD-anchor row."
    )
    rows = [["SGD+EWC" if r.anchor_method == "sgd" else "GAM+EWC",
             f"{r.a11:.4f}", f"{r.a21:.2f}", f"{r.a22:.2f}",
             f"{r.D0_train_local:+.6f}", f"{r.D0_test_local:+.6f}",
             f"{r.raw_trace_t0_train256:.3f}", f"{r.effective_weight_drift:.6f}"] for r in a.itertuples()]
    tex_table(FIG_DIR / "TABLE_A_controlled_continuation.tex", a_caption, "tab:rq2_controlled_continuation",
        "lrrrrrrr", ["Task0 anchor", r"$a_{11}$", r"$a_{21}$", r"$a_{22}$", r"$D_0^{\rm train}$",
        r"$D_0^{\rm test}$", r"$\mathrm{tr}(\widehat F_0)$", r"$\|\Delta_1\|_2$"], rows)

    tolerance = pd.read_csv(tolerance_path)
    q95 = tolerance.query("metric == 'Q95'").copy()
    q95.to_csv(FIG_DIR / "TABLE_B_Q95_tolerance.csv", index=False)
    q95_pivot = q95.pivot(index=["split", "epsilon"], columns="method", values=["prefix_radius", "right_censored"])
    q95_rows = []
    for (split, epsilon), row in q95_pivot.iterrows():
        cells = [split.capitalize(), f"{epsilon:g}"]
        for method in METHODS:
            radius = row["prefix_radius", method]
            cells.append((r"$\geq " + f"{radius:g}" + "$" if row["right_censored", method] else f"{radius:g}"))
        q95_rows.append(cells)
    q95_caption = (
        r"Appendix: prefix-safe tolerance radii based on the empirical 95th percentile of positive "
        r"Task0 CE damage, $G_{95}$, for historical SGD+EWC and visual GAM+EWC anchors. "
        r"Each positive radius uses the same 16 paired directions; these are direction statistics "
        r"from training seed 0, not training replicates. For each threshold, the reported radius is "
        r"the largest sampled radius such that every sampled radius up to it passes the threshold. "
        r"The sampled grid is $r\in\{0,0.5,1,2,5,10,20\}$. "
        r"$\geq 20$ denotes right censoring at the grid boundary, not a measured radius above 20. "
        r"Both deterministic train and test losses use the fixed Task0 20-class head. "
        r"These are finite-direction, finite-grid measurements, not worst-case neighborhood guarantees."
    )
    tex_table(FIG_DIR / "TABLE_B_Q95_tolerance.tex", q95_caption, "tab:rq2_q95_tolerance", "lrrr",
              ["Split", r"$\epsilon$", "SGD+EWC", "GAM+EWC"], q95_rows, "table")
    write_markdown_table(pd.DataFrame(q95_rows, columns=["Split", "epsilon", "SGD+EWC", "GAM+EWC"]),
                         FIG_DIR / "TABLE_B_Q95_tolerance.md")

    bridge = pd.read_csv(bridge_path)
    assert set(bridge.seed) == {0}
    assert set(zip(bridge.anchor, bridge.direction_method)) == {(x, y) for x in METHODS for y in METHODS}
    bridge.to_csv(FIG_DIR / "TABLE_B_existing_bridge.csv", index=False)
    bridge.to_csv(SOURCE_DIR / "B_existing_bridge.csv", index=False)
    bridge_rows = [[r.anchor.upper(), r.direction_method.upper(), f"{r.radius:g}",
                    f"{r.D_train:+.6f}", f"{r.D_test:+.6f}", f"{r.raw_trace_train256:.3f}"]
                    for r in bridge.itertuples()]
    bridge_caption = (
        r"Appendix: existing two-anchor by two-realized-direction bridge (training seed 0). "
        r"SGD and GAM abbreviate historical SGD+EWC and visual GAM+EWC checkpoints. "
        r"Each direction is the unit effective-weight displacement from Task0 to Task9 of its named "
        r"historical sequence, transplanted onto the named Task0 anchor; $r$ is the absolute Euclidean "
        r"radius in ambient coordinates of the 24 effective Q/V $BA$ matrices (scaling 1). "
        r"The signed train/test Task0 CE changes use each anchor's fixed 20-class head and deterministic "
        r"preprocessing. The raw, unnormalized empirical Fisher trace uses per-example unscaled CE "
        r"gradients on the shared 256 training examples. Radius zero reuses the anchor measurement "
        r"for both directions. This preserves the existing historical-sequence direction identity: "
        r"these directions are not the new controlled-continuation directions of Table~\ref{tab:rq2_controlled_continuation}."
    )
    tex_table(FIG_DIR / "TABLE_B_existing_bridge.tex", bridge_caption, "tab:rq2_existing_bridge", "llrrrr",
              ["Anchor", "Direction", r"$r$", r"$D^{\rm train}$", r"$D^{\rm test}$", r"$\mathrm{tr}(\widehat F)$"], bridge_rows)
    write_markdown_table(pd.DataFrame(bridge_rows, columns=["Anchor", "Direction", "Radius", "D train", "D test", "Raw trace train256"]),
                         FIG_DIR / "TABLE_B_existing_bridge.md")

    # Caption assertions are checked against every relevant row, so a later
    # source update cannot silently retain an unsupported favorable statement.
    neighborhood = pd.read_csv(REPORT / "B_neighborhood_deterministic.csv")
    for split in ("train", "test"):
        comparison = neighborhood.query("split == @split and radius > 0").pivot(index="radius", columns="method", values="G")
        assert ((comparison.gam < comparison.sgd) if split == "train" else (comparison.gam > comparison.sgd)).all()
    trace = neighborhood.query("split == 'train'").pivot(index="radius", columns="method", values="raw_trace_mean_train256")
    assert (trace.gam < trace.sgd).all()
    g_tolerance = tolerance.query("metric == 'G'").pivot(index=["split", "epsilon"], columns="method", values="prefix_radius")
    assert (g_tolerance.gam == g_tolerance.sgd).all()
    c_paths = pd.read_csv(REPORT / "C_path_summary.csv").set_index("method")
    assert c_paths.loc["gam", "S_path"] < c_paths.loc["sgd", "S_path"]
    assert c_paths.loc["gam", "drift"] > c_paths.loc["sgd", "drift"]
    # Reuse the current B caption; preserve its percent sign when exporting LaTeX.
    b_caption = (FIG_DIR.parent / "figures_redesign_20260920/B_caption.md").read_text().strip().replace("%", r"\%")
    c_caption = (
        r"Historical SGD+EWC and visual GAM+EWC sequences, training seed 0, re-evaluated in current "
        r"full FP32. The historical sequences use SGD-to-SGD and GAM-to-GAM optimizers, respectively, "
        r"so their future optimizers differ; Table A separately controls the future optimizer, including GAM-to-SGD. "
        r"(a) Task0 test accuracy and (b) signed Task0 test CE damage relative to each "
        r"method's Task0 anchor after completed tasks $t=0,\ldots,9$, using the fixed local 20-class "
        r"head and the same 307 deterministic test examples throughout; these are not all-seen-class "
        r"accuracies. (c) Measured endpoint directional sensitivity only at $t=9$, "
        r"$S_{\rm path}(9)=I_9/\|\Delta_9\|_2^2$, where "
        r"$I_9=\int_0^1\Delta_9^\top\widehat F(\theta_0+s\Delta_9)\Delta_9\,ds$. "
        r"The empirical Fisher is the mean of per-example outer products of unscaled CE gradients over "
        r"the same 256 stratified old-task training examples, with no trace normalization, floor, "
        r"or regularization and with the Task0 head fixed. "
        r"The effective-weight straight segment between the Task0 and Task9 endpoints is evaluated "
        r"at 9 composite-Simpson nodes; it is not the actual optimizer-step training trajectory. "
        r"Both 5-versus-9-node quadrature checks pass the 5\% gate and the matched-subset numerical "
        r"bound checks pass the stated 2\% tolerance. No $S_{\rm path}$ measurements are available "
        r"at $t=1,\ldots,8$, so no sensitivity trajectory is drawn. "
        r"(d) Displacement $\|\Delta_t\|_2$ is measured in the ambient coordinates of 24 effective "
        r"Q/V $BA$ matrices (scaling 1). GAM+EWC has lower endpoint sensitivity despite a larger "
        r"endpoint displacement. Curves connect task checkpoints as visual guides; this single-seed "
        r"case supplies no uncertainty estimate over training runs."
    )
    captions = {"Table A": a_caption, "Figure B": b_caption, "Figure C": c_caption,
                "Appendix Q95 tolerance": q95_caption, "Appendix existing bridge": bridge_caption}
    (FIG_DIR / "CAPTIONS.md").write_text("\n\n".join(f"**{name}**\n\n{caption}" for name, caption in captions.items()) + "\n")
    (FIG_DIR / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    for filename, caption in [("TABLE_A_controlled_continuation.md", a_caption),
                              ("TABLE_B_Q95_tolerance.md", q95_caption),
                              ("TABLE_B_existing_bridge.md", bridge_caption)]:
        path = FIG_DIR / filename
        path.write_text(path.read_text() + "\n" + caption + "\n")
    includes = ["% Requires \\usepackage{graphicx,booktabs}. Paths assume this figures/ directory is beside the main TeX file.",
                r"\input{figures/TABLE_A_controlled_continuation.tex}"]
    for name, filename, label in [("Figure B", "Figure_B_neighborhood", "fig:rq2_neighborhood"),
                                   ("Figure C", "Figure_C_sequence_and_endpoint", "fig:rq2_sequence_endpoint")]:
        includes.append("\\begin{figure*}[t]\n\\centering\n"
                        + "\\includegraphics[width=0.95\\textwidth]{figures/" + filename + ".pdf}\n"
                        + "\\caption{" + captions[name] + "}\n\\label{" + label + "}\n\\end{figure*}")
    includes.extend(["% Appendix tables (move to the appendix when integrating):",
                     r"\input{figures/TABLE_B_Q95_tolerance.tex}",
                     r"\input{figures/TABLE_B_existing_bridge.tex}"])
    (FIG_DIR / "latex_includes.tex").write_text("\n\n".join(includes) + "\n")
    record_provenance("Tables", [a_path, tolerance_path, bridge_path,
        REPORT / "B_neighborhood_deterministic.csv", REPORT / "C_path_summary.csv"], {
        "both_A_anchors_and_same_future_SGD_preserved": True,
        "A_a11_a21_a22_both_loss_splits_trace_and_drift_preserved": True,
        "A_GAM_minus_SGD_a22_percentage_points": a22_gap,
        "existing_bridge_anchor_direction_pairs": 4,
        "existing_bridge_kept_separate_from_controlled_continuation": True,
        "caption_comparisons_checked_against_every_relevant_data_row": True,
    })

if __name__ == "__main__":
    main()
