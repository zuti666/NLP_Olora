"""Write self-contained captions and LaTeX includes for the two existing-data plots."""
from paper_plot_style import FIG_DIR, load_verified_data
import json


def main():
    _, _, checks = load_verified_data()
    sgd = checks["methods"]["sgd"]
    gam = checks["methods"]["gam"]
    common = (
        r"Existing ImageNet-A checkpoints, training seed 0, re-evaluated in current full FP32. "
        r"These are supplementary ImageNet-A measurements, separate from the historical ImageNet-R main table. "
        r"The historical SGD+EWC sequence uses SGD for both the Task0 anchor and subsequent learning; "
        r"the historical GAM+EWC sequence uses GAM for both. Thus the future optimizers differ. "
        r"$T$ is the total number of learned tasks, $T=t_{\mathrm{code}}+1$: "
        r"$T=1$ is the Task0 anchor, $T=2$ is after Task1, and $T=10$ is after Task9. "
        r"Open markers indicate Task0 anchors; the dotted separator divides the anchor from subsequent learning. "
        r"Co-located zero anchors are drawn with an outer SGD circle and inner GAM diamond at the same coordinate. "
    )
    captions = {
        "Figure_C_stage_aware": common + (
            r"(a) Old Task0 test accuracy and (b) signed old-task test cross-entropy damage "
            r"$D_T=L_0^{\mathrm{test}}(W_{T-1})-L_0^{\mathrm{test}}(W_0)$, relative to each method's own anchor. "
            r"Both use a fixed Task0 20-class head and the same 307 deterministic test examples throughout. "
            r"Negative damage indicates a loss decrease. These are local Task0 accuracies, not accuracies among all seen classes. "
            r"All 20 actual checkpoint rows are shown without smoothing, clipping, or monotonicity enforcement. "
            f"GAM damage peaks at $T={gam['loss_peak_T']}$ ({gam['loss_peak']:.6f}) and decreases to "
            f"{gam['loss_endpoint']:.6f} at $T=10$; SGD damage also decreases at $T=2$ and $T=5$. "
            r"(c) Measured chord-averaged directional sensitivity only at $T=10$, "
            r"$\bar S_{10}=I_{10}/r_{10}^2$, where "
            r"$I_{10}=\int_0^1\Delta W_{10}^{\top}\widehat F_{\mathrm{emp}}(W_0+s\Delta W_{10})\Delta W_{10}\,ds$ "
            r"and $\Delta W_{10}=W_9-W_0$. Here "
            r"$\widehat F_{\mathrm{emp}}=N^{-1}\sum_{i=1}^N g_i g_i^{\top}$ uses unscaled per-example CE gradients "
            r"on the same $N=256$ stratified Task0 training examples, with the Task0 head fixed and no trace normalization, floor, or regularization. "
            r"Nine composite-Simpson nodes evaluate the Task0-to-Task9 effective-weight straight segment, "
            r"not the actual optimizer-step trajectory. The recorded 5-versus-9-node quadrature checks "
            r"pass the 5\% gate, and the matched-subset numerical bound checks pass their 2\% tolerance. "
            r"There are no intermediate-stage sensitivity measurements, so only two endpoint bars are drawn. "
            r"(d) Actual cumulative displacement $r_T=\|W_{T-1}-W_0\|_2$ from each sequence's own Task0 anchor, "
            r"in the joint ambient coordinates of 24 effective Q/V $BA$ matrices with scaling 1. "
            r"This is a displacement norm, not a sum of step lengths or a damage-tolerance radius. "
            r"Lines connect measured task checkpoints as visual guides. There is one training run per method; "
            r"the figure provides no uncertainty estimate across runs or isolated causal effect of the Task0 anchor."
        ),
        "Figure_r_vs_learning_stage": common + (
            r"Actual cumulative effective-weight displacement $r_T=\|W_{T-1}-W_0\|_2$ "
            r"is measured from each sequence's own Task0 anchor in the joint ambient coordinates "
            r"of 24 effective Q/V $BA$ matrices with scaling 1. "
            r"All 20 original checkpoint rows are drawn at their actual $T$ and displacement values, "
            r"without smoothing or coordinate offsets; connecting lines are visual guides between checkpoints. "
            r"This is neither the sum of successive step lengths nor an estimated damage-tolerance radius. "
            f"At $T=10$, the measured displacements are {sgd['endpoint_r_T']:.6f} for SGD+EWC "
            f"and {gam['endpoint_r_T']:.6f} for GAM+EWC. "
            r"Each curve is one training run (seed 0), so no uncertainty across runs is estimated. "
            r"The differing continuation optimizers prevent an isolated causal comparison of the anchors."
        ),
    }
    (FIG_DIR / "captions.json").write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n")
    (FIG_DIR / "captions.md").write_text("\n\n".join(f"## {name}\n\n{text}" for name, text in captions.items()) + "\n")
    snippets = []
    for name, text in captions.items():
        snippets.append("\\begin{figure*}[t]\n    \\centering\n"
                        + f"    \\includegraphics[width=0.95\\textwidth]{{figures/{name}.pdf}}\n"
                        + f"    \\caption{{{text}}}\n    \\label{{fig:{name.lower()}}}\n"
                        + "\\end{figure*}\n")
    (FIG_DIR / "latex_includes.tex").write_text("\n".join(snippets))


if __name__ == "__main__":
    main()
