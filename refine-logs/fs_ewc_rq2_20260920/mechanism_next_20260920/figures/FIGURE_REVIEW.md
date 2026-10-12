# Figure quality review

**Verdict: pass.** This is a same-family, provisional figure-quality review using the `paper-figure` checklist. I found no substantive correctness or interpretation change required before these artifacts are used for the scoped, single-seed mechanism account.

## Scope reviewed

- `PAPER_PLAN.md`
- `figures/CAPTIONS.md`, `figures/captions.json`, and `figures/latex_includes.tex`
- rendered `figures/Figure_B_neighborhood.png` and `figures/Figure_C_sequence_and_endpoint.png`
- the figure-generation scripts, provenance JSON, plotted source-data CSVs, and controlled-continuation Table A

## Correctness and fairness

### Figure B

The 2x2 design matches the available data. Line plots are appropriate for the sampled radius grid; the mean, empirical Q95, and sampled maximum are clearly distinguished; and the tolerance panel shows a discrete finite-grid outcome rather than implying a continuous robustness radius. The symmetric-log scales are stated in the caption and retain the exact zero measurements.

The plot and caption preserve the decisive counterevidence. GAM+EWC has lower train-side positive damage and raw train256 Fisher trace, while its test `G_mean` is higher at every positive sampled radius. The methods tie at all displayed `G_mean` tolerance thresholds on both splits. Nested markers expose exact ties without jitter, and `>=20` is identified as right censoring.

Replication and uncertainty are handled correctly: each positive radius uses 16 paired directions, those directions are not called training replicates, the reused origin is identified, and no confidence interval or worst-case guarantee is claimed. The comparison uses equal radii in the same 24 effective Q/V `BA` coordinates, with a joint Euclidean/Frobenius norm and scaling 1. The caption therefore gives enough information to interpret the ambient-coordinate quantity without confusing it with factor-coordinate perturbations.

### Figure C

The figure type matches the evidence. Panels (a), (b), and (d) show the ten measured task checkpoints; their connecting curves are identified as visual guides. Panel (c) is a two-bar endpoint comparison only at measured task 9, so it does not invent a sensitivity trajectory for tasks 1--8.

The caption now states that the empirical Fisher is the mean of per-example gradient outer products and distinguishes the SGD-to-SGD and GAM-to-GAM historical chains from Table A's controlled future-SGD continuation. It also identifies the integrated object as the Task0-to-Task9 straight endpoint segment in ambient effective-weight space, explicitly not the optimizer-step path. This prevents a path-length or causal-mediation interpretation. The numerical scale is legible: `S_path(9)` is annotated directly, while displacement remains on its own axis.

The single-seed limitation is explicit, and there is no uncertainty estimate that could be mistaken for cross-run variation. The statement that GAM+EWC has lower endpoint sensitivity despite larger endpoint displacement is exactly supported by the plotted values, but the caption does not promote this association to a causal chain.

### Table A and surrounding evidence

Table A retains both controlled-continuation rows and the unfavorable plasticity result: GAM-anchor `a22=72.05` versus SGD-anchor `a22=74.80`, a difference of -2.75 percentage points. Its caption states that both continuations use future SGD+EWC and that inherited Fisher matrices may differ, so the intervention is not presented as geometry-only. The historical bridge remains separate and preserves realized direction identity. No additional baseline or new training run is essential for the limited descriptive claims represented by Figures B and C.

## Required changes

None.

## Minimal, non-blocking improvements

1. In `figures/Figure_B_neighborhood.png`, panel label `(c)` sits close to the upper end of the long vertical y-axis label. In `figures/gen_fig_B_neighborhood.py`, move only that panel label slightly right/up, or shorten the axis label to `Raw Fisher trace (train256)`. This is a small readability polish; the current text is still decipherable.
2. For exact consistency with Figure C and Table A, `figures/gen_tables_and_captions.py` could describe Figure B's empirical Fisher as the **mean** of per-example outer products and say that “without normalization” means no trace normalization, floor, or regularization. The current caption is numerically consistent with the data, but this wording removes a possible sum-versus-mean ambiguity.

## Checklist result

- Data-to-plot agreement: pass
- Figure type versus data: pass
- Self-contained captions: pass
- Comparable conditions and fairness: pass
- Replication/uncertainty language: pass
- Counterevidence preserved: pass
- Endpoint chord versus optimizer path: pass
- Effective-weight ambient coordinates versus factor coordinates: pass
- Numerical scales and censoring: pass
- Readability at the reviewed rendered size: pass, with the minor panel-(c) spacing suggestion above
- Essential missing baseline or ablation: none for the scoped claims

