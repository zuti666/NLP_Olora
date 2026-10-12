# Independent figure-quality review (provisional)

**Decision:** pass for Figures B and C at their native 7.6-inch widths. This was an independent same-family provisional Step-7 review of the rendered PNGs, captions, generation code, validation records, and cited source CSVs. It did not re-audit the underlying experiments or compile the figures inside a paper manuscript.

## Figure B — pass

- The final 2×2 render is readable and balanced. Panel letters, axis labels, legends, curves, and markers do not collide or clip. The shared legend correctly limits train/test styling to (a), (b), and (d); panel (c) has its own Mean/$Q_{95}$ key.
- The visualization matches the measurements: (a–b) retain all train/test positive-damage summaries; (c) is raw empirical-Fisher trace on 256 training examples rather than test loss or a fixed-Fisher quadratic; (d) plots the recomputed prefix-safe $G_{\mathrm{mean}}$ tolerance radius. Nested strokes expose the exact method ties without jitter, and triangles plus the $\geq20$ tick truthfully indicate right-censoring.
- The caption is self-contained about ImageNet-A, seed 0, fixed Task0 heads, sample counts, $K=16$, own-anchor coordinates, positive damage, quantiles, symlog scales, and the finite-direction/non-worst-case limitation. It makes no claim that either method enlarges a protected region.
- Data/provenance checks pass: plotted coordinates match source rows; all 40 tolerance values and censoring flags were recomputed; the source-copy hashes match the report CSVs; the PDF is vector with embedded text.

## Figure C — pass

- The requested 1×4 order is correct: displacement, signed Task0 loss change, Task0 accuracy, final chord sensitivity. The revised artifact is 7.6×2.5 inches (2280×750 pixels at 300 dpi), with 9 pt axes/panel labels and 8 pt ticks, legends, and bar labels. All text is readable at native size; all ten stage ticks and both full method names fit; the wider bar panel is balanced; each of (a–c) has the requested internal legend with clear headroom and no data intersection. Negative and nonmonotonic observations remain visible.
- The caption accurately states ImageNet-A, seed 0, stage mapping, method-specific anchors, differing optimizer sequences, fixed 20-class head and 307 test examples, and the absence of across-run uncertainty or an isolated anchor effect. It does not reinterpret displacement as a tolerance radius.
- Panel (d) is scientifically faithful: only the measured final $T=10$ values are shown; $\bar S_{10}=I_{10}/r_{10}^{2}$ uses the Task0-to-Task9 straight effective-weight chord, 256 Task0 training examples, and nine Simpson nodes. It is not described as test loss or as the optimizer trajectory.
- Source integrity passes: the two displayed source CSV copies are byte-identical to `execution/report_seed0`; plotted line coordinates and the two bar heights match them; the signed values and bar annotations are truthful; the vector PDF is present.

The typography pass applies to the native 7.6-inch artifacts. Their scale and placement in an actual compiled paper were not tested in this review.
