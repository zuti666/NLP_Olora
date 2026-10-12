# Figure quality review

**Verdict: PASS — no blocking issues.**

Reviewed the five rendered PNG/PDF pairs in `output/` against `reference_preview.png`, `reference_style.md`, the frozen data, the independent metric audit, `export_verification.json`, and the finalized captions in `latex_includes.tex`. This is a bounded figure review; it does not request new runs or additional experiments.

## Rendered-artifact checks

| Figure | Curves / stages | Legibility and clipping | Legend and identification | Result |
|---|---:|---|---|---|
| Aircraft | 7 / 10 | Labels, ticks, markers, and extrema are clear; nothing is clipped | Seven entries are complete and below the axes; filename/caption can identify the dataset | PASS |
| Cars | 7 / 10 | Labels, ticks, markers, and extrema are clear; nothing is clipped | Seven entries are complete and below the axes; filename/caption can identify the dataset | PASS |
| CUB-200 | 7 / 10 | Labels, ticks, markers, and extrema are clear; nothing is clipped | Seven entries are complete and below the axes; filename/caption can identify the dataset | PASS |
| Flowers | 7 / 10 | High-valued CL-LoRA/GAM-EWCLoRA traces retain visible markers and headroom; nothing is clipped | Seven entries are complete and below the axes; filename/caption can identify the dataset | PASS |
| 5Datasets | 7 / 5 | Labels, ticks, markers, and extrema are clear; nothing is clipped | Seven entries are complete and below the axes; filename/caption can identify the dataset | PASS |

All five PDF pages are approximately 5.335 x 4.16 inches. At the intended `0.78\textwidth` placement in a full-width `figure*`, this is close to native size, so the 14 pt axis labels, 10 pt ticks, and 8.6 pt legend remain readable. Each PDF contains vector paths and embedded Times New Roman text, with no raster image object. No in-figure titles are present, as required by the paper-figure rules.

## Style and accessibility

- The four unchanged reference identities are preserved: gray circle (Vanilla LoRA), dark-blue square (InfLoRA), orange up-triangle (SDLoRA), and green filled-plus (EWCLoRA).
- The reference magenta star is correctly labeled **GAM-EWCLoRA**, not FS-EWC. OLoRA's light-blue diamond and CL-LoRA's purple down-triangle are distinct additions.
- All lines are solid with the reference 1.5 pt width and 4 pt markers. Light-gray grids, Times typography, axis sizing, and automatic y-range treatment match the reference language.
- The two-row legend preserves the requested reading order: Vanilla LoRA, InfLoRA, OLoRA, SDLoRA / CL-LoRA, EWCLoRA, GAM-EWCLoRA. It does not cover any data or collide with the x-axis label.
- Color is reinforced by seven distinct marker shapes. The closest hue pair, InfLoRA/OLoRA, remains separable by dark/light tone and square/diamond markers; likewise CL-LoRA/GAM-EWCLoRA use down-triangle/star markers.
- Omitting bands is the honest choice for one archived run per method. Nothing in the render suggests a confidence interval or replicated estimate.

## Data and claim honesty

- The bundle contains 35 curves and 315 plotted stage values: five datasets by seven methods, with Pets absent.
- `AA_t` is consistently defined as the equal-weight mean over tasks seen through stage `t`, including the current task. The final point is FAA, while the mean over all stage points is AAA.
- `plot_taskwise_aa.py` checks every rendered x/y series against the frozen `curves.json` values and checks that no uncertainty collections exist.
- `independent_metric_audit.json` independently verifies all 35 original-source hashes, all 1,645 required lower-triangle cells, all 315 stage means, and every FAA/AAA table entry. It reports no failure.
- EWCLoRA's visually strong curve is potentially easy to overinterpret because it is the eligible archived run with the highest final AA, chosen retrospectively rather than by a fixed matched protocol. The planned caption explicitly states this, which is necessary for a fair reading.

## Caption review

The finalized common caption is self-contained: it defines the metric, states the replication unit (`n=1` archived run per method), says that no uncertainty band is estimated, identifies the shared backbone/rank/batch size, and discloses the EWCLoRA selection rule. The finalized dataset-specific clauses correctly state:

- Aircraft: SDLoRA uses learning rate 0.02 while the other methods use 0.05.
- Cars and Flowers: EWCLoRA uses seed 1 while the other plotted runs use seed 0.
- 5Datasets: state the task order `CUB-200 -> FGVC-Aircraft -> Stanford Cars -> Oxford Flowers-102 -> Oxford-IIIT Pet`, EWCLoRA seed 2, and the 40-epoch InfLoRA/OLoRA/SDLoRA versus 30-epoch Vanilla LoRA/CL-LoRA/EWCLoRA/GAM-EWCLoRA split.

For CUB-200, the common caption is sufficient. `README.md` documents the dataset-dependent EWCLoRA coefficient/implementation differences and explicitly warns that these historical comparisons do not isolate method effects. Together with the caption's statement that the runs retain documented method-specific configurations, this avoids implying a completely matched training protocol.

No additional baseline, ablation, or experiment is needed for this five-figure deliverable.
