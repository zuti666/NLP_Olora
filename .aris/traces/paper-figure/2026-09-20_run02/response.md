# Bounded figure-quality review

**Verdict: PASS (provisional same-family review).** No required scientific or rendering fixes were found in the reviewed artifacts. Both figures are suitable as full-width supplementary ImageNet-A, seed-0 diagnostics. They must remain separate from the historical ImageNet-R main table, as their artwork and captions already state.

## Required fixes

None.

## Checks that passed

- The rendered PNGs were inspected directly. Labels, tick marks, legends, stage separator, open Task0 anchors, panel letters, negative damage values, and the small GAM sensitivity bar/value are visible without clipping or overlap at the intended full-width presentation.
- The mapping is correct and explicit: source `task=0...9` is plotted as total learned tasks `T=1...10`; `T=1` is the Task0 anchor and `T=10` is after Task9.
- The figures and captions identify the data as existing ImageNet-A seed-0 checkpoints and explicitly separate them from the historical ImageNet-R main table.
- The method labels correctly describe the full historical sequences: SGD -> SGD+EWC and GAM -> GAM+EWC. The caption states that the continuation optimizers differ and that the figure does not isolate a causal Task0-anchor effect.
- Panels (a) and (b) use the fixed 20-class Task0 head on the same deterministic 307-example test set. The caption distinguishes these local old-task accuracies from class-incremental accuracy among all seen classes.
- Signed damage is preserved. SGD is negative at `T=2` and decreases on arrival at `T=2,5`; GAM is negative at `T=2,3,4`, peaks at `T=7` (`0.073354`), and falls to `0.034347` at `T=10`. No smoothing, clipping, or monotonicity claim is present.
- `r_T` is correctly shown as the actual anchor-relative effective-weight displacement, not a sum of step lengths and not a damage-tolerance radius. The endpoint values (`1.197584` SGD, `1.400473` GAM) match the source CSV.
- Chord sensitivity is shown only at the measured terminal endpoint. The caption correctly defines the Task0-to-Task9 effective-weight straight chord, fixed Task0 head, stratified train `n=256`, nine composite-Simpson nodes, and distinguishes this chord from the optimizer-step trajectory and from unmeasured intermediate stages. The plotted values (`0.368058` SGD, `0.014793` GAM) match the path summary.
- The copied plotting inputs are byte-identical to the execution reports: trajectory SHA256 `d5a71364b125d17eac3dc69a8893750dca71e3d4fe3292175e7b2de74b359945`; path-summary SHA256 `f3fe9d0a82d9a5c47399a7c9314bcd3d6039a86a948b4eaa658583de7c6e1d57`.
- The validation receipts report vector PDFs with embedded text and no raster images. The generation scripts read verified CSV/JSON sources rather than hardcoding plotted data, retain source coordinates, and produce no axes titles.
- `RESULTS_AND_MAIN_FIGURES.md` contains no material scientific misstatement within this review's scope. Its T mapping, endpoint values, nonmonotonic-damage description, chord-only interpretation, ImageNet-A/ImageNet-R separation, and warning against a unified causal claim agree with the reviewed evidence. Its phrase that GAM has smaller damage at larger terminal displacement is acceptable only as a descriptive observation of these two full method sequences; the report and captions supply that limitation.

## Nits

- The captions are unusually long. If page budget becomes binding, the quadrature and Fisher implementation detail could move to a methods or supplement paragraph, while retaining the dataset split, seed count, fixed-head distinction, continuation-optimizer difference, terminal-only chord statement, and lack of uncertainty/causal isolation.
- `Figure_r_vs_learning_stage` uses generous vertical space for stage and provenance text. It remains readable at full width; compacting that whitespace is optional.

## Reviewed provenance

- Rendered figures: `Figure_C_stage_aware.png/.pdf`, `Figure_r_vs_learning_stage.png/.pdf`
- Captions and includes: `captions.md`, `captions.json`, `latex_includes.tex`
- Plot generation: `gen_fig_C_stage_aware.py`, `gen_fig_r_vs_learning_stage.py`, `paper_plot_style.py`, `gen_captions.py`
- Validation: `Figure_C_stage_aware_validation.json`, `Figure_r_vs_learning_stage_validation.json`, `render_self_review.json`
- Primary plotted inputs: `../../execution/report_seed0/C_current_FP32_task_trajectory.csv`, `../../execution/report_seed0/C_path_summary.csv`
- Raw receipts used by validation: `../../execution/results_seed0_recovery1/{sgd,gam}/{trajectory,path}/summary.json`
- Scope/context report: `../RESULTS_AND_MAIN_FIGURES.md`

## Review limitation

This is a same-family Codex review rather than the requested GPT-5.6-Sol independent model review, so it is provisional. It was bounded to figure correctness, captions, readability, source agreement, and material statements in the context report; it is not an exhaustive code, checkpoint, or experiment-integrity audit, and no checkpoint weight files were rehashed.
