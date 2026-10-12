# Figure quality review

Review date: 2026-09-20

Scope: the four rendered PNGs, their plotted-data JSON sidecars, `latex_includes.tex`, `data_verification.json`, `export_verification.json`, selected fields from `source_manifest.json`, `taskwise_accuracy.csv`, and `final_faa_by_order.csv`. This was a bounded audit; no large JSONL was traversed and no training was run.

## Decision

**Pass for the two all-method main figures. No blocking correctness or rendering issue was found.** The two SeqLoRA-only renderings also pass visually and numerically. One minor caption fix is needed only if the optional SeqLoRA figures are inserted into the paper using the current generic caption comment.

## Blocking findings

None.

## Passed checks

### Numerical agreement

- `taskwise_accuracy.csv` contains 330 unique plotted rows: 150 for Llama-3.2-1B and 180 for Llama-3.2-3B. These counts are exactly the available method/model/order combinations times the ten learned-dataset stage cells per order.
- Every value in the four plotted-data sidecars matches `taskwise_accuracy.csv`: 150/150 points in the 1B all-method figure, 180/180 in the 3B all-method figure, and 30/30 in each SeqLoRA-only figure. There were zero missing keys and zero value mismatches.
- All 33 rows in `final_faa_by_order.csv` equal the recomputed mean of the four stage-4 dataset accuracies for the same model, method, and order.
- The axis semantics match the data: each dataset curve begins at the stage where that dataset is learned, so unseen tasks are not presented as learned-task performance.

### Source and exclusion integrity

- `source_manifest.json` contains 132 unique stage keys, and all 132 referenced metric files and all 132 referenced training-result files exist.
- Six representative sources spanning new 1B EWC/FS-EWC/TreeLoRA runs and historical 3B SeqLoRA/OLoRA/TreeLoRA runs were spot-checked. Their metric-file SHA256 values match the manifest, and every plotted dataset metric from those files matches the CSV.
- The historical Llama-3.2-3B SeqLoRA Order-1 records are masked correctly: stage 1 plots only DBPedia, stage 2 adds Amazon, stage 3 adds Yahoo, and stage 4 adds AGNews. The manifest explicitly records the masked unseen datasets, and the CSV contains only the ten learned-task cells.
- SC/TC aggregates do not appear as dataset rows. The only plotted dataset labels are DBPedia, Amazon, Yahoo, and AGNews.
- The three unavailable Llama-3.2-1B OLoRA final cells are listed as missing and are absent from the 1B CSV, plotted-data JSON, curves, summaries, and method legend. The older unmatched 1B OLoRA values were not reused. Both the in-figure note and main caption disclose the omission.
- The evidence description is appropriately limited: `data_verification.json` says raw predictions were not recomputed in this pass and identifies the basis as source stage metrics plus previously verified final integer counts.

### Caption and comparison integrity

- Both main captions identify the model, metric, three task orders, dataset/method encodings, 7,600 examples per dataset point, and the single available seed per method.
- The captions disclose seed 42 versus 1993, DDP8/GA8 versus DDP4/GA16, differing optimizer defaults and adapter budgets, and the FS-EWCLoRA to implemented GAM-EWC mapping. These disclosures prevent the lines from implying a strictly controlled equal-protocol comparison.
- The captions make no unsupported superiority claim and do not treat the three orders as uncertainty estimates.

### Render quality

- All four PNGs are 2250 x 1095 pixels (300 dpi output for a 7.5-inch-wide figure) and are readable at the intended full-width `figure*` size.
- No axis label, tick label, panel label, legend item, or the 1B OLoRA omission note is clipped. The repaired bottom labels have visible clearance.
- Legends occupy dedicated space above the axes and do not cover data. Dense crossings in the all-method figures remain traceable through the combined marker and line-style encoding.
- Dataset colors use the colorblind-oriented Okabe-Ito palette. Dataset identity is also recoverable from its first learned stage and the dataset-labelled x ticks.
- Vector PDFs are present for all four figures. The PyMuPDF export check confirms that each is a single 540 x 262.8 point page with embedded Times New Roman, all three Order labels, no raster image object, and no text bounding box outside the page.

## Minor required fix for optional SeqLoRA use

**Severity: minor; non-blocking for the two main figures.** The optional caption comment in `latex_includes.tex` says “for the stated Llama model.” That phrase is not self-contained if copied directly into a manuscript. Before using either optional figure, give it a model-specific caption, for example:

- 1B: “Task-wise normalized exact-match accuracy of SeqLoRA with Llama-3.2-1B across three task orders. Each point evaluates 7,600 examples from a learned dataset after the indicated training stage; curves begin when the dataset is first learned. All three orders use seed 42, DDP8, per-device batch size 1, gradient accumulation 8, and nominal batch size 64.”
- 3B: use the same wording with “Llama-3.2-3B.”

No plot regeneration is needed for this fix.

## Optional improvements

- Keep the all-method figures at `\linewidth` in a two-column `figure*`; scaling them to one column would make the six-method encodings too small.
- For maximum replication clarity, the main captions could add: “Orders are distinct task sequences, not independent seed replicates.” This is consistent with the manifest and is advisory because the captions already state that only one seed is available per method.
