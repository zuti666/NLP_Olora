# Figure review resolution

The independent GPT-5.6-Sol xhigh review passed (same-family, provisional). Both non-blocking suggestions were addressed:

1. Figure B(c) vertical axis label was shortened to “Raw Fisher trace (train256)”; root reopened the final PNG and confirmed the panel letter is separate.
2. Figure B caption now defines F_emp=(1/N) sum_i g_i g_i^T with N=256 and unscaled per-example CE gradients. “Raw” retains sample averaging and excludes trace normalization, floor, and regularization. The source caption generator and all derived captions/LaTeX were updated.

C already states that Fisher is the empirical mean, the path is an endpoint chord, and the historical future optimizers differ. Table A uses “class-incremental accuracies” without implying a CNN backbone. No numerical data or underlying experiment was changed by these edits.

Final caption-only generation passed existing table checks. No TeX compiler was available; the supplied vector PDFs/PNGs were rendered and inspected, while editable LaTeX snippets remain uncompiled.
