# Audited result figures

All three figures are generated from pinned copies of the audited raw CSV files in `inputs/` by `plot_results.py`. Numeric results are read from CSV, not entered into the plot script. Every arm has only seed 2 (`n=1`); there are no uncertainty estimates or error bars. The two experiment protocols are separate and must not be combined as replicates.

## Figure 1 — Independent Task0 learning curves

Files: `task0_learning_curves.png` and `.pdf`.

Left: held-out Task0 sample accuracy, denoted $a_{11}$, at each training epoch. Right: sample-averaged held-out cross-entropy in nats. The five displayed directions are CE ($g_0$), S ($g_1$), G ($g_3$), Mean ($(g_1+g_3)/2$), and Sum ($g_1+g_3$). Lines show the recorded epochs without smoothing. Dots mark the fixed final epoch 50; final comparisons use this endpoint and must not select each arm's peak accuracy. All five displayed runs completed 50 epochs, but this is an **incomplete Stage1 campaign snapshot: five of ten planned arms**. Pending or interrupted arms are not displayed. The held-out set has 307 samples; the accuracy is a sample mean, not a class-macro mean. Stage1 uses independent data/sampler RNG and its own training protocol. No Task0 measurements from the older ten-task experiments have been inserted into these curves.

Sources: `task0/epoch_curves.csv` and `task0/final_results.csv`, frozen as `inputs/epoch_curves.csv` and `inputs/final_results.csv`.

## Figure 2 — Ten-task direction composition

Files: `ten_task_composition.png` and `.pdf`.

The complete ten-task runs compare S ($g_1$), G ($g_3$), Mean ($(g_1+g_3)/2$), and Sum ($g_1+g_3$), with $\gamma=0$, outer radius $r=1$, and inner radius $\rho=0.05$. Left: task AAA. Right: task FAA. These are task-equally-weighted metrics, not sample-weighted accuracy across tasks. With $A_{t,j}$ the held-out sample accuracy on task $j$ after training stage $t$ and $T=10$,

$$
\mathrm{AAA}_{\mathrm{task}}=\frac{1}{T}\sum_{t=1}^{T}\left(\frac{1}{t}\sum_{j=1}^{t}A_{t,j}\right),\qquad
\mathrm{FAA}_{\mathrm{task}}=\frac{1}{T}\sum_{j=1}^{T}A_{T,j}.
$$

Each stage is evaluated at its fixed final epoch 50; no peak checkpoints are selected. Mean changes the scale of the combined direction through $\beta_1=\beta_3=0.5$, while Sum uses $\beta_1=\beta_3=1$. Both panels show accuracy in percent. These ten-task runs use the original sequential protocol and cannot be treated as continuations or replicates of independent Stage1. Only seed 2 is available.

Source: `ten_task/training_summary.csv`, frozen as `inputs/training_summary.csv`; rows are selected by their beta, gamma, and radius metadata. The original nonzero-gamma arm is excluded from this composition comparison.

## Figure 3 — Ten-task pure-g3 radius sweep

Files: `ten_task_g3_radius.png` and `.pdf`.

Task AAA (left) and task FAA (right) for complete ten-task $g_3$ runs at outer radii $r\in\{0,0.1,0.25,0.5,1\}$, with $\beta_1=0$, $\beta_3=1$, $\gamma=0$, and $\rho=0.05$. The horizontal axis preserves the numeric distances between radii; connecting segments guide the eye and are not a fitted model. Metric definitions and fixed final-epoch rules are the same as Figure 2. These runs belong to the original ten-task protocol, not the independent Stage1 Task0 sweep. The small differences between the nonzero radii do not establish a robust optimum from a single seed.

Source: `ten_task/training_summary.csv`, frozen as `inputs/training_summary.csv`; rows are selected by beta and gamma metadata and sorted by the recorded radius.

## Reproduction and checks

Run `PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B plot_results.py` from this directory. The script requires the audited source CSVs to match their pinned copies, validates the five 50-epoch curves and their final values, checks finite curve values, and checks text bounds before saving. Outputs are 300 dpi PNGs and vector PDFs. All three PNGs were visually reviewed: legends, titles, axes, result labels, and footnotes are readable and not clipped. `FIGURE_MANIFEST.json` records input hashes and output checks.
