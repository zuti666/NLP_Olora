**Figure A. Realized future-learning diagnostics on ImageNet-A.** EWCLoRA and FS-EWCLoRA use the formal training configuration with EWC coefficient $\lambda=2000$. From left to right: **Effective-Weight Drift**, **Task-1 Test-Loss Change**, **Task-1 Test Accuracy**, and **Path Fisher Sensitivity**. The horizontal axis in every panel is **Number of learned tasks**, indexed by $t=1,\ldots,10$. Each method and seed uses its own Task-1 anchor $\mathbf w_1$; $\mathbf w_t$ concatenates the 24 effective Q/V LoRA weight matrices after learning Task $t$.

The first panel measures the realized displacement
\[
\Delta\mathbf w_t=\mathbf w_t-\mathbf w_1,\qquad
d_t=\|\Delta\mathbf w_t\|_2.
\]
The second panel shows the **signed** test cross-entropy difference
\[
D_{1,t}^{\mathrm{test}}
=\widehat{\mathcal L}_{1,\mathrm{test}}(\mathbf w_t)
-\widehat{\mathcal L}_{1,\mathrm{test}}(\mathbf w_1),
\]
so negative values are retained. The third shows Task-1 test accuracy $A_{1,t}^{\mathrm{test}}$ in percent. Both test quantities use all 307 Task-1 test examples and the fixed 20-class Task-1 head. The first three panels contain all ten measured endpoints per method and seed, for 60 endpoint records in total.

The fourth panel, whose y-axis is labeled **Path Fisher sensitivity**, shows displacement-normalized directional path empirical Fisher:
\[
\overline S_{F,1}^{\mathrm{path}}(t)
=\int_0^1 \mathbf v_t^\top
F_{1,\mathrm{tr}}^{\mathrm{emp}}
\bigl(\mathbf w_1+s\Delta\mathbf w_t\bigr)
\mathbf v_t\,\mathrm ds,
\qquad \mathbf v_t=\frac{\Delta\mathbf w_t}{d_t}.
\]
Each straight anchor-to-endpoint chord is independently measured for $t=2,\ldots,10$. At $t=1$, $d_1=0$ and the normalized quantity is undefined, so no estimate is plotted. Fisher uses true-label per-example gradients on all 1,259 Task-1 **training** examples in the 24 effective Q/V matrices. It is raw diagnostic empirical Fisher, without EWC coefficient scaling, trace normalization, damping, floor, or added regularization. Adaptive nested Simpson quadrature uses 9 or 17 measured nodes per chord: 54 chords and 502 nodes in total. Integration nodes are locations along straight chords; they are not task stages or the optimizer's chronological trajectory. This directional path statistic is not an endpoint Fisher trace.

Curves are arithmetic means over three independent training seeds (0, 1, and 2). For path sensitivity, each seed's integral is divided by its own squared displacement **before** the three seed values are averaged. Standard-deviation shading is omitted; sample standard deviations remain available in the source CSV. These panels summarize the existing measurements without changing training, model evaluation, or measured values.

Provenance: Figure A is the reordered and relabeled version of former **Figure B2** in `../figures_titled_20260922/`. Displayed Task 1 corresponds to source `task=0`; throughout this caption, $t=\texttt{task}+1=\texttt{T}$. Source task IDs, checkpoint names, CSV columns, and raw records retain their original indexing. The original 18 chords at source tasks 1, 5, and 9 (displayed tasks 2, 6, and 10) and the 36 stage-completion chords are all retained. Sources: [endpoint records](../figures_titled_20260922/B2_trajectory_seed_data.csv), [path records](../figures_titled_20260922/B2_path_seed_data.csv), [means and sample standard deviations](../figures_titled_20260922/B2_mean_sd.csv), and [configuration and scope](../figures_titled_20260922/analysis_report/CONFIG_AND_SCOPE.json).
