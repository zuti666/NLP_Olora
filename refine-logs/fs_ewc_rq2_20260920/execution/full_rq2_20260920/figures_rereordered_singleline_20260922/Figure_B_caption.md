**Figure B. Controlled neighborhood diagnostics on ImageNet-A.** EWCLoRA and FS-EWCLoRA are evaluated around their formal seed-0 Task-1 anchors $\mathbf w_1$, trained under the EWC configuration $\lambda=2000$. From left to right: **Task-1 Accuracy**, **Loss Increase**, **Fisher Trace**, and **Tolerance Radius**. The first three horizontal axes are **Perturbation radius $d$**; the fourth is **Allowed loss increase $\tau$**. Here $d$ is a controlled diagnostic displacement and is distinct from the training method's neighborhood-radius hyperparameter $r$.

The accuracy panel averages Task-1 training accuracy over paired random unit directions:
\[
\overline A_1^{\mathrm{nbr}}(d)
=\frac{1}{K_D}\sum_{k=1}^{K_D}
A_{1,\mathrm{tr}}(\mathbf w_1+d\mathbf u_k).
\]
Accuracy is reported in percent. The loss panel takes the positive part **for each direction before averaging**:
\[
\overline D_1^+(d)
=\frac{1}{K_D}\sum_{k=1}^{K_D}
\left[
\widehat{\mathcal L}_{1,\mathrm{tr}}(\mathbf w_1+d\mathbf u_k)
-\widehat{\mathcal L}_{1,\mathrm{tr}}(\mathbf w_1)
\right]_+.
\]
It therefore differs from the signed test-loss change in Figure A. Loss and accuracy use all 1,259 Task-1 training examples, the fixed 20-class Task-1 head, $K_D=50$ paired directions, and 107 measured radii per method. At $d=0$ all directions coincide with the anchor, which is evaluated once.

The third panel averages raw neighborhood empirical Fisher trace:
\[
\overline R_{F,1}^{\mathrm{nbr}}(d)
=\frac{1}{K_F}\sum_{k=1}^{K_F}
\operatorname{tr}F_{1,\mathrm{tr}}^{\mathrm{emp}}
\bigl(\mathbf w_1+d\mathbf u_k\bigr).
\]
This uses 256 fixed Task-1 training examples, $K_F=16$ paired directions, and 42 measured radii per method. The trace averages true-label per-example gradient squared norms in the 24 effective Q/V matrices, with no EWC coefficient scaling, trace normalization, or damping. The measured Fisher peak at $d=17$ is retained. Anchors were trained with rank-10 LoRA; the diagnostic perturbations occupy the full effective-weight space and are not constrained to preserve rank 10.

The fourth panel is computed directly from the positive-loss curve on the measured radius grid:
\[
\widehat d_\tau
=\max\left\{
d_j:\overline D_1^+(d_i)\leq\tau\ \text{for every }i\leq j
\right\}.
\]
This is the largest measured radius whose measured prefix satisfies the allowed loss increase. It is an empirical grid statistic, without interpolation or a continuous-radius certificate. All eleven thresholds through $\tau=0.1$ are retained. Radii $d=21,22,23,24$ were not measured; the gap limits tolerance resolution, and lines between measured points are visual guides. Curves are direction means from one training seed, so no cross-seed uncertainty band is shown.

Provenance: Figure B retains the panel order and measurements of `../figures_rereordered_20260922/`, with smaller single-line titles and larger horizontal-axis labels. It is the reordered and relabeled version of former **Figure B1** in `../figures_titled_20260922/`. Displayed Task 1 is source Task 0. Source fields `radius` and `epsilon` retain their names and numerical values; they are displayed as $d$ and $\tau$, respectively. Source `G_train` is exactly the positive-loss quantity $\overline D_1^+(d)$ above. This update changes panel titles and horizontal-axis label sizes without changing training, model evaluation, or measured values. Sources: [loss and accuracy](../figures_titled_20260922/B1_dense_seed0.csv), [Fisher trace](../figures_titled_20260922/B1_fisher_seed0.csv), [measured tolerance](../figures_titled_20260922/B1_tolerance_seed0.csv), [loss-definition audit](../figures_titled_20260922/B1_LOSS_AUDIT.md), and [configuration and scope](../figures_titled_20260922/analysis_report/CONFIG_AND_SCOPE.json).
