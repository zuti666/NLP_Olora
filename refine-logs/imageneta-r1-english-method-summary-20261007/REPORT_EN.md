# ImageNet-A at r=1: method and gradient summary

After correcting the original GAM coefficient to gamma=0.1, 10 of the 11 rows in the supplied image have completed counterparts. The literal unprojected difference, h+−h−, has no matching completed result or identified queue entry. Its dash is a missing result, not zero accuracy.

Common training configuration: ImageNet-A, 200 classes in a fixed order, 10×20-class tasks; ViT-B/16 with augreg2_in21k_ft_in1k pretraining, LoRA rank10; outer radius r=1 and inner radius rho=.05; 50 epochs per task, batch128, SGD with zero momentum and weight decay; learning rate .04 for Task0 and .02 thereafter, cosine decay to 10% of each initial learning rate. EWC/Fisher, weight merging, and replay are disabled. Natural-step results use the original learning rates without additional final update normalization.

Values are task-macro FAA/AAA (%) and BWT (percentage points). Completed three-seed conditions use seeds0/1/2 and mean±sample standard deviation (ddof=1). † denotes seed2 only; its standard deviation is not defined. Higher is better for all three metrics, including negative BWT.

Define h+=g1+g3 and h−=g0+g2. Pi-perp-u(v) is the component of v orthogonal to u, computed using a global inner product over trainable LoRA parameters and the seen classifier. The positive probes are g0=g(theta), g1=g(theta+rho N(g0)), g2=g(theta+r N(g1−g0)), and g3=g(theta+r N(g1−g0)+rho N(g2)).

## A. Original execution protocol

Eight completed methods, 24/24 validated seed cells. Four gradient probes plus one clean CE forward; global RNG progression and native perturbation subtraction/restoration. The missing unprojected direction is listed for completeness.

| Method | Gradient direction | FAA ↑ | AAA ↑ | BWT ↑ |
|---|---|---:|---:|---:|
| Accelerated GAM (original) | $h_+-0.1\Pi^\perp_{h_+}(h_-)$ | 54.977±0.351 | 68.333±0.281 | -6.916±0.226 |
| Unprojected branch difference (gamma=1) | $h_+-h_-$ | — | — | — |
| GAM without orthogonal correction (gamma=0) | $g_1+g_3$ | 54.810±0.148 | 68.117±0.347 | -6.684±0.823 |
| Mean of g1 and g3 | $(g_1+g_3)/2$ | 50.373±1.162 | 66.470±0.890 | -6.885±1.010 |
| SGD-direction control | $g_0$ | 40.063±0.726 | 60.134±0.709 | -12.809±0.662 |
| SAM-direction control | $g_1$ | 45.559±0.511 | 63.204±0.885 | -9.800±1.156 |
| SGD + second local difference | $g_0+(g_3-g_2)$ | 46.921±2.200 | 63.128±2.683 | -7.134±0.801 |
| Outer SAM-probe gradient | $g_3$ | 50.537±0.870 | 66.203±0.588 | -6.389±0.308 |
| Outer-location gradient | $g_2$ | 48.451±2.510 | 63.924±1.134 | -7.354±1.757 |

The gamma=0 row is also the branch sum g1+g3; these are one experiment, not two independent conditions. Sum and mean are collinear at the same parameter state. The original GAM orthogonal correction can change the direction, so they should not be labelled “the same direction as original GAM.” The g3 row is an outer SAM-probe gradient; original GAM minus SAM also contains the orthogonal correction. “Mean” is used instead of attributing the setting to an unverified paper hyperparameter.

## B. Exact restoration and RNG replay protocol

Three three-seed methods plus one seed2 method, 10/10 validated seed cells. Four positive probes with exact full-state restoration and per-probe RNG replay; unused probes leave no final state change. These rows share the common training configuration above but belong to a different execution protocol from Panel A. Compare within each panel rather than treat both as one controlled ranking.

| Method | Gradient direction | FAA ↑ | AAA ↑ | BWT ↑ |
|---|---|---:|---:|---:|
| SAM + second local difference | $g_1+(g_3-g_2)$ | 49.610±1.746 | 65.985±0.382 | -6.621±0.845 |
| Sum of g1 and g2 | $g_1+g_2$ | 55.911±1.033 | 68.609±0.353 | -5.544±0.639 |
| Mean of g1 and g2 | $(g_1+g_2)/2$ | 51.225±1.559 | 66.837±0.262 | -6.243±0.928 |
| SAM + outer displacement residual† | $g_1+(g_2-g_0)$ | 46.873 | 64.775 | -8.224 |

The last row is algebraically g1+g2−g0, not g1+g2. Its seed2 result is listed without a standard deviation. The two g1+g2 rows extend the supplied image with the newly completed three-seed experiments.

## Formula corrections and remaining gap

Original GAM uses h+−0.1 Pi-perp-h+(h−). Setting gamma=0 removes the complete orthogonal correction and gives h+. Removing only the projection while retaining gamma=.1 would instead give h+−.1h−; no matching completed run or queue entry was found for that direction either. The image's h+−h− additionally sets the correction coefficient to1.

For a single strictly uniform execution-protocol table, full GAM, branch sum, branch mean, g3 and g0+(g3−g2) need exact-restoration/RNG-replay counterparts; the no-projection directions also remain unrun. No new training or queue changes were made for this summary.

## Sources and verification

All34 included receipts are distinct and complete. All included configurations were checked against the common settings, including the literal r=1 value, and their raw task matrices independently reproduce FAA/AAA/BWT. Three-seed means and sample standard deviations were recomputed. The native SAM baseline uses the predefined audited S-control for seed2, not the separate earlier physical repeat.

[Table CSV](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261007/METHOD_GRADIENT_FAA_AAA_BWT.csv) · [Per-seed source ledger](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261007/SOURCE_LEDGER_34.csv) · [Full cumulative report](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-gam-cumulative-results-20261007T065249Z/REPORT.md).
