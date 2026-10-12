# ImageNet-A at r=1: method, gradient and step-size summary

This version adds the Panel B baselines (SGD, SAM and g2 directions), the Panel A learning-rate and step-matched controls (SAM ×2 LR, Mean ×2 LR, Gmatch), and the update-norm ratio |U|/|g1|. Every included cell uses the literal outer radius r=1.

Common training configuration: ImageNet-A, 200 classes in a fixed order, 10×20-class tasks; ViT-B/16 (augreg2_in21k_ft_in1k), LoRA rank10; outer radius r=1, inner radius rho=.05; 50 epochs per task, batch128; SGD with zero momentum and weight decay; base learning rate .04 for Task0 and .02 thereafter, cosine decay to 10% of each initial rate. EWC/Fisher, weight merging and replay are disabled. The only deliberate deviation is the LR scale column: ×2 rows double both the peak and floor learning rates; the Gmatch row keeps the base schedule but rescales each step so that its effective LoRA-BA plus classifier displacement equals the independent same-seed SAM run at the same task and step.

Values are task-macro FAA/AAA (%) and BWT (percentage points); mean±sample SD (ddof=1) over the listed seeds. † seed2 only. ‡ seeds0/1 only (see note below). Higher is better for all three metrics.

|U|/|g1| is the per-step ratio of the final update-direction norm to the SAM-probe gradient norm at the same parameter state and batch, over all trainable LoRA and seen-classifier parameters, averaged over the 2550 steps of each run and then over seeds. It excludes the LR scale, so the step relative to a SAM step at the same state is approximately LR scale × |U|/|g1|. * marks ratios computed from seeds0/1 only, because the historical seed2 runs of those conditions have no per-step trace.

Define h+=g1+g3 and h−=g0+g2. The positive probes are g0=g(theta), g1=g(theta+rho N(g0)), g2=g(theta+r N(g1−g0)) and g3=g(theta+r N(g1−g0)+rho N(g2)).

## A. Original execution protocol

Four gradient probes plus one clean CE forward; global RNG progression and native perturbation subtraction/restoration.

| Method | Gradient direction | LR scale | \|U\|/\|g₁\| | FAA ↑ | AAA ↑ | BWT ↑ |
|---|---|---|---:|---:|---:|---:|
| Accelerated GAM (original) | $h_+-0.1\Pi^\perp_{h_+}(h_-)$ | ×1 | 2.97* | 54.977±0.351 | 68.333±0.281 | -6.916±0.226 |
| Unprojected branch difference (gamma=1) | $h_+-h_-$ | — | — | — | — | — |
| GAM without orthogonal correction (gamma=0) | $g_1+g_3$ | ×1 | 2.93* | 54.810±0.148 | 68.117±0.347 | -6.684±0.823 |
| Mean of g1 and g3 | $(g_1+g_3)/2$ | ×1 | 1.47* | 50.373±1.162 | 66.470±0.890 | -6.885±1.010 |
| Mean of g1 and g3, 2x LR | $(g_1+g_3)/2$ | ×2 | 1.46 | 55.219±0.475 | 68.423±0.369 | -6.550±0.306 |
| SGD-direction control | $g_0$ | ×1 | 0.88 | 40.063±0.726 | 60.134±0.709 | -12.809±0.662 |
| SAM-direction control | $g_1$ | ×1 | 1.00 | 45.559±0.511 | 63.204±0.885 | -9.800±1.156 |
| SAM-direction control, 2x LR | $g_1$ | ×2 | 1.00 | 46.048±0.675 | 63.437±0.743 | -10.601±1.254 |
| SGD + second local difference | $g_0+(g_3-g_2)$ | ×1 | 1.03 | 46.921±2.200 | 63.128±2.683 | -7.134±0.801 |
| Outer SAM-probe gradient | $g_3$ | ×1 | 1.96* | 50.537±0.870 | 66.203±0.588 | -6.389±0.308 |
| Outer SAM-probe gradient, SAM-matched step (Gmatch) | $g_3$ | S-matched (×0.78) | 2.02 | 48.373±1.108 | 64.932±0.704 | -7.963±0.739 |
| Outer-location gradient | $g_2$ | ×1 | 1.93 | 48.451±2.510 | 63.924±1.134 | -7.354±1.757 |

The gamma=0 row is the branch sum g1+g3. Mean ×2 LR is the same nominal update as the branch sum, so the Mean and Mean ×2 rows differ only in step size. Gmatch uses the g3 direction with the SAM run's step length.

## B. Exact restoration and RNG replay protocol

Four probes with exact full-state restoration, per-probe RNG replay, and a per-epoch data seed of seed×100000+task×1000+epoch. The SGD, SAM and g2 baselines evaluate the probe set {g0, g1, g1 at theta−rho N(g0), g2} instead of {g0, g1, g2, g3}; the g2 location is identical, and exact restoration means the unused probes leave no state change. Compare within each panel.

| Method | Gradient direction | LR scale | \|U\|/\|g₁\| | FAA ↑ | AAA ↑ | BWT ↑ |
|---|---|---|---:|---:|---:|---:|
| SGD-direction control‡ | $g_0$ | ×1 | 0.88 | 40.790±2.218 | 61.141±0.232 | -12.089±1.503 |
| SAM-direction control‡ | $g_1$ | ×1 | 1.00 | 45.773±0.258 | 64.118±0.138 | -9.478±1.080 |
| Outer-location gradient | $g_2$ | ×1 | 1.94 | 49.694±0.537 | 65.212±0.889 | -6.831±2.027 |
| SAM + second local difference | $g_1+(g_3-g_2)$ | ×1 | 1.09 | 49.610±1.746 | 65.985±0.382 | -6.621±0.845 |
| Sum of g1 and g2 | $g_1+g_2$ | ×1 | 2.92 | 55.911±1.033 | 68.609±0.353 | -5.544±0.639 |
| Mean of g1 and g2 | $(g_1+g_2)/2$ | ×1 | 1.43 | 51.225±1.559 | 66.837±0.262 | -6.243±0.928 |
| SAM + outer displacement residual† | $g_1+(g_2-g_0)$ | ×1 | 1.97 | 46.873 | 64.775 | -8.224 |

‡ The SGD and SAM baselines have no r=1 seed2 run. Their seed2 runs used r=0.25 and are excluded here. Their final update (g0 or g1) does not depend on r, and the r-dependent probe is discarded under exact restoration, so including them would be mathematically equivalent but not literally identical. For reference, the three-seed values with that cell would be: P01: FAA 40.574±1.613, AAA 60.873±0.492, BWT -12.084±1.063; P02: FAA 45.375±0.714, AAA 63.823±0.520, BWT -9.699±0.854.

## Remaining gaps

The unprojected difference h+−h− and the gamma=.1 unprojected variant h+−.1h− have no completed runs. No step-matched control exists for the branch sum or original GAM (|U|/|g1|≈3), and SAM was tested up to 2× LR only.

## Sources and verification

50 validated cells. Each configuration was checked against the common settings, including r=1 and the stated LR scale. Each raw task matrix reproduces its FAA/AAA/BWT, and each trace covers exactly 2550 steps. Per-seed values and trace paths are in the source ledger.

[Table CSV](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261012/METHOD_GRADIENT_STEP_FAA_AAA_BWT.csv) · [Per-seed source ledger](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261012/SOURCE_LEDGER.csv) · [Excluded cells](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261012/EXCLUDED_CELLS.csv) · [Previous version](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-english-method-summary-20261007/REPORT_EN.md)
