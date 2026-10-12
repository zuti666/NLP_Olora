# InfLoRA independent numerical check — 2026-09-24

Scope: `figures/lora_taskwise_20260918/aa_times/output/lora_taskwise_aa_compact_tighty` and its source manifest. This review parsed raw InfLoRA log dictionaries independently using Python `ast.literal_eval` and `statistics.mean`; it did not call the plotting parser. Original figures, logs, and plotting data were not modified.

## Findings

All four InfLoRA series agree with their selected raw logs. Each dataset uses exactly one seed-0 run. Each log has one seed header and one complete progression of accuracy rows (10 stages, or 5 for DomainNet); there is no evidence of an appended or repeated run in these four files. All raw SHA256 values agree with `sources.json`. Parsed task matrices match `output/parsed_raw_runs.json` exactly. Both per-stage AA and cumulative AAA match `output/stage_statistics.csv` and `parsed_raw_runs.json` to floating-point roundoff (maximum absolute difference 1.4211e-14 percentage points).

The figure uses task-macro accuracy: `AA_t = mean(a[t,1:t])`, `FAA = AA_T`, `AAA = mean(AA_1,...,AA_T)`. Log `total` / `(curve) top1 Acc:` values use a different, sample-weighted convention. This distinction matters particularly for ImageNet-A. Neither convention is an interpolation.

| Dataset | Figure FAA, task macro | Figure AAA, task macro | Log global top1 FAA | Log global top1 AAA | Seeds |
|---|---:|---:|---:|---:|---|
| ImageNet-R | 76.793000 | 82.6248218254 | 76.770000 | 82.567000 | 0 only |
| ImageNet-A | 47.566000 | 59.2268440476 | 48.330000 | 59.351000 | 0 only |
| DomainNet | 72.342000 | 78.9720666667 | 72.350000 | 78.982000 | 0 only |
| CIFAR-100 | 86.440000 | 91.8256785714 | 86.440000 | 91.826000 | 0 only |

The last-stage per-task accuracy vector on ImageNet-A is `[51.92,49.38,59.46,54.36,54.67,55.5,39.29,37.89,33.96,39.23]`; its arithmetic mean is 47.566, while the log `total` is 48.33. Thus a 47.566 plotted endpoint is faithful to the figure's definition, not a mistaken transcription of 48.33.

## Exact source logs and lines

- ImageNet-R: `/data/115-2/users/liying/low-rank-cl-new/logs/inflora/imagenet-r/t10/seed0.log`; task rows at lines 35, 48, 61, 74, 87, 100, 113, 126, 139, 152; final global top1 curve at 153. Manifest entry: `sources.json:1683`, path at 1686, binding warning at 1688–1689.
- ImageNet-A: `/data/115-2/users/liying/low-rank-cl-new/logs/inflora/imagenet-a/t10/seed0.log`; task rows at lines 35, 48, 61, 74, 87, 100, 113, 126, 139, 152; final global top1 curve at 153. Manifest entry: `sources.json:1709`, path at 1712, binding warning at 1714–1715.
- DomainNet: `/data/115-2/users/liying/low-rank-cl-new/logs/inflora/domainnet/t5/seed0.log`; task rows at lines 35, 48, 61, 74, 87; final global top1 curve at 88. Manifest entry: `sources.json:1657`, path at 1660, binding warning at 1662–1663.
- CIFAR-100: `/data/115-2/users/liying/low-rank-cl-new/logs/inflora/cifar100/t10/seed0.log`; task rows at lines 35, 48, 61, 74, 87, 100, 113, 126, 139, 152; final global top1 curve at 153. Manifest entry: `sources.json:1631`, path at 1634, binding warning at 1636–1637.

The final top1 curve matches the sequence of `total` values in all four logs. `top1 Acc with task` and `top1 Acc task` are different reported quantities and were not used.

## Every plotted stage

Values below are percentages, rounded only for display.

| Stage | ImageNet-R | ImageNet-A | DomainNet | CIFAR-100 |
|---:|---:|---:|---:|---:|
| 1 | 94.010000 | 75.000000 | 87.650000 | 98.400000 |
| 2 | 89.295000 | 66.805000 | 81.935000 | 94.700000 |
| 3 | 86.030000 | 66.266667 | 77.933333 | 94.066667 |
| 4 | 82.955000 | 62.447500 | 75.000000 | 93.475000 |
| 5 | 82.250000 | 59.030000 | 72.342000 | 92.840000 |
| 6 | 81.370000 | 58.586667 | — | 90.850000 |
| 7 | 79.832857 | 54.862857 | — | 90.214286 |
| 8 | 77.301250 | 51.983750 | — | 89.437500 |
| 9 | 76.411111 | 49.720000 | — | 87.833333 |
| 10 | 76.793000 | 47.566000 | — | 86.440000 |

## No numeric interpolation or smoothing

`plot_lora_aa.py:91` averages actual task scores; lines 130–134 average across available seeds; lines 139–144 plot one existing data value at each integer learned-task count. There is no spline, resampling, endpoint adjustment, or imputation. Straight lines simply join adjacent stage points. Lines 145–146 add a fixed ±0.2 pp band for these single-seed baselines. This band is an illustrative visual aid, not an empirical standard deviation or confidence interval. `restyle_aa_main.py:55,64,90–92` loads the same source runs and asserts numerical curve identity; its changed heights and y limits do not change values.

## Not bound to the historical paper table

Every InfLoRA manifest entry explicitly says `reference_only_unbound_to_table` and `different available seed0 run; not a reconstruction of reported table`. The historical reference table retained in the same manifest lists:

| Dataset | Historical table FAA | Historical table AAA | Manifest line |
|---|---:|---:|---:|
| ImageNet-R | 74.41 ± 0.63 | 80.31 ± 0.60 | 1901 |
| ImageNet-A | 50.75 ± 1.33 | 64.36 ± 1.01 | 1910 |
| DomainNet | 71.01 ± 0.05 | 77.75 ± 0.03 | 1892 |
| CIFAR-100 | 86.34 ± 0.76 | 91.33 ± 0.48 | 1883 |

The plotted data are valid reconstructions of the selected raw logs, but they are not reconstructions of these table values. The discrepancy is not explained solely by sample-vs-task weighting; the selected seed/cohort differs. Nothing here independently verifies the old table's experimental provenance.

## Training configuration and checkpoint evidence

The InfLoRA logs record the following common configuration at lines 5–20: seed 0, `load: vit_base_patch16_224_in21k`, shuffle False, Adam, learning rate 0.0005, weight decay 0, batch size 128. ImageNet-R and ImageNet-A each use 20 initial classes / increment 20 / 10 stages / rank 10. ImageNet-R uses 50 epochs, while ImageNet-A uses 20 epochs (each log line 16; final-epoch training confirmation at line 149). DomainNet uses 69 initial classes / increment 69 / 5 stages / rank 30 / 5 epochs; CIFAR-100 uses 10/10/10/rank 10/20 epochs.

Code trace under `/data/115-2/users/liying/low-rank-cl-new/`:

- `methods/inflora.py:13,26` constructs the InfLoRA `Net`.
- `models/net_inflora.py:66` passes `args['load']` with `pretrained=True`; lines 13 and 21–26 resolve the timm pretrained configuration and build the model.
- `models/vit_inflora.py:26` imports that resolver from timm; lines 141–143 declare the variant URL `https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` with 21843 classes; line 940 registers the model name.
- `README.md:41` and `environment.yaml:22` declare timm 0.6.12.

However, the selected InfLoRA logs record no resolved checkpoint path, download URL, weight hash, or historical code commit. The current code's default URL is supporting configuration evidence, not proof of the exact bytes used in the February runs. Therefore it is justified to state that the logged pretrained-model configurations differ from the FS run's explicit Hugging Face model identifier, and that identical checkpoints have not been established. It is not justified from these artifacts alone to claim the actual weight bytes necessarily differed.

For the selected ImageNet-A FS-EWC seed-0 log at `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224.log`, its last appended run records `init_epoch: 50` at line 797, `epochs: 50` at 801, and `Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)` at 824. The InfLoRA 20-versus-FS 50 epoch difference on ImageNet-A is directly evidenced. This is a comparison of historical method-specific recipes, not a controlled identical-training-recipe comparison.
