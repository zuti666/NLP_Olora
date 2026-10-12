# Historical reference audit: S3 Damp C3, ImageNet-A

Audit date: 2026-09-20. This audit did not edit training source or launch training. The parent campaign executes the eight new runs listed below.

## Validity and performance

Validated **6/6** expected seeds 0–5: each completed 10 tasks, has a unique matched raw metrics file, finite observed CNN matrix entries, and unchanged log/metrics SHA256 relative to the existing audit. Missing: 0. Failed: 0. Upper-triangle matrix nulls are structural, not failed measurements. Unobserved FWT/AFG fields are not training failures.

| Seed | FAA | AAA (stage mean) | BWT | Forget | Sample Final | Sample AIA |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.316 | 67.074 | -7.332 | 7.551 | 59.120 | 70.485 |
| 1 | 50.585 | 67.029 | -8.582 | 8.582 | 57.800 | 70.641 |
| 2 | 53.311 | 66.700 | -6.371 | 6.371 | 60.700 | 70.426 |
| 3 | 51.968 | 67.528 | -8.170 | 8.417 | 59.970 | 71.147 |
| 4 | 52.236 | 67.134 | -7.844 | 8.020 | 59.510 | 70.470 |
| 5 | 50.880 | 66.184 | -7.393 | 7.907 | 57.800 | 69.628 |
| Mean ± sample SD | 51.883 ± 1.005 | 66.942 ± 0.456 | -7.616 ± 0.771 | 7.808 ± 0.794 | 59.150 ± 1.171 | 70.466 ± 0.490 |

FAA is the mean of the final row over tasks; AAA is the mean over stages of the mean over seen tasks; BWT is final minus diagonal accuracy over old tasks. Forget uses each old task’s maximum from its learning stage through the final stage, minus final accuracy. Sample Final/AIA weight test examples and differ because task sizes differ. The raw JSON `cnn.final.AAA` is the lower-triangle overall mean (60.629818 for seed2), not the stage-average AAA (66.700178). Standard deviations use n−1.

## Exact reference and configuration

- Reference log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s2_gpu3.log`
- Config fields: log lines 1–81; runtime identity: 92–95; sample final/AIA: 730–735; full CNN matrix: 747–775.
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s2.yaml`
- Launcher: `/data/140-1/users/liying/CopyFlatCV/scripts/run_s3_damp_c3_multiseed.sh`
- Log SHA256: `c77fb17130f92982aa95e1e9bc1db4f26dd342d8236c7bc7b8d3bf583e856c13`
- Raw metrics SHA256: `bf1eba7b7fecc300ec79bcdf41a0faa1134a90d8fb36b880963925c3b0b14b85`

ImageNet-A, 200 classes, 10×20 tasks, fixed class order; timm ViT-B/16 `augreg2_in21k_ft_in1k`, Q/V LoRA rank10; 50 epochs/task, batch128, SGD, first-task LR .04, later .02, momentum/weight decay 0, cosine minimum LR ratio .1. EWC λ=2000, γF=.9, η=0, per-task and accumulated trace normalization on, damping=.2. GAM strength `lambda_flat=1`, ρ=.05, r=.2, β1=β2=β3=1, γG=.1. A on: LoRA factor snapshots epochs {20,40}, training-CE softmax path β=3, ω=.3, endpoint classifier.

**This run is already WMergeThenFisher.** Runtime explicitly records `fisher_snapshots=False`, `fisher_source=post_weight_merge`: merge A/B first, then estimate batch-gradient ΔW Fisher proxy. The old config `approximate_fisher_in_task=true` does not mean it estimated trajectory Fisher snapshots. Fisher uses the shuffled training loader and train transforms, batch128, up to100 batches; reference task0 used1259 examples and task9 used378.

## Native frozen-source compatibility

Recommended source: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-paper-gapfill-20260918/attempt2/code`. Manifest: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-paper-gapfill-20260918/attempt2/source_frozen.json`. **190/190 source hashes match**. Run native `src.main`; do not call `runtime_v3/protocol.py::install()`.

- The frozen `utils/factory.py` lines254–257 map the reference alias to WMergeThenFisher. Native `src.main` does not automatically import/install the frozen paper runtime.
- The frozen LoRA forward retains the historical four randomly initialized temporary Linear modules. Current worktree `backbone/lora.py` has removed them, changing random-stream consumption; do not silently substitute the current worktree.
- Git commit `7c6aaf2` is dated 2026-06-29 09:56:57 +0200, before the reference log starting14:18. Frozen `as1_normfisher_gam_intask.py`, `as1_normfisher_gam_intask_damp.py`, `optimer_PerturabtionType/gam.py`, and `backbone/lora.py` are byte-identical to that commit.
- Later WMergeThenFisher additions implement posterior/anchor diagnostics with defaults off. Later source differences checked include optional receipt/resume validations, unrelated datasets, extra evaluations in alternative training branches, and trace-normalization refactoring; both normalization switches remain true here.
- DataManager later removed a hardcoded data root. Set `data_root=/data/140-1/datasets` explicitly. Fix `train_num_workers=8`, `eval_num_workers=8` to legacy defaults.
- Native default Fisher is `batch_empirical`; make it explicit. Set `fisher_split=false`; use training input source. Preserve `ewc_preserve_fisher_rng=false`. Independent loader and clean-RNG isolation belong to the paper runtime and must not be added here.
- Keep `mechanism_grad_stats=true`, `compute_nme_class_means=true`, and diagnostic execution paths consistent. Disabling NME changes the number of forwards and therefore the legacy random trajectory.
- Keep task-end Fisher and GAM probes running for λ=0 and strength=0. Do not set `ewc_skip_task_fisher=true` for the zero-λ arm.
- No source-hash receipt from the June reference execution exists. This is a historical-configuration reproduction on an audited native snapshot, not a guarantee of bitwise replay. The new center is required for within-campaign comparisons.

## New matched campaign

Eight full runs, all seed2 and A on; center λ=2000 / strength1 / training r=.2 with innerρ=.05:

| Arm | λEWC | GAM strength | Training r |
|---|---:|---:|---:|
| center | 2000 | 1 | 0.2 |
| lambda0 | 0 | 1 | 0.2 |
| lambda200 | 200 | 1 | 0.2 |
| lambda20000 | 20000 | 1 | 0.2 |
| strength0 | 2000 | 0 | 0.2 |
| strength05 | 2000 | 0.5 | 0.2 |
| radius01 | 2000 | 1 | 0.1 |
| radius05 | 2000 | 1 | 0.5 |

Use the newly rerun center for each same-seed slice. Historical sixseed performance is a separate reference. This campaign is exploratory single-seed sensitivity; no seed error bars or claims of a statistically selected optimum. All failures and all points must remain visible.

## Reuse restrictions

- The 20260918 paper plan uses `sample_empirical`, a500-example budget, batch32, deterministic test transforms, isolated loader/clean/Fisher RNG, and an A-off center. Its completed centers do not substitute for this legacy A-on batch-proxy center.
- Historical joint ω=.05/.1/.2/.3/.4/.5 varied Fisher and weights together in an earlier learner. It is not a controlled ω sweep of this post-merge Fisher configuration.
- Snapshot-window runs, alternate learning rates, damping points, and λ results may use different versions or scientific settings; retain their provenance and do not splice them into these single-axis curves.
- Diagnostic perturbation radii do not replace training-r experiments. Filename r10 denotes LoRA rank10. Extra ΔW penalty radii define a different constraint.

Existing background audits (read as separate protocol strata):

- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/IMAGENETA_ABLATION_REUSE_AUDIT_20260918.md`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/LOGS_TRY_HYPERPARAM_TASK_RADIUS_AUDIT_20260918.md`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/IMAGENETA_EXPERIMENT_PLAN_20260918.md`
- `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/results/logs_try_audit_20260918/launch_inventory.csv`

## Per-seed raw sources

### Seed 0

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s0_gpu1.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s0.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/0/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s0_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `b0b9978b8e4472fe781c43748e87f61d41415e67cc6921f2607eaaaa3e0eb0b7`
- Metrics SHA256: `d9fdede281f64740f53c61f6dc60200d424679c9f09c4a69a3594a2c0c80ac96`

### Seed 1

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s1_gpu2.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s1.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/1/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s1_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `62e55fa99eb94f64043a2c15c7afc5b0039aa2ef8a952fa38b826624eb87e985`
- Metrics SHA256: `e9e4d5ea260bd7d3229bbf2bf7e0d0442f72009c8402237c2deca135fc829ec2`

### Seed 2

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s2_gpu3.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s2.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/2/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `c77fb17130f92982aa95e1e9bc1db4f26dd342d8236c7bc7b8d3bf583e856c13`
- Metrics SHA256: `bf1eba7b7fecc300ec79bcdf41a0faa1134a90d8fb36b880963925c3b0b14b85`

### Seed 3

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s3_gpu4.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s3.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/3/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s3_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `d9e5dc32aeb979b1c9568c6e32b85e653f8748488d4cb2ce4e376f95d0d952de`
- Metrics SHA256: `ee79afd35b418eadce11028513a4f678d155e886ede993893e56cb984e94ad8d`

### Seed 4

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s4_gpu5.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s4.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/4/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s4_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `67aad7c3a30138dce718a121aea7d99eeb1091f965a6e1d68cdf61b9b27bbeba`
- Metrics SHA256: `291cd645ad4a03580a923db551f92cb777e93c33cb5021a79a01be3666bfd255`

### Seed 5

- Log: `/data/140-1/users/liying/CopyFlatCV/logs_try/logs_s3_damp_c3_multiseed/s3_damp_c3_alpha02_int20_s5_gpu6.log`
- Config: `/data/140-1/users/liying/CopyFlatCV/config_save/s3_damp_c3_alpha02_int20_s5.yaml`
- Matrix: `/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/5/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s5_vit_base_patch16_224_cl_metrics.json`
- Log SHA256: `4e5addf905d4a1ee80412b7c1ce783eec1a21514520f1fad49178bfcd9921dbd`
- Metrics SHA256: `e6f3c045d001187287f6a33726e49d792e618cb4cc20d0720897975dfa9db18a`

