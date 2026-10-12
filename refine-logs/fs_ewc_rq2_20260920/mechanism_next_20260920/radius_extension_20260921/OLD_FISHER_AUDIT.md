# Audit of historical radius Fisher fields

Audited 2026-09-21. Read-only inspection of both user-named log trees and associated `outputs_logs` trees; this report is the only new file. No GPU work.

The remembered **81-radius sweep exists**, and every radius contains 100 Fisher quadratic values. These are evaluations of a fixed, trace-normalized diagonal Fisher at the task-0 anchor, not freshly estimated Fisher traces at perturbed parameters. Historical raw anchor traces and full-model taskwise raw traces also exist; they are distinct quantities.

## Radius records and exact fields

Directory: [/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial).

There are four files: SGD/GAM × `endpoint_train`/`endpoint_test`. For example, [SGD train JSONL](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial/s1c_sgd_r_fqa_from_task0_s0_task00_endpoint_train_fqa_partial.json) and [GAM train JSONL](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial/s1c_gam_r_fqa_from_task0_s0_task00_endpoint_train_fqa_partial.json). Each contains exactly 81 JSONL records, `radius = 0, 0.25, ..., 20`, `K = 100`; every `per_u` array has length 100. The earlier sibling `stage2_r` contains two train files with 55 radii ending at 3, so the 81-point sweep is specifically `stage2_r-2`.

Each record has `radius`, `radius_idx`, `radius_count`, `K`, `main_prefix`, `extra_prefixes`, `per_u`, `mean`, `hessian_ref`. Here `extra_prefixes=[]` and `hessian_ref=null` throughout. The six `per_u` fields are:

- `loss_delta`, `perturbed_acc`, `acc_delta`, `linear_term`, `linear_term_abs`;
- `endpoint_train_fisher_quad` or `endpoint_test_fisher_quad`.

`mean` contains `loss_delta_abs`, `loss_delta_signed`, `base_acc`, `perturbed_acc_mean`, `perturbed_acc_std`, `acc_delta_mean`, `acc_delta_std`, `linear_term_abs`, `linear_term_signed`, plus the corresponding prefix's `fisher_quad`, `error`, `signed_error`, `relative_error`.

Exact representative `mean.*_fisher_quad` values (JSONL lines 1, 41, 81):

| Optimizer | Prefix | r=0 | r=10 | r=20 |
|---|---|---:|---:|---:|
| GAM | endpoint_test | 0 | 3.531601578288246e-06 | 1.412520850863075e-05 |
| GAM | endpoint_train | 0 | 3.533234121277928e-06 | 1.412105939380126e-05 |
| SGD | endpoint_test | 0 | 3.530829644660116e-06 | 1.41231830639299e-05 |
| SGD | endpoint_train | 0 | 3.533682502165902e-06 | 1.413635891367448e-05 |

For SGD/train, the first direction's `per_u.endpoint_train_fisher_quad[0]` is respectively `0`, `3.582690766279484e-06`, `1.4141181736704311e-05`. Direction index is an ordinal within each radius: directions are freshly sampled inside the radius loop, not one stored direction reused across all radii.

The [corresponding SGD aggregate metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/sgd_r_fqa_from_task0/imageneta/0/s1c_sgd_r_fqa_from_task0_s0/exp_run/20/s1c_sgd_r_fqa_from_task0_s0_vit_base_patch16_224_cl_metrics.json) use `mechanism.steps["0"].endpoint_train_fqa_r10_fisher_quad_mean` (and `_std`, `_fisher_available`), with radius tags such as `r0p25`. The anchor summary has `loaded_fisher_tensors=24`, `loaded_fisher_numel=14155776`, `loaded_fisher_sum=0.9999999906867743` (GAM: `0.9999999774154276`). Train and test prefixes use this same stored Fisher; the prefix changes the loss/accuracy dataset, not the Fisher's estimation split.

## What was computed

For a unit direction u and radius r, the saved scalar is

`Q_F(r,u) = 0.5 * r^2 * u^T diag(F_anchor / (trace(F_anchor)+eps)) u`.

It has **the factor 0.5 and no EWC lambda multiplier**. The radius loop evaluates loss and accuracy at `theta_anchor + r*u`, restores the anchor, and computes Q_F from the unchanged Fisher vector. It never differentiates the perturbed model to re-estimate Fisher in this loop. Therefore `Q_F(0,u)=0` is required; it does not mean the anchor's Fisher trace is zero. For uniform sphere directions and trace≈1, its expectation is approximately `r^2/(2*14155776)`, explaining the nearly equal SGD/GAM means.

Current source evidence: [stored-F caller](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:188), [flatten before sweep](/data/140-1/users/liying/CopyFlatCV/utils/mechanism_eval.py:2536), [perturb/evaluate/restore](/data/140-1/users/liying/CopyFlatCV/utils/mechanism_eval.py:1390), [quadratic formula](/data/140-1/users/liying/CopyFlatCV/utils/mechanism_eval.py:1422).

The original stored F is a **square-of-minibatch-mean-gradient proxy**, not a per-example empirical Fisher:

`F_raw[j] = (1/N) * sum_batches |batch| * (d mean_batch CE / d DeltaW[j])^2`.

I also checked historical repository snapshot `cbcde36ff815cc060f937e7cc812c8870690e237` (2026-06-22), because current loaders have changed since these runs. In that snapshot, `models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:209–215` passes `train_loader`; lines 518–556 perform one mean-CE backward per batch and the formula above. `as1_normfisher_gam_intask.py:519–529` calls that raw estimator, and lines 227–233 / 706–708 apply trace normalization. These agree with the June run logs; the snapshot is supporting historical source, not a claim that an exact June-14 source hash was saved.

The [SGD stage-1 metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/sgd_initlr0.04_lr0.02_e50_task0_shared/imageneta/0/s1c_sgd_task0_shared_e50_lam2000_s0/exp_run/20/s1c_sgd_task0_shared_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) and [GAM stage-1 metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/gam_initlr0.04_lr0.02_e50_task0_shared/imageneta/0/s1c_gam_task0_shared_e50_lam2000_s0/exp_run/20/s1c_gam_task0_shared_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) preserve `mechanism.steps["0"].in_task_fisher_final_raw_trace = 29.875783436000347` and `26.164499521255493`, respectively. These are **single anchor raw traces**, not radius curves. `approximate_fisher_in_task=false`, `in_task_fisher_used_path=false`, snapshots=0. The run used batch size 128 and at most 100 batches.

## The two log families describe different experiments

| Property | `sgd_gam_s1c` radius sweep | `sgd_gam_s1c_fullmodel` |
|---|---|---|
| Dataset | ImageNet-A (`imageneta`) | ImageNet-R (`imagenetr`) |
| Model | `ewclora_normfisher_gam_intask`, LoRA rank 10 | `ewcon_probe`, full-model tuning |
| Backbone | `vit_base_patch16_224` | `vit_base_patch16_224` |
| Class schedule | 20 initial +20, 10 tasks, seed0, no shuffle | Same class counts/order settings |
| Examined axis | Radius around task0, classes [0,20), train and test loss | Task sequence: stage1 task0, stage2 tasks1–9 |
| Fisher coordinates | 24 query/value Delta-W blocks, 14,155,776 entries | 150 trainable backbone blocks, 85,798,656 entries; classifier excluded |
| Normalization | Trace-normalized anchor | Raw task/all-seen proxies |

Configuration evidence: [radius log](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/sgd_seed0.log:6) and [fullmodel log](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c_fullmodel/staged/stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/shell/sgd_T10_seed0.log:6). Both load the timm ViT-B/16 `augreg2_in21k_ft_in1k` checkpoint.

## Full-model taskwise traces

The fullmodel tree contains no `stage2_r` folder, radius JSONL, or `_fqa_r*` metrics. Its [SGD T metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c_fullmodel/stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/trainer_outputs/logs_inc/ewcon_probe/sgd_fm_T10_lr0.01_e50/imagenetr/0/s1c_fm_sgd_T10_e50_lam2000_s0/exp_run/20/s1c_fm_sgd_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) and [GAM T metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c_fullmodel/stage2_T/T_from_task0_fm_lam2000_lr0p01_e50/trainer_outputs/logs_inc/ewcon_probe/gam_fm_T10_lr0.01_e50/imagenetr/0/s1c_fm_gam_T10_e50_lam2000_s0/exp_run/20/s1c_fm_gam_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json) contain two different raw trace quantities:

- `accum_fisher_trace` / `accumulated_fisher_total_mass`: unnormalized online accumulation `Omega_t = 0.9*Omega_(t-1) + F_task,t`. Each task F is the NCM cosine-logit (`tau=16`) mean-CE batch-gradient-square proxy over that task's 20 training classes, using shuffled train batches of size128, up to100 batches. The metadata explicitly says `fisher_is_per_sample_empirical=false`.
- `full_fisher_trace`: recomputed at the current task endpoint on **all seen training classes** using ordinary classifier logits and CE, not the NCM logits above. Uses batch128, shuffle=false, train transforms, model eval, no batch cap (`full_fisher_max_batches=null`), and the same batch-gradient-square formula. It is a raw diagonal trace, not an exact full Fisher matrix and not a per-example empirical trace.

At zero-based task9 (200 classes, 24,000 all-seen training examples), `full_fisher_trace` is SGD `2635.698486328125`, GAM `240.57546997070312`; `accum_fisher_trace` is SGD `439.24615478515625`, GAM `161.9191436767578`. At task1 (40 classes, 5,484 examples), the full traces are SGD `394.668701171875`, GAM `43.737762451171875`. `fisher_num_samples` instead refers to the current-task NCM estimator (2,889 at task1; 2,860 at task9).

Source: [NCM estimator](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1514), [online accumulation](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:1346), [all-seen estimator](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:2771), [trace fields](/data/140-1/users/liying/CopyFlatCV/models_CL/EWCon_Probe.py:2888). Historical snapshot `cbcde36...` has the same definitions at lines923–968, 792–809, 2159–2209, respectively, with no trace-normalization step.

## Search coverage and reuse limit

Checked all six historical radius JSONL schemas, all 16 `sgd_gam_s1c` aggregate metrics files and all four `sgd_gam_s1c_fullmodel` aggregate metrics files, and file inventories including ignored files. The only Fisher-bearing radius metric suffixes are `fisher_available`, `fisher_quad_mean`, `fisher_quad_std`; there is no stored per-radius recomputed raw pointwise/neighborhood Fisher trace in these inspected trees. Other S1C task-trajectory files do contain raw trace fields such as `task0_fisher_current_trace`, but their axis is trained-task endpoint, not perturbation radius.

These old radius data can support loss/accuracy and fixed-anchor quadratic comparisons. They cannot supply the current requested raw Fisher trace at each perturbed radius. Scaling an old quadratic by its anchor's raw trace would still leave it a fixed-anchor quadratic, not a newly evaluated Fisher trace.
