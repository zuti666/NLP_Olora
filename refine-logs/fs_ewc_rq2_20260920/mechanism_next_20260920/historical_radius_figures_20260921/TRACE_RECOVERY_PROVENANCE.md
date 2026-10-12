# Historical fixed-anchor trace recovery: provenance and limits

Audited 2026-09-21 from the historical JSONL records, stage1/stage2 logs and metrics, and CPU-only checkpoint inspection. No experiment or GPU computation was performed.

## Quantity recovered from the quadratic

The saved historical quantity is `Q_F(r,u) = 0.5 * r^2 * u^T F_norm u`, with a fixed diagonal anchor Fisher and unit directions sampled from normalized isotropic Gaussians. For `d = 14,155,776`, uniform-sphere averaging gives

`T_hat(r) = 2 * d * mean_u(Q_F(r,u)) / r^2`, for `r > 0`.

This is a Monte Carlo estimate of `trace(F_norm)` at the fixed anchor. At `r=0` the inversion is undefined (`0/0`); do not set the inferred trace to zero. Historical source snapshot `cbcde36ff815cc060f937e7cc812c8870690e237` in `/data/140-1/users/liying/CopyFlatCV` supports the direction/formula definitions in `utils/mechanism_eval.py:812,1275,1310,1340`: fresh directions are drawn inside each radius loop. The snapshot postdates the June-14/15 execution and is supporting source evidence, not a saved exact execution-source hash.

The plotted training-prefix records contain 80 nonzero radii and 100 directions per radius: 8,000 direction draws per optimizer. Independent recomputation gives:

| Method | Pooled normalized trace estimate | Direction-only SEM |
|---|---:|---:|
| SGD | 1.000089513150975 | 0.00010261035276413494 |
| GAM | 1.000068756046705 | 0.00010200552287740958 |

The test-prefix records are not included in these summaries. This SEM describes direction sampling only; it is not uncertainty across training seeds or datasets. Variation with radius after division by `r^2` is sampling variation for a fixed Fisher, not measured Fisher evolution with radius.

## Exact source of the optional raw anchor scale

The scale field in both stage1 metrics files is:

```python
metrics["mechanism"]["steps"]["0"]["in_task_fisher_final_raw_trace"]
```

- SGD: **29.875783436000347**, from [stage1 SGD metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/sgd_initlr0.04_lr0.02_e50_task0_shared/imageneta/0/s1c_sgd_task0_shared_e50_lam2000_s0/exp_run/20/s1c_sgd_task0_shared_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json).
- GAM: **26.164499521255493**, from [stage1 GAM metrics](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/gam_initlr0.04_lr0.02_e50_task0_shared/imageneta/0/s1c_gam_task0_shared_e50_lam2000_s0/exp_run/20/s1c_gam_task0_shared_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json).

Both records have `approximate_fisher_in_task=false`, `in_task_fisher_used_path=false`, and `in_task_fisher_snapshots=0`. Thus the recorded raw scale is the endpoint-only Fisher scale, with no path-Fisher mixing.

The additional scale is supplied by these known anchor logs. The normalized quadratic records alone do not identify the original raw scale.

## Link from the raw scale to the Fisher used in the sweep

1. The stage1 logs record the endpoint-only raw Fisher, then save the corresponding task0 checkpoint: [SGD endpoint-only calculation](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage1/task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/sgd_seed0.log:468), exact mechanism fields at line475, and checkpoint save at line476; [GAM endpoint-only calculation](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage1/task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/gam_seed0.log:468), exact mechanism fields at line475, and checkpoint save at line476.
2. The radius sweep explicitly loads those same method-specific Fisher checkpoint paths: [SGD stage2 load](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/sgd_seed0.log:126) and [GAM stage2 load](/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/shell/gam_seed0.log:124).
3. Original Fisher checkpoints: [SGD fisher_state_0.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/sgd/seed0/checkpoints/fisher_state_0.pt) and [GAM fisher_state_0.pt](/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/gam/seed0/checkpoints/fisher_state_0.pt). CPU `torch.load(..., map_location="cpu", weights_only=True)` inspection confirms the checkpoint, stage1 endpoint metrics, and stage2 loaded/endpoint metrics all agree on **24 Fisher blocks** and **14,155,776 coordinates**. Their maxima match exactly:

| Method | Checkpoint and stage1/stage2 Fisher maximum | Saved `fisher_past_trace_mass` |
|---|---:|---:|
| SGD | 0.001844932558014989 | 0.9999999906867743 |
| GAM | 0.0022466813679784536 | 0.9999999774154276 |

The checkpoints contain `fisher_past_delta`, `fisher_past_trace_mass`, and `last_task_new_fisher_stats`; they do **not** contain the raw trace field. The raw-scale link is established by the original stage1 metrics and save/load logs plus matching Fisher summaries, not by a raw tensor stored in the checkpoint. This is historical provenance evidence, not a cryptographic proof that raw tensors can be reconstructed bit for bit.

## Two normalizations and finite precision

Historical `as1_normfisher_gam_intask.py` in snapshot `cbcde36ff815cc060f937e7cc812c8870690e237` records `final_trace` before normalizing the raw endpoint Fisher (lines666–708); `_trace_normalize_fisher` divides by `trace(F) + eps` (lines227–233). At task0, `_accumulate_fisher` normalizes the new Fisher again when there is no old Fisher (lines813–815). Both execution logs specify `ewc_fisher_norm_eps=1e-12` and enable these normalizations.

Consequently, multiplying `T_hat(r)` by the recorded raw anchor trace is an **approximate raw-scale recovery**. Strict inversion would also account for the near-one second normalization factor and float32 rounding; multiplying by the raw trace alone is accurate to roughly the normalization's float32 precision, far below the direction-sampling uncertainty. Do not describe it as exact tensor reconstruction. As a consistency check, CPU per-block float32 sums of the saved checkpoints are 1.000000016996637 (SGD) and 0.9999999704305083 (GAM), while the historical flattened endpoint traces are 1.0 and 0.9999999403953552; reduction order/device roundoff explains these tiny differences.

## Interpretation boundary

The optional raw-rescaled curve is `raw_anchor_trace * T_hat(r)`: repeated direction estimates of one **fixed raw anchor Fisher proxy**. The underlying estimator is the weighted square of minibatch-mean gradients,

`F_raw[j] = (1/N) * sum_batches |batch| * (d mean_batch_CE / d DeltaW[j])^2`.

It is not the per-example empirical Fisher. Neither division by `r^2` nor multiplication by an anchor scale produces Fisher recomputed at `theta_anchor + r*u`, a radius-dependent neighborhood trace, or a per-example empirical trace. A different SGD/GAM vertical level in the raw-rescaled plot comes from the already recorded raw anchor scales; it is not new raw-scale information recovered from the normalized quadratic values alone.
