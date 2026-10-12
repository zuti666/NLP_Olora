# Existing EWC scale audit (CPU only)

The existing seed0 SGD and GAM Task-1 Fisher sidecars are already globally trace-normalized. Reuse them unchanged for the lambda sweep; retain `ewc_normalize_fisher=true` and `ewc_normalize_accumulated_fisher=true`. No Fisher reconstruction or recomputation is needed or authorized by the latest instruction.

| Anchor | Stored Fisher trace (FP64 sum) | Matrices / elements | Max entry | Stored original raw trace scalar |
|---|---:|---:|---:|---:|
| SGD | 1.000000053770413 | 24 / 14,155,776 | 0.00159069022629 | 34.06988796591759 |
| GAM | 0.9999999796979775 | 24 / 14,155,776 | 0.00165815197397 | 24.82307168096304 |

All blocks are nonnegative finite 768×768 tensors. The mean entry is about 7.06425e-8. All per-block traces, hashes, metadata and paths are in EWC_SCALE_AUDIT.json. The saved `fisher_past_trace_mass` is approximately 1 for both anchors; it is historical accumulation bookkeeping, not a missing raw-scale multiplier. The raw-trace scalars remain metadata only.

## Shared implementation and exact scaling

The formal EWC and GAM-EWC runs both instantiate `ewclora_normfisher_gam_intask` via `utils/factory.py:254`. The common learner inherits `as1_normfisher_gam`, which inherits `ewclora_youyue_fitArchitechture_gam`, then `ewclora_youyue_fitArchitechture`.

`ewclora_youyue_fitArchitechture.py:228` forms effective Q/V LoRA changes as BA. At line 270, protection is

`P = (lambda / 2) * sum_m sum_ij F[m,ij] * (BA[m,ij] - B0A0[m,ij])**2`.

There is **no matrix average and no element average in P**. `.sum()` is used; `fisher_mean` is a diagnostic, not another reduction. Eta is zero, dual ascent and projection are off in these runs. The CE is batch mean, and the stored Fisher estimator is a batch-gradient-square proxy, not the later report's per-sample raw Fisher.

Per-task global normalization is in `as1_normfisher_gam.py:177` (denominator trace + 1e-12). Accumulated normalization is in `as1_normfisher_gam_intask.py:822`; at Task 1 with no old Fisher it returns a normalized new Fisher. Gamma=0.9 affects subsequent accumulation; it does not multiply the Task-1 Fisher during Task-2 training. The Task-2 sidecar is updated after training, so its mass near 1.9 must not be mistaken for the mass used at the start of Task 2.

SGD uses `total_loss = CE + P + optional_delta_penalty` and backward once (`as1_normfisher_gam_intask.py:1172`). GAM computes the **same** P gradient with autograd (`ewclora_youyue_fitArchitechture_gam.py:259`) and adds it exactly once to the clean/shaping direction (`as1_normfisher_gam_intask.py:1019`). Its lambda is already included in P, so there is no second lambda multiplier. Resume clones the saved Fisher and restores its mass/metadata (`baseLoRA.py:734`); it does not rescale it or replace the current lambda configuration from the old checkpoint.

## Why lambda=2000 has a small visible effect

The C→C+P seed0 run is `formal_training/sequence_sgd_seed0` task index 1. There is no `B3_C_sgd_lam2000_seed0` directory. Its paired no-protection run is `B3_A_sgd_lam0_seed0`.

| Quantity, Task 2 seed0 | lambda=0 | lambda=2000 |
|---|---:|---:|
| Mean clean loss | 0.560805935487 | 0.560910116322 |
| Mean weighted EWC loss | 0 | 0.000255242659351 |
| Endpoint weighted EWC loss | 0 | 0.000378528073121 |
| Endpoint sum F ΔW² | 4.26772456068e-07 | 3.78528073121e-07 |
| a21 / a22 (%) | 83.06 / 76.38 | 83.06 / 76.38 |

The ratio of mean EWC loss to mean clean loss is 0.045505%. Protection is active: the Fisher-weighted drift decreases by 11.304%, and the effective Q/V displacement changes from 0.381290956 to 0.379516043. Equal rounded accuracy does not imply identical parameters or a disabled penalty. This is consistent with a weak overall force, but loss-scale evidence alone does not prove the complete causal explanation.

The native SGD branch did **not** record clean/Fisher gradient norms, so no observed SGD gradient ratio can be reported. As a separate GAM-EWC sanity check, `sequence_gam_seed0` task1 has mean clean norm 1.31872845031, mean Fisher norm 0.00237909847566; their ratio is 0.00180408519669. This is a **ratio of mean norms proxy**, not the mean of stepwise ratios. The separately recorded GAM mean of stepwise Fisher/clean ratios is 0.00194625199717; neither GAM number should be transplanted to C→C+P.

The seven-row report's three-seed C→C versus C→C+P averages are a21 82.41 vs 82.41 and a22 76.38 vs 76.38, while effective Q/V drift is 0.3775 vs 0.3749. B1 in that table is the native unnormalized batch proxy (SGD mean 31.441), not the stored normalized protection tensor and not the raw per-sample Fisher 1451.954.

## Sweep and stability

Recommend lambda grid **0, 2e3, 2e4, 2e5, 2e6, 2e7**, fixed seed0, fixed SGD anchor, only Task 2 trained for the original 50 epochs/lr=.02/batch128 and paired RNG. Reuse existing lambda0 and lambda2000 results after identity verification. Keep Fisher, normalization flags, scheduler, data and optimization unchanged. Add 1e6/5e6 only if needed to refine the transition.

CPU-only local curvature check at the unchanged SGD anchor gives largest P(lambda=1) Hessian eigenvalue approximately 2.89972462512e-05 in trainable A/B coordinates; the maximal block is `backbone.lora_vit.blocks.3.attn.qkv.v` (500-iteration FP64 relative residual 7.71e-08). The pure quadratic protection-only explicit-GD threshold at lr=.02 is approximately **3.4486e+06**. This is a local diagnostic, not a full-training stability guarantee. It predicts 2e7 may be numerically unstable at the unchanged learning rate.

At large lambda, record nonfinite loss/gradients and stop that run promptly; do not treat a divergent checkpoint as evidence for a retention/plasticity tradeoff. Check finite metrics, Task-2 learning, Fisher drift and old/new accuracy together. If a smaller learning rate is later used, label it as a separate learning-rate experiment rather than silently changing the lambda sweep.

Audit touched only this Markdown and its JSON companion. No GPU was used and no training or existing artifact was modified.
