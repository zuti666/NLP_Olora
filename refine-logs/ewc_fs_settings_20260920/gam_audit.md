# Current fine-grained GAM-EWCLoRA settings

Scope: the five datasets and the exact runs used by `figures/finegrained_taskwise_aa_20260920/data/curves.json`. This is not a new selection of a different FS-EWC campaign.

| Dataset (table order) | EWC lambda | GAM `grad_rho` | GAM `grad_norm_rho` | GAM `grad_gamma` | LoRA rank | Batch | LR | Epochs/task | Seed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Aircraft | 2000 | 0.01 | 0.10 | 0.02 | 16 | 128 | 0.05 | 40 | 0 |
| Cars | 2000 | 0.05 | 0.20 | 0.03 | 16 | 128 | 0.05 | 40 | 0 |
| CUB-200 | 2000 | 0.20 | 0.20 | 0.10 | 16 | 128 | 0.01 | 40 | 0 |
| Flowers | 2000 | not archived explicitly | not archived explicitly | not archived explicitly | 16 | 128 | 0.0025 | 40 | 0 |
| 5Datasets | 2000 | 0.20 | 0.20 | 0.10 | 16 | 128 | 0.02 | 30 | 0 |

The recorded common settings are `ewc_gamma=1`, `ewc_eta=0`, `ewc_max_batches=100`, `ewc_normalize_fisher=true`, `ewc_fisher_norm_eps=1e-12`, `ewc_dual_ascent=false`, and `lambda_flat=1`. There is no per-task tuning of lambda in these runs. `grad_beta_1/2/3=1` and `gam_adaptive=false` are recorded for Aircraft/Cars/CUB-200/5Datasets; Flowers does not explicitly record these fields either.

Flowers' original log, original metric file and extant original YAML were searched for the GAM radius, gradient decomposition, damping and merge fields. They do not expose them. The current implementation's defaults are not evidence that the historical run used those exact values.

Source paths, hashes, explicitly archived values, and log line numbers are preserved in `gam_selected_settings.json`; absent fields are JSON null. The metric file of every selected run was re-hashed against its verified source hash.

## Why the manuscript paragraph cannot be attached unchanged

The selected model identifier is `ewclora_normfisher_gam`. The factory maps it to `as1_normfisher_gam.Learner` ([factory.py](/data/140-1/users/liying/CopyFlatCV/utils/factory.py:181)), which inherits the explicit GAM-plus-past-Fisher learner. Its update combines `g_clean + lambda_flat * (g_gam - g_clean) + g_fisher` ([learner](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:24)).

This is separate from `as1_normfisher_gam_intask_damp_weightmergethenfisher` and from the squared-gradient finite-difference FS-EWC procedure written in the current manuscript. The latter explicitly rescales gradient differences by `2/s(u)` ([manuscript](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper/iclr2027_liying.tex:1465)); the selected GAM optimizer instead uses four gradients and a gradient-decomposition step ([frozen optimizer](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/optimer_PerturabtionType/gam.py:122)). Similar probe geometry is not proof of the same numerical algorithm.

In that GAM optimizer, `grad_rho` is the normalized-gradient probe displacement ([line 60](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/optimer_PerturabtionType/gam.py:60)) and `grad_norm_rho` is the gradient-difference ascent displacement ([line 92](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/optimer_PerturabtionType/gam.py:92)). `gam_grad_gamma` is a gradient-decomposition coefficient; it is **not** the Fisher damping beta or the EWC importance decay gamma. The three GAM beta coefficients likewise are **not** Fisher damping.

No trajectory checkpoint averaging is applied in the selected fine-grained GAM runs. In the new, frozen 5Datasets run, `in_task_weight_merge=false`, `approximate_fisher_in_task=false`, `ewc_fisher_damping_beta=0`, and `ewc_normalize_accumulated_fisher=false` are explicit in the JSON configuration. The task-end Fisher is evaluated on the trained endpoint model and then accumulated as `old * ewc_gamma + new` ([frozen task-end code](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:309)). An endpoint-only retained set could be described mathematically as `{final checkpoint}`, but must not be described as averaging multiple retained checkpoints.

Older Aircraft/Cars/CUB/Flowers logs do not explicitly record the later `fisher_mode`, accumulated-normalization, damping, or RNG-control switches. Their documented normalized-Fisher settings can be reported, but the frozen 5Datasets configuration must not be silently projected backward onto all historical runs.

## Reporting recommendation

For this result table, use the actual name GAM-EWCLoRA and the archived GAM configuration keys. Report the missing Flowers settings as unrecorded until a run-bound historical code snapshot is recovered. If reporting the manuscript's formal FS-EWC with checkpoint averaging, identify a different completed run and its matching parameters/result; renaming this row alone would not establish implementation equivalence.
