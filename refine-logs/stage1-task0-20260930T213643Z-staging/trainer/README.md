# Stage 1 independent Task 0 trainer

Run `task0_trainer.py --source FROZEN_SOURCE --config ARM.json --output-dir RUN_DIR` with one GPU exposed through `CUDA_VISIBLE_DEVICES`. The internal device is cuda:0. `--max-epochs 1 --max-batches 1` is smoke-only and is explicitly marked in the summary.

Config field `stage1_direction` accepts `ce`, `s`, `g`, `sum`, `mean`, `true_sam`. `gam_grad_norm_rho` controls the G radius sweep. This wrapper fixes seed2, ImageNet-A first20 classes, Q/V LoRA rank10+FC, batch128, Task0 initial LR .04, SGD momentum0/weight_decay0, and the native cosine scheduler (default eta_min=.004, not zero). The effective config is saved.

The original `incremental_train`, `_train`, and `_run_regularized_train` methods run unchanged. The main CE/S/G/sum/mean arms use the original clean-gradient cache plus four real GAM closures. CE only replaces the final decomposed gradient with g0; the other arms use native betas. True SAM uses the repository's real two-gradient SAM optimizer and an explicit standard two-step batch method. No after_task/Fisher estimator is invoked.

All ten arms use the same per-epoch data generator seed (`2*100000+epoch`) and independent augmentation RNG. Worker0 and the main-process loader path are isolated from optimizer RNG consumption. First-batch hashes, sampler-order hashes and epoch0 model/frozen-backbone hashes support cross-arm checks. Evaluation/checkpoint hooks restore RNG, model modes and buffers.

Outputs:

- `final_summary.json`: status=completed, completed_epochs, smoke flag, exactly Task0/20class checks, finite checks and checkpoint existence checks.
- `metrics.json`: each epoch's weighted train CE/sample accuracy, test CE/sample accuracy, 20class macro accuracy/counts, LR before/after scheduler, training/eval time, throughput and peak GPU allocation.
- `geometry.json`: first batch of each epoch; native g0..g3 norms/cosines/projection residual, layer norms, actual probe BA+FC displacement, and actual update. Actual SGD is checked from the post-unperturb parameters against a same-dtype SGD prediction. Native probe roundtrip drift and total-step relative discrepancy are recorded separately and never used to falsely reject a vanishing-LR step.
- `checkpoints/epoch_000`, `epoch_010`, ..., `epoch_050`: native-compatible cumulative `lora_w_a_0.pt`, `lora_w_b_0.pt`, `lora_meta_0.json`, `fc_state_0.pt`; plus optimizer/base-optimizer/scheduler/RNG in `training_state.pt` and model hash manifest. A smoke also saves its final epoch. The checkpoint epoch50 is saved before the native task-end saver increments backbone.task_id.

True SAM's g2/g3 are **not training probes**. Its first batch each epoch runs an additional isolated five-pass GAM diagnostic, fully restoring state/RNG, to obtain paired g0..g3 and validate the actual SAM update against native g1. The training-gradient count remains exactly two per batch; diagnostic passes have a separate counter and are included in reported wall time.

`test_task0_cpu.py` runs all six directions through an unchanged AST copy of the native epoch loop using a tiny synthetic LoRA model and the real frozen optimizers. It checks epoch/checkpoint hooks, common initial states and batches, and float32 low/zero-LR behavior. See `cpu_test_report.json`. This synthetic check complements the required GPU smoke; it does not establish dataset/model initialization correctness by itself.
