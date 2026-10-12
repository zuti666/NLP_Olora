# Paired GAM offline diagnostic

`paired_gam_diagnostic.py` reconstructs a single cumulative task-end LoRA adapter and the full saved FC from the frozen source. It never calls training or commits an optimizer update. Task numbers are zero based.

Required arguments: `--source`, `--base-config`, `--checkpoint-dir`, `--task`, `--output` (JSON path). Use `CUDA_VISIBLE_DEVICES=<one physical GPU>`; internally the default is `cuda:0`.

Smoke: `--task 0 --max-batches 1 --batch-size 128 --eval-per-task 1 --kl-match --save-vectors`. Formal: task 0/4/9, `--max-batches 4`, separately for g1/g3/sum source checkpoints. The same diagnostic seed and task generate identical augmented current-task train tensors across sources; hashes are emitted.

Use `TORCH_HOME=/home-local/liying/.cache/torch` and the existing `Pilot_new_local` Python environment. Model loading uses the same cached pretrained backbone and prohibits HuggingFace network fallback (`HF_HUB_OFFLINE=1`).

Outputs include:

- Real frozen GAM state g0–g3, five CE gradient passes including the clean cache; optional vector sidecar.
- Projection coefficient a without clipping, residual orthogonality, gradient norms/cosines, LoRA layer and FC norms.
- r=0/.1/.25/.5/1 probes: exact parameter distance, actual BA/FC changes and independent current-test-batch all-seen KL.
- Counterfactual g1/g3/sum/half-sum, a*g1, (1+a)*g1, both residual families and normalized direction interpolation; each effective BA+FC update is matched to the same raw g1 virtual eta=.02 update. This is a diagnostic step, not the nearly-zero task-end cosine LR.
- Independent current/old test losses, both all-seen and task-local CE. Old-task macro means average tasks equally; task0 has explicit NA.
- SAM matched to the actual g3 parameter probe distance, and optional KL-matched SAM using a radius grid plus up to ten bisections. No fallback is used for unmatched conditions.
- Full parameter, buffer, gradient, module-mode and Python/NumPy/torch CPU/CUDA RNG restoration checks. Nonzero stochastic dropout/drop-path or nonidentity LoRA scaling fails explicitly.

Successful evaluation emits `status="completed"`, finite and restoration checks, plus individual matching flags. A matching failure remains explicitly marked; it does not manufacture an equivalent update. An exception exits nonzero. Matched positive scale-only and g1 must reproduce the same parameter displacement; r=0 must reproduce g1 within a declared numerical tolerance.

`test_paired_gam_cpu.py` uses a tiny synthetic LoRA+FC model with the actual frozen GAM. It checks r=0 identity, restoration, six matched directions, and parameter/KL-matched SAM; it does not load a dataset or use a GPU. Its output is `cpu_test_report.json` beside the scripts.
