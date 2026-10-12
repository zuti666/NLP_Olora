# Native finite-response / HVP audit

The script is read-only with respect to native checkpoints and frozen source.
It restores every temporary probe and commits no SGD update. Its role is to
check the local finite-difference interpretation, not to certify long-horizon
Hessian mediation. No GPU job has been started as part of implementation.

```sh
python -B diagnostics/curvature_audit.py \
  --source FROZEN_SOURCE --config MATCHED_US_CONFIG.json \
  --checkpoint-dir TRAINING_OUTPUT/checkpoints/task_004 \
  --task 4 --output FRESH_OUTPUT/curvature_audit.json \
  --rho .05 --radius .25 --epsilons .05,.025,.0125 \
  --max-batches 4 --batch-size 128 --hvp-backend auto --device cuda:0
```

Frozen planned scope is matched US and matched UFull, each at tasks0/4/9.
Use4/4/3 batches respectively: task9 has only378 native training examples
(128/128/122), and task4's last batch has42. No repeat/padding is added.
Data seed is `training_seed*100000 + 70000 + task`, with isolated restoration
during each diagnostic, yielding the same batches across the two arm anchors.

The input must be a new component campaign checkpoint with `manifest.json`
containing task/seed/component rule/scale/source-GAM hash, class ranges,
model parameter SHA256 and native-file hashes. It restores via
`load_task_checkpoint(directory, task, dm)` and verifies the restored full
parameter hash and LoRA metadata. Complete input file trees are checked again
afterward. The learner's unused write path points to the new diagnostic output
directory, never the native checkpoint.

Each audit calls the frozen learner's real batch function and frozen GAM with
SGD learning rate0. It captures the four true closure states, including RNG,
buffers, modes, batchnorm controls, trainable tensors and gradients. Independent
checks reproduce native multiplication/addition/subtraction in the same
order; they retain the native finite-precision roundtrip. The extra clean
gradient pass plus four closures must produce exactly five forwards.

At closure0 and closure2, HVP is computed with a **detached** native-normalized
g0 or g2 direction. Same-point CE and first-gradient agreement are hard gates.
Native finite responses and forward/central finite differences at all three
epsilons are compared to the HVP with cosine, relative L2 error, norm ratio,
and maximum absolute error. Low alignment and poor epsilon trends are valid
results. Zero-norm cosine/ratio fields are explicit structural nulls.

With `--hvp-backend auto`, only an explicit unsupported-double-backward error
permits a local math-SDPA retry. The math-backend CE and first gradient must
agree with the native probe within the declared checks. OOM, NaN, a wrong
state, or mismatch is not masked. If both native and math backends explicitly
lack double backward, finite-difference results can still complete with
`hvp_all_available=false`; this does not support an autograd-HVP claim.
Native source and training attention dispatch are never modified.

Output schema:

- Top level: `schema_version`, `status`, `task`, `seed`, checkpoint manifest and
  hashes, source Python hashes, restored-model hash, script/config hashes,
  `batches`, and completion `checkpoint_unchanged`, `source_unchanged`,
  `hvp_all_available`.
- `batches[]`: batch/sample/input identities, `native.closure_count=4`,
  `native.training_ce_gradient_passes=5`, `native.training_forward_count=5`,
  three true `native.native_point_checks`, `deltaG_vs_deltaS`, and `restore`.
- `responses.original|outer`: native closure index, native/replayed CE,
  same-point gradient comparison, `hvp.status=available|unsupported`, backend
  and attempts, `native_finite_response_vs_hvp`, and `finite_difference_scan`.
- A successful diagnostic requires all restoration/numerical/protocol gates,
  not a minimum scientific cosine. Batches and checkpoints are not seeds.

CPU tests:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B \
  diagnostics/test_curvature_audit_cpu.py --source FROZEN_SOURCE \
  --formation-trainer FORMATION_STAGING/trainer \
  --output diagnostics/CPU_AUDIT.json
```

Tests compare autograd HVP to an analytic logistic Hessian, demonstrate that
the live parameter-dependent direction is detached, check decreasing FD
errors, exercise actual frozen GAM/learner probes, reject an injected wrong
outer probe, wrong task slice, nonfinite gradient, stochastic layer, altered
checkpoint and added source file. Unsupported-backend fallback is tested;
OOM must propagate. The analytic relative HVP error is approximately2e-16.
