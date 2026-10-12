# Native seed 0/1 runner

This directory supports N01–N14 only. It uses the frozen shared 190-file native source; all new code is observational except the previously declared final direction selection and Gmatch learning-rate scale.

The four direct native arms N01/N02/N04/N10 retain their historical beta/gamma configuration, invoke the original native `gradient_decompose`, and do not copy or replace its resulting gradient. Their new `native_beta` selector denotes observation only. A separate read-only FP32 oracle checks the original expression in its exact reduction order. CPU tests compare the entire four-step trajectory, all probe gradients, mechanism records, subsequent batch RNG and inputs against the completely unwrapped method.

N03/N05–N09/N11–N14 copy the audited historical LR helper. Their independent formula/oracle, effective BA+FC displacement and fail-closed matching tests are retained. `residual_selection.py` and `effective_step.py` are byte-identical to the previous implementation. `step_reference.py` now enforces the requested candidate/reference seed (0, 1, or historical CPU fixture 2) instead of unconditionally requiring seed 2.

Smoke JSON keeps 50 epochs and 10 classes/tasks metadata unchanged, and adds `max_train_tasks=2`. The per-instance `_run_regularized_train` wrapper passes `epochs=1` to the native loop while leaving the original cosine scheduler at `T_max=50`. The loader, gradient closures, random-number lifecycle, task evaluation and Fisher skip path remain native. Each smoke runs 18 updates / 2256 sample exposures. Formal jobs have no effective epoch cap and run 2550 updates / 299050 exposures.

Commands (from any working directory):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/native/run.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/suite.json --job smoke_N01_r1_k10_s0 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/native/test_seed01_native.py
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/native/validate.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/suite.json --smokes --output /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/audits/native_smoke_gate.json
```

The root worker owns GPU admission, console redirection, process lifecycle and `completed.json`. Do not invoke a GPU job directly outside that worker. The runner writes `outputs/<id>/final_summary.json` and a private `runtime/jobs/<id>/native_completed.json`; the scientific validator returns `direction_proof` for the root completion receipt. Gmatch requires the same-phase and same-seed N03 root completion receipt, with matching config/suite/run-pins hashes and intact direction trace.

The native smoke gate pairs all 14 conditions separately within seed, including input/target hashes, initial parameters, RNG before/after each update, parameter scope and task budget. It never compares seed 0 initialization against seed 1, and makes no cross-GPU bitwise full-training trajectory claim.
