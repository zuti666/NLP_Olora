# Fresh documentation check

Status: **passed**. Checked at 2026-10-09 20:27:23 UTC. Fresh reviewer: `fresh_context_agent /root/radius45_fresh_doc_check`.

Read `RUN.md` and the project `.aris/compute/local.md` ledger. Executed the three documented validation commands verbatim from `/data/140-1/users/liying/O-LoRA` using the existing `Pilot_new_local` Python, with no dependency installations. All return codes were 0; no invocation divergence.

- CPU unittest: 65 tests passed in 4.808 seconds with `CUDA_VISIBLE_DEVICES=''`.
- Trainer dry-run: CurrentG2-FixedBoth, CL10, seed2, 50 epochs, fixed global L2 inner and outer radii .05; 3 gradient reads and 1 update per normal step.
- Host queue dry-run: 8 jobs, 3 formal jobs, 4 smokes; only physical GPU4/5, maximum 2 parallel. Executed with host escalation. Successful host assertions checked `cudahpc40.cvc.uab.es` and `pid:[4026531836]` before returning without workers.
- All 45 `RUN_PINS.json` input hashes and all 190 source hashes match; source manifest matches suite hash. CPU implementation and independent implementation review prerequisites passed.

GPU execution remains unvalidated by this receipt. CUDA witnesses, four actual bounded two-task smokes and exact pairing gate are required before the three formal trainings. This reviewer executed no real queue or GPU training and changed no pinned inputs.

Exact commands, output and return codes:

### cpu_unittest

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/trainer:/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest test_response_step_cpu test_independent_response_step_cpu test_response_lifecycle_cpu test_independent_radius_driver_cpu test_queue_cpu
```

Return code: 0.

```text
.................................................................
----------------------------------------------------------------------
Ran 65 tests in 4.808s

OK
```

### trainer_dry_run

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/trainer/response_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/configs/formal_cl10_fixed_both_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/outputs/formal_cl10_fixed_both_s2 --dry-run
```

Return code: 0.

```text
{"status": "valid", "protocol": "current_g2_independent_inner_outer_exact_restore_rng_replay_v1", "response_method": "current_g2", "response_layout": "cl10", "seed": 2, "gradients_per_normal_step": 3, "updates_per_step": 1, "diagnostic_forwards_per_step": 1, "history_weight": 0.0, "history_source": null, "prediction_distance": "fixed_global_L2_inner_probe_radius", "response_radius_mode": "independent_inner_outer_v1", "gam_radii_active": false, "init_cls": 20, "increment": 20, "full_partition_tasks": 10, "classes": 200, "tasks": 10, "epochs": 50, "response_probe_variant": "CurrentG2-FixedBoth", "radius_variant": "fixed_both", "probe_mode": "independent_inner_outer_v1", "inner_probe_radius_mode": "fixed_l2", "outer_probe_radius_mode": "fixed_l2", "inner_rho": 0.05, "outer_R": 0.05, "probe_radius_coordinates": "global L2 over all trainable LoRA and current classifier tensors", "normal_gradient_calls": 3}
```

### host_queue_dry_run

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/suite.json --dry-run
```

Return code: 0.

```text
{"status": "valid", "jobs": 8, "formal": 3, "smokes": 4, "allowed_gpus": [4, 5], "max_parallel": 2}
```

