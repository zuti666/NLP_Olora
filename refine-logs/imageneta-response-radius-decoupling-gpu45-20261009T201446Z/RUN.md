# GPU45 radius decoupling reproducibility

Execution directory: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z`. Read EXPERIMENT_PLAN.md for formulas and settings. Existing environment is warm-reused unchanged: Pilot_new_local Python3.11/torch2.4.1+cu121; canonical environment spec SHA256 4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8. No dependency installations. Frozen source has190 files. environment_spec.json is byte-identical to the prior run; its historical resource metadata is overridden by suite.json allowed_gpus=[4,5],max_parallel=2 and executable CL10 configurations.

CPU validation (no GPU allocations), run verbatim:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/trainer:/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest test_response_step_cpu test_independent_response_step_cpu test_response_lifecycle_cpu test_independent_radius_driver_cpu test_queue_cpu
```

CLI validation, with CUDA_VISIBLE_DEVICES empty:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/trainer/response_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/configs/formal_cl10_fixed_both_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/outputs/formal_cl10_fixed_both_s2 --dry-run
```

Queue dry-run must run on actual host cudahpc40.cvc.uab.es in PID namespace pid:[4026531836], using host execution escalation in a sandboxed Codex session. It creates no workers:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/suite.json --dry-run
```

Real queue command (already authorized by user; only the primary agent launches it once, fresh doc reviewer MUST NOT execute it):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-radius-decoupling-gpu45-20261009T201446Z/suite.json
```

Use one detached host controller, exclusive runtime/controller.lock, unique output directories, shared `/tmp/olora_gamfs_gpu_locks` index+UUID leases inherited through worker/trainer. Check host PID/starttime identity; skip absent or busy GPU, do not preempt. Every leased worker runs a seeded strict FP32 CUDA kernel witness before model execution; formal jobs wait for four accepted2-task bounded smokes plus exact-pairing gate. TMPDIR/TMP/TEMP are short `/tmp/olora_r45_20261009T201446Z` to avoid multiprocessing Unix socket path failures.

RUN_PINS.json freezes executable code/configs/docs/static gates; source_manifest.json freezes source. CPU_IMPLEMENTATION_ACCEPTED.json and INDEPENDENT_IMPLEMENTATION_REVIEW.json are prerequisites. The doc reviewer writes gates/FRESH_DOC_CHECK.json/md without modifying pinned files or launching GPU work. Runtime/witness/receipts/RESULTS.md are mutable outputs, not input pins. The original full Current-G2 baseline is imported only with pinned sources/checkpoints/complete receipts. Smoke metrics are excluded.

Live progress: runtime/queue_state.json and runtime/jobs/<job>/console.log; raw output outputs/<job>/step_trace.jsonl, tasks.json, task_matrix.json, final_summary.json. collect_results.py writes RESULTS.md/CSV/JSON from accepted full training only. Exact formal commands are suite.jobs[*].command; smoke caps never carry to formal jobs.
