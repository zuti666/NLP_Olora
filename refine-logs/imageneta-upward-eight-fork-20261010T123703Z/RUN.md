# Reproduce the registered experiment

Execution host: cudahpc40.cvc.uab.es, PID namespace pid:[4026531836]. Run in the actual host namespace, not the restricted inspection sandbox. No SSH/package installation is needed. Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`; torch2.4.1+cu121. environment_spec.json pins the reused environment (fingerprint4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8). Source is read-only and pinned by source_manifest.json. The complete settings and questions are in EXPERIMENT_PLAN.md; suite.json is the executable66-job registry.

From this campaign directory, run the exact CPU tests:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/test_upward_residual_cpu.py
/home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/test_upward_local_audit_cpu.py
```

Each trainer CLI accepts --dry-run for CPU configuration validation. Example:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/upward_fork_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config configs/formal_task1_g2_s2.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-upward-eight-fork-20261010T123703Z/outputs/formal_task1_g2_s2 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/upward_single_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config configs/formal_single200_g2_s2.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-upward-eight-fork-20261010T123703Z/outputs/formal_single200_g2_s2 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/upward_local_audit.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --suite suite.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-upward-eight-fork-20261010T123703Z/outputs/formal_local_upward_s2 --seed 2 --dry-run
```

A fresh implementation/document review must pass before queue execution. CPU_IMPLEMENTATION_ACCEPTED.json, INDEPENDENT_IMPLEMENTATION_REVIEW.json and FRESH_DOC_CHECK.json hold real evidence, not placeholder approvals. RUN_PINS.json freezes all execution code/configs/manifests/docs. The scheduler dry-run verifies these and all output paths in the actual host namespace:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json --dry-run
```

For durable launch use start_new_session=True and append no unrelated environment/package changes. Equivalent manual command, only when no controller is live:

```bash
nohup /home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json > runtime/controller.log 2>&1 < /dev/null &
```

The queue protects GPU0/1 and admits only idle GPUs2-6 under physical+UUID leases. Each worker runs trainer/cuda_witness.py under the exact GPU UUID/env then its registered command. Training smokes get1epoch/2batches; full jobs get50epochs. Workers hold leases to exit, persist starttime-based PID identities and successful/failed exit records. Acceptance recomputes FAA/AAA/BWT, validates forbidden-operation counters, trainable scope, input/LR budgets, probes, snapshots, parent hashes and complete curves. All formal training depends on16smokes + pairing + offline smoke passing. Completion dependencies require accepted success, not simply process exit.

Inspect `runtime/queue_state.json`, `runtime/controller.log`, `runtime/jobs/JOB/console.log`, each output `step_trace.jsonl` and `final_summary.json`. `RESULTS.md`/CSV include accepted formal numbers only. Create `runtime/PAUSE` to pause future admissions; remove it to resume. Do not kill unrelated workloads. Do not edit frozen pins/code or change batch size/radii to recover a failed job: review and register a separately named revision if needed.

Bulk output root is explicitly fixed in suite.json to the exclusive cache/tmp campaign directory. `outputs` is a convenience symlink. The scheduler rejects paths outside that root and nonempty output reuse. Complete snapshots remain complete; old parent and other campaign artifacts remain read-only.
