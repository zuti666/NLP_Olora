# Run the frozen multi-seed supplement

Read EXPERIMENT_PLAN.md and ENVIRONMENT.md. Read-only dry-run (execute directly on actual host, outside sandbox):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/suite.json --dry-run
```

Launch exactly one detached controller after independent review, CPU_GATE, immutable source/config hashes, RUN_PINS and fresh document check pass:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/suite.json
```

GPU2–6 pool, max2, idle allocation only. GPU0/1 are reserved to their existing campaigns. Six actual model smokes plus same-seed pairing gate precede all six new formals. Old Mean12 seed2 remains in its own live queue, is imported only after successful accepted endpoint, and is never re-launched here. Runtime PAUSE stops future admission without killing live workers; any failure stops automatic new admission.

Results: summary.csv per-seed, THREE_SEED_SUMMARY.csv complete n3 only, results.json same-seed differences/SD/wins, RESULTS.md. Check runtime/controller.json with PID plus proc_starttime; watch per-worker console.log/outputs/step_trace.jsonl. All partial endpoints excluded from final aggregation.
