# ImageNet-A g0+B experiment

[Plan](EXPERIMENT_PLAN.md), [environment](ENVIRONMENT.md), [historical source audit](historical_inventory.json).

New jobs: four isolated GPU smokes, one input/initial-state pairing acceptance, two full g0B trainings. Only idle GPUs2–6; max2; GPUs0/1 excluded. No external notifications. Every job needs a seeded CUDA witness, verified source/config hashes, and a successful completion receipt.

Read-only queue verification:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0B-no-g3-20261005T164542Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0B-no-g3-20261005T164542Z/suite.json --dry-run
```

Root-managed detached launch invokes the same command without --dry-run after CPU checks and independent documentation review. runtime/PAUSE pauses future admission without interrupting active workers. A failed job stops further admission and preserves logs; it never relaxes scientific checks or automatically reruns a failed experiment.

Live state: runtime/queue_state.json; job logs: runtime/jobs/JOB/console.log; accepted final results: summary.csv and RESULTS.md. Check task coverage rather than process existence alone. Fixed code/config/document hashes live in RUN_PINS.json; dynamic outputs and audit receipts are not included.
