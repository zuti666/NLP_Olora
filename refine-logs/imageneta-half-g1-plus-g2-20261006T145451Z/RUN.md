# Run frozen half(g1+g2) campaign

Read EXPERIMENT_PLAN.md and ENVIRONMENT.md. CPU_GATE, independent code/schema review,4actual CLI checks,190source hashes and warm env spec must pass; freeze RUN_PINS then fresh independent documented invocation before launch. Real GPU kernel/model witnesses occur only after card claim.

Read-only host check:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/suite.json --dry-run
```

Detached host controller (once; do not duplicate a live controller):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/suite.json
```

OnlyGPU0/1,maxparallel2, shared index/UUID leases, idle admission, no preemption. First waits for2currentg0sumformalaccepted receipts, then2smokes+pairing and2formals. runtime/PAUSE stops new admission without killing workers. Failures halt further admission; preserve logs and frozen artifacts.

Progress runtime/queue_state.json, perjob runtime/jobs/<id>/console.log, outputs/<job>/metrics.json and step_trace.jsonl. Final summary.csv/RESULTS.md accept only10tasks/500epochs/2550steps/exit0 and validated config/trainer/trace/matrix hashes. Previous campaigns untouched.
