# Run this frozen campaign

Read EXPERIMENT_PLAN.md and ENVIRONMENT.md. CPU_GATE, independent code/schema/queue review,12actualtrainerCLI dry-runs, frozenRUN_PINS and fresh documented invocation must pass before launch. The environment is warm-reused; seededGPU witness and real model smokes occur only aftercardclaim.

Read-only host check:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/suite.json --dry-run
```

Actual detached controller command (onlyonce; neverduplicate a livecontroller):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/suite.json
```

The launch helper storesPID plus/procstarttime and stdout runtime/queue.log. EligibilityonlyGPU0/1. It firstwaitsfor4priorC/Dformalreceipts, then6smokes+pairing, then6formals. SharedGPUlocks arecompatiblewithallcurrentqueues. runtime/PAUSE stopsfutureadmissionwithoutkillingworkers. Newoutputsneverreuseoldpaths. Failedjobsstopadmission, evidencepreserved.

Progress runtime/queue_state.json, perjob runtime/jobs/<job>/console.log and outputs/<job>/metrics.json and step_trace.jsonl. Final results acceptedonlyforfull10-task/500-epoch/2550-update runs with recheckedconfig/trainer/trace/matrixhashes and exit0. summary.csv/results.json/RESULTS.md are regeneratedonjobcompletion.
