# ImageNet-A g1+(g3−g2) execution

新增一个方法、两个正式训练点（r=.25/1），seed2/rank10/rho.05。两次隔离短测加配对验收先于正式训练。仅空闲GPU2–6，max2；GPU0/1排除。既有g0B队列保持运行。

Read-only queue verification (execute on real host cudahpc40.cvc.uab.es, in its host PID namespace):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/suite.json --dry-run
```

Expected: status valid, jobs5, formal2, smokes2, allowed_gpus [2,3,4,5,6], max_parallel2. RUN_PINS.json must exist first. This command does not claim GPUs or create training outputs.

After CPU and independent code/document checks, the root launches this same command without --dry-run as a detached supervisor. Shared index/UUID leases and stable idle checks admit GPU jobs; each worker first runs a seeded FP32 CUDA witness. Full training depends on actual model/data short tests and their pairing gate. No external notifications or package changes.

Create runtime/PAUSE to pause future admissions without interrupting active jobs. A job failure halts further admissions, preserves evidence, and never silently changes protocol or retries. Do not invoke a second supervisor for the same run.

Live state: runtime/queue_state.json. Logs: runtime/jobs/JOB/console.log. Accepted full results: summary.csv / RESULTS.md. See EXPERIMENT_PLAN.md for formula, metrics and protocol limitations.

