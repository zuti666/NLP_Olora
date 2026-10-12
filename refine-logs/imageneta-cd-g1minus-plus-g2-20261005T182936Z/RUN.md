# ImageNet-A C/D g1−+g2 execution

用户已确认g1=g1−。新增C/D×r=.25/1四次正式训练；AC探针最后g2=C/g2、AC探针最后sum=C/sum，合并重复。seed2/rank10/rho.05。四次GPU短测和一个输入/初始状态配对验收先于四次正式训练。

Read-only verification in the real host PID namespace on cudahpc40.cvc.uab.es:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-20261005T182936Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-20261005T182936Z/suite.json --dry-run
```

Expected: status valid, jobs9, formal4, smokes4, allowed_gpus [2,3,4,5,6], max_parallel2. RUN_PINS.json must exist first. This command does not claim GPUs or create training outputs.

Root launches the same command without --dry-run as a detached supervisor after CPU and independent code/document checks. Each GPU job takes shared index/UUID leases and repeats a seeded FP32 CUDA witness. Actual model/data smoke and pairing acceptance gate formal training. No package changes or external notifications.

Only idleGPU2–6;GPU0/1excluded;max2. runtime/PAUSE pauses future admissions while active jobs continue. Failure halts new admissions and preserves evidence; no altered protocol or silent retries. Do not start duplicate supervisors.

Live state runtime/queue_state.json; logs runtime/jobs/JOB/console.log; accepted final endpoints summary.csv and RESULTS.md. Historical controls are linked in historical_inventory.json. See EXPERIMENT_PLAN.md for exact signs, update and metric definitions.

