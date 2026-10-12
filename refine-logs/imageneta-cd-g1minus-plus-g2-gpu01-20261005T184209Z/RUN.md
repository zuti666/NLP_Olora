# ImageNet-A C/D g1−+g2 on GPU0/1

用户明确指定物理GPU0/1，覆盖此campaign此前排除0/1的约束；其他campaign不变。四个正式训练(C/D×r=.25/1)，seed2/rank10/rho.05，最终U=g1−+g2不除2。四个实际GPU短测及配对验收先于正式训练。

Read-only verification on cudahpc40.cvc.uab.es in the real host PID namespace:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/suite.json --dry-run
```

Expected: status valid, jobs9, formal4, smokes4, allowed_gpus [0,1], max_parallel2. RUN_PINS.json must exist first. No GPU claim or training output from this invocation.

After independent GPU-delta/doc checks, root launches the same command without --dry-run as a detached supervisor. Shared index/UUID leases and stable idle checks; each worker repeats a seeded FP32 CUDA witness. Only physicalGPU0/1, max2, no preemption, no external notifications. Failures preserve evidence and halt new admissions. runtime/PAUSE pauses future work without interrupting workers. No duplicate supervisor.

Live state runtime/queue_state.json; logs runtime/jobs/JOB/console.log; accepted endpoints summary.csv/RESULTS.md. Model trainer is byte-identical to the prior queue. Source/config/docs frozen inRUN_PINS.json. SeeEXPERIMENT_PLAN.md andENVIRONMENT.md.
