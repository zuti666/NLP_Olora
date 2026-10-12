# ImageNet-A C/D g1−+g2 local environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No package changes. Canonical environment_spec.json SHA256 remains4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8; provider ledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. The spec's old validation section is provenance, not certification of this new run.

A fresh independent agent executes this read-only invocation verbatim in the real host PID namespace, outside the filesystem/GPU sandbox:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-20261005T182936Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-20261005T182936Z/suite.json --dry-run
```

Expected: status valid, jobs9, formal4, smokes4, allowed_gpus [2,3,4,5,6], max_parallel2. RUN_PINS.json must exist first. This invocation checks immutable source/config/code paths without claiming GPUs or creating training outputs. Independent receipt: gates/DOC_CHECK.json.

Actual GPU allocation remains idle-only under shared index/UUID leases; each worker repeats trainer/cuda_witness.py before its model/data training. Witness: seededFP32 forward/backward,TF32off. Sandbox NVML/CUDA failure is not a hardware diagnosis. Four two-task/one-epoch/one-batch smokes plus input/Task0 initial-state pairing gate four formal trainings. Kernel/CPU checks do not certify model execution; actual GPU acceptance remains pending until observed.

Suite uses offline HF/Torch caches in /home-local/liying/.cache; PYTHONDONTWRITEBYTECODE=1,PYTHONHASHSEED=0,CUBLAS_WORKSPACE_CONFIG=:4096:8,OMP/MKL/OPENBLAS threads8. Each child sees its selected physical UUID as logicalcuda:0. Only physicalGPU2–6 eligible;0/1excluded.

Source is the unchanged190-file frozen snapshot listed in source_manifest.json. Data /data/140-1/datasets/imagenet-a, train5981/test1519, fixed10x20. ViT-B/16 augreg2_in21k_ft_in1k,LoRA rank10,seed2,rho=.05,r=.25/1 read from configs. Probes g0/g1+/g1−/g2; C usesq=g0−g1− and D usesq=g1−−g0. Both finalU=g1−+g2, restoredbase before oneSGD; no averaging/projection/HVP/g3/EWC/Fisher/fusion/final matching. R0 replay perprobe, finalRNG=R1 fromg0. Common training50epochs/task,b128,LR.04 then.02cosinefloor.1,momentum/WD0.

