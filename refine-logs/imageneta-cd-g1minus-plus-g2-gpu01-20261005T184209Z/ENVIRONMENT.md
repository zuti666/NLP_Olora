# ImageNet-A C/D g1−+g2 GPU0/1 environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No package changes. Canonical environment_spec.json SHA256 remains4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8. Provider ledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. The spec's old resource_policy/validation is historical provenance; current user authorizes onlyGPU0/1 in suite.json. Runtime common.py andrun_queue.py enforce this new authorization.

A fresh independent agent runs this documented read-only invocation verbatim outside the sandbox in the real host PID namespace:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/suite.json --dry-run
```

Expected: status valid, jobs9, formal4, smokes4, allowed_gpus [0,1], max_parallel2. RUN_PINS.json must exist first. It verifies pinned source/code/configs without claiming GPUs or creating training outputs. Fresh receipt gates/DOC_CHECK.json.

All model training uses the supervisor with shared index/UUID leases and idle-only admission, never GPU2–6. Each GPU child is bound by UUID to logicalcuda:0 and repeats trainer/cuda_witness.py before its model/data smoke. Witness seededFP32 forward/backward, TF32off; sandbox NVML/CUDA failure does not diagnose hardware. Four2-task/1-epoch/1-batch smokes plus realized-input/Task0-entry-state pairing gate four50-epoch/task formals. No model acceptance claim before actual smokes.

OfflineHF/Torch caches in /home-local/liying/.cache; PYTHONDONTWRITEBYTECODE1,PYTHONHASHSEED0,CUBLAS_WORKSPACE_CONFIG=:4096:8,OMP/MKL/OPENBLASthreads8. Source190frozenfiles andtrainer9py unchanged. DatasetImageNet-A train5981/test1519/fixed10x20; ViT-B/16 augreg2_in21k_ft_in1k,rank10,seed2,rho.05,r.25/1. Probesg0/g1+/g1−/g2,Cq=g0−g1−,Dq=g1−−g0; finalU=g1−+g2,4CE/1naturalSGD,zero g3/projection/Fisher/EWC/merge/HVP/finalmatching. R0replay/allstaterestore/finalR1fromg0. Batch128,50epochs/task,LR.04then.02cosinefloor.1,momentum/WD0.

Original pending queue must be paused/drained/stopped with no formal begun before this new supervisor starts. Original short-test evidence is preserved; newGPU0/1smokes are isolated. No other campaign is paused or rerouted.
