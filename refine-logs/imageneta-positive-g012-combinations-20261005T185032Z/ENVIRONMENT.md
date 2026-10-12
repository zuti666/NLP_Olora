# ImageNet-A positive g012 combinations: local environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No packages changed. Canonical environment_spec.json SHA256=4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8. Provider ledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. Historical resource_policy/validation in the unchanged environment spec is provenance; current suite/runtime allow onlyGPU0/1.

A fresh independent agent executes this exact read-only invocation in the real host PID namespace outside the GPU/filesystem sandbox:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/suite.json --dry-run
```

Expected status valid, jobs13, formal6, smokes6, allowed_gpus [0,1], max_parallel2. RUN_PINS.json must exist first. The invocation verifies frozen source190files, runtimecode/configs and paths; it claims no GPU and starts no training. Fresh receipt gates/DOC_CHECK.json.

Every GPU worker must hold shared index+UUID leases, bind chosenUUID as logicalcuda:0, and run trainer/cuda_witness.py before actual model/data smoke. Witness seededFP32 forward/backward, TF32off. Sandbox NVML failure is not hardware diagnosis. Do not declare GPU model acceptance before6actual2-task/1-epoch/1-batch smokes and pairing pass. Previous4C/Dformals must all finish and their receipts/hashes pass before any newGPUwork; supervisor waiting is expectedwhilethosejobsrun.

OfflineHF/Torch caches /home-local/liying/.cache, PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL/OPENBLAS threads8. FixedImageNet-A train5981/test1519,10x20class/fixedorder,ViT-B/16 augreg2_in21k_ft_in1k rank10 seed2. rho.05,r.25/1,positivePq=g1+−g0;g0/g1+/g2/g3fourCE per1naturalSGD. g3computedunusedfinal; onlymode2 hastwoglobalg0projections. R0replay/allstateexactrestore/finalR1;noFisher/EWC/fusion/replay/HVP/finalmatching. Batch128,50epochs/task,LR.04then.02cosinefloor.1,momentum/WD0.
