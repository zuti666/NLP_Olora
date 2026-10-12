# ImageNet-A half(g1+g2): warm local environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No package changes. Canonical environment_spec.json SHA2564df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8; provider ledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. Old spec policy/validation is provenance; runtime/suite allow onlyGPU0/1. Source190 files unchanged.

After RUN_PINS exists, a fresh independent agent executes this exact read-only documented invocation in real host PID namespace, outside filesystem/GPU sandbox:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/suite.json --dry-run
```

Expected status valid, jobs5, formal2, smokes2, allowed_gpus[0,1], max_parallel2. No GPU claim or model training. Checks frozen code/source/config paths. Receipt gates/DOC_CHECK.json. CPU formulas, independent code/schema review and4actual trainer CLI checks must pass before freeze.

Controller waits for current two g0+g1+g2 formals to complete with hash-verified accepted receipts before any new admission. Then per-worker shared index/UUID leases, physicalUUID CVD to logicalcuda:0, idle-only memory<500MiB/util<5%, seededFP32 CUDA forward/backward witness TF32off, then2actual two-task/one-epoch/one-batch model smokes + input/Task0 pairing, then2formals. PendingGPUacceptance whilewaiting is expected, not failure. Do not declare actual model acceptance fromCPU/import only.

Same offlineHF/Torch cache /home-local/liying/.cache; OMP/MKL/OPENBLAS8threads, PYTHONDONTWRITEBYTECODE1, PYTHONHASHSEED0, CUBLAS_WORKSPACE_CONFIG=:4096:8. ImageNetAtrain5981/test1519 fixed10x20/rank10/seed2/rho.05/r.25,1. Nominal SGD LR.04then.02cosinefloor.1,momentum/WD0 unchanged. Positiveg0/g1+/g2/g3fourCE, g3computedunused; finalU=(g1_plus+g2)/2, direction_factor.5, naturalSGD scale=nominalLR (not halved again). No Fisher/EWC/merge/HVP/oppositeg1/projection/finalmatching. Fullmodelstate/R0replay/finalR1fromg0.
