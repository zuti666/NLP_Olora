# ImageNet-A g0+g1+g2 warm local environment

Warmreuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No packageschanged. Canonical environment_spec.json SHA2564df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8; providerledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. Oldspecresourcepolicy/validation isprovenance, current suite/common/run_queue hardallowGPU0/1only. Frozen source190files unchanged.

AfterRUN_PINSexists, a fresh independent agent mustexecute this documented READ-ONLY invocation VERBATIM inrealhostPIDnamespace (outsideGPU/filesystemsandbox):

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0-plus-g1-plus-g2-20261006T144354Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0-plus-g1-plus-g2-20261006T144354Z/suite.json --dry-run
```

Expected statusvalid,jobs5,formal2,smokes2,allowed_gpus[0,1],max_parallel2. NoGPUclaim/no trainingoutputs; checksfrozenruntime/source/configpaths. Freshreceipt gates/DOC_CHECK.json. ExactCPUformulatests, independentcode/schema review and4actualtrainerCLIchecks precedefreeze.

Actualqueueleasesphysicalindex+UUID, bindsCVD=UUIDlogicalcuda:0, idle-onlymemory<500MiB/util<5%. Everyallocatedworker runs trainer/cuda_witness.py (seededfiniteFP32CUDAforward/backward,TF32off) beforemodel/data smoke. SandboxNVMLfailureisnotahardwarediagnosis. Actualmodelacceptance requiresboth2task/1epoch/1batchsmokes+input/Task0pairing. Onlythentwofull10task/50epochspertaskformals. Noenvready/modelacceptanceclaimfromCPUorimportalone.

OfflineHF/Torchcaches /home-local/liying/.cache, OMP/MKL/OPENBLAS8threads,PYTHONDONTWRITEBYTECODE1,PYTHONHASHSEED0,CUBLAS_WORKSPACE_CONFIG=:4096:8. ImageNetAtrain5981/test1519,fixed200classes10x20;rank10seed2rho.05r.25/1;SGDLR.04then.02cosinefloor.1,momentum/WD0,b128. Positiveg0/g1+/g2/g3fourCE+1naturalSGD,finalU=g0+g1_plus+g2;g3computedunused,noaveraging/matching/projection/Fisher/EWC/merge/HVP/oppositeg1. CompleteR0statecapture/replay,finalR1fromg0. Previouscampaigncompleteandleftunchanged.
