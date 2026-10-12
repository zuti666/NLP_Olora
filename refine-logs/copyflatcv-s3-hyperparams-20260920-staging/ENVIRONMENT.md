# S3 reference hyperparameter campaign: local environment

Warm reuse of `Pilot_new_local` on `cudahpc40.cvc.uab.es`, RTX 6000 Ada 48 GiB.
Interpreter `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`.
Existing environment spec: `/data/140-1/users/liying/CopyFlatCV/.aris/compute/local-pilot-new-local-env-spec.json`, canonical SHA prefix `574b252e`. No installation or dependency changes.

Before witness, query physical GPU6; memory must be below500 MiB and utilization below5%. Then execute exactly:

```bash
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -c 'import torch, platform, socket, timm, torchvision; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); print("WITNESS", tuple((x@x).shape), torch.cuda.get_device_name()); print("VERSIONS", platform.python_version(), torch.__version__, torchvision.__version__, timm.__version__, socket.gethostname()); assert torch.isfinite(x@x).all()'
```

Expected `WITNESS (8, 8) NVIDIA RTX 6000 Ada Generation`, Python3.11.15, torch2.4.1+cu121, torchvision0.19.1+cu121, timm1.0.9. Filesystem sandbox hides GPUs; use existing escalation mechanism. Report literal output, spec hash, GPU availability, and deviations; do not fix commands silently.

Formal training uses a separately frozen copy of the September18 attempt2 source natively via `src.main`, without `runtime_v3/protocol.install`. Data `/data/140-1/datasets`; original batch-gradient Fisher, seed2, A enabled. All full runs have10 tasks and50 epochs/task. Only GPUs4/5/6 authorized. Each job has a unique output directory and log, and source/config SHA is checked before launch. June29 historical source identity is not fully guaranteed; a new center run is included and new curves use that center.
