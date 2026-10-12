# S3 mechanism ablation campaign: local environment

Warm reuse of `Pilot_new_local` on `cudahpc40.cvc.uab.es`, RTX 6000 Ada 48 GiB.
Interpreter `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`.
Existing environment spec: `/data/140-1/users/liying/CopyFlatCV/.aris/compute/local-pilot-new-local-env-spec.json`, canonical SHA prefix `574b252e`. No installation or dependency changes.

Before witness, query physical GPU6; memory must be below500 MiB and utilization below5%. Then execute exactly:

```bash
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -c 'import torch, platform, socket, timm, torchvision; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); print("WITNESS", tuple((x@x).shape), torch.cuda.get_device_name()); print("VERSIONS", platform.python_version(), torch.__version__, torchvision.__version__, timm.__version__, socket.gethostname()); assert torch.isfinite(x@x).all()'
```

Expected `WITNESS (8, 8) NVIDIA RTX 6000 Ada Generation`, Python3.11.15, torch2.4.1+cu121, torchvision0.19.1+cu121, timm1.0.9. Filesystem sandbox hides GPUs; use existing escalation mechanism. Report literal output, spec hash, GPU availability, and deviations; do not fix commands silently.

Formal training uses the 190-file frozen September18 attempt2 source natively via `src.main`, plus a separately hashed opt-in `utils/s3_mechanism_controls.py` observer/control. The observer saves factors without forwards/Fisher calls/RNG consumption. Endpoint-Fisher ablation relocates the one existing Fisher pass; all other variants retain native Fisher placement. There is no runtime_v3/sample-Fisher protocol installation. Data `/data/140-1/datasets`; original batch-gradient Fisher. Full runs have 10 tasks and 50 epochs/task; integration smoke runs have 2 tasks and 2 epochs/task and are never reported as formal accuracy. Initial authorization covered GPUs4/5/6; the user subsequently explicitly authorized GPUs0/1. This campaign now schedules physical GPUs0/1 only (max_parallel=2) because4/5/6 are occupied by another campaign. No dependency or environment changes. Every run uses a unique output directory; source and config hashes are checked. Prior seed2 center is reused after actual native same-state observer equivalence and per-boundary RNG/factor checks; separate GPU runs are not claimed bitwise reproducible.

The independent environment witness used then-idle GPU6 before reassignment. No environment change accompanied the reassignment. Actual smoke/full jobs verify their assigned GPU UUID and expose exactly one physical GPU.


Observer verification update: attempt1 cross-run equality failed before the first observer boundary (epoch1 snapshot CE already differed). Preserve that reproducibility failure; its source is unresolved, and do not attribute it to the observer. NATIVE_BOUNDARY_WITNESS.json instead tests six actual frozen native boundary paths at identical saved states: all50 factor/classifier parameters, gradients, buffers, modes and native metrics exactlymatch; RNGunchanged,0extra forward/Fisher. Formal eligibility requires this same-state witness and every real boundary invariant. Separate GPU-run matrix/tensor comparisons remain fully reported diagnostics, not a claim of bitwise reproducibility. This does not change training kernels/seeding/backend settings or any performance acceptance test.
