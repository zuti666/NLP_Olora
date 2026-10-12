# ImageNet-A g0+B local environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No package changes. The copied environment_spec.json has the unchanged canonical SHA256 4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8; the provider ledger is /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. The spec's historical validation section is provenance, not certification of this new run.

Execute the following documented read-only invocation verbatim in the real host PID namespace (outside the filesystem/GPU sandbox). It verifies pinned code/source/config paths and does not claim GPUs or create training outputs:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0B-no-g3-20261005T164542Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g0B-no-g3-20261005T164542Z/suite.json --dry-run
```

Expected output: status valid, jobs7, formal2, smokes4, allowed_gpus [2,3,4,5,6], max_parallel2. The immutable RUN_PINS.json must exist first.

Training always goes through this run's supervisor and shared index/UUID leases. Workers repeat trainer/cuda_witness.py before the actual model/data smoke; CUDA access in the sandbox is not a hardware failure. The kernel witness is seeded FP32 forward/backward, TF32 disabled. Four two-task/one-epoch/one-batch smokes plus a pairing gate precede the two full natural g0B trainings. No HVP or g3, no EWC/Fisher, no weight fusion.

The supervisor uses suite.json environment settings: offline HF/Torch caches at /home-local/liying/.cache; PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL/OPENBLAS threads8. Each GPU child sees its chosen UUID as logical cuda:0. Only physical GPUs2–6 are eligible; GPUs0/1 remain excluded.

Source is the unchanged 190-file snapshot listed in source_manifest.json. Data remains /data/140-1/datasets/imagenet-a, train5981/test1519, fixed 10x20 classes. ViT-B/16 pretrained augreg2_in21k_ft_in1k, LoRA rank10. rho=.05, r=.25 or1.00 genuinely read from config; each update uses3CE gradients and1SGD, restores base snapshot and advances RNG fromg0 only.

The old A_C controls are completed four-probe runs; the new runner's C short smokes and CPU removal test verify the three-probe implementation separately. This does not claim bitwise CUDA-gradient determinism.
