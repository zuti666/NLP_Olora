# ImageNet-A g1+(g3−g2) local environment

Warm reuse /home-local/liying/.conda/envs/Pilot_new_local/bin/python on cudahpc40.cvc.uab.es. No package changes. The environment_spec.json has the unchanged canonical SHA256 4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8; provider ledger is /data/140-1/users/liying/O-LoRA/.aris/compute/local.md. Its historical validation section is provenance, not certification of this new run.

Execute this documented read-only invocation verbatim in the real host PID namespace (outside filesystem/GPU sandbox). It verifies pinned code/source/config paths without claiming GPUs or creating training outputs:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/suite.json --dry-run
```

Expected: status valid, jobs5, formal2, smokes2, allowed_gpus [2,3,4,5,6], max_parallel2. RUN_PINS.json must exist first. Fresh independent agent execution is recorded in gates/DOC_CHECK.json.

All model training goes through the supervisor and shared index/UUID leases. Workers repeat trainer/cuda_witness.py after a GPU is claimed, before real model/data short tests. Witness is seeded FP32 forward/backward, TF32 off. Sandbox CUDA failure is not a hardware failure. The two two-task/one-epoch/one-batch short tests and input/initial-state pairing must pass before two full natural trainings. GPU validation is pending while the allowed cards are busy.

Suite environment uses offline HF/Torch caches at /home-local/liying/.cache, PYTHONDONTWRITEBYTECODE=1, PYTHONHASHSEED=0, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP/MKL/OPENBLAS threads8. Each child sees its chosen physical UUID as logical cuda:0. Only GPUs2–6 are eligible; GPUs0/1 excluded.

Source remains the unchanged190-file snapshot listed in source_manifest.json. Data: /data/140-1/datasets/imagenet-a, train5981/test1519, fixed10x20 classes. Model: ViT-B/16 augreg2_in21k_ft_in1k, rank10. Actual rho=.05 and r=.25/1 from config. Exactly4CE gradients and1SGD/update; g3 at outerpoint+rho N(g2), base restoration and R0 replay, final RNG=R1. Zero projection/EWC/Fisher/weight fusion/HVP; no final normalization or matching. Do not infer model acceptance from a CPU or kernel witness alone.

