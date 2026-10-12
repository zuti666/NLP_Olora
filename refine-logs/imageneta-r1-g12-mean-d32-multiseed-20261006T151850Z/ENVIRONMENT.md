# Warm local environment and documented check

Provider local conda on cudahpc40.cvc.uab.es, real PID namespace pid:[4026531836]. Use direct host tools.exec_command(require_escalated); SSH is not needed. Warm /home-local/liying/.conda/envs/Pilot_new_local/bin/python, Python3.11.15/torch2.4.1+cu121/timm1.0.9. Canonical spec SHA2564df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8 matches ledger /data/140-1/users/liying/O-LoRA/.aris/compute/local.md; no package changes.

A fresh independent agent executes this read-only command VERBATIM in the real host PID namespace after RUN_PINS exists:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-g12-mean-d32-multiseed-20261006T151850Z/suite.json --dry-run
```

Expected valid, jobs13, formal6, smokes6, allowed_gpus[2,3,4,5,6], max_parallel2. No GPU claimed by dry-run. Model acceptance requires actual allocated-GPU seeded CUDA witnesses and all six two-task smokes, then within-seed pairing. Shared index+UUID GPU leases, idle-only admission, no preemption. Static environment_spec validation notes are historical provenance; current runtime receipts show real witness/model acceptance.
