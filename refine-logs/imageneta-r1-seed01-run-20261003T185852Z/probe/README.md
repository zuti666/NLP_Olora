# Seed 0/1 four-probe repetitions

This directory adapts the sixteen P-family conditions only. The actual frozen
training code remains under `trainer/`. Seven numerical/lifecycle helpers are
byte-identical to `g012-r1-four-branches-20261003T072200Z/trainer`. The training
entry differs in two metadata fields: dry-run and final-summary `seed` now read
the resolved configuration. The intentional seed-2 surrogate passed to the old
base configuration validator in `native_lifecycle.py` remains unchanged; the
requested seed is restored immediately afterward.

The queue sets CUDA visibility and holds GPU leases. This entry does not select
GPUs or create another queue. Do not invoke a real training job outside that
lease-holding queue.

The following CPU-only command verifies the documented executable entry. It must
print `seed: 0`, `tasks: 2`, `epochs: 1`, `r: 1.0`, `rho: 0.05` and create no run
output directory:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/probe/run.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/suite.json --job smoke_P01_r1_k10_s0 --dry-run
```

Formal seed-1 matched baseline validation uses the same entry:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/probe/run.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/suite.json --job P16_r1_k10_s1 --dry-run
```

The runner checks both configuration hashes and the approved condition/seed/
budget, then invokes `trainer/four_probe_train.py` in the same process. Each smoke
executes two tasks, one epoch per task, one batch per epoch: two real SGD updates
and 256 sample exposures. The scheduler horizon stays 50. Each formal job
executes ten tasks and 50 epochs: 2,550 updates, 299,050 sample exposures. The
four-probe procedure restores state exactly, replays the entry RNG at each probe,
and advances final RNG only through the base gradient. It computes no `g3`,
Fisher, EWC, weight fusion, or gradient orthogonal projection.

After a job completes, the controller validates actual output evidence with:

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/probe/validate.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/suite.json --job smoke_P01_r1_k10_s0
```

`validate.py --suite ABS --smokes` validates all 32 P-family smokes and checks
initial model and batch pairing **within each seed**, with distinct seed-0 and
seed-1 initial model hashes. Matched conditions retain the historical target:
the current candidate state's positive first-probe gradient effective SGD step,
not a separate trained-S trajectory. Actual FP32 effective-step relative error
must be at most `1e-3`.

CPU evidence is recorded in `CPU_R1_DIRECTION_REPORT.json` (ten mathematical,
FP32-step, state/RNG-restoration and baseline-radius-invariance tests),
`CPU_SEED_CONTRACT_REPORT.json` (64 job configurations, 64 wrong-seed and 64
wrong-budget rejections), and `CPU_VALIDATOR_REGRESSION.json` (unchanged legacy
receipt checks, with old seed-2 evidence rejected as seed 0). CPU results do not
claim new GPU training has completed.
