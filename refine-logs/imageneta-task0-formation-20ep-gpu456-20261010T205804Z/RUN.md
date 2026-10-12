# Run the six-policy Task0 formation study

This is a new two-task20-epoch protocol, seeds0/1/2. No historical50-epoch parent or score is loaded. The only authorized physical devices are4/5/6; a single worker sees one leased UUID as logical cuda:0. Existing work is never preempted.

Working directory:

```bash
cd /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z
```

Warm environment: `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`; canonical inherited environment-spec SHA256 `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`. Native FP32 attention, torch2.4.1+cu121, TF32/AMP disabled. No installation. Workload and resource restrictions are defined by `protocol.json` and `suite.json`; the byte-identical `environment_spec.json` is an inherited environment reference, not the active50-epoch workload. See `.aris/compute/local.md` in the project root.

## Literal CPU checks for the independent document reader

These six commands require no GPU. Run them verbatim with CUDA_VISIBLE_DEVICES empty and OMP/MKL/OPENBLAS threads capped at2; do not alter a command to make it pass. Every dry-run must return JSON status `valid`, exactly20epochs/task and Task1SGD. The unit test must exit0. Logs and exit codes are independent evidence.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
/home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer -p 'test_formation*.py' -v
/home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -m unittest discover -s /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z -p test_queue_cpu.py -v
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/configs/formal_gam_residual_s1.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/outputs/formal_gam_residual_s1 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/configs/formal_sf_g2_then_sgd_s0.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/outputs/formal_sf_g2_then_sgd_s0 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/configs/smoke_sgd_then_sf_g2_s2.json --output-dir /data/115-2/users/liying/cache/tmp/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/outputs/smoke_sgd_then_sf_g2_s2 --max-batches 1 --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/suite.json --dry-run
```

The queue dry-run must execute in hostname `cudahpc40.cvc.uab.es`, PIDnamespace `pid:[4026531836]`; this is the actual host namespace. It verifies25jobs,18formal, sixsmokes, GPU[4,5,6], maxparallel3, source190 hashes, pinned input artifacts and exclusive output paths. It does not claim that CPU gates or CUDA tests have passed, and never leases a GPU. `RUN_PINS.json` must already be sealed by the author before this check.

## Authorized execution (controller operator only)

After CPU_IMPLEMENTATION_ACCEPTED, INDEPENDENT_IMPLEMENTATION_REVIEW and FRESH_DOC_CHECK all pass with evidence hashes, the actual-host operator starts one durable `run_queue.py --suite suite.json` process with the pinned interpreter and environment. This section is not part of the independent reader's six CPU commands. The controller holds its campaign controller lock, waits for idle physical4/5/6, and leases both shared index and UUID locks. The queue sets CUDA_VISIBLE_DEVICES to the leased UUID. No individual training job may bypass that admission.

Every worker first runs `trainer/cuda_witness.py` with seeded real CUDA forward/backward, finite gradients and TF32off, then its frozen command from `suite.json`. Sixshort runs each train BOTHtasks for20epochs with onebatch/epoch (40commits); they cover epoch11. Formal jobs depend on all six accepted short runs and `formation_smoke_pairing`. The pairing gate requires identical initial/model/input/LR/newFCrows and exact first-ten-epoch trajectories for SGD vs lateG2 and fullG2 vs earlyG2. AllTask1steps areSGD. A failed gate prevents formal admission and is preserved for inspection.

`runtime/queue_state.json` contains pending/running/completed/failed counts and actualworker PID/starttimes, GPUs and commands. `runtime/jobs/<id>/console.log`, exit.json, completed.json and CUDAwitness are evidence. The accepted result collector writes only complete20-epoch formal runs to `FORMAL_RESULTS.csv`, `AGGREGATED_RESULTS.json`, `PAIRED_SEED_DIFFERENCES.csv`, `TASK1_RETENTION_CURVES.csv` and `TASK0_LEARNING_CURVES.csv`. Short-run scores never enter formal tables. The collector's own output inventory is documented in `RESULTS.md`.

Native task checkpoint exports and epoch records are under `/data/115-2/users/liying/cache/tmp/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/outputs/<id>/`; `outputs` in this directory points there. No686MBfullbackbone snapshots are needed for this in-process two-task experiment.
