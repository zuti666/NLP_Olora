# RUN: G2 minimal shared-parent study

Execution directory: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g2-minimal-fork-gpu456-20261009T223549Z`. The active protocol is `suite.json` and `dev_split.json`; the copied `environment_spec.json` is immutable **historical environment/model provenance**, whose prior dataset budgets and wider resource policy are overridden by this suite. No environment install or source edits.

Host: `cudahpc40.cvc.uab.es`, PID namespace `pid:[4026531836]`. Warm interpreter `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`. Frozen source `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`, 190-file manifest SHA `e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2`. Same torch2.4.1+cu121/timm1.0.9 FP32/native execution. Environment fingerprint `4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8`.

Only physical GPU4/5/6, max3, idle UUID binding under shared `/tmp/olora_gamfs_gpu_locks` index+UUID leases. CUDA_VISIBLE_DEVICES is assigned exclusively by queue; each runner sees logical cuda:0. Never choose 0/1/2/3, never kill other jobs. Current older GPU45 campaign continues.

## CPU-only implementation and CLI

```bash
cd /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g2-minimal-fork-gpu456-20261009T223549Z
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s trainer -p 'test_probe_study_cpu.py' -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -m unittest discover -s trainer -p 'test_local_probe_audit_cpu.py' -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/probe_study_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config configs/formal_task1_R25_s2.json --output-dir outputs/formal_task1_R25_s2 --dry-run
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/local_probe_audit.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --reference-output-dir outputs/formal_task1_B0_s2 --output-dir outputs/local_probe_audit_s2 --dev-split dev_split.json --dry-run
```

All 14 training commands in suite accept `--dry-run`; both offline commands accept it. Full formal budgets have **no** max-epoch/max-batch flags; smoke only `--max-epochs 1 --max-batches 2`. Parent task0 trains once; each branch task1 only. Parent receipt must exist and bind the final summary before any fork. No auto-resume, no reuse of nonempty output.

Root creates frozen RUN_PINS after final tests/reviews. `gates/CPU_IMPLEMENTATION_ACCEPTED.json` pins test/review evidence. A fresh independent agent validates document/CLI/backend and writes `gates/FRESH_DOC_CHECK.json`; it may use host CPU read-only checks without taking a GPU. That gate pins reviewed docs/suite/scripts and requires no environment change.

## Host validation and launch

Run the following in the pinned **host** namespace, with the documented interpreter. A sandbox-only nvidia-smi/proc view is not GPU admission evidence.

```bash
cd /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g2-minimal-fork-gpu456-20261009T223549Z
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json --dry-run
/home-local/liying/.conda/envs/Pilot_new_local/bin/python run_queue.py --suite suite.json
```

Root launches the second command as a detached process and saves its PID/starttime in runtime/controller.json plus launch receipt. It remains alive after the interactive turn, admits only idle authorized UUIDs, runs a real CUDA finite forward/backward witness before each job, checks successful dependencies and pins before every admission, then executes native parent/fork/offline smoke acceptance before formal training. TMPDIR/TMP/TEMP use short `/tmp/olora_g2mvp_20261009T223549Z` for multiprocessing UNIX sockets. Zero preemption.

## Monitor

Read runtime/queue_state.json, runtime/controller.json, runtime/jobs/JOB/console.log, outputs/JOB/step_trace.jsonl and completed.json. Verify actual host PID/starttime and NVML UUID/PIDs; running label alone is insufficient. A completion requires zero worker exit plus validate_job.py budget/tensor/metric acceptance. results/summary.csv contains all job statuses; formal metric columns are filled only for accepted endpoints. results/comparison.csv contains only accepted formal Task1 endpoints; results/STATUS.md is regenerated on completion.

`touch runtime/PAUSE` pauses only future admissions; running jobs continue. Removing PAUSE resumes admission. No operator should change sealed suite or scripts; a necessary fix requires stopping only this controller, draining live jobs, a new version/pins/gates and explicit failure audit.
