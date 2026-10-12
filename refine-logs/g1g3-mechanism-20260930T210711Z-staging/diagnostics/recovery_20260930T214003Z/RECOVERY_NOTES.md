# Diagnostic queue recovery

The failed `g1_task9` job loaded its checkpoint successfully, then failed in
`paired_gam_diagnostic.py:479` while caching its training batches:

```text
ValueError: Not enough batches: requested 4, got 3
```

Task 9 has 378 training examples. With the original batch size 128 and
`drop_last=False`, its full pass is exactly `[128, 128, 122]`. The diagnostic
report is created after batch caching, so no `results/g1_task9.json` was written;
the queue's secondary `FileNotFoundError` is a consequence of the batch error.
This was not a terminal-checkpoint or task-index failure.

Only the recovery suite changes. It schedules the seven unfinished anchors,
sets the three task-9 jobs to `max_batches=3`, keeps the other four jobs at 4,
and sets concurrency to 1. Script, source, batch size, sampling seeds, transforms,
direction formulas, norm matching and all evaluations remain unchanged. No
samples are duplicated or padded to fabricate a fourth task-9 batch.

| Anchor task | Training examples | Diagnostic batch sizes |
|---|---:|---|
| 0 | 1259 | 128, 128, 128, 128 (fixed subset) |
| 4 | 426 | 128, 128, 128, 42 |
| 9 | 378 | 128, 128, 122 |

`g1_task0` and `g1_task4` are reused unchanged after validating their completion,
finite/restoration checks and 100/100 matched counterfactuals apiece. The old
suite, failed runtime, logs, scripts and outputs are preserved. The seven new
output paths were verified absent. The new runtime is
`diagnostics/runtime/formal_recovery_20260930T214003Z`; outputs stay under the
original `diagnostics/results/`, so the existing final summarizer sees all nine
anchors. GPU pool 0–6, idle thresholds and shared flock ownership are unchanged.

Deploy this staged `suite.json` to:

```text
/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/suite_recovery_20260930T214003Z.json
```

Root's preflight (this preparation did not run the queue or touch GPUs):

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -u /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/run_queue.py --suite /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/suite_recovery_20260930T214003Z.json --validate-only
```

Once all seven recovery jobs finish, an optional strict nine-anchor report check is:

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -u /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/summarize_diagnostics.py --results /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/results --output /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/diagnostics/summary --expected 9
```

For future single-task epoch checkpoints, place the original checkpoint filenames
inside distinct epoch directories, preserving the task ID and exact class mapping.
Track source arm/task/epoch/checkpoint hash separately. Epochs and diagnostic
batches are not independent seeds. If an epoch-50 checkpoint has the same digest
as an existing task-end anchor, cite that result rather than counting it twice.
Task-0 diagnostics correctly report old-task metrics as unavailable.
