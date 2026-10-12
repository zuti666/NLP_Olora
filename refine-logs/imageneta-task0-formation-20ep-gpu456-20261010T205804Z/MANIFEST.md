# Artifact inventory

- `EXPERIMENT_PLAN.md`: the frozen six-condition, two-task20epoch protocol and interpretation.
- `protocol.json`: active algorithm formulas, radii, seeds, data split, metrics and GPU456 policy.
- `configs/`: 18 formal training configurations and six acceptance-smoke configurations.
- `suite.json`: 25 explicit jobs and successful-acceptance dependencies.
- `trainer/formation_train.py`, `formation_step.py`: actual epoch-scheduled implementation.
- `RUN.md`, `RUN_PINS.json`: documented invocations and57 frozen input hashes.
- `source_manifest.json`: 190 original-source hashes.
- `environment_spec.json`: unchanged warm environment reference; active workload is defined in protocol/suite.
- `gates/`: author CPU, independent design/code, fresh literal-document evidence. Native GPU evidence is written under runtime/jobs.
- `runtime/LAUNCH_RECEIPT.json`, `controller.json`, `controller.log`, `queue_state.json`: durable controller and live admission/progress.
- `runtime/jobs/<job>/`: console, native CUDA witness, worker PID/exit and accepted completion receipt.
- `outputs/`: link to dedicated cache/tmp campaign outputs; each native training has Task0 and Task1 checkpoints, step trace and epoch metrics.
- `FORMAL_RESULTS.csv`, `AGGREGATED_RESULTS.json`, `PAIRED_SEED_DIFFERENCES.csv`: accepted formal results only.
- `TASK0_LEARNING_CURVES.csv`, `TASK1_RETENTION_CURVES.csv`: task learning and allseen40 retention curves; Task1epoch0 is post-expansion/pre-learning.
- `RESULTS.md`: live readable results; a queued or short-run experiment is never marked as a complete scientific result.
