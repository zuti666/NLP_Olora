# Seed 0/1/2 execution — 2026-09-22

User authorized physical GPUs 2, 3, 4, 5. Joost inventory found multi-seed ImageNet-A/R, CIFAR100 and DomainNet experiments, but no eligible result for the five target datasets.

This campaign contains 99 new formal runs + 6 reused, verified Sep18 formal seed0 runs, covering 7 methods × 5 datasets × seeds 0/1/2. The six reused runs are CL-LoRA on all five datasets and GAM-EWC on Het5. The 99-run policy is the conservative frozen-cohort option from the plan, not a claim that source recovery failed or historical scores are invalid. New seed0 results are included regardless of whether they improve historical scores. They are not used to select learning rates or seeds.

| Method | New full runs | Reused seed0 |
|---|---:|---:|
| Vanilla LoRA | 15 | 0 |
| InfLoRA | 15 | 0 |
| OLoRA | 15 | 0 |
| SDLoRA | 15 | 0 |
| CL-LoRA | 10 | 5 |
| EWCLoRA | 15 | 0 |
| GAM-EWCLoRA | 14 | 1 |

There are 9 additional diagnostic runs: one Flowers smoke for each method, plus native-EWC/Het5 and formal-GAM/Het5. Each exercises two tasks, one epoch per task, with rank16 and batch128. The trainers set effective nb_tasks=2 for capped smoke runs; those values are never pooled with full runs. Each full run waits for its implementation's smoke. New seed1/2 also wait for successful full seed0 in the same group; this verifies complete execution without imposing a score threshold. A failed job is parked, its descendants held, other groups continue. No automatic hyperparameter or batch-size changes, no preemption, no blind retries.

Frozen baseline sources come from Flatness_CV Git 9c4439b; EWC/CL/GAM use the Sep18 snapshot, with immutable file/config hashes. The original repositories are untouched. Per-dataset historical LR/epochs and method-specific differences are retained; this is a multi-seed archived-configuration table, not an equal-training-budget comparison. Details are in both preparation directories and suite.json. Pets is absent as a standalone column but is still the fifth Het5 component.

## Reproduction and monitoring

The existing Pilot_new environment is reused, spec e94a148f. No package installation is needed. Cached pretrained weights and all datasets are local/offline. Physical GPU UUID binding keeps each job's logical device at 0. Four seeded CUDA witnesses passed in env_witness.json.

Read-only launch validation (does not allocate a GPU or start training):

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/seed012_execution_20260922/launch_queue.py --dry-run
```

Actual detached launch, once preflight and runtime hashes are frozen:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/seed012_execution_20260922/start_background.py
```

The controller checks GPU compute processes/memory/utilization before every launch, keeps a per-GPU advisory lock, and runs at most one experiment per authorized GPU. It writes state.json, queue.log and logs/<job>.log. Receipts require complete finite CNN matrices, valid gradients/parameters and source/config/metric SHA256. Existing logs and receipts are never overwritten. After a controller interruption, inspect children before using launch_queue.py --resume; live children cause a refusal to avoid duplicates.

collect.py produces results.json, results.csv and SUMMARY.md. FAA and AAA_stage are recomputed from raw task matrices; sample SD uses ddof=1 over seeds, and incomplete groups are marked. Historical selected best scores are not mislabeled as new means.

Known provenance limits: historical data manifests bind ordered file paths/labels, not image bytes. Reused GAM-Het5 did not record a checkpoint byte hash at the historical launch; its model ID, frozen source, environment and currently resolved offline cache are checked. These limits remain documented rather than being presented as exact historical binary identity.
