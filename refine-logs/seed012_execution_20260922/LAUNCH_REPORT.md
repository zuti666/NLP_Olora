# Seed012 campaign launch status

Snapshot: 2026-09-22T16:09:50.331151+00:00. Live state is state.json; this report is not a live progress indicator.

Controller PID: 2561186; status: running. Physical GPUs: 2, 3, 4, 5, at most one job per device.

Joost multi-seed inventory: no eligible Aircraft/Cars/CUB/Flowers/Het5 run; available multi-seed datasets are ImageNet-A/R, CIFAR100 and DomainNet.

Cohort: 99 newly queued full runs + 6 validated archived seed0 results = 105 results, 35 groups × seeds0/1/2. Nine extra two-task one-epoch smoke runs are excluded from statistics.

| Stage | Status | Count |
|---|---|---:|
| formal | pending | 97 |
| formal | running | 2 |
| smoke | completed | 7 |
| smoke | running | 2 |

| GPU | PID | Current job |
|---:|---:|---|
| 3 | 2574989 | smoke_ewclora_het5_s0 |
| 2 | 2575838 | smoke_gam_ewc_het5_s0 |
| 4 | 2578313 | seqlora_flowers_s0 |
| 5 | 2578703 | inflora_flowers_s0 |

Collector verified: {'accepted_runs': 6, 'new_formal': 0, 'reused': 6, 'complete_groups_n3': 0, 'groups_total': 35, 'excluded': 0, 'not_completed': 99}.

Environment and dependency witnesses, CPU data/weight checks and independent documented-command dry-run passed. Full-run metrics require finite complete task matrices; errors park the affected group instead of changing hyperparameters.

Files: README.md (policy/reproduction), suite.json (all frozen jobs/configs), queue.log, logs/<job>.log, receipts/<job>.json, SUMMARY.md, results.json and results.csv.

Limitations remain documented: source recovery does not establish binding to every old run; the new campaign does not pool unbound historical seed0 values. Six reused formal runs have source/config/data evidence, but historical image-byte hashes and GAM-Het5 historical checkpoint-byte hash are unavailable.
