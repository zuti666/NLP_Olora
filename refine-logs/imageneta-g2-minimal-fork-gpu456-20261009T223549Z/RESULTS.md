# G2 minimal fork: completion and accepted results

Verified UTC: 2026-10-10T07:21:17.039787+00:00. Queue ended: 2026-10-10 02:09:25 CEST (2026-10-10T00:09:25.986744+00:00 UTC). **17/17 jobs accepted**, running0, failed0, pending0. This includes7 formal training jobs (one shared SGD Task0 parent+six Task1 branches),1 formal offline audit, and9 smoke/pairing jobs. Controller/monitor/all recorded workers exited; physicalGPU4/5/6currentlyidle by host audit. Actual jobs used5/6;4was occupied by the independent earlier campaign.

## Fixed protocol and scope

Seed2, ImageNet-A original200classes10×20fixed partition; this study trains **Task0 then Task1 only**, not10tasks. Each trained task50epochs, ViT-B/16 augreg2_in21k_ft_in1k, Q/V LoRArank10 +completeFC, batch128, ordinarySGD momentum/WD0, LR.04/.02 epochcosine minratio.1, strictnativeFP32. No EWC/Fisher/history/weight fusion/projection/HVP/clip/finalnormrescale/replay. Task0 train/dev1141/118, Task1 906/91; originaltest1519unchanged. All branches restore the identical accepted SGD parent before FC20→40 expansion. Final g2coefficient1. h=eta||g0||. All6850formaltraininggradientreads/2850SGD completed, fallback0; source190/pins57 unchanged.

Primary predeclared selection: Task1 epoch50 held-out-train current20class **devCE**. Officialtest accuracies/two-stageFAA/AAA/BWT are report-only. Dev accuracy uses current20class classification; test matrices use native argmax over allseenclasses. Parent-relative BWT includes FC20→40 competition and learned changes. Older full-data/fullCL10scores do not enter this table. Single seed; sampleSD notavailable.

## Six accepted Task1 branches

| Condition | Inner a | Outer r | Dev CE ↓ | Dev Acc (%) | Current Task Acc (%) | Old Task Acc (%) | FAA (%) | AAA (%) | BWT (pp) | ΔFAA vs SGD (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | — | — | 0.492012 | 84.615 | 75.980 | 81.430 | 78.705 | 81.698 | -3.260 | +0.000 |
| R25 | 0.05 | 0.25 | 0.484141 | 82.418 | 79.530 | 81.760 | 80.645 | 82.668 | -2.930 | +1.940 |
| S_F | h | 0.25 | 0.478829 | 83.516 | 80.310 | 82.080 | 81.195 | 82.942 | -2.610 | +2.490 |
| F_S | 0.05 | h | 0.484390 | 83.516 | 77.560 | 82.410 | 79.985 | 82.338 | -2.280 | +1.280 |
| S_S | h | h | 0.481411 | 83.516 | 76.770 | 82.740 | 79.755 | 82.222 | -1.950 | +1.050 |
| R100 | 0.05 | 1.0 | 0.549683 | 81.319 | 82.280 | 78.830 | 80.555 | 82.623 | -5.860 | +1.850 |

Shared Task0 parent's taskaccuracy: 84.690%. Task0 450updates; eachTask1 400updates. FAA is final mean of two task groups; AAA is equal mean over **two completed stages**, not50epochmean. Native grouped accuracies were rounded to two decimals; derived three-decimal display does not imply finer measurement.

## Findings limited to this screen

- Best by the preregistered endpoint criterion: **S_F**, devCE0.478829; it also has highest FAA81.195/AAA82.942. Compared with same-parent SGD: FAA+2.490, AAA+1.245, current-taskaccuracy+4.330, old-taskaccuracy+0.650pp. The larger measured gain is new-task learning, not only retention.
- R100 has highest currenttask test accuracy82.280%, but devCE0.549683 and BWT-5.860pp are worse. Bigger fixed outerradius is not unconditionally better in this fork protocol; do not select it from current test accuracy alone.
- Fixed versus h changes the whole distance policy, including initial scale; these results do not separately identify learning-rate versus gradient-norm causes. Single seed/two tasks cannot establish a fullCL10 optimum or Hessian mechanism.

## Formal local diagnostics

Completed **5 snapshots×4 fixed training batches**, 400 ordinarygradientreads,820 no-gradientlossreads, committedSGD0. Same-state restoration and disjoint TRAIN-only probe/holdout, no dev/test access passed. Inner.025/.05/.1/h; representative fixed-v outerpoints/halfpairs; same-radius SAM/opposite/random controls; crossedeta.02/.002 and actualFP32length-matched virtualupdates. These are passive local responses, not trained-method scores.

## Evidence

- [Completion audit](results/status_20261010/COMPLETION_AUDIT.json)
- [Actual host PID/GPU audit](results/status_20261010/HOST_PROCESS_GPU_AUDIT.json)
- [Accepted branch CSV](results/comparison.csv)
- [Queue state](runtime/queue_state.json)
- [Formal offline audit](outputs/local_probe_audit_s2/final_summary.json)
