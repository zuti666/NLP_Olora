# Shared positive four-probe sum / mean / second-difference trainer

This directory is a new copy. Existing campaigns, configurations, results and queues were not modified. The three conditions share the model/data/task lifecycle, probe positions, exact state restoration, RNG replay and natural SGD. Only final gradient arithmetic changes.

## Accepted sources and seed2 reuse

- Sum: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/trainer/{direction_step.py,four_probe_train.py}`. The selector preserves `U=_combine(g1,g2)` exactly.
- D32: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/trainer/direction_step.py`. The selector preserves `delta32=_combine(g3,g2,1.,-1.); U=_combine(g1,delta32)`. Its old enum `g1_plus_second_difference` becomes `g1_plus_d32`; arithmetic and locations stay the same.
- Queued mean: `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-half-g1-plus-g2-20261006T145451Z/trainer/direction_step.py`. It uses `0.5*g1+0.5*g2`. This trainer uses the explicitly requested `0.5*(g1+g2)`. CPU acceptance compares the four gradients, four rounded probe points, final U, RNG R1 and actual SGD output against each historical implementation, in FP32 and FP64, including reversed parameter order. All were bitwise equal on these finite nonlinear normal-range fixtures. This is not an assertion of universal bitwise identity at subnormal/overflow floating-point boundaries or across CUDA kernels.

Root handles receipt/config/protocol checks and decides reuse of the completed sum/d32 seed2 runs and the existing mean seed2 queued run. This trainer independently supports seeds 0, 1 and 2; it does not resume or trigger any old job.

## Probe construction and final formulas

For all conditions, `g0=grad L(phi)`, `phi1=phi+.05*N(g0)`, `g1=grad L(phi1)`, `phi2=phi+1.0*N(g1-g0)`, `g2=grad L(phi2)`, `phi3=phi2+.05*N(g2)`, `g3=grad L(phi3)`. The outer point is constructed from the original base phi, not phi1. Normalization uses all trainable LoRA factors and FC parameters with the existing FP64 global reduction and epsilon 1e-12. It is only used to construct probes.

| Enum | Final tensor arithmetic | Uses g3 in U |
|---|---|---|
| `g1_plus_g2` | `g1_plus+g2` | false |
| `half_g1_plus_g2` | `0.5*(g1_plus+g2)` | false |
| `g1_plus_d32` | `g1_plus+(g3-g2)` | true |

Each step makes exactly four ordinary CE autograd calls (`create_graph=false`) and one natural `torch.optim.SGD.step`. Mean scales the sum only once; all selectors retain original nominal LR=.04 at Task0, .02 afterwards, the original cosine schedule and LR multiplier=1. No projection, final norm matching, EWC/Fisher, merging, replay or HVP is added.

The generic base resolver clears boolean `*_probe` flags. This new resolver explicitly sets `compute_g3_probe=true`, `probe_gradient_count=4` and the requested `g3_used_in_final` after calling the base, so saved effective config agrees with custom execution. Traces and actual forward/counter accounting remain the authoritative budget evidence.

## State, RNG, seeds and validation

Before each probe, the copied transaction restores base parameter values, gradients, buffers, train/eval modes, delta-hook controls/caches and RNG R0. The evaluated rounded point is written with `copy_`. After probes, restore the base exactly and advance RNG to R1 from the g0 forward once, then install U and update once. Probe exceptions restore all entry state; update exceptions restore base parameters. Zero momentum and weight decay are enforced. Data seed rule remains `seed*100000+task*1000+epoch`.

Native lifecycle's temporary seed2 surrogate only satisfies the inherited single-task config validator; it immediately restores the requested seed list before training. Actual source training, controller identity, effective config, step trace and summary all use the requested seed. CPU tests directly exercise configuration and controller installation for all 9 seed-selector combinations.

Eight meaningful CPU tests cover independent nonlinear analytic gradients/points/U, derivative and update budgets, historical FP32/FP64 equivalence, same-state three-selector comparability, fourth-gradient influence only for d32, full stochastic state/RNG rollback, all four failure positions, every seed/controller condition, invalid configuration rejection, original LR in mean, zero-momentum guard and duplicate/failed update safety. Run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python trainer/test_three_directions_cpu.py
```

`CPU_GATE.json` records the acceptance result. No GPU training was executed by the implementing agent. Direct CLI remains `four_probe_train.py --source ... --config ... --output-dir ...`; optional `run.py --suite ... --job ...` translates the same job into that CLI and supports `--dry-run`.
