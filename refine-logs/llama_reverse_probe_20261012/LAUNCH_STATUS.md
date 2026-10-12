# Launch status — normalized reflected descent probe

Verified at 2026-10-12T10:55:34.607819+00:00.

- Controller PID2344327 is alive in the host PID namespace, detached; physical GPUs0–3 exclusively.
- All4 CUDA/BF16 witnesses passed;1B clean baseline two native smoke stages accepted.
- Active:1B clean-gradient/HF AdamW, Order1/seed0, dbpedia stage1; at least 55/218 real formal commits observed.
- Queued:1B SAM and reflected2g0−g1minus; then3B clean/SAM/reflected. Each own-chain two-stage smoke gate and first-window pairing gate precedes formal admission.
- Scope:6 complete Order1 chains,24 formal stages,12 smoke stages. Order2/3 and seeds42/1993 are catalog-only.
- No complete FAA/AAA/BWT yet; old label-boundary-bug scores are excluded.

Live state: [queue_state.json](runtime/queue_state.json). Results automatically update in [RESULTS.md](results/RESULTS.md) and [orders.csv](results/orders.csv).

New core: `src/optimer_reverse_probe.py`. Algorithm/settings: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). CPU/review:44 focused checks passed; frozen source hashes match fresh review.
