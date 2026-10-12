# Pure IncLoRA-G2: live progress and recovery

Snapshot: 2026-10-11T19:47:38.723364+00:00 UTC / 2026-10-11 21:47:38 CEST (Europe/Madrid).

**Formal training is running. No complete Order or final G2 FAA/AAA/BWT is available yet.**

| Model | Order | State | Accepted formal stages | Current progress |
|---|---:|---|---:|---|
| Llama-3.2-1B | 1 | Running: dbpedia, stage 1 | 0/4 | 138/218 actual G2 commits |
| Llama-3.2-1B | 2 | Queued | 0/4 | Starts after prior chain |
| Llama-3.2-1B | 3 | Queued | 0/4 | Starts after prior chain |
| Llama-3.2-3B | 1 | Queued | 0/4 | Starts after prior chain |
| Llama-3.2-3B | 2 | Queued | 0/4 | Starts after prior chain |
| Llama-3.2-3B | 3 | Queued | 0/4 | Starts after prior chain |

Accepted model smokes: **2/12**. Accepted formal stages: **0/24**, one running and23 pending. Accepted full Orders: **0/6**. No failure in the current retry2 attempt. Physical GPU0/1/2/3 are used together for each DDP4 stage; the six Orders run serially. Controller PID1627079, torchrun PID1627876, actual workers1627885–1627888 were verified on the host. The reservation's zero-use GPU snapshot records admission, not current utilization.

Both first-chain smokes passed independent checks: two real updates each, three CE gradient phases/two replay calls, exact base restoration and one final HF AdamW commit per step. Saved LoRA rank8→16, all historical adapter prefixes preserved byte-for-byte, finite FP32 tensors and16 unique predictions per smoke. These tiny smoke evaluations establish execution only and are not performance results.

All 138 observed formal update records have finite probe losses, final directiong2, three CE gradient phases/two replays and one HF AdamW commit. None uses fallback, Fisher, gradient history, projection, regularization hook or HVP. Training has not completed its first full evaluation. Final FAA is computed only after the four-task chain and its full30,400-example evaluation; AAA/BWT require the complete task matrix.

## Preserved failures and recovery

The original attempt passed four CUDA/BF16 witnesses but failed before model loading because its frozen source omitted six local metric-tokenizer files. The minimal launcher repair copies and pins those exact local assets. The first corrected attempt then hung at its initial NCCL collective before G2 training. Only that attempt's verified own process group was stopped; no other workload was interrupted.

A bounded four-rank diagnostic reproduced the default NCCL20-second timeout. With **NCCL_P2P_DISABLE=1**, all four ranks passed initial barrier, rank-coded all-reduce and final barrier in6.48seconds. This transport override is recorded in the active launch and worker environment. The original concurrent-stdout parsing record and its separately validated correction are both preserved.

Core G2, trainer, training entrypoint and all six model/Order configurations remain unchanged. The recovery affects asset packaging and transport only. Launcher/queue CPU checks:36/36 passed. Fresh literal documentation check and independent actual two-task model-smoke audit:PASS. Each later model/Order must pass its own two-task smoke before formal admission.

## Fixed scientific configuration

Seed0; pure Current-G2; actual transformers.optimization.AdamW, epsilon1e-6, weight decay0, betas(0.9,0.999). New LoRA rank8/alpha32/dropout0.1 on Q/V; cumulative saved adapter ranks8/16/24/32. BF16 forward with FP32 trainable LoRA, DDP4 and effective batch64. First-task LR0.001, later0.0001, constant schedule and historical explicit task caps:514 commits per full Order. No EWC/Fisher, model weight fusion or gradient projection. Virtual probes follow SGD; the real commit follows HF AdamW. Both probe radii are the unchanged dynamic eta*norm(g0), not a fixed vision-GAM r.

Order1: dbpedia→amazon→yahoo→agnews (218/78/156/62).
Order2: dbpedia→amazon→agnews→yahoo (218/78/62/156).
Order3: yahoo→amazon→agnews→dbpedia (156/78/62/218).

## Active records

- [Live queue state](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/runtime/queue_state.json)
- [Validated result table](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/results/RESULTS.md)
- [Actual host launch verification](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/LAUNCH_VERIFIED.json)
- [Independent model-smoke review](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_recovery_20261011T192153Z/GPU_MODEL_SMOKE_REVIEW.json)
- [Exact recovery commands](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_recovery_20261011T192153Z/ENVIRONMENT.md)
- [Validated NCCL comparison](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_recovery_20261011T192153Z/NCCL_DIAGNOSTICS_VALIDATED.json)

Current control root: `/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2`.
Bulk outputs: `/data/115-2/users/liying/cache/tmp/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2`.

The original and retry1 attempts, logs and frozen snapshots remain intact. Final result columns remain blank until their full completion gates pass. Archived IncLoRA baseline columns are paired seed0 controls, not newly measured G2 results. No external notifications were sent.
