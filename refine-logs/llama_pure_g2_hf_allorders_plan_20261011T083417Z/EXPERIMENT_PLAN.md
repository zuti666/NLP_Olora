# Pure IncLoRA-G2: Llama-3.2-1B/3B, Orders 1–3

Prepared for the user-authorized physical GPUs **0, 1, 2, 3**. Initial campaign: **seed0**, six complete Orders and 24 formal training/evaluation stages. Seeds42 and1993 are supported by the individual launchers but are not scheduled in this campaign.

## Method and fixed protocol

Only **IncLoRA-G2 with Transformers AdamW** is selected. Actual backend must be `transformers.optimization.AdamW`, epsilon `1e-6`, weight decay `0`, betas `(0.9, 0.999)`.

For the accumulated current-task cross-entropy gradient, over the fresh trainable LoRA parameters:

\[
g_0=\nabla L(\theta),\quad h=\eta\|g_0\|,\quad
g_1^-=\nabla L(\theta-\eta g_0),\quad a=g_0-g_1^-,\quad
g_2=\nabla L\left(\theta+h\frac{a}{\|a\|+\xi}\right).
\]

Both probes originate from the same restored base parameters. Restore base parameters again, submit **only g2**, clip its global norm to1, and commit **one** ordinary AdamW update. The prediction displacement is virtual SGD even though the final optimizer is AdamW. This campaign preserves `g2_inner_radius_mode=sgd_step` and `g2_outer_radius_mode=sgd_step`; it does not import the vision experiments' fixed `r=1` geometry.

The same accumulated microbatches and dropout RNG are replayed for the three gradients. Gradients use the global four-rank mean. Numerically zero/indistinguishable responses have explicit logged fallbacks; smoke requires actual non-fallback G2 steps. No EWC, Fisher, old-gradient cache, OLoRA penalty, orthogonal correction, Hessian-vector product, or model weight averaging is executed. IncLoRA's ordinary frozen-adapter concatenation is retained.

| Setting | Fixed value |
|---|---|
| Base models | Local Llama-3.2-1B and Llama-3.2-3B |
| Seed / data seed | 0 / 0 |
| New LoRA rank / alpha / dropout | 8 / 32 / 0.1 |
| LoRA modules | q_proj, v_proj |
| Saved cumulative adapter rank | 8 → 16 → 24 → 32 |
| Precision | BF16 forward; FP32 trainable parameters |
| Parallelism / effective batch | DDP4; batch1/device × GA16 ×4 =64 |
| Epoch / scheduler / warmup | Nominal1 epoch/task; constant LR; warmup0 |
| Learning rate | First task0.001; subsequent tasks0.0001 |
| Maximum gradient norm | 1.0, once before the final optimizer commit |
| Source / target / generation maximum length | 512 / 50 / 50 |
| Evaluation batch | 4/device |
| Prediction population | 7,600 examples per seen task; 30,400 at the final stage |

The explicit update caps below preserve the historical full effective-batch protocol and exclude extra tail-window updates.

## Six independent chains

| Sequence | Model | Task order | Formal updates by stage | Launcher |
|---:|---|---|---|---|
| 1 | 1B | dbpedia → amazon → yahoo → agnews | 218 / 78 / 156 / 62 | [Order1 1B](../../scripts_order1_llama1B/IncLoRA-G2_LLaMA3-1b.sh) |
| 2 | 1B | dbpedia → amazon → agnews → yahoo | 218 / 78 / 62 / 156 | [Order2 1B](../../scripts_order2_llama1B/IncLoRA-G2_LLaMA3-1b.sh) |
| 3 | 1B | yahoo → amazon → agnews → dbpedia | 156 / 78 / 62 / 218 | [Order3 1B](../../scripts_order3_llama1B/IncLoRA-G2_LLaMA3-1b.sh) |
| 4 | 3B | dbpedia → amazon → yahoo → agnews | 218 / 78 / 156 / 62 | [Order1 3B](../../scripts_order1_llama3B/IncLoRA-G2_LLaMA3-3b.sh) |
| 5 | 3B | dbpedia → amazon → agnews → yahoo | 218 / 78 / 62 / 156 | [Order2 3B](../../scripts_order2_llama3B/IncLoRA-G2_LLaMA3-3b.sh) |
| 6 | 3B | yahoo → amazon → agnews → dbpedia | 156 / 78 / 62 / 218 | [Order3 3B](../../scripts_order3_llama3B/IncLoRA-G2_LLaMA3-3b.sh) |

Each chain starts from the original pretrained model independently. Within a chain, each later stage loads the preceding saved cumulative adapter. Before that chain's formal training, its own two-task smoke runs2 updates/task with bounded128 train samples and16 predictions. These 12 smoke stages are separate outputs and never contribute to final performance statistics.

## Runtime admission and outputs

The six chains execute serially, each using all four authorized GPUs. The controller waits until all four cards are idle and obtains shared physical-index and UUID locks. It does not preempt existing processes. Fresh seeded FP32 CUDA and BF16 forward/backward witnesses run on each reserved card before any model training.

Source, commands, configuration, local models/tokenizers and data input hashes are frozen before launch. Every stage checks the frozen manifest and source. Failed or interrupted outputs are preserved and cannot be silently overwritten. The queue stops on training, runtime audit, adapter, prediction, or collection failure.

- Control root: `/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z`
- Bulk adapters, predictions, datasets cache and training logs: `/data/115-2/users/liying/cache/tmp/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z`
- Live state: control root `runtime/queue_state.json`.
- CUDA receipts: control root `runtime/cuda_witness.json` and per-card logs.
- Formal results: control root `results/orders.csv`, `stages.csv`, `across_orders.csv`, `RESULTS.md`.
- Per-stage validation: bulk stage directory `g2_script_verification.json`, `g2_runtime_audit.json`, `g2_step_trace.jsonl`.

The output location avoids the nearly full project disk. All six configuration files are under `configs/g2_llama/inclora_g2_llama{1b,3b}_order{1,2,3}.json`. Training source is `src/optimer_g2.py`, `src/uie_trainer_lora.py`, and `src/run_uie_lora.py`; launchers share `scripts_llama/run_g2_llama.py`.

## Acceptance and analysis

For each stage verify exact update budget, one real AdamW commit per update, actual HF backend/epsilon/decay, correct Q/V count, FP32 trainable tensors, finite loss/gradients, explicit response fallback counts, restored base origin, absence of Fisher/regularization/projection hooks, cumulative adapter geometry, and unchanged old-adapter prefix. Predictions must cover unique expected dataset/id pairs and match frozen test labels.

Let A[t,j] denote exact-match accuracy on task j after learning task t. Report:

- **FAA (%)**: mean final-stage accuracy across the four tasks.
- **AAA (%)**: mean of the four stages' seen-task average accuracies.
- **BWT (percentage points)**: mean `A[4,j] − A[j,j]` over the first three tasks.
- Per-task/stage accuracy, global updates, CE phase/replay count, fallback fraction, train/evaluation time and trace provenance.

Final FAA/AAA/BWT remain blank until all four stages of an Order pass. After three Orders finish for a model, report their arithmetic average for seed0. This is not a multi-seed mean or standard deviation. Compare FAA only against the same model, Order and seed0 archived IncLoRA row in `history_reuse_20260926/final_per_seed.csv`; preserve raw predictions for recomputation.

This experiment measures the complete implemented pure-G2 procedure. Relative to old SAM/GAM, it also changes probe geometry, RNG replay and distributed gradient construction, so an improvement alone does not identify a Hessian mechanism or isolate one gradient component.

See [ENVIRONMENT.md](ENVIRONMENT.md) for reproducible dry-run, preparation, execution and monitoring commands. Preparation and launch status are recorded separately; this plan itself makes no completed-training claim.
