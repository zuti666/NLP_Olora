# Pure IncLoRA-G2: final results and paired analysis

Report verified at 2026-10-12T08:53:42.554989+00:00. Client timezone: Europe/Paris. Campaign completed at **2026-10-12 04:36:20 CEST** (2026-10-12T02:36:20+00:00). Elapsed active retry2 campaign: 6:56:07.

**Complete and independently validated: 6/6 model×Order cells, 24/24 formal stages and 12/12 model smokes; failed0, missing0, invalid0 in the completed retry2 campaign.** The actual controller and workers are gone; physical GPU0–3 were idle at the read-only host verification. Earlier startup failures are retained separately and excluded from performance analysis.

## Fixed setting

Only seed0 was run. Llama-3.2-1B/3B each use Orders1–3, four tasks per Order. Q/V LoRA new rank8, alpha32, dropout0.1; cumulative saved ranks8/16/24/32 with exactly frozen historical prefixes. BF16 forward and FP32 trainable LoRA, DDP4, batch1/device×16 accumulation×4ranks=64. Actual transformers.optimization.AdamW: epsilon1e-6, weight decay0, betas(0.9,0.999), one final gradient clip at norm1. First-task LR0.001 and later LR0.0001, constant schedule, nominal1epoch with historical explicit update caps, max input512 and target/generation50.

Order1: dbpedia→amazon→yahoo→agnews (218/78/156/62 updates). Order2: dbpedia→amazon→agnews→yahoo (218/78/62/156). Order3: yahoo→amazon→agnews→dbpedia (156/78/62/218). Each model×Order has514 actual commits; all six total3084 commits.

The evaluated pure Current-G2 uses a descending virtual SGD prediction and a response-selected outer relocation:

$$g_0=\nabla CE(\theta),\quad h=\eta\|g_0\|,\quad g_1^-=\nabla CE(\theta-\eta g_0),\quad g_2=\nabla CE\bigl(\theta+hN(g_0-g_1^-)\bigr).$$

Restore BASE exactly, clip the submitted g2 once, then commit once with the stated HF AdamW. Both probe radius modes remain sgd_step; replay reuses the same microbatches/dropout state and global mean. No EWC/Fisher, gradient history, projection, HVP, orthogonality penalty or model weight averaging. Probe definition is unchanged from the accepted implementation.

## Metrics and primary results

All metrics below are percentages; BWT and absolute deltas are percentage points. FAA is final mean exact-match accuracy across four tasks (7,600 examples per task,30,400 per final evaluation). AAA averages four stages, each stage using the equal-weight mean of its seen tasks. BWT is the mean final-minus-learning-stage accuracy of the first three tasks; larger is better. LA is the mean accuracy immediately after each task is learned. Across-Order means average the three complete Orders within seed0. Seed sample SD is N/A; Orders are not independent seeds.

| Model | Order | G2 FAA | G2 AAA | G2 BWT | IncLoRA FAA | Delta FAA |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.2-1B | 1 | 64.503 | 67.366 | -10.908 | 65.599 | -1.095 |
| Llama-3.2-1B | 2 | 64.839 | 71.452 | -10.088 | 64.507 | +0.332 |
| Llama-3.2-1B | 3 | 68.286 | 63.575 | -6.873 | 67.470 | +0.816 |
| **Llama-3.2-1B** | **Mean** | **65.876** | **67.464** | **-9.289** | **65.859** | **+0.018** |
| Llama-3.2-3B | 1 | 71.345 | 74.152 | -5.075 | 68.694 | +2.651 |
| Llama-3.2-3B | 2 | 72.474 | 77.453 | -3.566 | 70.204 | +2.270 |
| Llama-3.2-3B | 3 | 72.688 | 68.237 | -2.531 | 71.602 | +1.086 |
| **Llama-3.2-3B** | **Mean** | **72.169** | **73.281** | **-3.724** | **70.167** | **+2.002** |

## Interpretation

1. **3B has positive paired FAA differences in all three Orders for seed0.** Mean FAA increases72.169−70.167=2.002pp; mean AAA improves3.136pp. LA improves1.151pp and BWT improves1.135pp, so this observation includes both better immediate task learning and better old-task retention. It does not establish multi-seed stability or a Hessian mechanism.

2. **1B has little mean final benefit.** Mean FAA increases only0.01754pp; two Orders improve, while Order1 loses1.095pp. Mean LA improves0.876pp and AAA improves0.997pp, but mean BWT worsens1.145pp. The learning improvement is nearly offset by worse retention. With four tasks, the exact metric identity FAA=LA+3/4 BWT explains this arithmetic: deltaFAA0.01754=0.87610+0.75×(−1.14474). This is a metric decomposition, not a causal proof.

3. **Evidence is model- and order-dependent.** The current setting is encouraging on3B and close to neutral on1B. Do not pool1B/3B into one universal improvement claim or compare this seed0 row against historical three-seed averages.

## Comparator integrity and costs

The independent reviewer recomputed24 current and24 same-model/Order/seed0 IncLoRA triangular evaluations directly from native JSONL. It checked7,600 unique IDs per task and canonical gold labels, verified prediction hashes, and reproduced all displayed current FAA/AAA/BWT and archived IncLoRA FAA to below1e-10. Selected non-target scientific settings, dataset/instruction/task files, adapter geometry, BF16/FP32 modes, effective batch, actual HF optimizer backend/epsilon/decay and all24 update counts match. Historical inactive EWC lambda differs textually(1 versus0) but baseline runtime confirms no active Fisher/EWC contribution.

G2 uses three CE gradient phases and two replay calls per commit, compared with one gradient phase for IncLoRA. Update budgets are matched; compute budgets are not. Current NCCL_P2P_DISABLE=1 is a recorded communication remedy after the separately archived startup timeout. Total formal cost:3084 commits,9252 CE gradient phases,6168 replay calls, zero fallback. Reported train runtimes sum10180.1334s and prediction runtimes sum13189.1804s; these are instrumentation sums, not a causal speed comparison.

## Validation and follow-up

Independent native CPU revalidation of every stage passed: all saved FP32 tensors finite, correct cumulative ranks and byte-identical historical prefixes; expected updates and16 smoke predictions or7,600×stage formal predictions; actual HF AdamW group settings; full normal-G2 traces; finite required observed training/probe/prediction metrics. All112 frozen source files,53 input files and manifest hashes remain intact. Host verification confirms no remaining owned process. No packages, algorithms, queues or unrelated workloads were changed in this result check. No new experiments or notifications were sent.

To test stability, the useful next confirmation is the unchanged setting at seeds42 and1993, prioritizing3B because all its seed0 Orders improve. This is a recommendation only; no additional training was enqueued. A mechanistic or equal-compute claim would require a separately designed control.

## Files

- [English result CSV](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/G2_RESULTS_ENGLISH.csv)
- [All paired AAA/FAA/BWT/LA results](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/INDEPENDENT_COMPARISON_REVIEW.csv)
- [Completeness ledger](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/COMPLETENESS_LEDGER.json)
- [Independent integrity audit](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/INDEPENDENT_INTEGRITY_AUDIT.json)
- [Independent raw-prediction comparison](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/INDEPENDENT_COMPARISON_REVIEW.json)
- [Host completion verification](/data/140-1/users/liying/O-LoRA/refine-logs/llama_pure_g2_hf_results_20261012T084626Z/HOST_COMPLETION_VERIFIED.json)
- [Original validated Orders](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/results/orders.csv)
- [Original triangular stages](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/results/stages.csv)
- [Completed queue state](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/runtime/queue_state.json)
