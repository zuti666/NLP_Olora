# Independent NLP table integrity review

**Overall verdict: FAIL for a controlled five-method comparison; PASS for numerical transcription/reproducibility from the located Llama artifacts.**  
`review_independence=same-family`  
`acceptance_status=provisional`

Scope was limited to the supplied Llama-3.2-1B/3B table and existence/configuration comparability of the inventoried T5-small/T5-large results. I treated `final_results.csv` only as a locator and read primary predictions, metric JSON, training state, adapter configuration, source, manifests, and console logs. I did not use executor or other-agent summaries.

The audit was exhaustive for the requested Llama cells: 30 order cells, 30 metric files, and 912,000 prediction rows. Every row's embedded label, ground truth, dataset, ID, and sentence matched the current four `CL_Benchmark` test files; independently recomputed normalized exact match equaled each stored metric to four decimals. All ten displayed averages equal the arithmetic mean of the three orders and round to the supplied value. The supplied values are at `refine-logs/nlp_table_audit_20260919/expected_table.json:13-83`.

| Model | Method | Recomputed Order-1 / Order-2 / Order-3 | Mean | Supplied values match |
|---|---|---:|---:|---|
| 1B | SeqLoRA | 66.9013 / 55.7105 / 68.0625 | 63.5581 | yes |
| 1B | IncLoRA | 64.5230 / 56.1941 / 65.4441 | 62.0537 | yes |
| 1B | OLoRA | 58.6053 / 53.1184 / 58.4211 | 56.7149 | yes |
| 1B | IncLoRA+EWC | 67.1941 / 61.9178 / 66.5362 | 65.2160 | yes |
| 1B | IncLoRA-GAM-EWC | 70.8355 / 70.0592 / 69.7007 | 70.1985 | yes |
| 3B | SeqLoRA | 67.8553 / 67.2632 / 67.4046 | 67.5077 | yes |
| 3B | IncLoRA | 65.2303 / 66.7434 / 69.5493 | 67.1743 | yes |
| 3B | OLoRA | 63.3651 / 62.9737 / 63.9243 | 63.4210 | yes |
| 3B | IncLoRA+EWC | 68.3586 / 69.1776 / 73.0395 | 70.1919 | yes |
| 3B | IncLoRA-GAM-EWC | 72.7138 / 73.5132 / 72.6612 | 72.9627 | yes |

Representative primary evidence includes `logs_and_outputs_llama_1B/order_1/SeqLoRA_Llama-3.2-1B_torch_AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4/4-agnews/predict_results.json:3-9,28`, `logs_and_outputs_llama_3B/order_1/EWCLoRA-GAM_Llama-3.2-3B_torch_GAM-AdamwHF_Lora8_Ba1-GA8_E1_LR1e-3_to_1e-4_lambda1.0_gamma1.0_seed-1993/4-agnews/predict_results.json:3-9,28`, and `experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_2/4-yahoo/predict_results.json:3-9,28`. The machine report contains the exact three source paths for every row.

## A-F checks

**A — PASS: dataset ground-truth provenance; custom metric.** Test examples are loaded from dataset JSON and label files (`experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/source/src/uie_dataset_lora.py:287-294,320-346,349-373,555-570`). References come from `Instance.label`, not model output (`.../source/src/run_uie_lora.py:1393-1404`). This is real dataset ground truth. The evaluator is a project-defined normalized generative exact match, not an official dataset package metric (`.../source/src/compute_metrics.py:31-49,78-97`). **Claim impact:** describing the score as custom normalized label exact match/accuracy is supported.

**B — PASS: denominator and raw-score validity.** The code divides matches by `len(references)` and multiplies by 100 (`.../source/src/compute_metrics.py:78-97`). Every Llama cell had 30,400 rows, four groups of 7,600, no duplicate self-normalization, and an exact four-decimal recomputation match. The new EWC verifier independently records 30,400 samples, group counts, and unrounded match counts (`experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_20260918T183137Z/full/order_2/4-yahoo/verification.json:18-48,50-65`).

**C — FAIL overall: Llama artifacts pass; T5-large coverage fails.** All 120 Llama stages had finite positive training records, positive steps, adapters, and model identity; an example is `.../SeqLoRA.../1-dbpedia/train_results.json:2-7`, `.../trainer_state.json:4-9`, and `.../adapter/adapter_config.json:2,11-16`. The new 1B EWC cohort is marked completed for all 12 full stages (`experiment_queue/.../state.json:2,38-52,92-97`). For T5, all 31 inventoried directories and 124 stages had usable training/state/adapter artifacts, and 942,400 selected evaluation rows recomputed consistently. However, the directory labelled T5-large IncLoRA Order-1 declares `t5-small` (`Logs_Short_Paper/Order-1_8/IncLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_redo1_seed-1993/1-dbpedia/adapter/adapter_config.json:2`), and its four final prediction files are byte-identical to the T5-small counterpart. Thus actual T5-large coverage is 14/15 method-order combinations. **Claim impact:** a complete T5-large five-method/three-order result set is unsupported.

**D — WARN: metric reachability proven only for the frozen new cohort.** The frozen runner selects the custom metric and passes it to `UIETrainer` (`.../source/src/run_uie_lora.py:1517-1531`); prediction calls and persists metrics (`.../source/src/run_uie_lora.py:1651-1686`); the trainer invokes the callback (`.../source/src/uie_trainer_lora.py:4438-4463`). Historical runs lack frozen executed-source identity. Current source hashes equal the new snapshot, but I do not extrapolate that fact to older runs. Raw-output recomputation still validates their reported numbers.

**E — FAIL: protocol and seed comparability.** Historical controls use bf16, 8 workers, batch 1, GA 8, global batch 64, and seed 42 (`Logs_Scripts_Short_Paper/Llama-3.2-1B_AllOrders_SeqLoRA/scripts_order1_llama1B/SeqLoRA_LLaMA2-1b.log:10-13,61-63,151,178`). Llama-1B GAM-EWC uses 5 workers, GA 8, global batch 40, and seed/data-seed 1993 (`Logs_EWCLoRA_GAM/order1_llama1B_ewclora_gam_seed1993_20260516_003719_tmux.log:96-103`). Llama-3B EWC/GAM use seed 1993 while controls use 42 (`Logs_Scripts_Short_Paper/Llama-3.2-3B_AllOrders_EWCLoRA/scripts_order1_llama3B/EWCLoRA_LLaMA2-3b.log:86-91`). The replacement 1B EWC uses seed 42/global 64 but DDP4, GA16, lambda 0.5, and a later frozen source (`experiment_queue/.../manifest.json:3-27,132-148`). Each table entry is one seed; “Avg.” averages orders, not seeds. T5 logs substantiate some nominal settings—e.g. SeqLoRA large Order-1 at `Logs_RedoOlora/run_logs/seqlora_all_orders_20260327_074758.log:40565` and EWC-large batch-8 at `Logs_Short_Paper/run_logs/run_all_ewc_t5large_ba8_seed-1993_20260518_172349.log:21-31,49516`—but launch/source evidence is unavailable for the full T5 inventory. **Claim impact:** score ordering is descriptive; causal method-superiority or robustness claims are unsupported.

**F — PASS: `evaluation_type=real_gt`.** Predictions are generated from the held-out test split (`.../source/src/uie_dataset_lora.py:271-281`; `.../source/src/run_uie_lora.py:1329-1337`) and compared with dataset labels. This is not a synthetic, self-supervised, simulated, or human evaluation.

## Required actions

1. Run the missing true T5-large IncLoRA Order-1 and exclude the mislabeled artifact.
2. Re-run comparisons from one frozen source with identical checkpoint hashes, precision, global batch, DDP/GA semantics, and shared seeds (preferably at least three); retain launch, source, dataset, and environment manifests.
3. Select and document one T5-large EWC Order-1 batch variant and use one evaluation pipeline.
4. Report mean and uncertainty across seeds separately from the mean across task orders, and name the custom normalized exact-match metric.

## Input integrity and limitations

SHA-256: supplied table `8cc14ae2c9243e5ebd9e7b085a87178a0bd4b886ac0fd06c6f02b64fa555636f`; path inventory `9646d939224a91767c7af95d470a76515b4cd1b56e95a68fa29c0656df1a1941`; 60 selected Llama prediction/metric files manifest digest `7621017999f12d15daa566855b576662d30f32571737ae8700efb6b351837022`; 170 selected T5 prediction/metric files digest `3f83f085dc432c931b66e1fa5892b1ad3078e58e3b887d21e2b1d808229b2e57`; 8 audited dataset files digest `cacd1f951f2f18d8d4d28865cbc6986e132ca5b36c7ab7093d50de073eeff620`; frozen source digest `cf7cd8a91b39a031fdaeaade9c759a7404d8e06191e13a3780c281dd4a02ef39`.

No Llama table-cell or selected T5 final-score check was sampled. Binary adapter weights/Fisher tensors were checked for existence/nonzero size and finite-verification metadata but were not independently re-executed or hashed. Historical executed source revisions, complete historical environment manifests, and launch settings for several T5 methods are unavailable; those gaps remain explicit rather than inferred from directory names.
