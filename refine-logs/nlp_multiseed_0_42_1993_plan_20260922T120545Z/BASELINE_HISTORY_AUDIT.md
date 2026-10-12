# NLP baseline multiseed history audit — 2026-09-22

Scope: SeqLoRA, IncLoRA, OLoRA+GAM (the current table entry displayed as OLoRA/olra), and TreeLoRA; Llama-3.2-1B/3B; orders 1/2/3; requested seeds 0, 42, 1993.

## Result

| Method | 1B verified historical seeds | 3B verified historical seeds | Complete orders | Missing seeds | Historical execution |
|---|---|---|---|---|---|
| SeqLoRA | 42 | 42 | 1, 2, 3 on both | 0, 1993 | 8 GPU × batch 1 × GA 8 |
| IncLoRA | 42 | 42 | 1, 2, 3 on both | 0, 1993 | 8 GPU × batch 1 × GA 8 |
| OLoRA+GAM | 42 | 42 | 1, 2, 3 on both | 0, 1993 | 8 GPU × batch 1 × GA 8 |
| TreeLoRA | 42 | 42 | 1, 2, 3 on both | 0, 1993 | 4 GPU × batch 1 × GA 16 |

All 24 selected runs have four real `predict_results.json`, `train_results.json`, and `trainer_state.json` files. All 96 stage train runtimes match the real training log. Each stage has positive training steps and runtime, and all seen datasets have prediction metrics. Seed 42 and `data_seed=None` are verified from TrainingArguments in the actual logs. No seed 0 or seed 1993 same-method/configuration result was found in the searched NLP roots.

The 48 missing method/model/seed/order sequences correspond to 192 task-training stages. Historical seed-42 SeqLoRA/IncLoRA/OLoRA+GAM have global batch 64 but DDP8/GA8, so they must be labelled as historical execution if combined with new DDP4/GA16 runs. A strict single-execution-protocol three-seed study needs 18 additional seed-42 sequences (72 task stages) for these three methods; existing TreeLoRA seed-42 sequences already use DDP4/GA16. This is a protocol distinction, not missing historical data.

## Sources and exclusions

- Selected source/config index: `figures/llama_taskwise_orders_20260920/combined_v6/run_sources.csv` and `protocol_configs.csv`. JSON companion to this report contains every absolute metric/log path, SHA256, parsed arguments and per-stage evidence.
- Main historical outputs: `logs_and_outputs_llama_1B/order_{1,2,3}` and `logs_and_outputs_llama_3B/order_{1,2,3}`.
- Main historical logs: `Logs_Scripts_Short_Paper/Llama-3.2-{1B,3B}_AllOrders_SeqLoRA` and `..._AllOrders_IncLoRA_OLoRA`.
- 54 relevant historical method/optimizer-variant logs were parsed and all record seed 42. SAM/RWP and SeqLoRA-GAM/IncLoRA-GAM are different methods, not additional seeds of the table's selected baselines.
- TreeLoRA campaign: `experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z`; manifest and all 24 full-stage training logs agree. Smoke outputs have only three stages and do not count as complete experiments.
- `experiment_queue/direct_1b_seed1993` has cache files only, with no training/prediction results.
- `logs_and_outputs_llama_1B/full_gpu6_20260811_seed42` contains SAM/FisherShape variants, not plain IncLoRA, and only order 1.
- CopyFlatCV archive: `/data/140-1/users/liying/CopyFlatCV/GAM-EWC/NLP_O-LoRA/results/inventory.csv`. 42 relevant Llama baseline/variant inventory rows refer to existing source runs; all archived prediction files checked were byte-identical to those sources. They are copies, not independent seed runs.
- `Logs_RedoOlora/run_logs/*llama*.log` are supervisors pointing to historical scripts/logs, not additional seed evidence.

The latest selected baseline prediction timestamp is 2026-09-20 12:31:27 UTC (TreeLoRA). No relevant non-smoke candidate prediction file with timestamp on/after 2026-09-21 was found under the two Llama output roots and experiment_queue. This is a bounded local-history audit; no claim is made about unmounted/remote disks. Visual and T5 experiment contents were not audited.
