# Independent short review of NLP multiseed plan

Date: 2026-09-22

Reviewed `EXPERIMENT_PLAN.md`, `SEED_ORDER_LEDGER.csv`, and `RUN_QUEUE_DDP4.csv` in `refine-logs/nlp_multiseed_0_42_1993_plan_20260922T120545Z`.

Verdict: **PASS for plan arithmetic, coverage, queue consistency, and method identity. No blocking error found within the requested review scope.** This is a plan review, not a declaration that training implementation or the OLoRA+GAM scaling proposal has passed runtime validation.

## Deterministic checks

- Ledger equals the full Cartesian product of 2 exact Llama models × 6 specified methods × seeds {0,42,1993} × orders {1,2,3}: 108 unique targets, no duplicate IDs or output paths.
- Exactly 36 targets have historical references; each listed directory and final metric file exists.
- Actions: 12 `REUSE_DDP4`, 72 `MISSING_SEED`, 24 `RERUN_DDP8_REFERENCE`. Thus 96 new sequences plus 12 reusable sequences yield the 108-target study.
- All 12 direct-reuse rows have historical DDP4/GA16; all 24 migration rows have DDP8/GA8. The minimum-supplement alternative has exactly 72 new rows and 36 historical rows.
- Queue has 96 unique rows with contiguous positions 1–96. Its keys are exactly the ledger excluding the 12 reusable targets. Every shared field matches its ledger row; no omission or extra run.
- New queue counts are SeqLoRA 18, IncLoRA 18, OLoRA+GAM 18, TreeLoRA 12, EWCLoRA 15, FS-EWCLoRA 15. The described phases sum 30 + 30 + 36 = 96.
- Every target uses GPU `3,4,5,6`, world size 4, batch 1, GA 16, global batch 64, and explicit `seed=data_seed`. The first four queue rows match the declared calibration order.
- Stage totals are consistent: 108×4=432 total, 36×4=144 historical, 12×4=48 directly reused, 72×4=288 minimum new, 24×4=96 migrations, 96×4=384 recommended new.
- Method field remains `OLoRA+GAM`; output paths use `olora_gam`, strategy is `olora`, and the semantic gate requires GA regularizer scaling and GAM replay validation. No pure OLoRA, SAM/FisherShape, T5, or visual experiment row appears in either CSV.
- EWC target lambda is 0.5; FS target lambda is 1 with OL regularizer zero; OLoRA+GAM 0.25 is expressly provisional and gated, not presented as a validated runtime setting.

## Nonblocking conditions already documented in the plan

The 12 historical reuse rows are conditional on frozen-input and method-implementation compatibility. GPU availability must be rechecked at execution. OLoRA+GAM GA/closure scaling and generalized launch/audit code must pass M0 before formal runs. No change to these conditions is needed for a planning deliverable.
