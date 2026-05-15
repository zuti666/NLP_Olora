# Experiment Tracker: Order-1 Full Overview

| Run ID | Milestone | Purpose | System | Backbone | Optimizer | Key Params | Priority | Status | ACC_T | Notes |
|--------|-----------|---------|--------|----------|-----------|------------|----------|--------|-------|-------|
| S0-fix | M0 | Standardize scripts | All | — | — | backbone→T5-large, noise_std=0.05 | MUST | TODO | — | Edit scripts before any new run |
| S1-seq | M1 | SeqLoRA baseline | seqlora | T5-large | adamw_hf | — | MUST | TODO | — | Lower bound |
| S2-inc | M1 | IncLoRA baseline | Inclora | T5-large | adamw_hf | — | MUST | TODO | — | Accumulative baseline |
| S3-olo | M2 | OLoRA vanilla | olora | T5-large | adamw_hf | — | MUST | DONE | 80.54 | From existing logs |
| S4-olo-sam | M3 | OLoRA + SAM | olora | T5-large | sam-adamw_hf | rho=0.05 | MUST | DONE | 81.77 | From existing logs |
| S5-olo-gam | M3 | OLoRA + GAM (best) | olora | T5-large | gam-adamw_hf | use best from GAM study | MUST | TODO | — | Depends on EXPERIMENT_PLAN.md results |
| S6-olo-rwp | M3 | OLoRA + RWP | olora | T5-large | adamw_hf+RWP | noise_std=0.05 | MUST | TODO | — | Fix: change backbone from T5-small |
| S7-seq-sam | M3 | SeqLoRA + SAM | seqlora | T5-large | sam-adamw_hf | rho=0.05 | MUST | TODO | — | |
| S8-seq-gam | M3 | SeqLoRA + GAM (best) | seqlora | T5-large | gam-adamw_hf | use best from GAM study | MUST | TODO | — | Fix: change backbone from T5-large (already correct) |
| S9-seq-rwp | M3 | SeqLoRA + RWP | seqlora | T5-large | adamw_hf+RWP | noise_std=0.05 | MUST | TODO | — | Fix: change noise_std from 0.02 |
| S10-inc-sam | M3 | IncLoRA + SAM | Inclora | T5-large | sam-adamw_hf | rho=0.05 | MUST | TODO | — | Fix: already T5-large ✓ |
| S11-inc-gam | M3 | IncLoRA + GAM (best) | Inclora | T5-large | gam-adamw_hf | use best from GAM study | MUST | TODO | — | Fix: change backbone from T5-small |
| S12-inc-rwp | M3 | IncLoRA + RWP | Inclora | T5-large | adamw_hf+RWP | noise_std=0.05 | MUST | TODO | — | Fix: already T5-small → change to T5-large |
| S13-forgetting | M5 | Forgetting matrix | olora+seqlora | T5-large | adamw_hf+SAM | — | NICE | TODO | — | Heatmap visualization |

**Known baselines (from existing logs, Order-1, T5-large, seed=1993):**
| REF | — | OLoRA vanilla | olora | T5-large | adamw_hf | — | — | DONE | 80.54 | |
| REF | — | OLoRA + SAM | olora | T5-large | sam-adamw_hf | rho=0.05 | — | DONE | 81.77 | Best so far |
| REF | — | OLoRA + GAM (rho=0.1, γ=0.3) | olora | T5-large | gam-adamw_hf | — | — | DONE | 78.03 | Underperforms |
| REF | — | IncLoRA vanilla | Inclora | T5-large | adamw_hf | — | — | DONE | 71.87 | |
| REF | — | IncLoRA + SAM (rho=0.05) | Inclora | T5-large | sam-adamw_hf | — | — | DONE | 66.29 | SAM hurts IncLoRA |
| REF | — | IncLoRA + GAM (rho=0.1, γ=0.3) | Inclora | T5-large | gam-adamw_hf | — | — | DONE | 63.17 | Both fail |
