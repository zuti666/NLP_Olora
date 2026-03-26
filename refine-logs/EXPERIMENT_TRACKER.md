# Experiment Tracker: GAM Tuning for OLoRA and IncLoRA (Order-1)

| Run ID | Milestone | Purpose | System / Variant | ρ | γ | Priority | Status | ACC_T | Notes |
|--------|-----------|---------|------------------|---|---|----------|--------|-------|-------|
| OA | M1 | OLoRA γ=0.1 | OLoRA-GAM | 0.10 | 0.10 | MUST | **TRAIN DONE** | — | All 4 rounds complete, awaiting eval |
| OB | M1 | OLoRA γ=0.05 | OLoRA-GAM | 0.10 | 0.05 | MUST | **RUNNING** (3-yahoo) | — | GPU3, ~17:05 done |
| IA | M2 | IncLoRA ρ↓ γ↓ | IncLoRA-GAM | 0.05 | 0.10 | MUST | **RUNNING** (1-dbpedia 29%) | — | GPU4, ~18:30 done |
| IB | M2 | IncLoRA ρ↓↓ γ↓↓ | IncLoRA-GAM | 0.02 | 0.05 | MUST | QUEUED | — | Starts after IA, ~20:00 done |
| OC | M3 | OLoRA ρ+γ co-sweep | OLoRA-GAM | 0.05 | 0.10 | NICE | TODO | — | After M1 results |
| OD | M3 | OLoRA ρ+γ co-sweep | OLoRA-GAM | 0.05 | 0.20 | NICE | TODO | — | Intermediate γ |
| IC | M3 | IncLoRA γ-only | IncLoRA-GAM | 0.10 | 0.10 | NICE | TODO | — | Isolate γ vs ρ |
| ID | M3 | IncLoRA ρ-only | IncLoRA-GAM | 0.05 | 0.30 | NICE | TODO | — | Isolate ρ vs γ |

**Baselines (already computed):**
| REF | — | OLoRA-GAM baseline | OLoRA-GAM | 0.10 | 0.30 | — | DONE | 78.03 | Current best GAM/OLoRA |
| REF | — | IncLoRA-GAM baseline | IncLoRA-GAM | 0.10 | 0.30 | — | DONE | 63.17 | Current best GAM/IncLoRA |
| REF | — | OLoRA-SAM target | OLoRA-SAM | 0.05 | — | — | DONE | 81.77 | Target to approach |
| REF | — | IncLoRA plain target | IncLoRA AdamW | — | — | — | DONE | 71.87 | Target to beat |
