# Experiment Plan: GAM Hyperparameter Tuning for OLoRA and IncLoRA (Order-1)

**Problem**: GAM underperforms SAM and even plain AdamW for both OLoRA and IncLoRA on the 4-task continual learning Order-1 benchmark. The question is whether better GAM hyperparameters can recover this gap.
**Method Thesis**: The gamma (gradient smoothing weight) in GAM is too aggressive at γ=0.3, reducing plasticity for OLoRA and causing forgetting for IncLoRA; reducing γ and possibly ρ should recover performance.
**Date**: 2026-03-25

---

## Baseline Results (Order-1_8, T5-large, seed=1993)

| Method | ACC_T | dbpedia | amazon | yahoo | agnews | Notes |
|--------|-------|---------|--------|-------|--------|-------|
| OLoRA-SAM (rho=0.05) | **81.77** | 96.88 | 76.88 | 75.03 | 78.31 | target |
| OLoRA plain | 80.54 | 98.34 | 74.87 | 72.56 | 76.41 | |
| **OLoRA-GAM** (rho=0.1, γ=0.3) | 78.03 | 97.37 | **55.51** | 71.28 | 87.95 | amazon bottleneck |
| IncLoRA plain | **71.87** | 87.46 | 63.97 | 64.93 | 71.13 | target |
| IncLoRA-SAM (rho=0.05) | 66.29 | 72.71 | 59.88 | 62.79 | 69.76 | SAM hurts |
| **IncLoRA-GAM** (rho=0.1, γ=0.3) | 63.17 | 60.74 | 42.93 | 59.87 | 89.13 | both retention+plasticity fail |

Current GAM config (both methods): rho=0.1, norm_rho=0.01, beta_1=0.05, beta_2=1.0, beta_3=1.0, gamma=0.3, adaptive=false

---

## Diagnosis

### OLoRA-GAM failure mode: over-smoothed gradient → low plasticity
- dbpedia retention: **97.37** (excellent, near-perfect)
- amazon plasticity: **55.51** vs 74.87 (plain), 76.88 (SAM) → −21 points
- GAM with γ=0.3 over-regularizes sharpness → the "smooth loss landscape" hurts task-switching (amazon after dbpedia is a harder distribution shift)
- Fix direction: **reduce γ** to restore plasticity while keeping flatness benefit

### IncLoRA-GAM failure mode: excessive perturbation destroys accumulated LoRA
- dbpedia final: **60.74** vs 87.46 (plain) → −27 points
- amazon final: **42.93** vs 63.97 (plain) → −21 points
- IncLoRA grows the adapter incrementally; GAM's gradient perturbation during task k disrupts the already-learned task 1..k-1 adapter weights
- Fix direction: **reduce ρ and γ** together to keep perturbation sub-threshold for the accumulated adapter

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
|-------|---------------|----------------------------|--------|
| C1: GAM with lower γ matches SAM on OLoRA | Shows GAM is a viable flat-landscape optimizer for grow-and-freeze CL | OLoRA-GAM ACC_T ≥ 80.0 (within 2 pts of SAM's 81.77) | B1, B2 |
| C2: GAM with reduced ρ+γ recovers IncLoRA retention | Shows GAM applicability to accumulative LoRA given mild enough perturbation | IncLoRA-GAM ACC_T ≥ 68.0 (within 4 pts of plain's 71.87) | B3, B4 |

Anti-claim to rule out: "γ=0.3 is not the bottleneck; the problem is something else (e.g. beta weights, rho)."
→ Falsified if lower-γ configs at same ρ show clear improvement.

---

## Paper Storyline

**Main paper must prove:**
- GAM achieves competitive results with SAM on OLoRA when γ is properly tuned (C1)
- GAM can recover IncLoRA CL performance with gentler perturbation (C2)

**Appendix can support:**
- Gamma ablation curve showing the transition region
- Full forgetting matrices across all checkpoints

**Experiments intentionally cut:**
- Multi-seed variance (only seed=1993 for now, given GPU budget)
- Order-2, Order-3 extension (separate future work)
- beta_1 / beta_2 / beta_3 ablation (already explored in SeqLoRA experiments; beta_2=1.0 confirmed best)

---

## Experiment Blocks

### Block 1: OLoRA-GAM gamma sweep (MUST-RUN)

- **Claim tested**: C1 — lower γ restores OLoRA plasticity
- **Why this block exists**: Current γ=0.3 causes amazon to drop to 55.51. Need to find the γ where flatness benefit and plasticity are balanced.
- **Dataset**: Order-1 (dbpedia→amazon→yahoo→agnews), T5-large
- **Compared systems**:
  - OLoRA-GAM γ=0.1 (Exp OA) — primary candidate
  - OLoRA-GAM γ=0.05 (Exp OB) — aggressive reduction
  - Baseline: OLoRA-GAM γ=0.3 (existing, 78.03) and OLoRA-SAM (81.77)
- **Metrics**: ACC_T (final row mean), full forgetting matrix
- **Setup**: rho=0.1, norm_rho=0.01, beta_1=0.05, beta_2=1.0, beta_3=1.0, adaptive=false; single GPU, 1 epoch/task, lr=1e-3, bs=8, grad_accum=2
- **Success criterion**: ACC_T ≥ 80.0 on at least one config → C1 supported
- **Failure interpretation**: If both configs remain ≤ 78, the bottleneck is not γ alone; may need to also vary ρ or beta_1
- **Table target**: Main comparison table (OLoRA block)
- **Priority**: MUST-RUN

### Block 2: OLoRA-GAM rho + gamma co-sweep (NICE-TO-HAVE)

- **Claim tested**: C1 extension — whether reducing ρ alongside γ helps further
- **Why this block exists**: If B1 shows γ reduction helps but not enough, ρ reduction might close the remaining gap
- **Dataset**: Same as B1
- **Compared systems**:
  - OLoRA-GAM ρ=0.05, γ=0.1 (Exp OC)
  - OLoRA-GAM ρ=0.05, γ=0.2 (Exp OD)
- **Setup**: Same as B1 but ρ=0.05
- **Success criterion**: ACC_T > best result from B1
- **Failure interpretation**: If ρ reduction doesn't help beyond γ reduction alone, γ is the dominant knob
- **Table target**: Appendix ablation
- **Priority**: NICE-TO-HAVE

### Block 3: IncLoRA-GAM perturbation reduction (MUST-RUN)

- **Claim tested**: C2 — gentler ρ+γ recovers IncLoRA retention
- **Why this block exists**: Current IncLoRA-GAM loses 27 pts on dbpedia. Need to test if reducing perturbation scope prevents this.
- **Dataset**: Same Order-1
- **Compared systems**:
  - IncLoRA-GAM ρ=0.05, γ=0.1 (Exp IA) — moderate reduction
  - IncLoRA-GAM ρ=0.02, γ=0.05 (Exp IB) — strong reduction
  - Baseline: IncLoRA-GAM ρ=0.1, γ=0.3 (63.17) and IncLoRA plain (71.87)
- **Metrics**: ACC_T, dbpedia final retention (key diagnostic)
- **Setup**: Same as B1 but lora_strategy=Inclora
- **Success criterion**: ACC_T ≥ 68.0 AND dbpedia final ≥ 75.0 on at least one config
- **Failure interpretation**: If dbpedia retention stays below 75 even at ρ=0.02, the failure is structural (IncLoRA's accumulated adapter is fundamentally incompatible with gradient perturbation at any scale)
- **Table target**: Main comparison table (IncLoRA block)
- **Priority**: MUST-RUN

### Block 4: IncLoRA-GAM rho-only ablation (NICE-TO-HAVE)

- **Claim tested**: C2 — isolate whether ρ or γ matters more for IncLoRA
- **Why this block exists**: Disentangle the contribution of each hyperparameter to the IncLoRA forgetting problem
- **Dataset**: Same
- **Compared systems**:
  - IncLoRA-GAM ρ=0.1, γ=0.1 (Exp IC) — reduce only γ, keep ρ=0.1
  - IncLoRA-GAM ρ=0.05, γ=0.3 (Exp ID) — reduce only ρ, keep γ=0.3
- **Success criterion**: Clear separation between IC and ID results identifies the dominant factor
- **Table target**: Appendix
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | GPU Cost | Risk |
|-----------|------|------|---------------|----------|------|
| M0: Sanity | Verify OA trains and logs look correct | OA round 1 only | loss < 0.15 at round 1; no OOM | ~20 min, 1 GPU | Low |
| M1: OLoRA γ sweep | Confirm γ reduction helps OLoRA | OA, OB | ACC_T > 78.03 on at least one | ~80 min each, 1 GPU | Medium |
| M2: IncLoRA ρ+γ reduction | Confirm retention recovers | IA, IB | ACC_T > 63.17 AND dbpedia > 70 | ~80 min each, 1 GPU | Medium-High |
| M3: Co-sweep | Find global optimum | OC, OD, IC, ID | Best > M1/M2 results | ~80 min each, 1 GPU | Low |

**Run order (GPU 3 → GPU 4 parallel):**
```
GPU 3: OA → OB → OC (sequential)
GPU 4: IA → IB → IC (sequential, start after M0 gate passes)
```

Eval after all training: run_eval_new_gam_exps.sh (batch, 4-parallel single-GPU eval)

---

## Compute and Data Budget

- Training: 6 MUST-RUN × 4 rounds × ~20 min/round = ~8 GPU-hours
- Eval: 6 models × 16 cells × ~5 min/cell = ~8 GPU-hours (parallelized to ~2 hours)
- Total: ~10 GPU-hours, ~5 hours wall-clock with 2 parallel GPUs
- No new data needed; all uses existing CL_Benchmark

---

## Risks and Mitigations

- **Risk**: γ reduction alone is insufficient for OLoRA (amazon stays low)
  → **Mitigation**: B2 provides ρ+γ co-sweep; if both fail, revisit beta_1 tuning
- **Risk**: IncLoRA is fundamentally incompatible with gradient perturbation
  → **Mitigation**: IB uses ρ=0.02 which is very gentle; if this still breaks retention, conclude that GAM is not suitable for IncLoRA and report the negative result
- **Risk**: GPU 4 becomes busy before IncLoRA experiments start
  → **Mitigation**: Run IA sequentially on GPU 3 after OA if GPU 4 is taken

---

## Final Checklist

- [x] Main paper tables are covered (OLoRA block + IncLoRA block)
- [x] Novelty is isolated (γ as the key knob, not just "different optimizer")
- [x] Simplicity is defended (same GAM, only 1-2 hyperparameters changed)
- [ ] Frontier contribution is N/A (this is optimizer tuning, not frontier-model work)
- [x] Nice-to-have runs (B2, B4) are separated from must-run (B1, B3)

---

## Experiment Configs

### MUST-RUN (4 experiments)

| Exp | Method | ρ | norm_ρ | γ | β₁ | β₂ | β₃ | Output dir suffix |
|-----|--------|---|--------|---|----|----|----|-------------------|
| OA | OLoRA-GAM | 0.10 | 0.01 | **0.10** | 0.05 | 1.0 | 1.0 | OLoRA-GAM_r8_bs16_e1_lr1e-3_Rho0p10_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993 |
| OB | OLoRA-GAM | 0.10 | 0.01 | **0.05** | 0.05 | 1.0 | 1.0 | OLoRA-GAM_r8_bs16_e1_lr1e-3_Rho0p10_NRho0p01_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993 |
| IA | IncLoRA-GAM | **0.05** | 0.01 | **0.10** | 0.05 | 1.0 | 1.0 | IncLoRA-GAM_r8_bs16_e1_lr1e-3_Rho0p05_NRho0p01_G0p10_B1_0p05_B2_1p00_B3_1p00_seed-1993 |
| IB | IncLoRA-GAM | **0.02** | 0.01 | **0.05** | 0.05 | 1.0 | 1.0 | IncLoRA-GAM_r8_bs16_e1_lr1e-3_Rho0p02_NRho0p01_G0p05_B1_0p05_B2_1p00_B3_1p00_seed-1993 |

### NICE-TO-HAVE (4 experiments)

| Exp | Method | ρ | norm_ρ | γ | Note |
|-----|--------|---|--------|---|------|
| OC | OLoRA-GAM | 0.05 | 0.01 | 0.10 | reduce ρ too |
| OD | OLoRA-GAM | 0.05 | 0.01 | 0.20 | intermediate γ |
| IC | IncLoRA-GAM | 0.10 | 0.01 | 0.10 | γ-only reduction |
| ID | IncLoRA-GAM | 0.05 | 0.01 | 0.30 | ρ-only reduction |
