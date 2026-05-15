# Experiment Plan: Order-1 Full Overview
# Flat-Minima Regularization for Continual LoRA (CL-LoRA)

**Problem**: Catastrophic forgetting in sequential LoRA fine-tuning — each new task overwrites representations learned by prior tasks.
**Method Thesis**: Flat-minima regularization (SAM / GAM / RWP) applied on top of different continual LoRA strategies (OLoRA / SeqLoRA / IncLoRA) improves backward transfer and reduces forgetting, because flatter loss basins are more robust to the weight drift induced by subsequent tasks.
**Date**: 2026-04-01

---

## Experiment Matrix (scripts_order1)

**Task Sequence (Order 1)**: dbpedia → amazon → yahoo → agnews  
**Data**: CL_Benchmark, configs/order1_configs/  
**Entry point**: `src/run_uie_lora.py` via torchrun (4 GPUs)  
**Common fixed hyperparameters**: LoRA rank=8, LR=1e-3, constant LR schedule, 1 epoch/task, batch=16, grad_accum=1, BF16, seed=1993

| Script | LoRA Strategy | Optimizer | Key Params | Model |
|--------|--------------|-----------|------------|-------|
| OLoRA.sh | olora | adamw_hf | — | T5-large |
| OLoRA_GAM.sh | olora | gam-adamw_hf | rho=0.05, norm_rho=0.2, gamma=0.03 | T5-small |
| OLoRA_SAM.sh | olora | sam-adamw_hf | sam_rho=0.05 | T5-large |
| OLoRA_RWPlora_Gaussian.sh | olora | adamw_hf + RWP | rwp_type=lora, noise_std=0.05, Gauss | T5-small |
| SeqLoRA_E1_Le3.sh | seqlora | adamw_hf | — | T5-small |
| SeqLoRA-GAM_E1_Le3.sh | seqlora | gam-adamw_hf | rho=0.02, norm_rho=0.2, gamma=0.03 | T5-large |
| SeqLoRA-SAM_E1_Le3.sh | seqlora | sam-adamw_hf | sam_rho=0.05 | T5-small |
| SeqLoRA_E5_Le3_RWPlora-Gaussian.sh | seqlora | adamw_hf + RWP | rwp_type=lora, noise_std=0.02, Gauss | T5-large |
| IncLoRA_E1_Le3.sh | Inclora | adamw_hf | — | T5-small |
| IncLoRA-GAM_E1_Le3.sh | Inclora | gam-adamw_hf | rho=0.05, norm_rho=0.2, gamma=0.03 | T5-small |
| IncLoRA-SAM_E1_Le3.sh | Inclora | sam-adamw_hf | sam_rho=0.05 | T5-large |
| IncLoRA_RWPlora-Gaussian_S1000.sh | Inclora | adamw_hf + RWP | rwp_type=lora, noise_std=0.05, Gauss | T5-small |

### LoRA Strategy Definitions

| Strategy | Description | Role |
|----------|-------------|------|
| **olora** | O-LoRA: orthogonal subspace continual learning; each task's LoRA update is constrained to be orthogonal to prior tasks' subspace | **Proposed method** |
| **seqlora** | Sequential LoRA: naive continual fine-tuning, each task inherits the prior adapter and trains freely | Catastrophic forgetting baseline |
| **Inclora** | Incremental LoRA: grows the adapter incrementally per task, accumulates weights across tasks | Accumulative adapter baseline |

### Flat-Minima Regularization Variants

| Optimizer | Mechanism | Hypothesis |
|-----------|-----------|------------|
| AdamW (baseline) | Standard adaptive gradient | No landscape regularization |
| + GAM | Gradient-based sharpness-aware minimization; modulates gradient with curvature signal | Wide minima → better cross-task weight sharing |
| + SAM | Sharpness-Aware Minimization; perturbs weights by rho in gradient direction, minimizes worst-case loss | Flat basin → more stable adapter across task boundaries |
| + RWP | Random Weight Perturbation on LoRA weights only; Gaussian noise injection during training | Data-augmentation-style flatness; low-cost alternative |

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|---------------|----------------------------|---------------|
| **C1**: Flat-minima regularization reduces catastrophic forgetting in continual LoRA | Core paper claim — connects loss landscape flatness to CL | At least one flat-minima variant improves ACC_T over vanilla baseline for each LoRA strategy | B1, B2 |
| **C2**: O-LoRA (orthogonal subspace) + flat-minima is complementary and superior to either alone | Shows the two regularization axes are orthogonal and additive | OLoRA+best_flat > OLoRA_vanilla AND > SeqLoRA+best_flat | B1, B3 |
| **C3**: Among flat-minima methods, SAM > GAM > RWP for CL settings | Informs practical optimizer choice for CL-LoRA | Clear ranking across all 3 LoRA strategies; no contradiction | B2 |
| **Anti-claim to rule out** | Gain comes from T5-large vs T5-small, not the method | All methods must be evaluated on same model backbone | B1 |

> **⚠️ Model Size Inconsistency Alert**: Current scripts use T5-large and T5-small inconsistently across methods and variants. This confounds comparisons. See Section "Issues and Fixes" below.

---

## Paper Storyline

**Main paper must prove:**
- OLoRA (vanilla) outperforms SeqLoRA and IncLoRA (vanilla) on ACC_T (orthogonal subspace is effective)
- At least one flat-minima variant (SAM preferred) consistently improves over vanilla within each LoRA strategy
- OLoRA + SAM achieves best overall ACC_T among all combinations

**Appendix can support:**
- Per-task forgetting matrices for each method × optimizer
- Ablation on RWP noise std and SAM rho
- Order-2, Order-3 results (separate scripts exist)
- GAM hyperparameter sensitivity (covered in EXPERIMENT_PLAN.md)

**Experiments intentionally cut:**
- T5-base backbone (resource constraint)
- Multi-seed variance (budget constraint; add 2 extra seeds for final camera-ready if accepted)
- LoRA rank ablation (rank=8 is standard)

---

## Experiment Blocks

### Block 1: Main Comparison Table — OLoRA vs Baselines (MUST-RUN)

- **Claim tested**: C1, C2
- **Why this block exists**: Establishes that OLoRA outperforms sequential and incremental LoRA; the central table of the paper.
- **Dataset**: Order-1 (dbpedia→amazon→yahoo→agnews)
- **Compared systems**:
  - SeqLoRA + AdamW (catastrophic forgetting lower bound)
  - IncLoRA + AdamW
  - OLoRA + AdamW (proposed, no flat-minima)
  - OLoRA + best flat-minima variant (from B2)
- **Metrics**: ACC_T (mean accuracy across all tasks at end of training), BWT (backward transfer), forward accuracy per task
- **Setup**: **Must use same backbone (T5-large) for all systems**; 1 epoch/task, bs=16, seed=1993
- **Success criterion**: OLoRA_vanilla > SeqLoRA_vanilla > IncLoRA_vanilla on ACC_T; OLoRA+flat-minima ≥ OLoRA_vanilla + 1.5 pts
- **Failure interpretation**: If OLoRA ≤ SeqLoRA, the orthogonal subspace constraint may be over-restricting plasticity for short task sequences
- **Table target**: Table 1 (main results)
- **Priority**: MUST-RUN

### Block 2: Flat-Minima Optimizer Comparison per Strategy (MUST-RUN)

- **Claim tested**: C1, C3
- **Why this block exists**: Determines whether SAM, GAM, or RWP most effectively reduces forgetting, and whether the benefit is universal across CL strategies.
- **Dataset**: Order-1
- **Compared systems** (per LoRA strategy):
  - vanilla AdamW (baseline)
  - + SAM (rho=0.05)
  - + GAM (current best config from EXPERIMENT_PLAN.md)
  - + RWP-Gaussian (noise_std=0.05)
- **Metrics**: ACC_T, BWT
- **Setup**: All on **same T5-large backbone**; fix other hyperparameters
- **Success criterion**: Consistent ordering across strategies (ideally SAM > GAM > RWP > vanilla); ACC_T improvement ≥ 1 pt for at least 2/3 strategies
- **Failure interpretation**: If ranking is inconsistent, no universal recommendation can be made; paper must qualify claims per strategy
- **Table target**: Table 1 (extended) or Table 2 (optimizer ablation)
- **Priority**: MUST-RUN

### Block 3: Novelty Isolation — Is Orthogonality Necessary? (MUST-RUN)

- **Claim tested**: C2 — the orthogonal subspace is the key differentiator, not just "different adapter init"
- **Why this block exists**: A reviewer will ask: "isn't OLoRA just SeqLoRA with a different initialization or projection?" This block falsifies that.
- **Dataset**: Order-1
- **Compared systems**:
  - SeqLoRA + SAM (best flat-minima on a non-orthogonal strategy)
  - OLoRA + AdamW (orthogonality without flat-minima)
  - OLoRA + SAM (both contributions)
- **Metrics**: ACC_T, per-task forgetting (especially task 1 retention after 4 tasks)
- **Success criterion**: OLoRA+AdamW > SeqLoRA+SAM, showing orthogonality alone beats a strong regularized baseline
- **Failure interpretation**: If SeqLoRA+SAM ≥ OLoRA+AdamW, the orthogonal constraint is not adding value beyond landscape regularization; need to revisit OLoRA design
- **Table target**: Table 2 (ablation) or dedicated ablation figure
- **Priority**: MUST-RUN

### Block 4: Simplicity Check — Is GAM Over-Complex? (NICE-TO-HAVE)

- **Claim tested**: Anti-complexity — SAM achieves comparable benefit with fewer hyperparameters than GAM
- **Why this block exists**: GAM has 5+ hyperparameters vs SAM's 1; if SAM matches GAM, recommend SAM as default
- **Dataset**: Order-1
- **Compared systems**: OLoRA+SAM vs OLoRA+GAM(best config) vs OLoRA+RWP
- **Metrics**: ACC_T, hyperparameter sensitivity (variance across configs)
- **Success criterion**: |SAM - GAM_best| < 1 pt on ACC_T, supporting the simpler SAM recommendation
- **Table target**: Appendix or footnote recommendation
- **Priority**: NICE-TO-HAVE

### Block 5: Qualitative Forgetting Analysis (NICE-TO-HAVE)

- **Claim tested**: C1 — flat-minima methods reduce per-task forgetting, not just average accuracy
- **Why this block exists**: ACC_T can mask heterogeneous forgetting patterns; forgetting matrix shows the full picture
- **Dataset**: Order-1
- **Analysis**: Full 4×4 forgetting matrix for OLoRA+AdamW, OLoRA+SAM, SeqLoRA+AdamW
- **Metrics**: R_{i,j} = accuracy on task i after training task j; BWT per task
- **Figure target**: Heatmap figure in main paper or appendix
- **Priority**: NICE-TO-HAVE (but generates a strong visualization)

---

## ⚠️ Issues and Required Fixes Before Running

### Issue 1: Inconsistent Model Backbone Across Scripts

The current scripts mix T5-small and T5-large within the same method family, making direct comparison invalid:

| Method × Variant | Current Model | Required Fix |
|------------------|---------------|--------------|
| OLoRA (vanilla) | T5-large ✓ | — |
| OLoRA + GAM | T5-small ❌ | Change to T5-large |
| OLoRA + SAM | T5-large ✓ | — |
| OLoRA + RWP | T5-small ❌ | Change to T5-large |
| SeqLoRA (vanilla) | T5-small | Choose one backbone for all |
| SeqLoRA + GAM | T5-large ❌ | Inconsistent |
| SeqLoRA + SAM | T5-small | — |
| SeqLoRA + RWP | T5-large ❌ | Inconsistent |
| IncLoRA (vanilla) | T5-small | — |
| IncLoRA + GAM | T5-small ✓ | — |
| IncLoRA + SAM | T5-large ❌ | Inconsistent |
| IncLoRA + RWP | T5-small ✓ | — |

**Recommended fix**: Standardize on **T5-large** for all methods in the main table. Run T5-small as a separate appendix scale study if needed.

### Issue 2: Inconsistent RWP Noise std

- OLoRA+RWP and IncLoRA+RWP: `noise_std=0.05`
- SeqLoRA+RWP: `noise_std=0.02`

**Recommended fix**: Standardize to `noise_std=0.05` across all methods for fair comparison.

### Issue 3: Inconsistent GAM rho

- OLoRA+GAM and IncLoRA+GAM: `rho=0.05` (current scripts)
- SeqLoRA+GAM: `rho=0.02`

**Recommended fix**: Standardize to `rho=0.05` (or use the best config from GAM tuning study).

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | GPU Cost | Risk |
|-----------|------|------|---------------|----------|------|
| M0: Fix scripts | Standardize backbone to T5-large, fix RWP std | Script edits | All scripts use T5-large | 0 | Low |
| M1: Baselines | SeqLoRA+AdamW, IncLoRA+AdamW on T5-large | 2 training runs | Loss converges, ACC_T in expected range | ~3 GPU-h | Low |
| M2: OLoRA vanilla | OLoRA+AdamW on T5-large | 1 training run | OLoRA ACC_T > SeqLoRA ACC_T | ~1.5 GPU-h | Medium |
| M3: Flat-minima sweep | All 3 strategies × {SAM, GAM, RWP} on T5-large | 9 training runs | At least 1 variant improves ACC_T | ~14 GPU-h | Medium |
| M4: Novelty isolation | Block 3 comparison | Already covered in M2+M3 | OLoRA+AdamW > SeqLoRA+SAM | 0 extra | Low |
| M5: Polish | Forgetting matrices, appendix exps | Eval only | Figures ready | ~4 GPU-h | Low |

**Sequential constraint**: M0 → M1 → M2 → M3 → M4 → M5

---

## Compute and Data Budget

- **M1 (baselines)**: 2 × 4 tasks × 20 min = ~2.5 GPU-hours
- **M2 (OLoRA)**: 1 × 4 tasks × 20 min = ~1.5 GPU-hours
- **M3 (flat-minima)**: 9 × 4 tasks × 25 min (SAM/GAM add ~25% overhead) = ~15 GPU-hours
- **Eval**: 12 models × 4 tasks × 5 min = ~4 GPU-hours (parallelizable to ~1 hour on 4 GPUs)
- **Total**: ~23 GPU-hours, ~6 hours wall-clock with 4 parallel GPUs

No new data required; all uses existing CL_Benchmark.

---

## Risks and Mitigations

- **Risk**: T5-large OOM with SAM (double forward pass doubles memory)
  → **Mitigation**: Reduce per_device_train_batch_size to 8, increase grad_accum to 2 for SAM variants

- **Risk**: GAM underperforms across all strategies (confirmed partially in EXPERIMENT_PLAN.md)
  → **Mitigation**: Use best GAM config from GAM tuning study; if still underperforms, report as negative result and recommend SAM

- **Risk**: OLoRA does not outperform SeqLoRA on short 4-task sequence
  → **Mitigation**: Check that orthogonal projection is correctly implemented; verify that subspace dimension is not too restrictive; consider evaluating on longer sequences (Order-2, 3)

- **Risk**: Results are confounded by model size differences (pre-fix)
  → **Mitigation**: Fix scripts first (M0); re-run any already-done experiments that used wrong backbone

---

## Final Checklist

- [ ] All scripts standardized to T5-large backbone (M0)
- [ ] RWP noise_std standardized to 0.05 across all methods
- [ ] GAM rho standardized across all methods
- [ ] Main comparison table (B1) covers OLoRA vs all baselines
- [ ] Flat-minima optimizer ranking (B2) is consistent and clear
- [ ] Novelty isolation (B3) shows orthogonality is necessary
- [ ] Nice-to-have runs (B4, B5) are separated from must-run
- [ ] GAM tuning follow-up (see EXPERIMENT_PLAN.md) feeds best GAM config into M3
