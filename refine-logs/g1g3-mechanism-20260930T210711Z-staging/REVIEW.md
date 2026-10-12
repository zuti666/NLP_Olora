# First-stage review: g1 versus g3 mechanism

**Review decision:** APPROVE WITH REQUIRED GUARDRAILS. The proposed first stage is compact enough to execute and covers the attachment's three priority experiments without promoting every diagnostic into a full 10-task run. Do not launch from this staging review.

## 1. Frozen first-stage scope

The formal training matrix should contain exactly five new seed-2 runs:

| New arm | Final update | `gam_grad_norm_rho` (`r`) | Purpose |
|---|---:|---:|---|
| `mean_sum_r1` | `0.5*g1 + 0.5*g3` | 1.0 | Same direction as the existing unaveraged sum, half its raw gradient scale |
| `g3_r0` | `g3` | 0.0 | Mathematical bridge to `g1`; also an implementation/reproducibility audit |
| `g3_r01` | `g3` | 0.1 | Low-radius point on the `r` intervention |
| `g3_r025` | `g3` | 0.25 | Intermediate-radius point |
| `g3_r05` | `g3` | 0.5 | Intermediate/high-radius point |

Reuse the existing pure `g1`, pure `g3` at `r=1`, and unaveraged `g1+g3` at `r=1` results. Keep `rho=gam_grad_rho=0.05`, `gamma=0`, Fisher/EWC/fusion/orthogonal correction off, and all other training settings frozen. `beta2` does not enter the final update when `gamma=0`, but the same four GAM gradient calls and clean-cache pass must still execute in every arm so call count and RNG consumption remain aligned.

Do not add the previously proposed fixed-`kappa=2.9` normalized full-training arms in this stage. The five runs cost approximately 12.1--12.35 GPU-hours from the observed 2.42--2.47 GPU-hours per run, before queue and I/O overhead.

## 2. Mathematical corrections required before interpreting results

### 2.1 Taylor language must remain local

The identities

`g1 = g0 + rho*H*u0 + O(rho^2)` and `g1-g0 ~= rho*H*u0`

are local approximations. At `r` as large as 0.5 or 1.0, the expansion of `g2` or `g3` around `theta` is not evidence that the measured vector is a Hessian-vector product or a "higher-curvature direction." Use **finite-radius gradient-change direction** for `N(g1-g0)` unless an explicit local HVP audit supports stronger wording. The formal `r` sweep measures the behavior of the chained probe; it does not validate a large-`r` Hessian approximation.

Likewise, the expansion `g1+g3 ~= 2*g0 + ...` is only a heuristic local decomposition. Outside the local regime there is no unique causal sense in which the clean gradient has been "added twice."

### 2.2 Keep the `a` and `1+a` coefficients straight

For the same checkpoint and batch, define

`a = <g3,g1> / (||g1||^2 + eps)` and `r_perp = g3 - a*g1`.

Then the two exact decompositions are:

- `g3 = a*g1 + r_perp`;
- `g1+g3 = (1+a)*g1 + r_perp`.

Therefore, a scale-only comparator for **`g3`** is `a*g1`; a scale-only comparator for **the sum** is `(1+a)*g1`. Mixing these answers two different questions. Log `a`, `1+a`, their signs, `||r_perp||/||g3||`, and the orthogonality residual. If `1+a <= 0` or is close to zero, the proposed sum-direction interpretation is unstable and that anchor must be reported separately rather than silently normalized.

### 2.3 Matched norm makes same-ray scale controls degenerate

`g1` and `(1+a)*g1` lie on the same parameter-space ray when `1+a>0`. Once both are rescaled to the same actual update norm, they are the same intervention. Thus a matched-norm comparison of these two cannot identify a scale effect. Scale is tested by the formal `mean_sum_r1` versus unaveraged sum comparison. The offline directional intervention should instead compare, at a common actual-update target,

`u_lambda = (1+a)*g1 + lambda*r_perp`,

with `lambda=0` as the scale-only endpoint and positive `lambda` adding the new component. The corresponding `g3`-specific family would use `a*g1 + lambda*r_perp`; do not combine the two families in one claim.

### 2.4 Orthogonality is coordinate-dependent

`r_perp` is orthogonal to `g1` in the Euclidean coordinates of trainable LoRA factors plus FC. LoRA has a bilinear, scale-nonidentifiable parameterization, so this is not an invariant statement about effective weights or functions. Call it a **new parameter-gradient component relative to g1**, and report effective-weight displacement/cosine as a secondary diagnostic. Do not claim that it is intrinsically a new function-space direction from parameter-space orthogonality alone.

## 3. Offline diagnostics that belong in this stage

All diagnostics must be computed on checkpoint clones, with the same checkpoint and same minibatch for every candidate. Save and restore Python, NumPy, CPU Torch, and CUDA RNG states around each branch. Never insert the extra probes into formal training: LoRA's temporary `Linear` construction consumes CPU RNG and would alter later training.

### 3.1 Same-state chain and `r` decomposition

At each selected anchor compute `g0`, `g1`, and `g3(r)` for `r in {0, 0.1, 0.25, 0.5, 1.0}`. Record at least:

- `cos(g1,g3(r))`, `cos(g0,g3(r))`, vector norms, and `||r_perp||/||g3||`;
- the exact raw parameter probe vector `epsilon3(r) = r*N(g1-g0) + rho*N(g2(r))` and its norm, rather than approximating its radius by `r+rho`;
- the loss at every probe point;
- `g2(r)` versus `g3(r)`, which isolates the final `rho*N(g2)` tail without another formal arm.

The `r` sweep is an intervention on the complete chained construction. It changes probe radius, probe direction, and the subsequent `g2`-dependent tail together. A monotone accuracy curve against `r` is a dose-response of this algorithm, not proof of the mediation chain `r -> curvature -> retention`.

### 3.2 Direction interpolation and perpendicular-component intervention

Keep both as offline screens:

- normalized direction interpolation `N((1-alpha)*N(g1) + alpha*N(g3))`;
- `u_lambda = (1+a)*g1 + lambda*r_perp` for the sum-specific question.

Use a small, predeclared grid and the same anchors/batches. An interior best `alpha` or positive `lambda` in a one-step counterfactual is evidence worth promoting, but it does not by itself establish long-horizon directional complementarity.

### 3.3 Match the actual LoRA-plus-FC weight update

The matching target must include every trained Q/V LoRA module and the classifier. For a LoRA module with effective update `s*B*A`, a candidate parameter step `(dA,dB)` induces the exact displacement

`dW_eff = s*((B+dB)*(A+dA) - B*A)`

`= s*(B*dA + dB*A + dB*dA)`.

Use the module's actual scaling and multiplication convention; do not assume it from names. The total target is

`sqrt(sum_modules ||dW_eff||_F^2 + ||dW_FC||_F^2 + ||db_FC||_2^2)`

for all trainable FC tensors. Apply the real optimizer parameter-group learning rates and scheduled LR when constructing the virtual step. Matching only LoRA factors, only `B*A`, or only the flattened parameter-gradient norm is insufficient. Because the LoRA map is bilinear, solve the candidate's single global scale against the exact reconstructed displacement (for example by a bracketed scalar search), then verify the achieved relative mismatch rather than assuming linear scaling.

### 3.4 One-step current/old-task counterfactual

Compute the direction from the current-task training minibatch, then evaluate the virtual update on an independent current-task evaluation minibatch and one independent held-out batch per old task. Report exact finite-step changes separately:

- current-task loss change, measuring immediate acquisition;
- macro average and per-task old loss changes, measuring immediate interference;
- optionally class margin/accuracy changes as secondary, noisier quantities.

Do not pool old examples into a size-weighted loss. Old-task data are diagnostics only and must never enter the formal update. These one-step effects are local proxies; they cannot alone establish that a direction causes final forgetting.

## 4. Radius-matched SAM controls

Prioritize the same-state **raw parameter-radius** control. For each `g3(r)` probe, compute `delta_r = ||epsilon3(r)||_2` and compare it with

`g_SAM_match(r) = grad L(theta + delta_r*N(g0))`.

This controls the Euclidean radius in the trainable parameter coordinates while changing the probe direction. It should be an offline comparison first. Parameter-radius matching does not fully remove distance as a functional confound under LoRA reparameterization, so the conclusion must be limited accordingly.

KL-radius matching is optional only if it is numerically controlled. Use the same examples and KL orientation for both probes, bracket the SAM radius, check local monotonicity, and record no-root/multiple-root cases. If multiple matches exist, predeclare a rule such as the smallest positive root. Do not silently turn an unstable per-batch root finder into a new full-training optimizer. Promote a KL-matched SAM run only after the matched radius is stable across anchors and a fixed or explicitly dynamic training rule has been specified.

## 5. `r=0` equivalence is a required audit, not an assumption

Mathematically, with deterministic gradients and exact restoration,

`r=0 => g2=g0 => g3=grad L(theta + rho*N(g0))=g1`.

The code path still performs later gradient calls, normalization, perturb/unperturb, and RNG-consuming forwards. Before treating `g3_r0` as equivalent to the existing `g1` arm, verify on several same-state batches:

- identical model parameters before both probes and after restoration;
- relative `L2(g3(r=0)-g1)` and cosine within a declared numerical tolerance;
- equality of probe losses/logits when RNG is restored;
- no stale `e_w_1_2`, missing-gradient, or parameter-order mismatch;
- identical four-call plus clean-cache execution and next-step RNG state for the formal arms.

If the audit fails, `r=0` is an implementation/RNG control rather than the SAM endpoint, and the `r` sweep cannot be presented as a clean continuous path from `g1` until the discrepancy is explained.

## 6. Outcome reporting and causal limits

Every formal arm must report the full task accuracy matrix, AAA, FAA, mean acquisition (matrix diagonal), final old-task accuracy, and per-task/mean forgetting or BWT. Existing matrices already show why this split matters:

- diagonal acquisition: `g1=53.436`, `g3=56.728`, `sum=61.583`;
- final old-nine-task mean: `44.911`, `51.074`, `54.076`;
- forgetting: `9.482`, `7.011`, `7.619`.

Thus the sum has stronger acquisition and higher absolute final old-task accuracy than `g3`, but slightly worse forgetting. Do not prestate that its benefit is improved retention. For `g3` versus `g1`, both acquisition and forgetting differ, so the first-stage question is how much each contributes.

With only seed 2, checkpoints and batches are paired diagnostic samples, not independent training replicates. Report paired distributions and stage/task stratification; do not claim cross-seed significance. Correlations such as `r`, radius, direction change, and FAA do not establish mediation with five settings from one seed.

## 7. Coverage of the attachment's priority experiments

1. **Scale confound (`g1`, `g3`, sum, mean-sum): satisfied.** The new `mean_sum_r1` plus three reusable arms directly tests the consequence of halving the unaveraged sum under the frozen schedule. It does not by itself prove that mixture direction is useful relative to `g3`; the offline matched-update interventions supply that screen.
2. **`g3 = a*g1 + r_perp` with matched norm: satisfied at the intended first-stage diagnostic level.** The offline `lambda` intervention matches the exact LoRA-plus-FC effective weight displacement and measures current/old-task one-step effects. A final-performance claim about `r_perp` remains conditional on a later formal run.
3. **`r=0 -> 0.1 -> 0.25 -> 0.5 -> 1`: satisfied.** Four new formal `g3` runs plus the reusable `r=1` run create the requested curve; the existing `g1` run is the semantic endpoint only after the `r=0` audit passes. Same-state radius-matched SAM is needed before attributing the curve to probe direction rather than distance.

## 8. Explicitly conditional follow-ups

Do not add these to the current five-run queue:

- one selected full `g1 + lambda*r_perp`-family arm, only if the offline effect is stable in sign across task stages and improves current/old loss tradeoff at matched effective update norm;
- one parameter-radius-matched SAM full run, only after same-state matching shows that distance alone does not reproduce `g3` and a train-time radius rule is frozen;
- a KL-radius-matched SAM run, only after the root is stable and auditable;
- a full direction-interpolation arm, only if an interior direction is consistently favored offline;
- HVP/Fisher/spectrum, rank/backbone/dataset, additional seeds, or LoRA-versus-full geometry studies.

The first stage is therefore sufficient for the attachment's top-three priorities, with the correct evidence boundary: five formal runs establish the scale and `r` curves, while the same-state offline package decides which single directional or matched-radius control, if any, deserves promotion.
