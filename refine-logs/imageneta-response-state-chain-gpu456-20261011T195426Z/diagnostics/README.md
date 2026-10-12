# Offline diagnostics (no training updates)

The exact curvature operator is the **Hessian of the inherited mean training
cross entropy with respect to all trainable Q/V LoRA factors and the entire
40-class classifier**. It is not Fisher, generalized Gauss–Newton, or a PSD
surrogate. `0.5 * g^T H g` may be negative. Both Hessian-vector products are at
the checkpoint base point; the second vector `H g0` is explicitly detached.

Each saved checkpoint at Task1 epochs 1, 10, and 25 uses the same two fixed
128-sample current-task training minibatches. Across six branches this gives
36 state/batch combinations. These do not count as independent training seeds.
Native S_F is `g2 = grad L(theta + .25 N(g0-grad L(theta-eta*g0)))`.
Native SAM is `grad L(theta + .05 N(g0))`. Offline radius scales are exactly
`1` and `0.5`, jointly multiplying S_F's inner step and outer radius, without
changing training. `eta` is the saved optimizer's **next-update LR after the
epoch scheduler step**; the completed epoch's LR is also recorded. At epoch25
this is the counterfactual SAM/S_F probe LR for epoch26, although formal
training next uses SGD.

## Pre-registered numerical checks

- Original gradient repeated at identical parameters, buffers, modes and RNG
  must agree byte for byte.
- Explicit math-attention / exact CE-mean microbatch gradient must match the
  native original-batch gradient: relative error at most `2e-4` and cosine at
  least `.99999`. Math attention is used only after this agreement check.
- Exact `H g0` and `H detach(H g0)` must agree at two adjacent points in the
  full central difference grid `[.001, .003, .01]` at relative error at most
  `.08`. Distances are absolute global trainable-parameter L2 displacements
  along the corresponding normalized vector. All grid values are retained.
- Absolute `||H g0|| <= 1e-12` is a degenerate curvature diagnostic.
- Native first-response and correction degeneracy threshold is
  `32 * eps(float32) * max(||g0||, 1e-12)`. Undefined cosines are `null` and
  the degenerate records remain in the denominator of reported valid counts.
  Genuine degeneracy is nonfatal and all remaining combinations are processed.
  An entirely degenerate pilot returns exit3 as inconclusive rather than
  claiming a successful numerical implementation or a failed training method.
- Complete trainable values, gradients, frozen version counters, registered
  buffers, modes, caches and Python/NumPy/Torch/CUDA RNG are restored exactly.

No gate is based on response agreement or accuracy. A numerical gate failure
halts offline Hessian claims, while separately validated formal training can
remain valid. Seed 2 is checked first, then seeds 0 and 1 use the same rules.
Microbatch size 16 limits double-backward memory. This exact mean decomposition
requires the inherited zero-dropout, no-training-BatchNorm architecture and is
checked against its native full-batch first gradient.

## Effective displacement comparison

At epochs 10 and 25, native SAM and S_F candidates are recomputed from the same
state and batch. The full effective map is **all unscaled LoRA `BA` matrices
plus every full classifier parameter**. SAM uses the saved native LR. A single
positive scalar is found for S_F by the exact quartic BA/FC displacement norm;
the smallest positive root in `(0,10]` is selected. The model is never updated.
Report native and matched effective-direction cosine, norm equality, FC
displacement separately, and every layer's exact `delta Q` split into first-
and second-order parts. Dense FP64 matrix calculations independently verify
the low-rank Gram polynomial and all finite-step identities.

This controls instantaneous effective step size, not accumulated trajectory
scale or the different native probe rules. Failure to match is reported rather
than replacing the candidate with another algorithm.

## CLI

```
PYTHON diagnostics/offline_diagnostics.py --run-root ROOT \
  --checkpoint S/checkpoints/epoch_001.pt \
  --checkpoint S/checkpoints/epoch_010.pt \
  --checkpoint S/checkpoints/epoch_025.pt \
  --checkpoint R/checkpoints/epoch_001.pt \
  --checkpoint R/checkpoints/epoch_010.pt \
  --checkpoint R/checkpoints/epoch_025.pt \
  --seed 2 --output-dir ROOT/diagnostics/results/seed2
```

`--smoke` selects the first checkpoint and first full 128-sample fixed batch,
and still executes all backend/HVP gates and both response scales. Output:
`diagnostic_results.json`, per-combination JSON, and `completion_receipt.json`.
Failed numerical validation returns exit code 2; all-degenerate diagnostics
return exit code3. No CUDA workloads are launched
by these files unless their CLI is explicitly invoked by the authorized queue.
