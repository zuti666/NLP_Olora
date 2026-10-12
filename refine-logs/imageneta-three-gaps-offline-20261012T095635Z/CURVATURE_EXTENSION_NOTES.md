# Exploratory fixed-checkpoint curvature convergence extension

This is new diagnostic code. The original campaign, original numerical failure,
checkpoints and preregistered three-distance gate are not modified. It performs
zero optimizer steps, and restores parameters, frozen version counters,
gradients, buffers, modes, runtime controls/caches and all RNG streams.

## CLI and fixed scope

Use the existing warm Python environment and deterministic execution variables
from the accepted campaign. A queue must lease only authorized GPU 4/5/6.

```text
python curvature_diagnostics.py --campaign OLD_CAMPAIGN --seed 2
  --checkpoint OLD_CAMPAIGN/outputs/formal_S_s2/checkpoints/epoch_001.pt
  --output-dir FRESH_PILOT_DIR --pilot --fp64-pilot-fallback
```

The pilot is fixed to formal S, seed2, epoch1, batch0 (128 fixed augmented Task1
training samples): the original failure point. `--fp64-pilot-fallback` is optional
and allowed only for this pilot. It triggers only after native/explicit FP32
first-order agreement passes and the FP32 HVP convergence test does not pass.

Full per-seed scope requires exactly six repeated `--checkpoint` arguments:
S/R branches x epoch1/10/25. Each uses two fixed training batches, for 12 cells
per seed and 36 cells overall. Scientific gate failure retains the cell and
continues; unexpected runtime exceptions raise and remain execution failures.

## Separate evidence gates

1. Native full128 gradient repeats exactly. Explicit math/microbatch16 gradient
   agrees with it (relative error <=2e-4, cosine >=.99999). First-order agreement
   alone is not evidence of higher derivative agreement.
2. At that same BASE, exact loss-Hessian products `q=H.detach(g0)` and
   `p=H.detach(q)` are checked by symmetric finite differences of gradients.
   The vector is normalized only to choose absolute parameter distances; the
   quotient restores the vector's original length. Both products must have two
   adjacent distances with relative error <=.08; no test-set selection occurs.
3. Original `[.001,.003,.01]` points and original pass/fail remain explicitly
   recorded. A separately labelled, predeclared exploratory grid is
   `[1e-5,3e-5,1e-4,3e-4,.001,.003,.01]`. All points and all failures are retained.
   Its acceptance cannot retroactively turn the original gate into a pass.
4. Optional pilot FP64 uses an independent deep clone, cast model/registered and
   cached floating tensors, exact verification of every represented BASE
   parameter, math attention and microbatch4. Its first gradient must agree with
   the original native FP32 gradient. HVP input remains original native FP32 g0,
   detached and converted to FP64, and the second vector is detached FP64 q.
   FP32/FP64 q and p differences are saved; FP64 original/extended gates remain
   separate. This checks a more precise realization of the same loss function;
   it does not establish exact identity with all FP32 higher derivatives.
5. Only after one independent HVP convergence gate passes, immutable native FP32
   candidate construction is used at formal `(a,r)` and joint half `(a/2,r/2)`.
   Compare `Delta_native=g2-g0` against `r*H.detach(H.g0)/||H.g0||`, not full g2.
   Candidate agreement is not used to tune numerical gates or choose cells.

The Hessian may be indefinite. No Fisher/GGN substitution and no assumed
nonnegative `g^T H g` energy interpretation is made. Degenerate gradients,
curvature and corrections retain explicit null/empty comparison outcomes.
The independent replicate is training seed, not checkpoints or minibatches.

## Outputs

- `checkpointXX_batchYY.json`: identity/checkpoint hash, fixed batch identity,
  actual saved next-update LR, original/extended FP32 and optional FP64 gates,
  every FD point, convergence classification, optional corrections, exact
  restoration receipt and zero training update counter.
- `diagnostic_results.json`: all cells, code/kernel hashes, aggregate number of
  original FP32 accepted cells and exploratory accepted correction cells.
- `completion_receipt.json`: accepted *execution*, scientific outcome (some
  converged or inconclusive), expected cell count, result hash, exact restoration.

Interpret `first_order_reference_disagreement` separately from
`higher_derivative_numerical_convergence`; after accepted HVP convergence, formal
vs joint-half correction error tests finite displacement bias. Improving error
at half scale supports such bias locally, not a long-term training causal claim.

## CPU validation

`python test_curvature_cpu.py` passes nine tests: indefinite quadratic exact HVP
and both correction scales; retained original failure with exploratory small
grid convergence; first-order reference mismatch blocking HVP; detached second
HVP against dense H; FD vector rescaling; zero gradient/curvature degeneracy;
nonfinite scientific outcome and restoration; FP64 clone/base/reference; and
adjacent-point rule rejecting separated/nonfinite passes. CLI `--help` also
passes. No GPU execution was performed by the implementation agent.
