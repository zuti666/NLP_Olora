# Four rows at manually selected λ = 20000

`train.py` continues only Task1 from the six completed formal Task0 parents.
The four `(Task0 parent, Task1 optimizer)` pairs are `ewclora=(sgd,sgd)`,
`fs_ewclora=(gam,gam)`, `task1_shaping=(gam,sgd)`, and
`task2_shaping=(sgd,gam)`, each at seeds 0, 1, and 2. This is the user's fixed
selection, not a claim that the preceding protection-threshold scan passed.

The native learner, architecture, data manager, optimizer, and trainer are
imported from the original frozen `execution/source`. Every frozen source hash,
completed same-seed parent identity, checkpoint, Fisher sidecar, metrics, and
parent RNG/runtime receipt is validated. The full Task0 state is restored,
including all frozen backbone weights and the old classifier head. The old
normalized Fisher is then replaced tensor for tensor by the corresponding
`prepared_fisher/{method}_seed{seed}/fisher_state_raw.pt`, never by multiplying
the normalized sidecar by a scalar. `PREPARED.json` binds actual parent method,
seed, checkpoint SHA, source SHA, payload SHA, paired fit inputs, positive raw
trace, and all 24 nonnegative Q/V tensors.

Both Fisher trace-normalization flags are false. The formal continuation uses
50 epochs, learning rate .02, batch size 128, four workers, cosine scheduling,
zero momentum/weight decay, and FP32 with TF32 disabled. All weight merging,
in-task Fisher approximation, extra delta-W penalties, and EWC projections are
disabled. Q/V adapters continue from the parent. Task1 new-head initialization,
sample order, and first augmented batch of every training/Fisher loader pass
must match the same-seed formal `sequence_{future_optimizer}` reference.

The native loss is unchanged. SGD uses CE plus the native EWC penalty in one
backward. GAM computes native clean-CE and EWC gradient caches, performs its
four CE-only closures, and applies the native direction projector once. The
auditor checks the projector tensor for tensor against the native formula
`clean + lambda_flat * (gam - clean) + fisher`; the base optimizer then runs
once after GAM restores the perturbed weights. Every batch records exactly one
native EWC evaluation, finite actual update gradients, and finite post-update
parameters. Extra SGD diagnostic differentiation is read-only and is counted
separately from the native objective evaluation.

Gradient components are sampled at zero-based steps 0, 1, 8, 16, … . SGD clean
gradients are measured as actual combined gradients minus a re-differentiation
of native EWC. GAM clean gradients come directly from the native unperturbed
clean-CE cache. Its actual-combined-minus-EWC residual is explicitly named
`grad_gam_direction_norm`, never clean CE. All norm/cosine measurements use
float64 accumulation; update tensors remain native FP32. Native mechanism
metrics are retained separately from these actual-update receipts.

Completion requires a full finite two-task CIL matrix, deterministic Task0
train/test local-head and seen-head evaluation, exact reload of every saved
Task1 model tensor, nonempty finite final Fisher, and all required checkpoint
sidecars. It also requires positive EWC and nonzero protection gradient at
zero-based step 1. Endpoint raw-Fisher weighted displacement uses the original
prepared Task0 Fisher: `d_F = sum(F * (BA_end - BA_parent)^2)`, with native
penalty `P = 0.5 * 20000 * d_F`. The auxiliary native `endpoint_eval` switch
remains false; ordinary trainer CIL evaluation and explicit local evaluation
are mandatory and checked by completion validation.

`--smoke` uses one epoch and isolated
`smoke/training/smoke_{arm}_seed0` outputs, producing `SMOKE_COMPLETE.json`.
`validate_smokes.py` is a CPU-only gate that rederives each smoke's identity,
revalidates all four complete outputs and actual input pairing, and writes
`smoke/VALIDATED.json`. Formal training checks that this gate and its four
receipts/manifests match the current driver and validator hashes before any GPU
selection. No fake smoke receipt is produced during preparation.

CPU validation completed on 2026-09-22:

- 27 discovery tests passed (18 train/smoke tests and 9 preparation tests), with
  `CUDA_VISIBLE_DEVICES=''` and the existing `Pilot_new` Python environment.
- Tests cover all 12 arm/seed configurations, all six real parent identities,
  invalid normalization/parent/seed/lambda/source boundaries, raw payload
  integrity, incomplete pairing, nonfinite gradients, incorrect provenance,
  and smoke/formal isolation.
- A tiny CPU network executed two real native GAM updates through the exact
  production auditor. EWC was zero at the anchor and positive with nonzero
  protection gradient on step 1; native clean/Fisher/combined mechanism norms
  agreed with independently observed actual update norms.
- `--validate-cpu` passed full ViT construction, exact full-state restoration,
  raw-Fisher loading and new-head initialization for `ewclora` and
  `task2_shaping`, using the real SGD seed0 payload (trace
  `34.069888104965315`). Their new-head hashes agree. Receipts are in
  `cpu_validation/cpu_ewclora_seed0/CPU_VALIDATED.json` and
  `cpu_validation/cpu_task2_shaping_seed0/CPU_VALIDATED.json`.

No GPU process, optimizer on GPU, package installation, or frozen-file edit was
performed in this preparation subtask. The other five raw payloads and all four
real GPU integration smokes remain explicit execution-stage dependencies.
