# Component horizon: scientific design and derivation review

## Target

Use a five-rule, ten-task, seed-2 screen to identify which *finite-probe update
construction* merits independent two-task formation/continuation experiments.
The screen does not by itself identify a Hessian mechanism or prove mediation.

## Status

**COHERENT AFTER REFRAMING / EXTRA ASSUMPTION.** The five directions and
candidate-state effective-norm control are well defined. Several causal and
Hessian interpretations in the attachment need the narrower wording below.

## Invariant object

The intervention is a complete learning rule and its resulting trajectory.
The norm control fixes one local scalar,
`||Δ concat(B_i A_i, W_FC, b_FC)||`, at each candidate's own state. It does not
fix a common cross-method trajectory, parameter step, function/KL step, or
old-task gradient projection. The Hessian diagnostic is in trainable-factor
parameter coordinates, not the Hessian in effective BA coordinates.

## Assumptions and notation

- Pure GAM, current-task sliced CE, no Fisher/EWC/fusion/SGV; rho=.05, r=.25.
- All five arms retain the clean-cache backward plus four native GAM probes,
  the same SGD, scheduler, data seeds, initialization, and task boundaries.
- `g0,g1,g2,g3` are the gradients at the four *actually captured native closure
  states*. Native FP32 add/subtract can make the outer base differ from an
  ideal exact-arithmetic theta by roundtrip error. Record that error.
- In exact arithmetic, `δS=g1-g0`, `δO=g2-g0`, `δG=g3-g2` and
  `g3=g0+δO+δG` are identities. They do not imply additive performance effects.
- For HVP claims, hold the batch, transforms, mode, RNG realization and probe
  direction fixed. Detach the normalization direction in autograd.

## Derivation strategy and map

1. Define the five learning-rule interventions.
2. Define each candidate-state g1 reference and actual effective-norm matching.
3. Measure long-horizon performance and separate acquisition from retention.
4. Audit finite-gradient responses against local HVPs only at preselected
   checkpoints; treat alignment as a local approximation diagnostic.
5. Select one candidate using frozen screening rules, then independently train
   a two-task factorial. A three-task persistence test remains conditional.

## Main derivation

### Exact identities

The rules are U0=g0, US=g1, UG=g0+(g3-g2), UO=g2, UFull=g3. Their definitions
are valid without invoking Hessians. UG uses a correction measured at another
state but applied at the candidate's final SGD base. UO is an update along the
gradient measured at the outer probe; the model is not permanently relocated
to that probe. Calling UO “relocation-only” is shorthand, not a literal state
transition intervention.

### Correct norm control

At the common *actual post-unperturb SGD base* of one candidate minibatch,
form a disposable FP32 g1 update using its nominal eta. Measure the actual
unscaled BA+FC change from those factor values in FP64. This is the target for
the candidate direction. Solve for the smallest positive candidate scalar,
then verify the **actual** FP32 update against the target. Record the scalar,
target, actual norm, error and probe-roundtrip displacement. Do not substitute
the earlier campaign's independent-S trajectory target.

Resolve the native `clean + (selected-clean)` arithmetic consistently. If the
training wrapper preserves that projector, both g1 reference and candidate
must pass the same operation sequence; otherwise the control silently mixes
raw g1 with post-projector candidate gradients. The g1 arm must reproduce its
own unscaled nominal update exactly (within the declared numerical audit),
with scalar eta rather than an unnecessary approximate root solve.

Matching rules generate different target norms at different candidate states.
Thus “equal local effective step relative to own-state g1” is accurate;
“every method has the same step length at each chronological step” is not.

### Approximation, with explicit remainder

For fixed u and a twice differentiable loss,

`[g(x+rho*u)-g(x)]/rho = integral_0^1 H(x+s*rho*u)u ds`.

This is a path-averaged Hessian action. It approaches H(x)u as rho→0 under
continuity. If H is L-Hessian-Lipschitz, the forward-difference error is at most
`(L/2)*rho*||u||²`. A high cosine at rho=.05 alone says little about magnitude
agreement; report relative error and norm ratio, and predeclare a short rho
grid (.05,.025,.0125). Central differences provide an additional numerical
check, not a replacement for the actual forward response used in GAM.

The normalization must be detached: differentiating `g(x)·N(g(x))` through
N(g(x)) computes a different derivative. For the outer response, x is the
actual closure-2 state and u is the detached native normalized g2 there.

### Metrics and accounting

Let a[t,j] be all-seen-head sample accuracy on task j after learning task t.
Use one matrix for all the following:

- AAA_task = `(1/10) sum_t (1/(t+1)) sum_{j<=t} a[t,j]`.
- FAA = `(1/10) sum_j a[9,j]`.
- acquisition/LA = `(1/10) sum_t a[t,t]`.
- BWT = `(1/9) sum_{j<9} (a[9,j]-a[j,j])`.
- peak forgetting = `(1/9) sum_{j<9} (max_{j<=k<=9} a[k,j]-a[9,j])`.

The maximum includes the final state, so forgetting is nonnegative. The exact
accounting identity `FAA = acquisition + .9*BWT` is not a causal decomposition.
Additionally retain pooled sample accuracy/CE and task-local CE/accuracy,
clearly separated from task-macro all-seen metrics. No task or minibatch is a
seed replicate.

## Frozen minimum scope and decision rules

1. Wait for the older 24 formation plus 84 future jobs and integrity receipts
   to finish. Do not revise this screen based on partially completed results.
2. Run nine fresh rules at r=.25, seed2: five matched rules plus natural-scale
   U0, UG, UO and UFull. Matched US is its own natural-scale identity control.
   Fresh natural/matched pairs must share the new sampler/RNG protocol.
   Existing natural-scale results remain historical context, because the new
   independent data-RNG policy differs from their native global RNG; no claim
   that a preexisting g0 baseline exists unless a full ten-task receipt proves
   it. Do not enlarge the first screen to r1 or all coefficients.
3. Primary endpoint AAA_task, with FAA a required guardrail; display all four
   planned contrasts UG-US, UO-US, UFull-UO and UFull-US and the U0 anchor.
   A suggested pre-data practical screen threshold is +1.0 percentage point
   AAA and no worse than -1.0 pp FAA, with no failed numerical audit. Freeze
   these values explicitly before launch if adopted; they are selection
   thresholds, not statistical significance or equivalence margins.
4. If no candidate passes that screen, report no candidate selected at the
   chosen resolution; do not conclude the directions are equivalent or that
   all prior gains were caused by scale. If several pass, choose largest
   AAA, breaking differences <.25 pp using FAA; if still tied, choose the
   simpler UO or UG before UFull, with UO before UG. This is an operational
   rule, not evidence that the selected component is uniquely causal.
5. Preserve acquisition and BWT guardrails. An apparent retention gain with
   lower new-task learning triggers matched-progress analysis, not a clean
   stability claim. Selection using seed2 remains exploratory. Confirm the
   chosen two-task factorial on paired seeds0/1/2, ideally add seeds3/4 for a
   main formation claim. No p-value from minibatch/task replication.
6. Curvature audit: matched US and matched UFull at task0/4/9, six checkpoints
   with the rho grid above. Use four fixed native no-repeat batches at task0
   and task4, three at task9 (378 samples, batches128/128/122). This is22 audited
   batches, not22 seeds. Task4's fourth batch has42 samples; report each batch
   and distinguish weighted/unweighted summaries. Do not select checkpoints
   or batches for high Hessian alignment. Optional location×direction2×2 is
   conditional and is not automatically queued.

## Interpretation boundaries

- UG beating US under the control supports usefulness of that finite-response
  construction as a learning rule. It does not alone identify outer Hessian
  change, because the states, probe directions, scalars and histories differ.
- UO approximately matching UFull cannot be established from nonsignificance
  or one seed. A later equivalence claim needs a prespecified margin and an
  uncertainty interval contained in that margin.
- Full beating both pieces is compatible with composition, but not proof of
  superadditive causal synergy. The scalar matching is nonlinear and each arm
  visits different states. A two-task factorial estimates trajectory effects,
  not a unique static-vector cause.
- Natural-scale advantage disappearing after matching is consistent with a
  magnitude/schedule interaction. It does not prove that magnitude alone was
  the cause: matching changes the complete schedule and state evolution.
- XS versus SS has the same future **rule, data and nominal schedule**, not
  the same gradient vectors, actual update norms or perturbation trajectory.
  Report post-expansion K0 and matched new-task progress, as in the existing
  future-probe protocol. Same future learning rule is the defensible phrase.
- SX versus SS needs the same Task0 origin and matched-progress/new-acquisition
  control before interpreting lower old loss as reduced interference.
- Existing finite-rho response geometry is not a Hessian spectrum or a global
  flatness measurement. Positive HVP audit is local, batch- and state-specific.

## Numerical acceptance versus scientific gates

Numerical gates (finite tensors, actual norm matching, original-state restore,
same-point gradients, data hashes) may fail a job. Scientific thresholds such
as HVP cosine must **never** turn an inconvenient valid result into a failed
job or trigger silent retries. Report low alignment and inconsistent rho
trends as findings. Backend fallback must be explicit and its same-point CE
and first-gradient agreement audited before interpreting its HVP.
