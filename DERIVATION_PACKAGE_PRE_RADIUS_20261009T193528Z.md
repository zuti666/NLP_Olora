# Derivation Package

## Target

Compare the implemented ImageNet-A A_C direction with the user's negative-first-probe proposal. The latest clarification on2026-10-05 fixes the outer position as theta2=theta0−r N(g1−−g0), and the final update as U=g2. This equals the historical C/g2 outer position, not D/g2. The earlier g0+g2 expression remains a distinct variant. This document is analysis only and does not add or modify an experiment queue.

Correction: the earlier version interpreted the proposal as theta0+r N(g1−−g0) and assigned it to D/g2. That assignment does not apply to the user's now-explicit minus sign. Historical D/C code and their measured results have not changed; the proposal's mapping and interpretation are corrected below.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

The path-integral intuition survives. The original equalities g2=g0+grad(g1−g0) and normalized Delta g=H do not. The corrected objects are a gradient difference, an averaged Hessian-vector product, and a gradient evaluated at a new parameter point.

## Invariant Object

The gradient field g(theta)=grad L(theta) of one fixed current-task minibatch loss. Its change along an explicitly fixed straight parameter path organizes the derivation. Gradient norm is a local geometric interpretation, not a replacement for the continual-learning performance objective.

## Assumptions

- theta denotes all trainable LoRA factors and classifier parameters in the current implementation, not the full frozen ViT weights or the effective Delta-W coordinates.
- Hold the minibatch, augmentation, dropout randomness and task class slice fixed for all probes. Existing code replays R0 for every probe and restores the base state.
- For exact path integrals, L is twice continuously differentiable on the paths considered.
- For stated local Taylor remainder orders, H=grad²L is locally Lipschitz, g0 is nonzero, and ||H0 n|| stays away from zero in the local regime.
- In analytic formulas N(v)=v/||v|| for nonzero v. Actual code uses v/(||v||+1e−12); odd symmetry N(−v)=−N(v) remains exact. Unit-distance interpretations assume this stabilizer is negligible.
- rho=.05 and r=.25 or1 are experimental settings, not certification that Taylor approximations are accurate at these finite radii.

## Notation

- theta0: original trainable parameters before the update.
- L, g(theta), H(theta): fixed minibatch loss, gradient vector, Hessian matrix.
- g0=g(theta0), n=N(g0), H0=H(theta0).
- theta1−=theta0−rho n, gm=g1−=g(theta1−).
- qC=g0−gm, qD=gm−g0=−qC.
- vC=N(qC), vD=N(qD)=−vC.
- theta2C=theta0+r vC, theta2D=theta0+r vD.
- g2C=g(theta2C), g2D=g(theta2D).
- UAC=g0+g2C−gm=g2C+qC; UD=g2D is the historical D update. The clarified user proposal has theta2*=theta0−rN(qD)=theta2C and U*=g2C.
- eta: actual SGD learning rate; it is distinct from a fixed probe distance rho.

## Derivation Strategy

Use exact definitions and a path integral first. Introduce small-radius Taylor approximations afterward. Keep probe direction, evaluation position and final SGD vector separate. Match these definitions to the actual branch selector and existing completed experiments.

## Derivation Map

1. The first gradient difference equals an integral of H along the straight first-probe path, applied to its tangent.
2. Its local approximation is plus or minus rho H0 n, a vector rather than an entire Hessian.
3. The two outer positions differ by an exact sign inversion of the same first-probe residual.
4. The outer gradient change is another Hessian-vector path integral; a local constant-H approximation yields a plus or minus H0² n term.
5. A_C and the clarified g2-only proposal have the same outer position; they differ only in adding the direct qC residual to the final update. Historical D/g2 instead probes the opposite outer position.
6. Existing C/g2 versus D/g2 isolates the outer sign; A_C versus C/g2 isolates the direct qC addition. Existing result reuse requires fixed rho=.05, not the new adaptive distance eta||g0||.

## Main Derivation

### Step 1: First-probe gradient accumulation — identity

Define the positive straight path theta+(t)=theta0+t n for 0<=t<=rho. Its tangent n is fixed at theta0, not recomputed along the path:

\[
g_1^+-g_0=\int_0^\rho H(\theta_0+t n)n\,dt.
\]

For theta−(t)=theta0−t n:

\[
g_1^--g_0=-\int_0^\rho H(\theta_0-t n)n\,dt,
\qquad
q_C=\int_0^\rho H(\theta_0-t n)n\,dt.
\]

The infinitesimal gradient change is dg=H(theta(t)) dot-theta(t) dt. The proposed integral of Delta g with respect to g0 is not the appropriate path integral: g0 is the fixed initial gradient, and the integration variable is parameter-path distance t. Multi-step gradient descent with a changing direction is a different curved path, where dot-theta(t) must also change.

### Step 2: Curvature interpretation — approximation and identity

For small rho:

\[
g_1^+-g_0=\rho H_0n+O(\rho^2),
\qquad
q_C=\rho H_0n+O(\rho^2),
\qquad
q_D=-\rho H_0n+O(\rho^2).
\]

Thus N(g1+−g0) approximates N(H0 n), not the matrix H0. A single directional probe cannot recover the whole Hessian. If H is constant along the path, the first two expressions are exact without the remainders.

For g0!=0, symmetry of the Hessian gives the exact derivative at the base:

\[
\nabla_\theta\|g(\theta_0)\|=H_0\frac{g_0}{\|g_0\|}=H_0n.
\]

Consequently vC is approximately a gradient-norm ascent direction, while vD is approximately its descent direction. This interpretation is local and concerns gradient norm, not necessarily loss. A finite step can leave the local regime. The original GAM paper uses gradient-norm ascent to approximate a neighborhood maximum and gives this derivative in Equation8: [primary paper](https://arxiv.org/html/2303.03108v3).

### Step 3: Outer positions and gradient changes — identity then approximation

For the same negative first probe:

\[
\theta_2^C=\theta_0+rN(g_0-g_1^-),\qquad
\theta_2^D=\theta_0+rN(g_1^--g_0)=\theta_0-rN(g_0-g_1^-).
\]

This historical C/D sign difference is exact, including the common numerical stabilizer; it is not merely a Taylor observation. Both outer probes start from theta0, not theta1−. Omitting r in an expression sets the nominal outer radius to1.

The latest user clarification is instead:

\[
\theta_2^*=\theta_0-rN(g_1^--g_0)
=\theta_0+rN(g_0-g_1^-)=\theta_2^C.
\]

The two minus signs cancel: one in the displacement, and one in the residual relative to qC. Thus this proposal samples the same outer point as C and A_C. It is locally a gradient-norm ascent outer probe, despite using a loss-descent first probe.

Holding each chosen v fixed along its own outer path:

\[
g_2^{C/D}-g_0=\int_0^r H(\theta_0+t v_{C/D})v_{C/D}\,dt.
\]

Therefore g2−g0 is a gradient difference between two positions. It is not grad(g1−g0): differentiating the vector-valued residual with respect to theta would produce a Jacobian matrix and include dependence of its probe position on theta.

If both rho and r are in the local regime and H0 n is nondegenerate:

\[
g_2^{C/D}-g_0
=\pm r\frac{H_0^2 n}{\|H_0 n\|}+O(r\rho+r^2).
\]

This is a second application of curvature in the constant-H local model, not a computation of third derivatives or proof of a long-term CL mechanism.

### Step 4: Final update — identity and local approximation

Implemented A_C:

\[
U_{A_C}=g_0+(g_2^C-g_1^-)=g_2^C+q_C.
\]

Clarified user proposal, with the minus sign applied to qD:

\[
U_*=g_2^C=g_0+(g_2^C-g_0).
\]

These are not the same final update: A_C adds qC, while the clarified proposal only submits g2C. Their first and outer probe positions are identical for the same rho and r. A_C does not use g3, g1+g3 or an orthogonal projection.

The expression g0+g2 instead corresponds to a third update rule; it doubles the base-gradient contribution in the local limit. If the intention is base gradient plus outer gradient change, the correct algebra is g0+(g2−g0)=g2.

Locally:

\[
U_{A_C}=g_0+\rho H_0n+r\frac{H_0^2n}{\|H_0n\|}
+O(\rho^2+r\rho+r^2),
\qquad
U_*=g_0+r\frac{H_0^2n}{\|H_0n\|}+O(r\rho+r^2).
\]

For comparison, historical D has UD=g0−r H0²n/||H0n||+O(r rho+r²). That opposite sign can attenuate or even reverse components of the submitted gradient; this is a local explanation to test, not a proven explanation of observed accuracy, and it does not describe the clarified proposal.

### Step 5: Fixed versus SGD-dependent first radius — identity

The user's theta1−=theta0−N(g0) eta||g0|| equals theta0−eta g0 for ideal normalization. Its probe distance is d=eta||g0||, which changes with the learning-rate schedule, minibatch and model. Current A_C, C/g2 and D/g2 experiments instead use fixed rho=.05. These rules coincide only when eta||g0||=.05. Using the adaptive distance also changes the amount of curvature averaging and normalized direction when H varies along the path. Existing fixed-rho results cannot certify this adaptive variant. The corrected outer-sign algebra remains exact for any shared first-probe distance.

### Step 6: Mapping to frozen implementation — code evidence

[direction_step.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/trainer/direction_step.py:158) selects cminus=gm−g0 for D, flips its sign for C, and constructs the outer point from the original base with a plus sign. Lines167–168 choose either g2 or A=g0+g2−first. Thus the clarified fixed-rho user proposal maps to C/g2; A_C maps to C/A. Both frozen suites already include these directions. Calling the clarified proposal D/g2 would mislabel the existing implementation.

## Remarks and Interpretation

- Loss-descent first probes can estimate H0 n, provided their residual sign is adjusted; the sign correction changes which outer neighborhood is sampled.
- Gradient-norm descent at a probe does not establish a flatter converged model or better continual-learning retention. A low-gradient probe can also provide a weak training update away from that probe.
- In a one-dimensional quadratic L(theta)=theta²/2 at theta0=1 and rho=.05, gm=.95. With r=1, historical D reaches0 and submits0 despite g0=1. C and the clarified proposal probe theta=2 and submit g2C=2; A_C submits2.05. The vanishing-update example pertains to historical D, not the clarified proposal. It demonstrates non-equivalence, not an explanation of the neural-network trajectories.
- The minimal existing comparison is g0, C/g2, D/g2, A_C. C/g2 versus D/g2 changes the outer residual sign with the same negative first probe and same final g2 rule. A_C versus C/g2 changes only the direct qC addition. Compare full curves and acquisition/retention metrics, not just final-gradient norm.

## Boundaries and Non-Claims

### Verified existing result comparison

The six fixed-rho, natural-step runs below are complete: seed2, rank10, rho=.05, ten tasks,500epochs,2550updates, exit0. Final-summary/matrix/trace hashes match completion receipts; task-macro AAA/FAA and diagonal-mean LA were recomputed from the matrices. No Fisher/merge/HVP work. Their common four-probe sequence includes an unused g1+ but no g3; the final D/C definitions are exactly those above.

| Direction | r=.25 AAA / FAA / LA | r=1 AAA / FAA / LA |
|---|---|---|
| D/g2, historical opposite outer probe | 46.887 / 24.999 / 40.117 | 37.359 / 23.187 / 27.351 |
| C/g2, clarified fixed-rho user proposal | 66.647 / 49.793 / 56.285 | 65.774 / 49.192 / 54.759 |
| A_C=C/A | 66.758 / 49.864 / 56.241 | 64.438 / 47.185 / 54.305 |

Sources: [r=.25 summary](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/summary.csv), [r=1 summary](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/summary.csv). Jobs in each suite: S1-05_O_D_natural_s2, S1-06_O_C_natural_s2, S1-09_A_C_natural_s2; full evidence in outputs/JOB/{final_summary.json,task_matrix.json,step_trace.jsonl} and runtime/jobs/JOB/{completed.json,exit.json}.

With the same g2-only final rule, D versus C shows a large observed effect of outer sign. LA differs by16.168pp at r=.25 and27.408pp at r=1, so the weak D endpoints include poor new-task acquisition and cannot be interpreted solely as forgetting. Adding the direct qC term gives A_C versus C/g2 differences of +.112AAA/+.071FAA at r=.25 but −1.336AAA/−2.007FAA at r=1. The direct residual is not consistently beneficial in these single-seed natural-step experiments. Neither comparison proves the local H² explanation. The adaptive eta||g0|| first-radius variant has not been certified by these fixed-rho results.

### Limits

- No new experiment is launched by this analysis. Previously requested queues remain untouched.
- No Hessian spectrum is recovered from one gradient residual, and no H² mechanism is established by a single-seed accuracy difference.
- Taylor formulas are local; actual r=.25 or1 can yield varying Hessians, cancellation and finite-radius effects.
- Natural-step comparisons include changes in both final direction and magnitude. Norm-matched results address a different experimental question.
- The user's adaptive first-probe radius and ambiguous earlier g0+g2 expression are separate, currently unverified variants.
- All mathematical derivatives refer to trainable LoRA/classifier coordinates and the current-task minibatch objective, not a Fisher estimator or a full-model Hessian.

## Open Risks

- q close to zero makes its normalized direction sensitive to finite precision and stochastic discrepancies.
- The local gradient-norm up/down interpretation should be checked using same-state probe measurements rather than asserted from final accuracy.
- Existing single-seed results establish observed behavior only; causal explanations need acquisition/retention traces and appropriate scale controls.
- Final intended update is interpreted as g2 from the user's last sentence; g0+g2 must be evaluated separately if that was intended instead.
