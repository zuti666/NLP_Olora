# Independent review: cached-history plus SGD-prediction response

Reviewed attachment: `/home-local/liying/.codex/attachments/8fe6fc89-7311-481f-86e7-f9e8ceb8e8c5/Pasted text.txt`.

Review scope: formulas and integration requirements, read only. No existing implementation or GPU process was modified.

## Exact first-version definition

Let the fixed ordered trainable parameter vector at the beginning of an update be theta, the current scalar SGD learning rate be eta, and the gradient of the current realized training loss be g. The proposed method does not use the old GAM radii r or rho. Its future probe is theta_f = theta - eta*g; its prescribed future distance is ell_f = eta*||g||. Both gradients in this update use the identical realized batch, loss/class slice, and stochastic forward realization.

After measuring g_f at theta_f, restore the complete base state before updating. With valid history (previous raw base gradient g_p and previous actual committed displacement s), ell_b = ||s|| and

    U = g + 0.5*(g - g_f) + ell_f/(2*ell_b)*(g_p - g).

Without valid history:

    U = 2*g - g_f.

Submit exactly one plain SGD update theta_new = theta - eta*U. Only after the submission succeeds, cache detached copies of g and the actual, measured displacement theta_new - theta. The historical gradient is neither the corrected U nor g_f. The displacement is neither the hypothetical -eta*g nor a reconstruction using the next learning rate.

If the future probe is zero or unresolvable in parameter dtype, the attachment defines U=g. Invalid historical displacement instead selects the no-history formula. No new clipping, projection, norm matching, replay of a previous batch, response normalization, or automatic learning-rate change belongs to this first version.

## Sign check and precise geometry

For a fixed differentiable loss, define u=g/||g||, v=-s/||s||. The exact integrated-Hessian identities are

    (g-g_f)/ell_f = [integral_0^1 H(theta-alpha*ell_f*u) d alpha] u,
    (g_p-g)/ell_b = [integral_0^1 H(theta+alpha*ell_b*v) d alpha] v.

Both signs in the attachment are correct. The two Hessian averages need not be equal; if they are approximately the same, their average is H*((u+v)/2), rather than automatically H*u. A direction v=-u cancels the leading fused response on a quadratic. This is meaningful and must not be hidden by separately normalizing the two responses.

For a quadratic with a constant symmetric Hessian, the identities and fusion are exact. With history, U = g + ell_f*H*(u+v)/2. Without history, U=(I+eta*H)g exactly. The latter is the gradient of L+(eta/2)||grad L||^2 when eta is held fixed for this update.

The historical version is not generally the gradient of that fixed regularized objective. Neither version is an inverse-Hessian method. No curvature guarantee follows from the formulas alone. In a centered positive quadratic without history, one eigendirection evolves with factor 1-a-a^2, a=eta*lambda_H. Its stability interval is 0<a<1, whereas vanilla SGD permits 0<a<2. Positive curvature can therefore amplify instability at the same learning rate; this is not an implementation defect.

## Most important minibatch limitation

In ImageNet training, cached g_p normally uses previous batch B_prev while g uses B_cur. Thus

    g_p-g = [grad L_Bprev(theta_prev)-grad L_Bprev(theta)]
            + [grad L_Bprev(theta)-grad L_Bcur(theta)].

The first term is a previous-loss path response; the second is a batch/loss change at the same parameter position. With stochastic forward functions, this second term also includes changes in stochastic realization. Dividing by ell_b does not remove it. When ell_f and ell_b are comparable, the extra historical term contains approximately one-half of the raw minibatch gradient difference even as eta becomes small. Therefore history may act partly as gradient averaging or trajectory-dependent noise filtering. The algorithm still implements the requested cache rule, but its historical component cannot generally be described as a pure Hessian response of the current batch.

A decisive counterexample has two linear batch losses, both with zero Hessian. Let g=[0,2], g_p=[2,0], eta=.01, and s=[-.02,0]. Then ell_f=ell_b=.02, g_f=g, and the historical rule gives U=[1,1]. Its nonzero correction occurs with no curvature at all. This should be a positive test of the requested cached-minibatch semantics, not a test expected to return plain g.

Do not clear history on every ordinary batch boundary in the history-enabled main version: doing so silently removes the proposed algorithm. Clear it on task/model-coordinate changes. A same-batch historical diagnostic would require reevaluating the previous point or retaining a batch and changing the protocol; that is a separate experiment with different cost.

## Integration requirements

1. Use the baseline scalar LR currently held by the optimizer, including task-specific LR and cosine decay. All groups must have the same LR; otherwise the scalar-distance formula needs a separately named generalization. Plain SGD requires momentum=0, weight_decay=0, dampening=0, nesterov=false, maximize=false. Maintaining a correct actual-displacement cache does not by itself make momentum-SGD equivalent to the specified formal update.
2. Preserve the baseline scheduler cadence. The inspected baseline `imageneta-positive-g012-combinations-20261005T185032Z/trainer/four_probe_train.py` wraps scheduler.step at epoch end and requires T_max=50. Therefore advance once per epoch, not once per batch. The attachment's per-step scheduler pseudocode must not change the established training configuration. The prediction never advances an optimizer or scheduler.
3. Snapshot all mutable state before probing: trainable parameters, gradients, registered buffers, module modes, temporary LoRA/forward caches, and CPU/CUDA RNG. Base and future probes replay the same pre-base RNG. After probing, restore base tensors/state and advance RNG exactly as one normal base forward would. The future forward must not advance persistent running statistics twice.
4. If actual post-update loss is recorded, use an isolated forward-only transaction after the formal update. Evaluate base/future/post losses on the same batch and stochastic realization for a meaningful delta, then restore committed model buffers/caches/RNG. A third forward is an explicit diagnostic cost; it must not become a third gradient call or change training trajectories.
5. Cache the ordered parameter signature and task/loss identity. A new task must invalidate history even when tensor shapes happen to match. Head expansion, replacement of Parameter objects, freezing/unfreezing, reinitialization, or changed coordinate order also invalidates it. Only caching shape lists misses replacement or semantic reinitialization.
6. Preserve one coordinate for every optimized trainable parameter; treat an unused raw gradient as zero with consistent instrumentation. Avoid changing the parameter subset between base and future probes. Cache clones, not detached views that are subsequently overwritten by optimizer or gradient assignments.
7. Record both nominal ell_f=eta*||g|| and actual prediction norm ||theta_f-theta||, prediction rounding relative error, actual committed displacement norm, and history validity/reset reason. FP32 tensor assignment can round an intended prediction to zero. Detect this from measured tensors; do not rely solely on nonzero nominal ell_f. An epsilon in a denominator changes the exact formula, so use the valid-history branch and fail explicitly on nonfinite values instead of silently imposing a coefficient cap.
8. The prior-update actual step belongs to eta_(t-1) and U_(t-1); never infer it from current eta or cached raw g. Cache only after a successful formal commit. A failed probe/failed optimizer update must leave both model and historical state unchanged.

## Necessary mathematical and integration tests

### 1. Non-diagonal positive-definite quadratic with nonparallel history

Use float64 H=[[3,1],[1,2]], theta=[.4,-.7], eta=.04, s=[-.03,-.02], theta_prev=theta-s. The eigenvalues are 1.38196601125 and 3.61803398875. Analytic expected values:

    g                = [ .5, -1.0 ]
    g_p              = [ .61, -.93 ]
    g_f              = [ .48, -.94 ]
    ell_f            = .044721359549995794
    ell_b            = .03605551275463989
    u dot v          = -.12403473458920829
    future response  = [ .4472135954999583, -1.341640786499875 ]
    history response = [ 3.0508510792387633, 1.941450686788304 ]
    U(history)       = [ .5782191040240648, -.9865878428937768 ]
    theta_new        = [ .3768712358390374, -.6605364862842489 ]
    U(no history)    = [ .52, -1.06 ]

Check the exact fused formula against an independent H multiplication. This catches all three common errors: using the wrong difference sign, assuming equal distances, and substituting -eta*g_p for the actual displacement. These expected values were independently checked with NumPy on CPU during this review.

### 2. Variable LR and actual-step cache

Run two or more steps on the same quadratic with eta sequence [.04,.02,.01]. Compare the implementation to an independent NumPy recurrence that measures actual parameter changes after each corrected commit. Verify cache_gradient == raw g, cache_displacement == theta_new-theta, and that the second step uses the previous committed step rather than either predicted step. Include a history direction nonparallel to g.

### 3. Zero curvature but changing minibatch

Use the linear counterexample above. Verify U=[1,1] with history and U=[0,2] without history. This validates the cross-minibatch cache rule and explicitly documents the limitation of a pure-curvature explanation.

### 4. Empty/invalid history and coordinate reset

History-on and history-off implementations must have identical first-step gradients, prediction, U, and final parameters. After an explicit task reset they must again agree for the first new-task step. After head expansion or replacement (including a same-shape reinitialization), old history must not be accepted. A valid ordinary next minibatch must retain history. Test zero actual previous displacement selecting no-history without division by zero.

### 5. FP32 unresolvable prediction

Choose parameters where a nonzero intended eta*g is below the parameter ULP. Verify measured prediction is unchanged, the defined U=g fallback is selected, and the actual zero committed displacement invalidates history for the next step. Also test exactly zero g and eta=0; all recorded diagnostics must remain finite with explicitly defined zero-norm conventions.

### 6. Stateful stochastic model transaction

Use a tiny dropout + mutable-buffer model. Verify both gradient probes use the same random mask, predictions restore exact base parameters before formal commit, buffer statistics advance only according to the designated base pass, frozen parameters remain unchanged, and global RNG equals the one-base-pass reference. Count exactly two ordinary gradient evaluations and one plain SGD update for each resolvable normal step. Count scheduler calls at the original epoch cadence, with none inside a probe. Repeat with the optional post-loss diagnostic enabled and verify identical final parameters, cache, and RNG to the diagnostic-disabled run.

### 7. Scope and work counters

On CPU fixtures and initial GPU smoke, require EWC/Fisher, weight merge, replay, final projection/decomposition, create_graph, HVP, and outer-location gradient calls all zero. Require at least one valid-history step after the first batch and one reset per task. Inspect logged ell_f and historical coefficient ell_f/(2*ell_b); do not infer that the history code executes from a config flag alone.

## Falsifiable first experiment

Compare (A) prediction mirror U=2g-g_f and (B) history fusion at w=.5, lambda=ell_f. Use the same ImageNet-A task/class order, rank10, seed2, all training hyperparameters, initial weights, batches, augmentations, and per-update stochastic realization. Both cost two new gradients per ordinary update. Report task-macro FAA, AAA, and BWT, plus actual gradient/forward calls. The hypothesis is that using the cached historical trajectory improves generalization/retention at equal two-gradient cost, not that the method guarantees flatter minima or lower curvature each update.

Minimum diagnostic fields are g dot U, ||U||/||g||, cos(u,v), ell_f, measured future displacement norm, ell_b, ell_f/(2*ell_b), ||g-g_f||, ||g_p-g||, historical validity/reset reason, and isolated base/future/post-commit loss. These are passive diagnostics and must not trigger norm matching, projection, or LR modification in version one.
