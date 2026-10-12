# FD/HVP training completion acceptance

`validate_job.py` keeps the inherited native lifecycle, no-Fisher/no-merge,
checkpoint, current-task loss slice, effective BA+FC step matching, and final
matrix checks. The new training branch additionally binds the first/second
response, correction coefficient and attention backend across the job, input
and effective configurations, summary, every step, and each checkpoint.

Each available HVP must come from the actual original closure 0 or outer
closure 2, use the detached native-normalized local gradient, pass same-point
CE/gradient comparisons, and restore parameters, frozen versions, buffers,
modes, gradients and RNG. Its correction includes `rho = 0.05`; both the exact
operation gate and the independently reported correction/HVP norm ratio are
checked. An unsupported HVP or finite-difference fallback cannot be accepted
as completed HVP training.

The actual selected, projected, and SGD gradient hashes are checked along with
their own-specification gates and exact SGD tensor prediction. Thus HVP
availability alone is insufficient. Native four closures/five CE-gradient
passes are counted separately from the extra HVP forward, first derivative,
and second derivative. Math controls must report all four actual SDPA flags,
including disabled cuDNN attention. Matched runs retain the positive actual
effective BA+FC norm constraint with relative error at most `1e-3`.

After paired jobs finish, call
`validate_paired_inputs(jobs, required_backend='math')` to check their actual
input/target/index digests, index order, augmentation seed, loss slice and
nominal LR at every step. Individual job validation cannot establish this
cross-job property by itself. This does not claim equal parameters, gradients,
effective updates, or learning trajectories.

CPU regression command (no GPU):

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B \
  review/test_fdhvp_validation_cpu.py --output review/FDHVP_VALIDATOR_CPU.json
```

The fixture runs real frozen native GAM, the actual new component wrapper,
and actual autograd HVPs on a small A/B+FC CPU model over two successive tasks.
Its completion envelopes and accuracy matrices are explicitly synthetic test
data. Every negative is tested only after its original complete fixture has
been accepted; `KeyError` or other incomplete-fixture errors do not count as a
successful rejection. Protocol changes must rerun the suite and retain the
reported executable hashes. GPU double-backward support and full native data
loading remain the separate smoke test's responsibility.
