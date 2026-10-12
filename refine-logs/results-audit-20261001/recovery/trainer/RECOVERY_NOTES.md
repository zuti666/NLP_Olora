# True SAM audit recovery v2

The immutable v1 campaign stopped True SAM at the first batch of epoch13 because its gradient differed from the independently sampled GAM-g1 by about `1.336878e-4` relative to GAM-g1. The saved SGD audit reported zero error against its same-dtype prediction. The blocking condition was therefore an invalid cross-implementation equivalence assumption, rather than evidence of an incorrect SGD update.

The two frozen optimizers calculate the same mathematical L2 norm with different floating-point reductions: GAM sums per-tensor squared-gradient sums and then square-roots the accumulated result (`gam.py:168-200`); SAM norms each tensor and norms the resulting vector (`optimer_sam.py:115-128`). They also restore perturbations differently: GAM subtracts saved perturbations, while SAM assigns the saved clean copy. These facts explain why bitwise equivalence is not guaranteed. They do not, without a paired GPU replay, quantitatively identify every source of the observed epoch13 discrepancy. Restoration occurs after the first perturbed gradient, so it cannot by itself explain g1 differences within one freshly restored paired anchor.

This version keeps the data generator, initialization, epoch loop, two-gradient SAM update, optimizer, hyperparameters, and main GAM arms unchanged. It does not loosen the cross-algorithm threshold: that threshold is removed from the definition of algorithm correctness and the difference is retained as a measurement.

On each sampled first batch, True SAM now has these independent checks:

1. Capture its actual g0 and independently reconstruct its own nonadaptive rho=.05 probe with the same tensor dtype and reduction order as frozen SAM. Compare every resulting parameter exactly; verify the saved clean parameter copies exactly.
2. At that actual SAM probe, independently differentiate explicit 20-class CE with `autograd.grad`. Restore all parameters, gradients, buffers, modes and RNG before executing the training forward/backward. Check actual gradients against this reference tensor by tensor; reject nonfinite values.
3. Confirm `second_step` restored the exact saved clean parameters before SGD; retain the existing same-dtype SGD prediction hard check.

`selected_gradient_relative_error` now refers to the independent gradient at SAM's own probe. `cross_algorithm_comparison` separately reports actual SAM-g1 versus isolated GAM-g1 relative/max-absolute error and cosine, with `used_as_correctness_gate=false`. The original g0..g3 geometry remains explicitly labeled as an isolated GAM diagnostic for True SAM. Every training batch still has two SAM gradient passes. The extra first-batch reference pass has a separate `independent_sam_probe_gradient_passes` counter; its time is included in throughput measurements.

CPU validation includes the original six-direction/native-epoch-loop and low/zero-LR regression, exact parameter/RNG equivalence against bare frozen SAM, an intentionally injected 1% cross-GAM discrepancy that does not alter training, and hard rejection of corrupted probe/gradient/nonfinite cases. The tests do not claim to reproduce the exact ViT GPU epoch13 discrepancy. A fresh GPU smoke remains necessary before deployment.

CLI is unchanged: `task0_trainer.py --source FROZEN_SOURCE --config ARM.json --output-dir NEW_RUN_DIR`. Use a fresh output directory; do not overwrite the old failed run. No resume behavior is added. The immutable v1 and frozen research source are untouched.
