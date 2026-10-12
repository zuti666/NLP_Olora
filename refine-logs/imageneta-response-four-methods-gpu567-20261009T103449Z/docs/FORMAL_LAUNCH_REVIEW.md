# Formal launch verification

Passed independent read-only verification on the actual host. The controller process and recorded start time match and remain alive. Exactly two formal runs are active: full-200 SGD on physical GPU 5 and current-only relocation on physical GPU 6. GPU 7 is absent. The worker and training processes are bound to the recorded GPU UUIDs; no campaign compute process runs on GPUs 0–4. Five remaining formal jobs are pending.

Both seeded CUDA kernel witnesses passed. Each method has at least ten independently checked real SGD updates with finite losses and positive committed parameter displacement. All restoration, deterministic RNG, purity, full-200 classifier and ordinary-SGD checks passed. SGD reads one gradient; current-only relocation reads three gradients and submits only g2 from a relocation originating at the base parameters. Each also performs one passive post-update diagnostic forward. Neither uses history, EWC/Fisher, model-weight fusion, projection, HVP or final norm rescaling.

The first-step base parameters, frozen parameters, new classifier initialization, realized input/target batch, learning rate, base loss and raw g0 gradient are bitwise paired. The first ten realized input/label/order identities also match between methods despite different gradient-probe budgets. Both commands request full 50-epoch training and include no bounded-smoke CLI limits. Frozen run pins, all 190 source files and accepted prior implementation evidence were reverified.

This verifies actual formal startup and implementation behavior. No new run is complete yet, so there is no complete performance conclusion.
