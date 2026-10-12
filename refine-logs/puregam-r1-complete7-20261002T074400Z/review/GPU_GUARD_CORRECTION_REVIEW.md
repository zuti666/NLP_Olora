# Independent review of the GPU release guard correction

**Decision: the revised evidence standard can support algorithm-implementation fidelity and release of the three performance runs. It does not establish cross-GPU numerical trajectory equivalence.** This is a review of the stated correction and existing raw smoke evidence, not certification of a future queue implementation that was not yet present when reviewed. Reviewed 2026-10-02; no GPU process started and no old script, threshold, receipt or output modified.

## Independently checked evidence

Evidence root: `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T073400Z`.

All five smoke jobs have completed receipts. Each has 18 training steps and 2,256 training examples across two tasks. I parsed all 90 trace rows and independently checked:

- Four fresh native closures, five model forwards and one actual SGD step per batch.
- Exact native g0/g1/g2/g3 cache checks, selected-direction checks, original clean-projector checks and actual-SGD checks.
- `selected_expected == selected_actual` and `projected_expected == projected_actual == actual_sgd_gradient`, including recorded gradient hashes/norms.
- Predicted versus actual post-SGD parameter hashes are equal.
- Empty Fisher/delta-penalty caches and unchanged RNG across the selection, projector and SGD audit segments.
- Trace SHA equals the summary binding and completed-receipt binding; summary SHA equals its completed-receipt binding.

All five jobs used extension SHA `8776dc581d93e18f967d6fc20d17ce9bce508a06b49393621792832bec5873fc` and native GAM SHA `0b37fca57a31952d911eba4ab99534b7952a6271c1f2c37dbf7a45b72fa5e40e`. The preceding independent CPU review established five selectors × four steps and eight negative tests, including exact agreement with a wholly unpatched native reference for both g1 controls. That CPU report is `review/LEGACY_DIRECTION_CPU.json`, SHA `15d5c1b94f883f14d91db8469b1b01786f176e23df4e324bc3a171f59c63efb5`.

The native g1 control ran on physical GPU 1 and the wrapped g1 control on GPU 2; both receipts identify NVIDIA RTX 6000 Ada Generation. Their 18/18 input hashes, target hashes, pre/post-batch RNG hashes, task/slice/order fields and learning rates are identical. Their CNN matrices and sample-weighted accuracy curves are also identical.

At the first batch, the first recorded probe's **trainable-parameter** hash is identical, but the native clean-gradient hash and g0 hash already differ. g0 is evaluated before any decomposition or final-direction selection. The g0 norm summaries are `4.327186423466665` and `4.327186423960445`; this norm difference is not a vector-error estimate. These observations show that numerical divergence is already present before the extension's differing decomposition-hook behavior. They are consistent with native execution variability across separate GPU runs, but do not isolate its exact kernel/backend cause or prove every later checkpoint difference is attributable to one cause.

The old guard correctly remains failed. `runtime/jobs/formal_g0_r1_k10_s2/queue_failed.json` records at `2026-10-02T07:38:53.320375+00:00`:

> Control parameter deviation exceeds predeclared tolerance: lora_w_b_1.pt/0/state_dict/weight

Its SHA is `e81e3467be5fedf32cc4cf9a0ddf59cdcf8a033ab3ec6ecdc359cd82438a14cd`. The old state shows five completed smokes, one formal job rejected by the guard and two formal jobs pending; the rejection happened before formal training launch. The old numerical threshold must not be changed or reported as passed.

## Why the replacement guard is appropriate

An algorithm-fidelity guard should test the specified operations at the actual state of a run. Here both controls independently verify that the four native probe caches are current, the unmodified native g1 decomposition produces its specified g1 value, the original FP32 projector executes `clean + (selected - clean)`, and the ordinary base SGD applies exactly the recorded gradient at the native restored point. All three new direction smokes also passed their own formula and update checks: g0, g2, and `g0 + (g3 - g2)`.

These exact within-run equalities, immutable native source, instance-only patch scope and CPU comparison against an unpatched oracle directly test the extension's intended change. They are a more targeted release criterion than demanding a separately executed native GPU trajectory stay within an unvalidated endpoint reproducibility tolerance. The cross-run failure remains useful reproducibility evidence and must be carried forward separately.

This is a transparent **change in the release question**, made after inspecting a failed control, not evidence that the original equivalence test passed. It supports executing the authorized performance experiments; it cannot retroactively establish cross-run checkpoint equivalence or a pre-registered numerical equivalence margin.

## Required binding of the new campaign

The new formal-only campaign should verify and pin all five exact completed smoke receipts, their trace and summary hashes, the CPU-oracle report, extension/native-source hashes, and the old failed-control receipt. It should retain the within-run exact checks for every formal training batch, rather than relying only on a summarized `passed` flag. It must still reject missing rows, wrong formulas, missing probes, nonfinite values, changed RNG, nonempty regularization caches, incorrect projection or an SGD mismatch.

Configs must retain the previously reviewed historical protocol and natural update scale; only the intended selectors, fresh output paths and required scheduling provenance may differ. The new GPU allowlist should be exactly `[4, 5, 6]`, with three formal jobs and the existing shared GPU locks. Old failed directories must not be reused, and unrelated curvature failures/old queue states must not be relabeled successful. This review did not run the new queue or inspect a deployed new allowlist.

## Remaining scientific limits

- GPU execution reproducibility at the old checkpoint tolerance is unproven and, for the observed pair, failed. Matching RNG/data does not imply numerically identical gradients or trajectories.
- Own-spec checks establish the specified calculation at its observed state, not that each complete run would equal a different independently executed run. The CPU oracle is stronger intervention evidence but is not a full ViT/GPU oracle.
- The new g0 still retains all native probes and floating-point perturb/unperturb operations; it is not ordinary one-forward CE-SGD. UG is a finite-rho correction, not an established exact HVP.
- Reused historical rows and new rows remain single-seed performance evidence under the intended same protocol. Small differences should not be described as significant, robust, or proof of universal component necessity. This guard correction should be disclosed if checkpoint reproducibility is later discussed.

Evidence snapshot pins: old `suite.json=13ceb61d20a0c26d6d9ce48b4466452489bd9145d32cfe4a45bae6cdaed6ebb1`; old `RUN_PINS.json=20e6e20a3913c2c79f275d8cb40b91b01a4265c2e1e376cd692b502b92588915`.
