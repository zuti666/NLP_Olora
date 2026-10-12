# TrueSAM v2 independent review

Reviewed on 2026-10-01 by the independent results-audit agent.

**Disposition: pass for the fixed Task-0 protocol, subject to the separately managed GPU validation and fresh-output run.** The patch checks the actual SAM algorithm against its own probe, gradient and SGD specifications. Cross-implementation GAM/SAM differences remain measurements. No blocking implementation defect was found in this scope.

## Artifacts and identity

- Reviewed `trainer/task0_trainer.py`, `trainer/v1_to_v2.diff`, `trainer/audit_core.py`, `trainer/test_recovery_cpu.py`, `trainer/RECOVERY_CPU_TEST_REPORT.json`, `trainer/cpu_test_report.json`, `trainer/RECOVERY_NOTES.md` and `trainer/PATCH_MANIFEST.json`.
- Compared the probe specification against frozen `s3-hparam-search-20260921/source/optimer_PerturabtionType/optimer_sam.py`, particularly `first_step`, `second_step` and `_grad_norm`.
- Independently recomputed the hashes below; all match `PATCH_MANIFEST.json`.

| Artifact | SHA-256 |
|---|---|
| Immutable v1 trainer | `b6632580c900ee91650d1f6a2c2ab8ae0e760348942bde69799ecf0cc79ae746` |
| Recovery v2 trainer | `1c6cc0b2e18073eea3aabd972c16b1ecceb3e12110d4fd70d17fbcff64c75659` |
| Recovery audit core | `9a98b92acebe66a4e5d9ea5d37abc718144c1fb68392f13ab3bc049e561eaf96` |

## Code findings

1. **Own-SAM probe:** `sam_probe_specification` uses the frozen SAM implementation's same-dtype norm-of-tensor-norms reduction, rho normalization and nonadaptive perturbation. Every sampled probe parameter must match the independent reconstruction exactly, and the saved clean parameters must match their captured originals exactly. Adaptive SAM is explicitly rejected.
2. **Independent self-gradient:** At the actual SAM probe, the reference uses explicit mean CE over exactly 20 logits and `torch.autograd.grad`. A `finally` block restores model parameters, gradients, buffers, training modes, BatchNorm auxiliary state and RNG through `Snapshot.restore`; `Snapshot.check` verifies parameters, gradients, buffers, modes and RNG. The subsequent training backward is compared tensor by tensor against this reference. Finite checks and the existing whole-vector relative-error gate remain enabled.
3. **Restoration and SGD:** Before the sampled base step, the restored parameters must equal the saved clean copies exactly. The base SGD update is checked against a same-dtype prediction starting at these restored parameters. The checker rejects nonzero momentum or weight decay. This avoids confusing finite-precision subtraction of the total displacement with an incorrect SGD operation.
4. **Cross-GAM measurement:** The isolated GAM diagnostic restores its snapshot before real SAM training. Actual SAM-g1 versus isolated GAM-g1 relative error, maximum absolute error and cosine are reported separately, with `used_as_correctness_gate=false`. The clean-g0 relative difference and same-gradient norm-reduction difference are also measured. The geometry source explicitly identifies the isolated GAM diagnostic. This is appropriate because the algorithms have distinct floating-point computation paths; mathematical equality does not require bitwise equality.
5. **Scope of gates:** Probe, independent gradient, restoration and SGD checks sample the first batch of each epoch. Every SAM batch checks clean/perturbed loss and existing gradient tensors for finiteness. This is not an every-batch proof of update equivalence. The fixed ViT protocol rejects stochastic dropout/drop-path modules and uses zero momentum/weight decay.

## Independent read-only CPU execution

The reviewer reran `test_recovery_cpu.main()` in a separate CPU-only process using `CUDA_VISIBLE_DEVICES=''` and `PYTHONDONTWRITEBYTECODE=1`; `torch.cuda.is_available()` was asserted false. Serialization was replaced in memory by a no-op, and directory creation was replaced by a check that the directory already existed. No test output, source file or checkpoint was modified.

The rerun reproduced:

- Exact parameter and RNG equivalence with bare frozen SAM.
- Exact own-probe reconstruction.
- Independent self-gradient maximum absolute error `0.0` on six synthetic model tensors.
- Same-dtype base SGD prediction maximum absolute error `0.0`.
- Injected 1% change to isolated GAM-g1 produced cross-comparison relative error `0.009900977909472928` while leaving the actual training update exactly unchanged.
- Rejection of the existing incorrect-probe, incorrect-self-gradient and nonfinite-gradient cases.

The reviewer additionally injected two independent negative cases in memory:

| Negative case | Injection | Observed outcome |
|---|---|---|
| Corrupted saved clean restoration | Add `0.01` to the first parameter's saved `old_p` before native `second_step` | Rejected: `Native SAM failed to restore its own saved clean parameters` |
| Corrupted SGD result | Register a base-optimizer post-step hook that adds `0.01` to its first parameter | Rejected by the retained SGD-result gate: `Actual update does not match selected gradient` |

Both additional cases reused an existing test directory and disabled all serialization. They did not alter the archived CPU report; this document records their execution evidence. The process exited successfully with `INDEPENDENT_READ_ONLY_CPU_REVIEW_PASSED`.

## Interpretation and remaining limits

- The old `1.3368781e-4` GPU discrepancy was a cross-SAM/GAM diagnostic threshold violation. Using that comparison as the definition of SAM correctness was inappropriate. Neither this CPU review nor the archived CPU report quantitatively reproduces or fully attributes that specific ViT GPU discrepancy. Different L2 reduction orders establish a plausible source of finite-precision differences, not a complete causal explanation.
- The independent CE check shares the model and CE implementation with training. It checks probe location, differentiation path, restoration and update behavior; it is not an independent validation of dataset semantics or model correctness.
- TrueSAM v2 has an additional sampled reference-gradient pass, separately counted as `independent_sam_probe_gradient_passes`. Report throughput as including the instrumentation cost; do not describe it as uninstrumented SAM runtime or as compute-matched to the five-gradient GAM arms.
- The plain `--max-epochs 14` option changes both `init_epoch` and `epochs`, and consequently the cosine schedule. Such a smoke would not reproduce the original 50-epoch schedule's epoch-13 state. This issue was reported to the root agent. The root agent states that a separate validation entry preserves `epochs/T_max=50` while limiting the execution budget to 14 epochs; that entry and its GPU outcomes are outside this independent patch review and should be verified in the deployment receipt.
- The five completed main arms retain v1 provenance; recovery runs should retain explicit v2 provenance and fresh output paths. The reviewed diff does not change main-arm gradient selection or the frozen optimizer implementation.

No GPU process was started by this reviewer. This review does not assert that the GPU preflight or the 50-epoch recovery has completed.
