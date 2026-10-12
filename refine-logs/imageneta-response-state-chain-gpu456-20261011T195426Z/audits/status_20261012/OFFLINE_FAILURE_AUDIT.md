# Offline numerical-gate failure audit — 2026-10-12

Read-only source/result review; no CUDA jobs, source edits, threshold changes, or original output overwrites.

The first formal offline job `offline_s2` returned **exit 2**, deliberately, at **2026-10-11 20:50:56 UTC**. This is a numerical validation failure, with no OOM, NaN, uncaught exception, or training update. The first state is **seed 2, S branch, Task1 epoch 1, fixed train batch 0** (128 samples). All restoration fields are true.

| Check | Result | Registered acceptance |
|---|---:|---|
| Native repeated gradient | exact | exact |
| Math/microbatch original gradient relative error | 0.0000209829 | <=0.0002 |
| Math/microbatch original gradient cosine | 0.999999999780 | >=0.99999 |
| H g0 FD errors at .001/.003/.01 | 0.008601 / 0.082600 / 0.111084 | two adjacent <=0.08 |
| H detach(H g0) FD errors at .001/.003/.01 | 0.501514 / 0.975109 / 0.984655 | two adjacent <=0.08 |

Both Hessian FD gates fail. The first has only one passing point; the second has none. Increasing distance increases disagreement, which is compatible with finite-distance truncation/nonlinearity; this is an interpretation, not an established root cause. Static code review found no concrete coordinate-order, scaling, base-state, or second-vector-detachment bug. The misleading boolean field `numerical_implementation_failed` means the declared numerical implementation *validation* failed; it does not establish an implementation defect.

Implementation checked: `offline_core.py` lines100–128 differentiates the weighted native CE mean, with a detached HVP vector, and uses central differences at ±distance*N(vector). Lines152–172 compute both exact products at the restored base and require two adjacent grid points. `offline_diagnostics.py` lines107–114 intentionally returns2; lines97–101 also couple effective-step matching to this unrelated Hessian gate. These files still match `RUN_PINS.json`. Eight CPU algebra tests previously passed and the short smoke checkpoint's Hessian gate passed, but that does not validate this later formal checkpoint.

Coverage: **1/36** state/batch combinations inspected, **0/36** accepted; **0/72** accepted correction-scale records and **0/24** effective-step matched records. Original seed0/seed1 diagnostic jobs never started. All 18 required formal branch/epoch1,10,25 snapshots are present. Formal training validity is independently assessed; this failure does not invalidate accuracy or native-gradient training.

No claim connecting the response to H²g is currently justified. Preserve the failed grid and registered threshold. A justified *separate exploratory follow-up* would study smaller FD distances with the same state and threshold (convergence vs roundoff), recording the original failure and all points; passing that new grid cannot be reported as passing the original protocol. Exact effective-step matching needs only native gradients and BA/FC algebra, so it can be recovered independently in a separately reviewed script without invoking or accepting Hessian correspondence. No follow-up was launched by this audit.

Evidence is indexed and SHA256 pinned in `OFFLINE_FAILURE_AUDIT.json`; causal record: `outputs/offline_s2/checkpoint00_batch00.json`; job exit: `runtime/jobs/offline_s2/exit.json`.
