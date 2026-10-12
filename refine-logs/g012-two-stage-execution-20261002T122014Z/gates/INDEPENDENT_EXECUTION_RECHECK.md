# Independent execution recheck

Reviewed at 2026-10-02 12:55 UTC, after the executor's fixes. Same fresh code-nonauthor reviewer and independent queue subreviewer; same-family semantic review remains provisional. This review used CPU-only tests, in-memory fixtures and read-only access to existing real smoke outputs. It did not launch or claim a GPU, change execution code/configs/pins, or control the running queue. Only this report was written.

**Verdict: PASS for Stage1 bootstrap and current output-schema compatibility. Stage2 remains conditional on a successful full M3 HVP gate; no passing initial math HVP evidence is claimed.**

No current Stage1 launch blocker was found. The earlier review is retained as a record of the original defects; this recheck records their disposition against the frozen updated files below.

## Fix dispositions

| Earlier issue | Current disposition | Evidence |
|---|---|---|
| R1: HVP completion accepted two top-level flags alone | Fixed for current gate schema. Backend/seed/reference mode, tasks0/4/9, exactly two 128-example batches each, current-task range, zero updates/forbidden work, Hstart/Hmid, same-point threshold, frozen h-grid and independently recomputed adjacent FD acceptance are checked. | `validate_job.py:15-54` |
| R2: Wrong-condition/no-HVP/NaN training trace accepted | Fixed for current trainer schema. Required summary check keys, every step's task/epoch/batch coordinates, branch/direction/mode/backend/seed, class range, finite CE, full restoration values, derivative accounting and actual matched error are checked. | `validate_job.py:98-162`; actual Controller CPU schema test and real smoke receipts below |
| R3: Worker launched before any launch intent was persisted | Fixed for safe recovery of recorded launch intent. Intent is persisted before Popen; worker publishes identity before CUDA work. Restart adopts a recorded worker or quarantines unknown launches. | `run_queue.py:55-65,163-179`, `job_worker.py:15` |
| R4: Null process identity crashed recovery | Fixed. Missing or incomplete process identities are treated as not alive and enter receipt processing. | `common.py:94-98`, `run_queue.py:79-98`; CPU calls with None and {} both returned False |

Recovery remains deliberately conservative: a crash between runtime-directory creation and launch-intent persistence can leave pending plus a directory; a restart before worker-identity publication can quarantine a live worker. These cases stop rather than duplicate work. Artifacts and inherited GPU leases remain preserved. This is an acceptable safe-stop limitation, not a new bootstrap blocker; host inspection is needed before manually recovering such an entry.

Completed dependencies are fully revalidated on controller restart (`run_queue.py:66-69`); their summary/report receipt hashes are rechecked before a dependent launch (`run_queue.py:125-130`). A long-running controller does not rehash every dependency trace/matrix at every launch. HVP checkpoint/source/code provenance is generated and checked by the pinned gate implementation; the result validator does not independently reopen every source/checkpoint file. These are assurance limits, not evidence that current artifacts are mismatched.

## Actual schema compatibility and CPU evidence

- The frozen queue dry-run succeeded with `CUDA_VISIBLE_DEVICES=''`: **51 jobs, 24 formal training, 24 smokes, allowed physical GPUs [2,3,4,5,6], max_parallel 5**. Source manifest, current RUN_PINS and training config pins passed. The two extra jobs are read-only mechanism diagnostics.
- An independent in-memory fixture used the actual `four_probe_train.Controller.install()` optimizer/batch handlers on a small CPU LoRA+FC CE model, with a real FC20-to-FC40 expansion and two steps per condition. All **24 distinct smoke conditions** produced JSON records accepted by the latest strict validator. The emitted update records came from the real four-probe and HVP implementations, rather than handcrafted trace dictionaries. Deliberately replacing each condition's row backend or inserting NaN CE was rejected. No files were written by this fixture. This establishes schema compatibility, not real-ViT numerical feasibility.
- Existing CPU mathematical tests passed: 8 direction tests, arbitrary-HVP analytic error 0/exact restore, and 4 archived-state/mechanism tests. The mechanism tests cover exact overlay with atomic rejection, common four-cell state/RNG, Sec/Hstart/Hmid quadratic identities and transactional failure recovery.
- The first three **existing real GPU smoke outputs** were read and independently passed `validate_job`: `smoke_S1-01_CE_natural_s2`, `smoke_S1-02_Splus_natural_s2`, `smoke_S1-03_Sminus_natural_s2`. Each has `status=completed`, 2 tasks and 2 verified updates. The reviewer did not run these GPU jobs. This confirms the stricter validator accepts the real trainer's current JSON schema.
- Directly applying `validate_hvp_gate` to the preserved failed initial math report raised AssertionError. Its failure cannot be used as M3 success.

The complete Stage1 formal budget is still gated by completion of all 16 Stage1 smokes. The three successful smokes above are not a claim that all smokes or any formal training have finished.

## Evaluation and mechanism diagnostics

`four_probe_train.py:24-32` now wraps observation/evaluation with the shared `response_hvp.State` transaction, so trainables, gradients, buffers, modes, named LoRA caches/controls and RNG are restored and checked. The actual evaluation wrappers use it (`four_probe_train.py:205-220`), and observation saves use it (`four_probe_train.py:238-250`).

The previously missing planned mechanism diagnostics have been implemented and scheduled (`build_suite.py:90-94,107-113`). They read the common A_P reference's tasks0/1/4/9, entry/epoch1/end trainable archives, pin reference assets, preregister at most two real batches per task before examining directions, and submit zero optimizer updates (`mechanism_diagnostics.py:189-206,255-306,308-383`). The Stage1 diagnostic compares P/D/C/R; the response diagnostic uses common math Sec/Hstart/Hmid endpoints and reports response error/cosine/norm ratio, while preserving state. This does not mix the different branches' training trajectories as common checkpoints.

During recheck, newly added `kind=diagnostic` jobs initially lacked an appropriate completion branch; this was reported promptly and fixed before the final frozen snapshot. The latest validator reads the declared `report_file`, checks all 12 task/point coordinates and 24 state-batches, correct backend/seed, zero updates, registry hash, restoration and expected candidates (`validate_job.py:58-92`). The queue's dependency hash path also honors `report_file` (`run_queue.py:127`). No diagnostic completion schema blocker remains. The diagnostics have not been asserted to have run.

## GPU and Stage2 constraints

Physical GPU0/1 exclusion, idle admission and index+UUID lease inheritance remain unchanged and effective in the queue path. Logical device0 remains the single admitted UUID, rather than physical GPU0. All current training jobs preserve the frozen seed2/rank10/rho.05/r.25/FP32 protocol and active zero-momentum, zero-decay pure SGD direction path.

The original native initial HVP gate remains failed for unsupported efficient-attention double backward. The original math initial HVP gate remains failed for lack of two adjacent accepted finite-difference h values (best relative errors .01169 Hstart and .01789 Hmid versus .005 maximum). No threshold/grid was changed and no failure record was overwritten. Stage1's native training can proceed because it does not require HVP. Every Stage2 smoke/formal still depends on a validated M3 task0/4/9, two-batch full HVP report. If M3 fails, Stage2 must stay unstarted.

## Frozen reviewed hashes

```json
{
  "trainer/four_probe_train.py": "bea9194a0ba4b73ae569f47d01da54dac0fd694519e9fb71b7740a124964d7ef",
  "validate_job.py": "0345dddb999ee949f09e7b4f9a56d29896ca493ac039cd6f69b268bb49862b1f",
  "run_queue.py": "2f6641fc00531356c4c378b092a980a84d1f624466e24330f24397eae7c4df49",
  "job_worker.py": "270e0e2a8b5204d8a0dcb9ed61efec0f6ea3d12e6923039d3d145b1749129dff",
  "common.py": "402391f3753335107a21bc7f2fef8a3c682619b175e76d453d5174f2905f6b26",
  "trainer/mechanism_diagnostics.py": "b580accfaff3938abb2ec0822de455704fa5a7812c7aa2532a481b9a04602941",
  "suite.json": "7fbf5889587c39341388ba889dac269b20014a3b88ffe5a2f034dd152646c5b1",
  "RUN_PINS.json": "a66b701757a179edb13ae3f57ce42fd0d6967816f7d6a191b0a38f62e61a64bc"
}
```
