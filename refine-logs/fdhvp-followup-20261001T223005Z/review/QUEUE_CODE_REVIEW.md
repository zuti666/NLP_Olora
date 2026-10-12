# FD/HVP queue independent review

Verdict: PASS for deployment review and parent-gated execution. No blocking defect found in this scope. No GPU, queue controller, waiter main, worker, or deployment was started by this review.

The 42-job graph contains 22 short GPU smokes, 17 potential full horizons, and 3 UG checkpoint curvature diagnostics. Conditional branches can be skipped; 17 is a registered maximum, not a promise that 17 horizons will run. `SUITE_DAG_INDEPENDENT.json` checks exact config pins/response identities, dependencies, smoke groups, and the UG Task0/4/9 checkpoint paths.

## Parent barrier and import isolation

`test_parent_handoff_independent.py` executes the real `parent_ready`, `verify_parent_release`, `verify_pins`, and `verify_gpu_smoke_release` functions against temporary fixtures. All 49 assertions passed; positive fixtures are explicitly required to pass, and unexpected exception types are fixture failures rather than successful negative tests. The JSON report binds the test and all implementation hashes before and after execution.

The fixture starts and observes a real CPU process in the current PID namespace, captures its identity, then proves it exited. It does not infer host-process exit from an invisible sandbox PID. Both an active queue controller and an active parent handoff process independently block release; incomplete or failed parents block; wrong namespace blocks even when both configured process IDs are inaccessible.

The test process deliberately already imports the child's `common`. The real `validate_parent.py` subprocess nevertheless imports a synthetic parent's `common` and `validate_job`, with explicit parent-directory/marker assertions, and validates each of all 25 raw outputs. This directly checks the cross-campaign import-isolation fix.

Before initial release, the waiter verifies parent pins and all raw completion outputs through the parent's validator. Later child launches recheck the validated receipt hashes, current parent pinned files, controller/handoff completion, and the common frozen source through child `verify_pins`. This receipt proof is not described as re-reading all historical parent model outputs at every launch.

Suite/pin manifest/pinned implementation/source/receipt/exit tampering, 24-of-25 completion, incomplete proof, and smoke group/count/hash/receipt substitutions were rejected. A full valid fixture was reaccepted after restoring tampered bytes.

## Conditional graph, smoke verification, and terminal collection

`apply_skips` runs before GPU selection. It verifies canonical upstream skip receipts and recomputes each job's own gate before writing `skipped.json`. `verify_skip` compares the exact current semantic gate result through `verify_skipped`, including bound evidence. A skipped dependency cannot silently become a successful training completion.

A pending runnable job must have every dependency completed, gate decision `ready`, current parent release, applicable group-specific GPU smoke proof, revalidated dependency outputs, current source/config pins, and an idle GPU. A skipped job never reaches the lease acquisition block. Group smoke proof binds the exact current membership, suite, frozen gates, and receipt hashes. The separate smoke verifier also checks real task-transition/input/head/LR pairing and the US identity control.

Collector runs after completed jobs and after propagated skips. It revalidates canonical skips, lists them separately, excludes smoke metrics from formal tables, and leaves absent/skipped numerical results absent. Final queue completion counts completed and skipped terminal jobs; it does not imply every registered branch trained. Existing conditional/collector CPU suites were not replaced by this review.

## Process and deployment checks

`start_waiting.py` and `deploy_run.py` check hostname and exact PID namespace before creating a waiter or copying the deployment. Deployment requires a fresh destination, pinned source, matching environment versions/spec/witness, and writes a new pin manifest. Waiting itself takes no GPU lease.

Queue idle selection checks GPU UUID and utilization twice around the shared flock. The lease FD is inherited by `job_worker.py` and again by the actual runner, keeping the lock alive through worker/controller death while the runner remains alive. Restarted running jobs need a real exit receipt; no missing exit becomes success. These FD and restart paths retain the prior independently tested implementation; this review did not launch a real GPU worker.

## Limits

CPU fixtures establish guard semantics and import isolation, not full ViT numerical correctness or GPU availability. Actual Task0-to-Task1 GPU smoke acceptance is still required by the queue. The parent barrier's raw-output validator is a synthetic implementation in this CPU test; real native completion schemas have their separate trainer/validator CPU and subsequent GPU checks.

## Pinned implementation hashes

```json
{
  "run_queue.py": "cedcdf6e284c740a7a446f8c5743e355c03f758bbd73b4873564870c2bfd9607",
  "wait_then_launch.py": "17895c280dec9a6daa71d5e3d84549afeb85705873df1651ed67eb761227ab90",
  "validate_parent.py": "65cf2a6ccaffc9fa0e4ba1cebfb7dd23c7f2f7374f48169b4dc55360bce1ab25",
  "start_waiting.py": "3a74e395c8c2247e7a2a24461c8f1b500d2f673aa46a5286f8d9b5d079db0bcf",
  "deploy_run.py": "2bf6d87b972bc092c798ef6ea5944a00d94b7c2c385e528ea7ac38d077db3ab3",
  "common.py": "a8b750e65b599445d97476a455cbd9f9fec94aab87b2f9023e6e1d0d17c16042",
  "validate_job.py": "528c66c4694aa5bae06c060422acee8ce288afb3db0a6cfb978cf251aa8375d4",
  "conditional_gate.py": "9fd504beace1c6d00dba47de4c16447f54a44ef723cf5ac1a0f91da5e487e010",
  "check_gpu_smokes.py": "1dd347559a4d4d29ab0d7499cb549026f7049ffe81ddb4be1167f23678a71f53",
  "collect_results.py": "3ea9dc1b7d1252ae6019e916c87311ac8ab8f1c15f76af307c4a868f90725064",
  "suite.json": "a654828718f3743813b53a23da5039a76b863ae88fca8fdb72f63b4eb7e5069f",
  "FROZEN_GATES.json": "a5f3b368eed77776da8e4d640fbfd678ccab9c6b89acdf8610f25c0432cd00c9",
  "job_worker.py": "658b451359124a72b3caaf4434d5cd6ea7f9612fdc50467e555edf73588f69c7"
}
```
