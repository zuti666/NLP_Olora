# Independent review: native r1 learning-rate controls

**Verdict: PASS for CPU correctness and static launch readiness.** No blocking issue remains in the reviewed implementation. This report approves readiness of the frozen files below; it does not assert that GPU smoke checks or the five formal experiments have completed. The reviewer did not launch GPU work, deploy files, restart queues, or change implementation/configuration files. Only this report was written under the campaign directory.

Reviewed staging directory: `/data/140-1/users/liying/O-LoRA/refine-logs/r1-lr-controls-20261002T191527Z`.

## Scope and protocol

The eight primary arms are S=`g1` at nominal LR, S1.5=`g1` at 1.5× LR, S2=`g1` at 2× LR, Sum=`g1+g3` at nominal LR, Mean2=`(g1+g3)/2` at 2× LR, Graw=`g3` at nominal LR, Gmatch=`g3` with its actual effective update norm matched to a separate S trajectory, and G025=`g3` with native outer radius 0.25. Original gamma=0.1 is not one of these eight arms.

Exactly five new full experiments are configured: S, S1.5, S2, Mean2, and Gmatch. Sum/Graw/G025 are reused primary controls whose protocol and raw results were independently verified. Old S is context only because a fresh S trace is required as Gmatch's independent target. Old Mean at nominal LR is supplementary, not Mean2. The initial table correctly contains three completed primary rows and five pending fresh rows; new accuracy entries remain empty.

All ten new jobs use seed 2, rank 10, rho 0.05, native full seen classifier heads, native global RNG/DataLoader handling, train/eval workers 8, mechanism evaluation/statistics enabled, and no EWC, Fisher, or fusion. Formal jobs use ten tasks of twenty classes, fifty epochs per task, 2,550 updates and 299,050 sample exposures. Five smoke jobs use two tasks and one epoch per task, eighteen updates and 2,256 exposures each. Peak classifier/backbone LRs scale together: base 0.04/0.02; S1.5 0.06/0.03; S2 and Mean2 0.08/0.04. Native cosine floors remain 0.1× the corresponding peak; native fifty-epoch scheduling is retained for formal jobs.

Each JSON uses the native gamma=0, beta=(1,1,0) configuration and must run through the selector-installing campaign runner. A bare native JSON invocation would not install the requested alternate direction. The plan and runner make this requirement explicit.

## Source and update semantics

I read the complete helper, reference loader/solver, CPU test, common validation/pin code, job runner, queue, smoke validator, collectors, table summarizer, suite, reuse report, and experiment plan. I inspected the pinned native GAM probe/decomposition/projector/unperturb/SGD sequence and the source scheduler behavior. All 190 pinned source file hashes and Python syntax checks passed.

The helper selects a direction only after the original four native probes and exact native decomposition. Original clean-gradient projection remains FP32, and native unperturb executes before the actual SGD callback. Native training still performs four closures and five CE evaluations per update. No controller, alternate DataLoader, added reseeding, or auxiliary optimizer update is introduced. Instance hooks are limited to the selected training learner and restore on exit or exception. Cache checks detect changed native inputs or gradients.

Effective geometry is measured at the actual post-unperturb point immediately inside the original base-SGD callback. The BA calculation includes the full finite update, including its quadratic factor term, plus the full seen FC head. FP64 Gram calculations measure geometry without materializing the large BA matrix; independent dense calculations verified them. Same-candidate-state projected-g1 geometry is explicitly a diagnostic and is never substituted for the Gmatch target.

Gmatch consumes a completed, separate S run of the same phase and scientific protocol. The loader verifies its receipt/config/summary/trace hashes, helper/source identity, complete budget, task sequence, expanding parameter/head shapes, inputs/targets, RNG before each update, and nominal LR. Runtime validation also verifies paired RNG after updates and revalidates the independent reference. Every target is the independent S row's actual effective joint BA+FC delta, with finite positive geometry. Source-file identities are rechecked after execution.

The solver measures actual FP32 cloned SGD updates at the candidate's restored point, searches for the smallest positive matching root, checks nonmonotonic cases, and fails on a missing, invalid, zero, nonfinite, or unreachable target. Live parameters, optimizer buffers, and RNG are not used for trial updates. Exactly one real native SGD update occurs. Temporary group LRs restore in `finally`, before scheduler advancement; the actual measured match must meet relative error <=1e-3. There is no fallback to the same-state counterfactual, no silently clamped match, and no additional real SGD trial.

## Independent executed evidence

The complete CPU suite was independently rerun with CUDA hidden, writing `/tmp/r1_lr_controls_independent_cpu.json` rather than overwriting the frozen implementation report:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B test_legacy_direction_cpu.py --output /tmp/r1_lr_controls_independent_cpu.json
```

It exited successfully with PASS. Its SHA-256 is `672ced059bd95f0f2ce69e718107af00e83e5cbcbde07c297bb23402845700f4`, identical to the frozen report. It exercises twenty selector/head cases with four native steps each against an independent selector oracle and unpatched native controls, native parameters/gradients/losses/mechanism statistics/RNG, numeric and negative checks, dense FP64 geometry, independent eighteen-step S-to-Gmatch matching through a 20-to-40-class head expansion, solver root/failure cases, and the Sum/Mean2 FP32 cancellation counterexample. The independent reference target demonstrably differs from same-candidate-state projected g1.

I additionally generated temporary CPU fixtures and exercised the production schema-3 validator:

- Accepted 38 actual native CPU rows: eighteen S, eighteen Gmatch, one Sum, and one Mean2.
- Rejected 26 corrupted records covering wrong schedule, unrestored LR, altered geometry, mutated counterfactual state, false matching/reference claims, wrong reference phase/job, zero target, excessive actual match error, live trial flags, and inconsistent matching prechecks.
- Compared 300 formal schedule positions against actual PyTorch `SGD` plus `CosineAnnealingLR` recurrence: multipliers 1/1.5/2, both task-phase nominal LRs, and all fifty epochs. All passed, including the scaled cosine floors.
- Injected an exception into the real Gmatch SGD callback. The callback observed matched LR `0.018468221137482033`; after failure, nominal LR 0.04 and all instance hooks restored exactly, and the exception propagated.

All ten suite/config hashes and their scientific protocol fields were checked independently. The frozen CPU report explicitly binds helper, reference, effective-geometry, residual, test, and native source hashes, and the run-pin verifier checks those bindings.

I independently recomputed all five reused controls from their complete native 10x10 lower-triangular accuracy matrices, preserving structurally unobserved cells. AAA, FAA, learning accuracy, BWT, and forgetting agree with reused results to numerical precision. Primary reused AAA/FAA (%): Sum 67.8608630952381/54.726; Graw 65.99317222222223/50.67900000000001; G025 66.22960873015873/50.79899999999999. Context S is 62.16553134920633/45.132, and supplementary Mean is 65.95094761904762/49.92699999999999. Historical beta/gamma/radius fields and all other shared scientific fields match their declared arms.

## Queue and GPU acceptance gate

The queue is restricted to GPU 1 with maximum concurrency one. Smoke Gmatch depends on smoke S. All five formal jobs depend on all five smoke jobs; formal Gmatch additionally depends on completed formal S. The production GPU gate validates ninety real smoke updates, paired initial state/input/target/RNG and expanding heads, full reference proof for Gmatch, and the Sum/Mean2 first-update actual joint norm within relative error <=1e-3. This is explicitly an initial-update magnitude check; it does not claim identical trajectories.

The runner verifies pinned code/source/environment/configuration and assigned GPU UUID, then validates the full update trace, budgets, metrics, reference immutability, and checkpoint hashes before an exclusive completed receipt. The scheduler rejects missing/invalid dependencies and failed jobs, prevents ambiguous reuse of partial output, checks true host namespace/PID identity and GPU availability, and uses both a campaign lock and an inherited device lease. Freeze/pin generation must bind this report before deployment, and the true-host validate-only pass plus five fresh GPU smoke checks must pass before any formal job proceeds.

## Limits of this verdict

CPU checks use small real PyTorch models and the pinned native GAM methods; they do not substitute for the five required GPU smoke checks on the full native learner. Historical controls do not have the new per-step traces, so this review does not retroactively claim their batches/RNG are paired with new runs. Gmatch isolates actual update norm against an independent S trajectory; candidate states and directions are allowed to differ. The result is one seed and supports no multi-seed significance claim. Sum at eta and Mean2 at 2eta are equivalent in exact arithmetic, but native FP32 clean projection can prevent bitwise equivalence and can lead to later trajectory differences. The campaign correctly does not gate on full-trajectory equality.

## Reviewed file hashes

The files below were unchanged during this independent review. A later edit requires revalidation and fresh run pins.

| File | SHA-256 |
| --- | --- |
| `legacy_direction.py` | `8c6b36c00aed4488d3171ed70ad2838b22cb80148c6807f31bff6b409eb38eaf` |
| `step_reference.py` | `9aeb7c1e0c1e70c9aa22cf95b506927bce4a62931a5a6af5fee3847bd05633d3` |
| `effective_step.py` | `97aa2b7bb87e8771f59b336b9b035e590bf13c8c7f514592ad2055430ed7abf3` |
| `residual_selection.py` | `4284c85df9700e1ef992cedc1bf1e2d2f784f687d680a605ace580c3ea83eab6` |
| `test_legacy_direction_cpu.py` | `3b618d5e1d95c9eceb44e9b7807f4a662857bc6de3ad2f17561cc36d89fedd59` |
| `common.py` | `b9a76f5217ab51a92caabd8d2f537dbfef590b9c873946e8d21326edde227aaa` |
| `run_job.py` | `495753468016da6d00902d11e21a9ff9b56229ab81d47c8ab4aa3bbec2b31b65` |
| `launch_queue.py` | `7085ea958d16f9f7927570b7140c271474aa6fd37f7164a6f2937eda0abf307e` |
| `validate_smokes.py` | `7dd6ef41604971e2f0e2862fe9f127c2e090474f9c4187ad5068b198ec73826e` |
| `summarize_controls.py` | `4ae979df8f63c5019f3c4b374545519741872a1da98f4c1ff927d36f52d936a3` |
| `collect_results.py` | `3dc10d8bbc29efc67866a212fc1ad6eb722535bbbdc1fae8d8cd03880de32940` |
| `suite.json` | `bf5f4b387bb67cc5569ac67a01cae0e1baff1538ceb5331da72b57ad6915e7e6` |
| `REUSE_VERIFIED.json` | `a4e8128c4708817830d5ac768ccb519240ea606e91ef096d07a4d418bfea2d94` |
| `review/LEGACY_DIRECTION_CPU.json` | `672ced059bd95f0f2ce69e718107af00e83e5cbcbde07c297bb23402845700f4` |
| `EXPERIMENT_PLAN.md` | `8342ceb23436443265ddc171197d093775e71b5b03f60108b639a9689581c858` |
