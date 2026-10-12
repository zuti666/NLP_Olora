# Independent implementation and configuration review

Status: **passed for implementation/configuration admission**. GPU smoke and formal training gates remain required; this review does not claim new performance results.

Three variants implement descending inner probe, `a=g0-g1`, positive outer relocation **from BASE**, and final `d=g2` only. FixedInner fixes inner global trainable L2 distance .05 and leaves outer `eta*||g0||`; FixedOuter reverses those choices; FixedBoth fixes both .05. Frozen original `response_step.py` and task layout are byte-identical. The independent code inherits ordinary FP32 torch SGD commit without final normalization, clipping, projection, history, Fisher or weight fusion.

An independently authored multi-tensor quadratic oracle checks exact probe positions against `H@theta+b` at eta .04/.004, state/cache/buffer/gradient restoration, identical R0 probe draws and exactly one R1 advance, g2-only final submission, ordinary FP32 SGD, explicit zero-gradient/zero-response/inner and outer FP32-unresolvable fallbacks, exception rollback, and cross-geometry checkpoint rejection. **FixedBoth at the same state has bitwise-identical probes and U for eta .04, .004 and 0; only actual SGD displacement changes with eta.** These checks do not use the author's unit-test results as their oracle.

The driver review independently compares complete resolved dictionaries for eight original method/layout combinations against the old driver: all equal. New registered variants preserve model/head/task CE/batch/epochs/SGD/epoch cosine/purity settings and reject unknown radii, history use, and silently ignored radius fields. All seven new configuration files differ from the old Current-G2 only by radius geometry and run metadata. Four bounded 2-task/1-epoch/2-batch smokes gate three full CL10 2550-step trainings. Suite authorization is exactly physical GPU 4 and 5.

Old Current-G2 result reuse independently checks 34 frozen run pins and all 190 source files, completed receipt/summary/trace hashes, and recomputes the native task matrix: AAA 62.87033373015873, FAA 45.252, BWT -9.652222222222223. This is reused baseline evidence, not a new training.

One early record-key interface issue (`radius_mode` versus `radius_protocol`) was identified, corrected, and retested before this passed review. No open implementation blockers remain. A literal fresh-document invocation check is a separate gate and is not implied by this review.

Evidence: `INDEPENDENT_ORACLE_CPU.py/json`, `INDEPENDENT_DRIVER_CHECK.py/json`, `INDEPENDENT_ORIGINAL_REUSE.json`, and hash/config details in `INDEPENDENT_IMPLEMENTATION_REVIEW.json`.

Execution gate code review additionally verifies: non-fallback actual probe distances use relative tolerance 1e-4/absolute 1e-8 and match displacement aliases; checkpoints require the independent format and registered geometry; new CPU admission references these new implementation gates, not old smoke acceptance; four real smokes compare Original against the reused formal first two updates; old baseline 107 receipt/result/checkpoint/pin files are frozen and verified; GPU leases use both physical index and UUID with a second idle recheck; final result tables exclude bounded smokes and require validated new formal receipts.
