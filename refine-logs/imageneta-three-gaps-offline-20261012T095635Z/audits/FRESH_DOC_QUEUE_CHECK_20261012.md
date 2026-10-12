# Fresh documented CPU and queue check

UTC: 2026-10-12T10:12:54Z

Status: accepted. Concrete launch blockers: none.

The literal RUN.md CPU command passed 16 tests in 0.119s (exit 0). The literal dry-run passed (exit 0): 12 jobs, GPU4/5/6, 353 pins, zero training updates. Historical source verification covers 190 files.

Formal scope is 36 curvature cells, 24 matching cells, and 30 checkpoint evaluations. Three independent pilots contain 1, 1, and 2 items respectively.

- GPU authorization and concurrency: pass. suite.json allowed_gpus=[4,5,6], max_parallel=3; run_queue.py:22 rejects other authorization; run_queue.py:84 gates active count; run_queue.py:89-102 picks only allowed index with pinned UUID.
- Host PID namespace: pass. suite.json hostname=cudahpc40.cvc.uab.es; pid_namespace=pid:[4026531836]; run_queue.py:41 rejects actual execution outside registered host/PID namespace; CPU dry-run returns before host/GPU operations at run_queue.py:39-40.
- Dual GPU leases and repeated idle query: pass. run_queue.py:93-100 acquires index and UUID flock leases, then reruns GPU query; queue_utils.py:36-49 checks no compute PID, memory<500MiB, utilization<5%; run_queue.py:99 releases both leases on failed requery.
- Worker and CUDA child inherit lease FDs: pass. run_queue.py:113 passes both leases to detached worker; job_worker.py:13 validates both FDs; job_worker.py:15-16 passes both leases to witness and diagnostic child; worker waits at line18.
- Execution acceptance is separate from scientific outcome: pass. queue_utils.py:61-68 checks zero submitted updates, hash receipt, expected fixed count, restoration, accepted_execution/completed_exploratory for curvature; queue_utils.py:69-71 rejects candidate comparisons when curvature gates are not passed; run_queue.py:64-67 logs scientific_outcome separately; scientific inconclusiveness does not mark execution failed; queue_utils.py:73-82 checks non-curvature completion, matching HVP_calls=0, historical evaluation agreement, logits hashes, and exact error partition.
- Independent pilot dependencies: pass. Three pilot jobs have no dependencies; Each of the nine formal jobs depends only on its own kind pilot; run_queue.py:85-87 gates on completed execution and validates dependency evidence.
- Immutable input/code paths and exclusive output paths: pass. Literal documented dry-run validated 353 implementation/input pins and historical 190-file source manifest; run_queue.py:17-21 checks SHA256 plus historical source containment; run_queue.py:27 restricts each output to bulk_output_root/job_id; line28 restricts interpreter and script; run_queue.py:45-46 rejects changed suite on resume; lines64,88 repeat verification; run_queue.py:104-108 refuses reused runtime job directory and nonempty output.
- Resume and surviving workers: pass. run_queue.py:43 enforces single controller flock; run_queue.py:49-52 revalidates completed results, adopts running workers, and refuses incomplete launching state; queue_utils.py:29-35 uses PID plus proc starttime and treats zombies as dead; run_queue.py:59-60 waits for adopted worker identity before reading exit evidence; Surviving diagnostic child retains both leases even if worker/controller dies; failed exit evidence halts new scheduling.
- Current scope counts and state counters: pass. suite formal totals: curvature=36, matching=24, evaluation=30; pilots are 1/1/2 items separately; Runtime counts are derived from per-job statuses with Counter at run_queue.py:79 and :117; Scheduler uses per-job status and active set, not cached counts; no stale hardcoded 32-cell scope.

No input/code changes, GPU query, CUDA kernel, or launch_queue invocation occurred. This acceptance does not assert GPU execution or scientific success; the registered host/PID namespace, live GPU idleness, CUDA witness, and output acceptance remain execution-time checks.

Exact command outputs and reviewed-file hashes are saved in FRESH_DOC_QUEUE_CHECK_20261012.json.
