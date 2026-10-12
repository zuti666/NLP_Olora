# Independent launch review

Reviewed UTC: 2026-09-22 16:06. Reviewer: independent `launch_reproduction_check` agent.

**Disposition: PASS for launching the frozen queue. No unresolved launch-blocking issue found.** This review did not launch training, instantiate a model, or initialize CUDA. GPU smoke/full execution remains the queue's next validation stage.

## Reproduction actually executed

Read README.md first, then ran its exact documented read-only command with the specified existing environment:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/seed012_execution_20260922/launch_queue.py --dry-run
```

Exit code 0; repeated after the collector correction and final runtime freeze:

```json
{"dry_run": "PASS", "jobs": 108, "formal": 99, "gpus": [2, 3, 4, 5]}
```

The read-only branch returns before querying or reserving GPUs, acquiring the queue lock, creating state, or spawning jobs. No `build_suite.py` invocation and no reviewer edits to runtime/source/configuration files were made.

## Findings and resolution

One concrete integration defect was found and reported before launch: the original six `suite.reused` entries used `status="accepted_reused"`, but the collector required `status="completed"`. They would all have been excluded. The main agent corrected the suite to the canonical completed status while preserving `reuse_validation_status`, strengthened archived-receipt identity/hash checks, retained the known historical `GAM-EWC` display-name alias, and regenerated runtime hashes.

The corrected collector was then executed independently against a copy of the actual suite in an isolated `/tmp` directory. It returned 0 with `accepted_runs=6`, `new_formal=0`, `reused=6`, `excluded=0`, `not_completed=99`, and 35 groups. No campaign results were written by that isolated check.

## Independently checked

- **Coverage:** 99 formal jobs + 9 diagnostic smoke jobs + 6 verified reused formal seed0 records. Formal plus reuse is exactly the Cartesian product of 7 methods, 5 datasets, and seeds 0/1/2: 105 unique identities and 35 groups, without gaps or duplicate identities. The seven Flowers smokes and two additional EWC/GAM Het5 smokes are present.
- **Configuration identities:** all 108 new jobs match their configuration's dataset and singleton seed; every config uses rank16, batch128 and logical device0. Formal task counts are10 for the four single datasets and5 for Het5, without a task cap. Smokes use two tasks and one epoch per task. Within all35 formal groups, configs differ only in seed, prefix, output root/tag or explanatory protocol note.
- **Dependencies:** all job IDs are unique, dependencies resolve without cycles, every formal job has the matching method smoke dependency, and new seed1/2 jobs depend on successful full seed0 in groups where seed0 is newly executed. Reused groups use verified existing seed0. No accuracy-threshold gate is present. Failed dependencies are held while independent groups can progress.
- **Hashes:** verified all108 configuration, source-manifest and data-manifest SHA256 bindings; checked every declared source file in all three trees (baseline160, EWC172, CL/GAM172); checked all six reused receipt/config/metric/log hashes. The frozen runtime manifest matched after the correction. The actual cached safetensors SHA256 and the HF `refs/main` file both match the preflight records.
- **Actual entrypoint:** in three isolated CPU-only subprocesses, imported each frozen `src.main`, replaced both training dispatch targets with capture/refusal functions, and called the exact runner argv `main(['--config', config, '--override', 'device=[0]'])` for all108 jobs. Baseline64, EWC17 and CL/GAM27 configs all parsed and dispatched correctly to incremental mode with matching seed/dataset/data root/logical device. `torch.cuda.is_initialized()` remained false in every subprocess. This verifies the actual main function without executing a trainer.
- **Data and pretrained weights:** configs resolve `/data/115-2/users/liying/datasets`, matching the suite environment. Pinned CPU-preflight evidence covers all five datasets, including Pets as Het5's fifth component; sample evidence contains real decoded train/test items. Offline cache settings are supplied to jobs. CPU weight-resolution evidence binds the default timm model to the same locally cached `model.safetensors` used explicitly by CL-LoRA. This review rehashed the cached weight, but did not repeat model instantiation or exhaustive image decoding.
- **Output identity and reader compatibility:** all108 `(outputs_root,prefix)` pairs are unique and resolve beneath this campaign. At the prelaunch check, no job receipt, job log or matching metrics file already existed; the required logs/receipts directories did exist. Trainer naming is `<prefix>_<backbone>_cl_metrics.json` with the matching `<prefix>_<backbone>.log`, consistent with runner discovery and log lookup. Trainer `cnn.matrices.final` is exactly the field consumed by the runner. Square upper-triangle NaN placeholders do not invalidate finite observed lower-triangle entries.
- **GPU scope and isolation:** suite and environment contract restrict physical GPUs to2/3/4/5 and concurrency to4. The scheduler queries current processes/memory/utilization, checks again under a per-UUID advisory lock, then passes one permitted GPU UUID through `CUDA_VISIBLE_DEVICES`. Each child requires exactly one visible GPU and uses logical device0. The existing environment witness records four successful CUDA checks on those GPU indices; this reviewer ran no additional CUDA witness.
- **Queue/runner integrity:** queue lock prevents duplicate active controllers; resume refuses recorded live children. Job logs use exclusive creation and the runner refuses existing receipts. Each child verifies source, config, data-manifest, checkpoint and frozen-runtime hashes before training. SGD gradients/parameters and observed task metrics must be finite, and metric/log SHA256 values are written to completed receipts. No retry, batch-size adjustment, preemption or score-based configuration selection was found.
- **Collector boundary:** only suite jobs with `stage='formal'` are considered, and only completed matching receipts enter aggregation. The nine smokes are skipped even if they later complete. Reused results are separately validated against their archived completed receipts/configs/metric bytes. FAA and AAA_stage are recomputed from observed CNN task matrices; sample SD uses `statistics.stdev`, hence ddof1; unavailable SD remains null for a single seed. Isolated collection accepted exactly the six intended formal reused records and no smoke results.

## Scope limits

This is a launch/dry-run review, not evidence that pending GPU smokes or full training will finish. The documented historical limitations remain: ordered paths/labels do not prove historical image-byte identity, and reused GAM-Het5 lacks a checkpoint-byte hash recorded at its historical launch. The current cache is verified. The campaign preserves archived method/dataset training budgets rather than claiming equal compute across methods.

## Reviewed frozen identity

```text
suite.json             71faec77b2d43a2224869c954cd6278ab63b949f86fc15291b9b6ebf31968baf
runtime_frozen.json    7b8b839d739449df1b05743bb98b48bfb0633038afb9f32366189da8edf140ac
run_job.py             f556b90915c674dc9640b0f283f9d8f15bb6791b8246cf7098c04136d8e269a4
launch_queue.py        d840d62b3cd8b05ede5f766486b273ad99cbbc37b2e2b9b1836625b18c727a66
collect.py             54f987aa0983fbfa4fbe7378aa6be2797341e619c098b341d3d402c41da706fb
```
