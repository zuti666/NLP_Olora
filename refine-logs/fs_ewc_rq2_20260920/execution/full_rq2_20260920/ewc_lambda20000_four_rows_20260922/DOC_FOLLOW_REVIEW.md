# Document-follow review: passed

Reviewer: `/root/lambda20000_fisher`. Mode: **existing-agent fallback** because the parent reported a thread-limit failure for a fresh reviewer. This reviewer did not author the run document or DAG, but did author `prepare_fisher.py`; the review is independent of the document authorship, not a fresh audit of the preparation implementation.

The dry-run command was copied verbatim from `RUN.md` and returned exit code 0 with `dry_run=true`, 24 stages, 24 jobs, and maximum concurrency 6. No `--execute`, GPU witness, preparation, training, or report command was run for this review.

| Check | Result |
|---|---|
| Stage composition | 6 preparations (1 CPU reuse + 5 native GPU Fisher passes), 4 seed0 smokes, 1 CPU gate, 12 formal continuations, 1 CPU report |
| Allowed resources | All GPUs 0–6; preferred GPUs for the four arms are 0/1/2/5; every GPU job permits fallback across all seven cards |
| Admission | No all-GPU barrier; `wait_for_all_gpus()` returns immediately when `required=false`; preparation checks only the selected GPU and its inherited UUID lock |
| Polling | 900 seconds; completed jobs can release dependencies immediately |
| Dependencies | All raw-Fisher preparations precede smokes; all four smokes precede CPU acceptance; every formal trial depends on acceptance; each arm runs seed0→1→2; report waits for all twelve |
| Actual CLI definitions | All job/preflight flags and required flags match the source parser definitions; twelve trial parent/method/seed/raw-Fisher paths and lambda=20000 match the specification |

No blocking discrepancy was found. One nonblocking document error was corrected by the parent: the state path is `campaign/pipeline_state.json`.

This dry-run establishes executable structure and available file fingerprints. Real GPU smoke acceptance and final scientific results remain future evidence. The exact command and reviewed file hashes are recorded in `DOC_FOLLOW_REVIEW.json`.

Final refresh: 2026-09-22T09:25:22.868551+00:00. After the report compatibility fix was frozen, the exact documented dry-run passed again with 24 stages / 24 jobs / maximum concurrency 6. Final `report.py` SHA256: `311adc316eecaf1e2fca4c3e57769fa99296e45a4732b600257bc6fb2ad0cb05`. Full dry-run output is saved in `cpu_validation/DOC_FOLLOW_FINAL_DRY_RUN.json`; prior findings remain preserved. No GPU job was launched by this review.
