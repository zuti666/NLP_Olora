# Residual/direction scan code and run inventory

This directory records a read-only audit of existing CopyFlatCV artifacts. No original source, configuration, result, queue, or training checkpoint was modified. No training or GPU job was launched.

Reproduce with `python build_inventory.py` (writes only files beside that script).

- `INVENTORY_SUMMARY.json`: 56 s3 suite files, 468 job records including repeated/smoke/recovery definitions; zero residual/interpolation-named jobs. These are record counts, not 468 unique scientific runs.
- `s3_suites_and_jobs.json`: every enumerated suite path/hash and job record.
- `metadata_candidate_search.json`: 1,209 config/suite/completed/manifest records checked, including arbitrary filenames inside configs directories; zero matches for the exact candidate-family terms.
- `logs_try_files.csv` and `logs_try_search.json`: all 267 files (221 log, 43 yaml, two txt, one out), with paths and hashes. Gitignored files were included via `rg --no-ignore`. The generic YAML interpolation hit describes endpoint weight blending and is excluded from gradient-direction interpolation evidence.
- `formal_vs_diagnostic_ledger.json`: the two principal campaigns' formal job/config/completion-receipt paths and hashes. The ten-task follow-up trains mean-sum and four radii; Task0 trains its ten listed arms. Diagnostic tensor artifacts are saved gradient vectors, not independently trained residual/interpolation models.
- `code_evidence.json`: frozen diagnostic script hash and exact line excerpts for the projection, candidate definitions, effective BA+FC norm, scalar matching, virtual eta, loss evaluation, and restoration.
- `protocol_and_endpoint_checks.json`: all fourteen source report paths/hashes, sampled evaluation counts, eta=.02, positive a ranges, and same-ray/endpoint numerical checks. Ten-task: nine anchors/33 batches/825 matched candidates; Task0: five anchors/20 batches/500 matched candidates. These batches are not independent seeds.
- `REPORT_REVIEW.json`: review of the parent REPORT.md. Its full-training/endpoint/KL distinctions pass; minor terminology suggestions are recorded without editing the report.

The absence conclusion is bounded to the enumerated directories and recognized job/config families. g1/g3/sum/mean have full-training endpoint runs, but these do not constitute training across residual coefficients or normalized interpolation coefficients with every update norm matched. All candidate scans under discussion are virtual single-step interventions, restored before the next candidate. Matching is on actual effective LoRA BA plus FC update L2, not raw A/B gradient norm or post-update prediction KL. Separate SAM controls may match probe KL; that is a different control.
