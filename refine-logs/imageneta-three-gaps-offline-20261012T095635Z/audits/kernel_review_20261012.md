# Independent offline kernel review — 2026-10-12

No concrete blocker found for the fixed read-only pilots. All 16 CPU tests passed independently with `Pilot_new_local`, `PYTHONDONTWRITEBYTECODE=1`, and CUDA hidden. No source changes, GPU calls, or network calls were made.

Two distinctions are necessary when accepting the pilot:

- `curvature_diagnostics.py:379,389` records execution success independently from numerical acceptance. An exit code of zero or `accepted_execution` may accompany zero converged cells. Use the per-cell gates and `candidate_comparisons_valid_cells`. Likewise `effective_match.py:91,95` may complete successfully with unavailable matches; inspect `accepted_matches` and each match status.
- `curvature_diagnostics.py:61,106,166,230` retains the original native FP32 gradient and represented BASE when testing the FP64 explicit Hessian. FP64 convergence cannot accept the FP32 Hessian, overwrite the original three-distance failure, or prove higher-derivative identity across precision. Its separate gates and q/p comparisons support an exploratory precision comparison.

The imported implementation consistently uses detached HVP vectors, restores direction magnitude after normalized finite differences, restores complete entry state, uses saved next-update LR, and matches the finite effective norm of all unscaled BA plus full FC weight/bias. Conditional20/joint40 error and CE competition partitions are correct. Evaluation checkpoints remain fixed at e0/e1/e10/e25/e50.

Independent raw-source checks confirmed common e0 parameter identity and LR for all three S/R seed pairs; all analysis input hashes; the reported layer-median stage summaries; all-step LoRA/FC/path magnitudes; and fit-matched test/epoch/LR summaries. No important interpretation correction was found in `results/state_analysis/RESULT_ANALYSIS.md:18,20,31,42,46,53,57-60`.

CPU success establishes algebra and orchestration. Actual ViT numerical acceptance and FP64 GPU feasibility remain pilot outcomes. Exact source hashes and validation scope are recorded in the JSON review.
