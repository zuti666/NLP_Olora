# Research Findings — current visual GAM+EWC RQ2

2026-09-20 user-selected target: current visual GAM+EWC, not the September17 paper algorithm.

Independent result-to-claim verdict: **partial**, confidence high within this one-seed scope; review_independence=same-family, acceptance_status=provisional, integrity_status=unavailable. Complete reviewer reasoning is in CLAIMS_FROM_RESULTS.md/json; full call trace is in TRACE_PATH.txt.

Supported: lower train256 raw empirical Fisher in sampled effective-weight neighborhoods; lower old-task test damage despite larger realized displacement in the historical full GAM chain; lower t9 straight-segment directional sensitivity.

Not supported: general test-neighborhood damage or tolerance improvement, pure geometry causal attribution from A, lower trace guaranteeing better a21/a22, optimizer-step path measurement, cross-seed generalization, or the complete shaping-to-tolerance-to-retention causal chain.

The train/test sign difference and endpoint sensitivity association are observations. Hypotheses such as direction alignment, fit/confidence differences, and sample-specific sensitivity remain untested explanations, not established causes.

Routing: retain completed A; GPU0/1 proposed test rawF plus explicitly labelled GAM-generated direction proxy; GPU2/3 proposed t1/t5 endpoint path integrals. No first-batch training. Dense loss-grid and seed replication are conditional on retained claims. Do not repeat broad Fisher-estimator comparisons or choose radii/epsilon after inspecting favorable outcomes. If test damage/tolerance remains unfavorable, narrow RQ2 to direction-dependent behavior under realized drift and retain the random-neighborhood counterevidence.

Current state: CPU organization and plots completed; figure quality review passed (same-family/provisional); additional GPU diagnostics are proposed, not queued. Refer to EXPERIMENT_TRACKER.md for final status.
