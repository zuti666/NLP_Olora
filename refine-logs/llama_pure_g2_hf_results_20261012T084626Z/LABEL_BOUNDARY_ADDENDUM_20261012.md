# Addendum: decoder defect and limits of the archived comparison

2026-10-12. The original report, raw predictions, manifest, frozen source and
unrounded scores are preserved. This addendum supersedes any interpretation
of that report as corrected-pipeline evidence for a G2 advantage.

The legacy decoder pipeline merged `Answer:` with five labels, inserted a
duplicate BOS, and misaligned target supervision. Both G2 and historical
IncLoRA share those defects. Current source now uses `decoder_token_ids_v2`;
clean validation needs fresh training of both methods under this version.

Using the same archived normalized-exact-match predictions, the mean
seed0 FAA delta over Orders1/2/3 remains +0.017544pp for1B and+2.002193pp
for3B. Excluding the five gold classes gives -0.859378pp and+0.713285pp,
respectively. This conditional sensitivity analysis is not a repaired
benchmark or proof of the defect's causal effect. Other supervision defects
remain in those predictions, and Orders1/2 share first-stage outputs.

Reproduced label-level evidence and code/validation plan:
[LABEL_BOUNDARY_ANALYSIS.md](../llama_g2_token_boundary_fix_20261012/LABEL_BOUNDARY_ANALYSIS.md)
and [IMPLEMENTATION_AND_PLAN.md](../llama_g2_token_boundary_fix_20261012/IMPLEMENTATION_AND_PLAN.md).
The live v2 collector no longer automatically reuses the historical baseline.
No corrected formal training was launched as part of this repair.
