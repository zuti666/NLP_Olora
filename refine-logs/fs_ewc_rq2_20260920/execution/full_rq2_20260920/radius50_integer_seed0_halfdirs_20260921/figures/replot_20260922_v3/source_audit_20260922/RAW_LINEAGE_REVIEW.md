# B1 v3 raw-source lineage audit

Status: **passed**. This independent CPU audit parsed every state in the 14 seed0 files feeding B1 and all supplemental origins. It did not import plotting or publisher functions, run models, modify inputs, or redraw figures.

## Counts and source partitions

| Source partition | Job | Raw records read (both methods) | Selected measurement records | Selected curve points | Selection |
|---|---|---:|---:|---:|---|
| formal_results | dense | 16,002 | 8,002 | 162 | r=0 plus indices 0–49 |
| formal_results | sparse_train | 194 | 194 | 14 | r=0 plus indices 0–15 |
| stage_path_completion_20260921/results | sparse_train | 290 | 288 | 18 | exclude repeated r=0; indices 0–15 |
| radius50_extension_20260921/results | dense | 1,202 | 600 | 12 | exclude repeated r=0; indices 0–49 |
| radius50_extension_20260921/results | sparse_train | 194 | 192 | 12 | exclude repeated r=0; indices 0–15 |
| radius50_integer_seed0_halfdirs_20260921/results | dense | 2,002 | 2,000 | 40 | exclude repeated r=0; indices 0–49 |
| radius50_integer_seed0_halfdirs_20260921/results | sparse_train | 642 | 640 | 40 | exclude repeated r=0; indices 0–15 |
| **Total** | | **20,526** | **11,916** | **298** | |

The 8,600 excluded dense measurements are direction indices 50–99 from the formal and coarse files. Another 10 records are supplemental repeat origins. The 11,916 selected records comprise 10,602 dense measurements and 1,314 Fisher measurements. Fisher locations all overlap the dense radius/direction grid, but are distinct job measurements. There are therefore 10,602 distinct selected geometric locations, not 11,916 independently located perturbations.

Dense contributes 107 radii per method: 0–20 in steps of 0.25, then 25–50 in steps of 1. Fisher contributes 42 radii per method: {0, 0.5, 1, 2, 5, 10}, 11–20, and 25–50. The stage partition supplies Fisher radii 11–19; formal data supply radius 20. The coarse partition supplies {25, 30, 35, 40, 45, 50}; integer refinement supplies the other 20 integers from 26 to 49. Radii 21–24 remain unmeasured.

## Independent checks

- All 20,526 raw records, their manifests, and summaries are finite; state IDs are unique within each job and every required radius/direction pair exists.
- All 308 source radius-summary entries were recomputed, including train/test signed damage arrays, positive-part mean damage, loss/accuracy, Q95, maxima, and all per-direction Fisher values. Stored source tolerances were also verified.
- All 214 dense CSV rows, 84 Fisher CSV rows, 512 first-three-panel plotted values, 22 v3 tolerance rows, and the eight prior tolerance-input rows agree with raw recomputation. Both published-source and v3 CSVs were compared.
- Maximum absolute CSV discrepancies: damage 9.92e-17, accuracy 2.85e-14 percentage points, Fisher 3.64e-12. Every tolerance radius and first-failure radius agrees exactly.
- The 100 persisted public direction objects match across methods, partitions, radii, and jobs, including hash, index, seed, and norm. Selected dense uses indices 0–49; Fisher uses indices 0–15.
- All supplemental origin train/test losses, accuracies, Fisher traces, and subset losses are exactly equal to the corresponding formal origins. They define each supplement’s damage baseline and validate continuity, but are excluded as extra plotted points.
- Coordinate size is 14,155,776; every train metric uses 1,259 examples; every Fisher estimate uses the fixed 256-example subset. Raw Fisher is nonnegative and has no normalization, floor, or regularizer.
- Canonical protocol manifests, coordinate-order hashes, configurations, data-manifest hashes, raw checkpoint identities, checkpoint files, source-code files, worker wrappers and receipt file hashes pass. Supplements change only their output directory, radius grid, and the explicitly authorized integer dense direction count.
- Restoration is recorded exact at every raw point, and final restoration checks report identical predictions, zero logit difference, and the anchor effective-weight hash.

## Plot-input interpretation

Panel 1 is the mean of max(loss_at_direction − partition_origin_loss, 0), not the signed mean loss difference. Panel 2 averages training accuracy over the same 50 dense directions. Panel 3 averages unnormalized empirical Fisher trace over 16 directions, with per-example gradients evaluated on 256 fixed examples. All plots use training seed 0 only.

Panel 4 uses the exact panel-1 dense curve, takes its cumulative maximum on the measured grid, and reports the preceding measured radius before the first value strictly above epsilon. It does not interpolate a continuous crossing. All 22 displayed thresholds have observed failures and are uncensored.

| epsilon | EWCLoRA / SGD safe radius | first failure | FS-EWCLoRA / GAM safe radius | first failure |
|---:|---:|---:|---:|---:|
| 0.005 | 8.25 | 8.5 | 9.25 | 9.5 |
| 0.01 | 11 | 11.25 | 12.25 | 12.5 |
| 0.02 | 14.5 | 14.75 | 16.25 | 16.5 |
| 0.03 | 17 | 17.25 | 19.25 | 19.5 |
| 0.04 | 19 | 19.25 | 20 | 25 |
| 0.05 | 20 | 25 | 20 | 25 |
| 0.06 | 20 | 25 | 25 | 26 |
| 0.07 | 20 | 25 | 26 | 27 |
| 0.08 | 25 | 26 | 27 | 28 |
| 0.09 | 26 | 27 | 28 | 29 |
| 0.1 | 27 | 28 | 29 | 30 |

The historical publisher metadata reports 53,646 original plus 2,644 new recomputed records because its validation chain traverses three training seeds before selecting seed 0. Those counts represent 56,290 rows across 34 distinct raw files, including excluded dense directions and repeated origins. They must not be described as the selected measurements used in v3. The current v3 script hashes the recorded raw-source files, reads the CSVs, and recomputes tolerance; this separate audit reparsed every relevant seed0 JSONL state and independently rebuilt the values.

## Result and limits

No data mismatch, duplicate selected curve point, nonfinite value, unexplained source identity change, missing direction, or fabricated radius was found. The retained 20-to-25 radius gap and single-training-seed scope remain limitations of the measured result. Hash and manifest agreement establishes persisted-file lineage; this CPU audit does not rerun GPU measurements or reconstruct direction tensors from their random seed.

Full source paths, per-radius lineage, origins, direction objects, check counts, differences, and fingerprints are in [raw_lineage_agent.json](raw_lineage_agent.json).
