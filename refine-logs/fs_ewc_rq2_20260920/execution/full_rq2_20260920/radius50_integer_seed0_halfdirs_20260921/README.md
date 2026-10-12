# Seed-0 integer radius refinement (50 dense / 16 Fisher directions)

This is a new exploratory extension. Existing three-seed artifacts remain unchanged.

- Seed: **0 only**; both EWCLoRA (`sgd`) and FS-EWCLoRA (`gam`).
- New radii: **26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49**.
- Reuse measured radii 25, 30, 35, 40, 45, and 50. Each job also repeats radius 0 for validation.
- Loss / accuracy: all 1,259 Task-0 training examples and the same **first 50 public directions** at every nonzero radius. All earlier seed-0 dense points are recomputed from their persisted first 50 individual directions; their former 100-direction means are not reused.
- Fisher: the same fixed **256 training examples and 16 directions**, unchanged from the existing Fisher protocol. This is raw per-example empirical Fisher trace: sample averaging only, no trace/parameter-count normalization or EWC multiplier.
- Final coverage per method: **107 dense radii** (0–20 in steps of 0.25, then every integer 25–50), and **42 Fisher radii** (the existing 16 radii through 20, then every integer 25–50).
- No new 21–24 measurements. The 20–25 gap therefore remains unresolved.

There are four GPU jobs: two methods × dense/Fisher. A dense job has 1,001 states (origin + 20 × 50); a Fisher job has 321 states (origin + 20 × 16). These jobs diagnose fixed anchors and do not retrain models. See [execution settings](RUN.md) and [live state](campaign/STATUS.md).

## Publication

`publish_radius50.py` is CPU-only. Default / `--check-only` performs a read-only inventory. `--execute` fails without all four validated results. It validates the prior completed coarse experiment and all new receipt hashes, origins, source identities, shared direction identities, and sample counts before combining measurements.

The final four-panel figure keeps Times New Roman, shared EWCLoRA/FS-EWCLoRA styles, and the epsilon ticks **0.005, 0.01, 0.02, 0.04**. Curves use only seed 0. No across-seed standard deviation, error bar, or confidence band is displayed. All new integer measurements have markers. The first three panels use axis padding beyond 0 and 50 to avoid clipped boundary markers.

The fourth panel uses the prefix-safe inverse of the first-50-direction training damage curve, computed separately for each method. Its value is the largest measured radius whose entire measured prefix satisfies the threshold. No radius interpolation is performed. If a method remains below the threshold at the scan limit, that censored point is omitted and documented in CSV. A passing radius 20 followed by failing radius 25 does not identify the boundary inside that gap.

Only after all validation and rendering succeeds will `report/radius50_complete.json` be written with `expected_jobs=4`, `validated_jobs=4`, and `automatic_numeric_and_layout_checks="passed"`. It also records all 12 validated reused coarse jobs, source hashes, output hashes, and the separate status of actual rendered-image review. New output is saved under `figures/radius50_integer_seed0_dense50_fisher16/`.

## Checks before launch

- Read-only inventory returns pending with zero completed new jobs; incomplete `--execute` exits 2 and writes no figures or completion receipt.
- A CPU replay validates all 12 existing coarse jobs and rebuilds 174 old dense and 44 old Fisher plot rows for seed 0 only.
- Independently checked training tolerance radii before the new integer measurements:

| Epsilon | EWCLoRA | FS-EWCLoRA |
| --- | ---: | ---: |
| 0.005 | 8.25 | 9.25 |
| 0.01 | 11.00 | 12.25 |
| 0.02 | 14.50 | 16.25 |
| 0.04 | 19.00 | 20.00 |

The last FS-EWCLoRA row is bounded by its measured failure at 25. These numbers use one seed and 50 dense directions; they are deliberately distinct from earlier three-seed/100-direction statistics.
