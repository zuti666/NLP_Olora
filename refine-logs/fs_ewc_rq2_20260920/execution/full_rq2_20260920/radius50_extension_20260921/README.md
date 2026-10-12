# B1 measured-radius extension to 50

The worker appends measured radii **25, 30, 35, 40, 45, 50**, with a repeated origin for identity checks. The original measurements and figures remain intact. The CPU publisher never starts training, model inference, or a GPU process.

The formal cohort is seeds 0/1/2 × SGD/GAM × dense/sparse_train: **12 extension jobs**. Each dense job has 601 states (100 directions per nonzero radius); each sparse_train job has 97 states (16 directions, the unchanged 256-example Fisher subset). Dense CE and accuracy use all 1,259 old-task training examples. Repeated origins are checked but not duplicated in the merged curves.

## Publisher commands

Use `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python` for rendering.

```bash
python publish_radius50.py --check-only
python publish_radius50.py --current-only
python publish_radius50.py --execute
```

- Default / `--check-only`: standard-library-only, read-only coverage inventory and SHA checks for existing completed job receipts. It does not publish or import Matplotlib.
- `--current-only`: recomputes all existing dense and 16-radius Fisher plotting values from raw states, checks provenance, and renders a separate radius ≤20 four-panel figure using ε = **0.005, 0.01, 0.02, 0.04**.
- `--execute`: requires all 12 complete extension jobs; independently checks raw states, receipt hashes, origin repeat, complete direction coverage, unchanged model/head/data/coordinates/code, exact allowed config changes, and final anchor restoration before producing the radius-50 version. An incomplete cohort exits with code 2 and creates no completion receipt; failed validation exits with code 1.

The full merged grid has **87 dense radii** (81 original plus six new) and **22 Fisher radii** (16 original plus six new). First-three-panel axes extend to 50 in the completed figure. Every new radius has a marker. The r=17 EWCLoRA Fisher spike and full cross-seed sample SD are retained without clipping or smoothing. Curves first average directions within each seed, then report the arithmetic mean and sample SD across the three training seeds.

## Tolerance interpretation

The fourth panel inverts the prefix maximum of each seed's measured damage curve. It uses no interpolation. For every method × ε × seed, `tolerance_bounds_per_seed.csv` lists:

- `largest_safe_measured_radius`: last measured radius before the first threshold failure;
- `first_failed_measured_radius`: first measured failure, blank when the scan remains right-censored;
- `bracket_width`, `right_censored`, and the scan limit.

If **any of a method's three seeds** is right-censored at an ε, that method × ε aggregate has blank mean/SD and is absent from the plot, including its connecting segment. The other method's uncensored point remains visible. Censored bounds remain in the tables.

The radius gap 20→25 is intentional. A safe value at 20 and failed value at 25 gives a measured bracket; it does not locate an exact continuous threshold or establish safety throughout the interval. The caption states this limit. No additional radius jobs are implied.

## Outputs and queue receipts

- Current plot: `figures/current_r20/Figure_B1_neighborhood_current_r20_epsilon4.{png,pdf}`.
- Final plot: `figures/radius50/Figure_B1_neighborhood_radius50_epsilon4.{png,pdf}`.
- Each figure directory includes per-seed CSVs, mean/SD CSVs, tolerance bounds, `CAPTION.md`, `include.tex`, and a validation JSON with source hashes.
- Current-only receipt: `report/current_r20_complete.json`.
- Final receipt: **`report/radius50_complete.json`**, duplicated as **`publisher_receipt.json`** for queue consumers.

The final receipt is written only after numerical/layout checks and PNG/PDF export. Required success fields are `status="complete"`, `expected_jobs=12`, `validated_jobs=12`, and `automatic_numeric_and_layout_checks="passed"`; `source_files` and `output_files` contain SHA-256 hashes. The publisher leaves `rendered_visual_review` pending because human/agent inspection of the actual image is a separate step.

Style reference: `../figures_formal_20260921/B1_merged_latest_20260921`. The publisher validates that prior style's artifact hashes and leaves every file there unchanged.
