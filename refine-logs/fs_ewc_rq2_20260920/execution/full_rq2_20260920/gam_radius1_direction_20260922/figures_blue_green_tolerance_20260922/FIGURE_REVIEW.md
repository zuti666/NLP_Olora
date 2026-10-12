# Final figure review: PASS

Reviewed the final rendered PNGs, their validation receipts, the plotting code,
README/captions, all four exported CSVs, and the referenced
`plot_b1.py::measured_tolerances` implementation. This was a CPU-only,
read-only audit of the experiment artifacts. No model evaluation or GPU work
was run.

## Rendered figures

- `task0_positive/Figure_task0_positive_blue_green_tolerance.png`
  (SHA-256 `b95298f2989951bfc37697ec6e5021969b377adaa3f7deffc50079ed0f4f070d`)
- `task1_negative/Figure_task1_negative_blue_green_tolerance.png`
  (SHA-256 `1e949c108e05624cd1bc92af75239201fea287ba188e9e4b9c0d83cc91aba971`)

Both final PNGs were opened and inspected at their rendered output. Each is a
single 1x4 row. Only blue EWCLoRA and green FS-EWCLoRA appear. The legends show
method names without parameter values, stay inside the axes, and do not cover
the plotted data. Labels, rotated epsilon ticks, SD bands, individual-seed
Fisher traces, and all panel edges are visible without clipping. The final
7.6 x 1.85 inch layout uses Times New Roman with 8.2 pt axis labels and 7.2 pt
ticks/legends; the vector PDFs embed Times New Roman. The Task0 Fisher axis uses
the final range `[0, 46027.64898113464]`, leaving clear space between the seed
peak and the upper-right legend.

## Data and summaries

- The selected table has exactly 132 rows: 2 methods x 2 directions x 3
  training seeds x 11 measured radii. It matches the same rows in
  `figures/all_per_seed.csv`; no third method enters any summary.
- For every selected row, signed `D_train` is nonnegative and equals
  `G_positive_train`. The first three panels therefore use one consistent data
  cohort and measured-radius grid.
- All 176 saved mean/sample-SD rows reproduce from the 132 per-seed rows with
  maximum floating-point difference `3.64e-12`. Panels 1-2 show mean plus/minus
  sample SD. Panel 3 shows the arithmetic mean and faint traces for all three
  seeds on a linear raw-Fisher axis. Its limits include every selected seed
  value.
- Panel 4 is mean-only, matching the reference presentation. Its per-seed
  radii and sample SD remain available in `tolerance_per_seed.csv` and
  `tolerance_mean_sample_sd.csv`.

## Tolerance inversion

- The plotting code directly imports and calls the reference
  `radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v2/plot_b1.py::measured_tolerances`.
- The seven displayed thresholds are exactly 0.25, 0.5, 1, 1.5, 2, 2.5, and 3.
  An independent prefix-maximum implementation reproduced all saved per-seed
  tolerance radii exactly.
- All 84 displayed seed-method-direction-threshold results have a last safe
  radius and an immediately following failed radius separated by 0.5. The safe
  prefix is at most epsilon and the next measured value is greater than
  epsilon in every case. No interpolation is used.
- The 72 rows for the six reference thresholds 0.005-0.05 all have tolerance
  radius 0 and first failure at radius 0.5; they are retained separately in
  the CSV.
- All 52 tolerance mean/sample-SD rows reproduce after per-seed inversion.
  There is no right censoring in the current 84 displayed results. The code and
  captions state the conservative future rule: leave a mean empty if any seed
  is right-censored rather than average only uncensored seeds.

## Provenance and disclosure

Both final validation receipts match every recorded source and output hash.
They record one fixed direction per seed, the measured radius grid, linear
unscaled Fisher, per-seed-first inversion, no interpolation, and no new model
evaluations. The captions accurately distinguish symmetric descriptive SD
bands from physical bounds, disclose the single fixed direction and
single-seed Fisher traces, and avoid claiming a continuous-radius or
all-directions guarantee.

No blocking issue remains.
