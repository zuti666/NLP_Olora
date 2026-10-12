# B1 radius50_integer_seed0_dense50_fisher16: four thresholds

[PNG](Figure_B1_neighborhood_radius50_integer_seed0_dense50_fisher16_epsilon4.png) · [PDF](Figure_B1_neighborhood_radius50_integer_seed0_dense50_fisher16_epsilon4.pdf) · [Caption](CAPTION.md)

Times New Roman and the prior one-row four-panel style are retained. All inputs are read-only. Only seed 0 is plotted, with 50 dense directions and 16 Fisher directions at all radii. No across-seed uncertainty is shown.

See `tolerance_bounds_per_seed.csv` for per-seed last-safe/first-failed measured radii. A missing first-failed radius means right censoring; the corresponding method×epsilon mean is blank in `fourth_panel_single_seed.csv`, and no point is drawn. No interpolation estimates the threshold within radius-grid gaps.

Reproduce: `python ../../publish_radius50.py --execute`.
