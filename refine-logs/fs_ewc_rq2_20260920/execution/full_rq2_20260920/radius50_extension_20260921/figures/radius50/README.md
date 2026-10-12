# B1 radius50: four thresholds

[PNG](Figure_B1_neighborhood_radius50_epsilon4.png) · [PDF](Figure_B1_neighborhood_radius50_epsilon4.pdf) · [Caption](CAPTION.md)

Times New Roman and the prior one-row four-panel style are retained. All inputs are read-only. Every plotted aggregate uses exactly seeds 0/1/2 and sample SD.

See `tolerance_bounds_per_seed.csv` for per-seed last-safe/first-failed measured radii. A missing first-failed radius means right censoring; the corresponding method×epsilon mean and SD are blank in `fourth_panel_mean_sd.csv`, and no point is drawn. No interpolation estimates the threshold within radius-grid gaps.

Reproduce: `python ../../publish_radius50.py --execute`.
