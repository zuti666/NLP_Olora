# Task0 formation: independent 2 x 20 epoch results

Updated UTC: 2026-10-10T23:46:54.921031+00:00
Queue: completed; {'pending': 0, 'running': 0, 'completed': 25, 'failed': 0}
Accepted formal runs: 18/18. Six bounded smoke runs and their pairing gate are excluded.

| Condition | Seed | Task0 end | Old start (40) | Old final (40) | New final (40) | Old drop | FAA |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGD throughout | 2 | 83.062 | 82.736 | 76.547 | 72.047 | 6.189 | 74.297 |
| SAM throughout | 2 | 83.388 | 83.062 | 77.850 | 72.441 | 5.212 | 75.146 |
| Residual GAM throughout | 2 | 83.388 | 82.736 | 77.850 | 72.047 | 4.886 | 74.949 |
| S_F g2 throughout | 2 | 84.691 | 84.039 | 79.479 | 72.441 | 4.560 | 75.960 |
| S_F g2 early, SGD late | 2 | 84.039 | 83.388 | 79.153 | 71.654 | 4.235 | 75.403 |
| SGD early, S_F g2 late | 2 | 83.062 | 82.736 | 77.199 | 72.047 | 5.537 | 74.623 |
| SGD throughout | 0 | 80.782 | 80.456 | 79.153 | 72.441 | 1.303 | 75.797 |
| SAM throughout | 0 | 81.433 | 81.107 | 78.827 | 72.441 | 2.280 | 75.634 |
| Residual GAM throughout | 0 | 81.433 | 80.456 | 78.827 | 70.866 | 1.629 | 74.847 |
| S_F g2 throughout | 0 | 82.085 | 81.759 | 80.782 | 71.260 | 0.977 | 76.021 |
| S_F g2 early, SGD late | 0 | 82.410 | 81.759 | 80.782 | 72.047 | 0.977 | 76.415 |
| SGD early, S_F g2 late | 0 | 82.085 | 81.759 | 79.153 | 72.441 | 2.606 | 75.797 |
| SGD throughout | 1 | 84.039 | 83.713 | 79.153 | 72.835 | 4.560 | 75.994 |
| SAM throughout | 1 | 84.039 | 83.388 | 79.479 | 72.441 | 3.909 | 75.960 |
| Residual GAM throughout | 1 | 83.388 | 83.062 | 79.805 | 70.866 | 3.257 | 75.335 |
| S_F g2 throughout | 1 | 84.691 | 84.039 | 81.433 | 72.047 | 2.606 | 76.740 |
| S_F g2 early, SGD late | 1 | 83.713 | 83.062 | 81.107 | 72.047 | 1.954 | 76.577 |
| SGD early, S_F g2 late | 1 | 84.039 | 83.388 | 80.782 | 72.441 | 2.606 | 76.611 |

| Condition | Accepted seeds | FAA mean +/- sample SD | Old drop mean +/- sample SD |
|---|---|---:|---:|
| SGD throughout | [0, 1, 2] | 75.363 +/- 0.928 | 4.017 +/- 2.488 |
| SAM throughout | [0, 1, 2] | 75.580 +/- 0.410 | 3.800 +/- 1.469 |
| Residual GAM throughout | [0, 1, 2] | 75.044 +/- 0.258 | 3.257 +/- 1.629 |
| S_F g2 throughout | [0, 1, 2] | 76.240 +/- 0.434 | 2.714 +/- 1.794 |
| S_F g2 early, SGD late | [0, 1, 2] | 76.132 +/- 0.636 | 2.389 +/- 1.672 |
| SGD early, S_F g2 late | [0, 1, 2] | 75.677 +/- 1.000 | 3.583 +/- 1.693 |

All Task1 updates use natural SGD. Method switching uses a continuous Task0 cosine schedule and model at epoch11. Old drop starts after head expansion, before Task1 training. The same official old/new tests use the full 40-class classifier.

Reported ratios are computed from exact test sample counts. Native rounded-per-task FAA/AAA/BWT remain in each run summary for provenance. Historical50-epoch results are never pooled. Partial seed cohorts are descriptive; formal paired conclusions require all three identical seeds. Formation/phase effects do not independently establish a trajectory-only causal mechanism.
