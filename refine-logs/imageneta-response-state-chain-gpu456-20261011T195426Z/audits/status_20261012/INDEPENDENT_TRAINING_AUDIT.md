# Independent training audit — 2026-10-12

All 6/6 formal branches passed validation; no failed or missing formal training cells. Three paired independent seeds: 0, 1, 2.
Verified 51 frozen execution pins and 190 native source files; 385776 observed float fields are finite.
Total: 2400 actual SGD commits and 4200 native gradient reads. Each seed has 400 paired augmented minibatches, identical initial expanded head, model, RNG and first base gradient.

| Branch | Epoch | Old accuracy (%) | New accuracy (%) | FAA (%) | Old decline (pp) |
|---|---:|---:|---:|---:|---:|
| S | 25 | 81.433 ± 2.280 | 76.115 ± 2.621 | 78.774 ± 0.741 | 3.040 ± 1.672 |
| R | 25 | 81.759 ± 0.862 | 78.215 ± 2.406 | 79.987 ± 0.779 | 2.714 ± 0.820 |
| S | 50 | 80.782 ± 1.629 | 78.346 ± 1.420 | 79.564 ± 0.212 | 3.692 ± 1.233 |
| R | 50 | 81.216 ± 1.047 | 80.184 ± 0.820 | 80.700 ± 0.443 | 3.257 ± 0.977 |

| Epoch | R−S FAA (pp) | FAA wins/ties/losses | R−S old (pp) | Old wins/ties/losses | R−S new (pp) | New wins/ties/losses |
|---|---:|---|---:|---|---:|---|
| 25 | +1.213 ± 0.832 | 3/0/0 | +0.326 ± 1.724 | 1/0/2 | +2.100 ± 0.227 | 3/0/0 |
| 50 | +1.136 ± 0.463 | 3/0/0 | +0.434 ± 0.752 | 1/2/0 | +1.837 ± 1.639 | 2/1/0 |

FAA improves in all three seeds at epochs 25 and 50. Final old accuracy improves only for seed 0 and ties for seeds 1 and 2; the mean gain mainly comes from new-task accuracy.

Queue status: `halted_failure`. Recomputed statuses: `{'completed': 13, 'failed': 1, 'pending': 2}`; stored queue counts are stale and should not be used as a live-process count.
Offline seed2 failed; seeds0/1 remain pending. This does not invalidate the accepted native training results, but the full three-layer mechanism study is incomplete.

The valid table is a two-task experiment with a common 50-epoch Task0 SGD parent, followed by 25 epochs SAM or S_F g2 and 25 shared SGD epochs on Task1. Do not mix with ten-task results or earlier 20-epoch formation experiments.
