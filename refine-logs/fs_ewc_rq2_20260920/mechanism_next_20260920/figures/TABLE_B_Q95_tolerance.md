| Split | epsilon | SGD+EWC | GAM+EWC |
| --- | --- | --- | --- |
| Test | 0.001 | 0 | 0 |
| Test | 0.005 | 1 | 1 |
| Test | 0.01 | 2 | 2 |
| Test | 0.02 | 5 | 5 |
| Test | 0.05 | 10 | 10 |
| Train | 0.001 | 1 | 2 |
| Train | 0.005 | 5 | 5 |
| Train | 0.01 | 5 | 5 |
| Train | 0.02 | 10 | 10 |
| Train | 0.05 | 10 | $\geq 20$ |

Appendix: prefix-safe tolerance radii based on the empirical 95th percentile of positive Task0 CE damage, $G_{95}$, for historical SGD+EWC and visual GAM+EWC anchors. Each positive radius uses the same 16 paired directions; these are direction statistics from training seed 0, not training replicates. For each threshold, the reported radius is the largest sampled radius such that every sampled radius up to it passes the threshold. The sampled grid is $r\in\{0,0.5,1,2,5,10,20\}$. $\geq 20$ denotes right censoring at the grid boundary, not a measured radius above 20. Both deterministic train and test losses use the fixed Task0 20-class head. These are finite-direction, finite-grid measurements, not worst-case neighborhood guarantees.
