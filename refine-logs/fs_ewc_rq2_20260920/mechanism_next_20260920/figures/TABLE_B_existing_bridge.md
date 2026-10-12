| Anchor | Direction | Radius | D train | D test | Raw trace train256 |
| --- | --- | --- | --- | --- | --- |
| SGD | SGD | 0 | +0.000000 | +0.000000 | 1465.497 |
| SGD | SGD | 0.25 | -0.000398 | -0.000103 | 1554.181 |
| SGD | SGD | 0.5 | +0.010850 | +0.016222 | 1799.319 |
| SGD | SGD | 1 | +0.085561 | +0.147516 | 3209.145 |
| SGD | SGD | 1.5 | +0.332180 | +0.517852 | 7572.045 |
| SGD | SGD | 2 | +0.970554 | +1.276820 | 15993.630 |
| SGD | GAM | 0 | +0.000000 | +0.000000 | 1465.497 |
| SGD | GAM | 0.25 | +0.002023 | -0.001218 | 1595.781 |
| SGD | GAM | 0.5 | +0.006002 | +0.001458 | 1600.829 |
| SGD | GAM | 1 | +0.020459 | +0.018917 | 1779.211 |
| SGD | GAM | 1.5 | +0.044087 | +0.051845 | 1989.137 |
| SGD | GAM | 2 | +0.080304 | +0.100854 | 2162.101 |
| GAM | SGD | 0 | +0.000000 | +0.000000 | 761.901 |
| GAM | SGD | 0.25 | +0.000266 | -0.008249 | 806.176 |
| GAM | SGD | 0.5 | +0.006800 | +0.000914 | 985.959 |
| GAM | SGD | 1 | +0.056009 | +0.111834 | 2067.648 |
| GAM | SGD | 1.5 | +0.232725 | +0.455334 | 5270.921 |
| GAM | SGD | 2 | +0.698221 | +1.115471 | 13016.053 |
| GAM | GAM | 0 | +0.000000 | +0.000000 | 761.901 |
| GAM | GAM | 0.25 | +0.002057 | -0.002281 | 724.649 |
| GAM | GAM | 0.5 | +0.005642 | -0.000945 | 767.457 |
| GAM | GAM | 1 | +0.017987 | +0.013447 | 950.242 |
| GAM | GAM | 1.5 | +0.039360 | +0.041232 | 1175.963 |
| GAM | GAM | 2 | +0.071637 | +0.088605 | 1340.090 |

Appendix: existing two-anchor by two-realized-direction bridge (training seed 0). SGD and GAM abbreviate historical SGD+EWC and visual GAM+EWC checkpoints. Each direction is the unit effective-weight displacement from Task0 to Task9 of its named historical sequence, transplanted onto the named Task0 anchor; $r$ is the absolute Euclidean radius in ambient coordinates of the 24 effective Q/V $BA$ matrices (scaling 1). The signed train/test Task0 CE changes use each anchor's fixed 20-class head and deterministic preprocessing. The raw, unnormalized empirical Fisher trace uses per-example unscaled CE gradients on the shared 256 training examples. Radius zero reuses the anchor measurement for both directions. This preserves the existing historical-sequence direction identity: these directions are not the new controlled-continuation directions of Table~\ref{tab:rq2_controlled_continuation}.
