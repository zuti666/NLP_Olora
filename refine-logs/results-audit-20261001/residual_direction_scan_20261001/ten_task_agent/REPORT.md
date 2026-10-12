# Ten-task residual-direction scan: independent paired audit

Existing offline diagnostics are complete: 9/9 anchors and 825/825 counterfactual records match the original reports. The three requested scan families account for 429 candidate-batch cells and 1,404 paired metric observations. All use training seed 2; 33 probe batches are not independent training seeds.

R = g1 + lambda * r_perp; S = (1+a) * g1 + lambda * r_perp, where S(lambda=1) is g1+g3; U = unit((1-alpha)*unit(g1)+alpha*unit(g3)). R/S grids are [0,.5,1,2], U is [0,.25,.5,.75,1].

Each candidate is rescaled to the same effective LoRA BA + FC norm as eta_ref=.02*g1. The paired contrast is CE(candidate) minus CE(the same family at coefficient 0), at the same checkpoint and probe batch. Negative is better. These are equal-probe-batch means; adding a residual also changes the matched whole-direction scale.

CL=current-local; CA=current-allseen; OA=old-task-macro-allseen; OL=old-task-macro-local. Old metrics are unavailable at Task0. These metrics use fixed test batches, not the full evaluation set.

## Lowest recorded mean-CE coefficient on each grid

The four entries in each cell are CL / CA / OA / OL. A grid minimum is exploratory, not a selected deployment coefficient.

| Anchor | R lambda | S lambda | U alpha |
|---|---|---|---|
| g1_task0 | 0 / 0 / NA / NA | 0 / 0 / NA / NA | 0 / 0 / NA / NA |
| g1_task4 | 0.5 / 0 / 2 / 0 | 2 / 0 / 2 / 0 | 0.75 / 0.25 / 1 / 0.25 |
| g1_task9 | 0 / 0 / 2 / 2 | 0 / 0 / 2 / 2 | 0 / 0 / 1 / 1 |
| g3_task0 | 0 / 0 / NA / NA | 0 / 0 / NA / NA | 0 / 0 / NA / NA |
| g3_task4 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| g3_task9 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| sum_task0 | 0 / 0 / NA / NA | 0 / 0 / NA / NA | 0 / 0 / NA / NA |
| sum_task4 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| sum_task9 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

All g3 and sum anchors have mean curves that worsen monotonically with increasing coefficient. g1_task0 R is nonmonotone, but coefficient 0 still has the lowest mean; its S/U curves worsen monotonically. At g1_task9, all three families monotonically worsen current means while improving old means. g1_task4 has mixed metric behavior and the only interior mean minima.

## Coefficient 1 versus 0 at the four focal anchors

Values are 1000 x paired CE difference (mCE); negative improves CE. R/S mean lambda=1; U means alpha=1.

| Anchor | Family | CL | CA | OA | OL |
|---|---|---:|---:|---:|---:|
| g1_task4 | R | -0.667422 | +2.401053 | -1.179546 | +0.098091 |
| g1_task4 | S | -0.765950 | +0.557582 | -0.651725 | +0.035254 |
| g1_task4 | U | -0.820179 | +1.158322 | -0.903659 | +0.053361 |
| g1_task9 | R | +6.563774 | +15.420082 | -3.686648 | -1.897768 |
| g1_task9 | S | +4.824761 | +10.299976 | -2.868574 | -1.532006 |
| g1_task9 | U | +5.462818 | +12.035174 | -3.203525 | -1.694361 |
| g3_task4 | R | +4.665711 | +11.619942 | +4.220255 | +1.131495 |
| g3_task4 | S | +2.335062 | +5.761277 | +2.123803 | +0.553444 |
| g3_task4 | U | +3.606478 | +8.962631 | +3.317036 | +0.884933 |
| g3_task9 | R | +5.805285 | +11.426094 | +6.874127 | +2.598150 |
| g3_task9 | S | +2.695695 | +5.646437 | +3.639088 | +1.361752 |
| g3_task9 | U | +4.733599 | +9.468861 | +5.823642 | +2.197351 |

At g1_task9, coefficient 1 worsens CA on all 3 probe batches and improves OA and OL on all 3; CL improves on 1/3 and worsens on 2/3. Thus the current/old tradeoff is strongest for CA/OA/OL. At g3_task9, all four metrics worsen on all 3 batches for all three coefficient-1 candidates. At g3_task4, R(lambda=1) worsens all four metrics on all 4 batches; S/U worsen old metrics on 4/4 and current metrics on 3/4.

## g1_task4 interior minima and sensitivity

R has the lowest CL at lambda=.5 (paired delta -0.000776347 CE). U has the lowest CL at alpha=.75, but alpha=1 differs by only 5.80e-7 CE. U has the lowest CA and OL at alpha=.25. Old-allseen improves to the largest recorded coefficient in all three families; R/S old-local worsens with coefficient.

U(alpha=.25) has equal-batch paired deltas [CL,CA,OA,OL] = [-0.000443608,-0.000163359,-0.000227727,-0.000010155]. The probe batches contain [128,128,128,42] examples. Probe-size weighting, provided only as a separate sensitivity calculation, gives [-0.000190437,+0.000453600,-0.000062198,+0.000048690]. CA and OL signs reverse. There is no robust joint optimum from these four batches.

## Full mean curves

Each array is 1000 x paired CE difference vs coefficient 0, ordered by the respective family grid.

| Anchor | Family | Metric | mCE array | Mean curve trend |
|---|---|---|---|---|
| g1_task0 | R | current_local | [+0.000000, +1.155660, +1.304224, +1.156539] | nonmonotone |
| g1_task0 | R | current_allseen | [+0.000000, +1.155660, +1.304224, +1.156539] | nonmonotone |
| g1_task0 | S | current_local | [+0.000000, +0.606611, +1.009464, +1.286089] | monotone_increasing |
| g1_task0 | S | current_allseen | [+0.000000, +0.606611, +1.009464, +1.286089] | monotone_increasing |
| g1_task0 | U | current_local | [+0.000000, +0.373840, +0.748977, +1.046181, +1.230717] | monotone_increasing |
| g1_task0 | U | current_allseen | [+0.000000, +0.373840, +0.748977, +1.046181, +1.230717] | monotone_increasing |
| g1_task4 | R | current_local | [+0.000000, -0.776347, -0.667422, -0.407518] | nonmonotone |
| g1_task4 | R | current_allseen | [+0.000000, +1.136967, +2.401053, +3.562815] | monotone_increasing |
| g1_task4 | R | old_allseen | [+0.000000, -0.842586, -1.179546, -1.318172] | monotone_decreasing |
| g1_task4 | R | old_local | [+0.000000, +0.054432, +0.098091, +0.151277] | monotone_increasing |
| g1_task4 | S | current_local | [+0.000000, -0.559508, -0.765950, -0.771934] | monotone_decreasing |
| g1_task4 | S | current_allseen | [+0.000000, +0.044767, +0.557582, +1.678916] | monotone_increasing |
| g1_task4 | S | old_allseen | [+0.000000, -0.346981, -0.651725, -1.034379] | monotone_decreasing |
| g1_task4 | S | old_local | [+0.000000, +0.014415, +0.035254, +0.072049] | monotone_increasing |
| g1_task4 | U | current_local | [+0.000000, -0.443608, -0.714134, -0.820758, -0.820179] | nonmonotone |
| g1_task4 | U | current_allseen | [+0.000000, -0.163359, +0.000224, +0.481774, +1.158322] | nonmonotone |
| g1_task4 | U | old_allseen | [+0.000000, -0.227727, -0.454821, -0.682987, -0.903659] | monotone_decreasing |
| g1_task4 | U | old_local | [+0.000000, -0.010155, +0.001976, +0.026636, +0.053361] | nonmonotone |
| g1_task9 | R | current_local | [+0.000000, +5.918454, +6.563774, +6.871835] | monotone_increasing |
| g1_task9 | R | current_allseen | [+0.000000, +13.276125, +15.420082, +16.545222] | monotone_increasing |
| g1_task9 | R | old_allseen | [+0.000000, -3.402211, -3.686648, -3.783979] | monotone_decreasing |
| g1_task9 | R | old_local | [+0.000000, -1.780570, -1.897768, -1.926449] | monotone_decreasing |
| g1_task9 | S | current_local | [+0.000000, +3.078583, +4.824761, +6.030841] | monotone_increasing |
| g1_task9 | S | current_allseen | [+0.000000, +6.227151, +10.299976, +13.670995] | monotone_increasing |
| g1_task9 | S | old_allseen | [+0.000000, -1.921670, -2.868574, -3.463349] | monotone_decreasing |
| g1_task9 | S | old_local | [+0.000000, -1.049789, -1.532006, -1.810055] | monotone_decreasing |
| g1_task9 | U | current_local | [+0.000000, +0.976660, +2.520952, +4.183598, +5.462818] | monotone_increasing |
| g1_task9 | U | current_allseen | [+0.000000, +2.056709, +5.204323, +8.830242, +12.035174] | monotone_increasing |
| g1_task9 | U | old_allseen | [+0.000000, -0.798497, -1.723419, -2.582304, -3.203525] | monotone_decreasing |
| g1_task9 | U | old_local | [+0.000000, -0.448512, -0.949901, -1.394192, -1.694361] | monotone_decreasing |
| g3_task0 | R | current_local | [+0.000000, +1.341537, +1.996487, +2.447054] | monotone_increasing |
| g3_task0 | R | current_allseen | [+0.000000, +1.341537, +1.996487, +2.447054] | monotone_increasing |
| g3_task0 | S | current_local | [+0.000000, +0.583529, +1.098678, +1.775667] | monotone_increasing |
| g3_task0 | S | current_allseen | [+0.000000, +0.583529, +1.098678, +1.775667] | monotone_increasing |
| g3_task0 | U | current_local | [+0.000000, +0.372201, +0.787660, +1.205310, +1.582235] | monotone_increasing |
| g3_task0 | U | current_allseen | [+0.000000, +0.372201, +0.787660, +1.205310, +1.582235] | monotone_increasing |
| g3_task4 | R | current_local | [+0.000000, +2.870952, +4.665711, +6.288360] | monotone_increasing |
| g3_task4 | R | current_allseen | [+0.000000, +7.082023, +11.619942, +15.738656] | monotone_increasing |
| g3_task4 | R | old_allseen | [+0.000000, +2.564780, +4.220255, +5.622879] | monotone_increasing |
| g3_task4 | R | old_local | [+0.000000, +0.667432, +1.131495, +1.525674] | monotone_increasing |
| g3_task4 | S | current_local | [+0.000000, +1.176329, +2.335062, +4.095283] | monotone_increasing |
| g3_task4 | S | current_allseen | [+0.000000, +2.887090, +5.761277, +10.184830] | monotone_increasing |
| g3_task4 | S | old_allseen | [+0.000000, +1.071714, +2.123803, +3.732875] | monotone_increasing |
| g3_task4 | S | old_local | [+0.000000, +0.273770, +0.553444, +0.997407] | monotone_increasing |
| g3_task4 | U | current_local | [+0.000000, +0.760509, +1.674746, +2.657797, +3.606478] | monotone_increasing |
| g3_task4 | U | current_allseen | [+0.000000, +1.873839, +4.137338, +6.584167, +8.962631] | monotone_increasing |
| g3_task4 | U | old_allseen | [+0.000000, +0.729509, +1.574926, +2.465501, +3.317036] | monotone_increasing |
| g3_task4 | U | old_local | [+0.000000, +0.189250, +0.412086, +0.651585, +0.884933] | monotone_increasing |
| g3_task9 | R | current_local | [+0.000000, +3.164389, +5.805285, +8.523672] | monotone_increasing |
| g3_task9 | R | current_allseen | [+0.000000, +6.544945, +11.426094, +16.311352] | monotone_increasing |
| g3_task9 | R | old_allseen | [+0.000000, +4.165099, +6.874127, +9.366214] | monotone_increasing |
| g3_task9 | R | old_local | [+0.000000, +1.559725, +2.598150, +3.568999] | monotone_increasing |
| g3_task9 | S | current_local | [+0.000000, +1.166710, +2.695695, +5.233862] | monotone_increasing |
| g3_task9 | S | current_allseen | [+0.000000, +2.599961, +5.646437, +10.386247] | monotone_increasing |
| g3_task9 | S | old_allseen | [+0.000000, +1.777185, +3.639088, +6.321760] | monotone_increasing |
| g3_task9 | S | old_local | [+0.000000, +0.659967, +1.361752, +2.387145] | monotone_increasing |
| g3_task9 | U | current_local | [+0.000000, +0.591327, +1.677073, +3.149840, +4.733599] | monotone_increasing |
| g3_task9 | U | current_allseen | [+0.000000, +1.356858, +3.604889, +6.486061, +9.468861] | monotone_increasing |
| g3_task9 | U | old_allseen | [+0.000000, +1.013928, +2.456985, +4.157937, +5.823642] | monotone_increasing |
| g3_task9 | U | old_local | [+0.000000, +0.376998, +0.918096, +1.561670, +2.197351] | monotone_increasing |
| sum_task0 | R | current_local | [+0.000000, +0.598863, +0.849515, +1.005054] | monotone_increasing |
| sum_task0 | R | current_allseen | [+0.000000, +0.598863, +0.849515, +1.005054] | monotone_increasing |
| sum_task0 | S | current_local | [+0.000000, +0.266716, +0.486195, +0.745222] | monotone_increasing |
| sum_task0 | S | current_allseen | [+0.000000, +0.266716, +0.486195, +0.745222] | monotone_increasing |
| sum_task0 | U | current_local | [+0.000000, +0.162154, +0.341222, +0.512749, +0.658661] | monotone_increasing |
| sum_task0 | U | current_allseen | [+0.000000, +0.162154, +0.341222, +0.512749, +0.658661] | monotone_increasing |
| sum_task4 | R | current_local | [+0.000000, +1.493921, +2.348189, +2.995585] | monotone_increasing |
| sum_task4 | R | current_allseen | [+0.000000, +5.057466, +8.133720, +10.560653] | monotone_increasing |
| sum_task4 | R | old_allseen | [+0.000000, +1.168560, +1.862820, +2.367165] | monotone_increasing |
| sum_task4 | R | old_local | [+0.000000, +0.167903, +0.286143, +0.372505] | monotone_increasing |
| sum_task4 | S | current_local | [+0.000000, +0.625068, +1.214009, +2.058553] | monotone_increasing |
| sum_task4 | S | current_allseen | [+0.000000, +2.087350, +4.111533, +7.110857] | monotone_increasing |
| sum_task4 | S | old_allseen | [+0.000000, +0.497952, +0.969160, +1.650795] | monotone_increasing |
| sum_task4 | S | old_local | [+0.000000, +0.069726, +0.141630, +0.253843] | monotone_increasing |
| sum_task4 | U | current_local | [+0.000000, +0.404638, +0.871471, +1.357658, +1.812767] | monotone_increasing |
| sum_task4 | U | current_allseen | [+0.000000, +1.353731, +2.946031, +4.635792, +6.251017] | monotone_increasing |
| sum_task4 | U | old_allseen | [+0.000000, +0.333283, +0.711702, +1.103278, +1.469344] | monotone_increasing |
| sum_task4 | U | old_local | [+0.000000, +0.048291, +0.105310, +0.167046, +0.226740] | monotone_increasing |
| sum_task9 | R | current_local | [+0.000000, +1.406792, +2.219518, +2.926973] | monotone_increasing |
| sum_task9 | R | current_allseen | [+0.000000, +3.549136, +5.734272, +7.719040] | monotone_increasing |
| sum_task9 | R | old_allseen | [+0.000000, +1.873324, +2.871383, +3.713885] | monotone_increasing |
| sum_task9 | R | old_local | [+0.000000, +0.630113, +0.968000, +1.253265] | monotone_increasing |
| sum_task9 | S | current_local | [+0.000000, +0.566800, +1.141964, +1.958358] | monotone_increasing |
| sum_task9 | S | current_allseen | [+0.000000, +1.387816, +2.850606, +5.014468] | monotone_increasing |
| sum_task9 | S | old_allseen | [+0.000000, +0.799570, +1.546351, +2.561334] | monotone_increasing |
| sum_task9 | S | old_local | [+0.000000, +0.268736, +0.520163, +0.863263] | monotone_increasing |
| sum_task9 | U | current_local | [+0.000000, +0.322684, +0.757120, +1.251636, +1.731701] | monotone_increasing |
| sum_task9 | U | current_allseen | [+0.000000, +0.767537, +1.849150, +3.123259, +4.397759] | monotone_increasing |
| sum_task9 | U | old_allseen | [+0.000000, +0.473289, +1.061304, +1.694712, +2.287982] | monotone_increasing |
| sum_task9 | U | old_local | [+0.000000, +0.159303, +0.357226, +0.570732, +0.770877] | monotone_increasing |

## Audit and claim limits

- Verified all 9 report SHA values against both anchor index and ledger, all completion receipt hashes, returncode 0, empty validation errors, report finite/restore gates, actual source config/code/checkpoint pins, and all 825 row-level values/fingerprints.
- Maximum effective-norm matching relative error: 1.0015848387992116e-05, below the original .001 tolerance.
- Per-arm probe batches: Task0=4, Task4=4, Task9=3, for 33 total; the three Task9 batches are the recorded completed recovery set.
- All 225 summary candidate rows were independently recomputed. Source batch win counts use a 1e-7 CE threshold.
- This is local one-step behavior at existing task-end checkpoints, not 13 newly trained methods. No AAA/FAA improvement, universal residual benefit, optimal coefficient, seed-level significance, or causal account of the full training gap follows.
- Worse than the matched g1 direction does not necessarily mean worse than taking no update. E.g. g3_task9 R(lambda=1) still lowers CL versus its original checkpoint by approximately .004966 CE.
- Parameter-gradient-space orthogonality does not imply noninterference with old tasks. Coefficients are not equivalent across R/S/U; a differs by anchor and batch.

Artifacts: residual_scan_audit.json contains original file hashes and audit evidence; family_metric_curves.csv contains full paired curves and sensitivities; paired_batch_contrasts.csv contains each paired observation; coefficient1_contrasts.csv contains focal contrasts. Original source files were not changed.
