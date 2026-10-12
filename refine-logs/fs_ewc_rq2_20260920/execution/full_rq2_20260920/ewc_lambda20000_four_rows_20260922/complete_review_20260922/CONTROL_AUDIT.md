# Lambda-zero controls audit

All 9 controls passed fresh CPU verification of saved assets, parent identity, 50 epochs, input/head pairing, frozen backbone/old head, and effective Q/V displacement.

| arm | seed | a11 | a21 | a22 | BWT | A2 | dQV | raw-F weighted squared drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla | 0 | 87.30 | 83.06 | 76.38 | -4.24 | 79.720 | 0.38129096 | 1.454008922e-05 |
| vanilla | 1 | 86.64 | 83.06 | 76.77 | -3.58 | 79.915 | 0.35967808 | 1.708049434e-05 |
| vanilla | 2 | 87.30 | 81.11 | 75.98 | -6.19 | 78.545 | 0.39154208 | 2.300746268e-05 |
| shaping_only | 0 | 86.97 | 83.71 | 81.50 | -3.26 | 82.605 | 0.50302108 | 6.551885816e-06 |
| shaping_only | 1 | 86.97 | 83.06 | 81.50 | -3.91 | 82.280 | 0.48702259 | 7.70005721e-06 |
| shaping_only | 2 | 87.95 | 83.39 | 79.53 | -4.56 | 81.460 | 0.51687681 | 9.981208989e-06 |
| task1_shaping_no_ewc | 0 | 86.97 | 86.64 | 72.83 | -0.33 | 79.735 | 0.36338164 | 5.164425561e-06 |
| task1_shaping_no_ewc | 1 | 86.97 | 87.30 | 74.41 | 0.33 | 80.855 | 0.34177180 | 7.750961522e-06 |
| task1_shaping_no_ewc | 2 | 87.95 | 85.67 | 71.65 | -2.28 | 78.660 | 0.36975829 | 9.117572125e-06 |

Native saved CIL matrices rounded to two decimals per seed across all rows; a11_exact retained separately.

Historical controls retain true normalization flags, but native penalty is multiplied by lambda=0, mean recorded EWC loss is zero, dual/projection/extra penalty are false. Flags only change inactive stored Fisher or diagnostic quantities and do not alter these two-task trained weights.

Every input index, head initialization and first augmented batch in each of 50 training plus final fit-Fisher passes are verified; later augmented batches were not independently hashed.

Valid matched EWC-vs-zero comparisons exist for Vanilla/EWCLoRA, Shaping-only/FS-EWCLoRA, and Task1-shaping-no-EWC/Task1-shaping. No SGD-to-GAM lambda-zero arm is present, so Task2-shaping does not have that direct no-EWC control.

sum over 24 Q/V effective BA matrices of prepared Task0 raw batch-gradient Fisher times squared endpoint-minus-parent BA; FP64 reduction; same raw Fisher as matched EWC arm.

Historical files were only read. No model inference, training, or GPU was used. All per-file hashes and configuration differences are in CONTROL_AUDIT.json.
