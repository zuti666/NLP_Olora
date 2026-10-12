# Mean50 r=0 identity debug

Two batches on the original failed physical GPU1; no training or committed checkpoint update.

The original failed JSON did not save its triggering value. This isolated run reproduced a threshold exceedance, not necessarily the exact original numeric value.

| batch | r0g3/r1g1 rel | within-r0 g3/g1 rel | across-radius g1 rel | r0 probe3−probe1 L2 |
|---|---:|---:|---:|---:|
|0|2.729198828e-05|1.741398408e-05|1.769620473e-05|1.380490506e-06|
|1|0.00121557173|0.001154286737|6.192893618e-05|2.095987068e-06|

All independent native-operation parameter checks are exact. Actual-probe CE autograd checks pass with maximum relative discrepancy 1.022e-7. SGD at diagnostic lr=0 leaves post-unperturb parameters exactly unchanged. Full snapshots restore parameters, gradients, buffers, modes and RNG. Active dropout/drop-path list is empty.

At r=0 the real-number identity does not remove native float32 add/sub roundtrip. These probes differ in coordinates; repeated g0/g1 computations also vary slightly across scans. This evidence supports replacing the cross-probe approximate identity as a hard correctness specification, but does not uniquely identify the GPU kernel or amplify-error mechanism. This is not a SAM-vs-GAM norm reduction difference: both scans use the same GAM implementation.

Proposed v2 retains the native path and old 5e-4 comparison as reported measurement, adds exact same-dtype operation checks plus independent CE at each actual point, and preserves all finite/matching/restore gates. It does not loosen 5e-4. Old outputs/scripts remain immutable.

## Instrumentation boundary

On GPU, instrumented versus uninstrumented native GAM r1 g3 differs relatively by 0.00184658 (batch0) and 0.00412692 (batch1); same-point CE checks remain near 1e-7. CPU controls are bitwise identical. Thus the recovered Mean result is explicitly schema2 alongside four schema1 anchors, not a claim of bitwise reproduction.
