# Registered HVP numerical revalidation

This is a new, zero-update diagnostic. It reads the original run, checkpoints
and source; it does not modify the original gate or queue. It does not launch
training and no shard can set `full_stage2_allowed=true`.

Use the same Python environment as the original campaign. Physical GPU
selection and idle-GPU locking belong to the external launcher; the script
itself accepts the visible device, normally `cuda:0`.

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python \
  /data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/code/hvp_full_revalidation.py \
  --task 0 --device cuda:0 --output-dir /tmp/hvp-task0-new-output
```

Repeat for tasks 4 and 9 in separate output directories. Each task contains
both registered batches, each batch contains Hstart and Hmid. Approximate cost
is 10–15 minutes per task on an A6000, extrapolated from the preceding
151-second single-response diagnostic; this has not yet been measured for the
new script. Raw tensor storage is roughly 1–2 GB per task shard.

## Acceptance

All original thresholds are retained: relative L2 <= 0.005, cosine >= 0.999;
same-state ordinary vs graph-gradient relative L2 <= 0.00002. A response passes
only if all of the following are true:

1. Production FP32 ordinary and higher-order-graph first gradients agree.
2. Reference FP64 ordinary and higher-order-graph first gradients agree.
3. FP64 HVP agrees with FP64 centered FD at two adjacent fine-grid points.
4. Production FP32 AD HVP agrees with reference FP64 AD HVP.
5. At the same two adjacent fine-grid points, production FP32 AD HVP also
   agrees with FP64 centered FD.

The fixed fine grid is `0.001, 0.0003, 0.0001, 0.00003, 0.00001, 0.000003`.
The production FP32 centered FD scan is also saved as a numerical saturation
diagnostic. A passing FP64 reference alone cannot authorize Stage2.

The prior worst-case Task9/batch0/Hstart production FP32-versus-FP64 HVP
discrepancy was about 0.905%, already above the unchanged 0.5% threshold.
Consequently a renewed failure is an expected, scientifically meaningful
outcome, not a reason to loosen the threshold.

## Objective and coordinates

Points and physical directions are built by the original registered FP32 math
backend probe. Hstart uses the saved FP32 phi1; Hmid uses the saved FP32 rounded
midpoint; d is the saved FP32 phi2-minus-phi1 displacement. Each coordinate,
input, frozen weight and floating registered buffer is promoted exactly to
FP64. No FP64 re-estimation of the probe direction is used.

FP64 evaluates two disjoint 64-sample chunk CE means weighted by 0.5 each, so
the mathematical scalar loss is the same 128-sample mean CE. The native model
has no training BatchNorm or nonzero dropout/drop-path; these are asserted.
Historical saved LoRA eval factors must be empty. Chunking changes GEMM shapes
and reduction order, so a cross-precision discrepancy cannot be attributed
uniquely to dtype. Within-reference FD/HVP agreement is still directly tested.

## Artifacts and status schema

`hvp_revalidation_report.json` contains original full source/checkpoint/config/
trainer/queue pins, original model and input/target hashes, exact batch IDs,
per-response raw tensor filenames and hashes, FP32 and FP64 scan rows, actual
perturbations, effective direction errors, per-block errors, state-restoration
checks, and protected-file hashes after execution.

Relevant paths are:

- `batches[i].responses.hstart|hmid.status`
- `batches[i].responses.hstart|hmid.acceptance_checks`
- `batches[i].responses.hstart|hmid.failed_checks`
- `batches[i].responses.hstart|hmid.fp64_reference.production_fp32_vs_fp64_hvp`
- `batches[i].responses.hstart|hmid.float64_scan[j].joint_accepted`
- `shard_numerical_pass`
- `full_stage2_allowed` (always false for this diagnostic)

An exit code of 0 means this task shard passed all numerical checks. Exit 1
means numerical failure; another exception is recorded as `status=error`.
All three task shards and the independent first-gradient consistency audit
must be reviewed before any launch decision. The original queue and original
gate remain unchanged.

Only CPU AST syntax validation has been run at code handoff; no GPU work was
started by the implementation agent.
