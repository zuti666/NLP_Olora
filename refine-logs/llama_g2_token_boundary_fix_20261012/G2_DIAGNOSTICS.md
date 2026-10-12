# Current-G2 diagnostics added on 2026-10-12

The diagnostic additions preserve the Current-G2 probes, BASE restoration,
realized effective batch, RNG replay, global mean gradients, final clipping,
and exactly one ordinary base-optimizer commit. No optimizer, scheduler,
weight precision, gradient reducer, Fisher, projection, or fallback rule was
changed. Historical frozen experiment sources and logs were not edited.

## Gradient and update metrics

`src/optimer_g2.py` emits the following in `last_stats`; UIETrainer copies
these to every `g2_step_trace.jsonl` row:

| Field | Meaning |
| --- | --- |
| `g2_cos_g0_g1` | Cosine between clean and descent-probe task gradients |
| `g2_cos_g0_g2` | Cosine between clean and outer-probe task gradients |
| `g2_g2_minus_g0_norm` | Global L2 norm of the actual pre-clipping gradient difference |
| `g2_g2_minus_g0_relative_norm` | The previous norm divided by `||g0||` |
| `g2_response_relative_norm` | `||g0-g1|| / ||g0||` |
| `g2_actual_update_over_h_ref` | Actual parameter commit distance divided by raw predicted-SGD distance `eta * ||g0||` |
| `g2_actual_update_over_inner_radius` | Actual commit distance divided by realized inner displacement |
| `g2_actual_update_over_outer_radius` | Actual commit distance divided by realized outer displacement |
| `g2_gradient_clip_scale` | Submitted gradient norm after clipping divided by its norm before clipping |

All gradient comparisons use the full trainable parameter vector, after the
existing global mean reduction and before clipping. A missing gradient is a
zero coordinate, as in the optimizer's existing `_grads()` implementation.
Norms and cosine scalar reductions accumulate in FP64; stored trainables
and gradients retain their original dtype. This does not run an additional
distributed reduction. `g2_gradient_diagnostics_schema` is
`CurrentG2GradientDiagnostics_v1`.

An uncomputed gradient produces `null` diagnostics, including when G2 falls
back to `g0`; we do not pretend that a `g2` probe was computed. Cosine for a
zero vector and a ratio with a zero denominator are undefined and produce
`null`. Legitimate zero differences produce `0`. Diagnostics are JSON-safe
and contain no NaN/Inf.

## Optional BF16 parameter-cast proxy

The core accepts `quantization_diagnostics_interval`, supplied by the trainer
from `g2_quantization_diagnostics_interval`. Its default is **0**, disabled.
For interval `N > 0`, sampling is scheduled at optimizer steps `1, 1+N,
1+2N, ...`. Computing the proxy costs extra casts/reductions, so e.g. `20`
provides occasional samples without running it on every update.

`g2_quantization_proxy` is a nested object with:

- schema `CurrentG2BF16WeightCastProxy_v1`;
- kind `parameter_cast_to_bfloat16`;
- scope `all_trainable_parameters`;
- `sampling_due`, `sampled`, `interval`;
- `proves_forward_visibility: false`;
- `inner` and `outer` objects, or `null` if not sampled/not constructed.

For each constructed probe, temporary copies of BASE and that actual probe
point are cast to BF16. The proxy reports `changed_coordinates`,
`changed_fraction`, `intended_displacement_norm`,
`represented_displacement_norm`, `represented_over_intended_norm`, and
`cosine_represented_vs_intended`. `source_parameter_dtypes` records the
actual source dtypes; no stored weights are cast or overwritten by the
diagnostic. The intended vector is the realized full-precision parameter
displacement, not an ideal direction unaffected by storage rounding.

**This is a hypothetical weight-cast proxy. It does not prove that a real
autocast forward sees, fails to see, or faithfully preserves the probe.**
Actual kernels, LoRA composition, activation rounding, and model state also
matter. The proxy is not a fallback criterion. If a finite stored weight
overflows when hypothetically cast to BF16, `status` becomes
`nonfinite_bf16_cast` and affected measurements are `null`; the diagnostic
does not reject an otherwise valid optimizer step.

The cadence is observational and is deliberately excluded from the strict
algorithm checkpoint configuration. Existing CurrentG2 optimizer checkpoints
remain loadable; loading one does not overwrite the currently requested
diagnostic cadence. The existing `fixed_l2` radius modes remain supported
and separate from the learning rate; this change does not silently replace
the historical `sgd_step` defaults.

## Verification

- **26/26 optimizer tests passed**, including exact quadratic-oracle
  diagnostic values, all supported radius combinations, missing-gradient
  coordinates, computed zero `g2`, zero-gradient/zero-response fallbacks,
  clipping scale, BF16 proxy dtype/cast/zero/overflow behavior, observational
  cadence, and checkpoint compatibility.
- The existing **12-step bitwise equality with the original visual SGD
  Current-G2** and RNG/transaction restoration tests still pass.
- A tiny CPU-autocast BF16 linear model constructs an FP32-distinguishable
  inner perturbation whose BF16-cast coordinates do not change; its true
  replay response is zero. This validates the diagnostic edge case. It does
  **not** establish that the completed Llama runs had the same behavior.
- **10/10 trainer tests passed**, including actual random tiny Llama BF16
  forward / FP32 LoRA, HF and torch AdamW, full and short accumulation
  windows, trace serialization, constructor passthrough, and a two-rank CPU
  Gloo global-mean oracle. The loopback test required execution outside the
  filesystem/network sandbox; `CUDA_VISIBLE_DEVICES=''` throughout.
- Enabling a cadence of 2 preserves four AdamW commits and RNG states
  bitwise compared with diagnostics disabled.

Reproduction logs: `G2_OPTIMIZER_TESTS.log` and `G2_TRAINER_TESTS.log` in this
directory. Python:
`/data/140-1/users/liying/conda_envs/nlora_req/bin/python`.
No new formal GPU training or model downloads were started for these checks.
