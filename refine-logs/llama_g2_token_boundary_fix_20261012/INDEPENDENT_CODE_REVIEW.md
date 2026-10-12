# Independent code review — 2026-10-12

**Conclusion: no unresolved blocker in the reviewed token-boundary fix or
Current-G2 diagnostic changes.** No GPU training, downloads, installs, or
source edits were performed by this reviewer.

## Actual observations

- `src/uie_collator.py` encodes prompt and answer independently, disables
  automatic special tokens, concatenates IDs directly, and retains target
  supervision independently of source truncation. Left-padding offsets are
  correct. Training and inference use the same unpadded prompt prefix.
- Target-only labels and `loss_mask[j - 1]` supervise the first target at
  position `j` through the final prompt logit, and likewise supervise EOS.
  Prompt/padding positions remain excluded. `src/model/llama.py` is identical
  to the historical retry2 snapshot; its existing causal shift matches this
  convention. The legacy OLoRA collator reuses the corrected decoder method.
- A focused CPU witness with a tiny decoder tokenizer lacking both BOS and
  EOS also produced exact target supervision and matching training/inference
  prefixes. This checks token construction, not additional decoder model
  implementations.
- Exact comparison with the retry2 frozen optimizer shows the additions are
  diagnostic reads/casts and metadata. Probe math, reduction ownership,
  fallback rules, restoration, clipping, and the single commit remain intact.
  Trainer changes relative to that snapshot only pass the diagnostic cadence
  and record pipeline/cadence metadata.
- A separate focused CPU witness compared four HF AdamW commits using BF16
  autocast and dropout: historical optimizer versus live optimizer with BF16
  proxy cadence 1. Parameters, complete optimizer state, Python/NumPy/Torch
  RNG, and every pre-existing statistic matched exactly after every commit.
  All four steps executed three CE phases; CUDA remained uninitialized.
- Missing probe gradients and zero-vector/zero-denominator diagnostics are
  represented as `null` appropriately. The BF16 cast proxy uses temporary
  copies, explicitly does not prove actual forward visibility, and reports
  hypothetical cast overflow without rejecting a valid step. Reviewed tests
  cover zero computed g2, fallbacks, missing parameter gradients, clipping,
  proxy overflow, cadence, and checkpoint compatibility; broad suites were
  not repeated by this reviewer.

## Identity finding and resolution

The initial launcher accepted historical profiles for live v2 execution while
leaving their pipeline identity unset. This was reported promptly. The landed
`validate_execution_profile` guard now rejects these profiles before GPU
reservation or source freezing; historical dry-plan inspection remains
available. A focused guard check confirmed this behavior and confirmed all
12 corrected profiles have unique names and explicit v2 identity.

The live collector now excludes the historical baseline by default for v2,
rejects mixed pipeline identities, and leaves unavailable paired metrics
empty, including aggregate baseline means. Historical frozen sources were
only read during this review.

The initial review excluded the then-unfinished precision helper; the
addendum below covers its completed implementation. These checks establish
implementation correctness and observational behavior on the focused CPU
cases, not a new trained Llama performance result or a validated best probe
radius.

## Precision helper addendum

Read-only review covered `scripts_llama/diagnose_g2_probe_precision.py`,
`tests/test_g2_probe_precision.py`, its documentation, and
`PROBE_PRECISION_TINY_CPU.json`. No unresolved blocker remains after the
focused fixes described below.

The helper creates no optimizer and makes zero commits. It requires FP32
masters throughout the model, compares BF16 autocast with autocast disabled,
and applies the same stored FP32 displacement in both modes using the shared
direction `-N(g0_bf16)`. Each measurement restores BASE, captured RNG and
registered buffers first. `finally` restores probe parameter values,
pre-existing gradient objects and contents, buffers/nonpersistent flags,
module training flags and captured RNG on success or exception.

The zero-radius floor is measured by actual repeated forward/backward calls.
Noise-corrected response is `(g0-g(radius))-(g0-g(0))`; linearity compares it
with the correspondingly corrected response at h. Cosines return `null` for
zero vectors. Raw BF16/FP32 response cosine and the separate BF16 weight-cast
proxy have distinct meanings. The proxy cannot establish forward visibility.

The review caught a real autocast-cache defect: under an existing CPU BF16
autocast context, the helper reused BASE weight casts and reported h-response
norm `0.0`, while the standalone witness measured `0.013101960805662831`.
The owner disabled autocast weight caching for diagnostic forward/backward
passes and added a regression that primes the surrounding cache. The same
focused witness now reports `0.013101960805662831` in both contexts, restores
weights, and leaves CUDA uninitialized.

Additional landed fixes constrain CUDA RNG/capability checks to the selected
device, reject a mixed-device model, reject nondefault fresh-base LoRA
settings when loading an adapter, and make CLI Task/Dataset prefixes match
formal profiles by default. New/saved parameter scopes and LoRA-setting
provenance are recorded explicitly. Distinct fixed radii now retain distinct
response keys rather than colliding after eight-digit formatting.

The API accepts an already-collated fixture; the CLI reconstructs a selected
canonical example and identifies its parameter-point scope. It does not
claim to recover an exact historical stage point. The saved tiny CPU report
explicitly identifies a random tiny model, zero commits and no trained Llama
checkpoint; its interpretation correctly excludes conclusions about actual
1B/3B BF16 probe visibility. No full-size model diagnostic, GPU run, or broad
test rerun was performed by this reviewer.
