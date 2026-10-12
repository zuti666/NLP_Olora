# Llama pure G2: decoder repair, diagnostics and validation plan

Date: 2026-10-12. This is a code/configuration change with CPU validation.
No corrected full-model GPU training, queue or new method result was started.
The completed retry2 experiment and its frozen source remain unchanged.

## What the old evidence establishes

Both local Llama-3.2 tokenizers merge the old jointly encoded `Answer:` and
five answers across the prompt/target boundary: dbpedia Athlete, Educational
Institution and Natural Place; amazon negative; yahoo Society & Culture.
The old decoder collator also inserted a duplicate BOS and derived the loss
mask from a separately tokenized label length containing BOS. Consequently,
removing the five categories does not repair all old supervision errors.

The official metric uses normalized exact match, task-equal FAA, and the
same archived seed0 predictions. Deltas below are G2 minus IncLoRA, averaged
over Orders1/2/3, in percentage points:

| Model | Original full-label delta FAA | Delta after excluding five gold labels |
|---|---:|---:|
| Llama-3.2-1B | +0.017544 | -0.859378 |
| Llama-3.2-3B | +2.002193 | +0.713285 |

The final column is a conditional sensitivity analysis, not a repaired
benchmark. The affected five classes contribute 1.391447 of the 3B raw
2.002193-point difference by correct-sample counting; that is descriptive,
not proof that the defect caused the difference. Several affected classes
have substantial accuracy (e.g. 1B Order1 Natural Place: 80.8194%/53.8175%
for G2/IncLoRA), so the assertion that all except Athlete have zero accuracy
is incorrect. Order1 and Order2 share identical first-stage dbpedia results;
they are not independent repetitions. The scratchpad's 1B +0.026316 mean
uses strict exact match; it must not be mixed with normalized exact match.

Full class statistics, token IDs, filtered denominators and raw prediction
hashes are in `LABEL_BOUNDARY_ANALYSIS.md`, the associated CSV/JSON files,
and `LABEL_BOUNDARY_MANIFEST.json`. None of these single-seed results supports
a corrected-pipeline or statistically established G2 advantage.

## Landed implementation

- `src/uie_collator.py`: prompt and answer are encoded separately with
  `add_special_tokens=False`; their token IDs are concatenated directly.
  Explicit single BOS and target EOS, target-only labels, and predictor-position
  loss masks supervise the complete first answer token. Source and target
  have independent budgets. Source truncation retains BOS and the prompt tail,
  including the answer cue; training and generation use the same prompt IDs.
- `src/uie_collator_OLoRA.py`: decoder mode shares the implementation. Encoder–
  decoder collation is unchanged. The new decoder identity is
  `decoder_token_ids_v2`.
- `src/optimer_g2.py`: adds `g2_cos_g0_g1`, `g2_cos_g0_g2`,
  `g2_g2_minus_g0_norm`, `g2_g2_minus_g0_relative_norm`,
  `g2_response_relative_norm`, clipping scale, and actual-update/probe-radius
  ratios. These use the globally averaged, pre-clipping gradients. Undefined
  ratios/cosines and uncomputed probes are JSON `null`.
- `src/uie_trainer_lora.py` and `src/run_uie_lora.py`: pass diagnostic sampling
  cadence, write pipeline identity and metadata to audit files, and retain
  normal step traces.
- Optional `g2_quantization_diagnostics_interval` samples a temporary BF16
  weight-cast proxy; default 0 disables it. The proxy explicitly records
  `proves_forward_visibility=false`. It changes neither stored precision nor
  fallback decisions.
- `scripts_llama/diagnose_g2_probe_precision.py`: measures actual same-BASE,
  same-batch, replayed-RNG gradient responses at 0, h/2, h, 2h and optional
  fixed radii under BF16 autocast and full FP32. It reports the h=0 noise
  floor, response linearity and BF16/FP32 response cosine, restores parameters,
  buffers, gradients, mode and RNG, and commits no optimizer update.
  Each probe disables the autocast weight cache, including when called inside
  another autocast context. CUDA RNG is scoped to the selected model device;
  mixed-device models are rejected. Fresh and saved adapter settings and CLI
  prompt prefixes are explicit, so diagnostic parameter points are identified.
- `scripts_llama/run_g2_llama.py`: real execution requires an explicitly
  versioned v2 profile; old profiles remain inspectable via dry-run. Acceptance
  checks pipeline identity and the newly required direction diagnostics.
- `scripts_llama/collect_g2_orders_results.py`: v2 runs cannot automatically
  reuse the old baseline CSV. Missing corrected baselines leave paired fields
  empty; supplied baselines must carry the matching pipeline identity and
  pinned provenance.

G2's mathematical probes are unchanged:

\[
g_0=\nabla L_B(\theta),\quad
g_1=\nabla L_B(\theta-qN(g_0)),\quad
a=g_0-g_1,\quad
g_2=\nabla L_B(\theta+RN(a)).
\]

After exact BASE restoration, only g2 is clipped and submitted to one HF
AdamW step. There is no Fisher/EWC, history, orthogonal projection, model
averaging or final direction normalization. Saved adapter concatenation is
the existing IncLoRA mechanism. Dynamic-radius old results still mean
q=R=h=learning_rate*||g0||; independent radii are an explicit configuration
change, not a silent replacement of that definition. Norm similarity alone
does not establish gradient-vector similarity.

## Candidate configurations

Twelve files in `configs/g2_llama/corrected_v2/` cover both models, all three
Orders, and two candidate settings. Twelve corresponding executable wrappers
are in the requested `scripts_order{1,2,3}_llama{1B,3B}` folders. All have
`status=planned_not_launched`.

| Setting | Candidate A | Candidate B |
|---|---:|---:|
| Inner mode / q | fixed_l2 / 0.05 | fixed_l2 / 0.05 |
| Outer mode / R | fixed_l2 / 0.05 | fixed_l2 / 0.20 |
| Diagnostic proxy cadence | 20 steps | 20 steps |
| Planned seeds | 0,42,1993 | 0,42,1993 |
| Decoder version | decoder_token_ids_v2 | decoder_token_ids_v2 |

Both q and R are global L2 distances over the current trainable FP32 LoRA
coordinates, independent of learning rate. They are calibration candidates,
not proven optimal radii. Shared settings remain: rank8 per new task,
alpha32, dropout0.1, Q/V; saved ranks8/16/24/32; BF16 forward; HF AdamW,
epsilon1e-6, weight decay0, betas0.9/0.999, clip1; DDP4, batch1, accumulation16,
effective batch64; constant LR0.001 in the first stage and0.0001 afterwards;
source512, target/generation50. Task-specific capped updates are
dbpedia218, amazon78, yahoo156, agnews62, total514/Order.

Orders are 1: dbpedia→amazon→yahoo→agnews;
2: dbpedia→amazon→agnews→yahoo;
3: yahoo→amazon→agnews→dbpedia.

## Staged validation, currently unlaunched

1. Run the precision diagnostic on each actual model at a declared parameter
   point and correctly collated realized batch before expensive continuation.
   Compare h=0 replay noise, h/2/h/2h response, and q=0.05/R=0.05 or0.2. Check
   finite values, response/noise ratio, BF16/FP32 agreement, clipping, actual
   update/probe distances and g0–g2 vector changes. The current tiny-model CPU
   artifact validates the tool; it is not this full-model acceptance.
2. Pilot Order1/seed0 for both models with a freshly trained v2 IncLoRA
   baseline and the two fixed-radius G2 candidates. That is two baseline
   chains plus four G2 chains (24 formal stages), each with its own GPU smoke
   gate. Keep all data, precision, optimizer, seed and capped updates matched.
   Fresh baseline uses `optimizer_type=adamw_hf`, the same current collator,
   FP32 trainable LoRA, and no G2 optimizer; the G2-only launcher does not
   implement baseline execution. Do not invoke old baseline shell defaults,
   whose device count/accumulation/output paths differ from this frozen budget.
3. Freeze the radius using diagnostics and a declared training/dev criterion,
   then complete both baseline and selected G2 at models×Orders×seeds
   2×3×3. This is 18 chains/72 formal stages per method, with compatible
   pilot chains reused. The three-seed catalog for both candidate radii is
   broader; its availability does not mean all 36 G2 chains are scheduled.
4. If separating the tokenization repair from the radius intervention is
   necessary, add an explicitly versioned v2 dynamic-radius G2 control. The
   archived v1 dynamic-radius scores cannot isolate either intervention.

Report per-seed FAA/AAA/BWT and paired differences using normalized exact
match and a common denominator. Average Orders within each seed before
cross-seed aggregation; use sample SD (ddof=1), n=3. Also report compute,
CE phases, real commits, fallback, clipping, gradient cosines/differences,
radius/update ratios and precision diagnostics. Do not treat stages or shared
first-task results as independent seeds.

## Reproducible inspection

```bash
bash scripts_order1_llama1B/IncLoRA-G2-FixedQ005-R020-TokV2_LLaMA3-1b.sh \
  --seeds 0 42 1993 --run-root /tmp/llama_g2_v2_example --dry-run
```

All twelve literal wrappers pass `bash -n` and this standard-library dry-plan
check, expanding 144 candidate formal stages plus72 smoke stages with zero
CUDA/model imports or training output directories. Receipt:
`PRESET_DRY_RUN_CHECK.json`; repeat with `python3 check_corrected_presets.py`
from this directory, or its repository-relative path.

The main acceptance tests cover real local 1B/3B tokenizers, native tiny
Llama causal loss/backward, the native BF16/FP32-LoRA G2 window, HF/Torch
AdamW and two-rank CPU Gloo. Core math retains bitwise comparisons with the
old optimizer. Independent review is recorded in `INDEPENDENT_CODE_REVIEW.md`.
The completed suites total **109 passed CPU tests**: collator16, G2 core26,
Trainer10, precision helper14, corrected integration7, launcher/queue36.
The independent helper review caught and verified a fix for stale casts under
nested autocast; full-size Llama probe visibility remains unmeasured.
Test logs and the precision-tool guide are in this directory. No package
install, old result replacement, or new formal GPU experiment occurred.
