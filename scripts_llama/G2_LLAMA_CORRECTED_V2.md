# Pure G2 Llama — corrected decoder pipeline

The archived seed0 six-Order experiment completed on 2026-10-12. Its prompt/
target tokenization has since been found defective. Raw historical scores and
frozen sources are preserved, but they are not corrected-pipeline baselines.

Current code separately encodes prompt and answer and supervises every answer
token, including the first, with one BOS and target EOS. The decoder version
is `decoder_token_ids_v2`. G2 additionally logs gradient cosines, relative
gradient changes, clipping and real-update/probe-radius ratios.

The mathematical update remains: g0 at BASE; g1 at BASE-q*N(g0);
g2 at BASE+R*N(g0-g1); restore BASE; clip and submit only g2 to one AdamW
step. Here g1 is the new algorithm's **descending** probe, distinct from
the ascending GAM/SAM probe using the same historical notation.

Twelve HF AdamW candidate profiles cover Llama-3.2-1B/3B and Orders1–3:

| Parameter | Value |
|---|---|
| Profiles | `configs/g2_llama/corrected_v2/*.json` |
| q / R | fixed_l2, q0.05; R0.05 or0.2 |
| Seeds available | 0,42,1993 |
| Adapter | Fresh rank8/task, alpha32, dropout0.1, Q/V |
| Precision | BF16 forward, FP32 trainable LoRA |
| Optimizer | Transformers AdamW, epsilon1e-6, WD0, clip1 |
| Effective batch | 4 DDP ranks × batch1 × accumulation16 =64 |
| LR | First stage0.001; later0.0001, constant |
| Update caps | dbpedia218/amazon78/yahoo156/agnews62 |
| EWC/Fisher, projection, averaging | Disabled |
| Status | Planned; no corrected formal training launched |

Inspect a three-seed plan without CUDA or output creation:

```bash
bash scripts_order1_llama1B/IncLoRA-G2-FixedQ005-R020-TokV2_LLaMA3-1b.sh \
  --seeds 0 42 1993 --run-root /tmp/llama_g2_v2_example --dry-run
```

Matching wrappers exist in each requested model/Order folder, for both
R005/R020. Real execution requires explicit four idle physical GPUs and a
fresh run directory, and performs its own bounded smoke gate. The following
is an execution recipe, **not an already executed command**:

```bash
bash scripts_order1_llama1B/IncLoRA-G2-FixedQ005-R020-TokV2_LLaMA3-1b.sh \
  --seeds 0 42 1993 --gpus 0,1,2,3 \
  --run-root experiment_queue/llama1b_order1_g2_tokv2_q005_r020
```

Use fresh v2 IncLoRA controls. The live collector leaves paired scores empty
until corrected baseline data with matching version and pinned provenance
is supplied. Legacy profiles are inspection-only in the live launcher;
their old frozen runners/results are untouched.

The optional sampled BF16 cast proxy is not proof of actual forward
visibility. `diagnose_g2_probe_precision.py` provides a separate same-point
BF16/full-FP32 response diagnostic without optimizer commits; its full-model
diagnosis remains to be run. See its `--help` and the precise scope guide.

Detailed code audit, old-result sensitivity analysis, and phased baseline/
three-seed validation plan:
[IMPLEMENTATION_AND_PLAN.md](../refine-logs/llama_g2_token_boundary_fix_20261012/IMPLEMENTATION_AND_PLAN.md).
CPU test/review evidence and literal twelve-wrapper receipts are in the same
directory. These checks do not establish a best radius or a new FAA result.
