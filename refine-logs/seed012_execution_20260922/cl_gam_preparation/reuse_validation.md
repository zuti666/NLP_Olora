# Formal seed0 reuse validation

PASS_WITH_DOCUMENTED_LIMITATIONS

Accepted 6 / 6 records. Verified all 172 frozen source files against original and prepared copies; verified recorded runtime hashes. Ordered training/test paths, labels, and task boundaries match both new baseline and EWC preflight manifests for all five datasets.

| Method | Dataset | FAA | AAA stage | Receipt/hash, config/hash, metric/hash, log/hash |
|---|---|---:|---:|---|
| CL-LoRA | aircraft | 29.284000 | 26.528942 | PASS |
| CL-LoRA | cars196 | 34.508000 | 42.169252 | PASS |
| CL-LoRA | cub200 | 78.291000 | 80.878104 | PASS |
| CL-LoRA | flowers | 97.822000 | 98.334387 | PASS |
| CL-LoRA | het5 | 66.984000 | 61.986800 | PASS |
| GAM-EWCLoRA | het5 | 83.370000 | 81.155667 | PASS |

CL-LoRA weight path and SHA match the explicit historical config binding: `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`.

- Historical data manifests contain paths and labels rather than historical image byte digests.
- GAM-Het5 historical pretrained revision and bytes were not bound in its archived receipt/config.
- Reuse verifies evidence and row-local protocol compatibility; it is not a claim of bitwise retraining reproducibility.

Full machine-readable paths, digests, checks, recomputed metrics, and seed1/2 configuration differences: `reuse_validation.json`. No training launched.
