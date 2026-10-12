# Radius 25–50 measurement worker

This adapter invokes the unchanged formal `FullRunner`. The source hashes,
formal coverage receipt, anchor/model/data/head/coordinate identity, public
direction identities, and repeated origin are checked. Only the isolated output
directory and the job's radius grid differ from the formal seed config.

| Job | New radius grid (origin repeated) | Public directions | Evaluation | Fisher | Expected states per seed/method |
| --- | --- | --- | --- | --- | --- |
| `dense` | 0, 25, 30, 35, 40, 45, 50 | indices 0–99 | all 1259 training and 307 test examples | none | 601 |
| `sparse_train` | 0, 25, 30, 35, 40, 45, 50 | indices 0–15 from the same library | all 1259 training and 307 test examples | fixed 256 training examples | 97 |

There are 3 training seeds × 2 methods × 2 jobs = 12 jobs. Methods retain their
existing mapping: `sgd` is EWCLoRA; `gam` is FS-EWCLoRA. This formal cohort uses
ImageNet-A and must not be relabeled as the historical ImageNet-R cohort.

The direction generator, direction seed, sample subset, checkpoints, and all
measurement code stay unchanged. Fisher is the per-example true-label CE
gradient-square mean in the 24 effective Q/V matrices, with fixed old 20-class
head. There is no trace normalization, regularizer multiplier, or Fisher floor.
Test evaluation is retained in raw results for provenance even where the figure
only displays training metrics.

Example CLI (the queue should set `CUDA_VISIBLE_DEVICES`):

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u diagnostic_radius50.py --config configs/dense_seed0.json --method sgd --job dense --device cuda:0
```

Replace the seed, method, and job consistently. Each job has its own unchanged
formal config plus the two authorized edits:
`configs/{dense,sparse_train}_seed{0,1,2}.json`.
Use `--validate-only` for CPU-only source/config/reference validation. Run
`python check_worker.py` to validate all 12 CLI combinations, replay the numeric
validator on completed formal records, and check that injected identity,
normalization, nonfinite, incomplete-state, and summary faults are rejected.

Results go to `results/seedS/M/J/`. A job is successful only when
`extension_receipt.json` exists with `status=complete`, the hash-bound summary,
manifest, and states still match, and all identity/origin/result checks pass.
The adapter independently recomputes damage/accuracy/Fisher aggregates from all
raw rows and checks exact public directions, finite numbers, real drift,
sample counts, and final anchor restoration before writing the receipt.

The unchanged runner also emits tolerance estimates from its extension-only
grid. These must **not** be used as the combined experiment's tolerance values:
the publisher must merge the original dense 0–20, step-0.25 data with the new
25–50 measurements, then apply the prefix-safe threshold rule per seed before
averaging across seeds. A boundary between 20 and 25 is only bracketed at this
resolution; interpolation is not an additional measured radius.

The frozen helper `stage_path_completion_20260921/diagnostic_extension.py` is
verified against its SHA-256 before import. Neither that helper nor the original
diagnostic code is modified by this extension.
