# Integer-radius supplement, 25–50

This isolated adapter reuses the unchanged formal `FullRunner` and the validated
coarse-extension worker. Only the output directory and the active measurement
grid differ from each formal seed configuration. Existing results are preserved.

The missing radii are **26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42,
43, 44, 46, 47, 48, 49**. Radius 0 is repeated for anchor verification. Completed
radii 25, 30, 35, 40, 45 and 50 are reused by the publisher, yielding every
integer from 25 through 50. This supplement does not measure radii 21–24.

| Job | Public direction indices at each nonzero radius | Evaluation | Fisher | States per seed/method |
| --- | --- | --- | --- | --- |
| `dense` | 0–99 | all 1259 training and 307 test samples | none | 2001 |
| `sparse_train` | 0–15 from the same library | all 1259 training and 307 test samples | fixed 256 training samples | 321 |

There are 3 training seeds × 2 methods × 2 jobs = 12 jobs, with 13932 measured
states including 12 repeated origins. The 13920 nonzero states add new
measurements. `sgd` denotes EWCLoRA and `gam` denotes FS-EWCLoRA. This is the
formal **ImageNet-A** cohort, not the historical ImageNet-R cohort.

Model checkpoints, the fixed old 20-class head, data order and sample subset,
direction generator and seed, and formal code remain unchanged. Both methods,
all training seeds and all radii use identical direction identities. Fisher is
the unnormalized mean of per-example true-label CE squared gradients in the 24
effective Q/V matrices, without EWC lambda, a floor, or trace normalization.
Raw results retain test evaluation even when the figure only displays training
metrics.

Example CLI, with GPU selection supplied by the queue:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u diagnostic_radius50.py --config configs/dense_seed0.json --method sgd --job dense --device cuda:0
```

Configurations are `configs/{dense,sparse_train}_seed{0,1,2}.json`. Their only
changes from `../configs/seedS.json` are `output_dir` and the active radius grid.
Use `--validate-only` for a CPU-only preflight. `check_worker.py` passed all 12
CLI combinations, replayed the numeric validator on 12 completed formal and 12
completed coarse-extension jobs, and rejected six controlled faults: nonfinite
metrics, changed direction identity, normalized Fisher, altered summary,
incomplete states and changed direction count. Frozen original source hashes
were checked before and after verification. See `worker_cpu_check.json`.

Results go to `results/seedS/M/J/`. Success requires a complete
`extension_receipt.json` with schema `rq2_radius50_integer_extension_v1`, matching
summary/manifest/states hashes, and successful model/data/coordinate/direction,
origin, finite-value, numerical-summary and final-restoration checks. A
partially written states file is not evidence that the job has completed.

The unchanged runner's extension-only tolerance values are not the combined
experiment's tolerance estimates. The publisher must merge the original 0–20
grid, the completed coarse extension and this supplement, apply the prefix-safe
threshold rule separately for each seed, and only then average across seeds.
There remain unmeasured intervals 20–25. A threshold crossing there remains
bracketed; this supplement cannot turn it into an exact measured boundary.

The frozen validation helper
`../stage_path_completion_20260921/diagnostic_extension.py` is SHA-256 checked
before import. No original diagnostic or completed-result files are changed.
