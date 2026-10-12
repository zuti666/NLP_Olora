# Seed-0 integer-radius supplement, 25–50

This isolated adapter uses the unchanged formal `FullRunner`. The latest user
request keeps only training seed 0, halves the dense direction count to 50,
and keeps the Fisher direction count at 16. Existing results remain unchanged.

The missing radii are **26, 27, 28, 29, 31, 32, 33, 34, 36, 37, 38, 39, 41, 42,
43, 44, 46, 47, 48, 49**. Radius 0 is repeated to verify the anchor. The publisher
reuses completed radii 25, 30, 35, 40, 45 and 50; the merged grid covers every
integer from 25 through 50. This supplement does not measure radii 21–24.

| Job | Public direction indices at each nonzero radius | Evaluation | Fisher | States per method |
| --- | --- | --- | --- | --- |
| `dense` | 0–49 | all 1259 training and 307 test samples | none | 1001 |
| `sparse_train` | 0–15 | all 1259 training and 307 test samples | fixed 256 training samples | 321 |

There are 1 seed × 2 methods × 2 jobs = **4 jobs**, totaling **2644 states**
including 4 repeated origins. `sgd` is EWCLoRA; `gam` is FS-EWCLoRA. This is the
formal **ImageNet-A** cohort, not the historical ImageNet-R cohort.

All checkpoint, sample subset, data order, model, 20-class old-task head,
effective-weight coordinates, direction seed and direction generator settings
remain identical to the formal experiment. Dense measurements select the first
50 directions from the original 100; both methods and all radii share that
prefix. Fisher retains all 16 original directions. Fisher is the unnormalized
mean of per-example true-label CE squared gradients across 24 effective Q/V
matrices, without an EWC lambda multiplier, trace normalization or Fisher floor.

Config files are `configs/dense_seed0.json` and
`configs/sparse_train_seed0.json`. Both change only the isolated `output_dir`
and the active radius grid; the dense config additionally changes
`dense_directions` from 100 to 50. The sparse config preserves 16 directions.
Example CLI, with GPU assignment supplied by the queue:

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u diagnostic_radius50.py --config configs/dense_seed0.json --method sgd --job dense --device cuda:0
```

Use `--validate-only` for CPU-only preflight. `check_worker.py` checks all four
CLI configurations, replays validation on the four seed-0 formal jobs and four
completed coarse-extension jobs, and validates the production dense direction
count against the first 50 directions of actual completed raw rows. Those
four derived summaries exist only in memory for checks; they are never saved
as experimental measurements. Six controlled faults test rejection of NaNs,
changed directions, normalized Fisher, altered summaries, incomplete state
counts and changed dense direction count. Old source hashes are checked before
and after verification. Evidence is saved in `worker_cpu_check.json`.

Results go to `results/seed0/M/J/`. Completion requires
`extension_receipt.json` with schema `rq2_radius50_integer_seed0_halfdirs_v1`,
matching summary/manifest/states hashes, and successful checks of the anchor,
model, data, coordinates, directions, finite values, numerical summaries and
final restoration. A partially written states file does not establish success.

The unchanged runner also writes tolerance estimates from its extension-only
grid. Do not interpret those as combined tolerance estimates. The publisher
must derive old dense statistics anew from the matching first 50 directions,
merge all original and new radii, then apply the prefix-safe threshold rule to
seed 0. Old 100-direction averages and three-seed means must not be mixed with
this seed-0 50-direction result. Fisher uses the original 16 directions
throughout. Unmeasured radii 21–24 remain a gap, so a threshold crossing between
20 and 25 stays bracketed.

The SHA-256-pinned helper
`../stage_path_completion_20260921/diagnostic_extension.py` and original formal
measurement code are unchanged. No GPU launch is performed by these CPU checks.
