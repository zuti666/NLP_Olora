# Five additional pure-GAM runs

The suite contains exactly five new seed-2 jobs: mean(g1,g3) at outer radius 1,
and g3-only at outer radii 0, 0.1, 0.25 and 0.5. The completed g3-only radius-1
run is reused. The preceding original, sum and g1 runs are also included as
read-only reference rows in the combined report, without new training.

The mean arm changes beta1=beta3=0.5; lambda_flat remains 1. All new runs have
gamma=0. The g3 radius sweep changes only gam_grad_norm_rho relative to the old
g3 configuration. Inner rho=0.05, rank=10, seed=2, epochs, learning rates, data,
environment and all other scientific settings remain unchanged. CONFIG_DIFFS.json
records the exact differences and reference hashes.

Training uses the unchanged 20260921 frozen source. Gamma-zero decomposition
arithmetic, the identity direction callback and all four GAM probes are retained.
Fisher estimation/EWC, extra delta-W penalty and weight fusion remain disabled.
No strict no-decomposition implementation is introduced by this campaign.

The four runtime scripts are byte-identical to the previous completed queue.
The GPU pool is physical indices 0..6, with at most five training jobs in parallel.
The existing /tmp/olora_gamfs_gpu_locks/gpuN.lock flock is held by the training
child via an inherited descriptor. A GPU must have memory usage below 500 MiB
and utilization below 5%. Diagnostic workers must share that lock protocol.

All suite/config output paths point to the final campaign/training directory.
Staging is preparation only. Before deployment, run:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -u preflight_cpu.py
```

After root deploys the directory, validate on the execution host:

```sh
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python -B -u /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/launch_queue.py --validate-only
```

Root owns the formal launch. This preparation agent has not started any queue.
The final-report entry in suite.json points to summarize_g1g3.py. There is no
four-arm-only summary hook. The summary validates reused receipts/config/source
and metrics hashes and reuses the established complete-matrix validator. It
reports task-macro AAA, FAA, mean diagonal acquisition, BWT, forgetting and
sample-weighted AAA/FAA. Forward-transfer native NaNs remain structural NAs.

`summarize_g1g3.py --allow-incomplete` can refresh a partial report after deployment;
without that option it returns nonzero until all five new jobs are complete.
G1G3_COMPARISON.json/.csv/.md are its outputs. Reference paths are deliberately
left pointing to their original completed campaign; they are never copied into
the new queue as newly completed jobs.
