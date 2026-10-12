# Three-point probes — warm environment and exact invocations

Repository: `/data/140-1/users/liying/O-LoRA`.
Warm Python: `/data/140-1/users/liying/conda_envs/nlora_req/bin/python`.
Reuse `nlora_req@0715917e`: canonical sorted compact JSON SHA256 of
`.aris/compute/nlora_req_local_spec.json` is
`0715917e1ad74c80ae4defaf61a143c33d13857323b67d09ead666a792fa61b2`.
Torch2.1.1+cu121, Transformers4.28.0, datasets1.17.0, repository-local PEFT.
No environment rebuild or new packages. Physical GPUs0/1/2/3 only.

Literal read-only, CPU/stdlib dry-run, with no output directory or training:

```bash
python3 scripts_llama/run_three_point_probe_llama.py --config configs/three_point_probe_llama/inclora_threepoint_central_llama1b_order1_tokv2_rho005.json --seeds 0 --run-root /tmp/three_point_probe_docs_20261012 --dry-run
```

Control root:
`/data/140-1/users/liying/O-LoRA/experiment_queue/llama_three_point_probe_hf_gpu0123_20261012T105940Z`.
Checkpoint/log/cache root:
`/data/115-2/users/liying/cache/tmp/llama_three_point_probe_hf_gpu0123_20261012T105940Z`.

Prepare the8-chain Order1/seed0 frozen manifest without allocating CUDA:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python scripts_llama/run_three_point_probe_queue.py --prepare --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_three_point_probe_hf_gpu0123_20261012T105940Z --artifact-root /data/115-2/users/liying/cache/tmp/llama_three_point_probe_hf_gpu0123_20261012T105940Z --config configs/three_point_probe_llama/inclora_threepoint_g0_llama1b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_g1_plus_llama1b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_backward_reflected_llama1b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_central_llama1b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_g0_llama3b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_g1_plus_llama3b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_backward_reflected_llama3b_order1_tokv2_rho005.json --config configs/three_point_probe_llama/inclora_threepoint_central_llama3b_order1_tokv2_rho005.json
```

Actual host controller invocation:

```bash
NCCL_P2P_DISABLE=1 /data/140-1/users/liying/conda_envs/nlora_req/bin/python /data/140-1/users/liying/O-LoRA/experiment_queue/llama_three_point_probe_hf_gpu0123_20261012T105940Z/source/scripts_llama/run_three_point_probe_queue.py --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_three_point_probe_hf_gpu0123_20261012T105940Z --wait-seconds 30
```

Launch as a durable detached host process, save stdout/stderr and record
PID/start ticks. The existing two-point queue retains its GPU leases until
it finishes; the new controller waits for all4 idle cards and shared locks.
It repeats seeded CUDA/BF16 witnesses after admission, then each method's
native two-stage smoke and first-window pairing gate before formal training.
Waiting, CUDA witness success and model smoke success are distinct states;
none is a formal accuracy result.

Progress: control-root `runtime/queue_state.json`. Accepted results:
`results/RESULTS.md`, `results/orders.csv`, `results/stages.csv`. Bulk logs
are under artifact-root `logs/`. `runtime/PAUSE` and
`runtime/STOP_AFTER_STAGE` affect future admission, leaving the active
stage to finish.

All four configs use `optimizer_type=threepoint-adamw_hf`,
`probe_rho=0.05`, `probe_eps=1e-12`; mode values are exactly
`g0`, `g1_plus`, `backward_reflected`, `central`. The three-read trace and
audit use distinct filenames and schemas from the old two-point cohort.
