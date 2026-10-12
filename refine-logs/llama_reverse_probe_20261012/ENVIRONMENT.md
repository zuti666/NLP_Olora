# Reflected probe — warm local environment and literal commands

Repository: `/data/140-1/users/liying/O-LoRA` on cudahpc40.cvc.uab.es.
Reuse `/data/140-1/users/liying/conda_envs/nlora_req/bin/python`, environment
nlora_req@0715917e, unchanged canonical spec
`.aris/compute/nlora_req_local_spec.json`, canonical SHA256
0715917e1ad74c80ae4defaf61a143c33d13857323b67d09ead666a792fa61b2.
Torch2.1.1+cu121, Transformers4.28.0, datasets1.17.0, repository-local PEFT.
No packages are installed. Only physical GPU0/1/2/3 are authorized here.
NCCL_P2P_DISABLE=1 is set in each new worker environment, carrying forward
the tested transport remedy from the completed G2 campaign.

Fresh literal dry-run, no CUDA query/import, output tree or training:

```bash
python3 scripts_llama/run_two_point_probe_llama.py --config configs/two_point_probe_llama/inclora_probe_reverse_llama1b_order1_tokv2_rho005.json --seeds 0 --run-root /tmp/reflected_probe_docs_20261012 --dry-run
```

Pilot control root:
`/data/140-1/users/liying/O-LoRA/experiment_queue/llama_reflected_probe_hf_gpu0123_20261012T103400Z`.
Checkpoint/log/cache root:
`/data/115-2/users/liying/cache/tmp/llama_reflected_probe_hf_gpu0123_20261012T103400Z`.
These are a new six-chain Order1/seed0 attempt, unrelated to old queues.

Prepare a frozen manifest without allocating CUDA:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python scripts_llama/run_two_point_probe_queue.py --prepare --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_reflected_probe_hf_gpu0123_20261012T103400Z --artifact-root /data/115-2/users/liying/cache/tmp/llama_reflected_probe_hf_gpu0123_20261012T103400Z --config configs/two_point_probe_llama/inclora_probe_sgd_llama1b_order1_tokv2_rho005.json --config configs/two_point_probe_llama/inclora_probe_sam_llama1b_order1_tokv2_rho005.json --config configs/two_point_probe_llama/inclora_probe_reverse_llama1b_order1_tokv2_rho005.json --config configs/two_point_probe_llama/inclora_probe_sgd_llama3b_order1_tokv2_rho005.json --config configs/two_point_probe_llama/inclora_probe_sam_llama3b_order1_tokv2_rho005.json --config configs/two_point_probe_llama/inclora_probe_reverse_llama3b_order1_tokv2_rho005.json
```

The controller invocation, from the actual host, is:

```bash
NCCL_P2P_DISABLE=1 /data/140-1/users/liying/conda_envs/nlora_req/bin/python /data/140-1/users/liying/O-LoRA/experiment_queue/llama_reflected_probe_hf_gpu0123_20261012T103400Z/source/scripts_llama/run_two_point_probe_queue.py --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_reflected_probe_hf_gpu0123_20261012T103400Z --wait-seconds 30
```

It must run as a durable detached subprocess, with stdout/stderr logged and
its host PID/start time recorded. Pending work waits for all four idle cards,
without preemption. Native model smoke has not passed merely because the
controller started or a CUDA witness passed.

Progress: control-root `runtime/queue_state.json`. Accepted results:
`results/RESULTS.md`, `results/orders.csv`, `results/stages.csv`. Logs are in
the artifact root. PAUSE and STOP_AFTER_STAGE sentinel files in runtime/
pause future admission or stop after the active stage, respectively.
