# Pure IncLoRA-G2 startup recovery

Work directory: `/data/140-1/users/liying/O-LoRA`. Reuse warm `/data/140-1/users/liying/conda_envs/nlora_req/bin/python` and canonical environment `nlora_req@0715917e`. No package installs, optimizer changes, algorithm changes, task-order changes, or seed changes.

The first campaign stopped on the first model smoke at2026-10-11T09:07:44Z. All four seeded CUDA/BF16 witnesses passed. `compute_metrics.py` eagerly loads `../data/gpt2tokenizer` relative to frozen source; its six local files were missing from the snapshot. Failure occurred before model loading or a submitted G2 update. Original failed control/source/logs remain preserved at `experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z`.

The corrected shared launcher copies the local tokenizer into `source/data/gpt2tokenizer`, includes it in frozen source hashes, and rejects missing assets before GPU admission. The queue also pins the original six asset files in its input manifest. G2 core/trainer/entrypoint are unchanged.

First literal read-only invocation, which must leave its run root absent and allocate no CUDA:

```bash
python3 scripts_llama/run_g2_llama.py --config configs/g2_llama/inclora_g2_llama1b_order1.json --seed 0 --run-root /tmp/llama_g2_recovery_20261011T192153Z --dry-run
```

Meaningful CPU regression: real frozen training entrypoint import, local metric tokenizer tokenization, denied network connections, and uninitialized CUDA, plus missing-asset rejection before GPU inspection:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python -m unittest scripts_llama.test_run_g2_llama_cpu scripts_llama.test_g2_orders_queue_cpu -v
```

Prepare the fresh six-chain seed0 attempt without allocating CUDA:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python scripts_llama/run_g2_orders_queue.py --prepare --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2 --artifact-root /data/115-2/users/liying/cache/tmp/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2 --config configs/g2_llama/inclora_g2_llama1b_order1.json --config configs/g2_llama/inclora_g2_llama1b_order2.json --config configs/g2_llama/inclora_g2_llama1b_order3.json --config configs/g2_llama/inclora_g2_llama3b_order1.json --config configs/g2_llama/inclora_g2_llama3b_order2.json --config configs/g2_llama/inclora_g2_llama3b_order3.json
```

Actual frozen entrypoint CPU-only startup check before GPU launch:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NLTK_DATA=/data/140-1/users/liying/O-LoRA/.cache/nltk_data PYTHONPATH=/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/source/src /data/140-1/users/liying/conda_envs/nlora_req/bin/python /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/source/src/run_uie_lora.py --help
```

The first corrected attempt was preserved after an initial NCCL collective hung before model loading. A bounded four-rank comparison reproduced the default20-second collective timeout; with `NCCL_P2P_DISABLE=1`, all four ranks passed initial barrier, exact rank-coded all-reduce and final barrier in6.48seconds. See `NCCL_DIAGNOSTICS_VALIDATED.json`; concurrent stdout marker parsing was reconciled against four complete JSON rank payloads. This is a recorded transport remedy, not a historical protocol claim or a change to G2/AdamW/batch/LR.

Durable host controller's literal command (redirect stdout/stderr and record PID/process start time):

```bash
NCCL_P2P_DISABLE=1 /data/140-1/users/liying/conda_envs/nlora_req/bin/python /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/source/scripts_llama/run_g2_orders_queue.py --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2 --wait-seconds 30
```

Only physical GPU0/1/2/3, shared index/UUID leases, idle admission, no preemption. Each fresh reservation repeats the seeded per-card FP32 CUDA and BF16 finite forward/backward witnesses. Every model/Order first passes its own two-task bounded smoke before formal training. Six Orders run serially using DDP4. Seed0; actual Transformers AdamW, epsilon1e-6, decay0; rank8/alpha32/dropout.1; effective batch64; historical task orders and514 formal updates per Order.

The full unchanged scientific protocol and six executable wrapper locations are in [the original experiment plan](../llama_pure_g2_hf_allorders_plan_20261011T083417Z/EXPERIMENT_PLAN.md). Recovery's current control root is the fresh retry path above. Live state is its `runtime/queue_state.json`, validated results are `results/RESULTS.md` and `results/orders.csv`. No final FAA/AAA/BWT is emitted for incomplete Orders.
