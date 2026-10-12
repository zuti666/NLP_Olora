# Pure-G2 all-Order execution environment

Repository and command working directory: `/data/140-1/users/liying/O-LoRA`.
Use the existing `/data/140-1/users/liying/conda_envs/nlora_req/bin/python` without installing or changing packages. Canonical environment spec: `.aris/compute/nlora_req_local_spec.json`, canonical JSON SHA256 `0715917e1ad74c80ae4defaf61a143c33d13857323b67d09ead666a792fa61b2`, computed from `json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()`. The formatted file's separate raw SHA256 is `e45ab90933d3e0968a8743569b207fbf3de0fd202178c0b07dafbed644099301`. Runtime is torch2.1.1+cu121, transformers4.28.0, datasets1.17.0, accelerate1.4.0 and repository-local PEFT0.3.0.

The following literal invocation is read-only, performs no CUDA allocation, and must leave `/tmp/llama_g2_allorders_documented_dry_run` absent:

```bash
python3 scripts_llama/run_g2_llama.py --config configs/g2_llama/inclora_g2_llama1b_order1.json --seed 0 --run-root /tmp/llama_g2_allorders_documented_dry_run --dry-run
```

All six wrappers support `--seed 0 --gpus 0,1,2,3 --run-root <fresh-directory>` and `--dry-run`. The campaign controller uses their same configuration and stage construction APIs, with a single immutable source snapshot and GPU reservation across the six chains.

CPU suite for the newly generalized launcher and queue/collector:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python -m unittest scripts_llama.test_run_g2_llama_cpu scripts_llama.test_g2_orders_queue_cpu -v
```

Prepare exactly six seed0 chains. This hashes local inputs and freezes source without allocating CUDA. Both output roots must be fresh:

```bash
python3 scripts_llama/run_g2_orders_queue.py --prepare --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z --artifact-root /data/115-2/users/liying/cache/tmp/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z --config configs/g2_llama/inclora_g2_llama1b_order1.json --config configs/g2_llama/inclora_g2_llama1b_order2.json --config configs/g2_llama/inclora_g2_llama1b_order3.json --config configs/g2_llama/inclora_g2_llama3b_order1.json --config configs/g2_llama/inclora_g2_llama3b_order2.json --config configs/g2_llama/inclora_g2_llama3b_order3.json
```

Launch the frozen controller in a durable host process, with stdout/stderr redirected to the control root `runtime/controller.log` and PID/start-time evidence recorded. Its literal foreground command is:

```bash
/data/140-1/users/liying/conda_envs/nlora_req/bin/python /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z/source/scripts_llama/run_g2_orders_queue.py --queue-root /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z --wait-seconds 30
```

The controller admits only physical GPU0/1/2/3. It checks idle memory below500MiB, utilization at most5%, absence of compute PIDs, and shared index/UUID leases. It binds children by GPU UUID. On each card it runs the canonical seeded8×8 FP32 matmul sentinel followed by finite BF16 forward/backward, saving actual logs before model smoke. No other cards or queues are modified.

Read progress and regenerate accepted result tables without starting training:

```bash
cat /data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z/runtime/queue_state.json
python3 - <<'PY'
from pathlib import Path
import runpy
root = Path('/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z')
queue = runpy.run_path(str(root / 'source/scripts_llama/run_g2_orders_queue.py'))
queue['verify_manifest'](root)
collector = runpy.run_path(str(root / 'source/scripts_llama/collect_g2_orders_results.py'))
collector['main'](['--queue-root', str(root)])
PY
```

Creating this queue's `runtime/PAUSE` file pauses after the current stage, releases its own GPU leases, and keeps a durable waiting controller. Removing that file resumes idle admission and fresh CUDA witnesses. `runtime/STOP_AFTER_STAGE` stops after the current stage. These are manual controls, not scheduled actions. Failed/stopped runs require inspection and a new preserved attempt rather than automatic reuse.
