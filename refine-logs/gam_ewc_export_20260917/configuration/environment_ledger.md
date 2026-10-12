# Local CUDA environment ledger

### env: nlora_req@0715917e

- how: direct Python environment at `/data/140-1/users/liying/conda_envs/nlora_req`
- tier: `{cpus: shared, mem_gib: shared, gpus: 1x NVIDIA 48GB per queued order}`
- weights:
  - `/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B` (4.7G)
  - `/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-3B` (12G)
- validated: 2026-08-25 Tier 1 import/CLI/model-tokenizer witnesses passed; the exact seeded CUDA witness is embedded in `scripts_llama/wait_and_run_gamfs_llama.sh` and must pass after a GPU is claimed, before either smoke starts.
- agent-follows-doc: 2026-08-25 fresh-agent dry-run passed verbatim for both 1B and 3B; GPU execution remains gated on the exact CUDA witness and two-stage smoke checks.
- gotcha: invoke `/data/140-1/users/liying/conda_envs/nlora_req/bin/python` directly because legacy `torchrun` shebangs and `/data/140-0` cache/model paths are stale. Use repository-local `src/peft`; do not install external PEFT.

### 2026-09-16 IncLoRA+EWC supplement on this host

- environment spec hash is still `0715917e`; Python/PyTorch/Transformers/Datasets imports match the existing environment. No package rebuild.
- hardware observed: 10x Quadro RTX 6000 24 GiB; GPU9 was idle and claimed by UUID `GPU-b1be0dea-597f-5d31-a637-124e1d8d0ab2`. This hardware reports BF16 unsupported; supplement uses FP32.
- seeded CUDA matmul/backward witness passed on GPU9. The detached runner repeats the spec's seed0 matmul witness and records `preflight.log`.
- gotcha: this host lacked NLTK punkt. Downloaded the official NLTK punkt data to repository `.cache/nltk_data`, using `SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt` to retain TLS verification. Runner sets `NLTK_DATA` explicitly; the actual training CLI `--help` passed in offline mode.
- agent-follows-doc: fresh reviewer executed the exact dry-run in `scripts_llama/INCLORA_EWC_SUPPLEMENT.md`; 10 commands, config paths, and NLTK lookup passed. Real two-task smoke is recorded separately in the run directory.
- invocation: `python3 scripts_llama/run_inclora_ewc_supplement.py --run-dir experiment_queue/inclora_ewc_supplement_20260916T213647Z --gpu auto --orders 2 3 --precision fp32` (already launched; do not execute again with this run directory).
