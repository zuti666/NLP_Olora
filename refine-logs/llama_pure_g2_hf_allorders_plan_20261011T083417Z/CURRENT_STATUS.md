Updated: 2026-10-11T10:49:35+02:00 (Europe/Paris).

Six seed0 pure IncLoRA-G2 chains are prepared and owned by the durable controller PID 1216462 on cudahpc40.cvc.uab.es.

Current status: **waiting_gpus**; **0/24 formal stages**, **0/12 smoke stages** accepted. Physical GPUs0–3 are occupied by other users, so no G2 CUDA witness, model smoke or formal training has started. All four cards must become idle before automatic admission. No other processes or queues were changed.

Settings: Transformers AdamW, epsilon1e-6, weight decay0, rank8/alpha32/dropout0.1, DDP4, global batch64, seed0. CPU launcher/queue/collector tests34/34 and independent documentation/source review passed. Previously validated pure-G2 core/trainer/entrypoint unchanged. Source106 files and local model/data/baseline inputs are frozen.

The first unexecuted preparation is preserved under `_prelaunch_v1`; a prediction-ID integrity correction was accepted before the current source snapshot and any GPU launch.

See [experiment plan](EXPERIMENT_PLAN.md), [execution environment](ENVIRONMENT.md), [launch evidence](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z/LAUNCH_VERIFIED.json), [live state](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z/runtime/queue_state.json), and [accepted results](/data/140-1/users/liying/O-LoRA/experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T083417Z/results/RESULTS.md). Final FAA/AAA/BWT remain blank until a complete four-stage Order passes.
