# Four response methods: full ImageNet-A validation on GPU 5/6/7

Reuse `/home-local/liying/.conda/envs/Pilot_new_local/bin/python` without installing packages. Actual host: `cudahpc40.cvc.uab.es`; host PID namespace: `pid:[4026531836]`. All physical GPU admission is restricted to 5, 6, 7. This host currently exposes only indices 0–6: 7 is absent and skipped, so 5 and 6 provide two parallel slots. Cards 0–4 are never admitted. Only idle cards enter under shared physical-index and UUID leases; each worker must pass the seeded CUDA witness before training. Temporary path: `/tmp/olora_r567_20261009T103449Z`.

The trainer is byte-identical to the accepted implementation campaign: 34 CPU checks, eight real GPU implementation smokes and one pairing gate. `gates/PREVIOUS_IMPLEMENTATION_ACCEPTED.json` revalidates the old receipts, traces, checkpoint tensors, frozen code and source; it does not claim a new formal result.

## Read-only configuration check

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/trainer/response_train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/configs/formal_single200_history_g2_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/outputs/formal_single200_history_g2_s2 --dry-run
```

Expected: history_g2, single200, 200-class head, one task, 50 epochs, seed2/rank10; dynamic SGD prediction radius, full previous final gradient cache; normal three gradient reads and one actual SGD per batch. Old GAM fixed radii are inactive. This command does not train.

## Read-only queue check in the actual host namespace

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/run_queue.py --dry-run
```

Expected JSON: status valid, jobs7, formal7, smokes0, allowed_gpus [5,6,7], max_parallel2. Frozen source190 and execution pins must pass. Execute through the approved actual-host route, since the sandbox PID namespace has no NVML.

## Execution and progress

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/run_queue.py
```

The root agent launches this command once detached. Before any later manual invocation, inspect runtime/LAUNCH.json, runtime/controller.json and runtime/queue_state.json to avoid duplicate launches. Controller.lock also prevents a second controller.

Four single200 jobs run first (SGD/current_g2/history_fusion/history_g2). All must complete the full 50 epochs and pass trace/checkpoint/metric acceptance before the three CL10 jobs start (SGD/current_g2/history_g2, 10 tasks ×50 epochs). CL10 history_fusion is imported only from its pinned, accepted existing HRF run; it is not retrained. No smoke budget flags appear in formal commands. Failure stops further admission and lets active work finish; runtime/PAUSE stops new admission without interrupting active training.

/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-four-methods-gpu567-20261009T103449Z/collect_results.py updates RESULTS.md, RESULTS.csv and RESULTS.json. Completed results require zero-exit receipts and full-budget acceptance. Pending/running metrics are blank. Single200 reports final test accuracy; BWT is undefined. CL10 reports task-equal FAA/AAA/BWT. Per-epoch single200 test curves are observational; epoch50 is fixed and no best-epoch selection occurs. No average across these two protocols is reported.
