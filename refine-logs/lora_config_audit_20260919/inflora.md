# InfLoRA configuration audit (2026-09-19)

Scope: the six InfLoRA cells in `cllora_comparison_20260918/results.json`, their raw log last-run blocks, existing source YAMLs, and the **currently readable Flatness_CV implementation**. No experiment was run or modified. This report does not assert that today's source is the immutable source used in March 2026. Explicit log values are historical evidence; unlogged defaults and implementation behavior below are current-source deductions.

## 1. Conclusion

- The five single-dataset InfLoRA runs match Vanilla LoRA on the recorded common settings inspected: seed, backbone family, class order/task split, batch size, initial/subsequent learning rate, epoch count, optimizer, momentum, weight decay, scheduler name, and replay-memory fields. This is not a guarantee of identical historical source, checkpoint bytes, transforms, data files, or RNG streams.
- **5Datasets has a real training-budget mismatch: InfLoRA 40/40 epochs versus Vanilla 30/30 epochs.** Other recorded common settings match.
- Across all six InfLoRA runs, rank is 16 and the method construction is the same in current source: 12 ViT blocks, Q/V adapters, information-initialized frozen A, trainable current B, frozen earlier task adapters, one expanding linear classifier, no learnable adapter scaling. LR and number/class sizes of tasks vary by dataset. `total_sessions` is consumed by DualGPM, so the numerical threshold schedule also differs.
- `lamb=0.7`, `lame=0.99`, `min_lr=0.1*stage_lr`, and loader workers 8 are **not recorded** in the six selected log parameter blocks or YAMLs. They are the defaults in the current implementation, with no injection found in the current entry/config/trainer path. They must not be presented as independently proven historical runtime values.

## 2. Per-dataset recorded settings and derived schedule

All six: seed 0; ViT-B/16 at 224; rank 16; batch 128; SGD; initial/subsequent momentum 0 and weight decay 0; cosine scheduler; no rehearsal buffer (`memory_size=memory_per_class=0`, `fixed_memory=false`); `class_shuffle=false`. Both initial and later task LR are the same within each dataset.

| Dataset | Classes per task | T / total_sessions | InfLoRA epochs init/later | Vanilla epochs init/later | LR init/later | Current inferred cosine eta_min | Current DualGPM thresholds, first → last |
|---|---|---:|---:|---:|---:|---:|---:|
| Aircraft | 10×10 | 10 | 40/40 | 40/40 | 0.05 | 0.005 | 0.7 → 0.961 |
| Cars196 | 16 + 9×20 | 10 | 40/40 | 40/40 | 0.05 | 0.005 | 0.7 → 0.961 |
| CUB-200 | 10×20 | 10 | 40/40 | 40/40 | 0.01 | 0.001 | 0.7 → 0.961 |
| Flowers102 | 12 + 9×10 | 10 | 40/40 | 40/40 | 0.0025 | 0.00025 | 0.7 → 0.961 |
| Pets37 | 5 + 8×4 | 9 | 40/40 | 40/40 | 0.0025 | 0.00025 | 0.7 → 0.957777778 |
| 5Datasets | 200,100,196,102,37 | 5 | 40/40 | 30/30 | 0.02 | 0.002 | 0.7 → 0.932 |

`T` above agrees with selected metrics task counts. The 5Datasets `increment: 1` is not a one-class task protocol: DataManager consumes explicit dataset task splits [200,100,196,102,37]. Filenames containing `t20c10` are not reliable task-count evidence.

## 3. Which parameters are actually consumed

| Setting / mechanism | Effective value in current source | Cross-dataset behavior / evidence |
|---|---|---|
| LoRA rank | 16, read from `lora_rank`; fallback 10 unused | Same six; `inflora.py:232–234` |
| Attached blocks/channels | All 12 ViT-B blocks, Q and V only; A=[16,768], B=[768,16], no biases | Same six; `_build_inflora_backbone` omits `lora_layer`, `lora.py:478–481,568–587` |
| Scaling / dropout | `learn_alpha=False`; IdentityScale applies multiplier 1; no LoRA dropout or alpha/r scaling in this wrapper | Same six; `inflora.py:252`, `lora.py:423–428,537–541,211–226` |
| A initialization | Current-task second-moment matrix collected before task training, projected against prior DualGPM basis if task>0; top-16 left singular directions, copied transposed with scale 1/sqrt(3); Q and V use identical A basis for each block | Data-dependent, not a fixed random A; `inflora.py:342–365`, `lora.py:838–853` |
| B initialization / training | Zero B; only current task B becomes trainable; A, original ViT and previous adapters frozen | Same rule six; `lora.py:657–663,190–201`; `inflora.py:333–377` |
| Prior adapters | One new rank-16 Q/V adapter per task; saved earlier A/B are loaded and summed with current contributions | Task count changes accumulated capacity; `lora.py:506–529,180–226`; not one fixed rank-16 lifetime adapter |
| `lamb`, `lame` | `args.get('lamb',0.7)`, `args.get('lame',0.99)` | Both absent from all six YAML/log parameter blocks; current defaults, `inflora.py:217–218` |
| `total_sessions` | 10/10/10/10/9/5; read explicitly | Actually consumed; fallback `int(200/increment)` never used for these six; `inflora.py:147–149,826` |
| DualGPM threshold | 0.7 + (0.99−0.7) × t/T, t=0,…,T−1 | T=10: [.700,.729,.758,.787,.816,.845,.874,.903,.932,.961]; Pets T=9 step .032222222; Het5 T=5 [.700,.758,.816,.874,.932]. Does not reach .99 |
| Basis dimension / policy | Data-dependent SVD energy selection; `remove` or `retain` subspace per block | No fixed `topk`/retained rank parameter from YAML; current basis energy uses squared singular values of the second-moment matrix; `inflora.py:828–907` |
| Projection location | Projection acts on current-task moment matrix before A initialization; subsequent training optimizes B via CE | No separate orthogonality/distillation/EWC penalty in SGD loss; not a YAML regularizer setting; `inflora.py:350–365,574–585,739–753` |
| Covariance data/sampling | All current-task training examples, every token including CLS, all 12 qkv inputs; XᵀX / number_of_tokens per block | Two full passes/task: before training and after training; same shuffled training loader/augmentations; no batch/sample cap; `inflora.py:342,411–429,777–822` |
| KMeans keys | Full current training set features, L2 normalized, k=5, random_state=0 | Called unconditionally per task despite “optional” comment; `all_keys` is only initialized/appended in current executable code, not consumed by CNN classifier/routing; `inflora.py:285,293–321` |
| Classifier | Single expanding SimpleLinear(768,C_seen), bias enabled; retain old rows; fc parameters enabled for SGD | No task oracle / per-task head routing; `inc_net.py:1418–1488`, `backbone/linears.py:17–33` |
| CE scope | First task all current logits; later CE uses only new-class logits, targets offset by known_classes | Old head rows have zero CE gradient; with this run's momentum=0 and wd=0 they do not update even though the whole fc tensor has `requires_grad=True`; `inflora.py:574–585,609–613,739–753` |
| NME prototypes | Separately computes means from all seen training classes with test transforms | Auxiliary NME evaluation, not the main-table CNN metrics. This is not the KMeans key mechanism; `inflora.py:288–291`; `baseLoRA.py:370–410`; `baseLearner.py:146–156` |
| Optimizer selector | `optimizer_type='sgd'` is read; plain `optimizer='sgd'` is descriptive/redundant here | No SAM/GAM/RWP branch active; stages read explicit LR/momentum/wd; `inflora.py:152`, `baseLearner.py:555–568,762–763` |
| Cosine scheduler | PyTorch CosineAnnealingLR; T_max=`epochs`=40 for initial and later stages; eta_min=0.1×stage LR, reset each task, step once at end of each epoch | Same default as current Vanilla; **different from CL-LoRA's default eta_min=0**. Historical `min_lr` not logged; `inflora.py:381–409,591–592,766–767`; `baseLearner.py:535–539`; `seqlora.py:1335–1360` |
| Loader settings | batch128; train shuffle=true; test=false; default train_num_workers=8 used for both | No explicit generator/worker-init supplied here, so method-specific extra passes consume RNG; `inflora.py:266–278` |
| Test cadence | Every 5 training epochs plus post-task evaluation; no early-stop/best-test checkpoint selection in these loops | Same rule six; `inflora.py:594–600,769–775` |
| Generic evaluation options | flat_eval, feature_flat_eval, feature_cka_eval, feature_proto_eval, attention_probe_eval, both linear-probe gates false | `feature_cka_max_samples`, `feature_proto_max_samples`, `flat_eval_dataset_fraction` etc are inactive evaluation parameters, not InfLoRA covariance sampling knobs |

Current parameter counts from these explicit shapes: B trainable per task = 12×2×768×16 = **294,912**, versus current A+B = **589,824** stored scalars/task. The enabled classifier tensor contains 769×C_seen scalars; only current-class rows receive CE gradients for later tasks. Prior adapters add storage as tasks accumulate, while remaining frozen. Thus identical `rank=16` does not imply the same trainable parameter count or lifetime adapter budget as Vanilla LoRA.

DualGPM's first-task chosen basis width uses `max(1, sum(cumsum(S²/(sum(S²)+1e−12)) < threshold))`; subsequent updates use the remove/retain policy and update basis widths from data. One must not translate `lame=.99` into “99% retained information every task” or claim a fixed per-layer rank for those bases. The actual per-layer widths are not present in the selected raw `.log` files; their `print()` output was not captured there.

## 4. Evidence boundaries and entry-point checks

Current `src/main.py:73–98` parses config/mode/override, loads YAML, and applies explicit CLI overrides. `utils/config.py:5–23` returns the YAML dictionary without inserting method defaults. The current trainer contains no `lamb`, `lame`, `min_lr`, or `train_num_workers` defaults. Therefore the absent keys fall through to the defaults documented above when these dictionaries are run through current source. This still does not prove March's unlogged historical defaults.

The selected original logs explicitly report pretrained hub identifier `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`; the configs and current builder agree on ViT-B/16. Historical checkpoint revision/SHA and package versions are not pinned by these records. Data roots/data_split/transform settings are absent from the six YAML/log parameter blocks; method loaders share DataManager, but a byte-identical historical dataset/transform pipeline cannot be certified from these fields alone. Current data directory reconstruction is separate evidence, not a historical snapshot.

Current source comments at `inflora.py:132–136` describe an older incomplete scaffold (“information-guided initialization could be added”). They contradict the executable implementation at `342–365` and `826–909` and should not be used to conclude that information-guided initialization/DualGPM are disabled.

## 5. Exact selected source paths

Raw log citations below point to the start of the last run block, which is the block used for the selected metrics. Earlier appended blocks are not separate selected cells.

### Aircraft

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/aircraft/0/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224.log:1) (block starts line 1; 1 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_sgd_t20c10_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_aircraft/inflora_inr_sgd_t20c10_r16.yaml:33).
- Metrics: [inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/aircraft/0/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40/exp_run/10/inflora_inr_sgd_aircraft_t20-E40_rank16_lr005_epoch40_vit_base_patch16_224_cl_metrics.json); selected FAA=40.958000000, AAA_stage=41.227975397. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

### Cars196

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cars196/0/inflora_inr_sgd_cars196_t20-E40_rank16_lr005/exp_run/20/inflora_inr_sgd_cars196_t20-E40_rank16_lr005_vit_base_patch16_224.log:1) (block starts line 1; 1 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/inflora_inr_sgd_t20c10_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cars196/inflora_inr_sgd_t20c10_r16.yaml:33).
- Metrics: [inflora_inr_sgd_cars196_t20-E40_rank16_lr005_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cars196/0/inflora_inr_sgd_cars196_t20-E40_rank16_lr005/exp_run/20/inflora_inr_sgd_cars196_t20-E40_rank16_lr005_vit_base_patch16_224_cl_metrics.json); selected FAA=34.345000000, AAA_stage=42.201332143. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

### CUB-200

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cub200/0/inflora_inr_sgd_cub200_t20-E40_rank16/exp_run/20/inflora_inr_sgd_cub200_t20-E40_rank16_vit_base_patch16_224.log:1892) (block starts line 1892; 5 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/inflora_inr_sgd_t20c10_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_cub200/inflora_inr_sgd_t20c10_r16.yaml:33).
- Metrics: [inflora_inr_sgd_cub200_t20-E40_rank16_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/cub200/0/inflora_inr_sgd_cub200_t20-E40_rank16/exp_run/20/inflora_inr_sgd_cub200_t20-E40_rank16_vit_base_patch16_224_cl_metrics.json); selected FAA=60.608000000, AAA_stage=70.601210317. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

### Flowers102

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/flowers/0/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16/exp_run/10/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16_vit_base_patch16_224.log:474) (block starts line 474; 2 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/inflora_inr_sgd_t20c10_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_flower/inflora_inr_sgd_t20c10_r16.yaml:33).
- Metrics: [inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/flowers/0/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16/exp_run/10/inflora_inr_sgd_flowers_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json); selected FAA=60.060000000, AAA_stage=72.979944841. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

### Pets37

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/pets/0/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16/exp_run/4/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16_vit_base_patch16_224.log:462) (block starts line 462; 2 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/inflora_inr_sgd_t20c10_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_oxfordPet/inflora_inr_sgd_t20c10_r16.yaml:33).
- Metrics: [inflora_inr_sgd_pets_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/pets/0/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16/exp_run/4/inflora_inr_sgd_pets_ep40_lr00025_t20_rank16_vit_base_patch16_224_cl_metrics.json); selected FAA=62.178888889, AAA_stage=70.397578924. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

### 5Datasets

- Log: [selected last-run block](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/het5/0/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224.log:1) (block starts line 1; 1 appended run block(s)).
- YAML: [task/seed settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/inflora_inr_sgd_het5_r16.yaml:11); [method and optimizer settings](/data/140-1/users/liying/Flatness_CV/config_exps_paper1_PAC/exp1_rebuttel_5datacombine/inflora_inr_sgd_het5_r16.yaml:32).
- Metrics: [inflora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224_cl_metrics.json](/data/140-1/users/liying/Flatness_CV/outputs_logs/logs_inc_lora/inflora/sgd/het5/0/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40/exp_run/1/inflora_inr_sgd_het5_t5_rank16_lr002_epoch40_vit_base_patch16_224_cl_metrics.json); selected FAA=77.748000000, AAA_stage=77.016266667. These are CNN lower-triangle-derived task-macro metrics, not NME or pooled sample-weighted total accuracy.

## 6. Current-source anchors and fingerprints

These SHA-256 values identify the files read for this audit only; they do not retroactively pin historical experiment source.

- [models_LoRAbasedCL/inflora.py:139](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/inflora.py:139) — `ae999e8457ddd5c4eed09818a1ad9cf44c5afa7d6f60caac3165fc0db89ecdd3`
- [backbone/lora.py:448](/data/140-1/users/liying/Flatness_CV/backbone/lora.py:448) — `6148efe40bdb3c5a2a478badbc9c996506e342658fa9160b9bed76b255018abd`
- [utils/inc_net.py:1418](/data/140-1/users/liying/Flatness_CV/utils/inc_net.py:1418) — `0e8c164f975cc933f4ff7eceebb49ee0f49804fd34bcbb71ead30a3187cb6565`
- [models_LoRAbasedCL/baseLoRA.py:370](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/baseLoRA.py:370) — `748365165894c45e27506215a1d010d1ccdc33c976f695d5d0c10905a6e3e78c`
- [models_CL/baseLearner.py:518](/data/140-1/users/liying/Flatness_CV/models_CL/baseLearner.py:518) — `69c8467a94a9904f0c2b8039477082f0bd76f7b6899a1e6c1a99529a60fd63db`
- [models_LoRAbasedCL/seqlora.py:1335](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:1335) — `ae1622be6176facdbbf18fb25a6cd720c136f6381dfd8c7801b224b72c815731`
- [src/main.py:73](/data/140-1/users/liying/Flatness_CV/src/main.py:73) — `1e1a6c156f2325824cf193753b2360fddaf6820491281fc8f57672a9e28a066b`
- [utils/config.py:5](/data/140-1/users/liying/Flatness_CV/utils/config.py:5) — `7388d8a500ff855f661f1033e4fbc0a69091e2015352a1b489cc8431647654a9`
- [utils/data_manager.py:18](/data/140-1/users/liying/Flatness_CV/utils/data_manager.py:18) — `335d027cf168780f48537dc237ea8df2bbe47169e0e9d9400155abeda0d59662`
- [utils/data.py:1300](/data/140-1/users/liying/Flatness_CV/utils/data.py:1300) — `2c02ac940c77ae898ac18fe9d4e5b797fdfa7d75c22a638c92256382bc6f185a`
