# GAM-EWCLoRA / CL-LoRA seed 可用性有界复核（2026-09-22）

当前 35 条曲线中，本次负责的 10 个 dataset-method 都仍展示 **n=1，seed={0}**。所选原始 metrics/log 的 SHA-256 全部与 source_manifest.json 一致，完整最终矩阵的已观测下三角全部有限。受检的原 run 根与 fine-gapfill-attempt2 中，没有发现可直接补入这些历史单次结果的同配置新 seed。不能用 n=1 报告样本方差/标准差为 0。

另在邻近 `gam-ewc-gapfill-launch-20260917/attempt2` 找到 CUB 与 Flowers 的 `wrapper_lam2000` seeds 1、2：各自可以形成独立 n=2 cohort；但其 RNG 控制和冻结实现不能与旧 seed 0 确认等同，因此本报告不合并成 n=3、不修改图表。它们是本次额外定位的候选，不能据此断言是在 9 月 20 日之后才产生。

## 当前展示与可直接延续的 cohort

| Dataset | Method | 当前 n | 当前 seeds | 原根内可确认的同配置 seeds | 额外候选 |
|---|---|---:|---|---|---|
| aircraft | CL-LoRA | 1 | {0} | {0} | 受检范围未找到 |
| aircraft | GAM-EWCLoRA | 1 | {0} | {0} | 受检范围未找到 |
| cars196 | CL-LoRA | 1 | {0} | {0} | 受检范围未找到 |
| cars196 | GAM-EWCLoRA | 1 | {0} | {0} | 受检范围未找到 |
| cub200 | CL-LoRA | 1 | {0} | {0} | 受检范围未找到 |
| cub200 | GAM-EWCLoRA | 1 | {0} | {0} | 独立 RNG-control cohort {1,2}，不能直接混入 |
| flowers | CL-LoRA | 1 | {0} | {0} | 受检范围未找到 |
| flowers | GAM-EWCLoRA | 1 | {0} | {0} | 独立 RNG-control cohort {1,2}，不能直接混入 |
| het5 | CL-LoRA | 1 | {0} | {0} | 受检范围未找到 |
| het5 | GAM-EWCLoRA | 1 | {0} | {0} | 受检范围未找到 |

## 受检范围与排除项

- 输入：`/data/140-1/users/liying/O-LoRA/figures/finegrained_taskwise_aa_20260920/data/curves.json` 与 `/data/140-1/users/liying/O-LoRA/figures/finegrained_taskwise_aa_20260920/source_manifest.json`；manifest 写明 one selected run per method/dataset，无 SD/CI。
- 历史 GAM：`/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/{aircraft,cars196,cub200,flowers}`。四个 dataset 根均仅有 seed 0 子目录；Aircraft/Cars/Flowers 各一个 metrics，CUB 四个。CUB 另外三个为 λ500、λ5000、dual λ500，原 log 也都 seed 0，既不是不同 seed，也不是 λ2000 的重复实验。
- 曾检查相同相对位置的 `Flatness_CV/logs_inc_lora/...` 四个 dataset 路径，均不存在；这不是全 Flatness_CV 搜索。
- `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2` 的完整 configs/receipts/outputs、suite.json、state.json、collected.json：14 项＝7 formal＋7 smoke，全部 seed 0；formal＝CL 六 dataset（含图已去掉的 Pets）＋GAM Het5。正式输出仅这 7 个 metrics，state completed，没有新增 seed 配置或回执。smoke 仅 2 tasks，不能算重复实验。
- 在 `CopyFlatCV/refine-logs` 与 `O-LoRA/refine-logs` 仅浅列名称含 gapfill/cllora 的邻近目录。`gam-ewc-fine-gapfill-20260918` 初次尝试的 smoke 失败、formal 被 held，且仍全 seed 0；排除。`gam-ewc-five-dataset-gapfill-20260917` 是计划/证据目录。`gam-ewc-paper-gapfill-20260918` 顶层是另一个 emp/mc S/A 协议，未将其记录混入。
- 针对浅列表中发现的 `gam-ewc-gapfill-launch-20260917/attempt2`，仅进一步检查 suite/state/metrics 和 CUB、Flowers `wrapper_lam2000` 四个候选的 config/receipt/raw/log。排除同批 `pure_gam` 和 `wrapper_lam0`。未扫描全盘，也不声称其他目录绝无更多结果。

## CUB、Flowers 的独立 n=2 候选 cohort

四个候选都完成 10 tasks，metrics 记录 eligible=true/status=complete；本次独立复算 raw 下三角并核对 raw/log SHA。每个 dataset 的 seed 1 vs 2 配置仅 `seed`、`loader_seed`、`prefix` 不同。共有设置：model `ewclora_normfisher_gam`、rank 16、λ2000、40 epochs/task、SGD+GAM、batch128；CUB LR .01、Flowers LR .0025。

候选原 log 第 33–34 行明确 `ewc_preserve_fisher_rng=True`、`ewc_isolate_clean_rng=True`，第 66–67 行为独立 loader RNG/对应 seed；原历史 CUB/Flowers seed 0 log 未记录这些开关，且没有相同冻结源码版本证明。候选冻结 source_root 为 `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-rngctl-20260916/code`，suite snapshot SHA `47d16884bec5558fb2c1fbc13f692a04ecd1028bbb04bc057d638882a0b8199d`。Flowers 候选 GAM rho=.2/norm_rho=.2/gamma=.1，历史 Flowers log 不提供这三个参数。因而尚不能认证其与当前图 seed 0 除 seed 外完全相同；不把未记录自动当作 false，也不把当前源码默认当历史事实。

统计定义：每阶段 AA 为该阶段已见任务准确率的算术平均；FAA=最后一个 AA，AAA_stage=全部阶段 AA 的平均。下表方差为样本方差（ddof=1，单位 pp²），SD 单位 pp。仅汇总候选 seeds 1、2，不包含图上旧 seed 0。

| Dataset | n / seeds | FAA mean | FAA SD | FAA variance | AAA mean | AAA SD | AAA variance |
|---|---|---:|---:|---:|---:|---:|---:|
| cub200 | 2 / {1,2} | 73.116500000 | 0.638517423 | 0.407704500 | 75.893790873 | 0.633980155 | 0.401930837 |
| flowers | 2 / {1,2} | 82.850500000 | 1.779787768 | 3.167644500 | 89.535979762 | 1.472409012 | 2.167988298 |

候选原始文件（逐项可追溯，不只依赖汇总统计）：

### cub200 seed 1

- Raw: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/outputs/logs_inc_lora/ewclora_normfisher_gam/gam_gapfill20260917_wrapper_lam2000_run/cub200/1/gam_ewc_gapfill_20260917_cub200_wrapper_lam2000_s1_run/exp_run/20/gam_ewc_gapfill_20260917_cub200_wrapper_lam2000_s1_run_vit_base_patch16_224_cl_metrics.json`
- Config: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/configs/cub200_wrapper_lam2000_s1_run.json`
- Receipt: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/attempt2/receipts/cub200_wrapper_lam2000_s1_run.json`
- FAA=73.568000000, AAA_stage=76.342082540; raw SHA `b17c8ae1b57bcfac3f8696d78e9f07140d25c1efa39d49b1241aad97f535c0f7`。

### flowers seed 1

- Raw: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/outputs/logs_inc_lora/ewclora_normfisher_gam/gam_gapfill20260917_wrapper_lam2000_run/flowers/1/gam_ewc_gapfill_20260917_flowers_wrapper_lam2000_s1_run/exp_run/10/gam_ewc_gapfill_20260917_flowers_wrapper_lam2000_s1_run_vit_base_patch16_224_cl_metrics.json`
- Config: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/configs/flowers_wrapper_lam2000_s1_run.json`
- Receipt: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/attempt2/receipts/flowers_wrapper_lam2000_s1_run.json`
- FAA=84.109000000, AAA_stage=90.577130159; raw SHA `c3632858f8b1743d2b7eb0839ab8d222418181d28fcd937057ef06001ca90ef5`。

### cub200 seed 2

- Raw: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/outputs/logs_inc_lora/ewclora_normfisher_gam/gam_gapfill20260917_wrapper_lam2000_run/cub200/2/gam_ewc_gapfill_20260917_cub200_wrapper_lam2000_s2_run/exp_run/20/gam_ewc_gapfill_20260917_cub200_wrapper_lam2000_s2_run_vit_base_patch16_224_cl_metrics.json`
- Config: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/configs/cub200_wrapper_lam2000_s2_run.json`
- Receipt: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/attempt2/receipts/cub200_wrapper_lam2000_s2_run.json`
- FAA=72.665000000, AAA_stage=75.445499206; raw SHA `0e00f452b0838ffda10895c7a34254039c154db5d40935a0bf3c955104401bed`。

### flowers seed 2

- Raw: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/outputs/logs_inc_lora/ewclora_normfisher_gam/gam_gapfill20260917_wrapper_lam2000_run/flowers/2/gam_ewc_gapfill_20260917_flowers_wrapper_lam2000_s2_run/exp_run/10/gam_ewc_gapfill_20260917_flowers_wrapper_lam2000_s2_run_vit_base_patch16_224_cl_metrics.json`
- Config: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/configs/flowers_wrapper_lam2000_s2_run.json`
- Receipt: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-gapfill-launch-20260917/attempt2/receipts/flowers_wrapper_lam2000_s2_run.json`
- FAA=81.592000000, AAA_stage=88.494829365; raw SHA `360e3e3a82d635aee50e372354e8a1301a58810763266b2a50411b616bdde43b`。

## 当前所选 10 个源路径

- aircraft / CL-LoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned/fine_gapfill_20260918/aircraft/0/cllora_aircraft_s0/exp_run/10/cllora_aircraft_s0_vit_base_patch16_224_cl_metrics.json`
- aircraft / GAM-EWCLoRA: `/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/aircraft/0/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_aircraft_t10c10_r16_s0_vit_base_patch16_224_cl_metrics.json`
- cars196 / CL-LoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned/fine_gapfill_20260918/cars196/0/cllora_cars196_s0/exp_run/20/cllora_cars196_s0_vit_base_patch16_224_cl_metrics.json`
- cars196 / GAM-EWCLoRA: `/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cars196/0/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0/exp_run/20/paperA_as1normfisher_lam2000_cars196_t10c20_r16_s0_vit_base_patch16_224_cl_metrics.json`
- cub200 / CL-LoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned/fine_gapfill_20260918/cub200/0/cllora_cub200_s0/exp_run/20/cllora_cub200_s0_vit_base_patch16_224_cl_metrics.json`
- cub200 / GAM-EWCLoRA: `/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/cub200/0/paperA_as1normfisher_lam2000_cub200_t20/exp_run/20/paperA_as1normfisher_lam2000_cub200_t20_vit_base_patch16_224_cl_metrics.json`
- flowers / CL-LoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned/fine_gapfill_20260918/flowers/0/cllora_flowers_s0/exp_run/10/cllora_flowers_s0_vit_base_patch16_224_cl_metrics.json`
- flowers / GAM-EWCLoRA: `/data/140-1/users/liying/CopyFlatCV/logs_inc_lora/ewclora_normfisher_gam/gam/flowers/0/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0/exp_run/10/paperA_as1normfisher_lam2000_flowers_t10c10_r16_s0_vit_base_patch16_224_cl_metrics.json`
- het5 / CL-LoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/cllora_aligned/fine_gapfill_20260918/het5/0/cllora_het5_s0/exp_run/100/cllora_het5_s0_vit_base_patch16_224_cl_metrics.json`
- het5 / GAM-EWCLoRA: `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/outputs/logs_inc_lora/ewclora_normfisher_gam/fine_gapfill_20260918/het5/0/gam_ewc_het5_s0/exp_run/100/gam_ewc_het5_s0_vit_base_patch16_224_cl_metrics.json`
