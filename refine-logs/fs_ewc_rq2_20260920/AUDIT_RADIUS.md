# 跨 r 实现与旧数据审查

审查日期：2026-09-20。只读检查 CopyFlatCV；本目录新增 CPU 重分析脚本和衍生 CSV。未启动训练、模型推理或 GPU 作业。未发现适用 AGENTS.md。下列源码行号指当前磁盘版本，不能替代 6 月运行时源码哈希。

## 1. 本地原始记录与用户整理目录对应

- 用户的 `fqa_r/lora/Fig_stage2_r-2-lora-imageneta/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50` 对应服务器原始目录：
  `/data/140-1/users/liying/CopyFlatCV/logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/`。
- `fqa_partial/` 中存在 `s1c_{sgd,gam}_r_fqa_from_task0_s0_task00_endpoint_{train,test}_fqa_partial.json`，共 4 个 JSONL。每文件完整 81 行，半径 0:0.25:20，K=100。每行保存 `per_u.loss_delta`、accuracy、linear term 和固定锚点 Fisher 二次式。没有保存 direction seed/hash/ID、扰动点 raw trace。
- 同 tag 的 `outputs_logs/sgd_gam_s1c/stage2_r-2/.../trainer_outputs/.../*cl_metrics.json` 保存 base loss、参数空间/维数、Fisher 总质量等。完整原始路径与 SHA256 见 [radius_reanalysis_manifest.json](radius_reanalysis_manifest.json)。
- 用户的 `fqa_r`、`Fig_stage2...` 不是仓库根下现存目录，不能把展示目录名当作服务器训练入口。
- 用户指定的 `full-fisherImportant/topF_batch_1pct_cached_g01` 对应 `logs/fullmodel_multifisher_fqa/topF_batch_1pct_cached_g01`。**该 tag 两分支均失败**：`sgd.log:282–295`、`gam.log` 末尾为 `strict_load_fisher=true` 下缺 `outputs_logs/fisher_stage1/cache_{sgd,gam}_batch/fisher_task00.pt`。没有此 tag 的原始 radius JSON。存在汇总 CSV 不代表诊断完成。
- 临近 `logs/fullmodel_multifisher_fqa/random_batch_cached/fqa_partial`、`topF_batch_10pct_cached/fqa_partial` 有 JSON，但不同扰动分布/配置，不能顶替指定 top1% 作业。`stage2_r-full-imagentr` 展示目录尚未得到唯一原始映射。

## 2. LoRA 主线执行链

1. `scripts/run_s1c_stage2_r_fqa_from_task0.sh:12` 读取 `config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml`；`:69–76` 设共享 Task0 manifest、日志、结果路径；`:165–167` 检查 Task0/Fisher/metrics。
2. Launcher `:83–120` 开启 resume-only endpoint train/test、`flat_eval_param_scope=delta_w`、固定 anchor Fisher、r-grid/K100；`:183–193` 调 `python -m src.main --override ...`。
3. `src/main.py:58,133` → `src/trainer.py`。`utils/factory.py:254–258` 将 `ewclora_normfisher_gam_intask` 映射到 `models_LoRAbasedCL/as1_normfisher_gam_intask.py`，继承 `as1_normfisher_gam.py` 与 `ewclora_youyue_fitArchitechture_gam.py`。
4. `models_LoRAbasedCL/ewclora_youyue_fitArchitechture_gam.py:135–199` 按 train/test 调 `utils/mechanism_eval.py:evaluate_anchor_metrics`，`:196` 输入的是 `_fisher_past_delta`（训练时存的保护矩阵）。
5. `utils/mechanism_eval.py:2466–2599` 构造稠密 DeltaW 注入、缓存输入、基准 loss/梯度和半径采样；`:1197–1803` 聚合、输出 JSONL。

配置确为 ImageNet-A、ViT-B/16、LoRA rank10、200类/10任务/每任务20类、lambda2000、gamma0.9、Task0 50epochs lr0.04，后续50epochs lr0.02、SGD(momentum0)、cosine、batch128；固定 class order/no replay。依据 YAML `:16–44,55–77` 和运行日志前100行。历史方法名为 **GAM / AS1**；按用户已明确的映射视为FS-EWC shaping实现，不再增设身份验证任务。历史关闭averaging的诊断与A-on实验仍需分开标注。

## 3. r 到底是什么

- 该主线是 **稠密 DeltaW 坐标扰动**，不是 LoRA A/B 因子坐标，亦不是保持 rank10 的流形内扰动。
- `mechanism_eval.py:321–378` 找到 Q/V（也兼容另一分支 K/V）矩阵的 external slots，每槽尺寸768×768，共24槽、d=14,155,776；在每槽上加可微稠密张量；head 不参与扰动。实际计数经 metrics 验证。
- `:833–858` 使用标准高斯除全向量 L2 norm；`:1398` 乘 r，因此 r 是所有槽拼接后的 Frobenius 范数。
- `backbone/lora.py:205–227` external delta 加到 BA，然后经 `scaling_factor[0]` 作用。若新 C 要宣称“有效权重”路径，必须把 scaling、历史 adapter contribution 与当前分支 BA 一致计入，不能默认 `BA` 差就是完整部署函数位移。
  跨T专项核验现已确认本主线 `learn_alpha=false`、IdentityScale、单个持续更新adapter，因此本资产可简化为BA差；完整证据见 [AUDIT_TASK_ASSETS.md](AUDIT_TASK_ASSETS.md)。其他变体仍需重查。
- 因子接口 `summarize_radius_lora_factor_fisher_quadratic_error` 在 `mechanism_eval.py:1806`，区分 tangent radius 和 actual induced DeltaW（包含二阶交叉项）。用户给 `run_imagenet_r_task1_fqa_lora_factor_adam_gam.sh:15` 调的父脚本 `run_imagenet_r_task1_fqa_adam_gam.sh` **当前缺失**，不能直接重放该 launcher。

## 4. 配对与数据变换边界

- `mechanism_eval.py:1356,1390–1397` 在每个半径内部重新抽随机方向。因此旧 K100 不是100条贯穿 r 的射线，不能按索引做跨 r 射线 bootstrap 或宣称方向配对。
- 跨方法都设 seed0，但无局部专用 generator、无保存 direction IDs；前置模型操作会消耗全局 RNG。**同 seed 不足以验明跨方法方向完全相同**。旧 per-u 只能作为各自采样分布，不能作已验证的 paired direction 证据。
- `src/trainer.py:1026–1037` resume train 是 `source=train, mode=train`；`utils/data.py:300–311,719–723` 是 RandomResizedCrop+flip。`mechanism_eval.py:71–92` 只取消 shuffle，不改变 transform；`:212–232,2540–2550` 缓存已经抽样的图像张量。
- 因此一个 run 内所有 r 的 base/perturbed 输入固定、r0偏差严格为0，但 train 是“一次缓存的随机增强视图”；train/test 之间和方法之间未保存样本张量哈希。新的256样本评估必须固定 sample IDs、分层选择和确定性 test transform（或明确保存增强视图及其hash）。
- LoRA主线 r0 只含Task0前20类 head，无旧/新head竞争问题。新 C 若在t9使用200类head，会混入扩展head影响；旧任务诊断应冻结Task0 head/mask，另报标准CL accuracy。
- Launcher看似 `flat_eval_batch_size=64`，但 DeltaW FQA 分支 `:2464` 调 `_make_eval_loader(loader,None)`，沿用128 batch。`:2545` 实际控制取样上限是 `flat_eval_loss_max_batches=20`；单改 `flat_eval_fqa_max_batches` 不足以改变该分支预算。

## 5. Fisher：可复用什么，缺什么

- 两方法载入24个矩阵，trace均约1：SGD `0.9999999906867743`，GAM `0.9999999774154276`。随机方向下 Q≈r²/(2d)，例如 r20 两者约1.41e-5。这不是“两个模型raw敏感度差不多”的证据，而是trace归一化后的几何结果。
- `mechanism_eval.py:2538,1422–1427` 只算固定 anchor 的 `0.5*r²*uᵀF_anchor*u`；没有在 `theta+r*u` 重新求F，因此不能从旧Q曲线恢复邻域 Fisher trace，也不能反推出路径F。
- 训练默认 `as1_normfisher_gam.py:128–130` 是 `batch_empirical`；`:381–432` 对 batch均值CE求梯度再平方，`:289–296` 返回trace归一化版本。新论文的逐样本 empirical Fisher 与这个保护矩阵不同。
- **现存通用诊断函数名也有陷阱**：`mechanism_eval.py:1103–1168` 的 empirical_fisher_vector_product/DeltaW版本实际上是 batch均值梯度外积，不能直接贴“逐样本 empirical Fisher”标签。
- 可复用 `as1_normfisher_gam.py:434–489` 的 sample_empirical 逐样本梯度计算逻辑，但新评估最好仅流式积累 `mean(||g_i||²)` 和 `mean((g_i·u)²)`，不存完整 F、不改EWC buffer，不调用会trace-normalize的上层入口。
- 路径二次型若定义的是完整 empirical F，需逐样本 `(g_i·Delta)²`；`sum(diag(F)*Delta²)` 会丢交叉项，是另一项近似，不能冒称同一指标。

## 6. 已完成的CPU重分析

运行 [reanalyze_radius.py](reanalyze_radius.py) 得 [radius_damage_seed0.csv](radius_damage_seed0.csv)（324行）和 [radius_tolerance_seed0.csv](radius_tolerance_seed0.csv)（40行）。完全使用原始 `per_u.loss_delta`：G=mean(max(deltaL,0))，Q95是正向损伤线性插值95%分位。prefix radius为首次超阈值前的最后网格点，不线性插值，不代表连续球保证。阈值继承用户提供计划；不声称历史预注册。

| split | method | G(r10) | Q95(r10) | G(r20) | Q95(r20) |
|---|---|---:|---:|---:|---:|
|train|SGD|0.015052|0.021390|0.070704|0.085250|
|train|GAM|0.010819|0.016757|0.058342|0.071434|
|test|SGD|0.016966|0.037694|0.075242|0.122806|
|test|GAM|0.017091|0.038894|0.078166|0.120965|

| split/metric/method | eps .001 | .005 | .01 | .02 | .05 |
|---|---:|---:|---:|---:|---:|
|train G SGD|2.75|5.75|8.25|11.5|17.25|
|train G GAM|3|7|9.5|12.75|18.75|
|train Q95 SGD|1|4.25|6.25|9.25|15.25|
|train Q95 GAM|1.5|4.75|7.75|11|17|
|test G SGD|1.5|4.75|7.5|11|16.75|
|test G GAM|1.5|4.75|6.75|10.75|16.25|
|test Q95 SGD|.25|1.75|3|6.25|11.5|
|test Q95 GAM|.25|1.75|3.25|6|12.25|

训练扰动表现改善；测试正向损伤均值未一致改善，有些阈值容忍半径反而略缩小。GAM的更高绝对测试accuracy（r0 87.622% vs85.342%）不能替代“相对自身起点损伤更小”：r20损失增量如表。必须保留test结果，不能将train结论扩写为泛化稳健性已经确认。

## 7. 最小新增诊断实现要求

1. 沿用Task0恢复和external DeltaW接口，先r0复现旧函数/旧head；新确定性变换的结果独立标记，不能假定等于旧增强缓存值。
2. 单独保存方向bank生成规范和hash，跨两方法、所有半径、train/test共用。r0只算一次。16方向 × 6非零半径 +1锚点 =97点评估/方法；256样本的raw trace成本需按逐样本梯度实测吞吐。
3. 对同一扰动点同步报loss损伤与raw trace；raw trace固定同一256分层旧训练样本。禁止用已trace-normalized保护F替代。
4. actual方向小扫描使用两个模型各自完整Task0→t9有效位移，并交叉施加到两个anchor。先验收端点函数等价、delta coordinate顺序/scale/head。
5. t9九节点直接求逐样本方向导数平方；积分误差可比较5/9节点（5节点是9节点子集），需要时17节点；无需默认计算全部中间任务路径。
6. 旧train/test dense曲线可用作seed0探索证据；正式paired dense曲线需重新使用方向bank跑前向，不能把旧索引强行当作已配对。
