# RQ2 跨 T 实现、原始资产与可恢复性审查

审查日期：2026-09-20。源仓库 `/data/140-1/users/liying/CopyFlatCV` 只读；未训练、未启动 GPU、未做网络前向。上级目录和仓库中未找到 `AGENTS.md`。

## 1. 已找到完整 seed0 资产

本报告选取跨 T 的 `Ttraj_fisher_lam2000_lr0p02_e50` 版本，因为它同时保留逐任务旧任务损伤、漂移和动态 Fisher 形状记录。不要和早期 `T_from...`、`Ttraj_from...`、`Tcompare_task0_shared...` 版本混合；这些版本也存在，但诊断项目不完全相同。

- Task0 根：`outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50`。
- 两个 Task0 模型：上述根下 `{sgd,gam}/seed0/checkpoints/task_00.pt`；对应 `lora_w_a_0.pt`、`lora_w_b_0.pt`、`fc_state_0.pt`、`fisher_state_0.pt` 全部存在。
- Task0 路径清单：上述根下 `manifests/{sgd,gam}_seed0.env`，均包含可用的原始指标路径与 checkpoint 路径。
- 跨 T 根：`outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50`。
- 两条序列模型：上述根下 `{sgd,gam}/T10/seed0/checkpoints/task_01.pt` 至 `task_09.pt`，各任务所有上述 sidecar 均存在。
- 原始逐任务 JSON：跨 T 根下 `trainer_outputs/logs_inc_lora/ewclora_normfisher_gam_intask/{sgd,gam}_from_task0_T10_lr0.02_e50/imageneta/0/s1c_{sgd,gam}_from_task0_T10_e50_lam2000_s0/exp_run/20/s1c_{sgd,gam}_from_task0_T10_e50_lam2000_s0_vit_base_patch16_224_cl_metrics.json`。
- 原始日志：`logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/{sgd,gam}_T10_seed0.log`。日志训练日期为 2026-06-22。
- 原有整理：`result_summary/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/task0_trajectory.csv`；另有 `GAM-EWC/results/logs_try_audit_20260918/task0_trajectory.csv`。

本次独立从原始 JSON 重算，并 CPU 检查 checkpoint：

- [task_trajectory_recomputed.csv](task_trajectory_recomputed.csv)：20 行，两方法 × Task0–Task9；损伤全部相对自身起点重新计算，保留 train/test、local/full 分类器、当前任务与已见任务平均 accuracy。
- [task_assets_manifest.json](task_assets_manifest.json)：20 个 checkpoint 的真实路径、大小、元数据、sidecar 状态、历史 log 配置与小文件 SHA256；没有把大 checkpoint 仅凭文件名认定可用。
- [audit_task_assets_cpu.py](audit_task_assets_cpu.py)：可重复的只读 CPU 审查脚本。

## 2. 实验配置和跨 T 逻辑

`config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml` 及原始两条日志前 96 行一致记录：ImageNet-A，固定类别顺序（`class_shuffle=false`），200 类拆成 10×20；ViT-B/16，rank10；`ewclora_normfisher_gam_intask`；seed0；batch128；Task0 lr0.04、以后 lr0.02；每任务50 epochs，cosine；momentum0、weight_decay0；EWC λ2000、gamma0.9、eta0，单任务 Fisher trace 归一化和累计 Fisher 归一化均开；in-task Fisher averaging、project、dual ascent、weight merge、delta-W penalty 关。

GAM 分支 `lambda_flat=1`，base optimizer SGD，rho0.05、norm-rho0.2、gamma0.1、beta1/2/3均1。`scripts/run_s1c_stage2_T_fullseq_from_task0.sh:191` 载入各自 Task0 manifest，`:210`–`:220` 按 `name` 决定未来继续 GAM 或 SGD，`:233` 设置从Task0恢复，`:236` 指定逐任务保存。因此旧跨 T 比较的是 **SGD→SGD 与 GAM→GAM**，GAM 在后续任务仍生效。

`utils/factory.py:254`–`:258` 将该名称映射到 `as1_normfisher_gam_intask.Learner`。基类 `models_LoRAbasedCL/seqlora.py:23`–`:26` 明确是单个固定大小的 adapter 持续训练，`:281` 的 after_task 只更新已见类别，不重置/合并 A/B。`ewclora_youyue_fitArchitechture.py:228`–`:233` 用每层 Q/V 的 `B@A` 定义24个有效权重块，总维数14,155,776。

**不要把每个 task checkpoint 的 BA 累加。** 此数据系列的每个 checkpoint 都是同一 adapter 的累积训练状态，应计算 `BA_t−BA_0`。其他 fresh-LoRA/merge 变体需要另一套恢复规则，不能套进本主实验。

主线 scaling 为恒等：`backbone/lora.py:465` 默认 `learn_alpha=false`，`:541`–`:545` 构造 `IdentityScale`，`:427`–`:431` 直接返回输入；`utils/inc_net.py:62`–`:63` 没有覆盖该默认，`seqlora.py:585`–`:586` 也显式设置false。全部20个checkpoint均没有scaling/wrapped参数key。因此虽然forward `:226`–`:227` 经过wrapper，本资产的有效权重确为BA，没有另一个alpha/r系数。推广到其他LoRA变体须重新检查scale。

## 3. checkpoint 的 CPU 验证已通过

使用 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`（torch2.4.1+cu121），`torch.load(map_location='cpu', weights_only=True, mmap=True)` 读取全部20个 manifest。

- 每个 manifest 都含350个 state key；均有完整 `network_state`。
- 每个任务的 Task0 分类器前20行权重和 bias 与本方法Task0 **逐位相等**。
- 所有非 LoRA backbone 参数与各自 Task0 **逐位相等**。
- 全部20个 checkpoint 的24块 `BA_t−BA_0` 范数重算与 JSON 的最大绝对误差为 `1.8371766e−8`。
- 每任务 A/B、FC、Fisher sidecar 全部存在。

所以 seed0 的 C/t9 与 S04 不存在 checkpoint 缺失问题。该结论是张量层资产核验，尚未做恢复后前向输出一致性测试。

代码恢复接口见 `models_LoRAbasedCL/baseLoRA.py:252`–`:288`；`:840`–`:886` 载入当前 A/B 后清空历史 `saved_A/saved_B` 并把内部 `task_id` 设0，防止把累积 adapter 重复相加。完整网络 manifest 写入见同文件`:393`–`:424`。离线诊断宜直接恢复完整 network_state，并禁止隐式下载/替换 pretrained backbone。

## 4. 原始数据支持到哪里

| 指标 | SGD→SGD | GAM→GAM |
|---|---:|---:|
| Task0 初始 test local accuracy (%) | 85.3420 | 87.6221 |
| Task9 后旧任务 test local accuracy (%) | 80.1303 | 85.3420 |
| Task9 后旧任务 test full accuracy (%) | 64.1694 | 72.3127 |
| Task9 train local loss 增量 | 0.188107 | 0.039556 |
| Task9 test local loss 增量 | 0.260343 | 0.034413 |
| Task9 test full loss 增量 | 1.033760 | 0.713910 |
| Task0→Task9 有效权重距离（CPU重算） | 1.19758410 | 1.40047259 |
| Task9 已见任务 macro CNN accuracy (%) | 40.9940 | 52.0780 |
| Task9 新任务 CNN global accuracy (%) | 34.62 | 53.85 |
| Task9 动态旧任务 normalized Fisher trace | 1.0000001 | 1.0000000 |

这是一例“位移更大、相对自身起点的旧任务损伤更低”的 seed0 真实序列证据。不能直接变成“仅 Task0 shaping 引起保留改善”的因果结论，因为后续优化器不同；也不能把同EWC规则解释为保护矩阵相同。新A表需固定后续 optimizer。

Task1 的 CNN class-IL 数字也可读到：SGD→SGD `(a11,a21,a22)=(85.34,82.08,75.20)`，GAM→GAM `(87.62,81.76,81.50)`。第二行仍不是目标 `GAM→SGD`。相应 test local loss增量是 −0.006824/−0.015311，表明短期 loss 改善可为负，不能擅自截零；正文也不应仅凭a21认定shaped起点更好。

## 5. 旧跨 T 诊断的测量限制

1. `src/trainer.py:501`–`:567` 的 `_compute_task0_local_delta_fisher` 先对batch mean CE反传，再平方并按batch size加权。它是 **batch-gradient square proxy**，不是逐样本 empirical Fisher。`:565` 会trace归一化。旧 `task0_fisher_current_trace≈1` 不提供raw敏感度信息；字段 `fisher_raw_quadratic` 的 raw仅指未乘1/2，并不表示输入F未归一化。
2. `src/trainer.py:228`–`:277` 同时评价 local Task0 mask `[0,20)` 与当前所有已见类别full head。前者可用于固定旧任务局部几何；后者随着任务数加入新类别竞争，目标发生变化，不能混在同一固定损伤定理里。
3. Task0 head 的前20行本次已证明全程未变；但旧代码没有另存独立 anchor head 对象，而是切当前head。新离线路径应显式固定Task0 head和mask，并检查端点 logits/loss。
4. `src/trainer.py:200`–`:223` 对train用`mode=train`，抓一次随机增强后缓存为CPU固定batch；对test用`mode=test`。两方法各自进程是否抓到完全相同随机增强未存hash，不能只凭样本数量声称两方法train输入逐像素配对。旧test更直接。新诊断应固定同一ID清单并用确定性eval transform；若需要复现旧train曲线，另注明缓存增强口径。端点sanity优先核对历史test；新deterministic train只要求在新协议下路径端点与直接评估一致，不能要求与旧随机增强train loss逐值一致。
5. 历史 Fisher/max-batches=100、batch128，但实际旧任务1259训练样本，因此遍历完所有旧任务样本。损伤batch64×20足以覆盖1259训练及307测试。当前 `/data/115-2/users/liying/datasets/imagenet-a/{train,test}` 第一20类计数也恰为1259/307，尚不等于图像内容hash完全相同。
6. 旧T0 `fisher_state_0.pt` 只有normalized张量、trace_mass和normalized统计，未保留每样本Fisher；不能由sidecar还原新定义的raw empirical Fisher。

## 6. t9 离线路径的最小实现方案

- 统一参数坐标为24个Q/V有效权重块，`Delta=BA_9−BA_0`，不是A/B直线；A/B直线通常对应非线性的BA路径。
- `backbone/lora.py:205`–`:216` 已支持 `external_delta_w_q/v` 加到当前BA。因此加载Task0原始网络，在节点s设置外部权重为 `s*Delta`，就实现精确有效权重直线；9节点`s=0,1/8,...,1`，还原后输出须与T9模型Task0 head吻合。
- 同文件`:220`–`:224` 已有有效权重梯度hook，可复用获取逐样本 `g_i(s)`。raw trace为`mean_i sum_blocks ||g_i||²`。
- 若论文 `F_emp=mean_i g_i g_i^T`，方向敏感度须直接累积 `mean_i (sum_blocks <g_i,Delta>)²`；先把梯度平方成diagonal再乘Delta²会丢交叉项，测的是另一个量。保存raw trace和方向项可避免存储完整F矩阵。
- 单位位移路径敏感度 `S_path=integral mean_i(g_i·Delta)^2 ds / ||Delta||²`，9点先用复合Simpson，复用5点作误差检查；9/17点追加只在误差明显时做。
- 固定256训练样本，分层同ID，两方法同样本、同确定性transform；单独测全307 test旧任务loss/accuracy。新C的所有任务已有loss/acc/drift无需重训。
- S04 用两方法各自归一化的真实Delta方向交叉放到两个anchor，共2 anchors×2 directions×6 radii；`r={0,.25,.5,1,1.5,2}`。r=0节点按anchor去重。明确超出终点半径的点是同方向外推。

## 7. A表补分支和源码兼容性

当前脚本把anchor方法与future optimizer绑定，不能只改seed/T就得到目标表。需将两个字段分开，并从GAM Task0 anchor继续Task1时显式 `optimizer_type=sgd`、`optimizer_type_by_task=sgd`（或清空该schedule）、`gam_base_optimizer=sgd`；`lambda_flat=0`可作明确记录。`as1_normfisher_gam_intask.py:1180`–`:1208` 证实SGD分支只有CE+EWC+默认关闭的delta penalty，不调用shaping，lambda_flat不参与该分支。保持主配置其它项目不变；只补缺失的GAM→SGD Task1 50epochs，既有SGD→SGD在前向/版本sanity通过后复用。

以下是具体执行前的兼容性检查，不能声称已完成：

- 旧默认Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python` 已不存在；本次可用Pilot_new已验证torch读取，但尚未验证全部timm/torchvision训练依赖。
- 历史日志的预训练权重是 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`。最新代码仍默认 `timm.create_model(...,pretrained=True)`；`baseLoRA.load_task_checkpoint` 恢复sidecar而不恢复base state，故需保证pretrained backbone精确相同，或使用完整manifest恢复。不能依赖当前timm默认模型别名自动一致。
- 当前源码新增了identity/producer receipt校验，旧checkpoint没有这些字段。`baseLoRA.py:559`–`:566` 中严格校验默认关闭；不要直接套用其它新实验的`resume_require_experiment_identity=true`，也不要伪造旧receipt。
- 本次审查来源是6月原日志和tensor资产，当前代码在9月有新增逻辑。离线恢复端点一致性必须先通过，若需补A训练则冻结所用源码与依赖。不要混入当前S3/Fisher averaging/weight merge配置。

首轮真正缺的是 B 的raw逐样本敏感度、S04实际方向扫描、C的t9路径，以及A的固定未来SGD缺失分支。四格、全部中间路径和full-model整条路径都不是资产恢复所必需。
