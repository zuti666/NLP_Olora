# ImageNet-R 两阶段：有效 Q/V 权重 empirical Fisher 续评规范

2026-09-20。本轮只读源码、JSON 元数据和 CPU checkpoint/sidecar；未运行 GPU、训练、模型前向评估或队列，也未改动原实验代码。这里区分代码中 task0/task1 与论文第一/第二任务。

## 1. 可直接复用什么

可以复用 `rq2_common.py` 的逐样本 EF 测量核心：`EffectiveQKV`、`PerExampleCollector`、`fisher`。需要新增 ImageNet-R 数据与端点恢复适配，不能直接运行原 ImageNet-A `diagnostic.py` 或 `endpoint_a.py`。

本次目标 cohort 是 `ratiofs_l12c20_full_twostage_s345_v1`，路径位于 `CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage`，模型为 `ewclora_normfisher_fishershape`。主行 `control`、`fs_ewc`，seeds 3/4/5；当前计划支持行是 `factored_s1`。本次还预查了 `factored_s2` 的恢复可行性，不因此扩大必做范围。它使用 **Ratio-FS**，不能标成 GAM。

| 行 | 第一任务起点来源 | 第二任务 | 本轮技术核验角色 |
|---|---|---|---|
| control | native anchor | SGD + EWC | 主比较 |
| fs_ewc | ratio anchor | Ratio-FS + EWC | 主比较 |
| factored_s1 | ratio anchor | SGD + EWC | 计划中的起点作用支持行 |
| factored_s2 | native anchor | Ratio-FS + EWC | 额外可用资产，不列本次必做 |

这组是持续更新同一组 Q/V 低秩因子的实现；不沿用旧 `low-rank-cl-new` 六行表的 fresh K/V adapter、projected-GAM 或其旧漂移解释。

**已有 raw EF 结果存在。** `imagenetr_endpoint_region_pilot_s345_v1` 的 sgd/ratio seeds 3/4/5 使用同一批 anchor，但测量的是因子和分类头坐标。当前待补的是**有效 Q/V 坐标的两阶段端点 EF**。另一个 `imagenetr_endpoint_region_heldout_s345_v2` 属于独立 80/20 训练 cohort，不可拼进这里的旧表。

## 2. checkpoint 恢复接口与限制

设 `BASE=CopyFlatCV/outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/logs_inc_lora/ewclora_normfisher_fishershape`，`TAG=ratiofs_l12c20_full_twostage_s345_v1`。精确相对路径模板：

```text
# group: control/factored_s2 -> native；fs_ewc/factored_s1 -> ratio
BASE/ratiofs_stage1_{group}_{TAG}/imagenetr/{seed}/lora_ratiofs_stage1_{group}_{TAG}_s{seed}/exp_run/checkpoints/
BASE/ratiofs_full_twostage_{setting}_{TAG}/imagenetr/{seed}/lora_1x1_{setting}_{TAG}_s{seed}/exp_run/checkpoints/
```

第一任务 anchor 有 `task_00.pt` 完整 `network_state` 及 `lora_w_a_0.pt`、`lora_w_b_0.pt`、`fc_state_0.pt`、`lora_meta_0.json` 等。已检查的 control seed3 完整 state 有 350 张量，含完整 backbone、48 个 A/B 张量和 20 类 head。本次所查 control/fs_ewc/factored_s2 共九条第二任务目录均只有 factor、head、metadata sidecar，没有完整 `task_01.pt`；factored_s1 的相同资产形式见之前的 `new_r_ratio_pair_assets.json`。

恢复顺序：

1. 创建相同 ViT-B/16、rank10 Q/V 包装，`pretrained=False`，严格加载 anchor 的全部 `network_state`。禁止从另一缓存/网络下载的 backbone 替换它。
2. 在原生 LoRA 模型上验证 anchor sidecar 与完整 state 的 A/B、head 一致。再复制模型，按 `lora_meta_1.json` 的层/通道顺序覆盖端点 A/B，并恢复 **完整 `fc_state_1.pt` 40 类 head**。
3. 不把端点因子另加成新 adapter；不把 anchor 的 BA 再相加。该分支的 sidecar 已是当前累积因子。对每层断言 `task_id==0`、`saved_A/saved_B` 为空、缩放为 IdentityScale；否则停下重新判定生命周期。
4. 原生重建模型通过数值检查后，才执行 `convert_effective`，冻结 base 与 classifier，eval 状态下开放 24 个有效 Q/V 矩阵供求导。

源码依据：完整保存入口 [baseLoRA.py:290](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/baseLoRA.py:290)；普通 resume 不是完整 backbone 的独立恢复保证；[baseLoRA.py:840](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/baseLoRA.py:840) 恢复当前因子并清空历史 branch，注释明确否则会双计数。`SeqLoRA._train` 在 task0 仅构造一次 LoRA、后续继续更新，见 [seqlora.py:1512](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/seqlora.py:1512)。原 forward 的 Q/V 注入见 [lora.py:174](/data/140-1/users/liying/CopyFlatCV/backbone/lora.py:174)。

本轮 CPU 检查发现上述 control/fs_ewc/factored_s2 九个 anchor/endpoint metadata pair 都精确列出 layer0–11、Q/V、task_id0；九个 pair 的 endpoint head 前20行 weight/bias 与对应 anchor **逐比特相等**。因此这些行的旧 local20 head 在前后比较中固定。不同方法可以有不同旧 head，不能宣称跨方法 head 相同。新20行的绝对范数不是新 head 的训练位移，因为其训练前初始值未单独保存。

这里仍有资产边界：第二任务只保存 sidecar，完整端点由“原 backbone 在该训练协议中冻结”的代码/日志证据与 anchor 重建，而非独立端点全量快照。应在结果 manifest 写明 `endpoint_restoration=anchor_backbone_plus_endpoint_sidecars`，同时检查原日志配置及重建 accuracy。不能把未保存的端点 backbone 称为已逐张量比较过。

## 3. 主测量：旧训练样本、旧20类 CE、有效 Q/V 坐标

固定旧任务训练样本集合 S，N=|S|，定义

\[
w=\operatorname{vec}\{W_{\ell,Q}^{\rm eff},W_{\ell,V}^{\rm eff}\}_{\ell=1}^{12},\quad
g_i(w)=\nabla_w[-\log p_{\rm old20}(y_i\mid x_i;w)],\quad
\widehat F_{\rm old}^{\rm emp}(w)=\frac1N\sum_{i\in S}g_i(w)g_i(w)^\top.
\]

固定 backbone 的 Q/V base 权重加 BA 与直接以 BA 增量作坐标具有相同导数和位移；本实现用 24 个 BA 有效矩阵，各768×768，共14,155,776个坐标。不包括 K、其余 backbone、A/B 因子或 head；不限制扰动留在 rank10 流形。

主输出为

\[
T_{\rm old}(w)=\frac1N\sum_i\|g_i(w)\|_2^2,
\quad q_d(w)=\frac1N\sum_i(g_i(w)^\top d)^2.
\]

每个样本先算梯度平方，再对样本平均。`rq2_common.py` 使用每样本 CE、`losses.sum().backward()` 与线性层 hook 提取每个样本梯度；batch size 只影响吞吐，不能将 batch 平均梯度的平方替代此量。[PerExampleCollector:150](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:150)、[fisher:326](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:326)。方向内积先跨所有块相加再平方，保留完整外积的交叉项。实现不物化完整矩阵；不要宣称已保存其全部对角元素。

raw 表示不做 trace/max/均值归一化，不加正 floor/damping，不把 EWC 或 Ratio-FS 正则加入 CE。样本数 N 的平均仍然必须保留。模型 eval、禁用 autocast 与 TF32；梯度/矩阵乘为 FP32，累计使用 float64。该量独立于训练保护矩阵：目标训练日志中的 normF、damping 与 batch_empirical Fisher 不能用于代填这里的 raw EF。

## 4. 数据与 loss scope

使用原 `DataManager('imagenetr', ...)` 与真实 class order 解析文件/标签；已查目标日志为 class_shuffle=False、顺序0–199、init_cls=increment=20，因此旧标签0–19、新标签20–39。避免从另一数据根目录随意选一个同名 ImageFolder。

旧任务 **source=train，mode=test**：`Resize(256,bicubic)→CenterCrop(224)→ToTensor`，没有 Normalize、随机裁剪或翻转。它是训练样本上的确定性测量，不是训练增强下的期望 EF。来源：[utils/data.py:300](/data/140-1/users/liying/CopyFlatCV/utils/data.py:300)、[iImageNetR:682](/data/140-1/users/liying/CopyFlatCV/utils/data.py:682)、[DataManager:88](/data/140-1/users/liying/CopyFlatCV/utils/data_manager.py:88)。

首轮复用256张比例分层抽样协议：先按旧训练各类数量作 Hamilton 分配，再固定 seed20260920 类内无放回抽样，各方法/训练seed共用完全相同 ID。保存真实类别计数、class order、全样本路径/标签、选中索引、文件 hash、解码 tensor hash、Pillow/torchvision 版本。不能沿用 ImageNet-A 的1259/307数量断言或旧 manifest；也不能默认为每类等量。若改全旧 train，结果必须改写 sample count 和协议名。

| 字段 | 数据 | logits 与标签 | 意义 |
|---|---|---|---|
| old_local20_anchor_trace | 固定旧 train S | anchor logits0:20；标签0–19 | 主指标：旧知识起点敏感度 |
| old_local20_endpoint_trace | 同一旧 train S | endpoint logits0:20；标签0–19 | 主指标：第二任务后旧知识敏感度 |
| new_local20_endpoint_trace，按执行计划必做 | 固定新 train 子集 | endpoint logits20:40；标签减20 | 新任务自身 endpoint 敏感度 |
| old_seen40_endpoint_trace，可补 | 固定旧 train S | endpoint logits0:40；原标签 | 含新旧类别竞争的旧数据 EF |

原 `RestoredNetwork.forward` 固定切旧20类，见 [rq2_common.py:93](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:93)。仅报告主指标时可保留此口径；补新20/seen40时必须显式 scope 接口，不能只更换数据集标签。

旧 train/test 的 local20 loss 增量使用同一旧 head，定义 `L_old(theta1)-L_old(theta0)`；accuracy 另报原论文 CIL a11/a21/a22。旧local20 EF 与 seen40 的 a21 是不同 loss 范围，不要混名。没有原始 Task1 训练前40类随机新 head 快照时，不能把20类 anchor CE与40类 endpoint CE称为同一函数的损伤；也不要为重建端点重抽新 head。

## 5. 正确位移：持续因子与 fresh adapter 分开

目标持续 Q/V 分支在同一冻结 base 上：

\[
\Delta_{\ell,c}=B_{\ell,c}^{(1)}A_{\ell,c}^{(1)}-B_{\ell,c}^{(0)}A_{\ell,c}^{(0)},
\quad r=\sqrt{\sum_{\ell,c\in\{Q,V\}}\|\Delta_{\ell,c}\|_F^2}.
\]

`delta_norm` 的历史平方量应与 r² 比对；不能直接把它叫 L2 范数，也不能把终点 `||BA||` 当本段位移。训练代码按有效 BA 取差的接口见 [ewclora_youyue_fitArchitechture.py:228](/data/140-1/users/liying/CopyFlatCV/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:228)。

一般有多个历史分支时，forward 有效权重是 `W_base + Σ_j s_j B_j A_j`，位移必须取两个端点整体之差。**只有**新任务创建全新 adapter、初始 B=0、旧分支/base/scales全冻结，且比较同一可加 forward 的 merge 前端点，才有本段 `Δ=s_new B_new A_new`。若执行 factor-wise 相加、非加法 merge、重置/缩放改变，需重建真正 forward 的有效权重后再求差。这些条件不能从旧 fresh-K/V 表移植到本目标。

可低成本加方向诊断：对每行自己的 `d=Δ` 同时测两端 q，然后报告 `q/||Δ||²`；r=0 时明确 undefined，不做隐式除零。若以后做两阶段路径，沿本行 `w0+sΔ` 固定旧head，显式改为 task0/task1，不能沿用原 `diagnostic.py` 的 task0/task9 硬编码。

## 6. 已有 endpoint_region 工具的复用边界

[run_imagenetr_endpoint_region_eval.py:381](/data/140-1/users/liying/CopyFlatCV/scripts/run_imagenetr_endpoint_region_eval.py:381) 确实循环每个样本调用 autograd，将每个梯度向量逐坐标平方、求和、除N，得到 raw empirical-Fisher **对角向量**；其 sum 是该坐标的合法 EF trace。因此不能把它笼统批评成 batch proxy。

但该脚本的模型构建 [902行](/data/140-1/users/liying/CopyFlatCV/scripts/run_imagenetr_endpoint_region_eval.py:902) 从20类 anchor恢复后，以固定 head seed 新扩展40类；不是本次已训练第二任务 endpoint。它以所有 trainable LoRA A/B 与 classifier 求导，`_forward_logits` 不切旧20；trace 坐标为 `trainable_lora_factors_plus_classifier`。基于该对角向量算的 `Σ_j F_jj d_j²` 缺少完整 EF 的交叉项，也不同于 rq2 hook 的 q。

现行数据构建 [980行](/data/140-1/users/liying/CopyFlatCV/scripts/run_imagenetr_endpoint_region_eval.py:980) 从旧 train 划出20% validation，再按每类前缀选诊断样本。它不等同于本方案完整旧 train 分布的比例分层256子集。复用旧已训练 pilot 时，这20%不能自动称为训练未见样本；v2独立留出训练 cohort 则另有定义。既有 JSON 的 `protocol_version`、`evidence_role` 和真实 sampler manifest 应各自保留，不根据现行源码反向覆盖旧版本定义。

可借用其 ImageNet-R loader、严格 anchor 恢复和 manifest 思路；不要直接把其 trace 放到有效 Q/V 列，也不要以该脚本的新随机 head替换真实端点 head。`run_imagenetr_endpoint_region_pipeline.sh` 默认 phase=all 包含训练；本次纯评估不能调用默认 pipeline。

## 7. 最小适配与必须的 sanity

实现独立 ImageNet-R 评估入口和输出目录，保留原结果。最小增量为：数据/样本 manifest 适配；完整 anchor＋endpoint sidecar 恢复；显式 old_local20 loss scope；按 anchor/endpoint 输入而非 task9；继承正确逐样本 hook 及精度设置。无需新增训练，首批两主行×三seed可完成；同seed control 与 factored_s2 的 native anchor 测量可按完整模型、head、data、code hash复用，避免重复。

上线评估前检查：

1. 记录输入 SHA256、metadata、实际日志配置、parent chain；检查 rank10、24块Q/V、全部缩放1、没有历史分支或训练可变 base。对新 fs_ewc/factored_s2 配对补齐 parent provenance，不能将以前仅 control/factored_s1 的旧审查 JSON 当作它们已验证的证据。
2. 原生恢复模型与有效矩阵模型在相同确定性样本上的 logits/CE相符；量化并记录容差。对端点重算 a11/a21/a22 与历史日志对照，若精度或图像解码环境有差异，明确旧值/新值和原因，不静默强行覆盖。
3. 两个真实样本逐个 autograd 对照 hook 的 trace与方向内积；batch1与batch8结果在FP32容差内一致，N与最后不满batch都正确。有限差分检查真实Δ的方向导数；不拿旧 ImageNet-A sanity 代替新恢复/数据/scopes验证。
4. 所有方法相同样本hash和预处理；eval、TF32关闭、无autocast；r=0重复稳定；设置 `w0+Δ` 复现endpoint的旧local20输出。检查旧head固定与Δ非零；若测新20/seen40，每个scope独立验证标签切片。

每行至少保存：seed、setting真实更新规则、anchor/endpoint资产hash、samples/manifest、loss_scope、parameter_scope、两端 raw trace、两端同口径CE、旧train/test损伤、有效Q/V r和r²、CIL accuracy；可附真实方向q与单位位移敏感度。raw测量与训练保护Fisher字段分开命名；不将小于三seed的完成结果包装为三seed统计。

相关已有核心审计：[ImageNet-A EF实现审计](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta_two_stage_ewc_gam_plan_20260920/AUDIT_FISHER_IMPLEMENTATION.md)。该文的核心估计器数学定义可复用，ImageNet-A结果值、样本数、task9路径、lambda2000及GAM标签均不移植到本目标。
