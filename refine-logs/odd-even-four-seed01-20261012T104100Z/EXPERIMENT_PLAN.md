# ImageNet-A 四种局部方向：补充 seed0/1

日期：2026-10-12（Europe/Madrid）。用户授权补齐正向、校正反向、中心差分、基点四方向的三seed比较，并最新明确只用物理GPU4/5/6。新增8次完整训练，复用同协议seed2四项；没有扩大半径、rank或方法范围。

## 数学定义：四组如何获取

令theta0是全部可训练Q/V LoRA A/B与已见FC参数，g(theta)=当前任务同一个增强batch的CE梯度。N(v)=v/(||v||+1e-12)，范数覆盖所有上述可训练参数。

1. 保存完整基点参数、buffers、模式、缓存、梯度引用及随机状态R0。
2. 在theta0读g0=g(theta0)，记录该前向后的随机状态R1。
3. 恢复theta0/R0，在theta_plus=theta0+0.05*N(g0)读取g_plus。
4. 恢复theta0/R0，在theta_minus=theta0-0.05*N(g0)读取g_minus。
5. 恢复theta0；只保留g0前向后的R1；根据下面的选择项设p.grad=U，提交一次普通SGD：theta_new=theta0-eta*U。

所有组均执行g0、g_plus、g_minus三次普通梯度读取。基点组和正向组也读取未用于U的其它探针，用于保持共同预算和诊断。探针位置不是已经接受的优化步骤，不在theta_plus或theta_minus上提交最终SGD。

定义s1=(g_plus-g_minus)/2，e1=[(g_plus-g0)+(g_minus-g0)]/2。这里s1是向量，不是旧GAM强度标量；s1/e1不要求正交，不除以rho。

| 名称 | update_direction | 实际FP32更新算式U | 代数形式 | 新seed |
|---|---|---|---|---|
| SAM方向 | g1_plus | 直接复制g_plus | g0+s1+e1 | 0、1 |
| 校正反向 | backward_reflected | g0+(g0-g_minus) | g0+s1-e1 | 0、1 |
| 中心差分 | central | g0+0.5*(g_plus-g_minus) | g0+s1 | 0、1 |
| 基点对照 | g0 | 直接复制g0 | g0 | 0、1 |

反向组不是直接用g_minus；先求g0-g_minus再加回g0。中心组不是(g_plus+g_minus)/2；它加入的是两侧梯度的半差。实际代码保留父实验的FP32括号顺序，代数重排不保证逐位相等。

## 统一配置

| 项目 | 固定设置 |
|---|---|
| 数据 | ImageNet-A，5981train/1519test，类别0…199固定顺序，10×20任务 |
| 骨干 | ViT-B/16 augreg2_in21k_ft_in1k，冻结 |
| 可训练参数 | Q/V LoRA rank10（无alpha/rank缩放）+已见FC |
| 训练 | 每任务50epochs，batch128保留尾批；每次完整10任务 |
| SGD | momentum=0、weight_decay=0 |
| 学习率 | Task0=.04，后续=.02；cosine T_max50，最低比例.1 |
| 精度/后端 | FP32、math attention、确定性、TF32/AMP关闭 |
| 内层半径 | gam_grad_rho=.05 |
| 外层半径 | 不存在，outer_radius=null；gam_grad_norm_rho=0.0只是兼容字段 |
| 最终步长 | natural，不归一化U，不匹配范数，不除rho |
| 关闭 | EWC/Fisher及其计算、融合、投影/正交修正、SGV、HVP/create_graph、replay |

所有配置从相应父seed2配置精确复制，仅允许seed、prefix、outputs_root三项变化。device=[0]是UUID绑定后的逻辑CUDA0，不是物理GPU0。旧beta/gamma/optimizer_type兼容字段不决定四方向；真实入口odd_even_train.py调用OddEvenStep，再用torch.optim.SGD提交。

公共关键字段：method_name=odd_even_local_v1，probe_branch=local_symmetric，attention_backend=math，step_mode=natural，lambda_flat=1，ewc_lambda=0，ewc_skip_task_fisher=true，in_task_weight_merge=false，task_endpoint_weight_merge=false，approximate_fisher_in_task=false。

## 种子与公平比较

每个seed独立从预训练骨干和该seed的LoRA/FC初始化开始。批次和增强使用原规则seed*100000+task*1000+epoch；同seed四方向的实际输入、targets、sample indices及新FC行初始化必须配对。不同seed的参数、增强和梯度允许不同，不能跨seed要求逐位相等。

| 条件 | seed0 | seed1 | seed2既有AAA/FAA |
|---|---|---|---|
| SAM | 新完整训练 | 新完整训练 | 63.102/44.809 |
| Backward | 新完整训练 | 新完整训练 | 62.615/43.714 |
| Central | 新完整训练 | 新完整训练 | 62.994/44.327 |
| CE | 新完整训练 | 新完整训练 | 60.232/39.260 |

父结果只使用odd-even-rho005-gpu123-20261003T185629Z的OE01–04；不能使用旧native四探针三seed结果。PARENT_REUSE_AUDIT.json绑定原配置、完成回执、矩阵与源码证据。旧两探针Backward是同seed实现复现，不作为独立重复。

## 实现变更及验收

odd_even_step.py数学代码与父文件字节一致。odd_even_train.py仅参数化原硬编码seed2的配置、dry-run及summary，默认仍为2；cuda_witness.py增加--seed。支持seed0/1/2，但本suite只注册0/1。

1. 原数学/回滚/生命周期CPU测试及seed传播测试。
2. 16份真实训练入口--dry-run，检查实际配置/seed/预算；队列负向测试、源190文件、资产、父结果验证。
3. 独立代码审核，冻结代码/配置/文档；新代理原样执行ENVIRONMENT.md的只读命令。
4. 8个短测，每项2任务×1epoch×1batch，共2SGD；覆盖FC扩展和任务后无Fisher/融合。
5. CPU按seed分组的四方向配对；全部通过后放行8正式。失败停止新增启动，保留已有证据，无自动改设置或重试。

每正式2550SGD、7650训练CE、60次隔离重放诊断CE；诊断不更新模型。新增8正式合计20400SGD、61200训练CE和480诊断CE。每短测6训练CE与12隔离诊断CE；短测不能作为正式成绩。

## 资源、记录和分析

只允许物理GPU[4,5,6]，最多3GPU任务并行，统一共享编号+UUID双锁；加锁前后fresh检查显存<500MiB、利用率<10%。不抢占其他任务。每worker在实际卡执行当前seed的FP32 CUDA witness。CPU配对隐藏GPU。

按父训练测得每条约1.75–1.80GPU小时，8条约14–15GPU小时；三卡持续空闲约5–6小时，加短测及资源等待。不承诺占用中的卡何时空闲。

主指标任务等权AAA，辅以FAA、LA、BWT、Forget和逐任务矩阵；样本加权指标独立标注。逐步保存g0/g+/g-/s1/e1/U范数、loss、方向余弦、LoRA BA+FC实际位移、输入身份、恢复与计算计数。三个seed全部完整验收后，才输出均值、样本SD(ddof=1)和相对g0/SAM的同seed配对差与胜出数。

自然步长仍同时包含方向与幅度效应。三seed可检查重复性，但不自动证明Hessian机制或统计等价。本轮不注册负s1、额外系数、外层半径或matched扩展。

实际进度见runtime/queue_state.json；正式新结果summary.csv；包含父seed2的THREE_SEED_SUMMARY.csv及配对比较由collector生成。
