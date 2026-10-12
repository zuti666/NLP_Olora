# ImageNet-R GAM-EWC 1×1 implementation review

2026-09-20。按 experiment-bridge Phase 2.5 的方法、超参数、数据/指标、ground truth、随机性与失败处理清单检查。本审查者未参与这批production文件实现，但继承了会话上下文；这是独立同系列审查，不是零上下文fresh-agent认证。只读代码并运行CPU小模型验证，没有GPU任务、训练或队列修改。

**结论：当前指定协议下未发现阻止交付代码的blocking错误。无需修改训练算法。** 尚未进行GPU训练sanity；代码交付通过不等于50epoch实验或GPU端到端恢复已经通过。

审查范围：`scripts/imagenetr_gam_ewc_1x1/{train.py,learner.py,config.json,launch.py,evaluate.py,eval_common.py}`、`tests/test_imagenetr_gam_ewc_1x1.py`，并追读实际使用的CopyFlatCV GAM、AS1 Fisher、LoRA生命周期与checkpoint实现。

## 已确认正确

- **真正原生GAM：** `learner.py:additive_step`调用`optimer_PerturabtionType/gam.py:GAM.step`，4次CE closure。没有Ratio-FS替换，没有额外第5次clean CE。`gEWC`在扰动前由clean参数独立求导缓存，GAM callback只相加，不投影、不进入inner CE。
- **对照和保护设置：** plain SGD-EWC与GAM(base SGD)-EWC同LR/任务/数据/seed设定；lambda=1.2e6、gamma=1、每任务Fisher trace归一化、累计不再归一化。`Learner._compute_delta_fisher`在父类归一化结果上只执行一次`(1-beta)*F+beta/P`，beta=.2，与参考R Ratio-FS保护矩阵处理一致；shaping算法独立替换为GAM。
- **两种坐标没有混淆：** 训练GAM扰动在可训练A/B与头参数上；EWC在有效Q/V `BA`差分上定义并链式回传；离线raw Fisher明确在24个有效Q/V矩阵上计算，头不纳入其坐标。
- **共同新初始化：** `train.py:139`仅从已有R完整checkpoint取300项冻结base，保留新初始化A/B与head。现存两份seed3 `CPU_VALIDATED.json`的350项初始网络hash相同，新增head hash相同，Fisher为空；未误用历史trained adapter作为新Task0。
- **两阶段恢复：** `train.py:112`要求本方法自己的Task0完成标记与一致配置；`patched_load`先复用原Fisher/lifecycle逻辑，再strict加载完整`network_state`、逐张量检查，并设置任务计数。Task1独立启动与连续运行都在任务入口重置对应RNG；新head在其后创建。
- **随机性：** 独立DataLoader generator和worker增强RNG不受GAM额外probe次数影响；4次CE probe重放首probe随机状态，最终只保留一次clean forward的模型RNG推进。新增head、数据清单、全epoch样本索引和首batch像素/标签有receipt；不是仅声称相同seed就证明逐输入相同。
- **保存与失败：** 每任务保存完整Task0/Task1权重、Fisher和RNG，检查冻结base不变及训练参数有限。GAM异常恢复clean参数和RNG，取消callback，抛出失败；已有run拒绝覆盖。`launch.py`默认仅打印计划，必须`--execute`且显式GPU才启动。
- **真实标签评价：** `eval_common.py:60`对old-local20/new-local20/seen40做显式标签范围检查；accuracy和CE都与dataset ground truth比，未用教师/另一模型预测作真值。Task0/1旧20类local CE用于损伤；a21/a22独立使用40类CIL。
- **逐样本EF：** `PerExampleCollector.add`从每样本线性层输入与adjoint恢复该样本有效权重梯度，`losses.sum().backward()`保留每样本未缩放梯度，先逐样本平方再平均。方向二次型先跨层汇总内积后平方；没有用batch均值梯度平方冒充逐样本EF。固定同一256训练样本与确定性transform，两端输入tensor记录一致性有检查。
- **有效位移：** `audit_checkpoint_pair`计算`BA1-BA0`，校验非adapter base和旧20行head不变，排除新分类头。有效模型转换后检查native logits一致，并在真实模型评估入口用逐样本autograd核对hook；输出区分old-at-Task0、old-at-Task1和new-at-Task1 raw trace。

## 非阻塞说明与验证边界

1. 原生GAM的beta1=beta3=1是两个方向之和。即使rho/normrho/gamma都归零，其方向是`2*gCE`而非`gCE`；测试已明确验证这个原生尺度。不要把当前同LR对照表述为已完全隔离“仅平坦度项、其他有效步长相同”。按用户“原生GAM”要求，不建议偷偷除2改算法。
2. 4次/1次指CE网络forward-backward次数。启用EWC后还有一次只经过BA二次项的`autograd.grad`，不应把全部autograd工作宣称只有4/1次；它不增加完整图像网络前向。
3. 当前独立重跑的是CPU小模型测试；两分支CPU完整初始化receipt只证明构造/初始化、不是ViT训练forward或Task0训练后恢复通过。后续正式长跑前应先使用现有`--epochs 1`小任务协议核验真实GPU训练、Task0→Task1恢复及EF入口，本轮用户未要求启动，因此这里不执行。
4. 启动器以显式失败而非恢复已有半截epoch为策略；这在代码中清楚，不把它描述成支持精确epoch续训。无需为当前请求添加恢复框架。

## 独立执行的CPU验证

- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .../Pilot_new/bin/python -m unittest discover -s tests -p test_imagenetr_gam_ewc_1x1.py -v`：**7/7通过**。覆盖独立四probe数学公式、EWC不改变GAM探测点、SGD等于CE+EWC、probe RNG重放、异常复原、原生零半径尺度、有效BA保护链式求导。
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .../Pilot_new/bin/python scripts/imagenetr_gam_ewc_1x1/evaluate.py --self-test`：**通过**。old/new/seen三scope，逐样本autograd核对、microbatch 1/2/5及尾批一致；最大trace绝对误差8.89e-16，方向内积误差6.53e-16。
- 只读检查两份CPU初始化receipt：`initial_network_sha256=f756cda90a597cd0f2217d221e58d9052bd8d5616b4b9a12ac29e51cc65d80cc`，两方法相同；均明确`forward_tested=false, training_started=false`。

Blocking修复要求：**无**。本次未修改production代码。
