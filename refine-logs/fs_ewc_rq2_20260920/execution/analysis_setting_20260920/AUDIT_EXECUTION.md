# 两任务续训实际执行审计（2026-09-20）

范围：只读检查已完成的两支 Task1 续训、冻结源码、配置、日志、checkpoint 和 RNG 收据；CPU 读取 tensor 做等值/哈希检查，没有启动训练、GPU 推理或修改已有实验。机器可读检查结果见 [AUDIT_EXECUTION.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/analysis_setting_20260920/AUDIT_EXECUTION.json)。历史结果及最终指标口径由本目录其他分析整合。

## 1. 实际比较的是什么

`A_sgd_to_sgd_seed0` 恢复历史 SGD Task0 的完整 network + 自身 Fisher；`A_gam_to_sgd_seed0` 恢复历史 GAM Task0 的完整 network + 自身 Fisher。**两支 Task1 都是普通 SGD + EWC，GAM 塑形在 Task1 已关闭。** 比较变量包含起点的 LoRA A/B、旧分类头和自身 Fisher，不能将差异纯粹归因于“平坦性”这一个单独变量。

配置入口 [train_continuation.py:85](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py:85)，`continuation` 模式强制 `optimizer_type=sgd`、`lambda_flat=0`、`max_train_tasks=2`、`resume_from_task=0`。冻结 trainer [src/trainer.py:929](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/src/trainer.py:929) 将总任务数截到 2，[src/trainer.py:966](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/src/trainer.py:966) 从 task 1 开始，故本次仅新训练 Task1 一次；`completed_tasks=2` 表示包含恢复来的 Task0。

| 项目 | 两支共同实际设置 |
|---|---|
| 数据 | ImageNet-A，本地 `imagenet-a/train`、`imagenet-a/test` |
| 类别 | 不 shuffle；Task0 类 0–19，Task1 类 20–39；原数据有 200 类/10 任务，但本次止于 Task1 |
| 模型 | ViT-B/16，LoRA rank 10，12 层 Q/V；同一组 LoRA A/B 连续更新 |
| Task1 数据 | 997 张训练，seen test 合计 561 张；batch 128，每 epoch 8 batch |
| 优化 | SGD，lr 0.02，50 epoch，cosine，eta_min **0.002**，momentum=0，weight decay=0 |
| 正则 | EWC λ=2000，η=0；各自 Task0 的 trace-normalized Fisher；无 replay |
| 关闭项 | GAM/flatness、梯度投影、dual ascent、in-task Fisher、weight merge、delta-W penalty |
| 数值 | FP32，生产 runner 显式关闭 matmul/cuDNN TF32 |

`init_lr=0.04/init_epoch=50` 仍在配置中，用于 Task0 规则，本次没有重新训练 Task0。cosine 下限不是 0，来自 [seqlora.py:288](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/seqlora.py:288) 默认 `0.1*lr`。

真实训练增强由 [utils/data.py:300](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/utils/data.py:300) 定义：224 RandomResizedCrop（scale 0.05–1、ratio 3/4–4/3）+ RandomHorizontalFlip(0.5) + ToTensor，没有 Normalize。诊断中“train split”固定测试变换则是 Resize256 + CenterCrop224 + ToTensor；二者不能混称相同训练损失。沿用历史变换本身不是此次两分支差异。

## 2. 确实执行了 SGD 分支及其损失

注册入口 [factory.py:254](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/utils/factory.py:254) 的继承链为：

`as1_normfisher_gam_intask → as1_normfisher_gam → ewclora_youyue_fitArchitechture_gam → ewclora_youyue_fitArchitechture → seqlora`。

它没有走另一个会强制 GAM 的 `GAM_EWCLoRA_lambda` 类。优化器 [baseLearner.py:765](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_CL/baseLearner.py:765) 返回 `torch.optim.SGD`；原队列训练 stdout 也有 `_build_optimizer: sgd`。实际更新 [as1_normfisher_gam_intask.py:1180](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py:1180) 走非 GAM 分支：清梯度，计算 task CE + Fisher penalty + 已关闭的额外 penalty，backward，SGD step。

训练目标为

\[
L = \operatorname{CE}(z_{20:40}(x),y-20)
  +\frac{2000}{2}\sum_{\ell\in\{12\text{层的Q,V}\}}
    \sum_{ij}\widetilde F_{0,\ell,ij}
    \big[(B_\ell A_\ell)_{ij}-(B_{0,\ell}A_{0,\ell})_{ij}\big]^2.
\]

CE 仅切新任务 20 类：[ewclora_youyue_fitArchitechture.py:263](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:263)。Fisher penalty 的真实乘数含 **1/2**：[同文件:292](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:292)。日志 `EWC(λ·F·Δθ²)` 名称省略了 1/2，`λ*D_F` 则确实是没有 1/2 的另一列；不是把 λ 乘了两次。

EWC 在有效权重 `BA` 坐标中计算，24 个 768×768 矩阵，共 14,155,776 坐标（不是直接在 rank-10 的 A/B 元素中惩罚）。各自 Fisher 已加载并固定用于整个 Task1；归一化分母是全体 tensor 总和 `trace(F)+1e-12`，[as1_normfisher_gam.py:200](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam.py:200)。本次重读 Task0 Fisher：SGD trace=0.9999999952，GAM trace=0.9999999791；均 24 tensor、无负值或非有限值。

`batch_empirical` 的实际估计是 batch-mean gradient square proxy：batch CE 均值的梯度平方 × batch size，再除总样本数；**不是逐样本 empirical Fisher**。实现 [as1_normfisher_gam.py:348](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam.py:348)、[同文件:381](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam.py:381)。Task1 结束才用第 51 次 loader 估 Task1 Fisher，并按 γ=0.9 累加/重新归一化；这个末尾的新 Fisher 不会回头改变已完成的 Task1 训练。

## 3. 恢复、参数更新和配对是否可靠

恢复首先调用旧框架加载生命周期和 Fisher sidecar，再对 `task_00.pt['network_state']` 作 `strict=True` 完整恢复且逐 tensor equality assert，[train_continuation.py:239](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/train_continuation.py:239)。从模型名推断或只加载 LoRA 的风险在此由完整 network restore 避免。Task1 新 head 在同一 task RNG 重置后初始化；两支新 20 行的 hash 完全相同。

冻结 ViT 的 300 个 state tensor 保持不变，仅 LoRA A/B 与分类头设为可训练。旧 head 行并非独立 `requires_grad=False`，但局部 CE 不使用旧行、EWC 不依赖 head，且 momentum/weight decay 均为 0，因此旧 20 行无更新；runner 会 assert，**本审计另行读取 before/after checkpoint，确认两支旧 head 的 weight/bias 前 20 行逐元素完全不变、300 个冻结 tensor 全部不变**。新 head 总共 40 行，因此训练 log 在扩头前列出的 trainable params=384020 不是扩头后的数量；扩头后按结构为 399400（LoRA 368640 + head 40×769）。

独立校验结果：

- 两支 config 仅 6 个字段不同：prefix、optimizer_tag_override、resume checkpoint、resume metrics、output root、save checkpoint 路径。全部训练超参相同。
- 两份 RNG 收据解析后逐条相同：task seed=20261929，同一新 head hash，同 dataset IDs，每轮完整样本 index hash 相同，每轮第一 batch 的像素+标签 hash 相同。
- 每支 51 次完整 shuffled loader 遍历，均 997 样本/8 batch：50 次优化 epoch + 1 次 endpoint Fisher；并非训练了 51 epoch。
- 冻结快照清单 199 个源码文件的当前 hash 全部匹配；两支 run_manifest 的 source_manifest/runner hash 都匹配当前文件。
- 两支原 anchor checkpoint/Fisher 文件当前 hash 匹配训练时收据；完整 Task1 checkpoint、完成收据和 COMPLETE.json 均存在。

收据没有记录每个 batch 的全部增强后像素；因此可以确认完整采样顺序、首 batch 增强结果及生成器/worker 配置配对，不能将收据表述为每一张增强图像都逐字节审计过。

## 4. 与负向精度差有关的可支持判断

| 实际观测 | SGD 起点 | GAM 起点 |
|---|---:|---:|
| 50 epoch 平均训练 CE | 0.56100484 | 0.60433106 |
| 平均训练 EWC penalty | 0.000215778 | 0.000091691 |
| 末端有效 BA 位移的L2平方（日志 `delta_norm`） | 0.14058718 | 0.12241307 |
| 末端有效 BA 位移的L2范数 | 0.37494957 | 0.34987580 |
| 末端 EWC penalty | 0.000329497 | 0.000140443 |
| Task1 local-head test accuracy | 88.19% | 87.40% |
| Task1 seen-40-head test accuracy | 74.80% | 72.05% |

`CurTask_Test_accy` 使用 [seqlora.py:301](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/seqlora.py:301) 的 `logits[:,task_start:task_end]`，这里为新 20 类。CNN 新任务 74.80/72.05 则在全部 40 类竞争，不是同一口径。GAM 分支的局部新类准确率也低 0.79pp，扩大到 seen-40 竞争后差 2.75pp：数据支持既有新任务适应差异，也有旧/新 head 竞争差异；不能仅用“遗忘更严重”解释平均下降。

EWC λ 相同，GAM 分支观测的 EWC **损失**更小，故不能从现有证据宣称“GAM 起点惩罚更大所以学不动”。另一方面，非 GAM 更新分支没有记录 EWC 梯度范数，损失小不严格等价于梯度影响小。较小位移也可能与起点 A/B 因子、有效参数化、旧 head/Fisher、学习动态共同有关；当前一组配对 seed0 不能给出唯一因果结论。

审计未发现此负向差值由错用 GAM optimizer、两分支 lr 不一致、少训练、丢失 Task0 backbone/head/Fisher、旧头意外更新或配对数据顺序不同造成。要检验“只塑形 Task0 就必然提升后续 SGD+EWC”，这组结果本身是一个负向观测；是否与历史链路一致需要匹配历史 Task0→Task1 的 optimizer schedule、seed/RNG 协议及评价口径。
