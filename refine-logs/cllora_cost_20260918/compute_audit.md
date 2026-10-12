# CL-LoRA compute audit (frozen attempt2 implementation)

只读静态审核；未修改实验代码、未启动模型训练或大模型运行。以下算量为编码器调用/Transformer block 次数，不是实测 FLOPs 或端到端加速比。

审计源码根目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code`。具体配置为 12 层 ViT，shared 层 0–5，specific 层 6–11，Q/V LoRA，dropout/attention dropout/drop path 均为 0。

## 训练和推理调用量

- 首任务每个 batch：一次 12 层 CE forward，一次 backward/optimizer step。
- 后续任务每个 batch：KD student 的新 shared 6 层、KD teacher 的旧 shared 6 层，然后完成一次 KD backward/step；再重新执行当前分支完整 12 层 CE forward/backward/step。共 24 个 block forward，约两个完整 ViT forward 的 block 数；反向经过 KD student 的 6 层和 CE 的 12 层，不经过冻结的 teacher 编码器。
- 第 T 个任务之后，每个测试 batch 调用 `interface`，循环全部 T 个任务分支，每支完整执行 12 层，合计 12T 个 block forward。没有真实任务标签路由；每支只生成所属类别的分数，最后拼接全部已见类。
- 任务末原型计算 `replace_fc` 对当前任务训练集遍历 T 次，每次调用一个完整编码器分支；合计当前训练样本数 × T 个 full-ViT forward。它在 `no_grad` 和 encoder eval 模式下执行。
- trainer 常规路径的 `eval_task` 只有一次全已见测试集扫描；源码另一次调用属于 resume-eval 分支，不能误算成日常重复评估。

## Teacher 没有 detach 的实际影响

`forward_kd` 返回 `proxy_fc(new_features), proxy_fc(old_features)`；loss 对 teacher softmax **没有 detach**。但是 `_freeze` 已冻结全部基础权重、旧 shared A/B，输入图像也不要求梯度，所以旧 encoder 的 6 层本来就不建立反向图。Teacher 仍然需要一次 6 层 forward，这是生成目标特征的成本。

Teacher logits 有梯度，是因为两支共用当前可训练的 `proxy_fc`（weight 和 sigma）。对

`L = -lambda * sum(q_teacher(phi) * log p_student(phi, theta))`

求导时，当前代码包含 teacher 分布对共享 head 参数 phi 的导数项。Detach teacher 将删去该项，从而改变 proxy head 的 KD 更新，接着改变 CE 更新和后续训练。它**不是保持本次实现结果的无损优化**，也不会省掉 teacher 的 6 层前向。仅为冻结 old encoder forward 加 `no_grad` 基本是显式化已有行为；若把 `proxy_fc(old)` 也包进去，则改变算法。

## 可以共享前六层，但须缓存完整 tokens

`interface` 所有分支默认 `use_new=True`，前 6 层不读取 task-specific 权重。因此在同一个 eval batch 内可只计算一次 patch embedding、位置编码和 shared 0–5 层，再让每个分支执行 specific 6–11 层。总 block 数从 `12T` 降至 `6+6T`：

| 已见任务 T | 原实现 | 共享前缀后 | block forward 减少 |
|---:|---:|---:|---:|
| 5 | 60 | 36 | 40.0% |
| 9 | 108 | 60 | 44.4% |
| 10 | 120 | 66 | 45.0% |

这相当于编码器 block 数的约 1.67–1.82 倍改善上限量级，实际时间需实测。推理依然随 T 线性增长，不能称为单分支推理成本。

必须缓存第 5 层输出的**全部 tokens、尚未执行最终 norm**的张量。现有 `Encoder(shared=True)` 已执行 `self.norm(x)[:,0,:]`，只剩 CLS，不能直接拿其返回值继续后 6 层。以 batch128、197 tokens、768维、FP32 计，完整前缀张量约 73.9 MiB；可只缓存当前 batch。

训练中 KD 的 student 前缀不能直接复用于 CE：两个阶段之间已 `optimizer.step()` 更新 shared B 和 proxy head。复用会使用旧参数激活，改变训练。新/旧 KD shared 分支的权重也不同，不能合并成一个前缀。冻结的 patch embedding 可在同一 batch 内复用，但不是主要 ViT block 成本。

## 原型只算当前 branch：预测公式等价，重放轨迹须另行处理

`replace_fc` 为当前类别在所有历史 branch 上填充原型；但 `interface` 只读取行任务与列 branch 相同的对角块。当前类别在历史 branch 上的 off-diagonal 原型在现有预测接口中从未读取，后续 `update_fc` 只是把它们复制到更大矩阵，未来也不会转成对角块。普通训练使用独立 `proxy_fc`，不读取这些原型。

因此，固定同一编码器参数和同一组样本均值时，只为当前 branch 计算当前类原型，保留历史任务对角原型，可保持当前接口的预测公式。末任务原型编码器 forward 可由 T 遍当前训练集降为一遍。分类器也可只保存每个任务的 C_t×D 原型，替代当前 C_seen×(T·D) 的宽矩阵；有效原型存储减少到 1/T。

但本次 `proto_loader` 设置了 `shuffle=True`，每遍还创建 DataLoader iterator。删除前面的遍历会改变随机数消耗、当前 branch 的样本累加顺序，以及后续训练 batch 顺序；浮点均值也可能有细微差异。故“公式等价”不自动等于“归档运行逐位重现”。实现时必须单独处理 loader RNG 消耗和累加顺序，或明确作为新运行并验证指标；也可先保留旧分支 loader 遍历、跳过其 GPU forward，节省主要编码器成本而保留原随机数路径，但此方案仍保留 CPU 解码成本。

此结论限定当前已关闭额外探针的正式配置和 `interface` 预测路径；其他未来直接消费完整 fc 矩阵的工具需要同步适配。

## 证据位置

- [learner](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:25)：冻结规则、KD/CE 两步更新、prototype loader。
- [network](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:18)：encoder 前向、KD 共享 head、interface 对角块、replace_fc 全分支遍历。
- [archived attention](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_archived_vit.py:300)：shared/new/old 分支选择与 specific branch 操作。
- [trainer main CL loop](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/src/trainer.py:1110)：训练后一次 eval_task。
