# Llama-1B IncLoRA+EWC：batch64 / seed42 对齐重跑

用户于本会话要求与旧 SeqLoRA / IncLoRA / OLoRA 的有效 batch64、seed42 对齐，并重新运行、区分结果。本文件替代原 batch40/seed1993 补全计划作为本轮执行配置。

## 参考与本轮配置

旧三类基线的 Order1/2/3 实际日志均为 BF16、8 GPU × BS1 × GA8 = batch64，seed42，data_seed=None。完整审查记录见 `../bf16_ewc_execution_20260918/historical_batch_audit.json`。

当前服务器仅有 7 张 RTX 6000 Ada 48GB。可执行配置采用 **4 GPU × BS1 × GA16 = batch64**，seed42，省略 CLI data_seed 以保持历史 None；安装的 Transformers4.28 分布式 sampler 在此情况下使用 seed42。固定四卡并逐卡核验 BF16，不按临时空闲卡数改变协议。

这对齐了有效 batch、随机种子、每个优化器更新的样本集合及每任务更新数；物理 GPU 数和 GA 与旧八卡运行不同。LoRA dropout 随机流及浮点归约顺序仍可能不同，不能声称八卡逐位复现。若用户要求物理八卡原样执行，需等具备八张可用卡的设备。

- 新协议 ID：`bf16_b64_seed42_ddp4_bs1_ga16_lam0p5`。
- 方法：纯 IncLoRA+EWC，新增 rank8、alpha32、dropout0.1、q_proj/v_proj，冻结历史 adapter。
- 精度：BF16 AMP，FP16 false，参数保持 FP32；BF16/FP16 full_eval false，TF32 沿用并记录环境实际默认值。
- 优化器：实际 transformers.AdamW，betas(0.9,0.999)、eps1e-6、weight_decay0、correct_bias true；max_grad_norm1。
- 每任务1 epoch；首任务 LR1e-3，后续1e-4；constant scheduler、warmup0；长度512/50，evalBS4。
- EWC：gamma1，raw full-data Fisher，不归一化、不 damping；OLoRA 两项显式为0，无 GAM/SAM。
- **CLI EWC lambda=0.5**：当前循环每个微批先用 CE/GA，再加 EWC。GA8×lambda1 与 GA16×lambda0.5 均给相同名义累计约束强度8。这是累积步数的确定性补偿，不依据测试结果调参。记录参考 GA8、参考 lambda1、实际 GA16、实际 lambda0.5。
- 保持旧循环每 epoch 的取整语义：DBPedia218、Amazon78、Yahoo156、AGNews62次更新；对应尾部48/8/16/32样本未形成完整更新。Fisher仍遍历完整数据集。
- 无梯度检查点，group_by_length=false、dataloader_drop_last=false，与历史默认一致。

## 执行与验收

依次运行新 Order1（DBPedia→Amazon→Yahoo→AGNews）、Order2（DBPedia→Amazon→AGNews→Yahoo）、Order3（Yahoo→Amazon→AGNews→DBPedia），每个 Order 从原始基座开始。仅加载本 Order 上一阶段的 adapter/Fisher。

正式实验前强制执行独立 DBPedia→Amazon smoke：每任务128训练样本、2个更新步、16预测样本、每卡2个Fisher微批（全局8）。验证实际 BF16、四卡分片、优化器对象、跨卡参数一致、rank8→16、Fisher1→2、历史 adapter 不变、第二任务更新后 EWC 惩罚和梯度为正。smoke 不作为正式结果，正式阶段无样本上限。

新目录前缀为 `experiment_queue/inclora_ewc_bf16_b64_s42_ddp4_`。保存独立源码/配置/验证器快照、哈希、模型与数据文件哈希、完整依赖、各阶段 argv、日志和状态；失败立即停止，重新执行使用新 run_id。

## 结果严格分组

1. 旧 batch40/seed1993 运行 `inclora_ewc_bf16_ddp5_20260918T145252Z` 已完成12个正式阶段，原始文件与完成状态保留，不改为失败或覆盖。
2. 本轮 batch64/seed42 输出独立 JSON/CSV/LaTeX，显式写入协议 ID、batch、seed、world_size、BS、GA、实际 lambda；只汇总本轮三个 Order。新三 Order 未完整结束时 Avg 留空。
3. 与历史 SeqLoRA/IncLoRA/OLoRA 比较时注明旧8卡/GA8、新4卡/GA16及源码/环境来源差异。
4. 历史 EWC/GAM 的 batch40/seed1993 与本轮配置不同，不并入本轮均值，不标为同协议组件消融。
5. 新结果从四任务各7600条真实预测重算整数正确数和未舍入 exact match，再求 FAA；不会根据旧分数预填或按结果选择重跑。
