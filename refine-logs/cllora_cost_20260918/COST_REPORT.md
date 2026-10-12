# 本次 CL-LoRA 的保存、训练和推理代价

对象是已产生六组结果的冻结版本 `cllora_aligned`，不是对所有 CL-LoRA 实现的泛化判断。证据来自实际执行源码、完成 receipt、训练日志和 CPU/meta 参数核算；本轮没有启动训练或 GPU 速度测试。

**主要代价是逐任务保留专属 adapter，并在推理时遍历所有已见任务分支。训练从第二个任务开始多一次蒸馏更新，任务结束还要构造原型。主干权重只保留一份，没有回放旧训练图片。相同 rank16 不代表与普通 LoRA 具有相同计算/存储预算。**

## 1. 实际结构与增长项

- ViT-B/16，宽度768、12个block，Q/V使用rank16。
- 前6个block使用共享LoRA；共享A冻结、B训练。另保留上一任务的共享LoRA副本用于蒸馏，不复制整个ViT教师。
- 后6个block每个任务保留一套专属LoRA，历史任务冻结；每个任务另有6×3个block权重（K分量也注册，尽管当前只启用Q/V）。
- 代码在初始化时通过 `ModuleList(... for range(n_tasks))` 一次性分配计划的全部任务分支。因此adapter的物理占用从开始就按计划任务数分配，而不是每到一个任务才追加。
- 分类器实际分配为 `[累计类别数 C, 768×已见任务数 T]`，还保留当前任务的训练用proxy head。推理只使用各任务对应的对角原型块。

每套6层Q/V LoRA的参数数目为 `6×2×2×768×16=294,912`。当前注册的LoRA总数是 `294,912×(T+2)`：T套专属、1套当前共享、1套旧共享。大分类器含 `C×768×T+1` 个参数；类数与任务数同时增长时，这一分配可呈二次增长，虽然很多块不参与当前推理。

## 2. CUB-200、10任务的参数存储

下表为FP32张量字节数，不是实测checkpoint文件大小；不包括梯度、激活、框架开销或压缩。

| 项目 | 参数/张量元素数 | FP32负载 |
|---|---:|---:|
| 共享ViT主干 | 85,798,656 | 327.30 MiB |
| 当前与旧共享LoRA | 589,824 | 2.25 MiB |
| 10套专属LoRA | 2,949,120 | 11.25 MiB |
| 大原型分类器 | 1,536,001 | 5.86 MiB |
| 当前proxy head与block weights | 15,541 | 0.06 MiB |
| 除主干外合计 | 5,090,486 | **19.42 MiB** |
| 当前全部state参数合计 | 90,889,142 | **346.71 MiB** |

部署时，旧共享副本、proxy head和非对角原型块可从专门设计的推理导出格式中移除。保留当前共享、全部专属adapter、block weights、每类一个768维原型后，CUB额外负载约 **12.96 MiB**。普通Vanilla LoRA的12层Q/V adapter加200类线性头约 **2.84 MiB**，均不含相同主干；因此即便精简，CL-LoRA的适配器/分类器部署负载仍约为其4.57倍。此精简导出尚未实现，不能将估算当作当前文件。

Het5（T=5、C=635）当前额外state约17.29 MiB、全state344.58 MiB；潜在精简部署额外负载约8.61 MiB。CUB和Het5类别规模不同，不能仅按任务数比例推算总存储。

## 3. 训练开销

首任务每batch只有分类CE更新。从第二个任务起，每batch按顺序执行：

1. 新共享分支跑前6层，旧共享分支也跑前6层，计算KD。
2. KD反向、共享B梯度重加权、一次SGD更新。
3. 当前完整12层分支重新前向，计算CE和block-weight正交项。
4. 再反向、再做一次SGD更新。

按Transformer block调用计，后续每batch有 `6+6+12=24` 层前向，而单次普通LoRA主干前向为12层。KD学生支反向经过共享6层，CE反向经过当前12层；旧共享encoder参数冻结，不进行其主干参数反向。**这是调用量核算，不是实测2倍训练时长。** 两次更新的顺序也是算法的一部分，不能随意合并成一次而称等价。

CUB日志每任务实际 `trainable=457747`，与共享B 147456 + 当前专属A/B 294912 + block权重18 + 当前20类proxy15361完全一致。它比普通LoRA某些阶段的可训练参数更少，但前向/反向执行更多，不能用可训练参数数目代替速度指标。

当前归档ViT注意力还显式计算 `q @ k.transpose` 并保存softmax注意力矩阵；224×224、patch16对应197个token，batch128、12heads时单份FP32注意力矩阵约227.40 MiB/层。实际峰值还取决于autograd保留及临时缓冲，不能把单矩阵大小直接当作整模型峰值或与历史实现的显存差。

教师logits没有detach：旧encoder本身冻结，但双方共享可训练proxy head，因此teacher概率仍向proxy head传播梯度。直接detach会改变训练更新，并不是已证明无损的加速。

任务训练结束还会对当前任务全部训练样本构造原型：`replace_fc` 遍历所有已见分支，因此第T任务需要 **T遍当前任务数据的完整ViT前向**。CUB最后任务为10遍。它不读取旧任务图片，但仍有额外数据加载和计算开销。

## 4. 推理开销最明显

`interface()` 每次预测都循环所有已见任务，对相同输入分别调用完整 `image_encoder`，算该任务类别的分数，再拼接成全已见类别logits；不依赖oracle任务标签。

| 已见任务数 | 当前CL-LoRA完整ViT前向次数/输入batch | 普通单分支LoRA |
|---:|---:|---:|
| 1 | 1 | 1 |
| 5 | 5 | 1 |
| 9（Pets） | 9 | 1 |
| 10（CUB等） | 10 | 1 |

所以10任务下有10倍主干前向调用量，**不是已经测得10倍端到端延迟**。共享权重不等于共享本次前向计算，当前代码没有缓存前6层结果。

可优化方向是缓存前6层的完整token张量，在后6层分支分别继续计算；层调用量由 `12T` 降为 `6+6T`。T=10时为120降至66，仍相当于5.5次12层主干的层调用量。必须缓存中间token，不能直接使用现有 `shared=True` 输出，因为它已经归一化并仅取CLS。

另一处冗余是原型：当前推理只读取每任务的对角块，理论上仅计算当前任务对应分支的原型、紧凑保存 `C×768` 即可。要保持训练轨迹，还需控制DataLoader shuffle消耗的RNG和浮点累加顺序；本轮只定位机会，没有改变运行代码或宣称已优化。

## 5. 这批运行的实际资源记录

| 数据集 | 总墙钟分钟 | 峰值PyTorch allocated显存 |
|---|---:|---:|
| CUB-200 | 96.82 | 12.81 GiB |
| Cars196 | 103.25 | 12.81 GiB |
| Aircraft | 92.27 | 12.81 GiB |
| Flowers102 | 40.29 | 12.81 GiB |
| Pets | 45.73 | 12.80 GiB |
| Het5 | 201.86 | 12.80 GiB |

墙钟来自receipt的结束减开始，包含初始化、训练、原型、评估与检查，不是纯训练kernel时间。GPU为本次环境记录中的Quadro RTX6000 24GB，batch128。显存来自 `torch.cuda.max_memory_allocated()`，不等于nvidia-smi的总使用量或reserved峰值。历史Vanilla/GAM-EWC没有完整匹配的硬件/计时/显存证据，不据这些值声称可靠的跨方法速度或显存倍数。

GAM-EWC也有自己的训练代价：本次冻结代码每batch执行一次clean梯度计算，再做4次GAM closure前向/反向，即5次完整主干任务梯度计算，另有Fisher正则代数与任务末Fisher估计。其Q/V有效ΔW Fisher与参考ΔW各有 `24×768×768` 个FP32元素，约54 MiB，各自为训练状态，在线累计后不按任务逐套保留；推理不需要这些Fisher状态，仍走单分支。因此不能笼统说CL-LoRA在训练、保存、推理所有维度都比GAM-EWC更重。

## 6. 实际模型保存情况

六项配置 `save_task_checkpoints=false`，没有开启trainer使用的 `save_final_model`。正式CL-LoRA输出仅有6份日志和6份metrics JSON，**没有可加载的训练后权重**。源码中的 `save_old_shared_lora()` 是在内存中复制共享adapter，用于下一任务蒸馏，不是保存到磁盘。

因此现有结果可以追溯和复算，但这些输出不足以直接恢复训练后模型进行部署或追加效率测量。若以后要保存/恢复，还需要保留配置、task_sizes、当前task、分类器形状、对应基础权重标识；只dump裸state_dict不能自行恢复Python列表里的任务边界。若需要中途续训，还要额外保存optimizer/scheduler和RNG状态。

## 7. 证据和本轮改动

- [训练循环与两次更新](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:54)
- [逐任务完整推理](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:107)
- [原型构造](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:120)
- [保存开关](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/src/trainer.py:1204)
- [结构化成本与来源SHA](cost_summary.json)、[CSV](cost_summary.csv)
- 专项：[存储核验](storage_audit.md)、[运行时间核验](runtime_audit.md)、[调用路径核验](compute_audit.md)

| 本轮文件 | 类型 | 内容 |
|---|---|---|
| build_cost_summary.py | 新建 | 读取冻结源码/config/receipt，验证SHA和训练参数计数，核算张量负载 |
| cost_summary.json / .csv | 新建 | 理论负载与实际墙钟/显存记录，明确二者口径 |
| COST_REPORT.md及三份专项审核 | 新建 | 成本解释、证据和可优化点 |

没有修改生产训练/推理代码，没有向运行中任务插入计时器，没有新GPU实验。
