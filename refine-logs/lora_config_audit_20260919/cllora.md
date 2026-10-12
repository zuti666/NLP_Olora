# CL-LoRA 六数据集：冻结配置与实际实现核验

生成时间UTC：2026-09-19T13:02:10.166459+00:00。只读原实验，未训练、未更改配置或结果表。

六个正式config均与完成receipt中的SHA一致；CL-LoRA learner、aligned network和archived ViT三份实际运行源码均与source_frozen.json一致。以下CL-LoRA结论有运行冻结证据；Vanilla源码对照只引用现存可读文件，不冒充旧运行源码快照。

## 结论

**CL-LoRA所有方法专属超参数和实现规则在六个数据集上相同。** 差别仅为数据集及任务边界、每任务epochs、基础学习率、命名。不能说所有训练细节都与Vanilla相同：除方法机制、head和初始化之外，cosine最低学习率与loader设置也存在差异或历史未记录项。

## 六个数据集实际变化

| 数据集 | 任务边界（每任务新类数） | T | 首/后续LR | 首/后续epochs |
|---|---|---:|---:|---:|
| cub200 | 20×10 | 10 | 0.01 / 0.01 | 40 / 40 |
| cars196 | 16+20×9 | 10 | 0.05 / 0.05 | 40 / 40 |
| aircraft | 10×10 | 10 | 0.05 / 0.05 | 40 / 40 |
| flowers | 12+10×9 | 10 | 0.0025 / 0.0025 | 40 / 40 |
| pets | 5+4×8 | 9 | 0.0025 / 0.0025 | 40 / 40 |
| het5 | 200/100/196/102/37 | 5 | 0.02 / 0.02 | 30 / 30 |

这六组rank=16、batch=128、seed=0、无回放、class_shuffle=false，基础LR和epoch与当前所选Vanilla六格的日志字段一致。Het5的increment=100只是配置占位；实际DataManager给的是不等类数五任务流，CL代码以get_task_size与累计边界计算，不用task_id×increment假设。

## 跨数据集相同的方法专属设置

| 项目 | 六数据集相同的实际规则 |
|---|---|
| LoRA位置 | shared blocks 0–5；task-specific blocks 6–11；msa=[1,0,1]，只作用Q/V |
| Rank与缩放 | rank16；shared直接加BA，无alpha/r缩放；specific乘每个block的可学习Q/V权重 |
| Shared初始化 | A=Uniform(0,1)随机矩阵QR分解所得Q转置，正交行；B=0；共享A训练期间冻结 |
| Specific初始化 | A=Kaiming uniform(a=√5)，B=0；每任务一个specific分支，当前分支A/B更新，旧分支冻结 |
| 旧shared教师 | old A/B初始0；每个任务后复制当前shared作为下一任务教师；旧shared冻结 |
| Block weights | 每任务6×3参数，初始化Uniform(.5,1.5)；只训练当前任务权重；前向只用Q/V列；正交项使用完整18维向量 |
| KD | 从task1起启用；kd_ratio=5，temperature=2；shared前6块输出经同一当前proxy头计算teacher/student soft targets；不额外乘T² |
| KD更新次序 | 每batch先KD backward、shared-B按旧B行范数重分配梯度并optimizer.step；再重新CE前向、CE+orth backward与第二次step |
| Shared-B重分配 | 旧B每行L2范数归一化为均值1；全零退化时缩放=1 |
| 正交项 | orth_ratio=1e-4；当前block weight与每个旧任务weight的绝对归一化内积取均值，范数分母+1e-8 |
| Proxy头 | 每任务新建768→当前新类数CosineLinear，无bias；特征和权重L2归一化，sigma可训练且初始1；weight初始化Uniform(−1/√768,+1/√768) |
| 最终prototype头 | 使用当前任务训练图像、test-mode变换；按类对未归一化特征求均值；冻结、无SGD训练，预测时再归一化 |
| Prototype存储/读取 | fc扩大为已见类数×(768×已见任务数)，旧块复制，新块置0；为当前类计算全部已见分支原型，预测只取各任务类对应的对角分支 |
| 推理 | 输入分别经过全部已见task-specific分支，拼接全部已见类cosine分数；不输入真实task label选分支 |
| 最终head sigma | fc.sigma初始化1，后续复制旧值且保持冻结；不从训练proxy.sigma复制，因此推理尺度保持1 |
| 优化器分组 | 所有requires_grad参数同一SGD组：shared B、当前specific A/B、当前block weights、proxy weight/sigma；没有独立head LR倍率 |
| SGD | momentum=0、weight_decay=0；每个任务重新构造optimizer；task0每batch一次step，后续每batch两次step |
| Scheduler | 每任务CosineAnnealingLR(T_max=epochs)，每epoch末step，eta_min=0，六数据集相同 |
| Backbone dropout | ViT默认drop_rate=attn_drop_rate=drop_path_rate=0；无额外LoRA dropout |
| Precision | 冻结执行路径未使用autocast/GradScaler或half/bfloat转换，模块按默认float32构造；不是mixed precision配置 |

KD细节：forward_kd的teacher分布没有detach；同一个可训练proxy头参与两支。旧shared/backbone冻结，但teacher分支仍可对proxy参数产生梯度。这是冻结代码的实际规则，不能写成固定teacher classifier的标准KL实现。

## Loader与训练路径

- train：当前任务train数据、train变换、shuffle=true。test：全部已见类test数据、test变换、shuffle=false。prototype：仅当前任务train数据、test变换、shuffle=true。
- 三者由同一个_loader构造，batch128、num_workers=train_num_workers=2、pin_memory=true；drop_last没有显式传入，已从运行环境PyTorch DataLoader源码确认默认false；persistent_workers默认false。配置eval_num_workers=2并未由此_loader单独读取。
- 所有CL数据使用统一DataManager的变换构造，无CL专属输入归一化。Prototype故意使用test-mode变换构建训练集均值，属于head机制。
- 配置中optimizer_type=sgd、scheduler=cosine与实际硬编码一致；修改这两个字符串本身不会改变CL的优化器实现。
- run_job还对SGD梯度/更新后参数做有限检查，CL learner再检查loss/grad和任务末参数；这些检查不裁剪、不缩放正常梯度，增加运行开销。

## 相对Vanilla：相同字段与非方法差异

| 项目 | CL-LoRA（冻结运行证据） | Vanilla（当前所选日志/现存源码） | 结论 |
|---|---|---|---|
| checkpoint | augreg2_in21k_ft_in1k，revision及文件SHA固定 | 日志相同hub ID，历史revision/SHA未绑定 | 名称一致，不是逐字节已证同权重 |
| rank/batch/LR/epochs/seed/task流 | 如上表，rank16/b128/seed0 | 所选六格日志同这些字段 | 共同预算字段匹配 |
| cosine最低LR | eta_min=0 | 现存Flatness_CV seqlora.py默认min_lr=.1×LR；冻结CopyFlatCV seqlora默认亦同；旧日志未打印min_lr | 现存代码显示非方法scheduler差异，历史值尚无逐运行见证 |
| workers | 固定2 | 现存seqlora默认8；旧所选日志不含train_num_workers | 执行默认差异；不能反向保证旧运行实际8 |
| pin_memory / persistent_workers | 全部六数据集true / false；裸DataLoader不继承idata.loader_options | 现存Aircraft数据类覆盖为true / true，其余五数据集DataManager默认false / false；历史日志未打印 | Aircraft的pin相同但persistent不同；其余pin默认不同；历史实际值未完全见证 |
| drop_last | 默认false | 现存loader同样默认false | 代码层相同，历史未显式记录 |
| precision | 默认float32，无AMP | 现存SGD路径无AMP；旧日志没有精度字段 | 无证据表明故意改精度，不能做历史位级等价保证 |
| 分类头 | 当前任务cosine proxy训练，最终训练集prototype cosine分类 | IncrementalNet/SimpleLinear，带bias、不做cosine归一化，保留旧class权重 | 方法机制差异，不能把CL解释成单一LoRA正则替换 |
| A/B初始化 | shared正交A冻结/specific Kaiming A，B0 | Vanilla LoRA Kaiming A/B0，单分支持续训练 | 方法机制差异 |
| 数据/Flowers映射 | 当前冻结data manifest与统一loader | 历史图片清单缺失；旧Flowers目录索引映射与当前官方image-ID代码有差异 | 不能声称完全同一图片级协议 |

Vanilla现存split_param_groups在未设置fc_lrate/init_fc_lrate时使用单参数组；本表所选Vanilla日志未记录这两个head倍率字段。CL明确单组，无head倍率。Aircraft的loader覆盖来自[现存iAircraft.loader_options](/data/140-1/users/liying/Flatness_CV/utils/data.py:109)，不能将DataManager的fallback当成所有数据集最终默认。

## 可追溯源文件

- [CL learner：冻结/loader/SGD/KD/orth](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/cllora_aligned.py:22)
- [CL network：cosine head/prototype/推理](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_aligned.py:35)
- [CL初始化与BA缩放](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_archived_vit.py:199)
- [CL block weight和dropout默认](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/backbone/cllora_archived_vit.py:427)
- [Vanilla现存训练和min_lr](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:1318)
- [Vanilla现存loader](/data/140-1/users/liying/Flatness_CV/models_LoRAbasedCL/seqlora.py:229)
- [Vanilla线性head](/data/140-1/users/liying/Flatness_CV/backbone/linears.py:12)
- [Vanilla当前DataManager](/data/140-1/users/liying/Flatness_CV/utils/data_manager.py:221)
- [冻结快照中Vanilla默认最低LR](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code/models_LoRAbasedCL/seqlora.py:288)
- [有限检查wrapper](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/run_job.py:33)
- [aircraft冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_aircraft_s0.json)
- [cars196冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_cars196_s0.json)
- [cub200冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_cub200_s0.json)
- [flowers冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_flowers_s0.json)
- [het5冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_het5_s0.json)
- [pets冻结配置](/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/configs/cllora_pets_s0.json)

未依据当前源码填补旧日志缺失参数，未推断未保存的历史运行行为。
