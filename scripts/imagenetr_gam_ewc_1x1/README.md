# ImageNet-R：原生 GAM-EWC 与 EWC 两阶段实现

2026-09-20。代码和 CPU 检查已完成；未启动本实验的 GPU 训练或真实端点评估，尚无新实验结果。

## 对照与范围

| 分支 | 第一任务（代码 Task0，20 类） | 第二任务（代码 Task1，新增 20 类） |
|---|---|---|
| EWC | SGD + CE | SGD + CE + EWC |
| GAM-EWC | GAM(CE)，base SGD | GAM(CE) + clean-point EWC gradient，base SGD |

这是完整方法的两阶段比较。它同时改变初始解和后续优化器，不能单独归因于初始解 shaping。若要落实 RQ2 的固定未来优化器对照，另需复用 GAM Task0，增加 GAM→SGD+EWC 分支；当前入口尚未实现这一交叉分支。邻域 sweep、经验容忍半径、路径积分及十任务序列也不属于当前入口的输出。

参考来源：

- `CopyFlatCV/scripts/run_verify_imagenet_r_1x1_lora_ratio_full_twostage.sh`：两阶段数据、EWC 配置和保存/恢复方式。
- `CopyFlatCV/scripts/run_imagenet_r_task1_fqa_lora_factor_adam_gam.sh`：区分 LoRA 因子坐标与有效权重坐标。该文件是诊断包装器，新实现不依赖其父 shell 脚本。
- `CopyFlatCV/optimer_PerturabtionType/gam.py`：实际原生 GAM 更新，未用 Ratio-FS 代替。

## 固定配置

配置文件为 [config.json](config.json)。ImageNet-R、固定类别顺序、ViT-B/16、连续 Q/V LoRA rank10、20+20 类；每任务 50 epochs、batch128、lr .02、cosine、momentum0、weight decay0；训练 seeds 3/4/5，先试 seed3。

EWC λ=1.2e6、gamma=1；训练保护矩阵沿用 batch empirical、每任务 trace 归一化、累计不再次归一化、damping beta=.2。GAM rho=.05、norm_rho=.1、gamma=.1、beta1/2/3=1；无 SGV、无 EWC 梯度投影、无 trajectory averaging。

两个分支的 base optimizer 默认都是 SGD。若需 Adam 对照，应复制配置并把 `comparison_base_optimizer` 改为 `adam`，重新明确两条分支的学习率，使用独立结果目录。

只从配置指定的历史 ImageNet-R 完整 checkpoint 读取冻结的预训练 backbone；所有 A/B 和分类头重新初始化，不读取其训练过的 Ratio-FS adapter。每个 seed 两分支共享初始参数、任务划分及可控数据顺序。源码、配置、初始化、新分类头和 DataLoader 输入均有 hash/receipt。

原生 GAM 使用四次 CE forward/backward。EWC 梯度在扰动前的 clean 权重单独计算，再加到 GAM 合成梯度；它不参与 GAM 的探测。**原生 beta1=beta3=1 时，零扰动极限也是 2gCE**；同学习率不表示已控制相同有效步长。EWC 单独求导不增加图像网络前向。

## 诊断定义

- accuracy：a11 用旧 20 类头；a21/a22 用全部已见 40 类头，BWT=a21−a11。
- damage：完整旧 train/test 集上，固定旧 20 类 local head，纯 CE 的 endpoint−anchor；保留正负号。
- drift：全部有效 Q/V 块的 `sqrt(sum_l ||B_l,1 A_l,1 − B_l,0 A_l,0||_F²)`，排除分类头。适配器跨任务持续训练；初始 Task0 的 BA=0 不代表 Task1 起点的 BA=0。
- raw empirical Fisher：固定、分层选择每任务 256 个训练样本，确定性预处理，先逐样本梯度平方再平均；在有效 Q/V 权重坐标测量，排除分类头。分别输出 old-at-Task0、old-at-Task1、new-at-Task1；不与训练用 normalized Fisher 混淆。
- 两个旧任务端点还输出真实更新方向的完整逐样本二次量；**这不是路径积分**。

端点评估检查完整权重恢复、冻结 backbone/旧头不变、有效权重转换前后 logits 一致，并在少量真实样本上核对 hook 梯度与逐样本 autograd。

## 入口

从仓库根目录执行。默认只打印命令，GPU 4 对应 EWC、GPU 6 对应 GAM-EWC：

```bash
bash scripts/imagenetr_gam_ewc_1x1/run.sh --gpus 4,6 --seeds 3
```

明确添加 `--execute` 才会启动；启动器检查两卡空闲，不创建等待队列。真实运行前可用独立目录做每任务 1 epoch 的端到端检查：

```bash
bash scripts/imagenetr_gam_ewc_1x1/run.sh --gpus 4,6 --seeds 3 \
  --epochs 1 --output-root /data/140-1/users/liying/O-LoRA/refine-logs/imagenetr_gam_ewc_1x1/smoke
```

上述命令同样只打印。正式配置不带 `--epochs`；`--phase all` 默认两阶段训练后评估和汇总。也可以用 `--phase task0`、`--phase task1`、`--phase evaluate` 分别执行；Task1 必须接续同目录本方法自己的已完成 Task0，配置和训练源码须一致。仅支持任务边界接续，不支持任意 epoch 的精确恢复；失败/重跑使用新目录。

默认结果目录：`refine-logs/imagenetr_gam_ewc_1x1/runs/{ewc,gam_ewc}/seedN/`。

| 文件 | 用途 |
|---|---|
| `run_manifest*.json`、`resolved_*.json` | 真实配置、源码版本和运行环境 |
| `checkpoints/task_00.pt`、`task_01.pt` | 两端点完整网络与配套 Fisher 状态 |
| `head_initial_task_*.pt`、`rng_receipts.jsonl` | 分类头实际初始化与数据随机性核验 |
| `evaluation/summary.json` | accuracy、damage、有效位移和 raw Fisher |
| `comparison/per_seed.csv`、`mean_sd.csv`、`comparison.tex` | 位于 output-root 下，配对检查后的汇总 |

`CPU_VALIDATED.json` 只代表模型/数据构造与初始化，不是训练完成标记。当前检查包含 GAM 更新、EWC 加法、LoRA 生命周期与完整状态恢复的 CPU 测试，以及 Fisher 逐样本/microbatch 一致性自检；真实 ImageNet-R GPU 训练及其阶段恢复仍待运行。
