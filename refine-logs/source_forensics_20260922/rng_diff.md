# LoRA forward 的 RNG 差异：源码和 CPU 最小复验

结论：**删除临时 Linear 初始化，在本次相同输入/参数的 CPU 最小案例中没有改变输出或可用梯度，但改变了进程全局 PyTorch CPU RNG 状态。差异从 task_id=0 就存在，也发生在训练模块切到评估状态时。** 这一证据不能换算成历史 FAA/AAA 差值，也不能证明完整多任务训练逐位等价。

## 五个来源的准确边界

`_LoRA_qkv_timm_train.forward` 的旧版本，在循环历史任务之前，无条件创建 Q/V 两组 A/B 临时 `nn.Linear`。4 个随机初始化随后未被用于当前任务数值计算；有历史任务时再把它们的权重替换为历史 A/B。新版用 `F.linear(F.linear(x,A.detach()),B.detach())` 计算历史贡献，保留两次线性运算的顺序。

| 来源 | 文件 SHA256 前缀 | train wrapper 每次 forward 的临时随机初始化 | 独立 eval wrapper |
|---|---|---|---|
| Flatness_CV 当前可读文件 | `6148efe40bdb` | 有，4 个 Linear | 有，4 个 Linear |
| CopyFlatCV Git HEAD | `303ed752ccf6` | 有，4 个 Linear | 有，4 个 Linear |
| Sep18 attempt2 冻结版 | `303ed752ccf6` | 有，4 个 Linear | 有，4 个 Linear |
| Sep16 rngctl 冻结版 | `b4c0233b91bd` | 无，历史项改为 F.linear | 有，4 个 Linear |
| CopyFlatCV 当前 working tree | `995119b1aea7` | 无，历史项改为 F.linear | 有，4 个 Linear |

Git HEAD 为 `456c4a66d34c4f252335b325624160e1625e1d3b`。HEAD 和 Sep18 attempt2 的整个 `backbone/lora.py` 字节相同。Sep16 和 working 的训练类变量名/注释不同，但本次案例数值和 RNG 行为相同。**不能把 working 的新实现外推到 Sep18 正式运行。** 五个来源的独立 `_LoRA_qkv_timm_eval` 类 AST 完全一致（SHA256 `49d7d07a661ee89b8b7ddfc2a215d7bffc0e705b8749f72b9efd3daa34e16f40`），仍然消耗 RNG。

完整路径、字节哈希、AST 哈希及捕获时间见 [rng_source_manifest.json](rng_source_manifest.json)。本目录保存了对应只读来源的独立文本副本，复验使用副本并检查哈希，未修改任一项目源码。

## 复验范围和结果

执行脚本只 AST 提取两个 wrapper 类，以及分类头 `SimpleLinear`，不导入训练入口，不加载数据集或预训练模型。环境是 `Pilot_new` 的 Torch 2.4.1+cu121；所有 tensor 位于 CPU，`CUDA_VISIBLE_DEVICES=''`，末尾验证 CUDA 从未初始化。

共 **60 个 CPU 案例**：5 个来源 × 2 个 wrapper 类 × task_id={0,1} × {train+backward、eval状态+backward、eval+no_grad}。小维度 batch=2、tokens=3、dim=6、rank=2，固定相同参数/输入/损失。为避免该类用 bool 属性覆盖 `.eval` 名称，评估状态通过 `.train(False)` 设置，这与父模块递归切换模式相同。

- 所有输出相对 Git HEAD **逐位相同**，最大绝对差为 0。
- train wrapper 在训练和评估状态下可用的参数梯度、输入梯度、ΔW hook 梯度均逐位相同；历史 adapter 未得到梯度。
- 独立 eval wrapper 的 `forward` 自带 `@torch.no_grad()`，因此它的案例只验证输出和 RNG，不能称作梯度验证。
- 旧 train wrapper 在 task0/task1、train/eval/no_grad 下均推进 CPU RNG；新 train wrapper 在本次案例中均不推进。
- 所有来源的独立 eval wrapper 均推进 CPU RNG。
- 每个会推进 RNG 的案例，推进后的状态都**严格等于单独构造对应 4 个 Linear 后的状态**。这精确定位了 RNG 消耗来源。

在默认 CPU tensor device 下，4 个初始化直接消耗的是全局 **Torch CPU generator**，不会直接调用 Python `random` 或 NumPy RNG。即使输入后来位于 CUDA，源码先在默认设备构造 Linear、再 `.to(x.device)`；复制设备不会随机初始化。因此按当前工程未设置全局默认 CUDA device 的代码，直接消耗仍为 CPU RNG。**本次没有进行 CUDA 运行或 CUDA RNG 实测**，不能把 CPU 数值逐位结论外推到所有 GPU 数值情形。

实际 rank16、dim768 时，每个 block 的这 4 个权重共有 `4×768×16=49152` 个初始化值；12 blocks 一次前向涉及 589824 个值。这是初始化元素计数，不是实测耗时或随机生成器内部计数。

## 对后续随机操作的传播：哪些已经测到

脚本从旧/新 wrapper 各自的 forward 后 RNG 状态出发，独立重置状态后分别执行下列后续操作。task0 案例中实际得到：

| 后续操作 | Git HEAD 旧 forward 后 | working 新 forward 后 |
|---|---|---|
| 无独立 generator 的 DataLoader 前 5 个样本 | 0, 5, 15, 8, 6 | 8, 13, 4, 1, 14 |
| 新 SimpleLinear 权重 SHA 前缀 | `6ea00a746e76` | `3ba2dc6fc97d` |
| RandomCrop 参数 `(top,left,h,w)` | (4,0,24,24) | (5,8,24,24) |
| 随后的水平翻转 uniform draw | 0.998963 | 0.403293 |
| DataLoader worker base-seed 所用的随机抽样 | 2045945648481788546 | 2816148791055203119 |

这些是**轻量 CPU 下游操作实测**，不是五个真实数据集完整训练轨迹。水平翻转用 p=.5 时，上述抽样会产生不同决策。DataLoader 演示采用 num_workers=0；worker base-seed 的生成规则单独实测，真实多 worker 数据流水线未运行。

源码连接也明确：

- `utils/data_manager.py:417` 的 `build_dataloader` 未主动创建独立 generator；本表相关 SeqLoRA 调用传入 shuffle/worker 等，未传 generator。InfLoRA 可读实现直接构造 DataLoader，同样未传 generator。
- 当前 Torch 的 `dataloader.py:601` 在创建 iterator 时执行 `torch.empty(...).random_(generator=loader.generator)`；`sampler.py:156` 在 generator=None 时从全局 CPU RNG 抽取 sampler seed。`_utils/worker.py:224` 令 worker seed 为 base_seed+worker_id。
- 主进程中 forward 改变 RNG 不会即时改写已经运行的 worker 的 RNG；它会影响后续从该全局 generator 抽 seed 的 iterator/worker 创建。是否、何时传播到实际样本次序和增强，还取决于 iterator 生命周期、prefetch 和 persistent_workers 等。
- `seqlora.py:328` 任务开始调用 `update_fc`；`utils/inc_net.py:425` 新建 `SimpleLinear` 后复制旧类别权重。新增类别行保留随机初始化，因此 head 初始化是另一条明确的传播路径。本次测试执行的是这个相同 `SimpleLinear` 类，未执行完整任务扩展训练。

## 方法调用链范围

以下是当前可读源码的调用链，不冒充未保存源码快照的历史运行绑定证明：

| 方法 | 训练相关路径 | 此次 train-wrapper 修改的适用范围 |
|---|---|---|
| Vanilla LoRA / SeqLoRA | `models_LoRAbasedCL/seqlora.py` → `LoRA_ViT_timm` → `_LoRA_qkv_timm_train` | 当前持续训练 adapter 路径；即使 task_id 始终0也会受旧 RNG 消耗影响 |
| EWCLoRA / GAM-EWCLoRA | EWC/GAM learner 继承 SeqLoRA；其有效 ΔW 收集也显式识别 `_LoRA_qkv_timm_train` | 同一训练 wrapper；多次前向/Fisher/评估的 RNG 隔离另需单独计入 |
| InfLoRA | Flatness_CV 可读 `inflora.py:230` 的 `_build_inflora_backbone` → `LoRA_ViT_timm(eval=eval_mode)` | 训练 eval_mode=false 使用 train类；单独重建 eval_mode=true 使用 eval类。不能只看到 SiNet 名称便断言不同 attention 类 |
| OLoRA | `olora.py:162` → `LoRA_ViT_timm(cur_task_index=...,learn_alpha=false,eval=eval_mode)` | 训练共享 train类，专用评估构造可切换eval类 |
| SDLoRA | `sdlora.py:177` → `LoRA_ViT_timm(...,learn_alpha=true,eval=eval_mode)` | 训练共享 train类；`_build_eval_backbone` 明确建立eval类 |
| CL-LoRA aligned | `cllora_aligned.py` → `backbone/cllora_aligned.py` → 独立 archived CL-LoRA attention | 不经过本次修改的 lora.py wrapper；不能把本差异泛化到CL-LoRA |

`LoRA_ViT_timm` 构造参数 `eval=False/True` 决定安装哪个类，而普通父模型 `.eval()` 只改变 training flag、不会把 train wrapper 自动换成 eval wrapper。因此必须区分“train wrapper 在评估状态运行”和“重建出的独立 eval wrapper”。本次两者分别检验。

## 对此前“核心文件没变”的限定和修正

此前 2026-09-22 13:18:47 UTC 的比较对象是 **working 与 Sep18 attempt2 冻结版**。本次约13:46 UTC复读时，这10个文件的 working SHA 与此前记录仍一致，没有证据显示这段时间刚发生改动。

但 working/frozen 相同**不等于 working/Git HEAD 相同**：`ewclora_youyue_fitArchitechture.py` working/frozen 是 `404955e17372e0198b0f3ab814f28dc9ec16bb54fb1590d798d2b1460f2f4334`，HEAD 是 `51d0175c3867546ad45e6f210e66c9d2cd4d87f39554f79556e0aad54738687f`；`src/trainer.py` 也与HEAD不同。EWC 差异包含 Fisher loader source 选择和来源记录逻辑，不能描述整个工作树为 Git HEAD 未改。三方哈希已保存在 manifest 的 `core_hashes`。

这些发现支持“补跑应冻结整套执行源码并记录有效 RNG 语义”，不支持把历史指标变化归因给单个修复，也不支持宣称已经复现了原始训练。

## 复验命令

```bash
CUDA_VISIBLE_DEVICES='' /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python \
  /data/140-1/users/liying/O-LoRA/refine-logs/source_forensics_20260922/verify_rng_diff.py
```

结果：[rng_diff.json](rng_diff.json)；复验脚本：[verify_rng_diff.py](verify_rng_diff.py)。本次未进行 GPU 实验、未训练模型、未修改两项目源码。
