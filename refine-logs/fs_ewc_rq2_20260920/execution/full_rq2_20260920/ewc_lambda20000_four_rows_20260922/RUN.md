# λ=20000：四组 × 三 seed，未归一化 Fisher

2026-09-22。用户明确指定共同 λ=20000；本批不是自动搜索选出的最优值，也不表示已达到旧任务保护门槛。历史扫描结果保留不变，选择来源见 `SELECTION.json`。

| 分支 | 第一任务 → 第二任务 | seeds | 优先 GPU |
|---|---|---|---|
| EWCLoRA | C → C+P | 0/1/2 | 0 |
| FS-EWCLoRA | C+F → C+F+P | 0/1/2 | 1 |
| Task-1 shaping | C+F → C+P | 0/1/2 | 2 |
| Task-2 shaping | C → C+F+P | 0/1/2 | 5 |

允许 GPU0–6 中任一空闲卡，取消排除 GPU0、1 和等待全部 GPU 空闲的旧约束。每 900 秒检查资源，作业结束可立即释放依赖。只使用无计算进程、显存低于 500 MiB、利用率低于 5% 的卡，并持有共享 GPU UUID 排他锁。同分支按 seed0→1→2 执行，四分支并行；指定卡暂忙时允许其它空闲卡接替。不会停止其它训练。

## 数据和科学设置

ImageNet-A、ViT-B/16、rank10 Q/V LoRA，延续同一 adapter。第一/第二任务对应代码 task0/task1。复用正式的六个 Task0 端点和完整状态，保持分类顺序、输入增强和配对 RNG。只训练第二任务 50 epochs，batch128，lr0.02，workers4，FP32，TF32 关闭。

两个 Fisher trace 归一化开关均为 false。每个不同父端点使用独立的原生 batch-empirical fit Fisher；SGD seed0 已有通过原始输入重放核验的 raw Fisher，直接复用，另外五个端点各重新计算一次。绝不从 trace 归一化矩阵逆推或混用其它 seed 的 Fisher。Fisher 阶段不训练或改变父模型权重。

原生 EWC 为 `0.5 * 20000 * sum(F * delta_effective_QV^2)`。SGD 加 CE 与 P；GAM 使用原生 shaping 更新并在方向投影中加入一次 P。记录保护梯度、P/CE、有效 Q/V 位移和 Fisher 加权位移。GAM 方向与 clean CE 梯度分别标注。

必需评估开启：Task1 训练后对 40 个已见类做 CNN/CIL 评估，并记录分组 a21/a22、BWT 和两任务平均；checkpoint 重载校验结果。配置中的 `endpoint_eval=false` 是未使用的附加诊断开关，不会关闭这里的正式终点评估。

## 自动执行顺序

共 24 个队列阶段：6 个 Fisher 准备（1 复用＋5 重算）→4 个 seed0 单 epoch 短测→1 个 CPU 短测验收→12 个正式续训→1 个 CPU 汇总。四个短测必须全部通过：父端点与 raw Fisher 身份、step1 非零 EWC 梯度、GAM 保护只加一次、所有数值有限、终点评估与 checkpoint 重载。短测不能替代正式结果。失败不会被当作完成，数值发散不自动修改 λ。

最终仅在每组 3 seeds 齐全有效时给均值±样本标准差。报告含 a11、B1、a21、a22、BWT、两任务平均、d_QV_cont、raw Fisher 和保护强度诊断。历史归一化 λ2000 与本批结果分开。现有 λ0 控制须再核验才用于保护收益对比；本批不额外增加 λ0 训练。

## 环境和执行命令

本机 `cudahpc40.cvc.uab.es`；复用 `.aris/compute/local.md` 的环境，spec hash `e94a148f`。Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。每个 GPU 作业先运行 `env_witness.py` 做 seeded CUDA 数值检查。

以下命令仅解析队列、检查依赖和文件哈希，不初始化 CUDA 或启动训练，可供独立文档核验：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/plan.json
```

预期 `dry_run=true`、24 stages / 24 jobs。控制器正式入口：

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/plan.json --execute
```

由持久后台进程执行，启动身份写入 `LAUNCH.json`，输出 `controller.log`。不要在已有活动控制器时再次执行。所有结果写在本目录，历史产物保持不变。运行期间不得编辑已冻结代码或 plan；状态见 `campaign/STATUS.md`、`campaign/pipeline_state.json`，最终汇总见 `report/RESULTS.md`。
