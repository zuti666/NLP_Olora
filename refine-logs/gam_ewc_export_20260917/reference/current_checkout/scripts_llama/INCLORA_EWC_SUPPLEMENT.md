# Llama-1B IncLoRA + EWC 补跑

入口：`scripts_llama/run_inclora_ewc_supplement.py`。默认串行运行缺失的
Order2、Order3；每个顺序从基座开始训练四个任务，阶段末预测所有已学任务。
旧 Order1 和其他已有结果不覆盖。

预览（不占 GPU、不创建输出目录）：

```bash
python3 scripts_llama/run_inclora_ewc_supplement.py \
  --run-dir /tmp/inclora_ewc_documented_dry_run \
  --gpu auto --orders 2 3 --precision fp32 --dry-run
```

运行（将 RUN 替换为本次唯一标识；前台命令适合在 tmux/nohup 中执行）：

```bash
python3 scripts_llama/run_inclora_ewc_supplement.py \
  --run-dir experiment_queue/inclora_ewc_supplement_RUN \
  --gpu auto --orders 2 3 --precision fp32
```

`--gpu auto` 只选择显存低于500 MiB、利用率不超过5%的单张卡；可指定
`--gpu 9`。无空闲卡则退出，不会抢占其他任务。GPU以UUID绑定，两个顺序
串行运行。项目内使用GPU文件锁避免同一启动器重复占卡。

实际训练调用既有环境
`/data/140-1/users/liying/conda_envs/nlora_req/bin/python`，不会安装依赖。
模型为 `/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B`。
NLTK punkt 数据已预置在仓库 `.cache/nltk_data`，runner显式设置`NLTK_DATA`。
训练离线执行，不会在训练阶段下载模型或NLTK数据。

固定实验设置：

- `lora_strategy=ewclora`，冻结历史LoRA，每任务新增rank8；`adamw_hf`。
- 只训练CE+EWC，显式设置 `olora_lamda_1=olora_lamda_2=0`；无GAM/SAM。
- seed/data_seed=1993；1 epoch；第一任务LR1e-3，后续1e-4。
- 单卡BS1、GA40，有效batch40；目标匹配旧Order1的5卡BS1、GA8。
- 名义EWC λ=1，实际CLI λ=0.2。旧/当前训练循环先将CE除以GA再加EWC，
  因此用 `1×8/40` 保持旧EWC/CE相对强度；直接传1会增加5倍。
- Fisher保持BS1、全训练集、raw Fisher、gamma1，无normalization/damping。
- 默认FP32；当前空闲Quadro RTX 6000不支持BF16。因精度、GPU数、随机数流
  不同，结果应标记为补跑，不能声称逐位复现旧BF16运行。

默认先执行两任务smoke（每任务2个更新步、80训练样本、16预测样本、2个
Fisher微批），验证adapter/Fisher链再执行完整实验。smoke和正式结果位于
不同目录，不会混入最终指标。正式训练不设置样本上限。

每阶段自动验证rank增长、历史adapter保持不变、Fisher形状和有限性、
Fisher更新计数以及训练/预测结果。失败后停止整个补跑并保留错误日志。

所有产物位于 `--run-dir`：

- `manifest.json`：配置及与旧结果的差别。
- `state.json`：当前顺序、阶段、worker PID、状态、日志。
- `source/`、`source_sha256.json`：启动时源码快照及hash。
- `commands/`：每阶段实际命令参数。
- `logs/`：每阶段训练及验证日志。
- `smoke/`：仅用于启动验证。
- `full/order_2/`、`full/order_3/`：正式训练、adapter、Fisher和预测结果。
- `final_results.json`：两个顺序完成后的最终准确率。

再次执行时需使用新的 `--run-dir`，避免覆盖失败记录或已完成结果。
