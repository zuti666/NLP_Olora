# FS-EWC RQ2：数据依赖与两线执行

2026-09-20。按用户要求核对现有8阶段、15作业后，采用 GPU0/1 与 GPU2/3 两条执行线。此次只调整调度依赖和物理 GPU 分配；科学参数、数据子集、checkpoint 来源、测量定义及训练 RNG 配对协议保持不变。原科学计划正文保留，当前执行安排由 `campaign_plan.json` 和本文件说明。

## 真实依赖

| 作业 | 必需的已有资产或前置结果 | 不依赖的本批作业 |
|---|---|---|
| sanity | 冻结源码、数据、两方法旧 Task0/Task9 checkpoint、历史复核值 | 其余本批作业 |
| neighborhood | 对应方法 sanity、Task0 原点测量、公共随机方向和固定数据 | bridge、trajectory、path、A 训练 |
| bridge | 对应方法 sanity、Task0 原点测量、两条旧 Task0→Task9 位移 | neighborhood、trajectory、path、A 训练 |
| trajectory | 对应方法 sanity、Task0 原点测量、旧 Task0–Task9 checkpoint | neighborhood、bridge、path、新 A checkpoint |
| path | 对应方法 sanity、Task0 原点测量、旧 Task0/Task9 checkpoint | neighborhood、bridge、trajectory、新 A checkpoint |
| A training | 原 Task0 完整 checkpoint、各自 normalized EWC Fisher、训练数据与协议；调度层要求共同 sanity | 所有 B/C 正式测量输出 |
| A endpoint / table_a | 对应 A 训练 checkpoint/metrics/manifest，sanity 和 Task0 原点测量 | neighborhood、bridge、trajectory、path |
| report | 两方法所有正式测量 receipt、A 随机配对核验、path 数值 gate | — |

依据：`diagnostic.py` 的四个正式 job 都调用 `require_sanity()` 和 `origin()`，之后直接读取既有资产，互不读取其他 job 输出。`train_continuation.py` 的配置和恢复逻辑只指向原 Task0 资产；`endpoint_a.py` 读取新 Task1、训练 manifest/metrics 和 sanity 原点；`postprocess.py` 最后收集全部 receipt。

因此，旧安排中的 path-before-training 是原先全阶段屏障和优先顺序，不是科学数据硬依赖。sanity 是测量正确性的共同前置，仍完整保留；不以结果是否支持研究预期作为开展其他独立实验的条件。

## 当前资源排程

前驱必须是整个 TreeLoRA 队列 `completed`，再确认 GPU0–3 连续空闲60秒。前驱状态文件为 `/data/140-1/users/liying/O-LoRA/experiment_queue/treelora_llama_both_bf16_b64_s42_ddp4_20260920T080615Z/state.json`。

| 位置 | GPU0 / GPU1，分别 SGD / GAM | GPU2 / GPU3，分别 SGD / GAM |
|---|---|---|
| 共同前置 | GPU0/1 witness → 双方法 sanity | 等待共同 sanity 完成 |
| 两线并行 | neighborhood → bridge | GPU2/3 witness → training → table_a → trajectory → path |
| 汇合 | 两方法 bridge 完成 | 两方法 path 完成且数值 gate 通过 |
| 最后 | CPU report 等待两线全部所需 receipt | 同左 |

GPU2/3 witness 分别在两条 training 作业的 `preflight_command` 执行，不额外增加正式作业数。GPU0/1 的 witness 仍在各自 sanity 前执行。

每张卡同一时刻只跑一个本实验作业；阶段内两个方法并行，两个方法都通过后才放行该线下一阶段。`neighborhood→bridge`、`table_a→trajectory→path` 中没有对应的数据生产/消费关系，这些边用于同卡串行和执行顺序。`training→table_a` 则同时是实际数据依赖。共同 sanity 后两条线独立推进，最终汇总等待两线。

## 必须保留的科学门槛

sanity receipt 必须同时满足 `status="complete"` 与 `passed=true`。训练 runner 自身不读取 sanity receipt，因此这条前置由上层调度器保证。

path receipt 必须同时满足 `status="complete"`、`quadrature_converged=true`、`bound_passed_with_2pct_numeric_tolerance=true`。`diagnostic.py::path()` 在求积或数值界检查失败时仍可能正常退出并写 `status="complete"`，不能只检查退出码或 status。两条线并行后，这些 gate 仍约束后继及最终完整报告，但不再阻止先行的 A 训练。

新 A 是普通或 shaped Task0→同一 SGD+EWC 的 Task1 续训；C 使用既有十任务链，其中 GAM 的旧 Task9 来自 GAM→GAM。C 的 `trajectory/path` 不读取新 A Task1，不能把两个分支混为一条训练轨迹。FP32 测量、原点 a11 复评、各自继承的 Fisher 和随机配对审计均保持原协议。

## 并发读写核查

| 位置 | 写入者与并发约束 |
|---|---|
| `results_seed0/{method}/data_manifest.json` | sanity 先创建；后续 `prepare_data()` 只读并逐项比较已有 manifest |
| `results_seed0/{method}/anchor_measurement.json` | sanity 先创建；后续 `origin()` 只读校验，不改共享原点 |
| `results_seed0/{method}/{job}/` | 各 job 独立写 manifest、states、progress、summary；禁止相同 method/job 重复启动 |
| `results_seed0/{method}/table_a/` | A endpoint 独立写自己的 data manifest 和 summary |
| `training/A_{method}_to_sgd_seed0/` | 每方法独立写训练 checkpoint、日志、RNG 收据和完成凭据 |
| 原 `CopyFlatCV` checkpoint 和 `execution/source/` | 本批任务只读；运行时 patch 仅在各自进程内生效 |
| `execution/unused_no_sidecars` | 仅作为诊断 LoRA 构造参数；当前 task_id=0 的构造路径不写 sidecar |
| `report_seed0/` | 等待两线完成后，由单个 CPU 汇总作业写入 |

sanity 通过后，同一方法的 neighborhood 与另卡 A 作业可并行；各进程的模型和 RNG 独立。GPU 物理编号未进入 diagnostic 测量协议 hash，GPU2/3 可读取 GPU0/1 sanity 产生的原点，但仍须通过自身环境 witness。

## 当前状态与归档

主计划继续使用 `campaign_plan.json`，接续入口继续使用 `wait_then_run.py --plan campaign_plan.json --execute`。新状态目录为 `campaign_seed0_dualline_gpu03/`；启动状态及 PID 以其中 `queue_state.json` 和 [QUEUE_STATUS.md](campaign_seed0_dualline_gpu03/QUEUE_STATUS.md) 为准。实际 pipeline 启动后才生成 `pipeline_state.json` 和 `STATUS.md`，当前新实验仍为 pending。

旧等待进程 PID `541721` 已在未启动本批 GPU 实验时安全停止；TreeLoRA 前驱未受影响。旧配置和文档归档在 `campaign_seed0_after_gpu03/before_dualline/`，替换凭据见 [SUPERSEDED_FOR_DUAL_LINE.json](campaign_seed0_after_gpu03/SUPERSEDED_FOR_DUAL_LINE.json)。不复用旧状态目录，也不把配置或排队状态当作完成结果。
