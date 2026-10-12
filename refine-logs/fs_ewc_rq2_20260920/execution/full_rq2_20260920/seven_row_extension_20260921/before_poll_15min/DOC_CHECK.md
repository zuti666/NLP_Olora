# 文档独立核验：PASS

核验时间：2026-09-21 16:45:03 UTC。核验者首次接触此补训目录；只读取本目录 RUN.md 与 .aris/compute/local.md 中 Pilot_new@e94a148f 相关条目，不读取训练实现。

在项目根目录按 RUN.md 的“只读/不启动GPU的文档核验命令”原样执行一次，未添加 `--execute`，退出码 0。实际输出与预期一致：

| 字段 | 实际值 |
| --- | --- |
| dry_run | true |
| stages | 10 |
| jobs | 10 |
| max_parallel | 6 |
| 输入文件指纹数 | 218 |

依赖结构已复现：两个 seed0 短测无前置依赖；`smoke_validation` 依赖两个短测；`shaping_only_seed0/1/2` 与 `task2_shaping_seed0/1/2` 均依赖 `smoke_validation`；`seven_row_report` 依赖全部六个正式续训。plan.json 的实际 SHA256 为 `f3caee8133ca463b7bb06f11ce6a5e181fe0e830eca6a1e874254ad1f0f6f208`。完整命令、依赖与 218 项指纹保存在 DOC_CHECK.json。

RUN.md 给出的 Pilot_new 解释器可直接运行该命令，Python 3.11.15、torch 2.4.1+cu121、torchvision 0.19.1+cu121、timm 1.0.9 及环境标签 Pilot_new@e94a148f 与 local.md 既有条目一致。FP32、关闭 matmul/cuDNN TF32、复用既有环境且不修改依赖的说明清楚；正式 GPU 作业的 seeded CUDA witness 被明确放在运行时前置检查。

结论：陌生使用者能够按给定命令复现队列结构，并从两份材料一致地识别所用环境。未发现本次文档核验命令或预期输出有误；无需修改 RUN.md。

本次仅核验文档与 dry-run，不重新计算环境 canonical hash，不重新查询依赖版本，不执行 CUDA witness、短测或训练，不据此声称正式结果有效。命令无 `--execute` 且返回 `dry_run=true`；本次没有另行进行进程或文件系统副作用监控。除 DOC_CHECK.md/json 核验报告外，本核验者未主动修改文件。
