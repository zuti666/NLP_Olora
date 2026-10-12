# 反向探针回加：独立 GPU0 队列

日期：2026-10-03。运行目录：`/data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z`。本目录准备阶段只物化清单与配置；没有冻结或启动训练。后续状态以本目录实际 gate、pins 和 runtime 收据为准。

## 方法和唯一范围

固定 seed2、rank10、rho=.05、FP32、math attention、自然 SGD。每批在同一 theta0 与 RNG R0 下计算 g0，然后仅在 `theta0 - .05*g0/(norm(g0)+1e-12)` 计算 g1_minus。按明确括号顺序计算 `U = g0 + (g0 - g1_minus)`，恢复 theta0 后唯一一次 SGD 更新 `theta0 - lr*U`。成功时只保留 g0 前向后的 RNG R1。没有外层探针；原始配置 `gam_grad_norm_rho=0.0` 为不参与方法的兼容字段，科学元数据 `outer_radius=null`。

| 作业 | 预算 | 依赖 |
|---|---|---|
| `smoke_RB01_Backward_reflected_natural_s2` | 2 tasks × 1 epoch × 1 batch；2 SGD、4训练CE、8隔离重放CE | 无 |
| `RB01_Backward_reflected_natural_s2` | 10 tasks × 50 epochs；2550 SGD、5100训练CE、40隔离重放CE | 短测完整证据验证通过 |

短测原始配置保留10任务、50epoch的正式科学设置，只通过入口 `--max-tasks 2 --max-epochs 1 --max-batches 1` 限制实际预算。两组均 batch128、首任务LR .04、后续 .02，cosine T_max50/min ratio .1，无momentum/weight decay、EWC、Fisher、merge、HVP、投影或最终方向归一化。正式各任务每epoch批数为 `[10,8,8,5,4,3,3,3,4,3]`。所有任务首批另执行两次完整的两探针零更新重放，共4隔离CE，原始tensor与RNG须逐位一致。

队列只含以上两个job，没有新增配对、诊断或参数扫描job。现有P/C/CD活动队列和Local-CD计划不由本队列修改。

## 准备、冻结与文档门禁

`build_suite.py --materialize` 已完成。它拒绝覆盖已有 suite、pins 或已启动队列，默认调用也不会物化任何内容。入口均为本目录 `trainer/reflected_train.py`。审计调度来自 `../g2-central-location-gpu01-20261003T105925Z`，新方法由本目录独立 reflected 实现提供。

必须使用新 `trainer/REFLECTED_CPU_GATE.json`，且其 reflected_step、reflected_train、test_reflected_cpu 的哈希与最终代码一致；复制的旧 G2/HVP gate 仅供历史上下文，不能验收新代码。另需输入资产收据、两条实际 trainer dry-run 收据及 `gates/INDEPENDENT_REVIEW.json` 的passed独立审查。准备齐全后由root执行：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z/build_suite.py --freeze
```

冻结保存所有本地训练与队列Python、配置、环境/源清单、输入资产、CPU证据、入口收据、科学计划与元数据。源清单必须仍为190个同哈希文件。禁止重冻已冻结或已启动的队列。独立代理随后原样执行 `ENVIRONMENT.md` 中的文档命令并保存新 `gates/DOC_CHECK.json`。该动态收据在pins之外，避免循环哈希；实际训练要求它通过。

## 调度与执行

仅允许物理GPU0、最多1个GPU作业；`device:[0]` 为UUID绑定后的逻辑设备0。沿用 `/tmp/olora_gamfs_gpu_locks` 的 `gpu0.lock` 和对应GPU UUID双锁，worker和trainer继承双锁描述符。分配时先fresh查询，取得双锁后再次fresh查询，两个结果均需显存<500MiB、利用率<10%，且UUID一致。忙卡只等候，不预占、不抢占；无失败重试。其他使用同一锁目录的诊断队列可先使用空闲卡，本队列不修改其状态。

每个worker训练前执行独立seed2 CUDA前向/反向witness，验证UUID绑定及TF32关闭。短测仅在实际raw trace、2CE/1SGD、4CE/task重放、源/配置/输入身份、任务矩阵、Fisher/merge/projection为零的证据全部通过后才允许正式作业启动。任意失败停止新启动并等待已运行作业退出。输出必须新建，不复用非空目录。原有队列与计划保持独立。

root完成全部门禁后方可执行下列命令；本准备任务没有执行：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z/run_queue.py
```

状态文件为 `runtime/queue_state.json`，作业日志为 `runtime/jobs/<job>/console.log`，输出为 `outputs/<job>`。`runtime/PAUSE` 仅阻止新启动。`collect_results.py` 生成本目录 `results.json`、`summary.csv` 与 `RESULTS.md`；未通过完成验证的数值只能作为进度。禁止外部通知。
