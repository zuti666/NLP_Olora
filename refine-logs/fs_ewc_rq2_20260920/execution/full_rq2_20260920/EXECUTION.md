# 完整 RQ2：三 seed 队列

用户本次授权：使用空闲 GPU 运行附件中的完整核心实验。范围为 B1/B2/B3；条件项 full-model M4 不包含在本批默认运行中。原 15/15 作业与历史表格均保留。

当前已启动正式队列（PID 1403027）；旧seed0两条50轮无EWC训练及端点评估已完成，补齐表见 [PILOT_RESULTS.md](PILOT_RESULTS.md)。该pilot不属于正式三seed汇总。

实时状态以 [正式队列](formal_queue/STATUS.md) 和 [旧 seed0 配对补全](pilot_queue_paired/STATUS.md) 为准；计划文件不是完成凭据。三 seed 表仅在所有必要收据、随机配对和数值检查通过后生成。

| 工作 | 数量 | 依赖与输出 |
|---|---:|---|
| 新 Task0 起点：SGD/GAM × seed0/1/2 | 6 个训练作业 | 从同一冻结预训练 backbone 重新初始化 LoRA/head |
| Task1…9 序列 | 6 个训练作业 | 恢复对应的新 Task0；SGD→SGD 或 GAM→GAM，EWC λ=2000 |
| B3 A/B/D 续训 | 9 个训练作业 | A：SGD起点→SGD λ0；B：GAM起点→SGD λ0；D：GAM起点→SGD λ2000 |
| B3 C | 复用 | 对应 SGD 序列的 task_01.pt，不读取最终 t9 的表格行 |
| 四格端点 | 12 个评估作业 | a11/a21/a22/BWT/平均准确率、旧任务 raw Fisher、有效权重位移 |
| B1/B2 | 54 个诊断作业 | 每 seed、每方法 9 项，详见下文 |
| 最终报告 | 1 个 CPU 作业 | 12 条 B3 结果、三 seed 均值±样本标准差、B1/B2覆盖与全部原始路径 |

总计 88 个正式作业、69 个新训练任务段，每段 50 epochs；另有 2 条历史 seed0 λ0 补全训练及端点评估。新训练所有分支统一 workers4、FP32、关闭 TF32、每任务固定随机协议。历史 seed0 pilot 使用 workers8，与旧 C/D 的采样及增强协议一致；不混入新三 seed 的统计。

GPU0/1 优先诊断；GPU2/3/4/5/6 承接训练。诊断可领取其他空闲卡。每次启动检查显存、利用率和计算进程，并非阻塞取得全局 UUID 锁；锁继承给 worker，防止调度器重启造成重复占卡。进程身份使用 PID＋启动 tick＋boot ID。所有独立任务按真实依赖并行，不要求前一类全部完成才启动后一类。

正式定义与 [formal_plan.json](formal_plan.json) 及 [三个 seed 配置](configs/) 一致：

- B1 dense：r=0:0.25:20，100 个公共方向，8001 个唯一状态，完整 train/test loss，只前向。
- B1 raw Fisher：train256 的 7 半径×16方向（97状态）；test307 的 4 半径×16方向（49状态）；完整 train1259 在固定首4方向、4半径（13状态）。
- B2 trajectory：t0…9 完整旧 train1259 Fisher、损伤与有效权重位移；t0/1/5/9 另有完整旧 test307 Fisher。
- B2 path：t1/5/9，完整旧 train1259 的逐样本方向量，9→17→33 嵌套节点，5% 求积稳定标准和同样本 trace bound；物理峰值保留。
- bridge：两个起点交叉使用真实 t9 位移方向；包含两条真实位移半径，以及同半径 K100 公共随机方向。

raw Fisher 是固定旧任务20类头、逐样本真实标签 CE 梯度在有效 Q/V 权重坐标下的 trace；不做保护矩阵的归一化、截断或加正则。表内 anchor/endpoint 使用同一旧 train256 样本；路径和轨迹使用1259样本，报告明确区分。位移同时保存 L2 范数及平方，避免将二者混写。

预检：199份冻结源码一致；SGD/GAM参考中的300个冻结 backbone keys完全相同，已排除训练过的LoRA/head；真实GPU前后向环境检查通过；新诊断 dense 与 sparse_train 的 GPU smoke 均通过，模型转换与恢复最大 logit 误差为0。

最初一次 pilot 用 workers4 启动后在配对审查中被发现不符合旧 C/D；仅该会话自己的进程组已停止，保留在 [被替代目录](pilot_queue/STATUS.md)，不会用于结果。现行 pilot 配置为 [pilot_plan_paired.json](pilot_plan_paired.json)。

训练完成要求：50/50日志、完整数据遍历、有效最终矩阵、checkpoint/Fisher/RNG凭据同时满足；诊断完成要求：退出码0与相应内容收据同时满足。失败会停止新派发并等待已经运行的任务结束；不自动改变batch size或科学设置。恢复要求原代码、配置、计划和已完成收据哈希一致。

最终结果位置：`final_report/RESULTS.md`、`RESULTS.tex`、`individual.csv`、`COMPLETE.json`。尚未产生 `COMPLETE.json` 时不能引用为完成的三 seed 结论。
