# 结果汇总器

`python collect.py`读取同目录`suite.json`、`receipts/ID.json`及suite明确列出的旧结果，仅汇总完成的formal任务；不训练、不导入实验工程、不搜索其他目录。`--root DIR`可指定独立测试目录。suite尚未生成时返回0并提示延期，不改输出。

输出逐文件原子替换：

- `results.json`：逐run身份/配置和指标SHA、CNN下三角、FAA/AAA_stage；35组n、missing_seeds、mean、sample SD/variance；excluded与not_completed。
- `results.csv`：35组统计，顺序为7方法、各方法内Aircraft/Cars/CUB/Flowers/5Datasets。
- `SUMMARY.md`：相同统计及排除原因；n=1的SD写unavailable，n=0无数值，n<3为partial。

校验：完成receipt的id/method/dataset/seed/stage/expected_tasks/config/config_sha256必须与suite一致；若exit_code存在必须为0。原指标文件SHA必须存在且匹配；有config路径时config SHA必须存在且匹配，并核查seed/dataset。任务数固定为前四数据集10、Het5 5。接受完整square或lower-triangular CNN任务矩阵，上三角NaN忽略，已观察下三角必须是0–100范围的有限实数。支持实际`cnn.matrices.final`及显式`cnn_task_accuracies`、`cnn_task_accuracies.final`、`cnn.task_accuracies`；不将top1 curve、NME或原AAA标量代替任务矩阵。若同时存在两种矩阵且下三角冲突则排除。

FAA为最终行任务均值；AAA_stage为每阶段已见任务均值的平均。receipt已有FAA/AAA_stage（或小写别名）必须与重算值一致。之后只对固定seed0/1/2计算样本标准差/方差（ddof=1），全精度计算。未完成、失败、无receipt任务不会进入统计；smoke排除在统计任务之外。

重复完成候选(method,dataset,seed)及重复使用同一指标文件全部排除，不自动挑first/last/best。若suite提供`protocol_sha256`，同组冲突全部排除。完整代码/数据/环境协议是否允许合并由suite准备的来源核查决定；collector不会因相似准确率推断来源等价。旧6项只从suite.reused明确名单读取，指标同样重算。

正常/仅有pending返回0；已保存明确校验排除项返回1；suite结构错误或顶层读取失败返回2并打印错误。原子替换以单文件为单位；跨文件阅读时以results.json内同一generated_at_utc/suite_sha256为快照依据。未对原配置、receipt或原始指标做任何修改。

验证：2026-09-22已用独立临时目录测试实际CNN矩阵schema、三seed均值/SD、n=1的null SD、下三角NaN/形状拒绝、重复候选全部排除、原文件SHA变化、pending不计入；全部通过，临时目录已清理。没有写入真实汇总文件或改原receipt。
