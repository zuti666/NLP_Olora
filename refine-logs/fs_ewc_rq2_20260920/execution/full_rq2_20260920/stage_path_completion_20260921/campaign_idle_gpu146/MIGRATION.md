# 空闲 GPU1/4/6 接续

用户于2026-09-21明确要求改用其他空闲GPU。原GPU0/3被其他计算占用；已观测GPU1/4/6空闲。

新安排保留27条已完成正式路径，接续9条B2路径和6个B1半径作业。B2优先使用GPU1/4；B1优先使用GPU6，GPU1/4空闲后也可参与B1，最多同时3个作业。B1/B2没有数据依赖，各自完整通过后分别自动汇总绘图。

原科学配置、模型checkpoint、诊断命令、输出路径、数值门及绘图源码均保持字节不变。变化仅为调度GPU/pool、CUDA环境witness的记录路径和调度阶段依赖。所有作业仍使用共享UUID锁、空闲检查及FP32/TF32关闭的CUDA witness。

停止旧控制器前核对PID/start tick/boot ID与命令行，暂停全部线程后再次确认没有活动作业或子进程。旧控制器退出且旧pipeline锁释放后才启动新控制器；旧锁继续由新进程及其子进程持有，阻止重复恢复旧队列。旧状态原文保存在 `original_pipeline_state.json`，旧目录记录迁移指向。

新队列先核验28份完成回执（27条路径+此前等待门）及其原始summary、manifest、states哈希，再放行剩余作业。实时状态见 [STATUS.md](STATUS.md)，迁移与启动证据见 `migration.json` / `launcher.json`。

结果继续写入上层原 `results/`、`figures/`、`report/`；不重复训练、不重新测量已完成路径。
