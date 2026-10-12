# 正式 Fisher 补点状态

独立只读核对 stage_path_completion_20260921 的配置、队列与输出目录。此次核对时，r11–19 补点尚无完成结果可加入图。

- 计划属于同一正式 ImageNet-A 三 seed cohort；radius_seed0/1/2 配置与原正式配置的 checkpoint 字典一致，半径计划为 [0,11,12,13,14,15,16,17,18,19]。
- 六个方法×seed半径任务排队中；radii 与 publish_B1 阶段 pending，当前等待 prior 阶段（历史 radius_extension_20260921 队列）。
- 指定的 results/seed0、seed1、seed2 尚不存在，未发现此扩展的 summary.json 或 states.jsonl。因此当前新版保留原正式来源的七个实测 Fisher 半径，没有遗漏已完成扩展点。
- 队列记录了原 plot_b1.py / plot_common.py 的代码指纹；本次只新增独立样式目录，不修改这些文件，也未启动、停止或重排任何 GPU 作业。

状态来源：../stage_path_completion_20260921/campaign/pipeline_state.json（相对于 full_rq2_20260920 根目录定位）。队列会继续变化，此说明仅记录本次核对时的状态。
