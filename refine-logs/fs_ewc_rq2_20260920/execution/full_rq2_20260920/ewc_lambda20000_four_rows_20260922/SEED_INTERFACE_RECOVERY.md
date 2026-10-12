# Seed 运行时字段兼容修复：attempt3

旧原生 trainer 在实际训练入口将 cfg.seed 从单元素列表改为整数。attempt2 四个短测在 raw Fisher 加载前因此停止，均未执行优化器更新。原始失败日志保留 campaign_attempt2/ 和 smoke/attempt2_failed_training/，修改前训练入口保留 attempt2_source/train.py。

修复仅为统一解析两种合法 seed 形式，禁止非整数/空列表/多元素；模型、优化器、数据、Fisher数值、λ、评估协议均不变。六份已完成 raw Fisher 原样复用。短测必须重新全部通过才放行正式续训。

活动入口 plan_attempt3.json；状态 campaign_attempt3/pipeline_state.json；启动记录 LAUNCH_attempt3.json。

CPU dry-run：
```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/ewc_lambda20000_four_rows_20260922/plan_attempt3.json
```
正式后台命令增加 --execute，原 RUN.md 的科学协议及全部空闲GPU授权继续适用。
