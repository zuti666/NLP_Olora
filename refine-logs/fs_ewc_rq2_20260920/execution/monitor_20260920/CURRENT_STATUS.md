# FS-EWC RQ2 状态快照

检查时间（马德里）：2026-09-20T17:30:27.312227+02:00
通过 10/15；运行 2；失败 0；排队 3
恢复队列PID：1341023；总状态：running

原邻域测量因参数 direction 重名中断；已修复，重新通过sanity并确认邻域实际采样越过原失败点。两条50epoch训练资产保留，日志显示 Already complete，未重新训练。

| 起点，后续均SGD+EWC | a11 | a21 | a22 | 两任务平均 | test local D0 | raw trace T1 |
|---|---:|---:|---:|---:|---:|---:|
| sgd | 85.67 | 84.36 | 74.80 | 79.58 | -0.019246 | 2008.3879 |
| gam | 87.62 | 84.04 | 72.05 | 78.05 | -0.013388 | 940.4125 |

accuracy单位为%；a21/a22为40个已见类别分类，local D0用固定20类旧任务head。a11采用当前FP32起点复评；训练随机配对验证通过。单seed结果，邻域/路径未完成前不作完整机制结论。

[实时状态](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/campaign_seed0_dualline_recovery1/STATUS.md)

旧十任务链t9路径测量已通过：SGD S_path=0.3680576，GAM S_path=0.01479309；均9节点，求积收敛与数值界检查通过。这是旧SGD→SGD/GAM→GAM序列，区别于新A续训。邻域与真实方向对照仍未完成。
