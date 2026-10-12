# RQ2：旧 ImageNet-R 主线与已有图的真实来源

用户已要求以真实历史 ImageNet-R two-task 表为主线，并适当扩展。本轮只做CPU图表与来源核查，不新增ImageNet-A训练，不把ImageNet-A曲线重标为ImageNet-R。

| ID | 角色 | 内容与来源 | 本轮动作 |
|---|---|---|---|
| Table R | RQ2主表 | 旧ImageNet-R six-row table，主对照control与setting1，seeds0/1/2 | 保留真实值及batch Fisher定义，配置/溯源报告 |
| Neighborhood | 主图科学设计；目前现成图是ImageNet-A辅助案例 | random train/test damage + rawtrainF + realized-direction bridge | 核对来源、限定结论，不改dataset标签 |
| C stage-aware | 用户指定现有C图的清楚重绘，**ImageNet-A辅助案例** | execution/report_seed0/C_current_FP32_task_trajectory.csv及C_path_summary.csv | 2×2：old-taskaccuracy、signedlossdamage、仅T10路径S、实际r_T；全部实际阶段T=1…10 |
| r(T) | 对C的独立位移图，**ImageNet-A辅助案例** | 相同trajectoryCSV | 画 r_T=||W_(T-1)-W_0||，不是损伤容忍半径；T1为Task0anchor |

不生成缺数据的ImageNet-R十任务loss/r曲线。不把GAM后期下降的loss强制画成单调增加。T定义为已完成任务总数，T=t_code+1；T1只有Task0，T2完成Task1，T10完成Task9。旧head固定20类；轨迹原始loss/testaccuracy为307个旧测试样本；方向积分为train256、Task0→Task9端点直线段。

可复现scripts、CSV、vectorPDF、300dpiPNG和self-containedcaptions置于figures/。同色表示同方法，dataset与seed必须出现在图片内的小型来源注释和caption；不改变原图。根agent会看PNG后安排独立图表复核。
