# B1：第四图展示五个新增阈值

[PNG](Figure_B1_added_thresholds.png) · [PDF](Figure_B1_added_thresholds.pdf) · [完整十档训练／测试结果](TEN_THRESHOLDS.md)

第四图现仅展示用户指定的五个训练阈值，刻度与数值一一对应；使用未经三位小数截断的逐 seed 原值求均值和样本标准差。前三图保留原有数据和样式，所有旧版本保留。本次没有模型训练或新的 GPU 测量。

| epsilon | EWCLoRA 均值 | FS-EWCLoRA 均值 |
|---:|---:|---:|
| 0.0025 | 5.750 | 6.583 |
| 0.0075 | 9.333 | 10.833 |
| 0.015 | 12.500 | 14.417 |
| 0.03 | 16.667 | 18.917 |
| 0.04 | 18.667 | ≥20（扫描上限） |

第四图的线段仅用于连接这五个选定阈值的汇总点，不作为中间阈值的精确曲线或半径插值估计。误差棒为三个训练 seed 的样本标准差。FS-EWCLoRA 最后一点的三个原始半径均截断在20，图上标注 ≥20，不用截断后的零标准差暗示真实边界没有不确定性。图中没有恢复此前去掉的上限平台、水平线或箭头。

`selected_thresholds_per_seed.csv` 与 `selected_thresholds_mean_sd.csv` 保存完整十档 train/test 数据，内容与上一版本完全相同；第四图实际使用的30个逐seed点及10个汇总点单独保存为 `fourth_panel_per_seed.csv`、`fourth_panel_mean_sd.csv`。epsilon 的定义仍是样本平均旧任务 CE 的方向平均正损伤阈值；训练N=1259，方向K=100，不是CE总和或逐样本上界。

复现：`python plot_b1_added_thresholds.py`。数据来源哈希、旧文件保留核对及各面板实际方法编码见验证JSON；完整图注见 `CAPTION.md`。
