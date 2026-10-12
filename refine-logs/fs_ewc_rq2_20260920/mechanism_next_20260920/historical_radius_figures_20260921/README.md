# 历史三联图：去除 r² 后的起点 Fisher 迹估计

- [归一化迹估计 PDF](Figure_B_historical_trace_estimate.pdf) · [PNG](Figure_B_historical_trace_estimate.png)
- [结合原始尺度还原起点 raw 代理迹 PDF](Figure_B_historical_raw_anchor_trace.pdf) · [PNG](Figure_B_historical_raw_anchor_trace.png)
- [之前的 Q_F 二次型图](Figure_B_historical_QF.pdf)
- [绘图数值](historical_B_plot_data.csv) · [逐方向数值](historical_B_direction_data.csv) · [估计汇总](TRACE_ESTIMATE_SUMMARY.json)

第一图仍是平均正损伤，第二图仍是训练准确率。所有输入来自同一批历史 ImageNet-A seed0 训练记录，不混入当前确定性 raw-Fisher 补测。模型、原始数据和补测队列均未修改。

## 1. 仅从旧 Q_F 反推的量

对各向同性单位球面方向，E[uu^T]=I/d，d=14,155,776。

$$\widehat{\operatorname{tr}\widetilde F_0}(r)=\frac{2d}{r^2}\overline Q_F(r),\qquad r>0.$$

必须除以 r²，且乘以2d；仅除r²得到的是平均单位方向二次型的一半。r=0时Q_F=0，公式为0/0，因此第三图省略这一点，不能填0或1。前两图仍有81个半径，第三图有80个正半径，每个半径100个方向。

这个估计针对固定、已做迹归一化的起点Fisher。已存迹为SGD0.9999999906867743、GAM0.9999999774154276。因此两个估计围绕1波动。每个半径并没有重新计算Fisher，曲线的细小起伏是有限方向采样与数值误差。第三图纵轴范围0.99–1.01，用于显示这些重建波动。

## 2. 结合外部日志还原旧起点 raw 尺度

旧Task0日志保存了归一化之前的trace：SGD29.875783436000347、GAM26.164499521255493。它们与旧r-sweep显式恢复的Task0 checkpoint对应，见 [原始尺度来源证明](TRACE_RECOVERY_PROVENANCE.md)。

令c0为对应方法日志中的原始迹，近似恢复

$$\widehat{\operatorname{tr}F_{0,\mathrm{raw}}}(r)\simeq c_0\,\frac{2d}{r^2}\overline Q_F(r).$$

原训练还对首次累积Fisher做了接近单位的再次归一化，因此这里是浮点归一化精度下的近似恢复。只靠归一化Q_F无法恢复这个尺度；必须引入额外的c0。

这恢复的是**旧起点的批平均梯度平方代理迹**，不是每个扰动位置的Fisher，也不是当前补测的逐样本empirical Fisher。两方法的尺度差来自已有Task0原始迹，不是r扫描新增发现的优势。原点仍不作为“由Q反推的点”补入。

## 3. 训练方向合并后的结果

各方法仅使用train前缀的80×100=8,000个方向，未混入test前缀。下述标准误只反映方向Monte Carlo采样，不反映训练seed或数据集不确定性。

| 方法 | 归一化迹估计 | 方向标准误 | 还原raw起点代理迹 | 方向标准误 |
|---|---:|---:|---:|---:|
| EWCLoRA | 1.000089513 | 0.000102610 | 29.878458 | 0.003066 |
| FS-EWCLoRA | 1.000068756 | 0.000102006 | 26.166298 | 0.002669 |

## 4. 解读与复现

除掉r²以后，每个半径估计的是同一个F0。曲线接近水平是旧诊断固定使用anchor Fisher的结果，不能说明真实的F(theta0+ru)随r不变。若正文要展示起点敏感度，直接报告已存原始anchor trace通常比重复80次反推更简洁；若要展示邻域敏感度，则仍需扰动位置的重新测量。

`python plot_historical_B.py` 自动生成两个版本；可用 `--trace-mode normalized` 或 `--trace-mode raw-anchor` 单独生成。Times New Roman、一行三列、图内图例；全部测点连接，标记每2.5半径显示一次。方向每个半径重新采样，跨方法配对未核验；训练图像是一轮缓存的随机增强。第三图仅显示均值，不额外绘制训练seed不确定性。

原始文件哈希及逐点核验见NORMALIZED_TRACE_PROVENANCE.json和RAW_ANCHOR_TRACE_PROVENANCE.json；历史Q_F版本的绘图脚本、图注和元数据存放在plot_revisions/before_trace_estimate_20260921T181229/。
