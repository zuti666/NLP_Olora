# 历史 81 点三联图：Loss、Accuracy、平均固定 Fisher 二次型

[PDF](Figure_B_historical_QF.pdf) · [PNG](Figure_B_historical_QF.png) · [绘图数据](historical_B_plot_data.csv) · [逐方向数据](historical_B_direction_data.csv)

使用两个方法的原始 Task0 `endpoint_train` JSONL，0–20、步长0.25，每个半径100个方向。三个指标来自同一历史训练测量记录，所有81个半径均绘制；为清晰起见，每2.5显示一个标记。字体Times New Roman，一行三列，横轴均为Weight drift。没有覆盖当前确定性raw-Fisher图或更改其GPU补测队列。

第一图是平均正损伤，第二图是平均训练准确率，第三图是

$$\overline Q_F(r)=\frac1K\sum_{k=1}^K Q_F(r,u_k),\qquad Q_F(r,u)=\frac12r^2u^\top\widetilde F_0u.$$

F是固定在Task0起点的对角Fisher代理，基于批平均梯度平方估计，并已做迹归一化；没有lambda系数。该图不是扰动位置重新计算的raw empirical Fisher。历史训练输入使用缓存的一次随机增强，各半径重新采方向，跨方法方向配对未核验；与当前确定性16方向补测的口径不同。

## 平均与范数的区别

- Q_F是非负标量，因此先取标量绝对值/范数再求平均与直接平均相同。
- 欧氏扰动范数 `||Delta||_2=r` 已作为横轴；`||u||_2=1`。
- 若指Fisher加权长度，它是 `sqrt(Delta^T F_tilde Delta)=sqrt(2 Q_F)`；先对每方向开平方再平均不等于 `sqrt(2 mean(Q_F))`。数据CSV分别导出 `Fisher_weighted_length_mean` 和 `Fisher_weighted_length_rms`，不与第三图平均Q_F混同。该长度来自正半定矩阵，严格说可以是半范数。
- 若指Fisher矩阵的Frobenius/谱范数，则不是本图的Q_F均值，不能从这个均值直接恢复。
- 再除以 `r^2` 会得到固定起点单位方向敏感度；不再包含扰动幅度效应，且r=0不定义。CSV保留该派生量供检查，没有在图中替换指标。

## 第三图为何几乎重合

各向同性单位球面方向满足 `E[u u^T]=I/d`，故

$$\mathbb E_u[Q_F(r,u)]=\frac{r^2}{2d}\operatorname{tr}\widetilde F_0\approx\frac{r^2}{2d},\quad d=14,155,776.$$

两方法维数相同且trace都约等于1，方向均值因此几乎重合；r²增长也直接来自定义。r=20时，重新从逐方向数组计算：

| 指标 | EWCLoRA | FS-EWCLoRA |
|---|---:|---:|
| 平均正损伤 G | 0.0707041592 | 0.0583422440 |
| 训练准确率 % | 85.7148531 | 88.5115171 |
| 平均 Q_F | 1.4136359823e-5 | 1.4121059735e-5 |

平均Q_F相对差仅0.108232%，理论均值为1.4128508391e-5。第一图G在该半径下降17.484%，但第三图不能作为“扰动邻域内raw Fisher更低”的证据。原JSON的mean字段以float32聚合，与这里按per_u数组重新平均存在末位舍入差，已校验在容差内。

完整原始路径与文件SHA256见PROVENANCE.json；指标来源及历史代码核查见相邻radius_extension_20260921/OLD_FISHER_AUDIT.md。复现：`python plot_historical_B.py`，仅需CPU。
