# B2：紧凑间距与路径 Fisher 公式

本版从 `../figures_titled_20260922/` 的三个 B2 CSV 重算均值，78组均值/样本SD与源表一致，54条路径均验证 `S_path=integral/drift**2`。没有修改数据或原版图。

- 顺序：Accuracy / Loss change / Weight drift / Average Fisher。
- 第四轴：$\overline S_{\mathrm{path},t}=I_t/\delta_t^2$；$\delta_t=\|W_t-W_0\|_2$，$t=T-1$。先逐seed归一化，再平均三个seed。
- 图例保持内部，第四幅左上角，无SD阴影。
- 画布7.6×1.65英寸；Times New Roman；标题10.5pt；wspace=0.38。
- 原颜色、点大小、线宽、数据及坐标范围保持不变。

运行 `python plot_b2_compact.py` 即可从源CSV重绘矢量PDF与300dpi PNG。
