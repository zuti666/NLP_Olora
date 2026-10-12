# B2：去掉标准差阴影的版本

按用户确认，四个逐任务子图均仅显示三seed均值曲线，去掉标准差阴影。

- [PNG](Figure_B2_stages_unified.png) · [PDF](Figure_B2_stages_unified.pdf)
- [绘图脚本](plot_b2_no_sd.py) · [图注](B2_caption.md)

保留7.6×1.65英寸、Times New Roman、蓝橙配色、标记、四个图内图例、δ、纵轴范围和全部实测阶段。B2_mean_sd.csv、B2_trajectory_seed_data.csv与B2_path_seed_data.csv和父目录逐文件SHA完全一致；SD仍保存在CSV中，只是不在图上绘制。B1及此前带阴影版本均未改动。

绘图对象检查确认四个坐标轴均无填充collection、各有两条均值曲线。版面和字体检查通过；图面复核见VISUAL_REVIEW.md。
