这版先展示真实后续学习，再展示受控邻域诊断。Figure A 对应旧 B2，Figure B 对应旧 B1；本轮仅调整顺序、标题、坐标文字与图注，保留已有数值和测量协议，没有重新训练或评估模型。

| 新图与下载 | 对应旧图 | 从左到右的面板标题 |
|---|---|---|
| Figure A：[PNG](Figure_A_realized_future.png) · [PDF](Figure_A_realized_future.pdf) · [图注与定义](Figure_A_caption.md) | B2 | Effective-Weight Drift → Task-1 Test-Loss Change → Task-1 Test Accuracy → Path Fisher Sensitivity |
| Figure B：[PNG](Figure_B_controlled_neighborhood.png) · [PDF](Figure_B_controlled_neighborhood.pdf) · [图注与定义](Figure_B_caption.md) | B1 | Task-1 Train Accuracy → Mean Positive Loss Increase → Neighborhood Fisher Trace → Measured Tolerance Radius |

Figure A 所有横轴统一为 `Number of learned tasks`。第四幅纵轴文字按要求精确使用 `Path Fisher sensitivity`，完整定义放在图注中。Figure B 前三幅横轴使用 `Perturbation radius d`，第四幅使用 `Allowed loss increase tau`，实际图中显示数学符号 $d$ 与 $\tau$。

两图采用相同的 7.6×1.65 英寸画布（PNG 2280×495）、Times New Roman、10.5pt 两行标题和 `wspace=0.38` 的紧凑间距。颜色、线宽、1.6pt 点标记、纵轴范围和图内 legend 保持一致；路径敏感度及 Fisher trace 的 legend 均在左上角。图 B 使用普通帽符号 $\hat d_\tau$，与图注中的 $\widehat d_\tau$ 表示同一估计量。

复现入口为 [plot_figure_a.py](plot_figure_a.py) 和 [plot_figure_b.py](plot_figure_b.py)，共用 [plot_style.py](plot_style.py)。在当前目录分别执行 `python plot_figure_a.py`、`python plot_figure_b.py` 即可重绘 PDF/PNG。脚本分别核对78组逐任务均值/样本SD、54个路径归一化值，以及22个离散容忍半径；所有图均直接读取已验证的源CSV。

| 区分 | Figure A：真实后续学习 | Figure B：受控邻域 |
|---|---|---|
| 位移 | $d_t=\|\Delta\mathbf w_t\|_2$，$\Delta\mathbf w_t=\mathbf w_t-\mathbf w_1$ | 人为施加的半径 $d$ |
| 损失 | $D_{1,t}^{\mathrm{test}}$：测试损失有符号变化，保留负值 | $\overline D_1^+(d)$：训练损失逐方向先取正部，再平均 |
| 准确率 | $A_{1,t}^{\mathrm{test}}$：Task-1 test accuracy | $\overline A_1^{\mathrm{nbr}}(d)$：Task-1 train accuracy 的方向均值 |
| Fisher | $\overline S_{F,1}^{\mathrm{path}}(t)$：位移归一化的方向路径敏感度 | $\overline R_{F,1}^{\mathrm{nbr}}(d)$：邻域内 raw empirical Fisher trace 的方向均值 |
| 容忍半径 | — | $\widehat d_\tau$：离散实测网格上的经验统计量 |

图中 Task 1–Task 10 对应原始数据的 `task=0,...,9`，展示索引满足 $t=\texttt{task}+1=\texttt{T}$。锚点记作 $\mathbf w_1$。原始 CSV 列名、checkpoint 名称和记录索引均保持原样。训练阶段的 neighborhood radius $r$ 保留方法中的含义，不再用于图中的受控扰动距离。旧版省略正部加号的轴标签已由本轮明确的 $\overline D_1^+(d)$ 定义替代，计算仍使用原来的 `G_train`。

两图均属于 formal ImageNet-A、EWC $\lambda=2000$ 配置；诊断 Fisher 不乘 $\lambda$，也不使用训练时的 Fisher 归一化。坐标空间为 24 个 effective Q/V LoRA 权重矩阵。Figure A 测试损失和准确率使用固定 20 类 Task-1 head 上的 307 个测试样本，路径 Fisher 使用 1,259 个训练样本。三 seed 的 60 个端点、54 条直线 chord 和 502 个积分节点全部保留；每条 chord 使用 9 或 17 个节点。路径敏感度先按每个 seed 自己的位移平方归一化，再取三 seed 算术均值；继续不显示标准差阴影。chord 是锚点至端点的直线，不是优化器实际经过的时间轨迹。

Figure B 保留 seed 0 的 1,259 个训练样本、50 个方向、每方法 107 个损失/准确率半径，以及 256 个训练样本、16 个方向、每方法 42 个 Fisher 半径。$d=0$ 只需评估一次锚点。扰动作用在完整 effective-weight 空间，不要求保持 rank 10。$d=21,22,23,24$ 未测量；$d=17$ 的实测 Fisher 峰值及 $\tau\leq0.1$ 的全部 11 个阈值均保留。容忍半径仅依据已测量网格，不作插值或连续半径保证。

原始绘图数据、字段解释和审计材料保存在 [上一版目录](../figures_titled_20260922/README.md)：Figure A 使用 [B2_trajectory_seed_data.csv](../figures_titled_20260922/B2_trajectory_seed_data.csv)、[B2_path_seed_data.csv](../figures_titled_20260922/B2_path_seed_data.csv) 与 [B2_mean_sd.csv](../figures_titled_20260922/B2_mean_sd.csv)；Figure B 使用 [B1_dense_seed0.csv](../figures_titled_20260922/B1_dense_seed0.csv)、[B1_fisher_seed0.csv](../figures_titled_20260922/B1_fisher_seed0.csv) 与 [B1_tolerance_seed0.csv](../figures_titled_20260922/B1_tolerance_seed0.csv)。旧版图注中的 Task 0、$r$ 和 $\epsilon$ 是历史显示方式，本文档和新图注采用最新显示约定。

[latex_includes.tex](latex_includes.tex) 按 Figure A → Figure B 顺序提供带完整 caption 的 LaTeX 引用片段；导入论文时由 LaTeX 自动确定最终 Figure 编号。
