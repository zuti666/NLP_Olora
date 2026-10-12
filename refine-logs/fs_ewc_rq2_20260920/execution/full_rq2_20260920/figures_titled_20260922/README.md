# B1 / B2：添加标题与调整图例

统一保留7.6×1.65英寸、Times New Roman、蓝橙配色、线宽与标记，B2继续无标准差阴影。缩短坐标轴绘图区的高度，为每幅英文标题留出空间；整体画布尺寸未变。

| 图 | 从左到右的标题 |
|---|---|
| B1 | Accuracy → Loss change → Tolerance radius → Trace of Fisher |
| B2 | Accuracy → Loss change → Weight drift → Average Fisher |

B2第四幅方法图例移到左上角。B1第二幅轴标按用户要求简化为 `Loss overline(D_0)`，去掉中括号和下标加号。**这仅简化显示符号，没有改变该曲线的正部计算。**

- **B1**：[PNG](Figure_B1_titled.png) · [PDF](Figure_B1_titled.pdf) · [精确定义与图注](B1_caption.md)
- **B2**：[PNG](Figure_B2_titled.png) · [PDF](Figure_B2_titled.pdf) · [图注](B2_caption.md)

## B1第二幅：实际计算的是平均正损伤

每个方向先在1259张训练图上平均交叉熵，再减去本方法自己的Task0锚点损失，然后取正部，最后平均50个方向：

\[
\Delta_k(r)=L_0(W_0+ru_k)-L_0(W_0),\qquad
G_0(r)=\frac1{50}\sum_k\max\{\Delta_k(r),0\}.
\]

图中简写的 `overline(D_0)` 在当前图注明确定义为上述G_0(r)，不能当作带符号的方向均值。214个实际绘图损伤点均非负。原始逐方向变化确实有负值：

| 方法 | 负变化数量/非零半径方向测量总数 | 最小原始变化 |
|---|---:|---:|
| EWCLoRA | 352/5300 | −0.002283700779 |
| FS-EWCLoRA | 315/5300 | −0.001479085580 |

例如EWCLoRA的r=0.25：带符号方向均值为−1.8894665×10⁻⁶，但图中的平均正损伤为2.8899102×10⁻⁵。两种统计不同；本轮没有把正损伤曲线替换成带符号曲线。B2第二幅仍为有符号测试损失增量，保留负值。

独立核验逐条读取10602条选中dense记录及6份原始states，重算与作图数据最大差约9.91×10⁻¹⁷。实现、原始负值、按半径统计及代码行号见 [B1_LOSS_AUDIT.md](B1_LOSS_AUDIT.md) 和 [JSON](B1_LOSS_AUDIT.json)。

## 复现与检查

[plot_titled.py](plot_titled.py) 重新读取并验证既有测量，所有7份数值CSV均与上一版逐文件SHA一致。B1仍为正式ImageNet-A seed0训练邻域；B2仍为正式三seed均值。此前所有图稿及原始测量未改动，没有GPU作业。

[PDF_TEXT_CHECKS.json](PDF_TEXT_CHECKS.json)核对两个PDF各有4个标题、4个完整方法图例；标题的实际横向顺序正确。生成记录见[GENERATION.json](GENERATION.json)，图面复核见VISUAL_REVIEW.md。LaTeX引用片段见[latex_includes.tex](latex_includes.tex)。
