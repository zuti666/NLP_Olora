# B1 四子图样式修订：独立新版本

[查看 PNG](Figure_B1_neighborhood_styled.png) · [矢量 PDF](Figure_B1_neighborhood_styled.pdf) · [英文图注](CAPTION.md)

本次只调整图面样式，以新目录独立保存。原三子图、原四子图及 B2 的图、代码和数据均保留，旧文件 SHA-256 核验未变。输入为此前已经核验的正式 ImageNet-A 三 seed 数据表；本次未重新评估模型，也未修改任何测量值。

| 项目 | 新显示 |
|---|---|
| 四图 EWCLoRA | 蓝色 `#0072B2`，实线，圆点；图例相同 |
| 四图 FS-EWCLoRA | 橙色 `#D55E00`，虚线，菱形；图例相同 |
| 第一图纵轴 | `Loss` 加 $D_t(\boldsymbol\theta,\boldsymbol\Delta)$，去掉正部方括号与下标加号 |
| 前三图横轴 | `Weight drift` 加 $r$ |
| 第三图纵轴 | `Fisher` 加 $\operatorname{tr}F_0^{\mathrm{emp}}$，移除 `Raw` |
| 第四图说明 | 移除 `scan limit` 字样、灰色箭头及上限横线；保留 epsilon=0.01 竖参考线 |

“scan limit” 原来用于说明扰动半径只扫描到 20。其图面文字可以去掉，但数据含义不变：曲线达到 20 是现有扫描上限处的右删失，不能说明两种方法超出扫描范围后的真实边界相同。此说明保留在图注。

第一图的纵轴用损伤量符号作简洁标记；实际曲线仍为每个方向先取正部、再平均的 $G(r)=100^{-1}\sum_k\max(D_k(r),0)$，没有改为有符号均值，也没有改为绝对 CE。第三图仍是未归一化、无 EWC 系数的经验 Fisher trace；删去 `Raw` 只简化显示文字。

字体仍为 Times New Roman，大小 7.6 × 1.85 英寸，四图均使用图内图例。复现命令：

```bash
python plot_b1_styled.py
```

脚本读取并校验上一版本 CSV 与 manifest 的哈希；所有 CSV 以完全相同内容复制到本目录。前三图的 338 项聚合值及第四图均值/标准差重新比对通过。程序还检查四图每个方法的实际颜色、线型、虚线参数、线宽、标记及标记尺寸完全一致。
