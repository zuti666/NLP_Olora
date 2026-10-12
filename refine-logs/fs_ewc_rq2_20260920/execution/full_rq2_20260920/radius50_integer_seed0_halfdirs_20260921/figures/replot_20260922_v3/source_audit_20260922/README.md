当前B1 v3图：原始数据、补点来源与绘图逻辑审查

已从14份直接供图的seed0原始状态文件重新聚合全部数据。14/14数据源完整，缺失0、失败0；4组新整数补点测量与CPU发布均已完成。此审查没有启动GPU，也没有修改图、实验结果或完成回执。

**文件入口**

- [当前PDF](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/Figure_B1_neighborhood_seed0_radius50_20260922_v3.pdf)
- [全部20,526条相关原始状态的平铺CSV](ALL_SOURCE_MEASUREMENTS.csv)：含训练/测试绝对CE、准确率、原点CE、正/负变化、Fisher、方向标识和原始文件行号。
- [实际用于绘图的11,916条状态](SELECTED_PLOT_MEASUREMENTS.csv)：排除了旧方向50–99和重复原点校验。
- [298行逐半径独立重算表](RECOMPUTED_BY_RADIUS.csv)：214行dense损伤/准确率及84行Fisher。
- [14个原始来源的完整路径、配置与SHA256](SOURCE_MANIFEST.json)
- [训练与诊断配置全集](TRAINING_AND_DIAGNOSTIC_CONFIGS.json)
- [Task0的1566张训练/测试图像清单与256张Fisher子集标记](TASK0_IMAGES.csv)
- [可复现本审查的CPU脚本](export_audit.py) · [审查数值检查](AUDIT_CHECKS.json)
- [四子图绘图函数](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/plot_b1.py) · [当前全部512个前三图数值](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/plotted_values.csv) · [当前22行阈值结果](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/tolerance_bounds.csv)

**1. 当前实验配置与锚点**

当前图使用正式 **ImageNet-A** cohort，不是CopyFlatCV的历史ImageNet-R结果。模型为ViT-B/16（vit_base_patch16_224），Q/V LoRA rank=10。训练配置为10任务、每任务20类、class_shuffle=false；本B1诊断仅使用seed0的Task0结束锚点和各方法自己的固定20类旧任务分类头。

| 配置 | EWCLoRA | FS-EWCLoRA |
|---|---|---|
| 原始方法标识 | sgd | gam |
| Task0训练 | 50 epochs，batch128，初始lr0.04，cosine；momentum0、weight decay0 | 相同 |
| optimizer_type / lambda_flat | sgd / 0 | gam / 1 |
| GAM设置 | 不启用GAM | rho=0.05，norm_rho=0.2，gamma=0.1，beta1/2/3=1 |
| 权重平均/合并 | weight_merge=false | weight_merge=false |
| 后续训练配置中的EWC | lambda=2000，batch_empirical，stored Fisher normalization开启 | 相同 |

训练配置中的归一化Fisher用于训练协议；第三图由独立诊断重新计算raw per-example empirical Fisher，未乘lambda，未沿用训练存储的归一化矩阵。此B1图也没有合入后面新训练的radius-1 intervention模型。

- [sgd训练manifest](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_training/anchor_sgd_seed0/run_manifest.json) · [Task0 checkpoint](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_training/anchor_sgd_seed0/checkpoints/task_00.pt)
- [gam训练manifest](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_training/anchor_gam_seed0/run_manifest.json) · [Task0 checkpoint](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_training/anchor_gam_seed0/checkpoints/task_00.pt)

Task0训练样本1259张，测试307张；第三图只使用固定的256张训练子集，按类别比例分配、subset_seed=20260920。每张图的路径、标签、SHA256和是否属于Fisher子集已导出。诊断预处理为PIL RGB→Resize256（bicubic）→CenterCrop224→ToTensor；按照源协议不加输入归一化。模型/梯度FP32，标量归约FP64，TF32关闭。

**2. 原始扫描和后续补点怎样合并**

| 批次 | 损伤/准确率半径 | Fisher半径 | 原始记录数（两方法） | 当前采用状态数 |
|---|---|---|---:|---:|
| formal_results | 0–20，步长0.25；旧100方向仅取前50 | 0、0.5、1、2、5、10、20；16方向 | 16196 | 8196 |
| stage_path_completion_20260921 | 不供给前两图 | 11–19的整数；16方向 | 290 | 288 |
| radius50_extension_20260921 | 25、30、35、40、45、50；旧100方向仅取前50 | 同六个半径；16方向 | 1396 | 792 |
| radius50_integer_seed0_halfdirs_20260921 | 其余20个整数半径；50方向 | 同20个半径；16方向 | 2644 | 2640 |
| 合计 | 每方法107半径 | 每方法42半径 | 20526 | 11916 |

新增20个整数为26、27、28、29、31、32、33、34、36、37、38、39、41、42、43、44、46、47、48、49。21、22、23、24不在上述任一原始文件中；其他目录与旧日志的广泛检索见 [DATA_SEARCH.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/DATA_SEARCH.md)。

旧100方向扫描在所有旧半径都从states.jsonl取direction index 0–49重新求平均；没有把100方向均值接到50方向均值上。Fisher沿用index 0–15。两方法和全部批次的同编号方向哈希一致。每次补测的r=0只校验锚点恢复与CE一致性，绘图不重复计数。

合计排除8600条旧方向50–99记录和10条重复原点记录；这些原始记录仍保留在ALL_SOURCE_MEASUREMENTS.csv，可按selection_reason过滤。其中1314个Fisher测量位置也出现在dense扫描中，但来自分别执行的诊断作业。11916指被采用的状态记录数；按方法×半径×方向去重后的几何位置为10602个。CSV中的空Fisher字段表示该forward-only损失状态没有计算Fisher，不表示Fisher为0。

**14份逐方向原始文件索引**

| 批次 | 方法 | job | 原始/采用状态 | 原始状态文件 |
|---|---|---|---:|---|
| formal | sgd | dense | 8001/4001 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_results/seed0/sgd/dense/states.jsonl) |
| formal | sgd | sparse_train | 97/97 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_results/seed0/sgd/sparse_train/states.jsonl) |
| formal | gam | dense | 8001/4001 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_results/seed0/gam/dense/states.jsonl) |
| formal | gam | sparse_train | 97/97 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/formal_results/seed0/gam/sparse_train/states.jsonl) |
| fisher_11_19 | sgd | sparse_train | 145/144 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/results/seed0/sgd/sparse_train/states.jsonl) |
| fisher_11_19 | gam | sparse_train | 145/144 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/stage_path_completion_20260921/results/seed0/gam/sparse_train/states.jsonl) |
| coarse_25_50 | sgd | dense | 601/300 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_extension_20260921/results/seed0/sgd/dense/states.jsonl) |
| coarse_25_50 | sgd | sparse_train | 97/96 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_extension_20260921/results/seed0/sgd/sparse_train/states.jsonl) |
| coarse_25_50 | gam | dense | 601/300 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_extension_20260921/results/seed0/gam/dense/states.jsonl) |
| coarse_25_50 | gam | sparse_train | 97/96 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_extension_20260921/results/seed0/gam/sparse_train/states.jsonl) |
| integer_25_50 | sgd | dense | 1001/1000 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/results/seed0/sgd/dense/states.jsonl) |
| integer_25_50 | sgd | sparse_train | 321/320 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/results/seed0/sgd/sparse_train/states.jsonl) |
| integer_25_50 | gam | dense | 1001/1000 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/results/seed0/gam/dense/states.jsonl) |
| integer_25_50 | gam | sparse_train | 321/320 | [states.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/results/seed0/gam/sparse_train/states.jsonl) |

每份states.jsonl的同目录manifest.json记录配置、checkpoint、数据、坐标及代码身份；summary.json为原始按方向汇总。SOURCE_MANIFEST.json列出全部三个文件的SHA256。源文件保留train/test loss、accuracy；只有sparse_train记录包含fisher_by_scope.subset.raw_trace。

**3. 半径和方向的定义**

原点W_m是方法m学完Task0后的有效LoRA权重。向24个Q/V有效BA矩阵（12层×Q/V，每个768×768）添加扰动；总坐标数14,155,776。第k个方向由CPU FP32 Gaussian生成，seed=20260920+104729*k，对所有矩阵联合单位化。

\[
W_{m,r,k}=W_m+r u_k,\qquad \|u_k\|_2=1,\qquad r=\sqrt{\sum_{\ell,c\in\{q,v\}}\|\Delta W_{\ell,c}\|_F^2}.
\]

每个点从锚点重新出发；不是逐半径累积更新。原始记录保存实际norm，且每点结束校验精确恢复。这里r是有效权重的全局范数，不是每层各加r，也不是LoRA因子A/B的范数。**锚点来自rank-10 LoRA训练，但随机扰动处于完整有效权重空间，未要求扰动仍保持rank-10。**

**4. 四个子图的精确计算**

对每个方法m，以自己的固定分类头与自己的锚点作对照：

\[L_{0,m}(W)=\frac{1}{1259}\sum_i\operatorname{CE}(f_m(W,x_i),y_i),\qquad D_{m,k}(r)=L_{0,m}(W_m+r u_k)-L_{0,m}(W_m).\]

① 第一图：\(G_m(r)=\frac1{50}\sum_{k=0}^{49}[D_{m,k}(r)]_+\)。先对训练样本平均CE，再减锚点CE，再取正部，最后对方向平均。既不是逐样本正损伤平均，也不是先平均方向再取正部。

② 第二图：\(A_m(r)=\frac1{50}\sum_{k=0}^{49}100\cdot\frac{\#\text{correct}_{m,r,k}}{1259}\)。绘制扰动后绝对准确率%，不绘制准确率下降量。

③ 第三图：在每个扰动点独立计算\(T_m(W)=\frac1{256}\sum_i\|\nabla_W\operatorname{CE}_i(W)\|_2^2\)，再对16个方向求均值。梯度相对于上述有效Q/V坐标，分类头和其余主干固定；是true-label per-example梯度，不是batch平均梯度的平方。minibatch=8只是计算批量，先求各样本梯度平方后平均。没有乘EWC lambda、r²，也没有按trace或参数总数归一化。除以256和16是定义中的期望平均。

Fisher源表也保存16方向的G_train和accuracy，这两列不供给前两图；前两图明确使用dense表的50方向数据。Fisher不覆盖全部107个半径，只覆盖42个实际测量点；缺少的点没有用固定Fisher惩罚或插值制造。

④ 第四图：从子图1同一dense dataframe，对半径升序计算前缀最大值：

\[\widehat r_\epsilon=\max\{r_j:\max_{i\le j}G_m(r_i)\le\epsilon\}.\]

等于阈值也通过。取首个大于epsilon的点之前的最后实测半径；不插值。若扫到上限仍不失败，则标记右删失且不绘制对应点。当前11档阈值均有后续实测失败点。epsilon对应平均每张训练样本的正CE增量容许值，不是样本总损失或百分比。r=0只评估一次，不伪造50/16次独立重复。

**5. 绘图脚本与处理顺序**

1. [plot_b1.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/plot_b1.py) 校验发布回执中的源文件和产物哈希，读取dense_training_per_seed.csv与fisher_training_per_seed.csv。
2. [publish_radius50.py](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/publish_radius50.py) 的recompute_prior_dense_prefix、validate_extension等函数从原始states重算旧前50方向与新补点，再合并并检查缺失/重复半径。
3. 绘图按r直接连接测量均值，不做平滑、插值补点或单调拟合。单seed不画跨seed误差条或标准差带。
4. 前两图每方法连接107个实际数值，但只在整数0–20和25–50标记，共47个marker；其余0.25网格点仍在线条顶点中。第三图42个Fisher点全部标记。
5. 第四图从子图1重算11档阈值；全部22个结果已重新核对。只调整0.005刻度文字的对齐方式，刻度和数据的x坐标未移动。
6. 当前样式为Times New Roman、7.6×1.85英寸、1×4；颜色SGD蓝/GAM橙，图例映射为EWCLoRA/FS-EWCLoRA。前三图marker=1.6pt、edge=.35pt；图四marker=2.7pt。x/y范围按数据设置，不裁去Fisher尖峰。

**6. 核查结果与应明确的限制**

- 重算214个损伤/准确率半径行及84个Fisher半径行，核对512个当前曲线数值与22个阈值结果；最大绝对浮点差为 3.64e-12，在校验容差内。150个发布源文件和12个发布产物哈希均一致。
- 第一图当前纵轴简写为\(\overline{D_0(\theta,\Delta)}\)，实际统计是平均正损伤G。若D_0在论文中定义为有符号损失差，这个简写不能直接当作有符号均值；必须在符号定义或caption保留正部约定。当前caption已说明positive change。本次审查没有改变用户此前指定的轴标签。
- 数值例：r=.25时，EWCLoRA有符号方向均值为−1.88947e−6，当前图值为2.88991e−5；FS-EWCLoRA分别为6.44692e−7和2.25949e−5。因此两个统计确实不同。
- EWCLoRA的r17 Fisher均值约6380.28，由direction14较高的68312.66拉升；该有限原始测量保留，未平滑或删点。其重复运行稳定性未在本次审查中测试。
- 当前为训练侧seed0诊断；不是三seed均值，也不能据此推断测试集保护区域。原始测试指标已导出，但不进入当前图。
- 保存的原始粒度是“每个半径×方向下的聚合CE、accuracy和trace”。生产collector的individual=False，未逐图保存梯度、完整Fisher矩阵或每个样本的trace；可从记录重算图中方向统计，不能仅靠这些记录还原每张图的梯度向量。
- 自动生成的图validation JSON沿用pending visual review字段，但单独[VISUAL_REVIEW.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v3/VISUAL_REVIEW.json)已记录完成的目视复核。这是状态字段未同步，不是数据计算失败。

本次只核查和导出数据，不增加实验、不覆盖现有图。若后续需要消除20附近容忍半径的平台分辨率限制，应补测同配置随机方向r=21、22、23、24；当前记录不能插值替代这些测量。

独立来源审查亦通过： [RAW_LINEAGE_REVIEW.md](RAW_LINEAGE_REVIEW.md) · [逐项机器审查记录](raw_lineage_agent.json)。该审查额外复核了全部308条含重复原点的来源半径汇总（train/test损伤、accuracy、Fisher mean/Q95/max），与原始记录一致。历史发布器的三seed验证计数不作为当前单seed图的样本量。
