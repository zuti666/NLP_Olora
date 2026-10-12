# S3 超参补查：其他同级目录（2026-09-21）

以两份 seed2/seed3 S3 参考日志为中心，补查本轮其他 agent 未覆盖的 9 个顶层目录，未按修改时间过滤。此处结果是补充扫描，不包括 logs_try、logs、logs_merge、logs-merge、refine-logs。未读取 checkpoint 或 GPU，也未修改原项目。

结论：这些目录存在大量 λ、扰动半径和 Fisher 形状超参实验，但未发现可以直接补入参考 S3 LoRA 10×50、GAM、α=.2、ω=.3 配置单轴曲线的新点。相似字段或文件名不能代替协议核对。

## 扫描范围

|目录|文本产物|关键词命中|ImageNet-A 日志|
|---|---:|---:|---:|
|logs_ewc|11|10|10|
|logs_expand|2|1|0|
|logs_inc_lora|28|2|2|
|logs_newData|40|9|7|
|logs_our_joost|2530|241|240|
|logs_run|92|24|24|
|outputs_logs_paperA|1|0|0|
|result_summary|1207|80|0|
|summaryResult|11|9|0|

合计 3922 个文本产物、376 个命中文件、283 份 ImageNet-A 日志。按 prefix/seed/model 联结控制台与内部日志得到 169 个候选组；该分组不证明重跑之间完全独立，不能当作样本量。

候选组状态：{'complete_console_10tasks': 136, 'no_complete_matrix': 19, 'nonfinite_training': 10, 'partial_matrix': 4}。complete_console_10tasks 只表示日志包含有限的十任务 CNN 矩阵，不表示十任务各 50 epochs。no_complete_matrix 表示本解析格式未获得完整矩阵证据，不自动宣称训练失败。非有限训练单独隔离。

## 最接近但协议不同的已有结果

### 全参数 GAM + WMergeThenFisher

`logs_run/fullmodel_imageneta_aligned` 确有 λ=2000/200000 及 ρ=.01/.05 点。实际可训练参数 85,798,656（首任务分类器创建前），参考 S3 LoRA 为 368,640；LR=.01/.01、γF=1、通常 20 epochs，均不同。20 epochs、快照间隔20只有末端快照，不能当作参考路径融合的 ω 扫描。

|日志|seed|epochs|λ|ρ|任务平均 FAA|状态|
|---|---:|---:|---:|---:|---:|---|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s0_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s0_gpu4.log)|0|20|200000|0.01|23.140|complete_console_10tasks|
|[wmergethenfisher_imageneta_fullmodel_e50_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/wmergethenfisher_imageneta_fullmodel_e50_gpu5.log)|0|50|2000|0.05|11.552|complete_console_10tasks|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_gpu5.log)|0|20|2000|0.01|24.832|complete_console_10tasks|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_s1_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_s1_gpu3.log)|1|20|2000|0.01|24.518|complete_console_10tasks|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_lr001_lam2000_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_lr001_lam2000_gpu5.log)|0|20|2000|0.05|1.827|complete_console_10tasks|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s2_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s2_gpu6.log)|2|20|200000|0.01|无有效最终值|nonfinite_training|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_s2_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam2000_s2_gpu3.log)|2|20|2000|0.01|17.303|complete_console_10tasks|
|[ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s1_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_run/fullmodel_imageneta_aligned/ewc_gam_wmergethenfisher_imageneta_fullmodel_e20_rho001_gamma005_lam200000_s1_gpu5.log)|1|20|200000|0.01|24.567|complete_console_10tasks|

λ200000 seed2 虽打印了十任务矩阵，训练 Loss/CE/GAM/EWC 已出现 NaN，不能作为正常最终结果。

### LoRA FisherShape 分支

`logs_our_joost/imageneta_lora_factorfs_dampnorm` 的模型名为 `as1_normfisher_fishershape`，其 `fs_alpha` 是优化器方向混合强度；阻尼参数为 `ewc_fisher_damping_beta`。它不是参考 S3 的 `ewc_fisher_damping_alpha` 扫描，也没有参考路径权重融合。当前实现构建 FisherShape 优化器（`models_LoRAbasedCL/as1_normfisher_fishershape.py:284`），日志打印的 fs_* 与模型名一致。以下均为20 epochs、seed0的相关背景结果：

|分支背景|变化参数|值|FAA|原日志|
|---|---|---:|---:|---|
|fs_alpha=.2,λ=0,momentum=0|FisherShape fs_rho|0.05|42.480|[factorfs_sameq_noewc_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_noewc_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu4.log)|
|fs_alpha=.2,momentum=0|EWC λ|400000|42.309|[factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam4e5_s0_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam4e5_s0_gpu6.log)|
|fs_alpha=.1,momentum=.9|EWC λ|150000|52.052|[factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam1p5e5_s0_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam1p5e5_s0_gpu4.log)|
|fs_alpha=.2,λ=0,momentum=0|FisherShape fs_rho|0.02|42.425|[factorfs_sameq_noewc_rho002_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu6.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_noewc_rho002_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu6.log)|
|fs_alpha=.1,momentum=.9|EWC λ|400000|51.496|[factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam4e5_s0_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam4e5_s0_gpu5.log)|
|fs_alpha=.2,momentum=0|EWC λ|2000|41.818|[factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2000_s0_gpu4.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2000_s0_gpu4.log)|
|fs_alpha=.2,momentum=0|EWC λ|20000|41.310|[factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e4_s0_gpu1.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e4_s0_gpu1.log)|
|fs_alpha=.2,momentum=0|EWC λ|200000|42.080|[factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e5_s0_gpu1.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e5_s0_gpu1.log)|
|fs_alpha=.1,momentum=.9|EWC λ|100000|51.852|[factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam1e5_s0_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam1e5_s0_gpu3.log)|
|fs_alpha=.1,momentum=.9|EWC λ|200000|52.624|[factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e5_s0_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_c010_mom09_batchfisher_imageneta_t10c20_r10_e20_initlr004_lr002_lam2e5_s0_gpu3.log)|
|fs_alpha=.2,momentum=0|EWC λ|80000|41.617|[factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam8e4_s0_gpu3.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_imageneta_t10c20_r10_e20_initlr004_lr002_lam8e4_s0_gpu3.log)|
|fs_alpha=.2,λ=0,momentum=0|FisherShape fs_rho|0.1|42.014|[factorfs_sameq_noewc_rho010_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu5.log](/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imageneta_lora_factorfs_dampnorm/factorfs_sameq_noewc_rho010_imageneta_t10c20_r10_e20_initlr004_lr002_lam0_s0_gpu5.log)|

这些结果可保留作独立方法/协议的历史背景，不能用于判断参考 S3 λ 或 r 的最佳点。`fs_alpha=0` 文件不等价于无 Fisher 阻尼；实际阻尼 beta 可能仍为 .2。

### 较早 GAM/EWC 目录及汇总文件

`logs_newData` 有六个完整十任务×10 epochs 的 LR 点，固定λ2000、GAM r=.1、无参考权重融合。FAA 41.849–46.850；另一个20 epochs尝试未检出完整矩阵。`logs_inc_lora` 的两个 ImageNet-A 记录是无路径融合的旧 GAM 模型。`logs_ewc` 为 Adam、10/20 epochs、λ=1e7 的旧 EWC，不是参考优化器。

`summaryResult` 70条 ImageNet-A汇总行全部标记 `ewclora_normfisher_gam`，是 CNN/NME 等重复评估记录，不能当作70个新的S3超参实验。汇总表只用于索引，未把其AAA定义直接与本轮阶段均值AAA混用。`result_summary` 的其他命中也未发现参考S3模型日志来源。

## 产物与指标定义

- [完整目录计数及逐日志参数](audit.json)
- [候选组CSV，含差异和关联日志](candidate_runs.csv)
- [全部文本文件路径](files.txt)
- [关键词命中文件](hits.txt)
- [可复现解析脚本](audit.py)

FAA为最终CNN任务行均值；AAA为逐阶段任务行均值再对阶段平均。控制台矩阵每格保留两位小数，本补充报告据此重算；不是原JSON更多小数精度。未来任务上三角 NaN 为结构性缺测，只有已观测三角需要有限；训练loss出现NaN则隔离。原日志 SHA256 已保存在 audit.json。
