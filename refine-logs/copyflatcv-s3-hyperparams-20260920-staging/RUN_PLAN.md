# 参考日志的配置与超参实验（2026-09-20）

本轮用户指定GPU4、5、6，执行主机为cudahpc40。以`s3_damp_c3_alpha02_int20_s2_gpu3.log`的历史配置为中心，保留A、batch Fisher和seed2，进行单轴敏感性实验。本轮与已运行的新冻结协议分别建档，避免混合估计器、A设置和随机流规则。

## 参考配置

ImageNet-A 200类、10任务×20类、固定类序；ViT-B/16 Q/V LoRA rank10；batch128；每任务50epochs；SGD，LR首任务.04、后续.02，momentum与weight decay均0；cosine最低LR为初始的.1。

λEWC=2000，Fisher历史衰减=.9，task/accumulated trace normalization开启，damping=.2。GAM强度s=1，内层ρ=.05，第二训练半径r=.2，β1/2/3=1，γG=.1。A在epochs20/40采样，训练CE softmax权重βpath=3、ω=.3；先合并LoRA权重再计算Fisher；分类头取endpoint。

历史seed2的sample Final=60.70%，任务等权FAA=53.311%，AAA=66.700178%，BWT=−6.371111。两种准确率口径分别报告。详细六seed核验见REFERENCE_AUDIT.md/reference_results.json。

## 本轮8项完整训练

| ID | λEWC | s | r | Seed |
|---|---:|---:|---:|---:|
| s3_center_s2 | 2000 | 1 | .2 | 2 |
| s3_lambda0_s2 | 0 | 1 | .2 | 2 |
| s3_lambda200_s2 | 200 | 1 | .2 | 2 |
| s3_lambda20000_s2 | 20000 | 1 | .2 | 2 |
| s3_strength0_s2 | 2000 | 0 | .2 | 2 |
| s3_strength05_s2 | 2000 | .5 | .2 | 2 |
| s3_radius01_s2 | 2000 | 1 | .1 | 2 |
| s3_radius05_s2 | 2000 | 1 | .5 | 2 |

三个切片共享本轮新中心。s=0仍执行GAM probes，λ=0仍执行Fisher估计；这是方向/保护强度对照，不能解读为同等算力最优比较。保持旧随机流规则；不安装新协议runtime_v3。另有两任务×两轮、interval1的验证短测，不纳入正式性能。

冻结源码来自`refine-logs/gam-ewc-paper-gapfill-20260918/attempt2/code`，190个文件全部SHA256匹配。GAM核心和旧LoRA随机前向与June29 git版本相同；原运行缺源码回执，不能保证历史逐位复现。因此新曲线使用新中心，旧六seed只作参考。当前工作树不用于训练。

## 已有实验去重

2026-09-18的15项新协议计划已经在另一台cudahpc15执行。截至本轮核验：9项核心对照+λ0已完成，λ200/20000、r.1/.5在运行，s.5已排队。此次不重复这些任务。其中心是sample_empirical、预算500/batch32、随机流隔离、A关闭；不能和本轮中心合并为同一曲线。

该新协议已完成三seed FAA：native EWC39.524±1.398，probe39.605±1.103，half50.444±.401，raw51.454±.767，+A51.405±.941。详见`GAM-EWC/results/imageneta_execution_20260918/RESULTS.md`。另一个batched Fisher lambda四方法验证计划也已经执行，协议不同。

## 执行与结果约定

GPU4/5/6每卡最多1个作业；每次启动前检查空闲，按UUID绑定。新目录独立保存配置/源码/日志/回执/指标。失败保留记录并暂停新作业；已有运行可完成。仅完整10任务、下三角有限且配置/源码哈希吻合的结果纳入汇总。

原seed2耗时约2.51小时；3卡、8次约3波，粗估8–12小时，依首个实测修订。训练以后台队列继续，当前仅验证启动，不冒充全部完成。结果为单seed探索性敏感性；报告全部点，不以benchmark test重新选默认，不画seed误差条。
