# S3 seed2 四轴超参搜索：复用已有seed2/seed3结果

本轮以用户指定的两份 June29 完整日志为科学配置基准，先检索同目录、同级目录和其他历史结果，再补齐缺失点。按用户最新要求，新增训练只用seed2，已有seed2/seed3都保留统计，不补跑seed3。未按日期限制历史检索。现有机制队列保持运行，本搜索使用独立输出与控制器。

## 历史检索范围和结论

- 原目录 `logs_try/logs_s3_damp_c3_multiseed` 及 `logs_try` 共44个实验子目录：205条已索引日志及对应原矩阵重新核验哈希；222个.log/.out，无新增未索引训练日志。
- `logs / logs_merge / logs-merge`：226份含超参键日志，102条ImageNet-A记录，含镜像；112条重点历史候选逐项记录参数、进度、分支差异。
- 其他9个顶层目录 `logs_ewc / logs_expand / logs_inc_lora / logs_newData / logs_our_joost / logs_run / outputs_logs_paperA / result_summary / summaryResult`：3922个文本产物，376命中，283份ImageNet-A日志联结成169个候选组。136组有有限十任务矩阵、10组训练非有限、4组局部矩阵、19组未解析到完整矩阵；这些组数不是独立实验样本量。
- `outputs_logs` 补查1318个内部.log，最终核对62个ImageNet-A同alias运行块、49个prefix；只新增一个停在初始化的S5 sanity，无遗漏的同中心非中心超参点。证据见 [内部日志补查](audits/audit_historical/outputs_internal_scan.md)。原始checkpoint只关联已有结果，不复制大张量。
- 今天的两个冻结campaign：8个参考超参任务全部完成；18个机制任务在核验时14完成、4运行。所有完成结果核对了原始矩阵、训练500个task/epoch、配置和源码收据。

| 历史目录/实验 | 找到的点 | 可复用范围 |
|---|---|---|
| logs_s5_phase1_lambda_damp | λ5000/10000及α=.001/.05/.1/.2/.3，共12尝试 | 7条仅完成Task0、5条无训练记录；不能当最终成绩 |
| logs_s1h_endpoint_damp_ablation | α=0/.1/.2，FAA50.546/50.562/50.490 | 完整旧阻尼曲线，但λ40000、A关闭、seed0 |
| logs_step2d_damp_batchfisher_ablation | C1/C2/C3/b0完整 | 使用旧路径Fisher；C3打印参数相同也不等于当前运行分支 |
| logs_step2d_weightmerge_* | ω=.05/.1/.2/.3/.4/.5六点完整 | 同时改变Fisher和权重混合，可复用旧联合曲线 |
| logs_s4a_wmergethenfisher_AB / logs_s6_ablation | 4/1条完整 | α、窗口、间隔、uniform等背景不同 |
| logs_merge/s7_merge_epoch_design | seed1六种完整窗口 | 可复用窗口构造实验，ω=1、β=0，不能填当前正ω轴 |
| logs_s2d_delta_r005_ablation / logs_euc_lambda | 额外ΔW或欧氏边界 | 不是GAM训练r，不能混用 |
| logs_run/fullmodel_imageneta_aligned | λ2000/200000、内ρ=.01/.05等 | 全模型约85.8M参数，LR/epoch/Fisher累积不同；非有限λ点已隔离 |
| logs_our_joost | λ与fs_rho等多个旧点 | FisherShape等不同方法和预算，保留为相关历史结果 |

上述旧曲线均保留，不因seed0/1或日期较早而丢弃；截至补查之前，没有发现参考S3背景下可新增抵扣本次四轴缺失点的历史运行。详细证据分别见 [原目录与同级目录审计](audits/audit_historical/HISTORICAL_HPARAM_AUDIT.md)、[其他目录补查](audits/audit_other/README.md)、[新近已完成结果核验](audits/audit_existing/existing_campaign_audit.md)。

## 参考设置与指标

| 执行记录 | seed2 FAA / 样本准确率 | seed3 FAA / 样本准确率 |
|---|---:|---:|
| 用户指定June29原日志 | 53.311 / 60.70 | 51.968 / 59.97 |
| 同科学配置的冻结源码新执行 | 53.419 / 60.70 | 52.006 / 59.97 |

两类都是实际结果，不能互换或择优当成同一次中心。本搜索曲线使用冻结源码新执行中心；两份历史日志保留作原始参照。冻结190个原生文件与已完成8项参考搜索完全相同；机制队列只额外增加控制/观测hook，复用的中心与α点已通过任务边界无扰动验证。历史June29完整源码与环境没有逐运行冻结，不声称位级复现。

固定ImageNet-A、10×20类、每任务50epochs、batch128、ViT-B/16 LoRA rank10、不打乱类别、不回放；首任务LR=.04、后续=.02；GAM λflat=1、ρ=.05、β1/2/3=1、γG=.1；EWC历史γF=.9；快照epoch20/40，CE加权β=3，先融合权重后计算batch-gradient Fisher。中心λ=2000、r=.2、α=.2、ω=.3。

## 新搜索：固定其他设置，每次只变一个参数

| 轴 | 完整取值 | 新增任务数（仅seed2） |
|---|---|---:|
| EWC λ | 0、200、2000、20000 | 0 |
| GAM训练r，固定内ρ=.05 | .1、.2、.5 | 0 |
| damping α | 0、.1、.2、.5、1 | 2 |
| weightmerge ω | .05、.1、.2、.3、.4、.5 | 5 |

seed2共15个唯一配置，另纳入已有seed3中心，共16个配置×seed单元：复用9个、新跑7个。四张曲线各重复展示中心，因此逐轴表22行不是22次训练。不做四维笛卡尔网格。设置与来源登记在 [SEARCH_PLAN.json](SEARCH_PLAN.json)、[复用记录](reused_cells.json)、[全部曲线单元](grid_cells.json)。

ω沿用原配置键 `in_task_fisher_omega`；在已核验的WMergeThenFisher分支中它仅控制权重混合，路径Fisher快照关闭，不是旧Intask联合轴。原生ω=0还会关闭快照CE，因此正ω连续曲线单列；已有保留快照CE的 matched_no_merge seed2/3（FAA53.098/51.935）作为独立零融合控制，不冒充字面ω0。已有A模块关闭、均匀路径、后期路径也保留在审计表。

复用中心、seed2 λ0/200/20000、r.1/.5、α0/1。7个新点仅为seed2的正ω=.05/.1/.2/.4/.5及α=.1/.5；不新增任何seed3任务。所有任务都跑完整10×50，训练超参和数据预算不变。

## 执行与结果

GPU候选为本机0–6，最多7并行，每卡1任务；必须显存<500MiB且利用率<5%，启动前再次核对UUID与占用。其他实验保留运行。每作业独立目录，检查源码/配置SHA256；有非有限Loss、异常退出或矩阵不完整时停止发放后续任务并保留失败证据。

复用既有 `Pilot_new_local` 环境，环境spec hash与已通过的CUDA witness一致，8次完整参考训练及本日机制训练已实际使用，未安装或变更依赖。本轮只变标量超参，不改变训练实现。科学配置的单轴性和9份复用收据已由独立代理复核；原两seed草案在启动前缩减，保留归档，未启动过seed3补点。

- [新任务进度](EXPERIMENT_TRACKER.md)
- [全部轴表CSV](HYPERPARAM_RESULTS.csv)
- [逐seed与配对统计](HYPERPARAM_RESULTS.json)
- [已有seed2/seed3统计（含历史中心与融合对照）](EXISTING_SEED23_STATISTICS.json)
- [可读结果表](HYPERPARAM_RESULTS.md)

队列结束后自动汇总；中途可执行 `python summarize_search.py` 更新复用与新结果合并表。完整点才进入最终统计；均值纳入每点已有的完整seed2/3并显示实际n；n=1时样本标准差记为N/A，n=2时计算样本标准差。增量始终对同seed中心配对，不能将只有seed2的点直接减去中心的两seed均值。该固定网格用于探索性敏感性分析，测试集最高分不被包装成预先指定的确认性配置。
