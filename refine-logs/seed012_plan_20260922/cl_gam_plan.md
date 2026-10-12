# CL-LoRA 与 GAM-EWCLoRA seed 0/1/2 补齐计划（未启动）

## 追加数量与复用边界

推荐只要求每个 method×dataset 内三种子执行同一固定协议，不在本轮统一不同方法的 RNG / 调度机制。

| 范围 | 可复用 | 需要追加 |
|---|---|---|
| CL-LoRA 五组 | attempt2 正式 seed0，条件复用 | 每组 seed1,2，共10 |
| GAM-EWC Het5 | attempt2 正式 seed0，条件复用 | seed1,2，共2 |
| GAM-EWC Aircraft/Cars/CUB/Flowers | 原历史 seed0 无足够源码/默认值身份，保留作历史参考 | 新冻结协议 seed0,1,2，共12 |

**推荐合计24次**。若仅按历史表的种子缺口，名义下限为20次，但对历史4个GAM组继续声称同协议没有充分依据。若连上述6个正式seed0也不满足复用的数据/权重条件，则10组三种子全部新跑，共30次。

可选省算方案：CUB、Flowers GAM切换成受控 RNG campaign，复用其已有 seed1,2，各补seed0；这会使本分工24→20次，整张表保守99→95次。必须明确更新协议、重新计算中心值，历史74.22/76.47与86.75/91.76不再是该行中心。seed3不可替代指定seed0。

## 正式seed0证据核实

- 冻结源码172个文件当前全部与 `source_frozen.json` 匹配；manifest SHA256：`0babeaf969c8dd4492747e57aea82935083fa9f7248c04ee425f7ca41251eb3f`。
- 6个正式receipt对应config、原始metrics与log哈希全部匹配，全部退出0且任务矩阵完整。
- 五个数据清单文件哈希均匹配；清单冻结的是路径与标签及任务划分，没有图片内容SHA，因此不能声称图片字节历史身份已锁定。准备阶段应从同一冻结DataManager重建清单对比，核对数据文件未被替换。
- CL-LoRA权重当前文件SHA为 `32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b`，与5份正式配置完全匹配。
- GAM-Het5日志只记录同一timm模型ID并经离线cache加载；receipt未绑定revision/weight SHA。复用前核对当时/现有cache定位证据；若无法接受该限度，则单独重跑其seed0（24→25），或者采用全新30次方案。
- 不以新增加图片哈希/权重哈希倒推旧运行已绑定。

## 源码与入口

- CL与GAM-Het5必须使用 `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code` 的既有实际实现，或完全复制后核对所有源码哈希。
- 入口示意（本轮不执行）：在新campaign独立生成suite/configs/receipts，复制适配 `/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/run_job.py`，调用冻结 `src.main.main(["--config", CONFIG, "--override", "device=[0]"])`。
- 使用 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`（既有Python3.11.15、torch2.4.1+cu121），不可改写旧suite、旧config和旧receipt。
- 原run_job固定HERE并校验冻结文件；不能直接在旧目录添加job或给旧启动器重启。新启动壳只更改路径、job清单与种子，保留有限性检测与CUDA-only约束。
- 每份config单独 `seed:[s]`，调度子进程同时设置 `PYTHONHASHSEED=str(s)`；trainer会取列表元素为int，再set_random初始化Python/NumPy/Torch/所有CUDA，cuDNN deterministic=true、benchmark=false。
- 保留原suite环境：HF/TORCH cache、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、OMP/MKL/OPENBLAS_NUM_THREADS=2、CUBLAS_WORKSPACE_CONFIG=:4096:8、FLATNESS_CV_DATA_ROOT=/data/115-2/users/liying/datasets。

## 关键实际行为：不要只看配置flag

- CL-LoRA前6层共享、后6层任务专属；Q/V；rank16；KD=5、temperature=2、orth=1e-4，代理分类头加原型推理。首task每batch一次更新，后续KD与CE/orth两次。
- CL冻结源码 `models_LoRAbasedCL/cllora_aligned.py:42`直接构造DataLoader，workers取train_num_workers=2（train/test/prototype均如此）、pin_memory=true、persistent_workers默认false；不使用DataManager的deterministic_loader_rng。该配置flag即使新增也不能改变它。
- CL cosine eta_min=0（源码:68）；GAM继承seqlora eta_min=0.1×初始LR。为了复用seed0不能统一成相同minLR；若要统一必须新开协议并重跑全部相关seed。
- GAM-Het5 `ewc_preserve_fisher_rng=true`确实生效，日志每个task结束均显示preserve_rng=True。
- **GAM-Het5 config虽写ewc_isolate_clean_rng=true，但attempt2冻结GAM learner完全没有读取该flag，clean梯度计算没有RNG恢复。实际clean隔离未启用。**这是代码核查结论，旧protocol_note写RNG isolation enabled不够精确。
- attempt2冻结DataManager `utils/data_manager.py:418`也不读取deterministic_loader_rng；GAM-Het5实际用全局随机流。计划继续此实际执行才能复用旧seed0，文稿不能声称双RNG隔离。
- 若改用Sep16受控代码实现clean隔离与独立loader generator，GAM-Het5 seed0须重跑；CL还需真实改loader代码。仅改变JSONflag不构成统一协议。

## 明确冻结的训练超参

| 数据集 | 任务 | LR | epochs/task | GAM rho / normrho / gamma |
|---|---|---:|---:|---|
| Aircraft | 10×10 | .05 | 40 | .01 / .1 / .02 |
| Cars | 16+9×20 | .05 | 40 | .05 / .2 / .03 |
| CUB | 10×20 | .01 | 40 | .2 / .2 / .1 |
| Flowers | 12+9×10 | .0025 | 40 | .2 / .2 / .1 |
| Het5 | 200→100→196→102→37 | .02 | 30 | .2 / .2 / .1 |

两方法基础参数：ViT-B/16，rank16，batch128，SGD，momentum=weight_decay=0，cosine，class_shuffle=false，memory_size=memory_per_class=0。CL专属参数在所有数据集相同。

GAM均固定lambda_flat=1、ewc_lambda=2000、ewc_gamma=1、ewc_eta=0、ewc_max_batches=100、fisher_mode=batch_empirical、normalize_fisher=true、normalize_accumulated_fisher=false、norm_eps=1e-12、dual_ascent=false、skip_task_fisher=false、compute_full_fisher=false、damping alpha/beta=0、approximate_fisher_in_task=false、in_task_weight_merge=false、ewc_project_grad=false；GAM base optimizer=SGD，adaptive=false，beta1/2/3=1。

Flowers参数来源须精确标注：当前可读YAML `Flatness_CV/config_exps_paper1_PAC/exp_paperA_cross_dataset/as1_normfisher_lam2000_flowers_t10c10_r16.yaml`确定LR=.0025、40轮、lambda2000，但本身仍未写rho/normrho/gamma。这里明确选冻结 `seqlora.py:80–86` 默认 .2/.2/.1作为新campaign固定值，也是已完成受控Flowers配置使用值；这不是证明旧86.75运行用了这三个参数。

对新跑的四个历史GAM组，冻结attempt2 source并显式保留current-style默认：Fisher preserve=false（源码默认）、clean isolation实际无实现、全局loader；workers=2、persistent=false。这是新冻结协议，不能宣称补旧历史相同环境。Het5保留Fisher preserve=true区别。若父计划选择Fisher preserve=true给四组，也应明确新协议后全3seed运行；不可仅改单个seed。

## 已有耗时和显存（不查询当前GPU，不启动测试）

| 运行 | 历史单次时长h | 峰值allocated GiB |
|---|---:|---:|
| CL-LoRA/aircraft formal seed0 | 1.538 | 12.807 |
| CL-LoRA/cars196 formal seed0 | 1.721 | 12.809 |
| CL-LoRA/cub200 formal seed0 | 1.614 | 12.809 |
| CL-LoRA/flowers formal seed0 | 0.671 | 12.807 |
| CL-LoRA/het5 formal seed0 | 3.364 | 12.801 |
| GAM-EWCLoRA/het5 formal seed0 | 14.635 | 11.155 |
| GAM历史/cars196（log首末时间跨度） | 4.988 | 未记录 |
| GAM历史/aircraft（log首末时间跨度） | 3.071 | 未记录 |
| GAM历史/cub200（log首末时间跨度） | 5.275 | 未记录 |
| GAM历史/flowers（log首末时间跨度） | 1.616 | 未记录 |

formal时长来自receipt起止，含初始化/训练/原型或Fisher/评估；日志跨度不是严格作业walltime。GPU均为Quadro RTX6000 24GiB。显存为torch max_memory_allocated，不能当作nvidia-smi总占用。CL新增10次按历史估计17.82 GPU小时；GAM-Het5新增2次约29.27 GPU小时，合计已知47.09 GPU小时，另需四组GAM三seed。其他协议/设备不能用此数据声明精确加速比。

可选受控CUB每次6.34–6.39h、Flowers2.09–2.11h，峰值约11.14GiB。这只用于该替代协议排期，不是当前historical组精确预计。

训练前条件：只做文件/清单/hash比对并生成独立manifest；授权训练后单独做smoke与finite gates。失败应保留分母，不择优换seed；若改变稳定性超参，应整组重新冻结三seed，不混合不同配置。
