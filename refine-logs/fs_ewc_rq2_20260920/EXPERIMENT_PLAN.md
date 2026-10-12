# FS-EWC RQ2：服务器核验后的补实验计划

日期：2026-09-20。状态：完成源码、旧结果和部分 checkpoint 的 CPU 审查及旧数据重分析；没有启动训练、GPU 评估或新 Fisher 测量。

本计划以用户提供的三个附件及简化清单为准。接受 GAM 是 FS-EWC 的优化实现，不重新设立“方法身份验证”任务。本文专门安排 shaping 机制诊断；旧主线关闭 averaging，不能混入当前其他 A-on 实验。输出放在独立目录，保留 O-LoRA 原有 NLP 实验计划。

## 1. 决策与论文主张

保留“三个实验块、一张表、两张图”。服务器上 seed0 的 Task0 和 Task1–Task9 checkpoint 均在，B/C 首轮不需要重训。A 的 ImageNet-A、lambda2000、固定后续 SGD 对照尚缺 shaped→SGD 分支，预计只追加一个 Task1；若历史基线与新运行环境无法匹配，则从两个已有 anchor 各续训一个 Task1。

| 主张 | 当前能支持的部分 | 仍缺的最小证据 | 正文位置 |
|---|---|---|---|
| C1：shaping 后的解在诊断邻域内具有较低敏感度、更大经验损伤容忍半径 | seed0 train 随机扰动损伤/半径有利；test 不一致 | 扰动点的 raw per-example Fisher；固定后续规则的 A 对照；训练 seed 重复 | 表 A、图 B |
| C2：真实序列中较低旧任务损伤不能仅用移动更少解释，且伴随较低路径敏感度 | seed0 旧任务损伤小、有效权重位移反而大 | t9 的单位位移路径敏感度；训练 seed 重复 | 图 C |

要排除的解释：不同起点造成的最终 accuracy 误读；新任务没有学好；后续优化器不同导致的初始解归因混淆。A 固定训练规则，但各自 EWC anchor/Fisher 不同，不能宣称保护矩阵已经完全隔离。C 比较全序列训练方案，不单独归因于 Task0 shaping。

本项目不是引入新的 LLM/VLM 组件，不增设 frontier 实验。简化来自复用现有训练和共用离线诊断，不为证明简化另加消融。

## 2. 实际路径与复用边界

以下路径均相对 `/data/140-1/users/liying/CopyFlatCV`。

| 用户整理目录/用途 | 服务器原始资产 | 核验结论 |
|---|---|---|
| LoRA `fqa_r/.../r_fqa_from_task0_shared...` | `logs/sgd_gam_s1c/staged/stage2_r-2/r_fqa_from_task0_shared_lam2000_initlr0p04_lr0p02_e50/fqa_partial/` | SGD/GAM × train/test 共四份 JSONL，81 半径 × 100 方向，含 per_u |
| `task0_traj` / `Fig_T` | `logs/sgd_gam_s1c/staged/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/shell/` | 两种方法 step0–9 的 loss、accuracy、drift 和归一化 Fisher 指标 |
| 两个 Task0 anchor | `outputs_logs/sgd_gam_s1c/stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/{sgd,gam}/seed0/checkpoints/` | `task_00.pt`、Fisher 等资产存在 |
| 逐任务 endpoint | `outputs_logs/sgd_gam_s1c/stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50/{sgd,gam}/T10/seed0/checkpoints/` | 每方法 `task_01.pt` 至 `task_09.pt` 存在 |
| LoRA 1x1 upper bounds | `outputs_logs/verify_imagenet_r_1x1_lora_fixedfs_upper_bounds/` | 独立 task0/task1/joint40 上界，不能作为 sequential A pair |
| full-model ratio two-stage | `outputs_logs/verify_imagenet_r_1x1_fullmodel_ratio_twostage/` | 指定目录仅发现 seed9 一 epoch smoke，不是正式两任务结果 |
| full-model fishershape | `outputs_logs/fishershape_imagenetr_1x1_fullmodel/` | 有 Task0/上界资产；不同数据/参数空间，作为辅助参考 |
| 邻近实际 LoRA two-stage pair | `outputs_logs/verify_imagenet_r_1x1_lora_ratio_full_twostage/` | ImageNet-R、lambda=1.2e6、seeds3/4/5，有 control/factored_s1；不能替代 ImageNet-A 主表 |
| `full-fisherImportant/topF_batch_1pct_cached_g01` | `logs/fullmodel_multifisher_fqa/topF_batch_1pct_cached_g01/` | 指定作业缺 Fisher cache 而失败，未找到该 tag 的 FQA JSON；邻近完成分支不能冒充它 |

详见同目录 `AUDIT_RADIUS.md`、`AUDIT_TASK_ASSETS.md`、`AUDIT_TWO_TASK.md` 及各自 CSV/JSON。用户 Windows 文件夹名不是服务器根目录原名，后续以本表和 manifest 定位。

## 3. 已核验的训练设置

配置入口：`config_Expand/sgd_gam_s1c/base_s1c_normaccum_e50_initlr004_lr002.yaml`，以实际 shell 日志覆盖值为准。

| 项目 | 固定设置 |
|---|---|
| 数据与模型 | ImageNet-A，固定类顺序，10×20 类；ViT-B/16，日志预训练标识 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| LoRA | rank10，12 层 attention Q/V；诊断坐标为 24 个有效权重矩阵，共 14,155,776 标量 |
| 训练 | 每任务50 epoch；Task0 lr=.04，后续 .02；batch128，cosine，momentum=0，weight_decay=0 |
| EWC | lambda2000，gamma=.9，eta=0，任务 Fisher 与累计 Fisher 均 trace-normalized，max_batches100，无 dual/projection |
| GAM | base SGD，beta1/2/3=1，rho=.05，norm_rho=.2，gam_gamma=.1，lambda_flat=1 |
| 关闭项 | approximate_fisher_in_task=false，weight_merge=false；不加新的 averaging、damping 或约束 |
| 当前证据 | seed0；B 的 train=1259、test=307，固定 Task0 local head；旧r-sweep train为缓存的随机增强视图 |

训练用 Fisher 保持历史定义；新增诊断 Fisher 独立计算。更换训练估计器、lambda 或 A-on 中心都构成另一实验，不用于补齐此处缺失 seed。

三个 `r` 必须区分：rank10 是秩；GAM `.05/.2` 是优化探测尺度；图 B 的 `0–20` 是有效权重诊断半径。不能直接比较它们的数值。

## 4. 旧实现的跨 r / 跨 T 逻辑

### 跨 r：同一个 Task0 anchor 上的独立离线扰动

入口 `scripts/run_s1c_stage2_r_fqa_from_task0.sh` → `src.main`/`trainer.py` 恢复 Task0 → `utils/mechanism_eval.py::summarize_radius_fisher_quadratic_error`。通过 `external_delta_w_coordinates` 在有效 Q/V 权重上注入加性扰动；每点还原原模型，测 CE 差。固定的 `omega_W` 仅用于 anchor Fisher quadratic 和 residual。

旧实现每个半径重新抽方向，且未保存可核对的公共方向库；即使 seed 相同，也不能宣布跨方法、跨 split、跨半径配对。原 `per_u.loss_delta` 足够做历史 G/Q95/容忍半径重分析，不能从旧二次型恢复扰动点 Fisher。旧train是一次缓存的随机crop/flip视图；新确定性train协议必须独立标注，历史dense damage与新raw trace只能先作为互补证据，逐点关系使用新测量同步产生的CE。用户的 `stage2_r-full-imagentr/gam_beta_sweep` 尚未得到唯一原始映射，其引用的full-model半径数字不列为本轮已核验结果。

### 跨 T：同一个 Task0 anchor 继续学后续任务

入口 `scripts/run_s1c_stage2_T_fullseq_from_task0.sh`，`TASK_COUNTS=10` 即恢复 Task0 后训练 Task1–9；不同 T 值本来会分别从 Task0 启动。已有 T10 覆盖全部 task endpoint，不要为 T1/T5/T9 再各训一条序列。

两条历史链是 SGD→SGD 和 GAM→GAM。轨迹记录固定 Task0 loss/accuracy、相对 Task0 的累计位移、旧任务动态 Fisher 等；`task0_fisher_normalize=true` 把 trace 固定到约1，现有动态二次型仍是归一化对角代理，不能作为 raw directional path integral。

**CPU checkpoint 核验所得的重要简化**：这条链是同一个 adapter 持续更新；全部20个 checkpoint 的 backbone 非 LoRA 参数和旧 head 前20行与自身Task0逐位相同。主线构造 `learn_alpha=false`，使用 `IdentityScale`，checkpoint无额外scale参数，且恢复时清空历史adapter贡献，因此本链 `W_eff=BA`。每层直接取 `B_t A_t - B_0 A_0`；不累加各 task 的 BA，不插值 A/B。重算位移与日志最大绝对误差1.84e-8。加载模型时仍断言scale恒等；其他变体须按 `W_eff=scale·BA+历史贡献` 定义梯度、半径和路径，不能直接套BA结论。

## 5. 实验 A：固定后续 SGD+EWC 的两任务表

- **目的**：改变继承状态后，在同一后续训练规则下是否减小旧任务损伤，同时保留新任务学习。
- **两行**：普通 Task0→SGD+EWC；shaped Task0→SGD+EWC，Task1 不再 shaping。
- **复用**：主配置 SGD 的 `task_00.pt/task_01.pt` 可作基线资产。尚未发现匹配主配置的 shaped→SGD 完整分支。优先从已有 shaped Task0 续一个 Task1，50 epoch；如果历史运行源码/随机过程无法与续训版本匹配，两个 anchor 各追加一个 Task1，总计两段，不重训 anchor、不补四格。
- **设置**：模型、数据、Task1 lr=.02/batch128/lambda2000/归一化/预算均固定。单独指定 anchor 来源与未来 optimizer，不能直接复用把 `name=gam` 同时绑定恢复目录和 optimizer 的旧循环。恢复各自 Fisher，明确它们不同；两行的新 Task1 head 初始化和数据顺序使用独立、共享的 continuation RNG。
- **实现开关**：主 learner 的 `optimizer_type=sgd` 分支只执行 CE+EWC，已经关闭 shaping；可显式 `lambda_flat=0` 标记。清空/固定 `optimizer_type_by_task` 防止它覆盖 optimizer。复用历史基线且声称严格随机配对，还需要旧Task1 head初始化、batch顺序、增强RNG的恢复/验证证据；本轮未建立该证据。无法建立时，用两个现有anchor同源各续一个Task1完成严格配对。只补一段可以作历史基线对照，但须明确其随机过程未验证配对。
- **指标**：a11、a21、a22；train/test 的自身起点 CE 差 D0；有效权重位移；旧任务 `T0(theta0)` 与 `T0(theta1)` raw trace。accuracy 使用相同 seen-class 评估定义；机制 CE 使用固定旧任务20类 local head，两者表头写清楚。新增 train trace 为固定256样本估计。
- **成功/失败解释**：D0 较小且 a22 没有明显恶化才支持 retention；若仅 a21 较高，表述为最终性能收益。报告实际 a22 差，不用事后调整容许下降阈值把少学新任务解释成成功。
- **已有 ImageNet-R pair**：seeds3/4/5 的 control/factored_s1 平均 a11=89.363/89.467，a21=85.900/85.957，a22=87.820/87.820；自身 accuracy 下降=3.463/3.510 pp。不能把这组接近结果写成 shaping 已明显降低损伤；它提供阶段开关模板和辅助证据。附件引用的 19.885→10.946、87.95→91.82 尚未对应到主配置原始 LoRA 记录。
- **优先级/位置**：MUST，正文表 A；无保护 pair 和完整四格为 CONDITIONAL。

## 6. 实验 B：邻域 damage / raw sensitivity / 经验半径

### B0：历史重分析（本轮 CPU 已做）

`D_k(r)=L0(theta+r*u_k)-L0(theta)`；`G=mean(max(D,0))`；`Q95=quantile(max(D,0),.95)`。容忍半径取从 r0 开始此前所有已测半径均合格的最大网格点，不能取最后一个偶然低于阈值的远点。固定阈值 `{.001,.005,.01,.02,.05}` 沿用附件；这是已有结果的事后重分析，不宣称预注册。

| 诊断 | epsilon | SGD | GAM |
|---|---:|---:|---:|
| train G 容忍半径 | .005 | 5.75 | 7.00 |
| train G 容忍半径 | .010 | 8.25 | 9.50 |
| train Q95 容忍半径 | .010 | 6.25 | 7.75 |
| test G 容忍半径 | .010 | 7.50 | 6.75 |
| test G 容忍半径 | .020 | 11.00 | 10.75 |

train 的优势不能外推到 test。图保留两种 split；半径达到扫描上限仍合格标记 right-censored。方向最大损伤不是连续球上的 worst-case。

### B1：新增稀疏 raw Fisher（MUST）

- 两个 seed0 Task0 anchor；r=`{0,.5,1,2,5,10,20}`；每个非零 r 用16个公共随机方向，r0只测一次。
- 方向在同一有效权重坐标按全向量标准高斯归一化；跨方法/半径/train-test 复用，使用专用 generator，保存 seed、模块顺序、dtype、方向哈希。允许按编号重建，避免把整套大向量常驻显存。
- 旧 train 固定分层256样本，按类样本量近似比例分配并保存 sample IDs/hash；两方法及 B/C 使用同一子集。用确定性 test transforms、eval 模式，无随机增强。全 train/test CE 单独报告；子集 CE 与 Fisher 同步保存。
- raw trace：`T0(theta)=mean_i sum_l ||g_i,l(theta)||²`；梯度针对上述24个有效矩阵，true labels，逐样本。raw 测量不加 clamp floor/damping、不归一化、不含 EWC/GAM loss。
- 这是有效权重 ambient space 的受控诊断；随机扰动可以超出 rank10 单步可实现的更新集合。论文明确此范围，不当作 LoRA 因子空间球或优化器训练半径。
- 正文图 B：(a) G/Q95-r，(b) train raw trace-r，(c) epsilon—容忍半径；16方向以均值为主，Q95/max 仅辅助。
- 支持标准：损伤和 raw sensitivity 的实测趋势共同支持 C1。若 trace 不降，保留损伤结果，停止声称该 trace 机制已成立；不删负结果。train-only Fisher 不能证明 test sensitivity 下降。

### B2：真实方向小扫描（MUST，图 B inset/附图）

`u_sgd=Delta_sgd/||Delta_sgd||`、`u_gam=Delta_gam/||Delta_gam||`，两个方向交叉施加到两个 Task0 anchor。r=`{0,.25,.5,1,1.5,2}`；保存 train/test D、256 train raw trace。各自 head 固定，不跨方法交换 head；这比较方向在各 anchor 上的作用。连接真实约1.2–1.4位移与旧随机 r5–20 的尺度差异。

### B3：directed probe（NICE-TO-HAVE）

仅能直接复用现有 GAM inner-search 时，按相同计算预算得到两个方向，再于两个 anchor 交叉评估 r=`{1,5,10,20}`。方向生成样本与 trace 子集分离。标为 GAM-directed proxy：旧 inner-search 可能在因子坐标/批均梯度 surrogate 上工作，映射到有效权重后不等于最大 raw trace 方向。无需另建复杂二阶搜索，不阻塞 B1/C。

## 7. 实验 C：真实序列与 t9 插值路径

复用每个 task 的 Task0 train/test CE、local accuracy、全 seen-class 性能与累计有效权重位移。新任务 a_tt 和全序列指标作为 plasticity 背景，不能只展示 retention。

已核验 seed0 t9：

| 指标 | SGD+EWC | GAM+EWC |
|---|---:|---:|
| Task0 train local CE 增量 | .188107 | .039556 |
| Task0 test local CE 增量 | .260343 | .034413 |
| Task0 test local accuracy 下降，pp | 5.212 | 2.280 |
| 相对 Task0 有效权重位移范数 | 1.197584 | 1.400473 |
| 最后任务的标准 seen-class accuracy，% | 34.62 | 53.85 |
| 最终各任务 accuracy 宏平均，% | 40.994 | 52.078 |

GAM 位移约大16.9%，损伤更小。这支持“不能仅用移动少解释”，不证明 anchor-only 因果，也不证明 raw path sensitivity 已下降。

新增只必做最终 t9：

`W(s)=W0+s*(W9-W0)`，s=`{0,1/8,...,1}`。固定 T0 local head、backbone、256 train IDs。无需完整 Fisher：

`q(s)=mean_i (sum_l <g_i,l(W(s)), Delta_l>)²`，`I=integral_0^1 q(s) ds`，`S_path=I/||Delta||²`。

必须先把所有矩阵的内积求和再平方，保留跨参数/模块项；不能用 `sum_j F_diag,j*Delta_j²` 替换。路径是实际 endpoint 之间的有效权重直线，不是 SGD/GAM 的逐步训练轨迹。零位移的 S_path 为 NA。

主图 C：(a) 各 task 旧任务损伤，(b) 损伤—位移散点/连线，(c) t9 S_path 对照；local accuracy 可放同图标注或附图。raw endpoint trace 同时记录，不要求每个 task 都重测 trace。

9节点 composite Simpson 与其中5节点结果比较，预定相对差5%为加密触发器（分母至少1e-12）。不稳定时17节点；仍不稳定保留未收敛标记。检查同一256子集、同一local head 的 D_sub²≤I，加上数值误差说明；不得拿全 train/test D 与子集 Fisher 检查该界。求积检查放附录，不当作方法有效性的主结果。

成功是较低 D 与较低 S_path 同时出现；若 S_path 不下降，收窄机制结论。seed0 t1/t5 为 OPTIONAL，不要求所有 seed 做中间路径。

## 8. 最小代码补足与测量检查

优先新增一个独立离线诊断入口，共用 load/坐标/数据/梯度实现，不修改历史训练目标。以下是待实现接口，不是已可运行命令。

| 改动 | 可复用代码 | 必须补的内容 |
|---|---|---|
| endpoint 恢复 | `models_LoRAbasedCL/baseLoRA.py` checkpoint restore；`utils/inc_net.py` | 核对 state key、类数、旧head、配置和数据哈希；载入后禁止触发训练 |
| 有效权重注入 | `utils/mechanism_eval.py::external_delta_w_coordinates`，`backbone/lora.py` additive path | 用 W0 作为基底直接注入 r*u / s*Delta；每点彻底还原 |
| Fisher scalar 测量 | `mechanism_eval.py` 的逐样本/有效权重梯度思路 | 同时返回 loss、raw trace、完整方向 q；microbatch 不改变逐样本定义 |
| 方向和子集 | 原 fixed batches 逻辑 | 显式公共方向 generator、样本清单、class-slice；固定 deterministic transforms |
| A 的追加分支 | `run_s1c_stage2_T_fullseq_from_task0.sh` 的 resume 参数 | 分离 anchor method 与未来 optimizer；只续 Task1；输出到新目录 |
| 结果格式 | 旧 JSONL/CSV 导出 | 保存 split、样本ID、point、coordinate、head、source SHA、梯度定义、是否归一化和数值检查状态 |

不能直接把 `scripts/compute_all_fisher_stage1.py` 改一个 mode 就当成新测量：该文件的 loader 硬编码 ImageNet-R、使用 train transforms，参数集合由 `requires_grad` 决定，且 diagonal 末尾 `clamp_min(1e-8)`。本任务应显式 ImageNet-A、有效权重坐标、eval transforms、无 floor；并直接计算 q，避免构造或截断整张矩阵。

测量 gate（与结果是否有利无关）：

1. CPU 资产/形状/位移检查已完成；接下来优先复现历史test local loss，并在新确定性train/test协议下验证 s0/s1 与直接载入的endpoint一致。旧r-sweep随机增强train loss不要求与新transform数值相同，也不与新trace逐点混配。端点无法重现先修恢复，不重训。
2. 同一点重复测量无随机增强差异；r0扰动量和D为0；||u||=1、||r*u||=r、参数作用域一致。
3. 小样本逐个 backward 验证向量化/分块实现；q非负，trace无归一化/floor；用方向有限差分检查梯度—路径一致性。累加标量用float64，先FP32，不默认AMP。
   现有 `mechanism_eval.py::empirical_fisher_vector_product` 及delta-W版本实际是batch均值梯度外积，不能按函数名直接复用为逐样本F；可复用 `as1_normfisher_gam.py` 的sample_empirical逐样本循环，但绕过上层归一化。
4. 先测一个 anchor 的256样本吞吐和内存，估算GPU时间；本轮没有GPU测速，不给虚构小时数。

环境：旧 shell 默认 `/home-local/liying/.conda/envs/Pilot_new_local/bin/python` 在当前主机不存在；当前默认 python 也无 torch。已找到并用于 CPU 检查的是 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`（torch2.4.1+cu121）。GPU评估前还需验证 timm/torchvision/数据预处理与源配置兼容，不照搬旧解释器路径。

## 9. 执行顺序、预算与停止条件

| 顺序 | 工作包 | 当前状态 | 训练需求/成本 | 完成 gate |
|---|---|---|---|---|
| 1 | S01 + S02：旧JSON/轨迹重分析、代码和资产清点 | CPU部分已完成；GPU端点/梯度sanity待做 | 无训练 | 来源可追溯，端点/坐标/梯度通过 |
| 2a | S03：seed0 sparse raw Fisher | 待实现/待评估 | 194个状态×256样本=49,664逐样本梯度 | 完整保存正负结果 |
| 2b | S04 + S05：actual-direction bridge + t9路径 | 待实现/待评估 | ≤24+18个状态×256；可复用r0 | 插值端点一致，求积可用 |
| 3 | S07：A补loss/raw trace及缺失分支 | 主配置缺shaped→SGD | 最少1个Task1；环境不匹配则2个 | 两行同后续规则、a22/D0可追溯 |
| 4 | S08 + S09：正式3 seeds与公共方向dense curves | 待做 | 若仅seed0可复用：4条完整序列=40个50epoch任务段；另有A分支 | 不以seed0正结果作为开展/删去seed的条件 |
| 5 | S06/S10/S11：proxy probe、中间路径、full-model、无保护pair | 条件/可选 | 单列预算 | 不阻塞主证据 |

首轮必须新增的梯度评估上界：`194+24+18=236`个状态，即60,416次逐样本梯度；r0去重后更少。若加入directed probes最多16个状态，总上界252，即64,512次逐样本梯度。另计全train/test前向、A两任务表、方向生成、求积加密和全样本稳定性检查。

正式 dense 配对：3 seeds×2方法×2 split×(1+80×100)=96,012次数据集前向，按r0只测一次计；无需新增训练，但不是零计算成本。新方向K100，r=0:.25:20；旧非配对 JSON 仅作为历史分析，不能混充正式公共方向采样。

正式每个seed至少都做B1 sparse raw Fisher和C/t9九节点。新增seed1/2这两项合计上界 `2×(194+18)×256=108,544` 次逐样本梯度，另计B2、A和加密；若B2也全seed执行，按seed0的236状态上界规划，即新增120,832次。不能把正式阶段预算只写成训练和dense前向。

B/C 共用训练，不为同一seed分别训练；seed1/2 每个方法只训一条10任务链，保存全部task。A 的shaped→SGD为单独Task1分支：每个正式seed各补一段，普通SGD分支复用对应完整序列的Task1；若续训协议不同，两行都同源重做Task1。多seed主表预算需额外记这2个新seed的分支，不能只计算四条完整序列。

结果按训练 seed 汇总均值、sample SD和每seed点；随机方向不充当独立训练重复。若旧seed0源码差异无法排除，先做同源seed0复现中心，再决定能否合并历史结果；预算另列2条序列，不悄悄把不同协议合成3-seed统计。

## 10. 论文落点与交付标准

当前 `GAM-EWC/paper/iclr2027_liying.tex` 的理论已转向 finite-displacement Fisher（约1124–1194行），但实验文字仍在讨论 inherited Fisher 的“可靠性”、quadratic residual / Fisher-Laplace validity（约2085–2200行）。下一轮写作应整体替换这段叙事，不仅给旧图换标题。当前方法段另含 averaged anchors；本 RQ2 无averaging要明确为 shaping 机制诊断，不能借用 A-on 的数值填此表。

- 表 A：改变继承状态，在相同后续训练规则下的 D0、accuracy/plasticity、位移和 raw trace。
- 图 B：等半径damage、邻域raw trace、经验容忍半径；train/test差异保留。
- 图 C：真实损伤—位移关系和最终路径敏感度；明确endpoint直线。
- 附录：数值检查、历史非配对曲线说明、可选directed/full-model复验。
- 删除主线中的 fixed-Fisher residual 优化主张、多估计器大扫描、默认四格、全seed中间路径和 full-vs-LoRA范数因果比较。

当前可以写：seed0 shaped anchor 在训练诊断集上具有更大经验容忍半径，完整序列在更大有效权重位移下产生更小旧任务损伤。尚不能写：test也一致改善、raw neighborhood/path Fisher已经下降、worst-case球已被验证，或初始shaping的作用已独立于保护矩阵。
