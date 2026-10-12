# RQ2 方法—理论对应审查与 sensitivity-directed probe 最小协议

审查日期：2026-09-20。只读代码、配置、checkpoint 顶层元数据及本地论文；没有 GPU forward、训练、队列变更或生产代码修改。本文不是结果有效性独立审稿，也不把实验目标视为已经成立的结论。

**用户范围确认：本次 RQ2 以“当前视觉 GAM+EWC 实验实现”为准。以下与 9 月 17 日稿的差异是方法溯源结果，不构成改用该稿算法或重训 A 的要求。当前资产可继续用于所选视觉实现的机制研究；保留历史源代码精确身份与参数坐标的界限。**

## 结论

**当前 B/C 的 Fisher 诊断确实测量了用户要求的 raw per-example empirical Fisher，C 的方向积分也直接对应本地 Proposition；但是 historical GAM 不能直接等同于此次核对的本地稿完整 FS-EWC 数值实现。** 障碍是实际更新公式、保留 anchor 的规则、训练 Fisher 的估计/归一化，以及训练与诊断坐标的差别。增加 directed probe 能完善历史 GAM 的机制诊断，不能消除方法身份差别。

本次核对的是 [2026-09-17 本地 LaTeX 副本](/data/140-1/users/liying/CopyFlatCV/GAM-EWC/evidence/iclr2027_paper_20260917/iclr2027_liying_source.tex:84)，不能确认它就是作者当前最终稿。若最终稿已经改变，需按最终版本重新做对应；不根据 FS-EWC/GAM 文件名推定等价。

## 1. 当前实验究竟执行哪个方法

`task_00.pt` 经 CPU `torch.load(..., map_location='cpu', weights_only=True, mmap=True)` 直接读取，顶层包含 `model_name=ewclora_normfisher_gam_intask`、`optimizer_type=gam`、`has_fisher_state=True`，不含算法源文件 SHA 或完整训练配置。相关 checkpoint 与十任务配置也见 [task_assets_manifest.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/task_assets_manifest.json:118)。

冻结源码 dispatch 与继承链如下，不能用同目录的其他 FisherShape 类替代解释：

1. [factory.py:254](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/utils/factory.py:254) 将该名称导入 `as1_normfisher_gam_intask.Learner`。
2. [as1_normfisher_gam_intask.py:16](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py:16) → `as1_normfisher_gam.Learner`；[as1_normfisher_gam.py:29](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam.py:29) → `ewclora_youyue_fitArchitechture_gam.Learner` → `ewclora_youyue_fitArchitechture.Learner` → `seqlora.Learner` → `baseLoRA.LoraBaseLearner` → `baseLearner.BaseLearner`。
3. 实际 optimizer 由 [baseLearner.py:619](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_CL/baseLearner.py:619) 导入 `optimer_PerturabtionType.gam.GAM`，并在第 674 行构造。它不是 `optimer_gam_filiter.py` 或 `as1_normfisher_fishershape.py`。
4. 当前分支覆盖的训练入口是 [as1_normfisher_gam_intask.py:1173](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py:1173)：SGD 分支直接求 CE+EWC；GAM 分支 clean cache、EWC cache 在前，CE-only closure 从第 1244 行开始。历史 sequence GAM `lambda_flat=1`，A 的 Task1 续训则切换到 SGD。

历史设置：ImageNet-A，ViT-B/16、Q/V LoRA rank10、10×20 类、seed0、batch128、50 epoch、Task0 lr=.04 / 后续 .02、cosine、momentum=0、weight_decay=0；GAM `rho=.05, normrho=.2, beta1=beta2=beta3=1, gamma=.1`；EWC `lambda=2000, gamma=.9, eta=0`，任务与累计 Fisher 均归一化。见 [训练协议](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/TRAINING_PROTOCOL.md:61)。

**溯源限度：** [source_frozen.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source_frozen.json:1) 固定了本次复现使用的源码，而不是历史 checkpoint 内自带的训练时源码。checkpoint 的类名/optimizer 和日志可对应到上述路径，但尚无历史启动时 source hash 证明每字节等同。复评通过只证明端点恢复与诊断数值可用，不证明更新算法身份。

## 2. 代码与本地稿：精确对应、surrogate、未对应

| 核对项 | 当前证据 | 判定与论文可用表述 |
|---|---|---|
| empirical Fisher 定义 | 稿件第 1005 行定义 `mean_i g_i g_i^T`；诊断在每张样本上平方梯度，再取均值 | **诊断定义对应**，须注明只在固定 Task0 head 的 effective Q/V 坐标、固定 256 张训练样本上测量 |
| finite-displacement bound | 稿件第 1140 行：`D² ≤ ∫ ΔᵀF(θ+sΔ)Δ ds`；诊断 path 第 373 行在直线段取节点 | **数学量对应**，本地代码做 Simpson 数值近似和收敛检查；不是实际 SGD 每步训练路径的积分 |
| neighborhood raw trace | 稿件第 1170、1257 行定义 trace 与球内最大值；现有 B 对 random directions 逐点重算 raw trace | **pointwise / sampled typical neighborhood 对应**；16 个随机方向的均值、Q95、最大值不等于球内 supremum |
| aggregate gradient surrogate | 稿件第 1449 行明确 `||mean_i g_i||²` 不是 `mean_i ||g_i||²` | **只属 surrogate**；GAM normalized gradient-difference direction 可解释为 aggregate-gradient sensitivity probe，不能说直接求 raw per-example trace 的最大方向 |
| 四梯度最终更新 | 稿件第 1465–1495 行为 `g0 + alpha * (2/s_z)(g3−g2) + EWC`；GAM 为下面的四梯度分解 | **公式不同**，相同的梯度求值次数不建立等价 |
| EWC 是否进入 inner-search | 稿件第 1506 行要求 EWC 在 search 外；代码 CE-only closure + clean-point EWC cache | **结构对应**；不消除最终 GAM 更新的差别 |
| 同 batch / stochastic realization | 稿件第 1501 行要求同 batch、同随机 realization；GAM closure 重用 inputs，但未在各 closure 之间显式 replay RNG | **同 batch 已对应；若有随机层，同 realization 尚未证实**。如果实际网络无随机层，需用模型配置证明，不能仅由输入同一得出 |
| 四次 CE 梯度成本 | optimizer 本身四次 closure；当前 learner 在它之前额外计算 clean gradient cache（第 1211 行） | 不能将整个 learner 直接报成严格四次 CE 梯度；冻结实现还存在一次额外 clean 梯度求值 |
| trainable parameterization | 稿件第 964 行要求 gradients / perturbations / importance 同一 trainable parameterization；GAM optimizer 对所有 trainable A/B 和 head 求梯度，EWC 与离线诊断用 effective BA | **坐标不同**；必须显式区分 factor-space surrogate 与 effective-weight 机制测量 |
| 保留 anchor | Abstract 第 97、104 行，Intro 第 566 行、method 第 1350 行要求 late-stage checkpoint average | 历史/当前所选配置没有该 averaging，`approximate_fisher_in_task=false`，当前 `in_task_weight_merge=false`，非 damp 分支；只能作为 shaping component / historical variant 的证据 |
| consolidation Fisher | 稿件第 1381 行写 diagonal per-example Fisher + `beta I`，第 1396 行 `gamma Lambda + Fhat`；训练为 batch-mean 梯度平方并 trace normalization，`eta=0`，累计 Fisher 再归一化 | **估计器、scale、damping 规则未对应**；不能把训练 stored Fisher trace 约等于 1 当作 raw sensitivity |

这里的对应判定只针对上述本地稿；没有宣称作者最终方法必须采用其中某个版本。

## 3. GAM inner-search 为什么不能直接称为 raw-Fisher maximization

令 `g0=g(phi)`，`g1=g(phi+rho*g0/||g0||)`；代码 [gam.py:60](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/optimer_PerturabtionType/gam.py:60) 完成第一次 perturb，恢复后 [gam.py:92](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/optimer_PerturabtionType/gam.py:92) 以 `g1−g0` 的单位向量做 `normrho` ascent。因此在小有限差分、同坐标、同 stochastic realization 下，该**方向**与 `∇||g_batch||² = 2 H_batch g_batch` 方向近似一致；归一化取消了正的有限差分比例。

随后令 `g2=g(phi+e_search)`，`g3=g(phi+e_search+rho*g2/||g2||)`。实际 [gam.py:129](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/optimer_PerturabtionType/gam.py:129) 在 beta 全为 1 时定义

`h_plus = g1 + g3`，`h_minus = g0 + g2`，`g_GAM = h_plus − gamma * (h_minus − proj_{h_plus}(h_minus))`。

learner 再执行 [as1_normfisher_gam_intask.py:1058](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py:1058) 的 `g_clean + lambda_flat*(g_GAM−g_clean) + g_EWC`。历史 `lambda_flat=1` 时是 `g_GAM+g_EWC`。这不等于本地稿的 `g0+alpha*(2/s_z)*(g3−g2)+g_EWC`。应通过一个冻结的小模型单步方向核验来确认最终候选实现，不能靠名称映射修正。

而且 `phi=(A,B,head)` 与 `W=(BA)_QV` 不同：`∇phi loss=Jᵀ∇W loss`（加上 head 项），所以 factor trace 涉及 `J Jᵀ`，不是 W-space trace；LoRA 的等价缩放 `A→cA,B→B/c` 还能改变 factor 梯度范数而不改变模型函数。把 factor search 位移转换到 effective W 后可做一个**GAM-generated direction diagnostic**，但它不再是固定 W 球中 raw Fisher 的 ascent guarantee。

## 4. 现有 raw 诊断与 path 应如何命名

[rq2_common.py:119](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/rq2_common.py:119) 把各 Q/V 的 BA 变成可直接扰动的 24 个 dense 矩阵，共 14,155,776 维；base backbone 和 Task0 的 20 类 head 固定。第 150 行 collector 用每例 adjoint 形成 effective-W 梯度，第 335 行用 `losses.sum().backward()` 保留每例尺度，最后第 348 行平均 trace 与 `(g_i·Δ)^2`。没有 normalization、floor 或 EWC 梯度。因此其 raw 定义是合适的；**不能复用这个 collector 直接求 trace 的二阶导数**，因为其 `no_grad`/`detach` 专为数值测量。

训练保护 Fisher 则见 [as1_normfisher_gam.py:381](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/models_LoRAbasedCL/as1_normfisher_gam.py:381)：每 batch 的 mean CE backward，平方 batch 梯度后乘 batch_size，再跨样本数平均；第 177 行和第 243 行分别处理 per-task/accumulated trace normalization。它和离线 raw Fisher 必须分栏。

建议图注写清：

- B：**Random and sensitivity-directed perturbations in the ambient effective Q/V weight space, with the Task0 head fixed.** 随机 dense perturbation 一般不在 rank10 LoRA 流形上；这是 ambient-space robustness，不是每个 perturbation 都可被 rank10 因子表达。
- C：**Straight-segment directional empirical-Fisher sensitivity from the Task0 anchor to each realized future-task endpoint.** 真实的是未来任务端点与累计位移；积分插值是一条诊断直线段。Task0-to-Task9 chord 一般也不全在 rank10 因子流形上，但与 Proposition 的直线段定义相容。
- `Sbar_path=I/||Δ||²` 与 `I`、`||Δ||` 分别保存。`I` 低可能来自位移小；`Sbar_path` 才衡量单位位移方向敏感度。
- bound 数值验证必须让 `D` 与 Fisher 使用同一个样本集合。现有 path 第 391 行用 Fisher 的 `subset_loss`，这一点是正确的；不能拿 test damage 与 train-subset `I` 当作同一 empirical bound 验证。

## 5. 最小新增 directed probe：16 个 evaluation 点，无重训

主协议建议采用**直接 W-space raw-Fisher gradient 单步 probe**，另将 GAM-compatible aggregate-gradient direction 作为可选对照；两者不互相冒称。

| 项目 | 预先固定的协议 |
|---|---|
| anchors | 当前 seed0 SGD Task0 与 GAM Task0，原 checkpoint 哈希固定 |
| 坐标/预测任务 | 与现有 B 完全相同的 24 个 effective Q/V 矩阵、固定各自 Task0 head、FP32、确定性 transform、无数据增强 |
| search data | 从 Task0 train 中除去已冻结 evaluation256 后，固定分层抽取 64 张，两 anchor 共用；记录 ID/像素哈希及 seed |
| calibration data | 若需要选择步数/有限差分精度，仅从剩余 train 再抽固定 64 张，与 search64/evaluation256 不交叠；单步 raw-gradient 固定协议无需模型选择即可运行 |
| evaluation data | 保留已有未参与 direction search 的 Fisher evaluation256；报告其 raw trace、subset damage。full train/test damage 可附报，但 full train 含 search 样本，需另列 train-excluding-search damage，不能称 full train 为 heldout |
| search objective | `T_search(W)=(1/64) sum_i ||∇W CE_i(W)||²`，true labels、不归一化、不加 EWC |
| 最小 search budget | 每 anchor 在其原点计算一次 `h=∇W T_search(W)`，用 `u=h/||h||`；两个 anchor 完全相同样本与预算。二阶梯度使用独立 autograd 路径逐样本/小 microbatch 累加，不能对现有 detached hook 反传 |
| evaluation points | 两个生成方向 `u_sgd,u_gam` **交叉施加到两个 anchors**，`r∈{1,5,10,20}`，即 `2 anchors × 2 direction sources × 4 radii = 16` 个新点；r0 复用同协议 anchor |
| norm | 对全部 24 个矩阵统一 Frobenius 范数；写回后核验 `||Wprobe−Wanchor||≈r`，记录实际半径和 direction hash；禁止逐层各归一化到 r |
| budget equality | 不按 test 结果给某一方法增加 steps/restarts，不丢弃产生高 sensitivity 的 direction；保存所有生成方向与生成目标值 |
| outcome | raw trace、relative-to-own-anchor trace、D_signed、positive damage、sampled max；direction-source × evaluated-anchor 交叉表必须完整保留 |

单步 direction 对所有 r 相同，正好节省 search 成本。它是局部敏感方向的有限半径 probe，不能保证 r=20 时仍接近最大值。如果需要进一步靠近球内高敏感点，可预注册每个 r 使用相同三步 projected ascent、固定步长 schedule 和相同 restart 数，投影到 W-space radius-r ball，仍只称 finite-search diagnostic。不得看到 test 结果后只给某一 anchor 扩充预算。

现有 random-direction mean/Q95/sample max 可以直接由逐点 raw states 重算。对同一 evaluation256，在 random+directed 的有限候选集合取最大值，是该**evaluation empirical objective** 球内最大值的 sampled lower bound；不称 exact supremum，也不是 population/test worst-case 的下界保证。有限 16 directions 的 Q95 是探索性分位数，不能当成严格 tail coverage。

### 能否直接复用 GAM inner-search

**不能直接调用完整 `GAM.step()` 代替这个诊断。** 它会执行外更新、在 factor/head 参数组中求 norm，优化 aggregate-gradient surrogate，且不是 raw trace。

可复用的只是 [gam.py:92](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/source/optimer_PerturabtionType/gam.py:92) 对 `g1−g0` 归一化的数学结构，独立写一个只选 direction、不做 optimizer step 的离线函数：

`g0=∇W mean CE_search(W)`；`g1=∇W mean CE_search(W+delta*g0/(||g0||+xi))`；`u=normalize(g1−g0)`。

这在 W 坐标下称 **GAM-style aggregate-gradient sensitivity probe**，使用同一个 fixed batch/确定性 forward，可用训练的 `delta=.05` 作为预注册参考，但 `.05` 从 factor-space 移到 W-space 没有半径等价含义。它只能回答“aggregate-gradient 敏感方向上的 heldout raw trace 是否较低”。若要说“方法自身产生的方向”，必须使用实际 factor/head inner-search、恢复模型、形成精确 `BA_after−BA_before`（不是简单 `B dA+dB A` 忽略二阶项），再统一 W-norm，并报告 head 位移被丢弃/冻结造成的坐标转换；这一分支仍是补充 surrogate probe。

## 6. P3 的实际收口条件与后续最省实验路线

1. 固定待投稿的 method 版本与源文件/配置哈希，做一张“论文更新公式—实现函数—最终参数”的签核表。若以上本地稿为目标，历史 GAM 属于 shaping-related variant，不能将图例简单改名成完整 FS-EWC。
2. 保留已经完成的 A/B/C，明确 historical GAM/SGD、数据集、模型、seed0 和 inference scope；它们仍回答 solution sensitivity 与 realized drift 的机制问题，不因方法对齐问题作废。
3. 若主张只针对已实现 GAM shaping component，应收窄文字并报告 surrogate→raw sensitivity 是经验验证，不是数学恒等。若主张完整本地 FS-EWC，则最小新增训练应先完成目标实现与 vanilla EWC 的 seed0 两任务/Task0 pilot，核验更新方向、head/数据 RNG、anchor averaging 与 Fisher 规则，再将同一离线 B/C runner 应用于新 checkpoint；无需先跑大量 Fisher estimator comparisons。
4. directed probe、更多 task-endpoint `Sbar_path,t`、多 seed 应在方法身份与统计口径固定之后安排。增加 seeds 能衡量复现稳定性，不能修复错误的算法标签。raw neighborhood Fisher、path integral 在当前 seed0 已有测量；新增 work 应针对覆盖范围和证据薄弱处，而不是把它们重新登记为“完全未测”。

另：O-LoRA 根目录的 [FisherShapedEWCLoRA_implementation.md](/data/140-1/users/liying/O-LoRA/FisherShapedEWCLoRA_implementation.md:127) 描述的是 LLaMA/两 CE 梯度的 FisherShape，默认 ratio correction，且第 363 行明确承认 factor-space 与 full-Delta-W 的区分；它不在本次 ImageNet-A historical checkpoint 的 dispatch 链上，不能作为 P3 等价的凭据。
