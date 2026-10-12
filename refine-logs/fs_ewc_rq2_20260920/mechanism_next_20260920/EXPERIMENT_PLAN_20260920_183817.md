# RQ2：已有证据与最小机制补测

日期：2026-09-20。范围：用户指定目录下的 ImageNet-A seed0 A/B/C；A 已完成并保留。本文是下一轮方案，不是已提交的新 GPU 队列。原恢复队列已完成 15/15 作业，`execution/report_seed0/RESULTS.md` 现在包含真实结果，旧 pending 描述已过时。

## 1. 判断与主张

当前数据可以继续检验机制，但尚不能写成“shaping → 邻域敏感度下降 → 容忍半径扩大 → 真实漂移损伤下降”的完整证据链。最重要的缺口已从“没有 raw Fisher”变成“训练侧敏感度改善能否对应测试损伤、实际方向和最终方法版本”。

仅保留两个主要待检验主张：

1. **局部性质**：在相同坐标、半径、方向和评估样本下，shaped 起点具有较低经验敏感度；是否同时带来更小旧任务损伤、更大容忍半径，必须分别检验。
2. **顺序学习**：在真实未来任务形成的端点位移下，较低单位位移方向敏感度与较小旧任务损伤共同出现，且遗忘差异不能单靠位移范数排序解释。

A 是已完成的干预对照表；B、C 分别检验上述两项。不把三块重新扩展为三套大规模训练，也不默认追加 A 的四格或 Fisher estimator comparisons。

当前能用的表述：**在这个 seed0 历史 GAM+EWC 案例中，训练子集的有效权重 Fisher trace 更低；实际十任务序列端点的位移更大、旧任务测试损伤更小，连接端点的单位位移 Fisher 积分更低。随机邻域测试平均损伤和平均损伤容忍半径没有显示相同改善。**

## 2. 当前配置与数据身份

| 项目 | 核验后的设置 |
|---|---|
| 数据 | ImageNet-A，固定 200 类，10 个任务，每任务 20 类；Task0 train=1259/test=307 |
| 模型 | ViT-B/16，日志预训练标识 `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` |
| 适配器 | Q/V LoRA rank=10，12 blocks；主干与旧 head 固定 |
| A | 普通或历史 GAM Task0 起点，各接同一 SGD+EWC Task1 协议；各自继承 head/Fisher；50 epochs、400 updates、配对随机过程通过 |
| 历史链 C | SGD→SGD 与 GAM→GAM 十任务链；后者未来任务仍使用 GAM，不能说只在 Task0 shaping |
| 保护项 | λ=2000，历史 batch empirical Fisher 经 trace normalization；不是本次 raw Fisher |
| raw Fisher | true-label、per-example CE，固定同一分层 256 个 Task0 **训练**样本；未归一化、无 floor、无 EWC 项 |
| 诊断坐标 | 24 个 dense effective Q/V BA 矩阵，共 14,155,776 维；固定旧任务 20 类 head |
| 评估精度/视图 | FP32，关闭 TF32；Resize256 bicubic→CenterCrop224→ToTensor，无 Normalize；诊断 train/test 均用确定性视图，训练本身仍有随机增强 |
| 重复 | 只有 training seed0；K=16 是配对随机方向，不是 16 个训练重复 |

trace、半径和范数依赖坐标。训练优化器中的 A/B 参数空间与这里的 dense BA 空间不同；dense 扰动可以离开 rank10 流形。此诊断衡量有效权重的局部性质，不能不经推导就称为直接测量原优化器的全部目标。

经验 Fisher 也不是一般意义上的 loss Hessian。true-label CE 梯度大小受当前拟合程度/置信度影响；起点训练 loss 不同，因此 raw trace 下降应称为经验敏感度下降，不能直接宣称已经隔离出与拟合质量无关的曲率变化。保存 clean loss 与 own-anchor trace 比值供解释，但不以归一化比值替代正文 raw trace。

## 3. A/B/C 实际证据状态

| 块 | 已完成 | 目前支持/不支持 |
|---|---|---|
| A | 两条 Task1 SGD+EWC 续训，端点、raw trace、位移、随机配对核验 | 表格完成；此两行 GAM 起点 a21=84.04 低于 84.36，a22=72.05 低于 74.80，位移也更小，因此 A 单独不证明“更少遗忘且非更少移动” |
| B | 2 anchors×7 radii×16 paired directions（r0 共用）；raw trace、Gmean/G95、prefix-safe radius；真实方向交叉 bridge | 所有半径训练子集 trace 约为基线 0.47–0.52；train Gmean 更低，但 test Gmean 在 6 个非零半径均更高；当前 Gmean 容忍半径两方法完全相同 |
| C | t0…t9 accuracy/loss/displacement；两条 t9 直线积分，各 9 nodes，求积与同样本界检查通过 | t9 位移 1.19758→1.40047，test loss damage .26013→.03435，S_path .36806→.01479；强 seed0 方向证据，但 t1/t5 的积分尚缺 |
| 方法归属 | checkpoint/config/source/paper 比对 | 不能默认历史 GAM 等于最终 FS-EWC；详见 METHOD_ALIGNMENT.md |

数值箭头均为 SGD→历史 GAM。A 的准确率是 seen-class class-IL；B/C 机制 loss 使用固定 Task0 head，C 主图 accuracy 也使用固定 20 类 head。A 要保留 a22，防止把新任务学习不足误写成无代价保护。

这里的 A 仅指本目录 `execution/report_seed0/A_two_task.csv` 的已完成两行；其他目录中的六行、轨迹平均、ImageNet-R 或 smoke 结果不并入本表。

## 4. 首批工作：不重训的两条独立诊断线

### P0：先锁定方法身份（CPU）

保存目标论文/算法版本、训练代码 SHA、完整开关、checkpoint SHA、参数坐标和数据哈希。历史 checkpoint 没有源代码内容绑定时，标为“historical GAM implementation; exact historical source identity unverified”。

若以当前视觉 GAM+EWC 为研究对象，可以继续离线测量现有资产；若以 9 月 17 日论文算法为最终 FS-EWC，先做更新公式、EWC 接入、averaging/damping 的一致性验证，再决定哪些训练资产可代表最终方法。不能通过重命名历史结果完成这个步骤。**用户已明确选择“当前视觉 GAM+EWC 实验实现”为本次目标。** 因此继续复用现有资产，不为追随 9 月 17 日稿另行重训；上面对论文版本的条件分支只保留为溯源说明。本轮需要固定当前实现的公式/开关与可用源码 SHA，并按该版本报告。

### P1a：解释 B 的 train/test 分歧（GPU0/1）

沿用已有 16 个公共方向，在固定半径 `{0,1,5,10,20}` 上对 **全部 307 个 Task0 test 样本**计算 raw per-example Fisher，并保留每样本 loss/directional products。它是评估指标，不回传训练，不用于搜索或挑方向。沿用同一 train256 结果作并列比较。

- 两方法各 `1+4×16=65` 个唯一状态，总 130；新增 `130×307=39,910` 个逐样本梯度样本求值，无训练。
- 主尺度预先固定 r=1，接近真实 drift；r=5/10/20 明确是 stress test，不把大半径分离当成真实更新区间的证明。
- 报告 trace mean/Q95/sampled max，Gmean/G95（train/test 分开），以及同方向的变化；不得事后只报告有利半径。
- 此 test Fisher 解释当前 seed0 测试行为，不构成新的独立验证集。若用它选择算法/超参数，正式确认需要新的训练 seeds 或独立数据。
- 不通过换 Fisher estimator、删除 test 曲线或任意改 ε 去制造机制闭环。

### P1b：sensitivity-directed bridge（GPU0/1，接 P1a）

在 r=`{1,5,10,20}` 加一个预设预算的 inner-search probe。搜索只用独立的训练 calibration 子集，禁止用 test 或最终 eval256 选择方向。方法内层方向与 raw empirical trace 的目标不完全相同，报告时称 **GAM-directed proxy diagnostic**；只有确实对同一 raw trace 做梯度搜索的实现，才称 **trace-directed probe**。二者不混写。

最低成本设计：每个源 anchor 只生成一个预先指定 inner-search 方向，将诱导的 effective ΔW 归一化；两源方向×两目标 anchors×四半径=16 个新状态。每个状态同时评估 train256 和 test307，新增 `16×563=9,008` 个逐样本梯度样本求值，另加两个方向的搜索成本（单独实测）。如果改成每个半径独立搜索，需要另记搜索数，不能继续写“两次搜索”。

必须记录实际 optimizer 分支、head 是否参与搜索、随机种子、batch、步数、方向 SHA、effective norm 和非零性，cross-evaluate 两个源方向以区分 anchor 与 direction。固定现有 random K16 作参照；其 sampled max 不是 supremum，搜索 proxy 也不是 exact worst case。不同搜索计算预算写入表注。

搜索 A/B 扰动映射到 effective W 后再归一化/交叉施加，是一种共同坐标的 proxy 测量；不宣称归一化后的任意 r 点就是优化器原本走到的点。若目标要检验理论中的相同 raw trace，另实现同坐标、同 calibration split 的直接 trace ascent，先做小规模求导/有限差分核验；不把此扩展暗中替换成廉价 proxy。

### P2：把 C 从末任务案例补到预设早/中/晚阶段（GPU2/3）

复用 t9，补 t=`{1,5}`。每个 target 从真实 checkpoint 重新构造 `Δ0→t`，验证旧 head 与 backbone 不变、s=0/s=1 精确恢复，计算：

`I_t = ∫_0^1 mean_i (g_i(θ0+sΔ)·Δ)^2 ds`，`S_path,t = I_t / ||Δ||²`。

- 主指标是方向积分，endpoint trace 为附录；跨矩阵先求和再平方，保留交叉项。
- 路径是 θ0 与 θt 的**直线段**，不是逐步 SGD 优化轨迹。主图写“endpoint-segment sensitivity”。
- 使用同一 train256；9-node Simpson 与其 5-node 子网格比对，差异>5% 时加到17，仍未收敛则33或标记不收敛，不展示未通过的正式积分。
- 2 methods×2 new targets×8 nonzero nodes=32 新状态；每个 anchor 另做一次原点梯度，同时计算 t1/t5 两个方向量，共 34 个状态求值，`34×256=8,704` 个梯度样本求值（不含求积加密、sanity 和最终新增配置自检）。
- 由于旧原点只存 t9 方向二次型，不能直接复用其标量充当 t1/t5 原点值。若新 runner 无多方向同次计算，实际需 36 passes=9,216，清楚记录。
- 当前 `diagnostic.py` 的路径恢复硬编码 Task0/Task9；只修改 `path_task` JSON 不会实现 t1/t5。先改 target-aware checkpoint/delta/cache key，再做 endpoint 与 q 的数值自检。
- bound 仅对同一256样本的 `D_subset²≤I` 做一致性检查；不能拿全 test loss 代入 train256 的 I。Δ=0 时 S 未定义，t0 画 NA，不伪填0。

### 两线依赖与资源

| 线路 | 建议顺序 | 真实依赖 |
|---|---|---|
| CPU | 结果组织、方法身份、补测实现/自检 | 已有 checkpoint 和数据；不依赖任何新训练 |
| GPU0/1 | test rawF → directed probe；0 SGD anchor，1 GAM anchor | 同方向/样本/坐标协议；directed cross-eval 等两个搜索方向资产 |
| GPU2/3 | t1/t5 path；2 SGD，3 GAM | checkpoint 恢复、target routing；**不依赖 B 或 A 的新输出** |
| CPU report | 更新 B/C、复核主张 | 等两线完成；保留失败/负结果 |

这是科学 DAG 的并行关系；不是所有阶段线性相关。部署时检查当前 GPU 作业/共享锁并等待释放，不能复用“先前已空闲”的快照。新输出必须使用新 protocol/run 目录，不能修改 recovery1 的 raw states，也不能把本方案当作已排队配置。

## 5. 容忍半径与正式重复：有条件的第二批

**半径分辨率**：当前 Gmean 的 grid tolerance 相同，意味着当前证据未显示优势，不等于真实连续半径必然相等。若正文必须主张更大 tolerance，在现有半径间对两方法、所有16方向使用同一个预定加密网格 `r=.25,.5,…,20`，固定 ε=`{.001,.005,.01,.02,.05}`。现有7个半径外再增加74个半径，共 `2×74×16=2368` 状态，full train/test各一遍为4736数据集前向；这一步只算 loss，不必全网格算 Fisher。说明是对已观察 seed0 的探索性分辨率复核，随后在独立 seeds 固定协议确认。不得依据哪种方法获胜决定保留哪些 ε 或半径；若仍无改善，删除广义容忍半径优势主张。

**正式重复**：当前 seed0 为机制 case study，不能用 direction bootstrap 冒充多训练 seed 置信区间。用户已选定当前视觉实现；保存其配置/源码后，先完成上述低成本诊断，再预注册补 seed1/2 的同设置比较。若存在完全匹配 checkpoint，直接复用；否则两方法×两新 seeds×十任务=4条链、40个 task-training segments。若历史 seed0 不符合最终算法，不纳入最终算法的3-seed统计：应另行明确新资产需求，不能把它悄悄算作三者之一。

每 seed 至少执行固定16方向的 B、真实方向 bridge、t9 path；只在预设 t1/t5/t9 报 C 的路径点。按 seed 成对报告方法差值与均值±样本标准差，n=3 明确写出，不承诺显著性。新 seeds 的 Task0/Task1 资产如可严格匹配已完成 A 协议，可以复用，但 A 额外分支不在本轮默认任务内。

不做的默认扩展：full-model 大表、很多 Fisher estimator、λ 大扫描、A 无保护四格、所有任务所有路径节点。只有保留“纯起点几何的因果效应”这种更强主张时，才需要 common protection/head 或保护开关等额外干预；C 的系统级关联本身不要求把 A 重新铺满。

## 6. 正文组织与结果判据

| 位置 | 内容 | 必须写出的边界 |
|---|---|---|
| A 一张表 | a11/a21/a22，ΔL0 train/test，raw trace T0/T1，||ΔW|| | 已完成的两行；配对未来优化协议但起点/head/Fisher 各自继承；保留 plasticity |
| B 一张2×2图 | Gmean、G95、raw neighborhood trace、ε→prefix-safe radius | train/test 损伤同时可见；目前 trace 仅 train256；sampled max非最坏值；同值曲线不能互相遮蔽；边界半径需标 right-censored |
| C 一张2×2图 | Task0 accuracy、test loss damage、S_path、actual displacement | 当前只有t9的 S 条形对照，不能补画不存在的 t 曲线；未来加t1/t5后只画实测点 |
| Appendix | 2×2 anchor/direction bridge，I与D_subset²/gates、seen-class accuracy、point trace、历史图、source mapping | 历史非配对方向与当前确定性视图分开；fixed-Fisher penalty/residual/linear term降为辅助 |

判据按每条主张独立执行：trace降低只支持 sensitivity；Gmean降低才支持平均正损伤改善；prefix-safe半径增加才支持该 ε/分布/坐标上的容忍范围增大。不能用其中一个指标代替另两个。真实方向好而随机方向 test Gmean 无改善时，RQ2 可以收窄为 **direction-dependent robustness under realized drift**，保留随机邻域反证。

这组实验可以建立与理论相符的机制证据；即使所有指标有利，单个 seed、相关指标、不同后续优化器也不能自动证明正式的因果中介链。

## 7. 输出与状态

- 已核验数据：`EVIDENCE_PACKET.json`、`claims.json`、`evidence_precheck.json`；12个摘要的 SHA 与原报告一致。
- 独立同系列主张判断：`CLAIMS_FROM_RESULTS.md/json`，semantic acceptance 为 provisional，按其结论执行后续分支。
- 方法版本审查：`METHOD_ALIGNMENT.md`。
- 新版图表与复现脚本：`figures/`，包含 A 表、B/C PDF/PNG、caption/LaTeX 与原始来源哈希。
- `EXPERIMENT_TRACKER.md` 区分本次完成的 CPU 整理与尚未执行的 GPU 补测。
