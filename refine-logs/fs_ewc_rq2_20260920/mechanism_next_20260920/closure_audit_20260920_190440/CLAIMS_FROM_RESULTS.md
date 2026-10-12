# RQ2 结果到主张闭环审查

- `claim_supported`: **partial**
- 完整链 `shaping → lower F → larger radius → less damage`: **no**
- 窄化后的方向依赖 seed0 case study: **partial**
- `confidence`: **high**（针对本次快照和所列本地证据）
- `review_independence`: **same-family**
- `acceptance_status`: **provisional**
- `integrity_status`: **unavailable**（未找到与本范围匹配的 `EXPERIMENT_AUDIT.json`；不阻断本次判断）
- 结果快照：**2026-09-20T19:12:31.478316Z**
- 方法范围：用户指定的当前视觉 GAM+EWC 实现；不把 9 月 17 日稿的方法身份差异当作阻塞项

## 总判决

当前材料足以排成“一张 controlled two-task 表 + 一张 neighborhood 图 + 一张 realized-future 图”，也足以报告一个诚实的单 seed、方向依赖机制案例；但不足以把两项主张连成广义或因果闭环。主要断点不是表图数量，而是证据符号：当前 ImageNet-A seed0 中，GAM anchor 的 train256 raw empirical-Fisher 较低，训练侧随机方向损伤也较低；然而 test `G` 在六个非零半径上全部更高，train/test 的 `G` prefix-safe 容忍半径全部相同。因此 `lower F → larger radius` 没有得到支持，`larger radius → less damage` 也无可用前提。

真实方向证据更强但范围更窄。2 anchors × 2 realized directions 的 bridge 中，在两种方向和五个非零半径的 10 个比较里，GAM anchor 的 `D_test` 均低于 SGD anchor；方向来源本身也造成很大差异。十任务历史链中，GAM→GAM 在 Task1–Task9 的每个保存端点都表现为更大 effective-weight 位移和更小 Task0 test loss damage；Task9 endpoint chord 的单位位移方向敏感度也更低。这支持“方向与 anchor 都重要，位移大小不足以解释损伤排序”的描述性结论，但不识别因果中介，也不能与 A 的 GAM-anchor→SGD continuation 无缝拼成同一干预链。

## 四层闭环状态

| 层次 | 状态 | 判断 |
|---|---|---|
| 设计 | **yes** | 两个主张、A/B/C 分工以及一表两图结构已经清楚；当前视觉 GAM 即本次 FS-EWC 优化实现，方法身份不再是前置条件。 |
| 实现 | **partial** | 既有 seed0 A/B/C runner 和表图导出已实现。`mechanism_next/followup_spec.json` 仍是 `proposed_not_queued` 且不是可执行 campaign；新 `full_rq2_20260920` 已生成 88-job formal plan 并通过 CPU/smoke contract checks，但 sparse-test 默认半径为 `{0,1,10,20}`（少于计划的 `r=5`），path 改用 full train1259 而既有 t9 是 train256，且 `bridge_full` 只评估 realized/random directions，没有实现 sensitivity-directed search。 |
| 执行 | **partial** | 既有 ImageNet-A seed0 的 A、随机邻域 B、2×2 realized-direction bridge、十任务端点曲线和 t9 chord 均完成。test raw Fisher、真正的 directed probe、t1/t5 chords 和匹配的正式多 seed 结果未完成。冻结快照中 formal plan 尚无 queue state，正式训练 `COMPLETE` receipt 为 0；已完成的两个 smoke 仅为 SGD dense 的 2 个状态和 GAM sparse-train 的 r0 单状态，不能计作新 B/C 证据。旧 workers=4 λ=0 pilot 已标记为 superseded；替代的 workers=8 两分支仍为 `running`。这些 λ=0 分支也不是当前机制断点的必要修复。 |
| 语义 | **no（完整链）/ partial（窄化 case）** | 完整 `lower F → larger radius` 链被当前 test-G 与容忍半径结果截断。若删除“larger radius”和因果中介措辞，现有结果可支持单 seed、坐标依赖、方向依赖的描述性案例。 |

## 两个主要主张

### P1：shaping 降低 neighborhood sensitivity，并扩大 damage-tolerant region

`claim_supported: partial`

支持部分：在固定 train256、固定 Task0 head、24 个 effective Q/V `BA` 矩阵坐标和 16 个 paired random directions 下，GAM/SGD raw-trace 比在所测半径上为 `0.468–0.520`；train `G` 与 `Q95` 在六个非零半径均较低。

不支持部分：当前 test raw Fisher 没有测量；CSV test 行中的 `raw_trace_mean_train256` 仍是同一个 train256 指标。更关键的是，test `G` 在六个非零半径均较高。train `G`、test `G`、test `Q95` 的全部已报告 prefix-safe 半径都相同；只有 train `Q95` 在两个 epsilon 上出现更大半径，其中 `epsilon=.05` 还是右删失。因而不能写成 split-robust neighborhood advantage 或一般的 larger tolerated radius。

### P2：较低方向敏感度允许真实 future drift 造成更少 old-task damage

`claim_supported: partial`

支持部分：Task9 时，GAM→GAM 的 drift 为 `1.400472589`，高于 SGD→SGD 的 `1.197584100`；test loss damage 为 `0.034347380`，低于 `0.260130529`。同一 train256 上，Task9 chord 的 `S_path` 为 `0.014793093`，低于 `0.368057602`，`D_subset` 为 `0.038097272`，低于 `0.163822279`；两条 chord 的求积与同样本数值 gate 均通过。

边界：只积分了 Task0→Task9 的 ambient effective-weight 直线段，不是 optimizer 实际走过的轨迹；完整对比是 GAM→GAM 对 SGD→SGD，未来任务优化器也不同。因此“较低 chord sensitivity 与较低 damage 共同出现”成立，“Task0 shaping 通过该中介因果地带来较低 damage”不成立。

## 箭头逐项判定

| 箭头 | 判定 | 证据结论 |
|---|---|---|
| `shaping → lower F` | **partial；窄范围 yes** | 当前 ImageNet-A seed0 的 train256 raw trace：anchor `1465.497 → 761.901`，共同 SGD+EWC continuation 后 `2008.388 → 940.413`。这是固定训练样本与坐标下的 empirical-gradient energy，不是 test Fisher、Hessian 或跨 seed 定律。 |
| `lower F → larger radius` | **no** | 当前 `G` 容忍半径在 train/test 的所有 epsilon 均打平；test `G` 在 6/6 非零半径反向。低 train raw F 没有转化为当前定义下的普遍更大半径。 |
| `larger radius → less damage` | **no / not demonstrated** | 当前没有观察到所需的 larger-radius 前提，也没有独立干预 radius 来识别该因果箭头。 |
| `lower realized-direction/chord sensitivity ↔ less realized damage despite larger drift` | **partial** | bridge 的 GAM anchor 在 10/10 个非零 anchor-matched comparisons 上 `D_test` 更低；C 的 t9 chord 同时出现低 `S_path`、低 damage 和更大 drift。只支持有限方向、单 seed 的关联。 |

## A：controlled two-task 表应怎样解释

当前 ImageNet-A A 是有效的 **package-level controlled continuation**：两个 inherited Task0 package 接同一 SGD+EWC Task1 规则，且 paired RNG 检查通过；但 inherited weights、old head 和 EWC Fisher 均随 package 改变，所以它不隔离单一 geometry 或 protection matrix。

当前结果不是正向 retention 表：

| Task0 anchor | a21 | a22 | test old-task CE change | drift | raw trace T0 | raw trace T1 |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 84.36 | 74.80 | -0.019246 | 0.374950 | 1465.497 | 2008.388 |
| GAM | 84.04 | 72.05 | -0.013388 | 0.349876 | 761.901 | 940.413 |

因此 A 支持“shaping 改变了 inherited solution/package”，不支持“在共同 future SGD+EWC 下提高 a21/a22”，也不能在本表里用“更大 drift 仍更少损伤”解释。当前渲染表只放了 T0，但 CSV 已有 T1；把 T1 加入表格不需要训练。

附件引用的 `19.885→10.946` 与 `87.95→91.82` 已经核实，但属于另一套 **ImageNet-R、fresh K/V LoRA、Adam、seeds 0/1/2、batch-gradient Fisher proxy** 的旧实验。control→setting1 的配对均值为：F1 proxy `19.8852→10.9458`、a21 `87.9467→91.8233`、BWT `-2.78→-1.52`、a22 `83.45→82.5667`。它可以单独支持该旧设置中的正向 two-task 结果；不能与当前 ImageNet-A/QV-SGD/raw-train256 的 B/C 数值拼成一条链。旧实验 18 份日志均 `save_ckp=False`，其 F2 和 endpoint base loss 对应新任务而非 old-task endpoint；若旧表强制要求 loss-based `D1`，现有日志不能补算，需重跑准确的旧分支。否则应省略 D1 并如实标注现有指标。

## B/C 当前最强的窄化解释

B 的 2×2 realized-direction bridge 对 anchor property 有支持：固定 direction source 与 radius 时，GAM anchor 的 test damage 在 10/10 比较中更低。不过，同一 anchor 上 GAM realized direction 通常也远比 SGD realized direction温和，尤其在大半径，因此 direction effect 很强；bridge 显示的是 anchor 与 direction 的共同结构，不是 anchor-only 中介证明。

C 能排除“只是移动更少”这一单一解释：GAM→GAM 在 Task1–Task9 每个端点都移动更多而 test loss damage 更小。它不能排除 future GAM optimizer、完整历史训练过程或其他 package 差异。主图中的 `S_path` 应称 **Task0-to-Task9 endpoint-segment sensitivity**；不能称 optimizer-trajectory sensitivity。

## 最小剩余工作

### 若保留附件中的完整广义链

1. **A 不必默认重训，也不必补完整四格。** 对当前 ImageNet-A 表直接加入已有 `raw_trace_t1_train256`，同时保留不利的 a21/a22 和 test CE。若改用旧 ImageNet-R 正向表，就将它作为独立 setting；D1 非必须时省略，D1 必须时才重跑对应旧分支。
2. **补最小 B 诊断，不做大型 Fisher-estimator 比较。** 用已有 16 个 paired directions，在预注册的 `r={0,1,5,10,20}` 上计算两 anchor 的全 test307 raw per-example Fisher；另做一个固定预算、独立 train-calibration split 的 sensitivity-directed finite probe，并在两 anchor 上交叉施加。当前 `full_rq2` 的 `bridge_full` 不是这个 search。
3. **若文字必须保留 `larger radius`，单独执行共同、预注册的 dense loss-only radius grid。** 无需在全网格计算 Fisher。若仍不扩大，就删除该箭头；不能把后续实验描述成会确保正结果。
4. **补 C 的 Task1/Task5 endpoint chords**，与现有 Task9 使用完全一致的数据、坐标、求积和命名口径，形成早/中/晚三点。该工作复用已有 checkpoints，不需要新训练。
5. **只有要写跨训练 seed 的正式结论时，再冻结上述协议并补 matching seeds 1/2。** 多 seed 衡量复现稳定性，不能修复 seed0 已经为负的 radius 箭头。

正在执行的 λ=0 两分支不是上述最小缺口，也不应成为接受当前链的前提。

### 若接受窄化的方向依赖 case study

不需要新增训练即可收口，但主张必须改写，并完成三个 CPU/reporting 修订：

1. A 表加入已有 T1 raw trace，保留 a21/a22 的负结果；不把旧 ImageNet-R 正结果混入当前表。
2. B 图同时保留 train/test 曲线、相同的 G-tolerance radius 和 2×2 bridge；删除“一般更大容忍半径”与 worst-case 表述。
3. C 图明确区分所有 task 的 realized endpoint damage/drift 曲线与仅在 t9 测得的 endpoint chord `S_path`；明确 A 是 GAM→SGD，C 是 GAM→GAM。

可用的窄化主张为：

> 在当前视觉 GAM+EWC 的一个 ImageNet-A seed 中，GAM Task0 anchor 在固定 train256 与 sampled random directions 上具有较低 raw empirical-Fisher 和较低训练损伤，但这一优势没有扩展到 test mean positive neighborhood damage 或当前 G-tolerance radius。对于两条 realized directions，GAM anchor 的有限位移损伤较低；在独立的历史完整 GAM→GAM 十任务链中，Task9 endpoint chord 也表现出更低的单位位移方向敏感度和更小的 old-task damage，尽管 endpoint displacement 更大。这是坐标与方向依赖的描述性证据，不是完整因果中介链。

## 直接读取的主要证据

- [A_two_task.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/A_two_task.csv)
- [B_neighborhood_deterministic.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_neighborhood_deterministic.csv)
- [B_tolerance_deterministic.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_tolerance_deterministic.csv)
- [B_actual_direction_bridge.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/B_actual_direction_bridge.csv)
- [C_current_FP32_task_trajectory.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/C_current_FP32_task_trajectory.csv)
- [C_path_summary.csv](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/report_seed0/C_path_summary.csv)
- [USER_TABLE_PROVENANCE.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/table_revision_20260920/USER_TABLE_PROVENANCE.json)
- [TABLE_REVISION.md](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/table_revision_20260920/TABLE_REVISION.md)
- [EVIDENCE_PACKET.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/closure_audit_20260920_190440/EVIDENCE_PACKET.json)
- [evidence_precheck.json](/data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/mechanism_next_20260920/closure_audit_20260920_190440/evidence_precheck.json)

确定性 pre-check 仅证明所列路径、哈希和数值存在，不把语义判断升级为 accepted。
