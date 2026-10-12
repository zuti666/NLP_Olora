# ImageNet-A：g0+g1+g2

新增一个最终方向，在外半径r=.25和r=1各进行一次十任务训练，统一seed2/rank10/rho=.05。沿用已完成正向g012批次的模型、数据、学习率、任务切分、随机状态与预算，不新增seed或其他半径。GPU0/1空闲时并行执行。

## 探针与最终更新

N(v)=v/(||v||+1e−12)，所有范数覆盖全部可训练LoRA和分类头参数。

\[
g_0=\nabla L(\theta),\quad
 g_1=\nabla L(\theta+\rho N(g_0)),\quad
 g_2=\nabla L(\theta+rN(g_1-g_0)),\quad
 g_3=\nabla L(\theta+rN(g_1-g_0)+\rho N(g_2)).
\]

\[
U=g_0+g_1+g_2,\qquad\theta\leftarrow\theta-\eta U.
\]

g1为原GAM正向探针，不是C/D的g1−。g3仍计算以保持原四探针预算，但不进入U。每探针恢复完整模型状态并重放R0，最终恢复基点，仅保留g0后R1，再一次普通SGD。不除以3，不做最终单位化/范数匹配，不投影/分解最终梯度。只有探针构造使用N，日志中的余弦/范数用于观察。

## 固定设置

| 项目 | 设置 |
|---|---|
| 数据 | ImageNet-A train5981/test1519，固定类序0…199，10×20类 |
| 模型 | SeqLoRA，ViT-B/16 augreg2_in21k_ft_in1k，rank10 |
| Seed | 2，各组独立从相同预训练初始化 |
| 训练 | 每任务50epochs，batch128，总500epochs/2550SGD更新 |
| 优化 | SGD momentum=0/weight_decay=0 |
| LR | Task0 .04，后续 .02；逐任务cosine，最低.1倍初始LR |
| GAM | rho=gam_grad_rho=.05；r=gam_grad_norm_rho=.25/1 |
| 更新模式 | P/g0_plus_g1_plus_g2/natural |
| 关闭 | EWC/Fisher、实际权重融合、replay、原GAM正交修正、SGV、HVP、最终归一化/范数匹配 |
| 数值 | FP32，TF32off，nativeattention，create_graph=false |

保留lambda_flat=1作配置兼容；beta/gamma不进入直接U构造。实际记录每步4CE/forward、1g3、1SGD，projection/Fisher/merge/HVP/oppositeg1均0。有效配置和计数优先于文件名。

## 比较与解释

与已完成g1+g2、g1+g2−g0在相同r下比较，历史六项结果已验收，直接复用、不重复训练。

在相同theta/batch状态下，定义d1=g1−g0、d2=g2−g0：

- 旧U3=g1+g2−g0=g0+d1+d2；
- 旧U1=g1+g2=2g0+d1+d2；
- 新U=g0+g1+g2=3g0+d1+d2。

本轮进一步检查增加基点梯度保留量是否继续有益，或超过合适范围。方向与总幅度同时变化，不把自然步长成绩单独解释为纯几何机制；只扫这一个新方向。单seed不报告显著性。

已有g1+g2任务等权AAA/FAA：r=.25为66.865/50.606；r=1为68.605/54.722。已有g1+g2−g0：66.832/50.552、64.775/46.873。历史P/g2：66.663/49.937、65.735/50.250。原始来源/哈希见historical_inventory.json。

## 执行与验收

2项真实模型/数据短测（每项2任务×1epoch×1batch，2SGD/8CE），配对Task0初始参数和实际增强batch身份后，接续2项完整正式训练（各2550SGD/10200CE）。已有两批已完成，无需等待旧队列。最多2GPU并行，只用物理0/1，领取共享index/UUID锁并检查memory<500MiB、util<5%，不抢占其他任务。每GPUworker先运行seededCUDA kernel witness，再实际模型短测。所有短测、计数、恢复与配对验收通过后才启动正式；失败停止新接续并保存证据。

主指标任务等权AAA，辅指标FAA、LA、BWT、forgetting，另存逐样本口径与10阶段三角矩阵。每步保存四loss、g0/g1/g2/g3/d1/d2/U范数、余弦、实际LoRA BA+FC更新位移、完整状态/RNG恢复和真实运算计数。最终指标须完整十任务、500epochs、exit0、config/trainer/trace/matrixhash和独立重算验证通过。

源/trainer/config冻结于RUN_PINS.json，旧批次不修改。计划中的“运行”依实际runtime/queue_state.json和LAUNCH_VERIFIED.json判断；没有最终结果前不报告成绩。
