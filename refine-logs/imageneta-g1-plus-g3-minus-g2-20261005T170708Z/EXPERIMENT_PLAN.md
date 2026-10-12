# ImageNet-A: g1+(g3−g2)

用户新增一个更新方向，在当前 r=0.25 和 r=1.0 两个设置各完成一次十任务训练。先进行两个短测和输入/初始状态配对检查，再接续两项正式训练。seed=2，LoRA rank=10，内层 rho=0.05。未找到此方向已有完整结果。

## 梯度与最终更新

令 N(v)=v/(||v||+1e−12)，phi 为同一训练 batch 的原参数：

- g0=grad L(phi)
- phi1=phi+0.05 N(g0)，g1=grad L(phi1)
- phi2=phi+r N(g1−g0)，g2=grad L(phi2)
- phi3=phi2+0.05 N(g2)，g3=grad L(phi3)
- U=g1+(g3−g2)，恢复 phi 后执行 phi←phi−eta U。

四次普通 CE 梯度、一次自然 SGD。g3 在外层位置 phi2 上继续作正向内层探针，不在 phi 或 phi1 上计算。各探针恢复完整状态并重放同一 batch 的 R0，最终 RNG 取 g0 后的 R1。方向不平均、不做最终范数匹配或单位化；只在构造探针时使用 N。投影调用为零。

## 固定设置

| 项目 | 固定值 |
|---|---|
| 数据 | ImageNet-A；train5981/test1519，固定类序0…199，10×20类 |
| 模型 | ViT-B/16 augreg2_in21k_ft_in1k，SeqLoRA rank10 |
| Seed | 2；每组独立从相同预训练初始化 |
| 训练 | 每任务50epochs，batch128，SGD momentum=0/weight_decay=0 |
| LR | Task0=.04，后续任务=.02；逐任务 cosine，最低为各自初始LR的.1 |
| GAM探针 | gam_grad_rho=.05，gam_grad_norm_rho=.25或1 |
| 最终更新 | P/g1_plus_second_difference/natural；直接构造U |
| 关闭模块 | EWC=0，skip_task_fisher=true，Fisher近似/归一化/阻尼、权重融合、replay、HVP、正交投影、SGV、最终匹配均关闭 |
| 数值 | FP32，TF32关闭；原生attention；无二阶autograd |

config 中保留的 beta/gamma 字段不参与此直接方向分支；最终方向以 four_probe_train.py / direction_step.py 为准。

## 新增队列

| 作业 | r | 预算 | 接续条件 |
|---|---:|---|---|
| smoke_g1_d32_r025_k10_s2 | .25 | 2任务×1epoch×1batch | 空闲GPU |
| smoke_g1_d32_r1_k10_s2 | 1 | 同上 | 空闲GPU |
| smoke_pairing_gate | 两组 | CPU输入/初始状态核验 | 两个短测成功且日志验收通过 |
| formal_g1_d32_r025_k10_s2 | .25 | 10任务×50epochs，共2550SGD更新 | 配对检查通过 |
| formal_g1_d32_r1_k10_s2 | 1 | 同上 | 配对检查通过 |

只等待物理GPU2–6，每卡空闲后独立接续；GPU0/1排除。最多同时两个GPU作业，共享index/UUID锁，不中断其他队列。短测或验收失败则停止新增接续，保留失败日志。

## 指标与比较

主要报告任务等权 AAA、最终任务等权 FAA；另存最终逐样本准确率、LA、BWT、forgetting 和三角任务矩阵。每步存四个loss、g0/g1/g2/g3及delta32/U范数、余弦、探针在有效权重空间的位移、实际更新幅度，以及真实forward/gradient/SGD/Fisher/融合/投影计数。逐任务完成状态与全预算核对后才接受最终指标。

已有 R0-replay g1+ 自然步长参考 AAA/FAA=63.234/44.578，可检验加入第二局部差分是否改善 g1。参考的最终公式与r无关，但多余探针序列与本组不同，因此不主张GPU位级等价。

A_C 的公式为 g0+g2(C)−g1−：它使用反向首探针和跨点残差。当前组使用正向首探针、以 g1 为更新基点、加入外层位置的局部差分 g3−g2，没有投影。与正在运行的 g0+B 也不同，后者只计算三个梯度并投影 g2−g1。

旧 UG=g0+(g3−g2)、旧Sum=g1+g3不等于本方法，历史native五CE/RNG协议结果只作背景。方向与幅度同时变化，单seed结果不能单独归因于几何机制。完整历史索引见 historical_inventory.json。

代码/配置冻结于 RUN_PINS.json；启动说明见 RUN.md，环境见 ENVIRONMENT.md；实时状态 runtime/queue_state.json；完成后 summary.csv / RESULTS.md。

