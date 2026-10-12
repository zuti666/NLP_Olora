# ImageNet-A：两组状态形成实验的配置、目的、结果和代码

整理日期：2026-10-12。第一组18/18正式训练、第二组6/6正式训练全部验收；两组各自使用配对seed0/1/2，不合并统计。第二组正式Hessian诊断未通过数值前提，机制证据仍不完整。本次仅整理已有代码和结果，没有启动新训练或修改冻结代码。

**两组实验研究的是不同位置的干预。** 第一组改变旧任务Task0的形成方式，随后整个Task1统一SGD；第二组固定同一个旧任务父模型，改变新任务Task1前半程，再统一SGD。第一组主要观察旧知识形成后的保持；第二组检验新任务更新所形成的状态及其持续影响。

| 项目 | 实验一：Task0形成与使用阶段 | 实验二：Task1状态形成及共同SGD后续 |
|---|---|---|
| 干预位置 | Task0 | Task1前25轮 |
| Task0 | 6种策略，每种独立训练20轮 | 每seed共用已验收的SGD50轮父模型 |
| Task1 | 所有组SGD20轮 | SAM或S_F g2训练25轮，再SGD25轮 |
| 方法切换 | Task0第11轮 | Task1第26轮 |
| 新训练规模 | 6组×3seed=18条两任务训练 | 2分支×3seed=6条Task1训练 |
| 方法内切换 | 保留模型/optimizer/scheduler/RNG，日程不重启 | 同左；还核对对象ID、参数引用及共享起点 |
| 主要基线 | Task0 SGD20→Task1 SGD20 | 共享Task0父模型后的SAM25→SGD25 |
| 主要问题 | 旧知识形成方式及早晚使用阶段是否影响后续保持？ | 特殊更新停止后，性能和有效状态差异是否持续？ |
| 完成情况 | 正式18/18；全部25个准备/短测/正式任务通过 | 正式6/6；正式离线0/3通过，seed2失败、0/1未启动 |

**两组共同固定的基础设置。**

| 项目 | 实际设置 |
|---|---|
| 数据 | ImageNet-A原有200类固定顺序0…199、10×20类划分的前两个任务；不是把全部200类重分为两个任务 |
| 类别范围 | Task0：0–19；Task1：20–39 |
| 固定train/dev | Task0 1141/118；Task1 906/91；dev从原train划出，各方法和seed不重划 |
| 官方test | 旧任务307张，新任务254张；不进入训练、探针或checkpoint选择 |
| 模型 | ViT-B/16，augreg2_in21k_ft_in1k预训练；主干冻结 |
| 可训练范围 | 全12块的Q/V LoRA，共24个模块，rank10；完整当前分类头 |
| LoRA有效权重 | W0+BA，实际缩放s=1，无额外alpha/rank |
| Batch | 128，drop_last=false；Task0每轮9批，Task1每轮8批 |
| 真正optimizer | torch.optim.SGD，momentum=0、weight_decay=0 |
| 初始LR | Task0 .04；Task1 .02 |
| Scheduler | 每任务连续epoch cosine，T_max等于本任务20或50轮；最低目标为初始LR的.1 |
| 损失 | 当前任务20类mean CE；Task0 logits[:,:20]，Task1 logits[:,20:40]、targets−20 |
| 正式准确率 | Task1阶段旧/新样本均用共同40类分类头，不提供task ID |
| 精度 | FP32、确定性运行，AMP/TF32关闭；沿用原attention与数据变换 |
| 关闭组件 | EWC/Fisher、replay、历史梯度、模型/权重融合、投影、clip、最终梯度范数匹配 |
| 方法比较 | 各方法按所需探针实际执行，SGD/SAM/S_F/残差GAM分别1/2/3/4次梯度读取；不是统一四探针预算 |

训练虽然把完整分类头加入optimizer，但Task1当前20类CE对旧分类头行没有直接梯度。无动量与WD时，这些旧行不因Task1 SGD直接改变；旧任务joint40表现还取决于共享LoRA/特征和新分类头竞争。不能把joint40准确率下降全部解释为旧类内部判别能力下降。

**实际算法定义。** 令θ为所有可训练LoRA因子与FC参数的拼接，g(x)=∇L_B(x)，η为当前真实optimizer学习率。所有探针使用同一已经增强的mini-batch，并恢复同一模型状态和RNG。最终只在原基点提交一次θ←θ−ηU；没有在探针点上更新或额外归一化U。

| 方法 | 探针与最终方向 | 梯度读取 |
|---|---|---:|
| SGD | g0=g(θ)，U=g0 | 1 |
| SAM | g1=g(θ+.05N(g0))，U=g1 | 2 |
| S_F g2 | g1⁻=g(θ−ηg0)，v=N(g0−g1⁻)，g2=g(θ+.25v)，U=g2 | 3 |
| 残差GAM | g1=g(θ+.05N(g0))；g2=g(θ+1.0N(g1−g0))；g3=g(θ+1.0N(g1−g0)+.05N(g2))；U=g0+(g3−g2) | 4 |

S_F的内探针距离a=η||g0||，仍随cosine和梯度大小变化；外半径固定.25，从原基点出发。残差GAM的内半径.05、外半径1.0；其g2来自上升SAM响应，**与S_F下降响应构造的g2不是同一个定义**。两组实验中的S_F则是同一实现。

SAM/残差GAM的方向单位化使用v/(||v||+1e−12)；S_F在范数高于阈值时直接除以范数，保留原实现的退化检查。正式验收不默许回退混入；失败/非有限/回退记录须单独检查。已验收正式数据没有该问题。

两组的independent_response_step.py、upward_residual_step.py、response_step.py、state_transaction.py、effective_step.py逐字哈希相同，见SOURCE_AND_RUN_MANIFEST.json。区别在训练控制器、任务/阶段与预算，不是重新实现了一个不同的S_F。

**实验一：改变Task0形成方式。** 六组均从相同seed对应的初始化开始，训练Task0独立20轮，再沿用原生顺序LoRA生命周期扩头20→40并训练Task1 SGD20轮。每个任务建立其普通optimizer及cosine日程；Task0内部第11轮切换只改变最终方向，不创建新的optimizer或日程。

| ID / condition | Task0 e1–10 | Task0 e11–20 | Task1 e1–20 | 每seed真实SGD提交 / 梯度读取 |
|---|---|---|---|---:|
| 1 / sgd | SGD | SGD | SGD | 340 / 340 |
| 2 / sam | SAM | SAM | SGD | 340 / 520 |
| 3 / gam_residual | 残差GAM | 残差GAM | SGD | 340 / 880 |
| 4 / sf_g2 | S_F g2 | S_F g2 | SGD | 340 / 700 |
| 5 / sf_g2_then_sgd | S_F g2 | SGD | SGD | 340 / 520 |
| 6 / sgd_then_sf_g2 | SGD | S_F g2 | SGD | 340 / 520 |

正式合计6120次SGD提交、10440次训练梯度读取。六组短测和配对门槛单独统计，不加入正式结果。GPU配对检查验证1与6、4与5的短测前10轮轨迹完全一致，每轮仅1批；完整340批正式输入的配对另由完成审计核对。同seed起点、输入顺序、增强和新FC行配对。

问题一比较2/3/4与1：不同形成方式能否留下更利于旧任务保持的模型？问题二比较5与6：同样90次Task0 g2更新，放前半还是后半更有效？问题三比较5/6与4：只用10轮g2能保留多少全程收益？早晚比较同时改变了η及a=η||g0||，因此是阶段策略对比，尚未分离阶段与探针尺度。

以下均为3seed均值±样本标准差，准确率%，下降量百分点。

| Task0策略 | Task0终点20类 | 扩头后旧起点40类 | 最终旧任务 | 最终新任务 | FAA | 旧下降↓ |
|---|---:|---:|---:|---:|---:|---:|
| SGD→SGD | 82.628±1.672 | 82.302±1.672 | 78.284±1.504 | 72.441±.394 | 75.363±.928 | 4.017±2.488 |
| SAM→SAM | 82.953±1.356 | 82.519±1.233 | 78.719±.820 | 72.441±.000 | 75.580±.410 | 3.800±1.469 |
| 残差GAM→GAM | 82.736±1.128 | 82.085±1.420 | 78.827±.977 | 71.260±.682 | 75.044±.258 | 3.257±1.629 |
| S_F g2→g2 | 83.822±1.504 | 83.279±1.316 | 80.565±.995 | 71.916±.601 | 76.240±.434 | 2.714±1.794 |
| S_F g2→SGD | 83.388±.862 | 82.736±.862 | 80.347±1.047 | 71.916±.227 | 76.132±.636 | 2.389±1.672 |
| SGD→S_F g2 | 83.062±.977 | 82.628±.820 | 79.045±1.794 | 72.310±.227 | 75.677±1.000 | 3.583±1.693 |

全程S_F相对SGD的配对FAA增益+.878±.728，三个seed均为正；最终旧任务+2.280±.651、旧下降−1.303±.862，三个seed也均改善。旧终点优势可以分解为**起点+.977与后续下降减少1.303**，所以不只是Task0初始准确率更高。新任务−.525±.820，存在保持与学习取舍。

早期g2相对晚期g2：最终旧任务+1.303±.862、旧下降−1.194±.498，均3/3改善；FAA+.455±.431，2/3改善。早期组相对全程组FAA仅−.109±.477，支持保留了较多收益，不能宣称等价。早期组下降最少，但其旧起点比全程低.543，旧终点仍低.217，说明单看下降量会误判。

SAM相对SGD的FAA+.217±.550、残差GAM−.319±.853，均仅1/3seed提高，没有一致整体优势。

**实验二：固定Task0父模型，研究Task1形成和持续影响。** seed0/1/2各复用已经完成的Task0 SGD50模型，不加载实验一的20轮模型。每seed只执行一次扩头20→40及原生任务边界处理，保存完整common_start，再复制给S/R；分支恢复后跳过再次扩头，避免第二次初始化新FC。

| 分支 | Task1 e1–25 | Task1 e26–50 | 每seed新增SGD提交 / 梯度读取 |
|---|---|---|---:|
| S | SAM，ρ=.05 | SGD | 400 / 600 |
| R | S_F g2，内θ−ηg0、外r=.25 | SGD | 400 / 800 |

新增正式训练合计2400提交、4200梯度读取。第26轮LR=.011，T_max=50、scheduler last_epoch=25；模型值、参数引用、optimizer/scheduler对象及RNG保持。全部400个增强batch和共享首g0配对。e25/e50为预定主时间点。

| 分支 | Task1 epoch | 旧任务 | 新任务 | FAA | 旧下降↓ |
|---|---:|---:|---:|---:|---:|
| S：SAM→SGD | 25 | 81.433±2.280 | 76.115±2.621 | 78.774±.741 | 3.040±1.672 |
| R：S_F g2→SGD | 25 | 81.759±.862 | 78.215±2.406 | 79.987±.779 | 2.714±.820 |
| S：SAM→SGD | 50 | 80.782±1.629 | 78.346±1.420 | 79.564±.212 | 3.692±1.233 |
| R：S_F g2→SGD | 50 | 81.216±1.047 | 80.184±.820 | 80.700±.443 | 3.257±.977 |

FAA配对优势从e25的+1.213±.832到e50的+1.136±.463，两个时间点均3/3seed提高。共同SGD阶段保留优势，平均没有放大。最终主要来自新任务+1.837±1.639；旧任务+.434±.752，仅seed0提高、seed1/2完全持平，因此不是一致减少遗忘的证据。

R最终clean train CE .142302±.000895高于S .123717±.003122，new20条件化test CE配对更低−.01401±.00653；预登记.05宽拟合箱共有31个共同首次进入记录，R条件化test CE均更低。匹配是近似的，epoch/LR可不同，31个箱不是31个独立seed，不能据此做独立样本显著性判断。

直接对比有效LoRA BA及FC，e25两分支联合距离.41868±.01550，e50 .41701±.01474；e50 LoRA BA差.28239±.00902、FC差.30684±.01181。有效模型差异在共同SGD后仍然存在，不只是因子分解坐标不同。分类头也明显不同，不能把收益唯一归因于LoRA。第二组没有Task1全程SGD分支，不能据此声称优于该基线；计划的同状态步幅匹配仅为离线候选诊断，也不等于完整的匹配步幅训练对照。

**第二组三层证据的记录和当前边界。**

1. 局部算法对应：S/R×e1/e10/e25×2个128样本固定训练batch×3seed，计划36组合；原生(a,r)及联合减半(a/2,r/2)共72条修正记录。用精确loss Hessian H，不用Fisher/GGN；q=Hg0、p=H stopgrad(q)，同基点计算，比较g2−g0与r p/||q||。
2. 状态形成：每轮记录所有24个Q/V模块的Q=BᵀB−AAᵀ、Qhat及相对共享起点的漂移，跨层median/IQR。固定首/中batch将实际Δ(BA)精确拆为BΔA+(ΔB)A+ΔBΔA，并记录ΔQ一阶/二阶项、分量范数/夹角、πA和FC更新。被动统计不新增反向传播。
3. 共同后续：每轮记录clean current train CE/accuracy、旧/新joint40 test CE/accuracy；e25与e50配对分析；保存每轮LoRA/FC，e0/e1/e10/e25/e50保存可恢复完整运行状态。另计划e10/e25共24组同状态有效步幅匹配，SAM候选保持原幅度，g2候选求统一标量使完整BA+FC有效位移相等，只离线比较，不提交训练。

正式离线第一点seed2/S/e1/batch0：原生重复g一致，math/microbatch16一阶梯度相对误差2.098e−5通过；登记距离.001/.003/.01下Hg0误差.008601/.082600/.111084，H detach(Hg0)误差.501514/.975109/.984655，不能满足每方向两个相邻距离均≤.08。按冻结规则退出2，seed0/1未接续；只尝试1/36组合，0/72正式修正及0/24有效匹配结果验收。无OOM/NaN，未发现明确实现错误，训练成绩仍有效，**目前不能确认H²g机制或排除更新幅度解释**。

**两组共同的指标口径。**

FAA=(A_old,end40+A_new,end40)/2，为任务等权。旧下降F=A_old,start40−A_old,end40，起点是扩头后、Task1尚未更新时。

第一组还保存扩头影响D=A_Task0,end20−A_old,start40。传统BWT=A_old,end40−A_Task0,end20，会包含扩头类别竞争；训练期BWT=−F。可选阶段AAA=(A_Task0,end20+FAA)/2，与逐epoch平均或此前十任务AAA不同。20轮和50轮两组不能合并、不能直接用FAA绝对值比较方法提升大小。

**代码定位与实际执行细节。** 以下A目录为实验一，B目录为实验二：

`A=/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z`

`B=/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z`

| 功能 | 实际代码 |
|---|---|
| A六组及Task1强制SGD | [formation_step.py:20](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_step.py:20)，active_method |
| A核对生效配置/普通SGD/当前20类CE | [formation_train.py:56](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_train.py:56)，resolve_config、Controller.install |
| A第11轮连续切换、指标与checkpoint | [formation_train.py:360](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/formation_train.py:360)，begin_epoch及后续save_task_checkpoint |
| A原始结果汇总和验收 | A/collect_results.py、A/validate_job.py、A/results/status_gpu6_20261011T064933Z/INDEPENDENT_COMPLETION_AUDIT.md |
| SAM及残差GAM真实探针 | [upward_residual_step.py:64](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/trainer/upward_residual_step.py:64)，shifted和compute |
| 两组同一S_F内/外探针 | [independent_response_step.py:127](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/independent_response_step.py:127)，compute；143取g0−g1，151从base构造外点 |
| 真正一次SGD提交 | [response_step.py:411](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/response_step.py:411)，apply_update，写入U.grad、optimizer.step并逐bit核对θ−ηU |
| 全状态/RNG恢复 | [state_transaction.py:54](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/state_transaction.py:54)，State；每次虚拟读重放R0，只提交基点读的R1 |
| B分支与e26方法切换 | [chain_step.py:9](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/chain_step.py:9)，active_method、ChainStep |
| B一次扩头后的共同起点及分支恢复 | [response_chain_train.py:163](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/response_chain_train.py:163)，save_common_start、restore_common；prepare任务调用继承的restore_parent，分支第二次update_fc严格no-op |
| B第26轮连续性验收 | [response_chain_train.py:211](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/response_chain_train.py:211)，begin_epoch |
| B clean训练/条件test/每轮保存 | [response_chain_train.py:183](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/response_chain_train.py:183)，evaluate_clean_train、finish_epoch、save_epoch_checkpoint |
| B所有层因子及实际更新精确分解 | [chain_statistics.py:15](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/chain_statistics.py:15)，factor_epoch、update_decomposition |
| B exact HVP及门槛 | [offline_core.py:100](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/diagnostics/offline_core.py:100)，hvp、numerical_gate、correction_diagnostic |
| B离线有效步幅匹配 | [offline_core.py:256](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/diagnostics/offline_core.py:256)，matched_effective_diagnostic |
| B配对/预算/完整性验收 | [validate_chain.py:41](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/trainer/validate_chain.py:41)，validate_output、validate_pair |

真正控制阶段的方法选择代码为：

```python
# 实验一：FormationStep 调用此函数，task==1永远是SGD
if task == 1:
    return "sgd"
if condition == "sf_g2_then_sgd":
    return "sf_g2" if epoch <= 10 else "sgd"
if condition == "sgd_then_sf_g2":
    return "sgd" if epoch <= 10 else "sf_g2"
return condition
```

```python
# 实验二：ChainStep，S/R只在前25轮不同
method = "sgd" if epoch > switch_after else (
    "sam" if branch == "S" else "sf_g2"
)
```

实际S_F几何的核心代码直接来自IndependentResponseStep：

```python
future = tuple(v.clone().add_(g, alpha=-eta)
               for v, g in zip(base, g0))
loss1, _, g1, _ = probe(future)
response = tuple(a - b for a, b in zip(g0, g1))
relocation = tuple(v.clone().add_(a, alpha=outer_length / response_norm)
                   for v, a in zip(base, response))
loss2, _, g2, _ = probe(relocation)
U = _clone(g2)
```

此摘录省略了退化/有限性/状态恢复/日志检查；完整实现见上方源码。outer_length=.25，内层是−ηg0，不是−.05N(g0)，外层也不是从future继续移动。

**生效配置与保留字段。** 两组都沿用历史模型工厂，因此JSON里仍可能有model_name含ewc、optimizer_type="gam"、gam_grad_norm_rho=1或response_method="sgd"等历史字段。这些不等于实际调用EWC/GAM优化器，也不等于S_F外半径1。

- 实验一由formation_condition_id、formation_task0_schedule、formation_task1_method和FormationStep选择方法；configured_beta_formula_active/configured_gamma_projection_active关闭。
- 实验二由chain_branch、chain_switch_after_epoch和ChainStep选择方法；R的fixed_outer_probe_radius=.25在实际kernel配置中确定。
- 逐step active_method、gradient_formula、inner/outer_actual displacement及组件计数是核对实际执行的依据。
- device=[0]是进程内逻辑CUDA编号；真实物理卡由worker的CUDA_VISIBLE_DEVICES和GPU UUID记录，不能据此说使用物理GPU0。

两组共同的关闭标志包括ewc_lambda=0、ewc_skip_task_fisher=true、approximate_fisher_in_task=false、in_task_weight_merge=false、task_endpoint_weight_merge=false；运行时禁用guard和逐步计数共同确认相关计算未执行。HVP第二组只在离线脚本中存在。

**复现入口与产物。**

实验一配置样例：[formal_sf_g2_then_sgd_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/configs/formal_sf_g2_then_sgd_s2.json)；CLI入口A/trainer/formation_train.py，参数为--source、--config、--output-dir。正式输出排除--max-batches=1短测。

实验二配置样例：[formal_R_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/configs/formal_R_s2.json)；CLI入口B/trainer/response_chain_train.py，参数同上；prepare_start配置先产出common_start.pt，再运行分支。

两组真正执行过的全部18+6配置、argv、输出目录和SHA均保存于本目录SOURCE_AND_RUN_MANIFEST.json；原suite.json记录完整依赖。57/51个执行冻结文件此次全部哈希一致。原生source190位置为/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source，均用同一冻结框架。既有warm环境为/home-local/liying/.conda/envs/Pilot_new_local/bin/python。

- 第一组详细结果：[RESULT_ANALYSIS.md](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/results/analysis_20261011T194904Z/RESULT_ANALYSIS.md)；[学习与保持曲线PDF](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-task0-formation-20ep-gpu456-20261010T205804Z/results/analysis_20261011T194904Z/FORMATION_RETENTION_CURVES.pdf)。
- 第二组详细结果：[CURRENT_STATUS_AND_ANALYSIS_20261012.md](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/CURRENT_STATUS_AND_ANALYSIS_20261012.md)；[性能与拟合曲线PDF](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/audits/status_20261012/factor_fit/PERFORMANCE_AND_FIT_20261012.pdf)；[因子与有效状态证据PDF](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-response-state-chain-gpu456-20261011T195426Z/audits/status_20261012/factor_fit/FACTOR_FIT_EVIDENCE_20261012.pdf)。
- 本目录STUDY_1/2_RAW_RESULTS.csv是原始逐seed表副本；STUDY_1/2_SUMMARY.csv与RECOMPUTED_STATISTICS.json为本次重算的各组均值/样本SD、配对差与胜负，不混合研究协议。

**共同允许的研究表述。** 在这两个配对、两任务协议中，响应重定位的原生g2更新改变了模型形成，部分性能差异能够持续至共同SGD后续训练。旧任务形成实验主要表现为更好的旧任务保持；新任务阶段切换实验主要表现为更好的新任务测试。两者提供互补证据，但不能把两套绝对成绩合并、宣称路径独立于终点决定性能、因子平衡是唯一原因、收益与步幅无关，或将尚未通过的H²g数值诊断当作已验证机制。

