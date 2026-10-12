# ImageNet-A 后续分量与 FD/HVP 实验队列

本地日期：2026-10-02（Europe/Madrid）。运行标识的时间戳使用UTC。此次用户授权将后续设置加入等待队列；保留现有 formation 与 component 队列的冻结配置和失败记录。

## 已经完成和正在进行

历史 seed2、r=1、rho=.05 的纯GAM十任务方向已完成原版gamma=.1、sum=g1+g3、g1、g3四组；任务等权AAA/FAA分别68.028517/55.341、67.860863/54.726、62.165531/45.132、65.993172/50.679。另已完成mean与g3半径0/.1/.25/.5，共9条历史完整轨迹。尚无同协议g0、g2、UG完整十任务结果，没有HVP修正真正用于全训练的成绩。

现有formation于UTC22:51全部120/120完成，其中24/24单Task0训练（8arms×3seeds）、84/84 future诊断和12项短测。其后接的旧component25项（10smoke+9十任务+6诊断）在UTC22:53因6项首批短测的评测解析断言停下，余19项待运行，尚无正式十任务成绩。根因是native评测返回全10个类别段，未见任务带结构性NaN，而wrapper错误地将这些未见任务一起算入当前行。

同配置的r=.25、seed2任务迁移到独立修复目录 `s3-component-horizon-recovery-20261001T230000Z`，只修正已见任务评测行提取，不修改旧失败记录、数据、训练或优化器设置。本新增队列等待修复批全部25项验收成功、controller及handoff进程退出后才放行。实时状态以各runtime文件的逐job状态为准，不能把Task0与future100步记作完整十任务。

## 新增注册范围

**共42项注册任务：22项短测、17项正式训练设置、3项checkpoint诊断。条件分支互斥或按信号选择，因此不表示42项都会运行。**

| 阶段 | 正式内容 | 短测 | 正式训练/诊断 | 接续条件 |
|---|---|---:|---:|---|
| r1核心 | g0、g1、UG=g0+(g3-g2)、g2、g3，均自然步幅 | 10 | 5次十任务训练 | 无收益门槛；父队列成功后执行 |
| UG轨迹审计 | UG自然版本Task0/4/9，FD/HVP同状态检查 | 0 | 3项诊断 | UG核心完整验收 |
| 第二次HVP | FD/FD、FD/HVP，各natural与matched | 4 | 4次十任务训练 | UG对g0达到探索收益门槛，且3项UG诊断HVP可用 |
| 第一次HVP | HVP/FD、HVP/HVP，各natural与matched | 4 | 4次十任务训练 | 第二次替换有预定义的绝对AAA差异 |
| 修正强度回查 | US与UG各lambda_corr=.5/2，自然步幅 | 4 | 4次十任务训练 | UG在lambda_corr=1下未达到收益门槛 |

r1核心短测对五种方向各验证natural/matched，以覆盖更新尺度实现；正式核心均为natural。US lambda_corr=1的自然/匹配版本保持同一名义LR更新，可作身份检查。其余方向不要求末态参数相同。

新增g1/g3与g0/g2/UG共用当前wrapper的固定数据RNG，提供新协议的严格配对。历史四组仅作为历史背景，不当作这次同批次的配对控制。

## 所有设置固定

- seed仅2，LoRA rank10；ViT-B/16，原始固定ImageNet-A 200类，10×20类。
- rho=`gam_grad_rho=.05`，r=`gam_grad_norm_rho=1`。
- 每任务50 epochs，batch128；SGD momentum=0、weight_decay=0；Task0 LR .04→.004，后续 .02→.002，cosine Tmax50。
- GAM最终gamma=0，无SGV、EWC、Fisher估计、Fisher归一化/阻尼、任务内及端点权重融合、重放或额外漂移惩罚；lambda_flat=1。
- 每task/epoch独立数据seed：`2*100000+task*1000+epoch`。配对核输入/标签/索引哈希、初始化、扩展分类头新行及名义LR。
- 每条完整训练2550步、299050样本呈现；短测两任务，每任务1epoch/1batch，仍使用Tmax50，验证扩头和旧任务保持过程。
- 所有任务先使用原生四梯度探测，加原learner独立clean-gradient pass；有HVP时额外0/1/2次前向和自动微分分别计数，不能将额外计算隐去。

## 真实HVP干预

定义g0、g1、g2、g3为当前原生GAM四位置梯度。用实际捕获的外层点theta2，而不是重新计算理想theta+r*v替代其FP32往返结果。

| 版本 | 外层方向所用响应 | UG最终更新 |
|---|---|---|
| FD/FD | g1-g0 | g0+lambda_corr*(g3-g2) |
| FD/HVP | g1-g0 | g0+lambda_corr*rho*H(theta2)*stopgrad(N_eps(g2)) |
| HVP/FD | rho*H(theta0)*stopgrad(N_eps(g0)) | 新外层点的g0+lambda_corr*(g3-g2) |
| HVP/HVP | rho*H(theta0)*stopgrad(N_eps(g0)) | 新外层点的g0+lambda_corr*rho*H(theta2)*stopgrad(N_eps(g2)) |

第一次HVP的rho乘积送入与原版相同的归一化函数，保留epsilon作用。第二次HVP的rho=.05必须保留，不能将修正放大约20倍。探针方向detach，不穿过生成探针位置的计算过程求导。额外HVP恢复进入额外求导前的参数、buffers、模式、梯度和RNG，原四探测仍保留。取得最终方向后恢复原生基点，只执行一次实际SGD。

HVP与对应FD控制**全程采用math SDPA**，包括clean pass、第一次/第二次探针及评测；禁用flash、memory-efficient和cuDNN SDPA。原native fused FD只能作为背景，不能作为math-HVP的唯一对照。HVP训练不可用或数值失败即失败，禁止静默退回FD。

所有matched臂匹配候选自身当前状态、同batch、名义LR沿原生g1的实际FP32反事实更新长度。有效坐标包含未缩放LoRA B@A和已见FC weight/bias，逐步真实应用SGD后要求相对误差<=1e-3。Hessian的参数坐标仍是LoRA A/B与FC，不能称为BA坐标Hessian。

## 在结果前冻结的条件

所有门槛读取 `FROZEN_GATES.json`；单seed探索，不用于显著性声明。

1. **UG收益信号**：新r1自然UG相对新r1自然g0，任务等权AAA至少+1pp，FAA至少-1pp。相对g1的差值也报告，但不作为这一门槛额外条件。
2. 信号通过并且UG三个checkpoint的HVP均available，接续第二次HVP的四臂。技术验收需要finite、同点CE/梯度一致、detach、恢复和SGD方向流；**不要求高cosine或低FD/HVP误差**，这些是待测量结果。
3. 信号未通过，接续两个来源各lambda_corr=.5/2的四臂；基准0复用g0，1复用US/UG。这里lambda_corr不是ewc_lambda或lambda_flat。
4. 第二次HVP自然/匹配四条都完成后，若任一尺度的HVP−FD任务AAA绝对差至少1pp，才接续第一次HVP四臂。负差也可触发进一步研究，不能描述为HVP改善。
5. 未选分支写入带配置、门槛和原始结果哈希的 `skipped.json`，数值表不填0；“未过门槛”不是修正无用的证明。HVP明确不支持属于记录的技术原因，不能填写虚构HVP训练成绩。

## 指标与诊断

主指标为任务等权AAA，另报FAA、LA、BWT、遗忘和样本加权AAA/FAA。保存每task矩阵、每epoch CE/accuracy/LR、全部任务checkpoint、每步实际BA+FC更新长度、修正与g0范数比/余弦、FD修正与实际使用修正的误差、HVP实际前向/求导次数与时间/显存。

UG诊断Task0/4取4个固定batch，Task9取3个；扫描仅第二差分的h=.05/.025/.0125，forward与central FD，对照自动微分HVP。报告相对误差E、cosine A、范数比K及近零标记，并验证同点损失/梯度和所有状态恢复。不可用记录unsupported，OOM/NaN/恢复错误作为任务失败。

## 执行与检查

新队列独立目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-fdhvp-followup-20261001T223005Z`。
只使用GPU0–6中显存<500MiB、利用率<5%的设备，最多7并行；共享已有GPU锁，不停止其他进程。父25项需要隔离Python进程读取其原validator，避免child的common.HERE污染父路径；绑定宿主hostname和PID namespace。新等待进程不分配GPU。

每组正式臂依赖自身短测完成与该组跨配置配对检查；不能用base组通过替代HVP组。发生任何真实失败，停止新增任务启动，正在运行的任务可完成并保留证据；不自动修改配置、降低batch或重试。

- `suite.json`：42项、依赖、命令与条件。
- `runtime/handoff_state.json`：等待/执行/失败状态。
- `runtime/queue_state.json`：实际pending/running/completed/skipped/failed。
- `results/horizon_metrics.csv`：仅验收完成的十任务数值。
- `results/skipped_jobs.json`：条件未选分支及证据。
- `results/progress_summary.json`：完成数与当前门槛。

额外seeds、独立两任务SS/XS/SX/XX及三任务persistence仍需十任务方向结果来确定研究对象。保持用户指定单seed，本批不登记这些未确定配置。它们与FD/HVP近似精度问题分别推进。
