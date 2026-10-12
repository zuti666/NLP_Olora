# 完整 RQ2 队列启动前审查

本审查只读取已有计划、源码及产物，并在用户授权的空闲 GPU6 上执行原有 `env_probe.py` 的 8×8 seeded CUDA 前向/反向。未启动训练、未修改既有队列或源码。依据原附件完整计划 `../table_revision_20260920/FULL_RQ2_PLAN_082549.md`，不能把此前精简队列 15/15 完成等同于完整 B1/B2/B3 完成。

## 环境与预训练来源

- 宿主机 `cudahpc40.cvc.uab.es` 的 GPU0–6 查询时均为 NVIDIA RTX 6000 Ada、2 MiB、0% utilization，无 compute process。此快照只用于本次检查，不是未来作业自动获得设备的许可凭据；每次启动仍查询。
- GPU6 UUID 为 `GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a`。微型 witness 成功并已退出，见 [env_witness_gpu6.json](env_witness_gpu6.json)。GPU6 已释放。
- 使用既有解释器 `/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`；Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9、CUDA12.1，与此前 warm-run witness 一致。未安装或修改依赖；TMPDIR/TMP/TEMP 指向 `/tmp`。
- witness 的 `cudnn_allow_tf32=true` 是脚本记录的环境默认值，不是正式测量设置。训练和诊断入口必须继续显式设置 `torch.backends.cuda.matmul.allow_tf32=False`、`torch.backends.cudnn.allow_tf32=False`，并记录生产精度。旧训练入口已有这两行；不能拿微型 witness 代替模型/数据/梯度 sanity。
- `source_frozen.json` 列出的199个文件全部逐文件 SHA256 匹配。manifest SHA256 为 `fe3b5ef75e89bab32010c1aa9a0581cbaeb2b068eaa2f4718f11a9cc314d0a2b`。
- CPU 重新读取两份历史 Task0 checkpoint，只提取 `is_frozen_backbone` 的300个 backbone state keys；两方法均得到状态哈希 `7b139492ff24e913929bcfb88af0685b9fa21bdf3ccd587a781215eddf7912c4`。48个 LoRA A/B keys 与 `fc.weight/fc.bias` 均被排除，见 [frozen_base_witness.json](frozen_base_witness.json)。
- 因而 fresh seed 可通过旧 `install_runtime` 的方式加载这300个相同且冻结的 pretrained backbone keys，同时保留按新seed初始化的 LoRA/head。不要把整份已训练 Task0 `network_state` 用作 fresh-seed 初始化；整份权重恢复只适用于明确声明的 continuation。
- 旧代码先将 timm `pretrained=False`，随后严格加载上述300keys，避免 alias 或远程下载改变预训练版本。新入口应保留严格 key 集合检查、逐tensor相等检查和 `frozen_backbone_loaded` 收据。每个 fresh sequence 需要与其真实训练seed一致的初始化/Task0 RNG witness。

## 既有 pipeline 扩展建议

旧 `../pipeline.py` 的 `validate_plan` 硬编码 GPU0–3、`max_parallel<=4`，且 job 使用静态 `gpu`。把计划里的 GPU 列表改为0–6还不够。旧训练入口 `--gpu` 也限制0–3；本次新入口已开始独立扩展，应以最终冻结代码为准。

必要修改建议如下，审查者未直接修改 scheduler：

1. 允许的设备集合显式冻结为物理0–6及其UUID，最大并行数不超过授权池大小。区分 CPU job (`gpu=null`) 与自动选卡 GPU job，不能用同一个null同时表示两种含义。
2. 派发前从空闲且未被本队列预留的GPU选择设备，原子预留，然后设置实际 `CUDA_VISIBLE_DEVICES=UUID` 与训练命令的 `--gpu` 参数。确保两者一致。旧逻辑中 `waiting_for_gpu` 会占线程池slot，静态卡忙可使空闲其他卡无法承接任务；动态设计应让busy任务保持pending，并轮询资源。
3. 原子设备预留需兼顾同机其他本项目队列：共享的按UUID锁优于每run私有 `gpuN.lock`。子进程继承设备锁FD可覆盖scheduler退出、子进程仍在导入/运行的窗口；启动前再次查 compute process、显存和utilization。不要终止其他作业来取卡。
4. 无卡可用与依赖死锁分别处理：有ready GPU任务但当前没有空闲卡时等待；只有依赖不可能满足才报错。等待应可响应失败/停止事件。
5. 保留旧runner的失败传播：发现失败立即停止新派发、已运行作业drain，不能失败后继续放行下游。保留独立run目录、不可变plan/code fingerprints、receipt要求及receipt SHA、PID/start_ticks/boot_id身份检查和恢复时活进程拒绝规则。
6. 每个CPU汇总作业也需要receipt完整性要求。最终 `complete` 必须来自预定全部必需作业及字段，而不是仅process exit=0，或仅某个四格表已生成。已有warm结果只有在输入checkpoint、数据/head、坐标、精度、源码/协议哈希相同且receipt有效时复用。

## 最关键的实验验收 gate

1. **样本、坐标、恢复一致。** ImageNet-A Task0 train1259/test307、固定20-class old head、24个Q/V有效BA矩阵、无诊断数据增强。保存样本ID、类别顺序、head/坐标/权重哈希；r0重复与扰动恢复一致，s0/s1能恢复真实endpoint。raw Fisher必须逐样本真实标签CE梯度，完整方向量是平均 `(g_i·Delta)^2`；训练batch-proxy及归一化缓存另存。
2. **三seed科学身份。** seeds0/1/2均按同一正式协议；同seed方法配对初始化、数据顺序、可控增强和新head。历史seed0与新训练存在精度/RNG差别时，不能悄悄混成同协议n=3。完整新链与warm先导结果应明确分层，或者先通过预先定义的复用一致性条件。
3. **B3四格。** 每seed拥有SGD/GAM Task0 × Task1 SGD、lambda0/2000四格；相同anchor的两种protection必须用同一checkpoint和Fisher缓存。保存50epoch完成、真实lambda、配对收据、a11/a21/a22/BWT、train/test旧head CE及D、新任务学习量、有效W位移与旧任务raw trace。D不是完整持续GAM方法。已测C/D有Fisher/位移，未完成A/B不得用NA占位后宣称四格完成。
4. **B1正式覆盖。** 每seed方法共享同一K100方向库；dense loss半径0:.25:20、train/test分别评估且保留逐方向原始值；sparse train256为r0+6×16=97状态/anchor；test307为r0+3×16=49状态/anchor。规定r0/1/10/20的固定方向子集还需全train1259核对。真实方向跨anchor交叉，并加入各自真实位移半径及相同半径随机对照。dense-only分支不应对8001状态逐点额外计算Fisher。
5. **B2正式覆盖。** 每seed每方法t0..9全部endpoint使用train1259测旧任务raw trace；test raw trace至少t0/1/5/9。t1/5/9各自真实有效W差值形成直线；正式路径同一train1259，9节点与5节点复合Simpson比较，不稳定继续17/33节点。求积误差阈值、bound数值容忍及加密规则预先冻结；`D²`与`I=integral q_Delta ds`用完全相同数据/head，不把subset D与full F混用；记录 `||Delta||² integral trace ds` 和所有负面结果。求积未收敛应清楚标识未通过数值gate，不能报告机制已验证。
6. **汇总不按结果选取。** 训练seed是统计单位；先每seed汇总，再mean及sample SD。K16/K100方向不能充作训练重复。train/test、signed/positive damage、norm/norm²、raw/normalized/batch proxy分别命名。所有预定epsilon和半径、负面seed/test均保留。机制结论须由结果判断，不作为队列“通过”的条件。

ImageNet-R full-model复验在原附件属于条件支持性工作；若根计划把其列入本次必做范围，则应有独立参数范围和receipt，不能用LoRA表代替，也不能把full与LoRA位移绝对值混做参数化因果比较。

本预检确认既有环境和冻结基底可用，并指出scheduler与完整覆盖所需gate；它不等于尚在编写的新训练/诊断脚本已通过完整运行验收。新代码冻结后仍需dry-run、针对新增调度/receipt逻辑的检查和正式sanity。
