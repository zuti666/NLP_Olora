# Llama-1B GAM-EWC：四卡、有效 batch64、seed1993

日期：2026-09-19。用户授权在当前服务器四张空闲 GPU 上重跑旧 GAM-EWC，改为有效 batch64，其余沿用旧实验，输出独立保存。三个 Order 均从原始基座开始，四任务串行；先完成两阶段真实 BF16 smoke。

|项目|新运行|旧 GAM-EWC|
|---|---|---|
|GPU / 每卡batch / GA|4 / 1 / 16|5 / 1 / 8|
|有效batch|64|40|
|seed / data_seed|1993 / 1993|1993 / 1993|
|精度|BF16 AMP，参数FP32|BF16|
|模型|本地 Llama-3.2-1B|同名旧路径；历史权重hash缺失|
|方法|ewclora_gam；纯 IncLoRA-GAM-EWC|ewclora_gam|
|LoRA|r8 / alpha32 / dropout0.1，q_proj/v_proj|相同|
|训练|每任务1epoch；首任务LR1e-3，其余1e-4；constant/warmup0|相同|
|长度|source512 / target50 / generation50|相同|
|EWC|lambda1 / gamma1 / project_eps1e-12|相同|
|GAM|rho0.02 / norm_rho0.2 / grad_gamma0.03；betas1/1/1；非adaptive；grad_reduce mean；perturb_eps1e-12|相同|
|基础优化器|实际 torch.optim.AdamW；betas0.9/0.999，eps1e-8，weight_decay0.01|候选历史源码相同|
|Fisher|全训练集 raw Fisher，禁用归一化和damping|相同估计规则|
|OLoRA正交/L2|显式系数0/0，并由独立opt-in路径跳过额外项|历史EWC-GAM不使用日志中的默认OLoRA参数|

GAM在每个optimizer update的clean point单次计算EWC梯度，四个closure均按GA平均CE。因此GA8→16仍保持lambda1，不能套用纯EWC的lambda0.5补偿。

保持历史local-GAM：每rank在自己的16个microbatch上进行四次GAM probe，投影EWC后只同步最终梯度。这给出总batch64，但不意味着与8卡各8个microbatch的非线性GAM数学等价，也不与seed42基线或新纯EWC合并。

任务顺序：O1 DBpedia→Amazon→Yahoo→AGNews；O2 DBpedia→Amazon→AGNews→Yahoo；O3 Yahoo→Amazon→AGNews→DBpedia。正式数据14000/5000/10000/4000，更新数218/78/156/62（DBpedia/Amazon/Yahoo/AGNews）；沿用旧完整GA窗口策略。Fisher仍使用全部训练样本。最终各任务测试7600条、每order30400条。

环境复用 `.aris/compute/nlora_req_local_spec.json` 的0715917e，无包重装。当前设备为RTX6000 Ada，启动时重新选择空闲卡并按UUID加锁，检查原生BF16与seed1993 matmul/backward。使用现有nlora_req解释器、本地PEFT和离线数据/模型。

每阶段检查：真实DDP4、BF16 forward、rank/device与采样分片、实际TorchAdamW默认值、四closure/update和每closure16microbatch、最终梯度同步计数及首末步跨rank哈希一致、非零GAM扰动、后续任务非零EWC缓存与投影作用、全数据Fisher、adapter rank8→16→24→32、历史adapter不变、预测计数和EM重算。

历史GAM的本地FP32扰动加减会产生微小rank参数舍入差异，因此记录参数hash、最大绝对差，并用allclose(atol1e-6,rtol1e-5)检查参数；初始参数和已同步梯度要求哈希一致。只广播审计用副本，不修改或广播训练权重。CPU舍入见证在 `GAM_FLOAT32_UNPERTURB_WITNESS.json`。

保存到新的 `experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_<UTC时间>/`，拒绝已存在目录。保存冻结源码/config、输入hash、运行环境、逐阶段命令/日志、逐rank审计、adapter/Fisher、预测和独立结果JSON/CSV/LaTeX。旧batch40 GAM指标和日志只复制到明确标注的historical_gam_b40_reference；不写回旧实验。

历史执行源码hash缺失；候选git898adf6与旧成功日志行号及GAM文件相符，具体证据见 SEMANTICS_REVIEW.md/json。新队列每个正式阶段必须通过验证后才进入下一阶段；失败即停止并保留失败目录。
