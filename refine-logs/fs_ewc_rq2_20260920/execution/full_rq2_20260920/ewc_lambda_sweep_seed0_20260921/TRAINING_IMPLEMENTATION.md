# Seed0 SGD→SGD：Vanilla 与 raw-Fisher EWCLoRA

本入口只复用正式`../formal_training/anchor_sgd_seed0`的已训练Task0权重，不重训Task0。Task1开始前，必须将旧归一化Fisher**全部替换**为准备阶段实际重新计算的原生、未归一化Task0 Fisher；不通过反乘旧trace恢复矩阵。Task1的`ewc_normalize_fisher=false`和`ewc_normalize_accumulated_fisher=false`固定不变。λ0为Vanilla；λ>0为EWCLoRA。优化器全程SGD，不进行GAM扫参。旧runner、冻结source和历史结果均不修改；先前未启动的normF候选位于`before_raw_fisher`，旧`cpu_validation`记录不代表rawF验证。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python train.py \
  --seed 0 --ewc-lambda 1000 --run-name raw_sgd_ewc_lam1000_seed0 --gpu 0 \
  --prepared-fisher prepared_fisher/fisher_state_raw.pt
```

示例相对路径以本目录为cwd。`--prepared-fisher`默认绝对路径为本目录`prepared_fisher/fisher_state_raw.pt`；其相邻文件必须为`PREPARED.json`。`--seed0`可替代`--seed 0`。`arm=ewc_lambda_sweep`、`mode=continuation`、`anchor_method=sgd`、`future_optimizer=sgd`固定。λ允许任意有限实数`0 ≤ λ ≤ 1e9`，入口不硬编码最终网格。正式Task1固定50轮，batch128、lr .02、momentum0、weight_decay0、cosine、workers4、全FP32和两处TF32关闭；lambda_flat=0、eta=0、gamma=.9，无投影、裁剪、双对偶调λ或权重合并。

准备接口约定：

- `.pt` payload含`fisher_past_delta`（24个raw Q/V tensor）、`raw_trace`和`stats`。允许准备脚本额外提供原生sidecar兼容字段。
- `PREPARED.json`需`status=complete`、seed0/task0、`parent_checkpoint_sha256`、`payload_sha256`、`source_manifest_sha256`、两normalize=false、`fisher_mode=batch_empirical`、`fisher_source=fit`、`tensor_count=24`、`numel`、`raw_trace`、`weights_unchanged=true`及`exact_inputs_receipt.matched=true`。
- 入口核验所有fingerprint和receipt身份。真实加载前还检查24个tensor的键/shape与原父Fisher一致、有限、非负、总元素数相同，且直接计算的trace与payload和receipt一致；不对新Fisher做归一化或缩放。
- 原父manifest/COMPLETE、源文件、完整网络、旧Fisher、metrics和RNG/runtime receipt仍按原协议检查。先逐张量验证原Task0完整恢复，再用新raw字典替换旧Fisher，清除旧normalized trace-mass，记录`prepared_raw_fisher_loaded`receipt。运行时实际λ、SGD和两false flags均断言验证。

`--print-config`在准备文件尚不存在时允许预览，identity明确`prepared_fisher.status=pending`及`execution_ready=false`；这不是可运行实验身份。实际训练和`--validate-cpu`都要求准备文件完整存在，缺失时立即失败。真实rawF CPU恢复检查必须在准备完成后运行，验证新rawF加载及Task1 head初始化，不做forward/backward。

`--smoke`只训练Task1一轮，不缩减数据，输出强制隔离到`smoke/training/{run-name}`，只写`SMOKE_COMPLETE.json(status=smoke_complete)`。正式默认`training/{run-name}`，不能写入smoke或CPU验证目录。正式训练要求50次完整训练loader遍历加1次fit-Fisher遍历，共51次；smoke为1+1次。Task1 head、各遍历dataset IDs、全部样本索引SHA、首batch增强像素/target SHA、generator seed、worker数和样本/批次数与原正式`sequence_sgd_seed0`配对。这里只比较每个样本索引和每轮首个增强batch，不声称所有增强batch都有逐像素hash。

`training_batch_metrics.jsonl`逐batch保存`loss`、`ce_clean`、`ewc`、实际SGD梯度`grad_combined_norm`、学习率和更新后参数有限性。在step0、1、8、16…采样梯度分量：重微分原生EWC penalty得到`g_ewc`，由实际`p.grad - g_ewc`得到clean分量，保存`grad_clean_norm`、`grad_fisher_norm`、`fisher_to_clean_norm_ratio`和`cos_clean_fisher`。step1使一轮smoke能够观察离开anchor后的非零EWC梯度。探测不增加模型forward，不改`.grad`、参数或RNG；范数用float64求和，原SGD仍用原float32梯度更新。clean是减法推导，在EWC远大于CE时有浮点抵消限制。旧基线不存在的梯度分量不能补造。

真实非有限CE/EWC/total loss、梯度或参数写`DIVERGED.json`，退出码86，不写正式完成标记。包含`status=diverged`、`identity_sha256`、`seed`、`ewc_lambda`、`smoke_only`、`reason`、`error`、`time_unix`、已完成batch数和prepared_fisher身份。其他异常（OOM、缺文件、身份/配对错误）是技术失败，不能吞作数值发散。发散不降低lr/batch、不改变λ。`--resume-last`只恢复能通过完整末端验证的Task1；部分训练不得混用旧receipt静默重跑。

正式COMPLETE沿用旧schema；`validation`含CIL准确率、有效QV BA位移、losses、gradients、batch metrics SHA、输入配对证据及prepared_fisher来源。`run_manifest.prepared_fisher`和`validation.prepared_fisher`均绑定payload与PREPARED文件SHA、原父checkpoint SHA、源manifest SHA和raw trace。最终验证另要求原生Task1 mechanism明确lambda_flat0、目标λ和未归一化累积Fisher，局部Task0 train/test CE/accuracy有限，完整checkpoint/Fisher保存并成功逐张量reload。

CPU回归覆盖λ范围和身份、两false flags、真实父身份、缺少准备时仅可预览、准备receipt/hash篡改拒绝、raw tensor shape/非负/trace校验、真实51遍历、smoke隔离、第二batch梯度采样，以及原生EWC诊断保持参数/.grad/RNG和SGD更新逐位相同。GPU Fisher准备、smoke和正式运行由外层调度负责。
