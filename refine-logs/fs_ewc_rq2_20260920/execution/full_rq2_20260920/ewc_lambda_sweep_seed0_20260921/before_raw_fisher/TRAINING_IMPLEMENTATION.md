# Seed0 SGD→SGD+EWC λ 扫描入口

`train.py`由已验证的`../seven_row_extension_20260921/train.py`派生，继续调用原冻结`execution/source`的原生SGD、EWC与Fisher实现。只恢复正式`../formal_training/anchor_sgd_seed0`的完整Task0网络和原始24个归一化Fisher张量；不重训Task0、不重算raw Fisher、不缩放或替换父Fisher。`ewc_normalize_fisher=true`、`ewc_normalize_accumulated_fisher=true`保持原状。旧runner/source和历史结果均不修改。

CLI默认并约束`arm=ewc_lambda_sweep`、`mode=continuation`、`anchor_method=sgd`、`future_optimizer=sgd`、`lambda_flat=0`。必须提供`--seed 0`（或`--seed0`）、`--ewc-lambda`、`--run-name`；GPU训练还必须提供`--gpu`。λ允许任意有限实数`0 ≤ λ ≤ 1e9`，入口不硬编码最终网格。默认父目录和metrics均为正式SGD seed0。显式传入不匹配的父目录会被拒绝。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python train.py \
  --seed 0 --ewc-lambda 100000 --run-name sgd_ewc_lam1e5_seed0 --gpu 0
```

正式固定只训练代码Task1，50轮，batch128、lr .02、momentum0、weight_decay0、cosine、workers4、全FP32及两处TF32关闭。EWC eta=0、gamma=.9，无GAM shaping、投影、裁剪、双对偶调λ或权重合并。`--smoke`仅将Task1预算变为1轮，强制写入`smoke/training/{run-name}`，只产生`SMOKE_COMPLETE.json(status=smoke_complete)`。正式输出默认`training/{run-name}`；正式模式拒绝smoke和CPU验证目录。不能传epochs缩短正式训练。

父manifest/COMPLETE、源文件、父模型/Fisher/metrics、父runtime/RNG receipts全部执行原身份和SHA验证。真实加载后逐张量验证完整网络和Fisher，并检查实际optimizer=SGD、两处λ一致、lambda_flat=0及两normalize flags=true。CPU验证不做forward/backward，只构建真实模型、完整恢复父状态、初始化Task1 head。

Task1每次运行在同一task边界重置共享RNG；新head、各遍历dataset IDs、全部样本索引SHA、首batch增强像素/target SHA、generator seed、worker数和样本/批次数均与`../formal_training/sequence_sgd_seed0`逐遍历比较。正式必须有50次训练遍历和1次fit-Fisher遍历，恰好51次完整遍历；smoke为1+1次，第二次对比正式iteration1。这里只核对每个样本索引和每轮首个增强batch，不声称全batch逐像素hash。

`training_batch_metrics.jsonl`逐batch记录`loss`、`ce_clean`、`ewc`、实际`grad_combined_norm`、实际学习率和更新后参数有限性。每8步（0、8、16…）重微分原生`_past_fisher_penalty`，观测`g_ewc`，由实际`p.grad - g_ewc`计算clean分量，再记录`grad_clean_norm`、`grad_fisher_norm`、`fisher_to_clean_norm_ratio`和`cos_clean_fisher`。这个探测只构造LoRA惩罚图，不增加模型forward，不写入`.grad`，不修改参数或RNG；所有范数用float64求和，原SGD更新继续使用原float32梯度。clean分量通过减法推导，在EWC远大于CE时存在浮点抵消限制。旧SGD基线没有这些分量统计，不能为旧基线补造数值，也不要求GAM梯度字段。

每batch在更新前检查原生CE/EWC损失及真实梯度有限性，更新后检查可训练参数，结束后继续核验loss、局部Task0 train/test CE与accuracy、CIL矩阵、有效QV BA位移、完整checkpoint/Fisher和reload receipts。损失/参数/梯度出现真实非有限值时写`DIVERGED.json`并退出86；包含`status=diverged`、`identity_sha256`、`seed`、`ewc_lambda`、`smoke_only`、`reason`、`error`、`time_unix`、已完成batch数，明确未改变超参数。发散不写COMPLETE，不能通过resume变为完成。其他异常（包括OOM、缺文件、身份/配对错误）写FAILED并保持非零退出，不能当作数值发散观测吞掉。

COMPLETE沿用原schema，`validation`另有`fisher_policy`、`losses`、`gradients`和`training_batch_metrics_sha256`；`gradients`包含combined mean/max、采样组件mean、样本数和来源说明。λ和精确父身份保留在manifest/requested_config中。`--resume-last`仅恢复已完成Task1边界且能通过全部验证的末端；部分Task1不得混用旧receipt静默重跑。

已完成CPU验证：`test_train.py -v`共8项通过，覆盖λ范围/身份、真实父身份、smoke隔离、真实51遍历、原生EWC探测保持参数/.grad/RNG及SGD更新逐位相同，以及非有限实际梯度拒绝。`--print-config`以λ1e9通过；`--validate-cpu`以λ1e8完整恢复24个Fisher张量和全网络通过，实际normalize两flags=true，Task1 head SHA为`05d16669f438d5ad33621d339fea510fef78c6c917131a70bc769873d592d7b2`。GPU smoke/正式运行由外层调度负责，CPU结果不代替GPU验证。
