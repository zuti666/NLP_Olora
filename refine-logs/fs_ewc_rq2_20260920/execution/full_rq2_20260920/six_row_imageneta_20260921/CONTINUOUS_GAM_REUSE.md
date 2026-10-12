# 持续GAM两任务结果复用核验

**可以直接复用正式 `sequence_gam_seed{0,1,2}` 的Task1 checkpoint和t01记录，作为“GAM Task0 → GAM Task1 + EWC（λ=2000）”这一行；无需重新训练。不能复用成无EWC的GAM→GAM，也不能取最终Task9指标代替两任务结果。**

所有数据来自本次正式训练协议。已逐seed重算父Task0 checkpoint、父Fisher、父metrics及Task1 checkpoint的SHA256，并与运行manifest、COMPLETE和trajectory记录交叉核对；检查全部通过。完整来源与具体字段见 [CONTINUOUS_GAM_REUSE.json](CONTINUOUS_GAM_REUSE.json)。本次仅只读CPU核验，未执行GPU评估。

## 实际配置与B3可比性

- 三个实际manifest均为 `optimizer_type=gam`、`lambda_flat=1`、`ewc_lambda=2000`；Task1机制记录也明确 `lambda_flat=1`、`lambda_ewc=2000`。50 epochs、lr=.02、cosine、batch128、momentum0、weight_decay0，FP32且TF32关闭。
- 每个seed与B3_D（GAM起点→SGD+EWC）恢复的是完全相同的正式GAM Task0 checkpoint、Fisher和anchor metrics；runner及冻结source hash也相同。
- 实际配置差异只有预期的optimizer_type、lambda_flat、总任务截止数及四项输出命名/路径。sequence截止10任务，B3_D截止2任务；`source/src/trainer.py:929`用于限制任务循环，具体任务scheduler按epochs=50设置，后续任务不会反向修改此前保存的Task1 checkpoint。t01矩阵与最终矩阵的前两任务部分一致。
- Task1 RNG比较显式忽略预期的optimizer_type标签差异后，新head、任务seed、训练loader配置以及51次shuffled loader遍历（50轮训练＋1次末端Fisher）的实际样本顺序和首batch像素/标签哈希全部相同。配对证据不声称逐像素验证每个batch。

## 可直接填入六行表的两任务结果

a11使用当前FP32的Task0评估；a21/a22使用训练CNN class-IL t01矩阵（40类联合推理，原记录两位小数），不能改用固定旧20类局部head。BWT=a21−a11，每个seed先求差再聚合；平均=(a21+a22)/2，为任务等权。该精度策略与现有B3一致。

| seed | a11 (%) | a21 CIL (%) | a22 CIL (%) | BWT (pp) | 两任务平均 (%) | 有效QV位移L2 | L2² |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 86.970684039088 | 83.71 | 81.10 | -3.260684039088 | 82.405 | 0.502006616052 | 0.252010642560 |
| 1 | 86.970684039088 | 83.06 | 81.50 | -3.910684039088 | 82.280 | 0.484501154233 | 0.234741368453 |
| 2 | 87.947882736156 | 83.39 | 79.92 | -4.557882736156 | 81.655 | 0.516177166950 | 0.266438867681 |
| mean±sampleSD | 87.296417±0.564186 | 83.386667±0.325013 | 80.840000±0.821462 | -3.909750±0.648600 | 82.113333±0.401819 | 0.500895±0.015867 | 0.251064±0.015870 |

三seed的Task1旧20类局部head accuracy均为88.273615635%，这不是表中的a21；a21必须使用83.71/83.06/83.39。位移是Task0→Task1有效24个Q/V权重矩阵差的Frobenius范数，排除分类头；训练metrics里的 `delta_norm` 实际为其平方，不能与L2混写。

## Fisher列：用户要求的起点B1可复用；三种Fisher口径必须分开

| seed | 起点 batch-gradient proxy B1 | 起点raw train256 | Task1 raw train256 | 起点raw fulltrain1259 | Task1 raw fulltrain1259 |
|---:|---:|---:|---|---:|---:|
| 0 | 24.823071680963 | 652.771968647185 | 未测 | 617.087324472501 | 771.305851590643 |
| 1 | 18.866997465491 | 657.794630660477 | 未测 | 597.125271431022 | 756.212569662147 |
| 2 | 22.108400069177 | 592.963797098538 | 未测 | 631.796747763872 | 756.612497206621 |

用户这张表的起点mathcal B1若按已选口径使用 `in_task_fisher_final_raw_trace`，三seed值24.8230716810 / 18.8669974655 / 22.1084000692均存在于同一正式GAM anchor的机制记录，可与其它GAM起点行共享。它是batch-gradient delta-W Fisher proxy，不是逐样本empirical Fisher。

起点raw train256同样现成可用。B2的Task1 endpoint只测了全1259旧train图的逐样本raw Fisher（以及全307 test），没有同一固定train256子集的endpoint诊断；不能用fulltrain均值、batch proxy或B3_D不同Task1 checkpoint的结果填充该列。这一可选诊断缺项不阻塞用户当前只要求起点B1的六行表。

## 复用边界

可复用：a11、a21、a22、BWT、两任务平均、Task1有效QV位移及其平方、起点batch-proxy B1、起点raw train256、固定旧head train/test D0，以及另列明确标注fulltrain1259的Task1 raw Fisher。

不可替换：a21不能用88.27%的local accuracy；两任务平均不能用Task9全十任务final指标；L2不能用日志delta_norm直接代替；fulltrain1259不能冒充train256；λ2000结果不能当成λ0结果。此复用为同源成对协议下的持续GAM实验行，仍不代表已经隔离起点几何、保护矩阵和后续优化器各自的因果作用。
