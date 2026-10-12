# 第一轮 raw-EWC λ 细化：独立入口

本目录只执行`refinement_spec.json`列出的六个seed0浮点λ试验。每个点继续从原正式SGD Task0完整权重训练Task1 50轮，并直接复用初轮实际计算的同一份raw Fisher及PREPARED凭证。Task0不重训，Fisher不重新计算，两种trace归一化仍为false，SGD、lr .02、batch128、FP32/TF32关闭、任务head、RNG及输入配对全部沿用初轮。

`train.py`直接复制`../ewc_lambda_sweep_seed0_20260921/train.py`，唯一文本差异是`--prepared-fisher`默认值从初轮自己的HERE路径改为`FORMAL / 'ewc_lambda_sweep_seed0_20260921/prepared_fisher/fisher_state_raw.pt'`。新目录与初轮同级，因此FORMAL父目录和EXECUTION/source解析不变；HERE使输出自然隔离到新目录。回归测试验证完整文件恰好这一个替换，并将此替换消去后比较完整AST；训练、Fisher恢复、有限数值检查、梯度诊断、51遍历验证和完成schema均未改动。已完成初轮GPU验证仍适用于相同训练逻辑，本轮不新增smoke作业；每个正式作业仍由外层执行CUDA witness和资源准入。

初轮runner SHA256：`90bd977e0e0dbb2f2e7b7d5bcaed390b7f035405ac16354c5101e29fbd417f44`。
本轮runner SHA256：`a03892b30727a5f8b6ea3545de480302d57cb4b9f1c3cd14a664ddff5e71c6c5`。

外层规范：

```json
{"round": 1, "seed": 0, "trials": [{"trial_id": "refine1_01_seed0", "ewc_lambda": 13160.740129524924}]}
```

`trial_id`是唯一安全目录名，直接作为原生`--run-name`，不通过整数截断或float格式化生成目录。λ必须为有限JSON数值，范围[0,1e9]；重复ID、重复数值、路径穿越、错误seed/round、CLI ID与λ不匹配均拒绝。规范可附加metadata；若提供initial_sweep、prepared_fisher和input_fingerprints，wrapper会核验其路径及所有SHA。正式根目录仍为本目录`training/<trial_id>`。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python run_trial.py \
  --trial-id refine1_01_seed0 --ewc-lambda 13160.740129524924 \
  --gpu 0 --resume-last
```

以上以本目录为cwd，GPU由外层调度代入；本实现工作未执行此训练命令。`--spec`默认本目录`refinement_spec.json`。wrapper明确传入共享prepared路径和完整浮点λ，并在同一进程调用原生runner，保持外层锁的生命周期。运行前计算完整原生identity；终态严格比较manifest与请求配置、runner SHA、prepared身份。只有原生COMPLETE退出0或者真实DIVERGED退出86且身份一致才写`TRIAL_RESULT.json`。OOM等技术失败保持非零，不伪装成观测。

TRIAL_RESULT保留初轮的`status=observed`、`outcome=complete|diverged`、seed、ewc_lambda、native_exit_code、native_receipt/SHA、manifest/SHA、identity_sha256及formal_accuracy_available，新增round、trial_id、spec路径/SHA、wrapper_sha256、prepared_fisher和recovered_terminal_divergence。`--resume-last`对已完成点调用原生完整复验；对于已写DIVERGED的终态，只在精确身份校验后恢复其观测回执，不重新训练，也不改变λ或超参数。部分训练仍按原生规则拒绝混入旧receipt静默重跑。

CPU验证：保留初轮13项有针对性回归，新增4项覆盖完整文本/AST差异、六个真实浮点点的无损传参及独立身份、错误规范/输入指纹拒绝、技术失败和数值发散区分，共17项通过。wrapper最后增加输入指纹约束后，受影响的4项细化测试再次通过。真实`--validate-cpu`以λ13160.740129524924完成全网络、24个共享raw Fisher和新head恢复：actual_raw_trace=34.069888104965315，两normalize=false，head SHA=`05d16669f438d5ad33621d339fea510fef78c6c917131a70bc769873d592d7b2`。回执位于`cpu_validation/cpu_refinement_float_restore/CPU_VALIDATED.json`。

共享raw payload SHA=`cfdcd801f773d3d33cb48885c84ff31f032a60bf4a4cccbdead5008d67aacd95`，PREPARED SHA=`76a06376cb87e5a21bcb92c26c9a66890b5fb280d36fa372bc57dcc54f1e3cf1`。CPU验证不包含forward/backward；本工作未启动GPU或控制器。外层全GPU0–6空闲首次准入、900秒轮询和最多6并发策略保持不变。
