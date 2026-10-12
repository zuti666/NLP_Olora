# GAM-EWC batch64 新运行结果

状态：completed；正式验证12/12，running0、failed0、pending0。结束于2026-09-19T18:13:54.833403+02:00（Europe/Madrid），总耗时2.9403小时，约2小时56分25秒。原supervisor102059及训练worker已退出；检查时7卡均0%利用率、各2MiB。

配置：Llama-3.2-1B；BF16；DDP4×BS1×GA16=64；seed/data_seed1993；EWC lambda1/gamma1；GAM rho0.02、normrho0.2、gradgamma0.03；实际TorchAdamW eps1e-8/weight_decay0.01；1epoch、首任务LR1e-3/后续1e-4。

|运行|Order1|Order2|Order3|Avg|
|---|---:|---:|---:|---:|
|新batch64，4×1×16，seed1993|69.8454|68.1250|70.6184|69.5296|
|旧batch40，5×1×8，seed1993|70.8355|70.0592|69.7007|70.1985|

新Avg相对旧结果变化为-0.668870个百分点。O1/O2下降，O3上升。batch、卡数、local-GAM窗口变化；不要归因于单一因素。新run保留seed1993，因此也不能当作与seed42纯EWC严格配对的消融。

48份正式rank审计均有限且协议正确；四closure/update、每closure16microbatch、最终同步梯度首末步hash一致，GAM扰动及后续EWC投影有效，Fisher全训练数据/32块，历史adapter不变。最大跨rank参数差1.452863216e-07，通过预先规定的allclose阈值。

结果始终独立保存，未改写旧运行：
- [完整结果JSON](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/final_results_gam_ewc_bf16_b64_s1993_ddp4_bs1_ga16_lam1.json)
- [结果CSV](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/results_by_order_gam_ewc_bf16_b64_s1993_ddp4_bs1_ga16_lam1.csv)
- [队列状态](/data/140-1/users/liying/O-LoRA/experiment_queue/gam_ewc_bf16_b64_s1993_ddp4_20260919T131729Z/state.json)
- [新旧结果对比](RESULT_COMPARISON_B64_VS_B40.csv)
- [正式运行记录检查](FINAL_RUNTIME_CHECK.json)

独立使用冻结normalize_answer重算91200条最终预测，全部整数计数/FAA一致；3份预测和3份指标的SHA256全部匹配。[重算记录](FINAL_RECOMPUTE_CHECK.json)。Order2精确FAA为68.1250%；原队列LaTeX使用Python保留2位的half-even舍入显示68.12，本报告保留4位避免舍入歧义。
