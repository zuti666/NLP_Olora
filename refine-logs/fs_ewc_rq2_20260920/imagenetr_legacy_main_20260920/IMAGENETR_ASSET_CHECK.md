# 真实历史 ImageNet-R 资产的有限范围只读核查

2026-09-20。核查了已索引的旧主表、best projected、phase5x5、Task0 FQA 和 true-GAM checkpoint 候选；没有进行全盘穷尽搜索，没有启动 GPU。

精确主表 `verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7` 的实际 run summary 均为两个任务；`sessions=10` 配合 `stop_task=2`，不构成十任务结果。该组 `save_ckp=False`，指定日志根未找到 `.pt/.pth/.ckpt`。主表核心均值19.885229→10.945784、87.946667→91.823333已核实。

真实 R 的其他可复用资产：

- `/data/115-2/users/liying/low-rank-cl-new/result_summary/best_gam_projected_4datasets/best_rho0p05_normrho0p5_lambda1e7/`：ImageNet-R seed2、55条逐任务 accuracy、10条机制记录；full projected 设置，未保存 checkpoint，没有旧任务 loss。`delta_norm` 为每次新 adapter 平方能量，缺少交叉项，不能还原累计 r(T)。
- `/data/115-2/users/liying/low-rank-cl-new/configs_sweeps/verify_imagenet_r_phase5x5/phase5x5_s012/`：五组×三seed、十任务；normrho=.2、前五任务 shaping。相应 `result_summary/summary_all_selectedbest/eval_long.csv` 中825条相关记录；无机制记录和权重。
- `/data/115-2/users/liying/low-rank-cl-new/logs/imagenet_r_task1_fqa_adam_gam/task1_fqa_s0_fulltrain_k100_r0to1_fine_firstorder_r1/`：`adam/ewclora/imagenet-r/t10/seed0/` 与 `gam_rho_0p05/ewclora_gam_factored/imagenet-r/t10/seed0/` 各有 `task_0.pth`、`task_0_endpoint.pth`。真实 R、seed0、Adam、GAM `.05/.2`，与主表 `.05/.5` 不同。现有27半径×100方向数据在对应 `result_summary`，不能把 log/trainer 两份重复记录当成独立方向。

## 有完整十任务 A/B/fc、但控制不匹配的一对

SGD checkpoint：

```text
/data/140-1/users/liying/CopyFlatCV/logs_our_joost/imagenetr_lora_factorfs_dampnorm/logs_inc_lora/ewclora_youyue_fitarchitechture/lorar_sgdplain_old_1x_noewc_initlr002_lr0015_e50_lam0/imagenetr/0/lorar_sgdplain_oldlearner_1x_noewc_imagenetr_t10c20_r10_e50_initlr002_lr0015_lam0_s0/exp_run/checkpoints
```

GAM checkpoint：

```text
/data/140-1/users/liying/CopyFlatCV/outputs_logs/config_Expand_lr_tuned/logs_inc_lora/ewclora_normfisher_gam/gam_sgd_initlr002_lr0015_e50_rho005_f100/imagenetr/0/as1normfisher_lam2000_imagenetr_t10c20_r10_sgd_initlr002_lr0015_e50_rho005_f100_s0/exp_run/checkpoints
```

两条都有 `lora_w_a_{0…9}.pt`、`lora_w_b_{0…9}.pt`、`fc_state_{0…9}.pt`、`lora_meta_{0…9}.json`。

共同设置：真实 ImageNet-R，seed0，10×20、固定class order；ViT-B/16 augreg2_in21k_ft_in1k，rank10，12层Q/V；LR .02→.015、50epoch/task、cosine、momentum0、weight_decay0。

关键区别：SGD分支 lambda=0；GAM分支 lambda=2000且训练 Fisher 按 trace 归一化，rho=.05、normrho=.1、gamma=.1。两次运行的历史时间也不同。缺少与 GAM 同 lambda/Fisher 规则的 native EWC 基线，故不能解释为受控 GAM 干预。

metadata 的 t0/t9 均为同一持续 adapter（task_id=0），不是旧主表的 fresh K/V 累积。因此每阶段有效位移应恢复缩放 s 后计算 `s*(B_T*A_T−B_0*A_0)`，不能直接相加全部 sidecar。冻结 base 在此位移差中相消；各自 `fc_state_0.pt` 可用于固定旧 head，但本轮未声明后续旧20行逐位相同。

可读数据为 `/data/115-2/users/liying/datasets/imagenet-r/{train,test}`；原配置的 `/data/140-0/datasets/imagenet-r` 现不可读。需重新记录类别/样本映射并核对原 accuracy。

本地冻结主干候选为：

```text
/data/115-2/users/liying/cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/snapshots/063c6c38a5d8510b2e57df480445e94b231dad2c/model.safetensors
```

两条日志只记录模型别名/resolve-main，没有历史 revision/hash，也没有完整冻结 backbone/Fisher checkpoint。原 accuracy、参数键、LoRA缩放、数据映射通过恢复校验后，原则上可以不重训补测 loss、raw Fisher、directional path；缓存文件存在本身不是历史恢复成功的证据。

`ratio_pair_asset_check.json` 证明的是另一 Ratio-FS 两任务资产；`evaluation_protocol.json` 自己仍标记 strict GAM pair 未核实，不能把它们当成真 GAM 十任务匹配对照的完成凭据。
