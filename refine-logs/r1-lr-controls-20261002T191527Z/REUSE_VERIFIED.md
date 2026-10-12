# ImageNet-A 原生10任务学习率/幅值对照：历史结果复用核验

五条历史seed2完整训练的配置、started/completed凭据、原始metrics、suite/source_manifest、console和运行脚本SHA256均重新核验。190个冻结源文件live哈希全部通过。全部原始CNN矩阵包含10任务与55个有限观察值；500个task/epoch身份和10条Fisher跳过记录完整。仅写新目录的本报告和JSON，没有训练或修改历史文件。

| 组 | 使用范围 | r | LR首任务/后续 | AAA task macro | FAA task macro | LA | BWT | forgetting |
|---|---|---:|---|---:|---:|---:|---:|---:|
| S | 历史背景；需重跑S作步幅参考 | 1 | 0.04 / 0.02 | 62.165531 | 45.132000 | 53.436000 | -9.226667 | 9.482222 |
| Sum | 复用主对照 | 1 | 0.04 / 0.02 | 67.860863 | 54.726000 | 61.583000 | -7.618889 | 7.618889 |
| Graw | 复用主对照 | 1 | 0.04 / 0.02 | 65.993172 | 50.679000 | 56.728000 | -6.721111 | 7.011111 |
| G025 | 复用主对照 | 0.25 | 0.04 / 0.02 | 66.229609 | 50.799000 | 57.317000 | -7.242222 | 7.365556 |
| Mean | 已有次级幅值对照 | 1 | 0.04 / 0.02 | 65.950948 | 49.927000 | 57.172000 | -8.050000 | 8.305556 |

S=g1；Sum=g1+g3；Graw=g3且r=1；G025=g3且r=.25；Mean=(g1+g3)/2且使用原名义LR。除β1/β2/β3/γ、prefix/outputs_root外，共享配置完整一致；G025仅额外改变该组预定变量gam_grad_norm_rho=.25。共同为ImageNet-A、seed2、rank10、rho=.05、50epochs/任务、batch128、SGD momentum/decay0、cosine；机制评估/统计开启、train/eval workers8，原生全局RNG/DataLoader，不按task/epoch重播种，Fisher/EWC/实际融合关闭。

Sum、Graw、G025可复用为本轮主对照；Mean为已有次级幅值控制。旧S仅作历史背景：五条均没有逐步batch/RNG/实际BA+FC步幅trace，无法提供独立S轨迹全部2550步的联合LoRA BA+完整seen FC位移范数。Gmatch禁止使用旧S作为步幅参考，必须先运行新S并记录全程每步范数。不能用任务终点checkpoint或逐任务平均梯度范数补造缺失步幅；checkpoint本次仅验证十任务文件集存在，旧凭据不绑定其哈希。

2550是相同source/data导出的完整训练预算；这里独立验证的是10任务、500个epoch身份和完整矩阵，未声称历史每步更新数与数学身份有trace证明。不存在严格跨GPU训练轨迹等价声明。

在CopyFlatCV的1158个config/suite字典与Flatness_CV的15个字典中，未找到全10任务持续S×1.5LR、S×2LR或Mean×2LR纯GAM完成实验。现有scaledcontrols是formation Task0，max_train_tasks=1，seeds0/1/2，不能代替本轮完整10任务控制。未使用r=.25隔离随机流component结果。

所有原始路径、SHA256、指标重算定义和证据限制见 [REUSE_VERIFIED.json](REUSE_VERIFIED.json)。
