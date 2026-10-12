# 两个缺失分支：训练入口与验证

本目录独立扩展已完成的正式RQ2，不修改旧`full_rq2_20260920/train.py`、冻结`execution/source`或任何历史结果。`train.py`由旧runner派生，继续调用完全相同的frozen learner；改变的是实验身份、允许的未来优化器与验证约束，没有改GAM、EWC、Fisher或数据算法。

| Arm | Task0父方法 | Task1未来优化器 | Task1 λEWC | 正式seed |
| --- | --- | --- | ---: | --- |
| `shaping_only` | GAM | GAM | 0 | 0、1、2 |
| `task2_shaping` | SGD | GAM | 2000 | 0、1、2 |

CLI显式分离`--anchor-method`和`--future-optimizer`，不接受含混的`--method`。两种arm与方法/λ的组合必须完全匹配。正式固定：Task0恢复后只训练代码Task1，50轮，batch128、lr .02、workers4、cosine scheduler、全FP32、两处TF32关闭；不能传epochs来缩减预算。旧Task0没有EWC历史，因此共享原六个已完成anchor即可。

父目录必须是`../formal_training/anchor_{sgd|gam}_seed{s}/checkpoints`，而非旧pilot或其他seed。启动前校验父manifest/COMPLETE一致、完成Task0的50轮验证、dataset/worker/precision/source/runner等身份、完整Task0与Fisher文件SHA、唯一Task0-only metrics SHA，以及父RNG/runtime receipt SHA。实际加载仍执行严格完整网络state恢复与逐张量相等检查，并恢复24个Fisher张量；不会只载LoRA，也不会重新下载预训练模型。

示例（GPU调度由外层queue代入）：

```bash
python train.py \
  --arm task2_shaping --anchor-method sgd --future-optimizer gam \
  --seed 0 --ewc-lambda 2000 --run-name task2_shaping_seed0 \
  --anchor-dir ../formal_training/anchor_sgd_seed0/checkpoints \
  --anchor-metrics ../formal_training/anchor_sgd_seed0/trainer_outputs \
  --output-root training --workers 4 --gpu 0 --resume-last
```

上述相对路径以本目录为cwd；queue生成绝对路径。`shaping_only`相应换成GAM父、λ0；seed1/2同样分别恢复同seed父。

`--print-config`同时执行只读父/source验证，返回config与identity，不训练。`--validate-cpu`在`cpu_validation/{run-name}`构建真实模型、完整恢复父网络/Fisher并初始化新head，不运行forward/backward或GPU。

`--smoke`是显式独立集成测试：只训练Task1 **1轮**，不截取数据，随后使用原生CIL评估、Task0 loss评估、fit Fisher及checkpoint保存/完整重载。它强制输出到`smoke/training/{run-name}`，仅写`SMOKE_COMPLETE.json(status=smoke_complete)`；不会写正式COMPLETE，不能混入正式表格。正式模式不加此参数始终50轮。两个smoke由外层`validate_smokes.py`验证后，queue才释放六个正式作业。

保存Task1后，新runner把实际`task_01.pt`严格载回模型，逐参数核对一致，保存完整重载receipt；旧head保持不变的断言、有效QV BA位移、Task0 train/test loss/CIL评估沿用旧runner。正式完成判定还要求：

- 恰好训练Task1，存在最终`50/50`日志、task_complete与精确父恢复receipt；冻结file logger仅记录末轮，不能误要求其包含每一轮tqdm输出。
- 50次完整训练loader遍历加1次完整fit-Fisher遍历，共51次，iteration顺序0…50。smoke相应为1+1=2次。缺失或不完整Fisher遍历也拒绝完成。
- 新head SHA及各遍历的dataset IDs、全部样本索引SHA、首batch像素/标签SHA、generator seed、worker数、样本/批次数均与同seed正式`sequence_gam`输入流一致。smoke的第二遍历对照正式iteration1，不是正式iteration50；这核对相同loader流位置，不要求smoke模型数值等于正式第50轮。
- 2×2有限CIL矩阵与`t01`一致，mechanism step1明确λ与shaping设置，Task1/Fisher/RNG checkpoint存在且有SHA，Fisher非空有限，Task0有效权重位移有限。

输入验证覆盖每个样本索引与每轮首个增强batch，不声称所有增强batch都有逐像素hash。GAM额外梯度探测的算法计算仍与SGD不同。`--resume-last`只恢复已保存且可完整验证的Task1末端；未形成完整checkpoint的部分Task1不会被静默重跑混入旧receipt，需保留失败目录并以新目录重新安排。

已完成CPU验证：

- `test_train_config.py -v`：7/7通过，覆盖六份正式配置、六个真实父身份、错误arm/父seed拒绝、smoke输出隔离、真实51遍历回归、真实final-only文件日志回归。
- `cpu_validation/shaping_only_seed{0,1,2}/CPU_VALIDATED.json`：3/3完整网络和24个Fisher恢复通过；实际future optimizer=GAM，λ0。
- `cpu_validation/task2_shaping_seed{0,1,2}/CPU_VALIDATED.json`：3/3完整网络和24个Fisher恢复通过；实际future optimizer=GAM，λ2000。
- 共6/6真实CPU恢复检查通过，两臂同seed的新Task1 head SHA一致：seed0=`05d16669f438d5ad33621d339fea510fef78c6c917131a70bc769873d592d7b2`，seed1=`17c7c4bf794328e23a3079d0268ce1a51d79e3d43d8f01cba232f481d5361c2f`，seed2=`f3691cae0907d4f54b6f06569d7ce45462168468d8bd28aa94c6fc1930e19371`。CPU验证没有证明GPU forward/backward、CIL数值或完整训练成功；这些由排队中的实际smoke和正式完成验证负责。

当前训练入口SHA256：`4e87c9312de8f40bfa5f2f6e6df15853e6ba65c7386ea8dcb4426effe8c1b492`。manifest与identity保存`arm/anchor_method/future_optimizer/task0_shaping/task1_shaping/smoke_only/parent_validation`；COMPLETE保留旧schema并增加准确率、位移与配对验证摘要。训练启动后入口须冻结；更改runner会导致既有identity/source校验拒绝复用。
