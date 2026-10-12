# ImageNet-A七行表：已授权的补训队列

用户要求将已有GAM→SGD（C+F→C）放入倒数第三行，并补齐Shaping only、Task-2 shaping两行，在空闲GPU自动开始。七行表预期21个seed单元，复用15个已完成单元，新增6个正式续训。

环境沿用已验证的Pilot_new@e94a148f，不安装或更换依赖。解释器：`/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python`；Python3.11.15、torch2.4.1+cu121、torchvision0.19.1+cu121、timm1.0.9。基础spec为上两级`environment_spec.json`，canonical SHA前8位e94a148f，与`.aris/compute/local.md`既有条目相同。运行时FP32、matmul/cuDNN TF32均关闭；每个实际GPU作业先重复seeded CUDA forward/backward witness。

**执行范围**

- Shaping only：正式GAM Task0父起点→GAM Task1，λ=0，seeds0/1/2。
- Task-2 shaping：正式SGD Task0父起点→GAM Task1，λ=2000，seeds0/1/2。
- 每段Task1为50轮、batch128、lr .02、workers4，保持原共享RNG协议和数据/类别顺序。无需重训Task0或十任务序列。
- 同一持续LoRA分支，Q/V有效权重，固定端点参考，无trajectory averaging或任务内权重融合。保留所有旧冻结文件。

**启动前检查与自动依赖**

1. CPU配置/完整父状态恢复、数据与评估代码审查、队列dry-run先完成。
2. 两个seed0短测各1轮，使用真实Task1完整数据、评估、Fisher和checkpoint流程。输出严格隔离在`smoke/training/`，只生成`SMOKE_COMPLETE.json`，不作为正式结果。
3. `validate_smokes.py`确认真实GAM梯度、λ0无EWC梯度、λ2000有EWC梯度、CIL测试矩阵、旧任务测试损伤、Fisher/完整checkpoint/RNG收据均有效后，才放行6段正式训练。
4. 正式6段完成且收据有效后，CPU `report.py`校验父状态/输入配对，复算有效Q/V位移，自动生成完整七行Markdown/LaTeX/CSV于`final_report/`。

`endpoint_eval=false`关闭的是额外endpoint诊断模块；正式训练器仍调用`eval_task()`并写CIL矩阵。`mechanism_eval=true`、`mechanism_grad_stats=true`，任务末Fisher、checkpoint和旧任务loss/accuracy/位移记录均开启。短测门槛会验证真实输出。

**只读/不启动GPU的文档核验命令**

在项目根目录执行以下原样命令。它只验证队列结构和输入文件指纹，不启动CPU训练或GPU作业，也不创建campaign状态。

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/seven_row_extension_20260921/plan.json
```

预期：dry_run=true，stages=10，jobs=10，max_parallel=6；6个正式训练stage依赖smoke_validation，最终report依赖全部6个正式训练。

**运行入口（由本次助手启动一次）**

```bash
/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -u /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/campaign_runner.py --plan /data/140-1/users/liying/O-LoRA/refine-logs/fs_ewc_rq2_20260920/execution/full_rq2_20260920/seven_row_extension_20260921/plan.json --execute
```

调度器在宿主机后台运行，日志保存在`campaign/controller.log`，实际PID/启动时刻记录在`LAUNCH.json`。不要重复启动；恢复前检查宿主机进程身份和pipeline锁，再使用`--resume`，且冻结代码/配置不得变化。

GPU池0–6，最多6个并行作业。每张卡必须显存<500MiB、利用率<5%、无计算进程且取得共享UUID锁才可启动；每15秒重查。GPU繁忙只等待，不停止其他任务。任何短测/训练失败或收据无效都会阻止后续作业启动，已运行作业结束后队列退出失败；不自动修改batch或科学参数。

实际状态以[campaign/STATUS.md](campaign/STATUS.md)和[campaign/pipeline_state.json](campaign/pipeline_state.json)为准。排队、CPU检查通过、短测通过和正式完成是四种不同状态。
