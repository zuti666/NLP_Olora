# 恢复 ImageNet-A r1 seed0/1 原队列

用户授权在物理GPU4/5/6继续原计划。运行目录为 /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z。本次只恢复已有60项正式训练：保留6项已完成和60项已通过短测；3项中断（N04s0/N04s1/N05s0）从Task0重新训练，51项未启动按原顺序继续。原代码、配置、suite和RUN_PINS均保持原字节及路径，Gmatch继续引用已完成的同seed N03/S。

中断输出和runtime完整移动到本目录 interrupted/，各文件hash和原路径记录于ARCHIVE_INDEX.json。原queue_state、controller和结果表备份到before_recovery/。先持有原controller.lock，检查旧controller/worker已经消失及所有证据，再仅将3个中断状态重置为pending；不覆盖6个完成结果，不重跑60个短测。所有正式任务保持10×50epochs；重跑不混入之前部分轨迹。

环境热复用Python `/home-local/liying/.conda/envs/Pilot_new_local/bin/python`，canonical environment spec SHA4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8；无包安装/升级。host cudahpc40.cvc.uab.es，真实PID namespace pid:[4026531836]，provider ledger为 /data/140-1/users/liying/O-LoRA/.aris/compute/local.md。资源只限GPU4/5/6,max3；共享索引/UUID锁与空闲检查延用原队列。

独立agent-follows-doc核验者只逐字运行下面这条命令（真实host exec_command require_escalated）。它不移动输出、不重置队列、不启动训练；只核验现有记录并在空闲GPU4共享锁下执行固定seed2环境内核测试，0次SGD。其PREFLIGHT.json写入本恢复目录。预期passed=true, formal_completed=6,smoke_completed=60,remaining=54,allowed_gpus=[4,5,6]，cuda_witness有限。

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/recoveries/resume_20261005T114228Z/recover.py --check-only --gpu-witness 4
```

验证者将独立报告写入本目录DOC_CHECK.json，包含passed=true/false、精确命令、退出码、差异。不要执行apply或启动队列。主agent待独立核验通过后执行同脚本--apply，再以start_new_session后台调用原run_queue.py（不复制/改写原队列代码），新的controller日志写本恢复目录controller.log；启动记录LAUNCH.json。恢复后由原控制器重新验证66项完成记录和短测门禁，每个新训练worker又执行一次CUDA witness。
