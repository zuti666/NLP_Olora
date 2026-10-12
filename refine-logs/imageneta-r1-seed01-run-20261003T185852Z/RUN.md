# ImageNet-A r=1 seed0/1 重复实验

本轮30种纯GAM条件，每种重新训练seed0、seed1，共60次完整10任务实验。统一rank10、r=1、rho=.05、50epochs/task、batch128。native28项和four-probe32项分别保留历史代码路径；EWC/Fisher、权重融合、replay关闭。原版GAM自身正交项和B残差投影仅在对应条件保留。seed2只读复用，不重训或算成新seed。

先运行60项短测（native每项2任务×1epoch=18steps，four-probe每项2任务×1epoch×1batch=2steps；scheduler Tmax50）。全部结果和同seed配对验收通过后自动开始60项正式训练。Gmatch进一步依赖同seed的S完成，禁止跨seed引用。首批正式顺序N01s0、N01s1、N03s0，随后优先N03s1。

GPU只用物理4、5、6，最多三路并行。索引+UUID锁，显存低于500MiB且利用率低于5%才接收；双重读取确认，不抢占。其他GPU无论是否空闲都不使用。

独立文档验证者逐字执行以下唯一命令（真实host require_escalated；只短CUDA内核测试，不训练）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/documented_check.py --gpu 4
```

预期dry-run结果120jobs、60formal、60smokes，allowed_gpus [4,5,6]、max_parallel3；之后G012_ENV_WITNESS打印有限forward/backward。验收保存gates/DOCUMENTED_INVOCATION_WITNESS.json。

正式队列命令（由主agent后台启动，不在已有controller运行时重复执行）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/run_queue.py
```

实际启动采用start_new_session，标准输出写controller.log。查看runtime/queue_state.json、runtime/jobs/<id>/console.log和各自exit/completed记录；queue包含短测及正式任务共120项，结果表只纳入完整正式训练。runtime/PAUSE仅暂停后续放行，不中断已运行任务。任何失败停止新增启动，保留全部日志；原计划和历史代码均保留。

summary.csv列出60条新实验当前状态；THREE_SEED_SUMMARY.csv按同协议同条件合并seed0/1和指定seed2，只有完整三seed才能报告三seed结论。单seed不报告标准差0。原始科学计划和历史配置索引见plan_snapshot/；可复现实际命令、参数、依赖和文件校验值在suite.json与RUN_PINS.json。
