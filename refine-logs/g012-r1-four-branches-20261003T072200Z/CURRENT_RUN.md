# 当前生效的运行说明

已启动ImageNet-A r=1的P/D/C/R探针与符号实验。新完整十任务训练12项；半径无关基准真实历史复用4项，不重新写旧completed。16项r1两任务短测全部通过，共32真实更新，实际外点距离r=1、无Fisher/融合/g3/正交修正、有效步长匹配最大误差约3.994e-8。

当前生效资源入口为 resource_extension_20261003T074000Z/resource_queue.py controller，资源池[1,4,5,6,3,2]、max_parallel3。GPU1原Mean2已完成释放，因此先在GPU1启动；GPU2原Gmatch及GPU3–6其他训练继续，只有实际空闲并通过共享锁/UUID检查后才能接收新任务。GPU0不在本池。原父RUN.md的GPU2/3声明由最新资源RUN.md覆盖；此前两个busy文档尝试均未启动训练、原失败记录保留。

P的O=g2正式训练已实际启动，首轮三epoch有限。其余11完整条件由同一后台控制器接续，全部正式新成绩仍待各自十任务/2550步验收。动态进度在runtime/queue_state.json；主表RESULTS.md、summary.csv、results.json自动更新。核心设计见EXPERIMENT_PLAN.md，实际资源/复现入口见resource_extension_20261003T074000Z/RUN.md。真实配置/代码/CPU与历史审计56文件哈希未变，启动和资源身份见LAUNCH.json、gates/FORMAL_START_CHECK.json。
