# Future probe 独立代码审查

结论：当前源码通过静态与独立 CPU审查，可进入future GPU smoke；完整native ViT/checkpoint集成仍需smoke通过，queue另审。

- 已修复原生 clean+(selected-clean) 后误要求 raw g1/g3 逐位一致的问题；现在分别按native decomposition/projector实际运算顺序、SGD预测更新验证，每步实际4closures+5forwards。
- 每checkpoint从native完整Task0权重独立加载，model hash/LoRA metadata/arm/seed/epoch/source/config绑定；原文件运行前后hash不变。
- 扩头保持旧head rows和backbone exact，新行同seed隔离RNG；未来K0在扩头后，local/allseen与扩头前状态分开。
- Task1训练仅local类别20..39，未来S保留rho=.05的原生g1；同seed/future step数据和LR一致。
- 100步按8batch/epoch推进50epoch cosine schedule，实际12.5epoch；不会将K节点当连续追加预算。
- 每步fixed diagnostic评测与K0使用同一batch packing，I/Iplus望远镜恒等检查成立。full/reporting另在固定K节点报告，不能把diagnostic Iplus称full-test累计破坏。
- 评测隔离RNG/buffers/modes/grads；CPU特意注入的随机/状态变化toy模型证明恢复。
- split核canonical sample IDs、labels、人数、互斥并集与singleton规则。实际Task0=307(102/205)；Task1=254(84/170)，class34单张只进入reporting。
- 完成reference summary绑定identity与trace SHA；source/head/data/LR/split在consumer侧匹配；smoke1显式与formal区分，接口兼容root manifest及S1.5/S2 endpoint probes。

CPU结果：data_rng/head_rng/observational/split/path/reference均PASS，native S和Gmatch各3步通过（matching容差1e-3），并拒绝损坏投影/错误引用等negative cases。

精确源码与测试SHA见 FUTURE_CODE_REVIEW.json；独立运行输出见 FUTURE_PROBE_CPU_INDEPENDENT_FINAL.json。
