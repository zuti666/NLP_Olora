# Queue 与完成判定独立审查

结论：最终5个脚本的静态审查与独立CPU严格测试PASS，无剩余阻断；可冻结后部署 `--smoke-only`。这不是GPU或正式实验已通过的声明。

独立证据：

- 120个唯一job组成无环DAG：12smoke → 24formation → 84未来轨迹(78 futureS+6switch)。全部正式形成训练依赖全部smokes；全部formal future依赖完整24formation cohort。Phase2/3自动启动关闭。
- 8份完整正例均accepted，包括500-step formation与100-step future；14个formation负例、20个future负例全部按预期原因拒绝。缺字段KeyError、无关提前失败或错误接受都会令测试失败，不再只是记录字符串。
- receipt六类字段(config/summary/trace/identity/evaluations/checkpoints)逐项篡改均拒绝；restart/dependencies/collector均校验。collector不会把只有raw completed summary的并行任务提前计为已完成。
- future实际eval内容finite、K节点完整、final_summary和最终raw一致、两任务/三subset/两labelspace的samplecounts逐项核。Task0=307/102/205；Task1=254/84/170。
- controller单实例锁、UUID idle-GPU选择、lease FD继承、失败后停止新启动逻辑正确。已执行CPU lease测试证明worker退出后训练child仍持锁，child退出才释放。
- 不覆盖已有输出、不重跑已验completed、不自动启动Task2或完整后续Phase2/3。Future Task1 100-step本就是明确授权的susceptibility probe。

已修复并保留失败证据：最初fixture缺字段导致正例和全部negative都KeyError；future逐步ownspec/LR/评测完整性缺口；collector只看summary；receipt覆盖不全。旧QUEUE_VALIDATION_INITIAL_INDEPENDENT.json仅用于记录该已修复缺陷，其lease子测试有效。

所有CPU fixture明确标识为合成JSON/checkpoint bytes，没有模型训练或实验结果。源SHA及最终测试SHA见QUEUE_CODE_REVIEW.json。

正式队列释放前，还需真实GPU12smokes及跨arm配对检查：同seed初始模型和每步数据、Sum/Mean2数值对照、future新增head rows与相同数据。审查者没有启动任何GPU。
