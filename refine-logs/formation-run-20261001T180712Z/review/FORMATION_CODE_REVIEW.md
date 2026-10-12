# Formation 独立代码审查

结论：当前固定源码通过静态与独立 CPU 审查，可进入 formation GPU smoke。该结论不覆盖 future_probe 或 queue。

- 有效映射包含所有 trainable LoRA A/B 与 FC，FP64低秩 Gram 计算实际 BA 差；16项显式矩阵对照通过。
- Gmatch 使用另一条同 seed S 的实际逐步 delta；hardening 后在训练前拒绝不完整、smoke/formal混用、错seed/source/config/hash/步序的参照。
- 每步检查 native decomposition 与 native projector 各自实际运算顺序、4 closures、实际 zero-momentum SGD；unperturb后的基点与probe roundtrip drift分开。
- 8arms各3步 CPU通过，Gmatch最大实际相对误差1.9700560896495168e-7；13个negative cases被拒绝。
- 24个seed/arm的50epoch schedule检查通过，Mean2整条LR（包括floor）为Sum两倍。CPU三步末态最大差5.960464477539063e-8，原生clean+(selected-clean)引入ulp差，不错误要求不同运算路径逐位等价。
- 固定seed/data/rho/rank/raw配置、500step参照、每步输入/标签/索引hash、数据RNG隔离均核对。

限制：GPU full model/真实数据/完整checkpoint恢复仍需smoke；零target当前按INVALID拒绝（安全fail-closed），没有实现设计建议的no-op；量化的是native unperturb后的SGD步长，不能把单列roundtrip数值漂移混同match目标。

源文件SHA、CPU报告SHA、详细审查项见 FORMATION_CODE_REVIEW.json。
