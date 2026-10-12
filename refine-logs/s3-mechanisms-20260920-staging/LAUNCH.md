# S3 机制验证：执行入口

2026-09-20：用户先授权 GPU4/5/6，随后授权空闲 GPU0/1。本轮当前只使用物理 GPU0、1，最多同时两项。其他实验不受修改或终止。

共18项正式训练，每项 ImageNet-A 10任务×50epochs；复用上一轮8项完整结果。权重融合中心/保持快照评估的无融合对照采用seeds1/2/3配对，其余构造与几何机制为seed2探索。先执行20个2任务×2epochs集成短测，包括保存边界开/关的跨运行复现性诊断，以及同一原生状态下的精确等价性验证。短测不作为正式准确率。

执行链：GPU空闲 → 20项短测 → 全量结果/参数有效性/同状态observer等价性检查 → 18项正式训练 → 自动汇总与CPU融合交叉项分析。任何短测失败、源代码或配置哈希不符都会阻止正式启动。显卡占用时保持等待，不终止其他任务。预计正式部分约45GPU小时，双卡约22.5小时，等待时间和不同算法的速度另计。

查看实时状态：

- `PIPELINE_STATUS.json`：全流程阶段；只有`formal_queue_active`才表示正式队列已启动，具体任务仍以tracker为准。
- `preflight/smoke_attempt2/EXPERIMENT_TRACKER.md`：20项短测进度。
- `preflight/smoke_review.json`：仅在全量短测完成后生成，不能预先视为通过。
- `EXPERIMENT_TRACKER.md`、`results.csv`：18项正式任务进度与结果。
- `MECHANISM_RESULTS.md`、`mechanism_summary.json`：同协议已有/新增结果，融合三seed配对差值。
- `boundary_analysis/`：训练完成后的factor融合与直接ΔW混合差异。

`REVIEWER_MECHANISM_PLAN.md`与JSON固定对照；`EVIDENCE_REUSE.md`记录既有证据及跨协议限制。190份父源码保持字节不变，仅增加一个单独哈希的边界控制模块。observer不新增forward/Fisher，不改变RNG；Fisher位置消融只移动既有的一次估计。未采集双Fisher余弦与SVD投影，不能据此宣称已验证。

当前状态请读取实时文件；本文只说明执行流程，不代表短测或正式训练已经完成。


修复记录：初次短测误用了nb_tasks/total_sessions限制，训练器会覆盖它们，导致observer开/关各实际运行10任务×2epochs，验收按预期拦截，正式任务没有启动。保留原smoke_campaign全部失败记录。新短测目录smoke_attempt2显式设置max_train_tasks=2，并检查启动日志与配置；其余科学参数未变。


Observer verification update: attempt1 cross-run equality failed before the first observer boundary (epoch1 snapshot CE already differed). Preserve that reproducibility failure; its source is unresolved, and do not attribute it to the observer. NATIVE_BOUNDARY_WITNESS.json instead tests six actual frozen native boundary paths at identical saved states: all50 factor/classifier parameters, gradients, buffers, modes and native metrics exactlymatch; RNGunchanged,0extra forward/Fisher. Formal eligibility requires this same-state witness and every real boundary invariant. Separate GPU-run matrix/tensor comparisons remain fully reported diagnostics, not a claim of bitwise reproducibility. This does not change training kernels/seeding/backend settings or any performance acceptance test.
