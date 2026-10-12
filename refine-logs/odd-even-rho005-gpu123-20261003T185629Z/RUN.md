# 三探针奇偶分量实验：GPU 1、2、3

日期：2026-10-03。当前独立目录 `/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z`。只准备本轮七组的配置与入口；实际状态以本目录运行收据为准。旧目录、已有结果和其他队列不作修改。

## 方法与预算

固定 seed2、ImageNet-A 10×20 类、ViT-B/16 Q/V LoRA rank10、rho=0.05、FP32、math attention。每批从相同 theta0 与 RNG R0 读取 g0、g_plus、g_minus 三个梯度；正负位置分别为 theta0 +/- rho*g0/(norm(g0)+1e-12)。令 s=(g_plus-g_minus)/2、e=(g_plus+g_minus-2*g0)/2。仅最终更新方向不同；探针顺序和计算预算一致。恢复 theta0 后执行唯一一次普通 SGD；成功后只保留 g0 前向后的 RNG R1。没有外层半径，兼容字段 gam_grad_norm_rho=0.0，outer_radius=null。

| ID | 最终方向 | 正式接续 |
|---|---|---|
| OE01_SAM_s2 | g_plus | 七组短测和CPU配对通过后首批 |
| OE02_Backward_s2 | g0+(g0-g_minus) | 同上，三探针重新训练 |
| OE03_Central_s2 | g0+(g_plus-g_minus)/2 | 同上 |
| OE04_CE_s2 | g0 | 首批三组全部完成后 |
| OE05_Negative_s2 | g_minus | 同上 |
| OE06_Average_s2 | (g_plus+g_minus)/2 | 同上 |
| OE07_ReflectedAverage_s2 | g0+(g0-0.5*(g_plus+g_minus)) | 同上 |

全部14份训练配置保留10任务、50epoch科学设置。7份 smoke 配置只通过实际入口 --max-tasks 2 --max-epochs 1 --max-batches 1 限制预算。每个短测2次SGD、6次训练CE、12次隔离重放CE；每个正式2550次SGD、7650次训练CE、60次隔离重放CE。另有一个无GPU无训练的七组配对作业，总计15job。每个任务首批进行两次三探针零更新重放，每任务6次诊断CE。

固定batch128、首任务LR=.04／后续=.02、cosine T_max50/min ratio=.1、momentum=0、weight decay=0。正式每任务每epoch批数 `[10,8,8,5,4,3,3,3,4,3]`，保留短尾批。无EWC/Fisher、权重融合、SGV、HVP/create_graph、外层探针、梯度投影或最终方向归一化。主要汇报任务等权AAA、FAA、LA、BWT和遗忘，保存完整任务矩阵；sample-weighted accuracy分开标注。

七组配对检查在同一初始任务0状态比较实际g0/g_plus/g_minus和探针位置身份；两个短测任务比较输入/增强、类别、冻结起点和新增FC行身份。更新方向不同后，训练得到的参数和梯度允许不同，不能强制相等。

## CPU准备与冻结

build_suite.py --materialize 拒绝覆盖已物化suite、pins或运行状态。入口为本目录 trainer/odd_even_train.py，实际引擎 trainer/odd_even_step.py。复制的旧测试证据不验收新方法。

冻结前必须有：gates/INPUT_ASSETS.json，当前 trainer/ODD_EVEN_CPU_GATE.json（至少15个CPU测试、3个最终代码哈希），14条真实入口 --dry-run 收据 gates/TRAINER_ENTRYPOINTS.json，以及独立审查 gates/INDEPENDENT_REVIEW.json，状态均passed。root准备完整计划后执行：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z/build_suite.py --freeze
```

冻结包含本地Python、配置、环境/190项源清单、实验计划、运行说明和CPU证据。已冻结或已开始的队列禁止重冻。随后独立代理原样执行 ENVIRONMENT.md 的CPU文档命令，并写 gates/DOC_CHECK.json。该动态验收收据不参与pins，避免循环哈希；实际启动要求其通过。

## GPU调度与运行

仅物理GPU1、2、3，最多3个并行GPU作业，device:[0]表示UUID绑定后的逻辑GPU0。共享 `/tmp/olora_gamfs_gpu_locks` 的物理编号+UUID双锁，worker和实际trainer继承双锁。分配前fresh查询，取得两把锁后再次fresh查询，显存必须<500MiB、利用率<10%，两次UUID一致。忙卡只等待；不预占、不抢占、不自动重试。每个GPU worker训练前执行新的seed2 CUDA witness，验证FP32前后向、UUID和TF32关闭。

7个短测先运行；每组完成须通过实际三CE/一SGD、六CE/task重放、身份、任务矩阵和零禁止模块的强验证。全部短测通过后执行CPU配对；配对通过才放行首批3个正式条件。首批3个全部完整通过后，其余4个依次占用空闲的授权GPU。任何失败停止新增作业，已运行任务自然退出。原队列与旧结果保持独立，禁止外部通知。

root完成门禁后执行以下实际后台控制器入口：

```bash
/home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z/run_queue.py
```

状态：runtime/queue_state.json。日志：runtime/jobs/<job>/console.log。训练输出：outputs/<job>。runtime/PAUSE只阻止新任务，不终止进行中任务。collect_results.py输出results.json、summary.csv、RESULTS.md；只有status=completed且完成证据通过的结果才可作为最终比较。
