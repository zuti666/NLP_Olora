# 第二轮 raw-EWC λ 细化：独立入口

本目录执行已冻结 `refinement_spec.json` 中的六个 seed0 浮点 λ，trial ID 为 `refine2_01_seed0` 至 `refine2_06_seed0`。两个区间 `[3000, 10000]`、`[173205.0807568877, 227950.70569547775]` 各取三个对数四分点；实际 λ 以规范中的完整 JSON 数值为准。每点从原正式 SGD Task0 完整权重继续训练 Task1 50 轮，复用初轮同一份 raw Fisher 和 PREPARED 凭证，SGD、lr .02、batch128、FP32、TF32 关闭、任务 head、RNG 和输入配对均沿用已验证协议。Task0 不重训，Fisher 不重算，两种 trace 归一化均为 false。

`train.py` 与 `../ewc_lambda_refinement_round1_20260922/train.py` 逐字节相同，SHA256 均为 `a03892b30727a5f8b6ea3545de480302d57cb4b9f1c3cd14a664ddff5e71c6c5`。其 prepared-Fisher 默认路径已指向初轮共享文件；同级目录下 FORMAL、EXECUTION 和 frozen source 的解析不变，HERE 将本轮输出隔离到本目录。训练 loss、更新、Fisher 恢复、梯度诊断、有限数值检查和完整终态验证未改动。

`run_trial.py` 只相对 round1 修改三处：规范要求 `round=2`、对应错误提示和 `TRIAL_RESULT.round=2`。wrapper SHA256 为 `8ef8dc386d7815f9fa76553b01b172c974cbf1fbb8d61b8625e1fc892b7762ea`；round1 wrapper SHA256 为 `b02d8a0273de735aaa4918c305689499bc4714a5734f7d130e005f8002152fc1`。回归测试将整个 wrapper 与仅做这三处替换后的 round1 文件比较。

wrapper 要求唯一安全 `trial_id` 和唯一有限 JSON 数值 λ（范围 `[0, 1e9]`），CLI ID/λ 必须精确匹配规范。`initial_sweep`、`prepared_fisher` 路径和所有 `input_fingerprints` SHA 校验保持不变。`--spec` 默认为本目录 `refinement_spec.json`，规范冻结 SHA256 为 `7dca77cc2e38ed9ade865ea0791e7f4f6332838435488c3f3bb7721e1b687f2a`。

`trial_id` 直接作为原生 `--run-name`，输出为 `training/<trial_id>`；浮点 λ 使用完整 `repr(float(...))` 传参。原生入口在同一进程运行，以保持外层 GPU 和 pipeline 锁的生命周期。manifest 的 identity、runner SHA、请求配置、prepared-Fisher 身份、seed 和 scope 均精确核对。只有完整两任务、Task1 50 轮的 `COMPLETE.json` 且退出 0，或身份一致的真实 `DIVERGED.json` 且退出 86，才能记录 `TRIAL_RESULT.json`。OOM 等技术失败不能记为已观测数值发散。完整回执字段、规范 SHA 和 wrapper SHA 记录方式均沿用 round1。

`--resume-last` 对已完成点调用原生完整复验；若已存在数值发散终态，只在精确身份校验后恢复观测回执，不重跑该点。部分训练继续遵守原生恢复规则。外层控制器负责本轮资源准入、可用 GPU 和并发策略，本文件不另行覆盖调度政策。

CPU 验证命令（在本目录运行）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python -B -m unittest -v test_train
```

2026-09-22 执行结果：17 项通过，测试耗时 11.440 秒，退出码 0。保留原有科学配置、raw-Fisher 恢复和非侵入梯度诊断回归，并验证本轮六个真实 λ 的无损传参、独立输出路径、round1/round3/非 seed0 规范拒绝、round2 观测标记、输入篡改拒绝、技术失败与数值发散区分。`test_train.py` SHA256 为 `422eb587bade8a6bcc0d1bc624d2d57843deb9fe12ee67f42f5a6a6b49059afd`。测试中的模型仅为 CPU 小型梯度夹具，本轮实现验证未启动 GPU、训练进程或控制器。

全网络 CPU 恢复使用同码既有证据：`../ewc_lambda_refinement_round1_20260922/cpu_validation/cpu_refinement_float_restore/CPU_VALIDATED.json`，SHA256 为 `d463d0eee617772bb6b0851a03aa6693e885a5a43b1833b85f197e40c49a2d1c`。该回执的 runner SHA 与本轮完全相同；已验证 24 个 raw Fisher 张量、raw trace `34.069888104965315`、两种 normalize=false、新 head SHA `05d16669f438d5ad33621d339fea510fef78c6c917131a70bc769873d592d7b2`。这属于 round1 在 λ `13160.740129524924` 下的完整构模/恢复证据，并非本轮新点的运行结果；本轮不重复巨大构模。既有回执明确 `forward_tested=false`、`training_started=false`。

共享 raw-Fisher payload SHA256 为 `cfdcd801f773d3d33cb48885c84ff31f032a60bf4a4cccbdead5008d67aacd95`，PREPARED SHA256 为 `76a06376cb87e5a21bcb92c26c9a66890b5fb280d36fa372bc57dcc54f1e3cf1`。所有既有 initial/round1 文件保持不变；本实现只创建 round2 的入口、测试和说明。
