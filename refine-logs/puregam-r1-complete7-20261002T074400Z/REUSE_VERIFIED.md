# 历史 r=1 四方向复用核验

审计 UTC：2026-10-02T07:19:09.299587+00:00。只读原实验；仅新增本审计 JSON 与 Markdown。

四条完整十任务结果可复用；g0、g2、UG=g0+(g3−g2) 尚未发现同协议完成结果。当前记录是 seed2 单次实验，不支持多 seed 或逐步 bitwise 配对声明。

| 方向 | Task AAA | Task FAA | Sample AAA | Sample FAA |
|---|---:|---:|---:|---:|
| original_GAM_gamma_0.1 | 68.028517 | 55.341 | 71.253 | 62.08 |
| g1_plus_g3 | 67.860863 | 54.726 | 71.103 | 61.49 |
| g3 | 65.993172 | 50.679 | 69.998 | 58.85 |
| g1 | 62.165531 | 45.132 | 65.842 | 51.88 |

## 核验范围

配置/runtime 配置、started/completed、suite/source manifest、runtime scripts、原始 metrics 与现存文件 hashes 全部匹配；190 个源码文件逐文件匹配。每条 10 tasks × 50 epochs = 500 个唯一 task/epoch；最终10×10矩阵55个已见元素有限，与receipt一致。未见任务45个NaN属于结构空值。

每条10次 TaskStart has_fisher=False、10次实际 task-end Fisher skip；10任务 EWC/Fisher loss、Fisher gradient norm均0，实际 merge applied=False。

共享配置完整保存在 JSON 的 shared_config，仅剔除 β1/β2/β3/γ 与 prefix/outputs_root。原四条实际差异仅 β1/β3/γ 和输出 provenance；β2固定1。

协议为 rank10、r1、rho=.05、seed2、batch128、50/50epochs、init_lr=.04/lrate=.02、cosine floor=.004/.002、momentum/weight_decay=0、纯GAM（Fisher skip、EWC=0、fusion关闭）。mechanism_eval=True、mechanism_grad_stats=True 必须保留；其他额外评测开关详见 shared_config。

原流程直接进入 native src.main，初始 set_random(2) 后使用原生全局 DataLoader RNG。没有逐step batch身份轨迹；新epoch重播种/诊断隔离wrapper不能直接当历史严格配对复现。

**Checkpoint限制：每条50个tensor文件＋10个meta JSON均存在，覆盖Task0–9；仅确认文件集合存在。旧receipt没有checkpoint hashes，未读取tensor，不宣称历史checkpoint hash绑定或内容验收。**

## 原始路径与绑定

历史根目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z`。目录本身无单一文件SHA；精确绑定见JSON逐文件字段。
Suite SHA256：`5d6e7ef40ba78a981348644caaa44284d76cfa23e71af4c0072573cc11aaa1e6`。
Source manifest SHA256：`e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2`。
Native source：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source`。

### original_GAM_gamma_0.1 — s3_puregam_original_r1_k10_s2

- Config：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_original_r1_k10_s2.json`；SHA `84bc993f0b769941d592f28bcc189156b1873e5d3022db6d55ccd1635f42fb29`。
- Completed：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_original_r1_k10_s2/completed.json`；SHA `b3e79baaf120f281ce7917768408a75f6661d863f32c213b39a0f548d19e793b`。
- 原始 metrics：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_original_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_original_r1_k10_s2/exp_run/20/s3_puregam_original_r1_k10_s2_vit_base_patch16_224_cl_metrics.json`；SHA `8bec3a82690f7ae87d445c214a581e6e52b29aefffb6ae67d1dce6d8082447fe`。
- Checkpoint目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_original_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_original_r1_k10_s2/exp_run/checkpoints`。
- 完成 UTC：2026-09-30T20:37:52.139908+00:00。

### g1_plus_g3 — s3_puregam_gamma0_sum_r1_k10_s2

- Config：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_sum_r1_k10_s2.json`；SHA `ec9a09efcbd44ea7d50bb6388b351d8f79c96d6d6a230eb75e3beb2801bd3ea9`。
- Completed：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_sum_r1_k10_s2/completed.json`；SHA `2050bedf478645dfaf8a7e3616f9374b46af91f9787e9044c9a777b3cce3a3b5`。
- 原始 metrics：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_sum_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_sum_r1_k10_s2/exp_run/20/s3_puregam_gamma0_sum_r1_k10_s2_vit_base_patch16_224_cl_metrics.json`；SHA `b76528fd29ecbc09476eedf958f914826d87f8a656a1b8598a008b698751e73e`。
- Checkpoint目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_sum_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_sum_r1_k10_s2/exp_run/checkpoints`。
- 完成 UTC：2026-09-30T20:37:44.515663+00:00。

### g3 — s3_puregam_gamma0_g3_r1_k10_s2

- Config：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_g3_r1_k10_s2.json`；SHA `3402fdd302fda0ab5aa84c92b548529b54e293a55571e0b0c111802b137fabc8`。
- Completed：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_g3_r1_k10_s2/completed.json`；SHA `1631a2b92db364ce8da50ca7c75595be815ddfda2e62bb034d882226bfa8d94e`。
- 原始 metrics：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g3_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g3_r1_k10_s2/exp_run/20/s3_puregam_gamma0_g3_r1_k10_s2_vit_base_patch16_224_cl_metrics.json`；SHA `202668d55bc43c0310beb2ce7bd38207807fca8590cfd452e6aabc79732afdba`。
- Checkpoint目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g3_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g3_r1_k10_s2/exp_run/checkpoints`。
- 完成 UTC：2026-09-30T20:40:38.700995+00:00。

### g1 — s3_puregam_gamma0_g1_r1_k10_s2

- Config：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_g1_r1_k10_s2.json`；SHA `01525a21328b77644a06126ba93bbb09648982a8872acddd07d990c2f8c7b557`。
- Completed：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/runtime/jobs/s3_puregam_gamma0_g1_r1_k10_s2/completed.json`；SHA `e9204681c10e4f2462c1fdca9a07989876006e17776a6104e35000139849142a`。
- 原始 metrics：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g1_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g1_r1_k10_s2/exp_run/20/s3_puregam_gamma0_g1_r1_k10_s2_vit_base_patch16_224_cl_metrics.json`；SHA `6810f533a16081569cca1dfc5d7f721e54ed35ffe48c15e36e3868fae4ea0bc5`。
- Checkpoint目录：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/outputs/s3_puregam_gamma0_g1_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_gamma0_g1_r1_k10_s2/exp_run/checkpoints`。
- 完成 UTC：2026-09-30T20:40:10.357134+00:00。

## 新配置审计

现已生成 suite 与8份配置：3 formal（g0/g2/ug）＋5 smoke（上述三条＋native_control/g1_control）。静态核验通过：formal共享配置逐键完整相同，仅输出prefix/root变化；native β=(1,1,0)、γ=0作为公共四探针路径，实际方向由suite direction_selector指定。各formal依赖全部5smoke。Python/source/environment保留历史值，source manifest hash完全一致。

Smoke另有明确预算例外：init_epoch=epochs=1、max_train_tasks=2，native scheduler T_max随此预算改变；其用途是最小执行/身份对照，不代表50epoch学习率轨迹验收。formal保持原50/50epochs。当前审计仅验证config/suite，不替代selector代码审阅或GPU验收。

当前suite SHA256：`0274cc3ede0372194c405f63a0f2aa90d872dcaa8770ea25afc591741ba34f6e`；审计 UTC：2026-10-02T07:20:23.700450+00:00。

历史检索边界：本轮结合既有 REUSE_AUDIT，重新枚举 CopyFlatCV/refine-logs、logs_try、GAM-EWC 中9011个JSON并解析958个ImageNet-A配置。r=.25 component、单Task0、早期10/20epochs SGD、r=0 g3均不用于填补r1三缺项。
