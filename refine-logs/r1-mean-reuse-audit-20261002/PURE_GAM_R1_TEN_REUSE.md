# ImageNet-A：原生 r=1 十方向复用汇总

完成10/10；原九方向表新增已核验的正常学习率Mean。
统一seed2、rank10、r=1、rho=.05、10任务×50epochs、无EWC/Fisher/权重融合；AAA/FAA为任务等权。

|方向|AAA (%)|FAA (%)|LA (%)|BWT (pp)|遗忘 (pp)|
|---|---:|---:|---:|---:|---:|
|原版 GAM，γ=0.1|68.029|55.341|61.719|-7.087|7.087|
|g1+g3，γ=0|67.861|54.726|61.583|-7.619|7.619|
|Mean：(g1+g3)/2，LR×1|65.951|49.927|57.172|-8.050|8.306|
|仅 g1|62.166|45.132|53.436|-9.227|9.482|
|仅 g3|65.993|50.679|56.728|-6.721|7.011|
|仅 g0|59.453|39.338|51.360|-13.358|13.604|
|仅 g2|63.838|49.989|54.856|-5.408|6.280|
|g0+(g3-g2)|64.102|47.870|54.714|-7.604|7.604|
|A：g0+(g2-g1)|64.595|49.807|55.375|-6.187|6.990|
|B：g1+(g2-g1)⊥g1|64.826|47.590|54.243|-7.392|7.561|

Mean：beta=(.5,1,.5)、gamma=0、lambda_flat=1；首任务LR=.04、后续=.02、cosine最低LR×.1。
Mean与原表共享配置相同，除方向参数和输出来源字段。原始矩阵/完成凭据已核验；历史Mean无逐步trace，不追溯宣称新A/B等级的逐步或跨GPU轨迹配对。
配置：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_meansum_r1_k10_s2.json
完成凭据：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/runtime/jobs/s3_puregam_meansum_r1_k10_s2/completed.json
原始矩阵：/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/outputs/s3_puregam_meansum_r1_k10_s2/logs_inc_lora/ewclora_normfisher_gam_intask_damp/s3_reference_native_expanded_scan/imageneta/2/s3_puregam_meansum_r1_k10_s2/exp_run/20/s3_puregam_meansum_r1_k10_s2_vit_base_patch16_224_cl_metrics.json

formation目录Mean2=半和+LR×2仅训练Task0，seeds0/1/2各500步。未找到匹配纯GAM的Mean2 LR×2完整十任务训练；不进入本表。
Mean相对对照差值（百分点）：
- g1+g3，γ=0：AAA -1.910，FAA -4.799。
- 仅 g1：AAA +3.785，FAA +4.795。
- 仅 g3：AAA -0.042，FAA -0.752。
