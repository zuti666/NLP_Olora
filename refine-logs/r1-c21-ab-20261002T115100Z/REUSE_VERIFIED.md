# r=1 原生协议：七条完成基线复用核验

七条完整训练均通过本次只读核验。前四条来自 /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z；后三条来自 /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z。只在新目录写入本报告与 JSON，未训练或修改历史实验。

| 方向 | AAA task macro | FAA task macro | LA | BWT | forgetting |
|---|---:|---:|---:|---:|---:|
| 原版 GAM，γ=0.1 | 68.028517 | 55.341000 | 61.719000 | -7.086667 | 7.086667 |
| g1+g3 | 67.860863 | 54.726000 | 61.583000 | -7.618889 | 7.618889 |
| g1 | 62.165531 | 45.132000 | 53.436000 | -9.226667 | 9.482222 |
| g3 | 65.993172 | 50.679000 | 56.728000 | -6.721111 | 7.011111 |
| g0 | 59.453081 | 39.338000 | 51.360000 | -13.357778 | 13.604444 |
| g2 | 63.837738 | 49.989000 | 54.856000 | -5.407778 | 6.280000 |
| g0+(g3−g2) | 64.101870 | 47.870000 | 54.714000 | -7.604444 | 7.604444 |

全部指标从各自原始 CNN final 10×10 下三角矩阵独立重算。AAA 为十阶段已见任务均值的等权均值，FAA 为最终十任务等权均值，LA 为十个对角线准确率均值；BWT 与 forgetting 为前九任务等权均值。55个已观察准确率均有限，45个未来任务 NaN 为结构占位。原生 JSON 的 final AAA 口径不同，未替用。

七条配置剔除且仅剔除 β1/β2/β3/γ/prefix/outputs_root 后，与完整 shared_config 相同。ImageNet-A、seed2、rank10、r=1、rho=.05；每任务50 epochs，batch128，首任务LR .04、后续 .02、原生cosine；原生全局RNG和DataLoader，不按任务/epoch重播种。mechanism_eval 与 mechanism_grad_stats 为 true，训练/评估 workers 均8。Fisher和实际融合关闭。

每条绑定 config、复制的运行配置、started/completed 凭据、原始 metrics、console、suite、source_manifest 与运行脚本 SHA256。190个冻结源文件哈希重新检查全部一致；完成凭据绑定的配置、suite、source和metrics哈希与当前文件相同。全部10任务×50 epochs 和跳过Fisher记录均核验。

前四条历史实验没有逐步数据/RNG/公式/SGD trace，其 checkpoint 仅核验文件集存在，旧完成凭据未绑定 checkpoint 哈希，不追认逐步或跨GPU配对。后三条在独立子进程调用其已有 validate_smokes.validate_job_audit：各2550步、四次probe/五次CE、原生cache/方向/projector/SGD身份、零RNG重播种/恢复及完成凭据绑定的checkpoint哈希全部通过。

跨GPU严格轨迹等价未验证，先前跨GPU等价失败的阈值和证据保持原状。结论仅用于单seed同协议基线复用。本报告未使用r=.25数值。新队列只需运行 A/B，不重新训练七条基线。

机器可校验的全部路径和SHA256见 [REUSE_VERIFIED.json](REUSE_VERIFIED.json)。
