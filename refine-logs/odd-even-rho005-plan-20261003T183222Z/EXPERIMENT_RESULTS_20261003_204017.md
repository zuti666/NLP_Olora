# 已有结果核验与本轮计划状态

核验时间：2026-10-03T20:40:17.851464+02:00。仅CPU读取和重算，未进行新训练或新模型诊断。

**完整性：**选定7份正式证据7/7通过，0失败、0缺失；它们属于三个协议组。新的七组局部面板尚未运行，4种公式已有历史完整结果、3种缺失。所有数据均seed2，不计算跨seed标准差。

| 组别 | 更新公式 | 后端/CE每步 | AAA | FAA | LA | BWT | Forget |
|---|---|---|---:|---:|---:|---:|---:|
| local_historical_native | g0 | native/4 | 60.337 | 40.140 | 51.005 | -12.072 | 12.240 |
| local_historical_native | g_plus = g0+s+e | native/4 | 63.234 | 44.578 | 53.703 | -10.139 | 10.269 |
| local_historical_native | g_minus = g0-s+e | native/4 | 51.090 | 25.368 | 45.184 | -22.018 | 22.018 |
| local_reflected_math | g0+(g0-g_minus) = g0+s-e | math/2 | 62.615 | 43.714 | 53.515 | -10.890 | 10.890 |
| outer_location_math | g(theta+.25*N(s+e)) | math/4 | 66.621 | 49.995 | 56.600 | -7.339 | 7.339 |
| outer_location_math | g(theta+.25*N(s-e)) | math/4 | 66.696 | 49.952 | 56.488 | -7.262 | 7.386 |
| outer_location_math | g(theta+.25*N(s)) | math/4 | 66.686 | 50.008 | 56.577 | -7.299 | 7.423 |

全部AAA/FAA为任务等权口径，遗忘单位百分点。新Backward最终样本加权准确率51.22%不能与其任务等权FAA43.714%混用。

可以得到的结论：

1. 新反向回加已完整完成，AAA62.615/FAA43.714。其旧启动记录尚是历史快照，完成证据以final_summary/runtime completed为准。
2. 同一旧native协议中g+优于g0，而g−较差；新math反向回加与旧SAM差−0.619AAA/−0.864FAA只是跨协议背景，尚未证明相对SAM收益。
3. math外层P/C/CD结果接近，但它们最终使用外点g2；不能填作局部g0+s及其±e结果。
4. 已有18批同状态诊断中e/s均值约.370，e/g0约.040，e并非零；成绩接近不能排除其作用。
5. 下一步固定rho=.05，先补B/Local-CD/SAM的共同协议，随后补CE/Negative/±e；不扫描系数。

复用范围：既有矩阵、配置、完成收据、18批诊断可以直接引用；新B训练结果的三探针实现复用需未来身份核验。旧native基线作为背景，新math共同协议需重跑。Local-CD旧suite.jobs仍为空；Avg/Ref暂无结果。

详见EXPERIMENT_PLAN.md；逐行原始文件与SHA见EXISTING_RESULTS.csv、RESULTS_AUDIT.json。
