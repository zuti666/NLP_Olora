# S3 机制实验：独立最终审查

审查时间：2026-09-21T15:11:47.689007+00:00。审查只读取 CopyFlatCV，写入本 O-LoRA 目录；未启动 GPU，未改源代码/配置/运行/原结果。

**18 个新增核心实验、8 个复用实验均完整完成，原始结果完整性检查通过。** 三 seed 的融合增益较小，尚不能声称其 FAA 正贡献或 FAA 实用等效；AAA 按预设区间标准等效。GAM 主训练路径贡献明显的证据来自 seed2，Fisher 多个准确率消融接近中心。另发现已有汇总器修正 seed2 误配时把主 cohort 改成 seeds2/3；该缩减没有预注册依据或 seed1 失效凭据，本审查保留原 seeds1/2/3 主分析。

## 完整性与口径

- 26 份 raw 10×10 CNN 矩阵、260 份中间阶段矩阵与最终矩阵一致；FAA、AAA、BWT、forgetting 共104个量从 raw 独立重算。26条 sample-weighted 曲线另从训练日志核验。
- 26份 completed receipt、config、suite、source manifest 及122个运行文件哈希引用均通过；190份原生代码完整保留，新source仅增加observer/control模块，共191份。没有 seed1 特有冻结协议或环境不一致。
- 26作业均有 task0…9 × epoch1…50 的完整日志，无非有限训练 loss；不是只凭最后checkpoint判定完成。20/20预检通过，正式18/18完成，0运行/排队/失败。
- 新18作业的180份边界archive共1,043,964,880字节全部核SHA，独立按原dtype/顺序重构48个因子张量，逐张量bitwise相等；180/180 observer前后RNG不变、干预开关符合配置。dense代数报告的哈希/身份/240×18层行通过，未在本审查重复全部dense乘法。
- 同状态 native observer witness 六个案例通过，0额外 forward/Fisher；跨两次GPU执行的完整轨迹bitwise相等曾失败，原因未解决，不能把同状态中性证明表述成跨运行完全可复现。
- FAA=最终10任务准确率均值；AAA=10个阶段各自已见任务准确率均值再平均。原raw `cnn.final.AAA` 是55个下三角cell直接平均（阶段权重不同），不能和本表AAA混写。中心seed2原字段60.624，本表阶段等权AAA66.745541。
- 此处是 ImageNet-A、200类、10×20任务、ViT-B/16、LoRA rank10、50轮、batch128、8workers、初始/后续lr .04/.02、无replay；Fisher为dense ΔW空间的batch-gradient平方proxy，原生增强/loader/RNG。与RQ2新FP32/逐样本Fisher协议、旧ImageNet-R表及20260918 sample-empirical/batch32/RNG隔离协议分开。

## 原预注册的三 seed 融合配对

`center − matched_no_merge`。后者保留20/40轮snapshot和质量CE前向预算，将merge start设为task10，十任务内不融合。差值单位均为百分点；±为样本SD。

| Seed | Center FAA | Matched no-merge FAA | ΔFAA | ΔAAA |
| --- | ---: | ---: | ---: | ---: |
| 1 | 51.139 | 51.050 | +0.089 | -0.084273 |
| 2 | 53.419 | 53.098 | +0.321 | +0.053323 |
| 3 | 52.006 | 51.935 | +0.071 | +0.060072 |

| 量 | 配对均值 ± SD | 95% paired t CI, df=2 | 预设等效界 | 判定 |
| --- | ---: | --- | --- | --- |
| ΔFAA | +0.160333 ± 0.139432 | [-0.186035, +0.506702] | ±0.5 | 正贡献证据不足；等效未成立 |
| ΔAAA | +0.009707 ± 0.081460 | [-0.192650, +0.212064] | ±0.3 | 满足预设实用等效区间 |

三组FAA均为正，但95%区间跨0，未满足预注册正贡献条件。FAA区间上限0.506702略超过0.5，不能四舍五入后宣布等效。AAA区间完全位于±0.3内。只有3个配对种子，且raw每任务准确率存到0.01pp；区间依赖小样本t模型，应保留这些限制。

## 汇总错误与未经说明的 cohort 变动

原汇总器 `summarize_mechanisms.py` 丢弃所有 `s3_` 前缀override，连科学干预 `s3_fisher_location=endpoint` 也被丢掉。因此endpoint分支被误认作无override中心并覆盖seed2的on_id。正确seed2中心是预注册复用 `s3_center_s2`（FAA53.419），不是endpoint-Fisher（53.454）。错误影响融合聚合，未污染26份训练raw。
原错误三seed ΔFAA=+0.172±0.159603，正确为+0.160333±0.139432；错误AAA区间曾越过+0.3，正确AAA区间在±0.3内，等效判断会改变。
当前CopyFlatCV汇总器于2026-09-21 15:01:34 UTC已被其他流程改为显式ID，`SUMMARY_CORRECTION.json` 时间15:02:48 UTC。该说明只解释误配，没有说明为何另将seeds2/3改称primary、seed1称legacy/supplementary。原计划（mtime 2026-09-20 20:35:18 UTC）仍明确seeds1/2/3；全部seed1 config/source/runtime/receipt检查正常，未找到基于有效性的排除证据。本审查没有改该文件，不能推断未知修改者动机，也不能把新的结果后cohort标签当预注册。
保留2/3作为敏感性展示：ΔFAA=+0.196000±0.176777，df1 t95=[−1.392276,+1.784276]；ΔAAA=+0.056698±0.004772，t95=[+0.013824,+0.099571]。删去seed1会改变AAA的符号/区间解释，故主结论仍采用预注册三seed。当前与备份代码SHA、mtime、计划及receipt路径见JSON `source_provenance`。

## Seed2 完整结果与机制对照

以下全部是单seed探索性控制，Δ均相对同协议中心53.419/66.745541；不与三seed均值混为一列。EWC/clean为task1…9的每任务平均梯度范数比再平均，百分比表示。SGD路径没有记录这一GAM专属梯度统计，记为未测。

| 实验ID | FAA | AAA阶段均值 | ΔFAA | ΔAAA | EWC/clean % |
| --- | ---: | ---: | ---: | ---: | ---: |
| s3_center_s2 | 53.419 | 66.745541 | +0.000 | +0.000000 | 0.144108 |
| s3_lambda0_s2 | 53.500 | 66.761436 | +0.081 | +0.015895 | 0.000000 |
| s3_lambda200_s2 | 53.405 | 66.714967 | -0.014 | -0.030574 | 0.015987 |
| s3_lambda20000_s2 | 53.426 | 66.683944 | +0.007 | -0.061597 | 1.001028 |
| s3_strength0_s2 | 38.036 | 60.165318 | -15.383 | -6.580223 | 0.155075 |
| s3_strength05_s2 | 51.986 | 65.964496 | -1.433 | -0.781044 | 0.102154 |
| s3_radius01_s2 | 52.462 | 66.052027 | -0.957 | -0.693513 | 0.122818 |
| s3_radius05_s2 | 54.978 | 67.886076 | +1.559 | +1.140535 | 0.181206 |
| s3_mech_a_nomerge_matched_s2 | 53.098 | 66.692217 | -0.321 | -0.053323 | 0.144498 |
| s3_mech_gam_half_direction_s2 | 51.834 | 65.692031 | -1.585 | -1.053510 | 0.101672 |
| s3_mech_sgd_ewc_s2 | 40.123 | 61.012882 | -13.296 | -5.732659 | 未测 |
| s3_mech_a_off_s2 | 52.150 | 67.163908 | -1.269 | +0.418367 | 0.125674 |
| s3_mech_fisher_endpoint_anchor_s2 | 53.454 | 66.850227 | +0.035 | +0.104686 | 0.145843 |
| s3_mech_a_uniform_s2 | 53.591 | 66.718142 | +0.172 | -0.027399 | 0.145998 |
| s3_mech_a_latest_s2 | 53.298 | 66.643848 | -0.121 | -0.101693 | 0.148722 |
| s3_mech_fisher_alpha0_s2 | 53.431 | 66.774426 | +0.012 | +0.028885 | 0.228749 |
| s3_mech_fisher_alpha1_s2 | 53.353 | 66.793216 | -0.066 | +0.047675 | 0.002803 |
| s3_mech_fisher_gamma0_s2 | 53.574 | 66.755154 | +0.155 | +0.009613 | 0.209086 |
| s3_mech_fisher_gamma1_s2 | 53.343 | 66.786403 | -0.076 | +0.040862 | 0.142765 |
| s3_mech_fisher_tasknorm_off_s2 | 53.230 | 66.665354 | -0.189 | -0.080187 | 0.143184 |
| s3_mech_gam_projection_off_s2 | 53.261 | 66.540675 | -0.158 | -0.204866 | 0.143878 |
| s3_mech_gam_rho025_s2 | 52.866 | 66.395754 | -0.553 | -0.349787 | 0.127157 |

另外四份非seed2新训练（均完成）：

| 实验ID | Seed | FAA | AAA阶段均值 |
| --- | ---: | ---: | ---: |
| s3_mech_center_s1 | 1 | 51.139 | 67.078499 |
| s3_mech_a_nomerge_matched_s1 | 1 | 51.050 | 67.162772 |
| s3_mech_center_s3 | 3 | 52.006 | 67.511071 |
| s3_mech_a_nomerge_matched_s3 | 3 | 51.935 | 67.450999 |

GAM：true single-forward SGD+EWC为40.123，中心高13.296pp；半完整GAM方向为51.834，低中心1.585pp，但高SGD11.711pp。支持该seed下完整GAM训练路径承担主要准确率差异，不能凭它证明泛化上的普适机制。旧lambda_flat=0仍执行GAM探测，只令组合 `g_clean + lambda_flat(g_GAM−g_clean)` 回到clean方向，结果38.036，不等于trueSGD；lambda_flat=.5也不等于将整个GAM方向乘.5。projection关闭只降.158pp，第一扰动rho减半降.553pp；第二扰动radius=.5的+1.559pp只是旧单seed敏感性结果，不能升级为选优baseline。
Fisher：lambda从0到20000的FAA落在53.405…53.500；阻尼α0/1、history γ0/1、task normalization off、endpoint-Fisher的FAA相对中心均在−.189…+.155pp，未支持该工作点下EWC必需或某个Fisher设计提高准确率。中心平均EWC/clean仅.144108%，clean/GAM余弦.952869，flat/clean范数比1.877472。α1虽准确率近似，却将EWC/clean降到.002803%，说明trace匹配并不匹配沿实际更新方向的梯度约束，不能写成Fisher几何完全无变化。
融合执行：whole-module-off相对matched-no-merge，FAA−.948pp、AAA+.471690pp。这项比较同时改变snapshot/质量前向与RNG消费；不能把其差值当纯权重融合贡献。

## Fisher、参数位移与质量权重到底测了什么

各任务raw `mechanism.steps` 已记录：current_fisher_raw_trace、归一化current_fisher_trace、累计fisher_past_trace_mass、ΔW delta_norm、fisher_weighted_drift、EWC penalty与梯度比、因子融合shift、snapshot CE与权重/effective_n。中心seed2 task1…9 raw Fisher trace均值17.644410（10.387028…28.287242），归一化trace≈1，累计trace mass从1.9到6.513216；ΔW范数均值.221211，Fisher二次位移均值1.884603e−7，因子merge shift均值.010363。这些分别是不同量，不能将归一化trace、累计mass或因子移动当作raw Fisher trace/新任务有效ΔW范数。
Observer另保存endpoint/retained因子、20/40轮snapshot、实际选择权重。新center seeds1/3共20个边界中，19次主要选epoch40；seed3 task2的训练CE=.5078422/.5080230，几乎打平但epoch20更低，于是其权重约.997527。这说明不能说权重总是选末个snapshot。
代码对两个CE先以population std标准化，再softmax(−βz)，β=3。只要差值非零且大于epsilon，两项z近似±1，权重便约.0024726/.9975274，effective_n≈1.017462，与CE差距大小几乎无关。uniform有效数2，latest有效数1；这是质量权重非常尖锐的直接实现/数值证据，不是其带来准确率收益的证明。

| 边界集合 | median 因子融合相对移动 % | median ΔW相对移动 % | median 交叉项/线性ΔW % | max 交叉项/线性ΔW % | median 交叉项/实际移动 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| s3_mech_center_s1 | 0.109530 | 0.414154 | 0.001261 | 0.009633 | 0.303199 |
| s3_mech_center_s3 | 0.105293 | 0.384772 | 0.001126 | 0.095856 | 0.287197 |
| s3_mech_a_uniform_s2 | 0.367459 | 1.321459 | 0.025218 | 0.184048 | 1.903461 |
| s3_mech_a_latest_s2 | 0.107067 | 0.380595 | 0.001079 | 0.008795 | 0.272560 |

边界代数证明因子平均并不严格等于ΔW平均：`B̄Ā−ΣwᵢBᵢAᵢ = −Σᵢ<ⱼwᵢwⱼ(Bᵢ−Bⱼ)(Aᵢ−Aⱼ)`；endpoint/path因子插值还增加 `−ω(1−ω)(B̄−Bₑ)(Ā−Aₑ)`，并有原dtype舍入残差。现有报告的恒等式残差约1e−16。上表分母均明确，交叉项相对完整ΔW很小，但相对实际位移可更大。旧seed2中心没有新observer archive，不能把endpoint干预seed2的archive冒充中心，也不能声称三seed中心均做了这个dense边界审查。
未测：同一checkpoint/loader的endpoint与retained配对Fisher cosine；best rank10 SVD残差；固定batch的counterfactual logit差；改成ΔW/SVD融合后的训练收益；未测GAM与Fisher所有多seed交互。observer实际不额外估Fisher、不额外forward，因此上述不能从当前archive报告凭空得到。

## 尚未执行的可选确认与结论边界

- uniform/latest没有达到|ΔFAA|≥.5或|ΔAAA|≥.3的复测阈值；endpoint-Fisher为+.035/+.104686pp，平均下一任务EWC/clean相对变化+1.203541%，未到10%阈值。
- Fisher准确率阈值未触发。但α1的EWC/clean相对中心降低51.419119倍：按“changes … by at least 2x”的双向倍数变化解释，触发damping两端α0/α1各补seed1/3，共4个可选确认。若原规则仅指增加而不含降低，必须明确说明这一口径；不能事后默默选择。本审查未启动这些作业。
- 中心交叉项/线性ΔW的median远低于1%，SVD训练触发条件第一项未满足；SVD残差与logit门槛本身也未测。
- optional accumulated-normalization-off/alpha0未跑；它是历史相关scale变化，不是可由一项固定lambda替代的no-op。radius=.5复现和lambda0多seed只在要主张相应改善/必要性时按原计划追加，核心18完成不能写成所有可选已完成。
- 可支持：指定协议/seed下GAM训练路径是主要准确率来源；融合配对FAA效应小且区间不确定、AAA按预设实用等效；Fisher消融在当前λ2000工作点准确率变化小；两snapshot权重几乎二选一、因子融合存在但较小的非线性ΔW交叉项。
- 尚不支持：权重融合显著提高FAA，融合FAA已证明等效，Fisher普遍无效，shaping通过降低Fisher而必然降低遗忘，quality weighting优于uniform/latest，SVD/ΔW融合会更好，或与RQ2另一训练/测量协议或旧ImageNet-R数据拼表得出跨设置结论。

复核入口：本目录 `audit_s3_final.py` 可仅用CPU重跑；`S3_FINAL_AUDIT.json` 保留26条原始路径、receipt/config/source SHA、全部统计、180个边界重构记录及源文件mtime。
