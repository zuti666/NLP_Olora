# ImageNet-A 正向、反向、中心比较：配置与完整结果

核验时间：2026-10-12T12:03:42.041321+02:00（Europe/Paris）。

范围：覆盖原点局部响应、第一段外点定位、跨点残差、末端差分正反号、联合后向替换；参数空间中点作为相关附表。只整理已完成实验，不运行或更改队列。所有表AAA/FAA为任务等权百分比；差值为百分点。

本报告索引 105 个不同输出目录的完整训练；复用行按来源去重。旧两探针和新三探针Backward虽是两次实际训练，但轨迹相同、seed相同，不能计成两份独立统计重复。各研究之间后端、读数预算或确定性协议不同，按表内比较。

| 实验家族 | 有效/登记 | 新增不同输出 | 复用输出 |
|---|---|---|---|
| local7 | 7/7 | 7 | 0 |
| backward2 | 1/1 | 1 | 0 |
| native025 | 16/16 | 16 | 0 |
| native1 | 16/16 | 12 | 4 |
| repeat01 | 32/32 | 32 | 0 |
| location3 | 3/3 | 3 | 0 |
| cdsum | 4/4 | 4 | 0 |
| positive_sum | 2/2 | 2 | 0 |
| terminal16 | 16/16 | 16 | 0 |
| joint4 | 6/6 | 4 | 2 |
| midpoint4 | 4/4 | 4 | 0 |
| stage2_math | 4/4 | 4 | 0 |

**共同设置**：ImageNet-A 5981 train/1519 test，固定200类0…199，10×20任务；ViT-B/16 augreg2_in21k_ft_in1k，冻结骨干，Q/V LoRA rank10（无alpha/rank缩放）与已见FC可训练。每任务50epochs、batch128、SGD momentum/WD=0、LR首任务.04/后续.02、cosine最低.1倍。rho=.05，FP32、TF32/AMP关闭。无EWC/Fisher/融合/replay/SGV/HVP。除明确matched行外都是自然步长；没有最终投影。默认seed2，r=1旧四分支另外补齐seeds0/1/2。

统一记号：N(v)=v/(||v||+1e−12)，g(v)=∇L(v)，g0=g(theta0)，g±=g(theta0±rho N(g0))；s1=(g+−g−)/2，e1=(g++g−)/2−g0。不除以rho。所有最终更新在恢复theta0后执行theta0−eta U，探针不是已接受优化步。

## 1. 原点局部响应：正向、校正反向、中心

math attention，同状态/RNG三读数；无外层r。七组全完成。

| 更新U | AAA / FAA | LA | Forget |
|---|---|---|---|
| g+=g0+s1+e1 | 63.102 / 44.809 | 53.477 | 9.631 |
| 2g0−g−=g0+s1−e1 | 62.615 / 43.714 | 53.515 | 10.890 |
| g0+s1 | 62.994 / 44.327 | 53.614 | 10.319 |
| g0 | 60.232 / 39.260 | 51.062 | 13.246 |
| g−=g0−s1+e1 | 50.801 / 25.235 | 45.578 | 22.603 |
| g0+e1 | 60.944 / 41.290 | 52.104 | 12.060 |
| g0−e1 | 59.262 / 37.932 | 50.505 | 13.970 |

g0+s1相对g0提高2.763AAA/5.067FAA；正向比中心仅高.108/.482，单seed不足以确立优势。单独+e1有效、−e1退化，不能称偶响应完全有害。直接g−与校正反向2g0−g−不是同一方法。旧两读数Backward=62.615/43.714，与本表新三读数矩阵及30权重文件一致，仅作实现复现。

## 2. 第一段探针及外层定位符号：P/D/C/R

g2^X=g(theta0+rN(qX))。C表示校正反向，不是中心差分；中心另记CD。

| 分支 | 首侧梯度 | 定位q |
|---|---|---|
| P | g+ | g+−g0 |
| D | g− | g−−g0 |
| C | g− | g0−g− |
| R | g+ | g0−g+ |

O_X=g2^X；A_X=g0+(g2^X−g_first)，P/R的g_first=g+，D/C的g_first=g−。native attention，每步g0/g+/g−/g2四读数。matched以候选自身当前状态raw g+的名义LR更新为参照，正标量匹配实际LoRA BA+全部已见FC位移；不是与独立SAM轨迹匹配。

| 条件 | r=.25 seed2 AAA/FAA | r=1 seed2 AAA/FAA |
|---|---|---|
| S1-01_CE_natural | 60.337 / 40.140 | 60.337 / 40.140 |
| S1-02_Splus_natural | 63.234 / 44.578 | 63.234 / 44.578 |
| S1-03_Sminus_natural | 51.090 / 25.368 | 51.090 / 25.368 |
| S1-04_O_P_natural | 66.663 / 49.937 | 65.735 / 50.250 |
| S1-05_O_D_natural | 46.887 / 24.999 | 37.359 / 23.187 |
| S1-06_O_C_natural | 66.647 / 49.793 | 65.774 / 49.192 |
| S1-07_A_P_natural | 65.900 / 49.141 | 65.842 / 49.325 |
| S1-08_A_D_natural | 62.695 / 45.153 | 52.065 / 35.434 |
| S1-09_A_C_natural | 66.758 / 49.864 | 64.438 / 47.185 |
| S1-10_O_R_natural | 48.943 / 26.857 | 38.561 / 24.355 |
| S1-11_A_R_natural | 2.169 / 0.485 | 2.006 / 0.882 |
| S1-12_A_P_matched | 65.509 / 48.042 | 63.325 / 44.015 |
| S1-13_A_D_matched | 63.085 / 45.166 | 58.756 / 43.944 |
| S1-14_A_C_matched | 65.516 / 48.103 | 63.632 / 45.062 |
| S1-15_A_R_matched | 2.031 / 0.554 | 2.318 / 1.111 |
| S1-16_CE_matched | 59.955 / 39.565 | 59.955 / 39.565 |

r=1表中g0、g+、g−和g0-matched四项复用r=.25的半径无关结果，没有重复训练。

### r=1：已补齐三seed

以下±为样本标准差（ddof=1），每行n=3；不能与上表单seed数字混写。

| 条件 | AAA mean±SD | FAA mean±SD |
|---|---|---|
| S1-01_CE_natural | 60.873±0.492 | 40.574±1.613 |
| S1-02_Splus_natural | 63.823±0.520 | 45.375±0.714 |
| S1-03_Sminus_natural | 49.274±4.481 | 17.482±8.889 |
| S1-04_O_P_natural | 65.212±0.889 | 49.694±0.537 |
| S1-05_O_D_natural | 38.477±1.814 | 24.434±1.244 |
| S1-06_O_C_natural | 65.911±0.623 | 50.071±1.002 |
| S1-07_A_P_natural | 65.664±0.229 | 49.874±0.749 |
| S1-08_A_D_natural | 51.996±0.455 | 35.522±0.754 |
| S1-09_A_C_natural | 65.039±0.543 | 48.460±1.869 |
| S1-10_O_R_natural | 39.278±1.577 | 25.302±1.066 |
| S1-11_A_R_natural | 2.018±0.095 | 0.883±0.200 |
| S1-12_A_P_matched | 64.416±0.981 | 46.488±2.343 |
| S1-13_A_D_matched | 57.726±1.098 | 43.413±0.795 |
| S1-14_A_C_matched | 63.013±0.775 | 43.811±2.482 |
| S1-15_A_R_matched | 1.967±0.391 | 0.968±0.458 |
| S1-16_CE_matched | 60.540±0.570 | 40.122±1.398 |

g2家族C−D配对增益27.434±1.254AAA/25.637±.321FAA，3/3同向；P−R同样3/3同向。C−P只有+.699±1.490AAA/+.378±1.528FAA，seed间方向不完全一致。证据支持外点定位差分的取向重要，不能声称校正反向稳定胜过正向。D/R较低遗忘伴随LA约29%，主要还包含学不会新任务，不能称保持更强。

## 3. 中心差分只用于外点定位

math attention；r=.25，seed2，四读数g0/g+/g−/g2，最终一律U=g2。

| 定位q | AAA/FAA |
|---|---|
| P: g+−g0=s1+e1 | 66.621 / 49.995 |
| C: g0−g−=s1−e1 | 66.696 / 49.952 |
| CD: (g+−g−)/2=s1 | 66.686 / 50.008 |

三者最大AAA差.074、FAA差.056，没有中心明显更好的证据。18个共同状态诊断中mean cos(qP,qC)=.738，mean cos(g2P,g2C)=.982，外点方向虽不同，读到的梯度仍很相近；这是描述性诊断，不是Hessian机制证明。

## 4. 负侧梯度与外点梯度相加

native，自然相加不除2。C/D目录名中的cd是两分支缩写，不是中心差分CD。P背景每步未用的第四读数为g3，C/D为g+；分别验收，不能称位级完全相同执行器。

| 分支与更新 | r=.25 AAA/FAA | r=1 AAA/FAA |
|---|---|---|
| P: g++g2P | 66.865 / 50.606 | 68.605 / 54.722 |
| C: g−+g2C | 66.462 / 50.479 | 67.628 / 51.461 |
| D: g−+g2D | 58.766 / 41.123 | 64.869 / 48.709 |

C在相同加法下优于D，但加入g−同时改变更新方向和幅度，不能将D组合的提升完全归因于选址改善。P组合不是正交投影版本。

## 5. 第二局部探针：侧别×残差符号×基底

strict native FP32；第一段固定正向qF=g+−g0，theta2=theta0+rN(qF)。g3±=g(theta2±rho N(g2))，c±=g3±−g2。每步4训练CE，最终基底为g0或g+；16组全部完整完成。

| ID | U | r=.25 AAA/FAA | r=1 AAA/FAA |
|---|---|---|---|
| B0 | g0+c+ | 64.914 / 47.706 | 65.176 / 47.584 |
| B1 | g_+ + c_+ | 65.368 / 48.181 | 65.335 / 47.562 |
| N1 | g0−c+ | 9.039 / 0.222 | 1.911 / 0.619 |
| N2 | g+−c+ | 39.599 / 2.213 | 1.720 / 0.554 |
| N3 | g0+c− | 19.743 / 0.814 | 1.934 / 0.456 |
| N4 | g_+ + c_- | 43.107 / 17.364 | 1.862 / 0.673 |
| N5 | g0−c− | 64.845 / 48.015 | 65.385 / 48.632 |
| N6 | g+−c− | 65.378 / 48.720 | 65.906 / 48.917 |

正侧+正残差、负侧+负残差都正常学习；另外四格明显退化。r=1四种错误取向的FAA仅.456–.673；r=.25部分组先学后退化，不能统称所有组Task0立即崩溃。这些均数值有限、完整结束，是学习失败而非程序失败。

## 6. 外点中心差分与联合后向替换

r=1、strict native。第一段F:qF=g+−g0；B:qB=g0−g−。第二段F:cF=g3+−g2；B:cB=g2−g3−。Uab=g0+cb(theta2^a)。F/B是局部读数与符号校正，不是外层半径r改负。

| 条件 | U | AAA/FAA | 训练CE/步 | 来源 |
|---|---|---|---|---|
| FF | g0+g3Fplus-g2F | 65.176 / 47.584 | 4 | 复用 |
| FB | g0+g2F-g3Fminus | 65.385 / 48.632 | 4 | 复用 |
| CPLUS | g0+(g3Fplus-g3Fminus)/2 | 65.197 / 48.170 | 5 | 新训练 |
| CMINUS | g0-(g3Fplus-g3Fminus)/2 | 2.064 / 0.651 | 5 | 新训练 |
| BF | g0+g3Bplus-g2B | 64.608 / 47.339 | 4 | 新训练 |
| BB | g0+g2B-g3Bminus | 65.468 / 48.935 | 4 | 新训练 |

令s2=(g3F+−g3F−)/2，e2=(g3F++g3F−)/2−g2F；FF=g0+s2+e2，FB=g0+s2−e2，Center±=g0±s2。Center+保持正常成绩，Center−完整训练后接近随机，说明末端共同响应的取向重要；尚缺本组相同strict/读数预算的g0零修正对照，不能说已证明s2是全部增益的必要来源。

BB不读取两处局部正侧梯度，65.468/48.935，与FF相比+.292AAA/+1.351FAA；支持这个g0+末端残差变体能联合采用校正后向估计。不证明完整原版GAM等价，也未减少四次梯度读取。

## 7. 相关附表：参数空间中点，不是上述中心差分

r=1 strict native；g1=g+，theta3=theta2+rho N(g2)；theta_m=(theta1+theta2)/2，g_m=g(theta_m)。中点在LoRA A/B+FC参数坐标取，不是BA有效权重融合，更不是梯度平均。

| U | AAA/FAA | 原生梯度次数 |
|---|---|---|
| g1+g2 | 68.439 / 54.227 | 3 |
| g0+g2 | 68.534 / 54.122 | 3 |
| 2gm | 68.237 / 53.867 | 3 |
| g1+g3 | 68.669 / 54.641 | 4 |

2gm比g1+g2低.203AAA/.360FAA，属于计划预设±.5范围内的描述性接近，不能称统计等价。g0+g2与g1+g2接近，g1仍用于外点构造，不能说删除第一探针。整体更新尺度不同，不能与g0+残差直接作纯方向优劣排序。

## 8. 相关附表：基底g0与只用跨点残差

r=.25，math普通梯度接续实验。

| 方向 | 步长 | AAA/FAA |
|---|---|---|
| g2−g+ | natural | 31.823 / 0.334 |
| g2−g+ | matched | 19.040 / 0.674 |
| g0+(g2−g+) | natural | 66.039 / 49.315 |
| g0+(g2−g+) | matched | 65.145 / 47.525 |

保留g0的成绩明显高于仅用跨点残差，matched也不能挽救仅残差；这些结果不等于局部HVP训练成功。

## 9. 完成、迁移与尚未执行

本表列出的完整成绩均来自已验收十任务训练，没有把短测或阶段指标当终点。原P/C/CD和C/D相加旧目录只做短测后切换GPU，正式结果分别在gpu01目录。Local-CD旧suite为plan_only/jobs=[]；其自然方向后来由七组odd-even完成，匹配版本仍未跑。

第二探针首版0正式、短测断言失败；recovery版14短测通过但真实配对未过，0正式，随后由strict版16正式完整替代。旧pending是冻结历史，不代表本表尚有16项等着运行。阶段2Hstart/Hmid自然/匹配4项仍未训练，FP32参考验收未全过；不能声称Hessian机制已验证。

仍需区分的未做对照：原点局部g0−s1和2g0−g+、局部Backward/Central的有效步幅匹配、末端strict同预算g0零修正。除旧r=1四分支外，本文核心中心/末端比较仍为seed2单次；不报告样本SD=0，不宣称统计等价。

## 10. 结论与来源

跨几个实验位置，最一致的现象是：下降侧梯度读数经过差分符号校正后可以有效；直接反向、或只翻转残差符号，常明显退化。中心差分可以保留正常表现，但尚无中心始终最优的证据；奇偶响应在不同位置和最终更新方式下作用不同。不能把测量侧别、差分取向、最终基底及更新尺度混为一个因素。

完整来源：[去重原始结果表](ALL_VALIDATED_RUNS.csv)、[完成账本](COMPLETENESS_LEDGER.csv)、[r1三seed16设置](R1_THREE_SEED_16_SETTINGS.csv)、[配对差](R1_PAIRED_CONTRASTS.csv)。所有source路径、矩阵SHA保留在CSV。

- local7: [odd-even-rho005-gpu123-20261003T185629Z](/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-rho005-gpu123-20261003T185629Z/summary.csv)
- backward2: [reverse-probe-reflected-update-20261003T161621Z](/data/140-1/users/liying/O-LoRA/refine-logs/reverse-probe-reflected-update-20261003T161621Z/summary.csv)
- native025: [g012-two-stage-execution-20261002T122014Z](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/summary.csv)
- native1: [g012-r1-four-branches-20261003T072200Z](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/summary.csv)
- repeat01: [imageneta-r1-seed01-run-20261003T185852Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-r1-seed01-run-20261003T185852Z/summary.csv)
- location3: [g2-central-location-gpu01-20261003T105925Z](/data/140-1/users/liying/O-LoRA/refine-logs/g2-central-location-gpu01-20261003T105925Z/summary.csv)
- cdsum: [imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-cd-g1minus-plus-g2-gpu01-20261005T184209Z/summary.csv)
- positive_sum: [imageneta-positive-g012-combinations-20261005T185032Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-positive-g012-combinations-20261005T185032Z/summary.csv)
- terminal16: [imageneta-second-probe-sign-deterministic-20261007T071400Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-deterministic-20261007T071400Z/summary.csv)
- joint4: [imageneta-center-joint-backward-execution-20261007T135648Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-center-joint-backward-execution-20261007T135648Z/summary.csv)
- midpoint4: [imageneta-g12-midpoint-interaction-execution-20261008T140723Z](/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g12-midpoint-interaction-execution-20261008T140723Z/summary.csv)
- stage2_math: [g012-stage2-revalidation-20261003/ordinary_stage2](/data/140-1/users/liying/O-LoRA/refine-logs/g012-stage2-revalidation-20261003/ordinary_stage2/summary.csv)
