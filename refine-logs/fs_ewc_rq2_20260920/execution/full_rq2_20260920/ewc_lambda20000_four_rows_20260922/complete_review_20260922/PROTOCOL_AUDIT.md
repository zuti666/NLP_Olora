# λ=20000 四分支：配置、代码与实际产物独立审计

审计时间：2026-09-22T12:01:54.536148+00:00。只读检查源代码、12个正式模型/Fisher checkpoint及日志，在CPU重算位移；未启动训练或GPU作业。

**结论：12/12正式续训通过本次独立检查。** 每条均完成50轮、400次真实优化器更新，训练Task2（代码task1），Task1模型复用。最晚正式完成时间为2026-09-22 12:47:53（马德里）。共同λ=20000，两个trace归一化开关均为false。

## 实际实验协议

| 项目 | 实际设置 |
|---|---|
| 数据 | ImageNet-A本地预分train/test；固定类序，前40类、20+20两任务 |
| 样本 | Task1 train1259/test307；Task2 train997/test254；合并测试561 |
| 模型 | ViT-B/16，rank10 LoRA，12层Q/V；48个A/B张量、24个有效BA矩阵 |
| 训练范围 | 同一LoRA分支续训；300个非adapter骨干张量与旧20类head均bitwise不变 |
| Task1 | 复用SGD/GAM × seeds0/1/2共6个已完成端点；不重训 |
| Task2 | 50epochs，batch128，lr0.02，cosine，momentum0，weight_decay0，workers4 |
| 新任务损失 | CE仅当前20类，不包含历史样本；新20类head参与训练 |
| EWC | λ20000，η0，rawF，endpoint anchor，无路径平均、无dual ascent/gradient projection/额外ΔW正则 |
| GAM | baseSGD；lambda_flat1；ρ0.05、normρ0.2、beta1/2/3均1、gamma0.1、adaptive=false |
| 精度 | FP32，CUDA matmul和cuDNN TF32均关闭 |
| 评估 | a11为Task1结束20类准确率；a21/a22为Task2结束40类joint CIL分任务准确率；A2=(a21+a22)/2；BWT=a21−a11 |

`endpoint_eval=false`关闭的是辅助endpoint分析开关；原生CNN的t01/final评估矩阵和旧任务local/seen评估均实际生成。未执行的上三角任务单元用NaN表示，已完成的3个矩阵单元均有限，不能把这些占位NaN误判为训练发散。

## Fisher与保护实现

$$\widehat F_1=\frac1N\sum_b |b|\left(\nabla_{\Delta W}\frac1{|b|}\sum_{i\in b}\mathrm{CE}_i\right)^2,\qquad P=\frac{\lambda}{2}\sum_k\widehat F_{1,k}(\Delta W^{(2)}_k-\Delta W^{(1)}_k)^2.$$

这里ΔW=BA，坐标为有效Q/V矩阵。Fisher是batch均值梯度先求和、后平方的原生proxy，使用旧任务全部1259张训练图及训练增广；不是逐样本empirical Fisher，亦不是固定old-train256诊断值。6份保护Fisher均24×768×768、finite且非负。SGD seed0复用此前已重新计算并核验的raw文件，另5份重新做一次原生Fisher pass，所有准备过程training_steps=0。

SGD每步对CE+P反向传播一次。GAM在未扰动端点分别缓存clean-CE和P梯度，4次shaping closure仅含CE，然后原生projector形成clean+lambda_flat×(GAM−clean)+grad(P)，lambda_flat=1时即GAM方向+grad(P)。实际优化器前每一步断言P只注入一次，且未改变原生更新。F表示GAM shaping实现，不能写成显式最小化Fisher trace损失。

以下为本次CPU直接读矩阵得到的尺度对照。旧checkpoint确为trace≈1；将本次raw矩阵除以trace后与旧矩阵的相对L2误差均<1.4e−7。故在同一参数位移处，新/旧保护项的系数比约为10×trace；这是系数尺度，不等于实际训练损失或梯度必增大同样倍数。

| 起点 | seed | raw trace | 旧归一化trace | 新λ20000/raw vs 旧λ2000/normalized系数比 | 等价normalized λ |
|---|---:|---:|---:|---:|---:|
| SGD | 0 | 34.069888 | 1.00000005 | 340.699× | 681397.8 |
| SGD | 1 | 26.555999 | 1.00000003 | 265.560× | 531120.0 |
| SGD | 2 | 33.696880 | 0.99999999 | 336.969× | 673937.6 |
| GAM | 0 | 24.823072 | 0.99999998 | 248.231× | 496461.4 |
| GAM | 1 | 18.866997 | 1.00000003 | 188.670× | 377339.9 |
| GAM | 2 | 22.108400 | 1.00000001 | 221.084× | 442168.0 |

相同raw λ不使SGD/GAM起点的保护总尺度相同；两者保留各自Fisher trace和方向结构。因此跨起点比较含起点状态及Fisher变化，同起点有/无EWC配对才是EWC增量效应对照。

## 真实更新、评估和数值核验

12个正式checkpoint共逐个检查350张量均有限；旧head和冻结骨干不变，adapter key集合与父模型一致。所有4800次base optimizer更新都有finite-parameter、finite-loss/gradient与single-P记录；12条第二步保护项和保护梯度均正。3个seed内四分支的新head完全一致，50个训练遍历加1个末尾Fisher遍历的全部样本顺序及首个增强batch hash完全一致。该配对证明涵盖每次遍历的首batch像素，不声称已比较每一个增强batch。

从实际保存的A/B以float64重算端点间有效位移，与现有报告d_QV_cont的最大误差<1.2e−10。训练末期checkpoint完整重载、CNN matrix与COMPLETE结果一致。

| 分支 | 更新数/seed | train mean P/CE | mean norm(P)/norm(clean CE) | Task2新数据 batch Fisher trace（mean±sample SD） |
|---|---:|---:|---:|
| ewclora | 400 | 1.221% | 5.174% | 28.031±1.406 |
| fs_ewclora | 400 | 2.056% | 6.161% | 18.624±2.040 |
| task1_shaping | 400 | 0.786% | 3.492% | 23.783±1.346 |
| task2_shaping | 400 | 4.394% | 10.092% | 19.790±1.708 |

P/CE为每seed训练全400步均值之比再跨seed平均；梯度比来自zero-based step0、step1、step8、16…共51个采样点，分母含所有可训练参数（含新分类头）的clean CE梯度；GAM更新方向范数另有记录，不能把该比例解释成P/GAM方向比例。

上述Task2 Fisher列是新任务数据端点proxy。保存的fisher_state_1.pt则为0.9F1+F2累计矩阵，逐条验证其trace与此公式相符、误差<1e−6。该末尾累计发生在Task2训练后，gamma=0.9不把Task2训练时保护F1先缩成0.9F1。旧任务逐样本raw Fisher在本次12续训并未重测，旧批相关值不能沿用。

## 解释边界

- λ20000是用户统一指定值，不是此前扫描证明的三seed最优值或达到预设≥1pp保护门槛的值。
- 数值与梯度日志证明EWC确实参与更新；位移缩小只证明约束更强，旧任务CIL准确率是否改善仍必须看同起点同未来优化器的λ0配对。
- 原生训练只拟合新20类；旧head被冻结，但表示可移动，新旧head在40类CIL中竞争。旧20类局部准确率与40类CIL准确率应分开解读。
- 三seed为当前协议下描述性证据；不能由均值相差小或SD重叠直接声称显著性、等价性或稳定优越。
- 改了λ和Fisher归一化两个设置，旧λ2000表与新λ20000表的差异不能仅归因于λ提高10倍。

机器可读全部逐trial证据见 [PROTOCOL_AUDIT.json](PROTOCOL_AUDIT.json)。

代码定位：原生P `execution/source/models_LoRAbasedCL/ewclora_youyue_fitArchitechture.py:270`；原生batch Fisher `.../as1_normfisher_gam.py:382`；关闭归一化路径 `.../as1_normfisher_gam_intask.py:721`；GAM组合 `.../as1_normfisher_gam_intask.py:1018` 与`:1173`；运行时只读单次注入断言 `train.py:665`；旧head断言 `train.py:997`。
