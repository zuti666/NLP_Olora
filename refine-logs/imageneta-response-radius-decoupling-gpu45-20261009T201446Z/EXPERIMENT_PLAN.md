# Current-G2 探测半径解耦：GPU 4、5

用户已授权仅使用物理 GPU 4、5；空闲调度，最多同时两项。四项短测（2任务×1epoch×2batch）及配对验收通过后，运行三项完整训练。

设 `g0 = grad L(theta)`，`h = eta*norm(g0)`。内点为 `theta - q*N(g0)`；`g1` 在该点读取。响应 `a=g0-g1`。外点重新从 BASE 构造 `theta + R*N(a)`，读取 `g2`，恢复 BASE 后仅提交 `theta <- theta - eta*g2`。所有范数为完整可训练 LoRA+分类器参数的全局L2；不对最终更新归一化。

| Method | Inner q | Outer R | Training |
|---|---|---|---|
| CurrentG2-Original | eta*norm(g0) | eta*norm(g0) | 已完成的原版结果复用；新跑短测核验 |
| CurrentG2-FixedInner | 0.05 | eta*norm(g0) | 完整10任务 |
| CurrentG2-FixedOuter | eta*norm(g0) | 0.05 | 完整10任务 |
| CurrentG2-FixedBoth | 0.05 | 0.05 | 完整10任务 |

公共配置：ImageNet-A固定5981/1519训练/测试、类别顺序0…199、10×20；seed2，ViT-B/16 augreg2_in21k_ft_in1k，LoRA rank10；每任务50epochs、batch128；SGD momentum/weight_decay=0，首任务LR .04、后续 .02，epoch cosine T=50、min ratio=.1；FP32/native/严格确定性，无AMP/TF32。

无历史缓存、EWC、Fisher、权重融合、梯度投影、HVP、replay、梯度裁剪或最终范数匹配。每正常更新3次训练CE梯度、1次真实SGD，另1次被动更新后CE诊断。若零响应或FP32点不可分辨，按明确fallback记录，不伪报3次。

每组完整2550更新、299050样本暴露。AAA/FAA为任务等权，BWT为最终减各旧任务刚学完准确率的平均，符号负值表示遗忘。短测准确率不能作为完整结果。除AAA/FAA/BWT/LA/遗忘外，记录每步LR、h参考、实际/理论内外距离、响应相对范数、g2-g0相对范数、实际参数/有效LoRA更新长度、CE变化、RNG/状态恢复、禁止工作计数。

原版复用源与107项哈希在 gates/ORIGINAL_RESULT_REUSE.json。已有AAA62.870、FAA45.252、BWT−9.652，仅seed2。三组新训练完成后逐组与原版配对比较；这是半径解耦验证，并非无学习率依赖、最优半径或Hessian机制的直接证明。历史三方法保持原定义。

代码：trainer/independent_response_step.py改变虚拟探针；trainer/response_train.py明确路由三变体；trainer/response_step.py保持原版逐字节相同，继承原生SGD提交逻辑。完整命令、输入SHA及依赖均列在suite.json。GPU4/5若忙，等待，不终止外部进程。
