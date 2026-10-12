# SGD与三种响应方法：实际项目实现

依据USER_METHOD_SPEC.txt重新实现并接入冻结ImageNet-A/LoRA工程。附件中的sandbox下载链接不作为当前机器已有代码；这里提供的是本项目真实可调用文件，使用Pilot_new_local的PyTorch 2.4.1。

## 统一定义

可训练参数集合是全部requires_grad=True的LoRA Q/V因子和当前分类头。范数在这个完整集合上计算。当前批次固定一次输入增强，探针重放同一个R0，训练RNG只前进到基点梯度后的R1。

\[
g_0=\nabla L_B(\theta),\quad h=\eta\|g_0\|,\quad
g_1=\nabla L_B(\theta-\eta g_0),\quad a=g_0-g_1.
\]

这里的g1是下降探针，与历史正向SAM/GAM实验的g1不是同一读取点。第一版外点半径固定为动态r=h，没有固定r=1或rho=.05探测。

| 方法/代码名称 | 正常新梯度 | 最终提交d | 缓存向量 |
|---|---:|---|---|
| 普通SGD / sgd | 1 | g0 | 无 |
| 当前响应重定位 / current_g2 | 3 | g2=grad L(theta+h*N(a)) | 无 |
| 历史响应直接融合 / history_fusion | 2 | g0+c | 上一步原始g0，source=base |
| 最终更新历史辅助重定位 / history_g2 | 3 | g2=grad L(theta+h*N(c)) | 上一步完整d，source=final |

两个历史方法均独立保存实际位移s=theta_current−theta_previous，ell=norm(s)。有效历史时：

\[
c=(1-w)(g_0-g_1)+w\frac{h}{\ell}(p-g_0),\qquad w=.5.
\]

history_fusion中的p是上一基点g0；history_g2中的p是上一真实提交向量d，正常为g2。没有历史时c=a，所以方法二回退到2g0−g1，方法三回退到方法一。

外点始终从原基点theta出发；最终g2不再加c或历史向量。缓存d保留完整幅值，不缓存N(d)、响应差分或带学习率/负号的位移。真实FP32位移从提交后的参数相减读得，独立保存。

## 实际核心分支（对应response_step.py）

```python
if method == "sgd":
    final = g0
else:
    h = lr * global_norm(g0)
    g1 = read_at_base_offset(-lr * g0, same_batch_and_R0=True)
    response = g0 - g1
    if method in ("history_fusion", "history_g2") and history_valid:
        response = 0.5 * response + 0.5 * h / ell * (cached_vector - g0)
    if method == "history_fusion":
        final = g0 + response
    else:
        g2 = read_at_base_offset(h * global_normalize(response))
        final = g2

restore_base_state_and_keep_base_R1()
optimizer.step_once_with(final)
actual_step = submitted_parameters - base_parameters
if method == "history_fusion":
    cache(g0, actual_step, source="base")
elif method == "history_g2":
    cache(final, actual_step, source="final")
else:
    history = None
```

上面为非退化数学分支。实现中仅两个历史方法写入缓存，两个无历史方法始终无缓存；所有缓存detach().clone()。

## 可直接调用的PyTorch接口

当前任务的类别边界为(lo,hi)，由TaskLayout.class_range(task)提供。

```python
import torch
from response_step import ResponseStep

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.04, momentum=0, weight_decay=0)
driver = ResponseStep(model, params, geometry=None, config={
    "method": "history_g2", "history_weight": 0.5,
    "radius_mode": "sgd_step", "radius_scale": 1.0,
    "eps": 1e-12, "backend": "native",
})

def closure(images, labels):
    logits = model(images)["logits"]
    loss = torch.nn.functional.cross_entropy(logits[:, lo:hi], labels - lo)
    return loss, logits

result = driver.compute(images, labels, closure, optimizer.param_groups[0]["lr"])
record = driver.apply_update(optimizer.param_groups[0]["lr"], optimizer)
```

CLI中的response_train.py安装同一接口到原生learner，不在驱动器之后再调用backward或第二次SGD。普通SGD真实只读一次梯度，不执行虚拟探针来补计算预算。

## 数值、任务与checkpoint约定

- h过小或预测点在FP32不可辨：提交g0，记录减少的读取次数。
- 历史真实位移无效：忽略历史；融合响应零时先尝试当前响应a，再为零则提交g0。
- 外点在FP32不可辨：显式提交g0。history_g2仍缓存这次实际提交的完整g0，source保持final。
- 虚拟探针均恢复原基点参数、grads、buffers、模式、LoRA缓存及随机流；沿用上一版完整事务政策。冻结参数版本改变直接报错。
- 实际更新仅为普通torch.optim.SGD；momentum/WD0，无裁剪、最终投影、QR、HVP、范数匹配、Fisher/EWC、replay和权重融合。
- 每个task/head/layout变化重建驱动器并清空历史；普通epoch边界保留历史。cosine调度器每epoch前进一次，不随探针推进。
- checkpoint同时保存方法、source=base/final、全幅值缓存、实际位移、参数布局、哈希、步数及外层optimizer/scheduler/RNG；加载不同方法或source立即拒绝。

训练器每步额外做一次同批次/R0无梯度post-update loss诊断，并恢复提交后状态。正常梯度/前向/SGD预算分别为SGD 1/2/1、current_g2 3/4/1、history_fusion 2/3/1、history_g2 3/4/1。单任务每epoch测试也属于被动评估，不参与更新或best-epoch选择。

## 机制解释范围

历史跨minibatch，差分包含样本/增强变化。方法三的缓存g2在上一虚拟外点读取，即使固定损失，也不能把k_previous−g0称为两个真实基点之间的梯度割线。本轮测试的是完整更新信号作为定位信息是否有用，不声称精确Hessian、逆曲率、无条件下降或新方法已优于SGD。

路径：核心trainer/response_step.py；接入trainer/response_train.py；划分/指标trainer/layout_lifecycle.py；原生源文件保持不变。
