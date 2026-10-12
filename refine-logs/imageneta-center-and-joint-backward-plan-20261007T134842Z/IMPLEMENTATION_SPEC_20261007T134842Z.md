# 独立v3执行器修改规格（尚未实施）

此文件是可实现的代码修改方案和伪代码，不是已运行的训练代码。禁止改动正在训练的严格执行目录；以后实施时复制其数据/模型生命周期和事务恢复基础设施到新目录，并重新固定源码pins。

## 显式字段

| 字段 | 取值 |
|---|---|
| first_location_estimator | forward / backward_corrected |
| second_residual_estimator | forward / backward_corrected / centered |
| residual_coefficient | +1 / −1；本轮仅centered支持−1 |
| update_base | g0（本轮固定） |
| gam_grad_rho / gam_grad_norm_rho | .05 / 1.0 |

不要同时保留含糊的旧first_probe_sign、branch、second_difference_sign和新估计器配置；配置解析器应拒绝冲突或派生并核验所有旧别名。原始gamma/beta字段不控制新显式U，日志必须写出实际活跃规则，不能仅根据model_name判断GAM执行。

## 最小compute流程伪代码

```python
state = capture_full_state(model)   # parameters, buffers, gradients, modes, caches, R0
base, R0 = state.base, state.rng
try:
    # eval_grad每次先恢复state和R0，再设置指定位置；CE保持普通正号
    g0, loss0, R1 = eval_grad(base, same_inputs, same_targets, R0)
    v0 = detach_normalize(g0)
    if first_location_estimator == 'forward':
        g1 = eval_grad(base + rho*v0, same_inputs, same_targets, R0).grad
        q = g1 - g0
    else:
        g1 = eval_grad(base - rho*v0, same_inputs, same_targets, R0).grad
        q = g0 - g1
    outer = base + radius*detach_normalize(q)
    g2 = eval_grad(outer, same_inputs, same_targets, R0).grad
    v2 = detach_normalize(g2)
    if second_residual_estimator == 'forward':
        g3plus = eval_grad(outer + rho*v2, same_inputs, same_targets, R0).grad
        residual = g3plus - g2
    elif second_residual_estimator == 'backward_corrected':
        g3minus = eval_grad(outer - rho*v2, same_inputs, same_targets, R0).grad
        residual = g2 - g3minus
    else:
        g3plus = eval_grad(outer + rho*v2, same_inputs, same_targets, R0).grad
        g3minus = eval_grad(outer - rho*v2, same_inputs, same_targets, R0).grad
        residual = 0.5*(g3plus - g3minus)
    U = detach(g0 + residual_coefficient*residual)
    state.restore()
    assert_full_state_restored(state)
    set_rng(R1)                      # 每个训练batch只推进一次前向随机状态
    install_gradients(U)
    optimizer.step()                 # momentum=WD=0，maximize=False
    assert_actual_params_equal(base - current_lr*U)
except BaseException:
    state.restore()
    raise
```

所有探针point由detach张量构造，autograd.grad(create_graph=False,retain_graph=False)。禁止对q、v0、v2求链式梯度。中心组g3−从outer减位移，不是从g3+位置连续移动。

## 当前必须修改的位置

以[冻结direction_step.py](../imageneta-second-probe-sign-deterministic-20261007T071400Z/trainer/direction_step.py)为源：

- 47–62的validate_factors与78附近的branch校验：目前第一探针只允许+1；新配置必须严格派生定位侧别。
- 174–193的compute：新增后向定位、中心双侧读取、由估计器派生4/5次普通梯度预算与U公式。
- 记录中的first_gradient_label、points/grads键、g3正负调用数、公式标签需要区分实际侧别，不把g1−记成g1+。
- 232–298的diagnose_sides：中心组已在训练计算两侧，直接从context计算s2/e2，新增梯度0；BB不测上升侧。当前“selected-side复用”标签不能用于没有单一selected-side的中心组。
- apply_update的实际SGD/完整基点核验保留；异常时恢复不能只退参数而遗漏其他状态。

以[冻结four_probe_train.py](../imageneta-second-probe-sign-deterministic-20261007T071400Z/trainer/four_probe_train.py)为源：

- resolve_config当前拒绝非P与非单侧方向；加入严格v3配置模式，不用旧默认值自动回退。
- 实际forward hook和训练/诊断计数多处写死4/1，改成由同一expected_budget函数派生。
- 更新计数、Task0首步配对张量、final_summary/receipt/validator字段同时升级；完整训练中心12750CE、单侧10200CE。
- 配对门区分F-family和B-family共同点；跨family只比较共同基点g0，不能要求theta2相同。
- 新validator不能沿用validate_job.py写死4的验收条件；不能为了通过把第五梯度藏进诊断计数。

## 接口级验收

1. 非线性光滑CPU损失：独立解析g，验证四个新U，检查e2非零、center±中的1/2及符号。只用二次损失会漏检e2删除错误。
2. 二次损失：qF=qB，同外点；FF/FB/BF/BB和Center+一致，e2=0；Center−保留相反修正。允许预先固定的浮点误差预算，不放宽失败阈值。
3. rho=0退化：末端修正0、U=g0；零梯度及eps不产生NaN。此项为执行器单元门禁，正式配置仍固定rho=.05。
4. 完整状态/RNG与异常恢复：包括buffer、缓存、模型mode、已有grad；最终R1只来自g0。新增正负读取不改变数据/增强随机序列。
5. FF/FB回归：同状态新旧点、梯度、U与实际SGD、RNG一致后才复用两条历史严格成绩。
6. 四组真实模型短测：2tasks×1epoch×1batch；中心10CE/2SGD、BF/BB8CE/2SGD；FC旧行继承、禁用计数0、BB正向局部梯度0。
7. 首步配对只断言共同量相同；训练后不同U导致不同轨迹是预期，不能要求不同方法全程梯度相同。

CPU测试不是模型性能证据；短测分数不是完整AAA/FAA。本次仅输出修改规格，以上验收尚未运行。
