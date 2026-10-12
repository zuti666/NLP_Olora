# 历史响应融合更新：算法、伪代码与实现约定

本次实现首版 History Response Fusion（HRF）。只使用历史原始梯度缓存、一次虚拟SGD预测和响应直接融合；不加入QR、Hessian求逆、方向优化、投影、更新范数匹配或g2重定位。历史不是重放数据：缓存中不保存用于再训练的旧样本。

## 精确定义

令当前可训练参数为\(\theta_t\)，当前优化器学习率为\(\eta_t\)，本批原始梯度为\(g_t\)。持久缓存只有上一步原始梯度\(g_{t-1}\)和实际提交位移\(s_t=\theta_t-\theta_{t-1}\)，以及有效性、布局和哈希元数据。

\[
g_t=\nabla L_{B_t}(\theta_t),\qquad
\theta_t^{\mathrm f}=\theta_t-\eta_tg_t,\qquad
g_t^{\mathrm f}=\nabla L_{B_t}(\theta_t^{\mathrm f}),\qquad
\ell_{f,t}=\eta_t\|g_t\|.
\]

两个梯度使用同一个已实现的输入批次和同一个前向随机状态。未来点只是探针，读出后完整恢复基点。

有效历史下\(\ell_{b,t}=\|s_t\|\)，采用\(w=.5,\lambda_t=\ell_{f,t}\)：

\[
U_t=g_t+\tfrac12(g_t-g_t^{\mathrm f})+
\frac{\ell_{f,t}}{2\ell_{b,t}}(g_{t-1}-g_t).
\]

无历史时（含各任务第一步）：\(U_t=2g_t-g_t^{\mathrm f}\)。无历史对照在所有步骤均使用这一式。正常步骤均需两次新梯度；缓存历史不重新计算梯度。

正式仅提交一次普通SGD：\(\theta_{t+1}=\theta_t-\eta_tU_t\)。缓存\(g_t\)而不是\(U_t\)或\(g_t^{\mathrm f}\)；缓存实际读得的\(\theta_{t+1}-\theta_t\)而不是预测位移。FP32真实位移与理论\(-\eta_tU_t\)可能存在舍入差异。

如果\(\ell_f\le10^{-12}\)或预测参数在FP32下完全不可辨，显式回退到\(U=g\)，记录原因及一次梯度预算。历史位移无效则仅关闭本步历史项，不做除零。每次任务切换、头扩张或可训练布局变化清空历史；同任务跨epoch保持缓存。

## 伪代码

```text
cache_gradient = None
cache_actual_step = None

for task in tasks:
    expand_classifier_and_reset_history()
    for epoch in range(50):
        for batch in paired_epoch_loader:
            base_state = snapshot_parameters_buffers_caches_modes_grads_and_rng()
            eta = optimizer.current_learning_rate
            g, loss_before = gradient(batch, base_state.parameters, base_state.R0)
            R1 = RNG_after_the_base_gradient
            ell_f = eta * global_norm(g)
            theta_future = FP32_SGD_prediction(base_state.parameters, g, eta)

            if ell_f <= eps or theta_future_equals_base_bitwise:
                U = g
                record_explicit_degenerate_predictor_fallback()
            else:
                g_future = gradient(batch, theta_future, same_R0)
                if history_enabled and cache_is_valid:
                    ell_b = global_norm(cache_actual_step)
                    U = g + 0.5 * (g - g_future)
                          + ell_f / (2 * ell_b) * (cache_gradient - g)
                else:
                    U = 2 * g - g_future

            restore_complete_base_state()
            set_training_RNG_to_R1()
            optimizer.step_once(U, eta)
            actual_step = submitted_parameters - base_state.parameters

            # Passive observation; no gradients or training-state changes.
            loss_after = forward_at_submitted_parameters(batch, same_R0)
            restore_submitted_state_and_R1()

            cache_gradient = detached_copy(g)
            cache_actual_step = detached_copy(actual_step)
            record_update_and_cache_diagnostics()

        scheduler.step_once()  # baseline cosine cadence is per epoch
    evaluate_all_seen_tasks_and_save_checkpoint_with_cache()
```

## 与此前配置的关系

模型、数据、增强、SGD学习率与任务协议沿用ImageNet-A基准。旧GAM的\(r=1,\rho=.05\)在本算法中不参与计算；动态未来距离为\(\eta_t\|g_t\|\)。有效配置明确标记`gam_radii_active=false`，终点summary的实际`r/rho`为null，旧值单独记为参考。因此不能把固定半径下降镜像成绩当作本算法的既有对照。

没有有效历史时，与固定半径SAM的领先阶联系也不是相同配置。固定\(\lambda=\rho\)留作未来独立版本；本轮不混入。

## 机制边界

固定损失且路径足够局部时，\(h_f=(g-g_f)/\ell_f\)近似\(Hu\)，\(h_b=(g_{t-1}-g)/\ell_b\)近似\(Hv\)，其中\(u=g/\|g\|,v=-s/\|s\|\)。融合响应对应两个输入方向的混合，不能总说更准确地估计\(Hu\)。

实际ImageNet训练的缓存来自上一minibatch，因此历史差分还包含样本和增强变化。CPU反例使用两个Hessian均为零的线性损失，历史修正仍非零。首轮只能验证缓存信息是否改善训练、泛化和保持；不能仅凭该公式或正面准确率宣布Hessian机制、无条件下降或更平坦。

## 记录指标

每步记录：原始及预测梯度哈希、缓存梯度与真实位移哈希、\(\ell_f,\ell_b\)、长度比、历史有效性、\(g^TU\)、\(\|U\|/\|g\|\)、\(\cos(u,v)\)、两个响应余弦、修正相对幅度、正式更新后同批次真实loss变化、梯度/前向/SGD次数及任务/epoch/样本身份。

每任务报告FAA、AAA、BWT、遗忘和刚学完准确率LA；AAA为主展示指标。所有最终准确率均从原始任务矩阵等权重算。每任务checkpoint保存算法缓存与状态哈希。额外post-loss为一次无梯度前向：正常预算为2次梯度、3次前向、1次SGD；诊断费用单独披露，不参与构造更新。

核心实现：`trainer/history_step.py`；训练接入：`trainer/history_train.py`；数值检验：`trainer/test_history_step_cpu.py`。
