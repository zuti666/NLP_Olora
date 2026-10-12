# 历史响应融合算法流程图

图中实线表示训练主流程，虚线表示不会改变训练轨迹的更新后损失诊断。正常可分辨步骤新增两次梯度计算；零或不可分辨预测直接回退到原始梯度。每次任务切换或参数集合变化清空历史，同任务跨批次保留缓存。未来探测使用同一个当前batch和R0；历史梯度直接复用上一batch的原始梯度，不重新计算，因此历史响应可能混有样本和随机实现变化。

```mermaid
%%{init: {"theme": "base", "forceLegacyMathML": true, "fontFamily": "DejaVu Sans, sans-serif", "themeVariables": {"fontFamily": "DejaVu Sans, sans-serif", "fontSize": "16px", "primaryTextColor": "#142B43", "lineColor": "#334155", "background": "#FFFFFF", "edgeLabelBackground": "#FFFFFF"}, "flowchart": {"htmlLabels": true, "curve": "linear", "nodeSpacing": 35, "rankSpacing": 38}}}%%
flowchart TD
    task_reset["Task / parameter-set change<br/>Clear the history cache"]
    current["$$\begin{gathered}\text{Current batch; save base state and R0}\\\\g_t=\nabla L_{B_t}(\theta_t)\\\\\text{Gradient read 1}\end{gathered}$$"]
    prediction["$$\begin{gathered}\text{Construct the virtual SGD prediction}\\\\\theta_t^f=\theta_t-\eta_tg_t\\\\\ell_{f,t}=\eta_t\lVert g_t\rVert\end{gathered}$$"]
    valid_future{"Future displacement<br/>nonzero and resolvable?"}
    plain["$$\begin{gathered}\text{Numerical fallback}\\\\U_t=g_t\end{gathered}$$"]
    future["$$\begin{gathered}\text{Same batch and R0}\\\\g_t^f=\nabla L_{B_t}(\theta_t^f)\\\\\text{Gradient read 2; restore full base state}\end{gathered}$$"]
    valid_history{"Compatible history and<br/>nonzero actual prior displacement?"}
    mirror["$$\begin{gathered}\text{No-history mirror}\\\\U_t=2g_t-g_t^f\end{gathered}$$"]
    fusion["$$\begin{gathered}\text{Fuse raw responses; } w=1/2\\\\h_t^f=(g_t-g_t^f)/\ell_{f,t}\\\\h_t^b=(g_{t-1}-g_t)/\lVert s_t\rVert\\\\U_t=g_t+\ell_{f,t}(h_t^f+h_t^b)/2\end{gathered}$$"]
    commit["$$\begin{gathered}\text{Commit once from the restored base}\\\\\theta_{t+1}=\theta_t-\eta_tU_t\\\\\text{One plain SGD update; RNG = R1 from } g_t\end{gathered}$$"]
    cache["$$\begin{gathered}\text{Cache detached raw gradient and actual step}\\\\g_{\mathrm{cache}}\leftarrow g_t\\\\s_{\mathrm{cache}}\leftarrow\theta_{t+1}-\theta_t\end{gathered}$$"]
    next_batch["Next batch<br/>in the same task"]
    diagnostic["Read-only post-update loss<br/>Same batch and R0<br/>No gradient<br/>Restore committed state and R1"]
    task_reset --> current
    current --> prediction
    prediction --> valid_future
    valid_future -- "No" --> plain
    valid_future -- "Yes" --> future
    future --> valid_history
    valid_history -- "No" --> mirror
    valid_history -- "Yes" --> fusion
    plain --> commit
    mirror --> commit
    fusion --> commit
    commit --> cache
    cache --> next_batch
    next_batch --> current
    commit -. "Passive diagnostic only" .-> diagnostic
    classDef input fill:#E9F4EE,stroke:#3D7A5B,color:#183C2B,stroke-width:1.5px;
    classDef probe fill:#EAF1FB,stroke:#4A6E9B,color:#142B43,stroke-width:1.5px;
    classDef decision fill:#F5F7FA,stroke:#687D91,color:#142B43,stroke-width:1.5px;
    classDef combine fill:#F1ECFA,stroke:#8064A1,color:#32214C,stroke-width:1.5px;
    classDef output fill:#FFF2E6,stroke:#B67B45,color:#53341C,stroke-width:1.5px;
    classDef side fill:#FAFAFA,stroke:#8D969F,color:#475569,stroke-width:1.3px,stroke-dasharray:5 3;
    class task_reset,current input;
    class prediction,future probe;
    class valid_future,valid_history decision;
    class fusion,mirror,plain combine;
    class commit,cache output;
    class diagnostic,next_batch side;
    linkStyle default stroke:#334155,stroke-width:1.8px;
```

图中 SGD 学习率沿用基线任务级日程，调度器仍每epoch前进一步；不在虚拟预测时调用优化器或调度器。算法没有固定的 GAM 内外半径，也没有QR、Hessian求逆、梯度分解或重定位扩展。
