# FS-EWC 正文定义与历史实验命名核对（2026-09-20）

结论：用户段落描述的是当前正文正式 FS-EWC：平方梯度敏感度整形、同任务晚期 checkpoint 等权平均、在保留的平均 anchor 上估计当前任务 Fisher 并加阻尼，再与历史重要度累加。当前 fine-grained 表的 GAM-EWCLoRA 是另一历史实现；不能将其 GAM 参数改名填入正文的 `r, δ, β, K_t`。论文中的旧 `FS-LoRA` / `FS-EWCLoRA` 字面名称不能证明实现一致。

只读检查以下文件；未修改论文、配置或实验：

- `P=/data/140-1/users/liying/CopyFlatCV/GAM-EWC/paper`
- `P/iclr2027_liying.tex`（主论文，正式方法定义）
- `P/6_experiments_appendix.tex`（实验附录草稿，区分源结果与待核实配置）
- `P/6_experiments_reorganized.tex`（实验重组草稿，尚有导入占位符）
- `/data/140-1/users/liying/O-LoRA/figures/lora_taskwise_20260918/aa_times/sources.json`，及其选中的四个 GAM+Merge seed-2 原日志开头配置。

## 1. 正文符号的确切含义

| 符号 | 当前正文定义 | 正文证据 |
|---|---|---|
| `r` | 参数邻域半径；`z = θ + r ĥ(θ)/(||ĥ(θ)||+ξ)` | `iclr2027_liying.tex:1481–1493` |
| `δ` | 有限差分 probe 的尺度；`s(u)=δ/(||g(u)||+ξ)`，`ĥ(u)=2/s(u) [g(u+s(u)g(u))−g(u)]` | `:1463–1477` |
| `β` | 当前 Fisher 重要度的加性各向同性阻尼，`F̂_t=Diag(F_emp(θ_t))+βI`，`β>0` | `:1376–1390` |
| `K_t` | 同任务、同参数化的选定晚期 checkpoint 索引集，`θ_t=(1/K)Σ θ_{t,k}`，等权平均 | `:1346–1359` |
| `γ` | 历史 Fisher 衰减，`Λ_t=γΛ_{t−1}+F̂_t`；不是 GAM 的 `gam_grad_gamma` | `:1393–1404` |

该正文用平方的 batch 聚合梯度作为 surrogate（`:1439–1450`），明确它不等同于逐样本 empirical-Fisher trace。两次差分估计共四次当前任务 loss-gradient；同 batch、同随机 forward realization，EWC 历史惩罚在敏感度搜索之外（`:1497–1506`）。Fisher 在平均以后、保留 anchor 上计算（`:1402–1405`、`:1514–1518`、`:1544–1549`）。

正文附录还明确：`ĥ` 包含完整的 `2/s` 因子；存在加性 stabilizer 时，直接归一化未缩放梯度差不是相同实现，除非 stabilizer 同时重标（`:3541–3545`）。因此，不能仅凭“两种代码都有四次梯度”和“都有两个 rho”建立数值映射。

**不能直接断言 `gam_grad_rho=δ`、`gam_grad_norm_rho=r` 或 `gam_grad_gamma=β`。** 前两者只能在逐项核对实际更新公式、归一化、EWC 放置、stabilizer 及梯度混合规则后确定对应关系；第三个明确不应被当作 Fisher 阻尼。配置中的 `ewc_fisher_damping_alpha`、`path_beta` 也不能仅凭名称认定等于正文 `β`。

## 2. 三份论文文件能否提供每个 benchmark 的正式数值

不能。这些文件给出符号定义与实验记录要求，没有查到正式 FS-EWC 按 benchmark 给出的 `r, δ, β` 数值表，也没有正式默认 `K_t` 与每个主结果 run 的绑定。

- `6_experiments_appendix.tex:29–39` 要求逐 run 记录代码/配置版本、split、optimizer，以及 `α,r,δ,ξ,λ,γ`、阻尼规则和 `K_t`；明确不能用其他 setting 的值填补缺项。
- `6_experiments_reorganized.tex:87–93` 要求恢复 Fisher 估计、阻尼、checkpoint window 等，并禁止把某个历史超参数推广到所有实验。
- fine-grained 匹配比较表仍为 `ExpImport`（`:179–205`）；caption 的同配置叙述是目标协议，不能反过来证明历史表已经匹配。
- 附录确有 checkpoint-policy ablation：`{50}`、`{10,20,30,40,50}`、`{5,10,...,45,50}`、`{30,40,50}`、`{40,45,50}`（`6_experiments_appendix.tex:309–322`）。主论文说明这是 ImageNet-A 的一个 checkpoint ablation（`iclr2027_liying.tex:2397–2401`），不是五个 fine-grained dataset 的默认集合。附录也声明 configuration identity / repetitions 待核实，full-path 不应写成 late-stage（`:300–311`）。

## 3. 与当前 fine-grained GAM-EWCLoRA 行的关系

附录保留的旧 `FS-LoRA (source name)` 行为 FAA `74.224 / 51.741 / 46.660 / 86.747 / 78.824`（CUB/Cars/Aircraft/Flowers/Pets；`6_experiments_appendix.tex:155`），与当前表 GAM-EWCLoRA 的四舍五入值相符。但该表 caption 明确说这些历史单 seed Track-1 数值**不能建立 final FS-EWC 值或 EWC-only baseline**（`:143–146`）；注释同样要求先核对方法/配置映射（`:138–140`）。

父线程原始日志/源码核对确认该行选中 `model_name=ewclora_normfisher_gam`，没有 checkpoint averaging；这个已确认差异足以排除“与当前正文完整 FS-EWC 相同”。本文文件本身没有绑定某个代码类为正式算法的已验证实现。Flowers 的 GAM probe 参数未记录，不能借其他 benchmark 或当前代码默认值补齐。

重组实验稿对历史 GAM/AS 数值也明确保留身份：`:261–270` 说明它们不能直接认证成 squared-gradient FS-EWC，且该诊断保留 terminal anchor。

## 4. 旧四 benchmark 主表的 `FS-EWCLoRA` 是另一个历史实现线

`aa_times/sources.json` 将旧论文 `paper_method=FS-EWCLoRA` 映射为实际描述标签 `method=GAM-EWC+Merge`，并限定为 historical implementation lineage（而非数学等价认证）。选中的原日志记录 `model_name=ewclora_normfisher_gam_intask_damp`。每个 dataset 抽查一个 included seed-2 run：

| 数据集 | `gam_grad_rho` | `gam_grad_norm_rho` | `gam_grad_gamma` | `ewc_fisher_damping_alpha` | checkpoint interval | `ewc_lambda` / `ewc_gamma` |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-100 | .05 | .1 | .1 | .2 | 5 | 2000 / .9 |
| ImageNet-R | .05 | .1 | .1 | .2 | 20 | 2000 / .9 |
| ImageNet-A | .05 | .2 | .1 | .2 | 20 | 2000 / .9 |
| DomainNet | .05 | .1 | .1 | .2 | 5 | 6000 / .9 |

这些是原字段值，不是正文符号重新命名。该表仅用于说明另一实现线和跨数据集差异；不替代逐 seed 全量核对。

ImageNet-A seed 2 的原 log（路径可从 `sources.json` 的 `dataset=imageneta, method=GAM-EWC+Merge, seed=2` 唯一取得）：

`/data/140-1/users/liying/CopyFlatCV/outputs_logs/step_ablation/logs_inc_lora/ewclora_normfisher_gam_intask_damp/gam_sgd_initlr004_lr002_e50_rho005_f100_intask20_wmerge_omega030_batchF_alpha02_sr00_wpathbeta30/imageneta/2/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2/exp_run/20/s3_damp_c3_alpha02_int20_imageneta_t10c20_r10_sgd_s2_vit_base_patch16_224.log`

- `:16–36`：model、λ=2000、ewc_gamma=.9、interval20、damping_alpha=.2、weight_merge=True。
- `:51–53`：GAM gamma=.1、rho=.05、norm_rho=.2。
- `:87`：`weight_merge=True`, `omega=0.3`。
- `:88`：`path_weighted=True`, `path_start_ratio=0`, `path_beta=3`。
- `:89`：实际日志类 `as1_normfisher_gam_intask_damp_weightmergethenfisher.py`，`fisher_snapshots=False`, `fisher_source=post_weight_merge`。

这条实现确有“先合并权重再 Fisher”的日志证据，但 path weighting / start ratio 0 与正文等权晚期平均尚不能视为相同规则；加上 GAM 与当前 `2/s` 平方梯度算法仍需逐式匹配，它不能自动成为正式 FS-EWC 的完整实现证明。`path_beta=3` 是另一命名空间，也不是正文 Fisher `+βI` 的数值。

建议在回答/表格中保留三个层次：当前 fine-grained **GAM-EWCLoRA（无平均）**；旧四 benchmark **GAM-EWC+Merge（含合并，历史 FS 别名）**；正文 **FS-EWC 数学定义（逐 benchmark 数值与代码映射尚待绑定）**。不修改论文或已选结果。
