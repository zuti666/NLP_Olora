# Experiment Plan: IncLoRA & OLoRA Multi-Optimizer × Multi-Order

**Problem**: 在 4 任务持续学习 benchmark 上，flat-minima (SAM/GAM) 和噪声扰动 (RWP) 优化器应用于 IncLoRA 和 OLoRA 策略时，是否能稳定降低遗忘，且在不同任务顺序下保持一致？
**Method Thesis**: IncLoRA/OLoRA + SAM/GAM/RWP 在 Order1/2/3 上均优于对应的 AdamW baseline，learned-only 评估下平均精度和 backward transfer 均有一致提升。
**Date**: 2026-03-26

---

## ⚠️ 运行前已修复的问题

### 问题 1：所有 order2/order3 脚本中 `/disk0` 路径
**已修复**：通过 `sed -i` 批量将 `/disk0/users/liying` → `/data/140-0/users/liying` 处理以下文件：
- `scripts_order2/`: OLoRA.sh, OLoRA_SAM.sh, OLoRA_GAM.sh, OLoRA_RWP.sh, IncLoRA-SAM_E1_Le3.sh, IncLoRA-GAM_E1_Le3.sh
- `scripts_order3/`: IncLoRA_E1_Le3.sh, IncLoRA-SAM_E1_Le3.sh, IncLoRA-GAM_E1_Le3.sh, IncLoRA-RWP_E1_Le3.sh, OLoRA.sh, OLoRA_SAM.sh, OLoRA_GAM.sh, OLoRA_RWP.sh

### 问题 2：order1 IncLoRA-GAM 硬编码端口
**已修复**：将 `--master_port=29501/29502/29503/29504` 替换为 `--master_port=$port`（随机端口），避免与并行运行的 OLoRA 脚本冲突。

---

## 配置一致性检查结果

| 配置项 | Order1 | Order2 | Order3 | 一致？ |
|--------|--------|--------|--------|--------|
| IncLoRA-Base: lora_strategy | Inclora | Inclora | Inclora | ✓ |
| IncLoRA-Base: batch | 8 | 8 | 8 | ✓ |
| IncLoRA-Base: lr | 1e-3 | 1e-3 | 1e-3 | ✓ |
| IncLoRA-Base: output 含 redo1 | ✓ (redo1) | ✓ (redo1) | ✗ (无 redo1) | 命名差异，非错误 |
| IncLoRA-SAM: rho | 0.05 | 0.05 | 0.05 | ✓ |
| IncLoRA-SAM: output 含 redo1 | ✓ | ✓ | ✗ | 命名差异 |
| IncLoRA-GAM: output 含 -005 | ✗ (无-005) | ✓ (-GAM-005) | ✓ (-GAM-005) | order1 命名不同 |
| IncLoRA-RWP: noise_std | 0.05 | 0.05 | 0.05 | ✓ |
| OLoRA-Base: lora_strategy | olora | olora | olora | ✓ |
| OLoRA-Base: Ba | Ba8-1 | Ba8-1 | Ba8-1 | ✓ |
| OLoRA-SAM: Ba | Ba16-1 | Ba16-1 | Ba16-1 | ✓ |
| OLoRA-GAM: Ba | Ba16-1 | Ba16-1 | Ba16-1 | ✓ |
| OLoRA-RWP: Ba | Ba16-1 (O1) | Ba8-1 (O2/3) | Ba8-1 (O2/3) | order1 不同，保留原样 |

---

## 训练→评估路径对照表

### Order1

| 方法 | 训练输出路径 | 评估 model_path | 一致？ |
|------|-------------|----------------|--------|
| IncLoRA-Base | Order-1_8/IncLoRA_T5large_..._Ba16-1_E1_LRe3_redo1_seed-1993 | 同 | ✓ |
| IncLoRA-SAM | Order-1_8/IncLoRA-SAM-005_T5large_..._Ba16-1_E1_LRe3_redo1_seed-1993 | 同 | ✓ |
| IncLoRA-GAM | Order-1_8/IncLoRA-GAM_T5large_..._Ba16-1_E1_LRe3_seed-1993 | 同 | ✓ |
| IncLoRA-RWP | Order-1_8/IncLoRA-RWP_lora_Gauss_std0p05_..._Ba16-1_E1_LRe3_seed-1993 | 同 | ✓ |
| OLoRA-Base | Order-1_8/OLoRA_T5large_..._Ba8-1_E1_LRe3_seed-1993 | 同 | ✓ |
| OLoRA-SAM | Order-1_8/OLoRA-SAM-005_T5large_..._Ba16-1_E1_LRe3_seed-1993 | 同 | ✓ |
| OLoRA-GAM | Order-1_8/OLoRA-GAM-005_T5large_..._Ba16-1_E1_LRe3_seed-1993 | 同 | ✓ |
| OLoRA-RWP | Order-1_8/OLoRA-RWP_lora_Gauss_std0p05_..._Ba16-1_E1_LRe3_seed-1993 | 同 | ✓ |

### Order2

| 方法 | 训练输出路径 | 评估 model_path | 一致？ |
|------|-------------|----------------|--------|
| IncLoRA-Base | Order-2_8/IncLoRA_T5large_..._redo1_seed-1993 | 同 | ✓ |
| IncLoRA-SAM | Order-2_8/IncLoRA-SAM-005_..._redo1_seed-1993 | 同 | ✓ |
| IncLoRA-GAM | Order-2_8/IncLoRA-GAM-005_..._redo1_seed-1993 | 同 | ✓ |
| IncLoRA-RWP | Order-2_8/IncLoRA-RWP_lora_Gauss_std0p05_..._seed-1993 | 同 | ✓ |
| OLoRA-Base | Order-2_8/OLoRA_T5large_..._Ba8-1_seed-1993 | 同 | ✓ |
| OLoRA-SAM | Order-2_8/OLoRA-SAM-005_..._Ba16-1_seed-1993 | 同 | ✓ |
| OLoRA-GAM | Order-2_8/OLoRA-GAM-005_..._Ba16-1_seed-1993 | 同 | ✓ |
| OLoRA-RWP | Order-2_8/OLoRA-RWP_lora_Gauss_std0p05_..._Ba8-1_seed-1993 | 同 | ✓ |

### Order3（yahoo→amazon→agnews→dbpedia）

| 方法 | 训练输出路径（stage 顺序） | 一致？ |
|------|---------------------------|--------|
| IncLoRA-Base | Order-3_8/IncLoRA_T5large_..._seed-1993/{1-yahoo,2-amazon,3-agnews,4-dbpedia} | ✓ |
| IncLoRA-GAM | Order-3_8/IncLoRA-GAM-005_T5large_..._seed-1993 | ✓ |
| OLoRA-Base | Order-3_8/OLoRA_T5large_..._Ba8-1_seed-1993 | ✓ |
| OLoRA-RWP | Order-3_8/OLoRA-RWP_lora_Gauss_std0p05_..._Ba8-1_seed-1993 | ✓ |

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C1: 平滑/扰动优化器降低 IncLoRA 遗忘 | IncLoRA 是增量 LoRA 策略的代表，优化器收益展示泛化性 | SAM/GAM/RWP 中至少一个在 3 个 order 上平均精度均优于 Base | B1, B2 |
| C2: 平滑/扰动优化器降低 OLoRA 遗忘 | OLoRA 是正交 LoRA 策略，验证优化器效果跨策略一致性 | 与 C1 相同标准，独立验证 | B1, B3 |

---

## Paper Storyline

- **Main paper must prove**: B1（主结果表）、B2（IncLoRA 优化器对比）、B3（OLoRA 优化器对比）
- **Appendix can support**: B4（order robustness 热力图），B5（training loss 曲线）
- **Experiments intentionally cut**: 多 seed 交叉验证（受算力限制，单 seed=1993）

---

## Experiment Blocks

### Block 1: Main Result Table (IncLoRA + OLoRA, 3 Orders)
- **Claim tested**: C1 + C2（联合）
- **Why this block exists**: 主表，展示所有 8 个 variant × 3 orders 的 learned-only 评估结果
- **Dataset**: CL_Benchmark (dbpedia, amazon, yahoo, agnews)
- **Compared systems**: IncLoRA-{Base,SAM,GAM,RWP} × {Order1,2,3}; OLoRA-{Base,SAM,GAM,RWP} × {Order1,2,3}
- **Metrics**: 平均精度（Average Accuracy across learned tasks at final stage），Backward Transfer（BWT）
- **Setup details**: T5-large, lora_dim=8, 1 epoch/task, lr=1e-3, seed=1993, GPU: IncLoRA on 0,1 / OLoRA on 2,3（4 GPU 总计）
- **Success criterion**: 至少 1 个优化器 variant 在所有 3 orders 上均超过 Base；跨 order 标准差较 SeqLoRA 不大幅增加
- **Failure interpretation**: 若所有优化器 variant 均不超过 Base → 优化器对 IncLoRA/OLoRA 无效，需检查超参或策略适配性
- **Table / figure target**: 主表 Table 1（与 SeqLoRA 结果并列）
- **Priority**: MUST-RUN

### Block 2: IncLoRA Optimizer Ablation (Order1)
- **Claim tested**: C1
- **Why this block exists**: 聚焦 IncLoRA，展示 SAM/GAM/RWP 相对 Base 的逐 task-stage 精度曲线
- **Dataset**: Order1，4 stage checkpoints
- **Compared systems**: IncLoRA-{Base, SAM, GAM, RWP}
- **Metrics**: 每 stage 的 learned-only 精度矩阵（上三角矩阵），最终 stage 平均精度
- **Setup details**: 同 Block 1
- **Success criterion**: 优化器 variant 在 stage 3-4 出现明显分化（遗忘曲线变平）
- **Failure interpretation**: 若曲线重叠 → 优化器在 IncLoRA 上无效
- **Table / figure target**: Figure 2（精度曲线），Appendix Table（完整矩阵）
- **Priority**: MUST-RUN

### Block 3: OLoRA Optimizer Ablation (Order1)
- **Claim tested**: C2
- **Why this block exists**: 同 Block 2 但针对 OLoRA
- **Dataset / split**: Order1，4 stages
- **Compared systems**: OLoRA-{Base, SAM, GAM, RWP}
- **Metrics**: 同 Block 2
- **Setup details**: 同 Block 1
- **Success criterion**: 同 Block 2
- **Failure interpretation**: 同 Block 2
- **Table / figure target**: Figure 3 或与 Block 2 合并为双列图
- **Priority**: MUST-RUN

### Block 4: Order Robustness Check (IncLoRA + OLoRA)
- **Claim tested**: C1 + C2 的 order robustness
- **Why this block exists**: 验证优化器收益不是某个特定任务顺序的 artifact
- **Dataset**: Order1/2/3 结果（来自 Block 1）
- **Compared systems**: 每个 variant 在 3 orders 的标准差
- **Metrics**: 跨 order 精度标准差，最优 variant 排名稳定性
- **Setup details**: 复用 Block 1 结果，无需额外训练
- **Success criterion**: 最优 variant 在 3 orders 中均排名 top-2
- **Failure interpretation**: 若 best variant 在不同 order 下变换 → 优化器选择对顺序敏感，需报告该局限性
- **Table / figure target**: Appendix Table（3 × 8 结果热力图）
- **Priority**: NICE-TO-HAVE（无额外计算，后处理即可）

### Block 5: Training Efficiency Note
- **Claim tested**: 实用性（4 GPU 可行）
- **Why this block exists**: 说明 IncLoRA+OLoRA 并行 4 GPU 的 wall-time 相比 SeqLoRA（8 GPU）的效率
- **Dataset**: N/A（仅记录训练时间）
- **Compared systems**: SeqLoRA（8 GPU）vs IncLoRA+OLoRA（4 GPU × 并行）
- **Metrics**: 总 wall-time（小时）
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost（估算 GPU-h） | Risk |
|-----------|------|------|---------------|-------------------|------|
| M0: Sanity | 验证 1 个 IncLoRA+OLoRA 脚本在 Order1 第一个 stage 正常运行 | IncLoRA-Base(0,1) + OLoRA-Base(2,3), 只跑 1-dbpedia | 无 OOM，loss 下降，保存 adapter/ | ~2h | GPU 冲突 / path 错误 |
| M1: Order1 全量训练 | 8 个 variant 全部训练完 Order1 | 4 批次并行（见 run_all_inc-olora.sh） | adapter/ 目录均存在，stage 4 有输出 | ~32h | 显存 / 长时稳定性 |
| M2: Order1 评估 | 获得主结果 | run_evalFlat_order1_inc_gpu01.sh + run_evalFlat_order1_olora_gpu23.sh | eval log 无 error，结果可读 | ~4h | eval_Whole_olora.sh 路径 |
| M3: Order2/3 训练+评估 | 验证 order robustness | 同上，scripts_order2/3 | 评估结果存在 | ~64h | 同 M1 |
| M4: Result Analysis | 构建主表 + 图 | 后处理脚本 | 主表完整，无 NaN | <1h | 路径解析 |

---

## Compute and Data Budget

- **Total estimated GPU-hours**: ~100h（Order1×8×4h + Order2/3×8×4h × 2 + eval）
- **Data preparation**: 无需额外准备，使用 CL_Benchmark
- **Human evaluation**: 无
- **Biggest bottleneck**: Order2/3 的连续训练总时长（每个 order 约 32h wall-time，4 GPU）

---

## Execution Commands

```bash
# Step 1: Order1 训练 + 评估
cd /data/140-0/users/liying/O-LoRA
bash scripts_order1/run_all_inc-olora.sh

# Step 2: Order2（Order1 完成后）
bash scripts_order2/run_all_inc-olora.sh

# Step 3: Order3（Order2 完成后）
bash scripts_order3/run_all_inc-olora.sh
```

每个 run_all_inc-olora.sh 内部已按顺序：
1. SAM batch（GPU 0,1 + 2,3 并行）
2. GAM batch
3. RWP batch
4. Base batch
5. 评估（inc GPU 0,1 + olora GPU 2,3 并行）

---

## Risks and Mitigations

- **Risk**: OLoRA.sh 的 Ba8-1（batch_size=8）与同时运行的 IncLoRA（Ba16-1）速度不匹配，导致 wait 时 OLoRA 先完成但 IncLoRA 仍在运行
  - **Mitigation**: `wait` 等待最慢的脚本，不影响正确性，只影响墙时效率

- **Risk**: IncLoRA-GAM order1 的输出名为 `IncLoRA-GAM_...`（无 -005），而 order2/3 为 `IncLoRA-GAM-005_...`；eval 脚本已分别对应
  - **Mitigation**: 已在各 order 的 eval 脚本中使用对应的 model_path ✓

- **Risk**: Order1 OLoRA-RWP 使用 `Ba16-1` 命名，order2/3 使用 `Ba8-1`；eval 脚本已分别对应
  - **Mitigation**: 已在各 order 的 eval 脚本中使用正确名称 ✓

- **Risk**: `eval_Whole_olora.sh` 的 `lora_strategy` 参数：IncLoRA 需传 `Inclora`（大写 I），OLoRA 传 `olora`（小写）
  - **Mitigation**: 已在所有 eval 脚本中正确设置 ✓

---

## Final Checklist

- [x] 所有 `/disk0` 路径已修复为 `/data/140-0`
- [x] order1 IncLoRA-GAM 硬编码端口已修复为随机端口
- [x] 训练→评估路径一致性已逐一核验（8 variant × 3 order）
- [x] 所有 `run_all_inc-olora.sh` 已创建（order1/2/3）
- [x] 所有 eval 脚本已创建（6 个：inc_gpu01 + olora_gpu23 × 3 orders）
- [x] GPU 分配：IncLoRA on 0,1 / OLoRA on 2,3（无冲突）
- [x] Must-run blocks 与 nice-to-have 已分离
- [ ] Sanity check（M0）通过
- [ ] Order1 主结果完成
- [ ] Order2/3 结果完成
