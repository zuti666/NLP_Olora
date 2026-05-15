# Experiment Plan: SeqLoRA Multi-Optimizer × Multi-Order

**Problem**: 在 4 任务持续学习 benchmark 上，flat-minima (SAM/GAM) 和噪声扰动 (RWP) 优化器是否能稳定地降低遗忘，且这一效果在不同任务顺序下保持一致？
**Method Thesis**: SeqLoRA + SAM/GAM/RWP 在 Order1/2/3 上均优于 AdamW baseline（SeqLoRA-Base），在 backward transfer 和平均精度上取得一致提升。
**Date**: 2026-03-26

---

## ⚠️ 运行前必须修复的问题

### 问题 1：run_all_seqlora.sh 中 RWP 和 Base 的 GPU 冲突（全部三个 order）

当前 order1/2/3 的 run_all_seqlora.sh 第二组并行调用时：
```bash
bash scripts_orderX/SeqLoRA_E5_Le4_RWPlora-Gaussian.sh &   # 默认 GPU 4,5,6,7
bash scripts_orderX/SeqLoRA_E1_Le3.sh &                     # 默认 GPU 4,5,6,7 ← 冲突!
wait
```
两个脚本均默认占用 GPU 4,5,6,7，会互相争抢显存导致 OOM。

**修复方法**：显式传入 GPU ID：
```bash
bash scripts_orderX/SeqLoRA_E5_Le4_RWPlora-Gaussian.sh 4,5,6,7 &   # RWP 用 GPU 4-7
bash scripts_orderX/SeqLoRA_E1_Le3.sh 0,1,2,3 &                     # Base 用 GPU 0-3
wait
```

### 问题 2：eval_Whole.sh 使用 lineraProbe_order1_configs（非问题）

eval_Whole.sh 写死了 `configs/lineraProbe_order1_configs/${Task_name}`。
该目录包含 dbpedia/amazon/yahoo/agnews 四个任务的测试集，**与训练顺序无关**。
用于 order2/3 的评估**完全正确**。✓

### 问题 3：eval 脚本 model_paths 写死了 seed-1993

如需使用其他 seed，评估脚本中的 model_paths 需手动修改。
默认 seed=1993 则路径完全匹配。✓

---

## 配置一致性检查结果

| 配置项 | Order1 | Order2 | Order3 | 一致？ |
|--------|--------|--------|--------|--------|
| SAM: GPU | 4,5,6,7 | 4,5,6,7 | 4,5,6,7 | ✓ |
| SAM: batch | 16 | 16 | 16 | ✓ |
| SAM: rho | 0.05 | 0.05 | 0.05 | ✓ |
| SAM: lr | 1e-3 | 1e-3 | 1e-3 | ✓ |
| GAM: GPU | 0,1,2,3 | 0,1,2,3 | 0,1,2,3 | ✓ |
| GAM: batch | 8 | 8 | 8 | ✓ |
| GAM: gam_grad_rho | 0.05 | 0.05 | 0.05 | ✓ |
| GAM: gam_grad_norm_rho | 0.2 | 0.2 | 0.2 | ✓ |
| GAM: gam_grad_gamma | 0.03 | 0.03 | 0.03 | ✓ |
| RWP: GPU | 4,5,6,7 | 4,5,6,7 | 4,5,6,7 | ✓ |
| RWP: batch | 16 | 16 | 16 | ✓ |
| RWP: noise_std | 0.05 | 0.05 | 0.05 | ✓ |
| Base: GPU | 4,5,6,7 | 4,5,6,7 | 4,5,6,7 | ✓ |
| Base: batch | 16 | 16 | 16 | ✓ |
| 所有方法: lora_dim | 8 | 8 | 8 | ✓ |
| 所有方法: epochs | 1/task | 1/task | 1/task | ✓ |
| 所有方法: lr | 1e-3 | 1e-3 | 1e-3 | ✓ |
| 任务顺序 | dbpedia→amazon→yahoo→agnews | dbpedia→amazon→agnews→yahoo | yahoo→amazon→agnews→dbpedia | 各异（正确） |

**结论：同种方法在三个 order 之间超参完全一致。✓**

---

## 训练输出 → 评估路径一致性检查

| 方法 | 训练输出路径 | 评估读取路径 | 匹配？ |
|------|-------------|-------------|--------|
| Order1 SAM | `Logs_Short_Paper/Order-1_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 (相对路径) | ✓ |
| Order1 GAM | `Logs_Short_Paper/Order-1_8/SeqLoRA-GAM_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order1 RWP | `Logs_Short_Paper/Order-1_8/SeqLoRA-RWP_lora_Gauss_std0p05_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order1 Base | `Logs_Short_Paper/Order-1_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order2 SAM | `Logs_Short_Paper/Order-2_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order2 GAM | `Logs_Short_Paper/Order-2_8/SeqLoRA-GAM_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order2 RWP | `Logs_Short_Paper/Order-2_8/SeqLoRA-RWP_lora_Gauss_std0p05_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order2 Base | `Logs_Short_Paper/Order-2_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order3 SAM | `Logs_Short_Paper/Order-3_8/SeqLoRA-SAM-005_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order3 GAM | `Logs_Short_Paper/Order-3_8/SeqLoRA-GAM_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order3 RWP | `Logs_Short_Paper/Order-3_8/SeqLoRA-RWP_lora_Gauss_std0p05_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |
| Order3 Base | `Logs_Short_Paper/Order-3_8/SeqLoRA_T5large_torch_AdamwHF_Lora8_Ba16-1_E1_LRe3_seed-1993/` | 同 | ✓ |

评估结果保存在：`${Model_path}/${Stage}/evalFlat-${task}-trainedLoRAwhole/`

---

## GPU 分配方案

| 方法 | GPU | Batch | 说明 |
|------|-----|-------|------|
| SeqLoRA-SAM | 4,5,6,7 | 16 | SAM rho=0.05 |
| SeqLoRA-GAM | 0,1,2,3 | 8 | GAM rho=0.05, gamma=0.03 |
| SeqLoRA-RWP | 4,5,6,7 | 16 | Gauss std=0.05 on LoRA weights |
| SeqLoRA-Base | 0,1,2,3 | 16 | **修复后：显式传 0,1,2,3** |

**每个 order 内的并行方案（修复 GPU 冲突后）：**
```
阶段1（并行）: SAM(GPU4-7) || GAM(GPU0-3)        → wait
阶段2（并行）: RWP(GPU4-7) || Base(GPU0-3)        → wait  ← 需传 GPU ID
阶段3（并行）: sam_gam eval(GPU0-3) || sgd_rwp eval(GPU4-7)  → wait
```

---

## 实验执行计划（顺序执行 Order1→2→3）

### 步骤 0：修复 run_all_seqlora.sh（三个 order）

编辑 `scripts_order1/run_all_seqlora.sh`、`scripts_order2/run_all_seqlora.sh`、`scripts_order3/run_all_seqlora.sh`，
将第二组并行改为：
```bash
bash scripts_orderX/SeqLoRA_E5_Le4_RWPlora-Gaussian.sh 4,5,6,7 &
bash scripts_orderX/SeqLoRA_E1_Le3.sh 0,1,2,3 &
wait
```

### 步骤 1：运行 Order1

```bash
cd /data/140-0/users/liying/O-LoRA
bash scripts_order1/run_all_seqlora.sh > logs_order1_seqlora.log 2>&1
```

**执行内容：**
1. [并行] SAM 训练 (dbpedia→amazon→yahoo→agnews, GPU4-7) + GAM 训练 (同顺序, GPU0-3)
2. [等待]
3. [并行] RWP 训练 (GPU4-7) + Base 训练 (GPU0-3)
4. [等待]
5. [并行] SAM/GAM 评估 (GPU0-3) + Base/RWP 评估 (GPU4-7)
6. [等待，完成]

**检查点：** `Logs_Short_Paper/Order-1_8/` 下应有 4 个方法目录，每个方法目录下有 1-dbpedia/2-amazon/3-yahoo/4-agnews 四个子目录。

**预计时间：** 约 5-7 小时

### 步骤 2：运行 Order2

Order1 完成后执行：
```bash
bash scripts_order2/run_all_seqlora.sh > logs_order2_seqlora.log 2>&1
```

**执行内容：** 同上，任务顺序为 dbpedia→amazon→agnews→yahoo
**检查点：** `Logs_Short_Paper/Order-2_8/` 下有 4 个方法 × 4 个 stage 目录。
**预计时间：** 约 5-7 小时

### 步骤 3：运行 Order3

Order2 完成后执行：
```bash
bash scripts_order3/run_all_seqlora.sh > logs_order3_seqlora.log 2>&1
```

**执行内容：** 同上，任务顺序为 yahoo→amazon→agnews→dbpedia
**检查点：** `Logs_Short_Paper/Order-3_8/` 下有 4 个方法 × 4 个 stage 目录。
**预计时间：** 约 5-7 小时

---

## 里程碑与决策节点

| 里程碑 | 目标 | 决策 | 预计时间 |
|--------|------|------|---------|
| M0 | 修复 GPU 冲突，验证脚本 dry run | 确认 GPU 分配不重叠后再运行 | 15 min |
| M1 | Order1 训练完成 | 验证 4 个方法 checkpoint 均存在 | ~4-6 hr 训练 |
| M2 | Order1 评估完成 | 查看 eval log 非空，ACC 在合理范围 | ~1-2 hr 评估 |
| M3 | Order2 训练+评估完成 | 同 M1/M2 | ~5-7 hr |
| M4 | Order3 训练+评估完成 | 同 M1/M2 | ~5-7 hr |
| M5 | 汇总结果 | 运行 /analyze-results 解析 eval logs | 30 min |

**总计预计时间：~16-22 小时（三个 order 顺序执行）**

---

## 存储需求

| 项目 | 估算 |
|------|------|
| 每个 checkpoint (~T5-large adapter) | ~50-100 MB |
| 每个方法 × 4 stages | ~200-400 MB |
| 4 个方法 × 4 stages × 3 orders | ~2.4-4.8 GB |
| eval 输出（预测文件） | ~500 MB/方法/order |
| **总计** | **~10-20 GB** |

---

## 风险和缓解措施

| 风险 | 缓解 |
|------|------|
| GPU 冲突（RWP ‖ Base）→ OOM | 步骤 0 修复前不运行 |
| 磁盘空间不足 | 运行前 `df -h /data/140-0` 确认 ≥ 50 GB 可用 |
| Port 冲突（torchrun） | 已用 `shuf -i25000-30000` 随机端口，概率极低 |
| Order2/3 中途 GPU 被占用 | 各 order 顺序执行，不与其他任务竞争 |
| eval 读取半完成的 checkpoint | 训练脚本内每轮之间有 sleep 4，wait 确保完成 |

---

## 最终 Checklist

- [ ] 步骤0：修复三个 order 的 run_all_seqlora.sh（RWP 传 4,5,6,7；Base 传 0,1,2,3）
- [ ] 确认 /data/140-0 磁盘空间 ≥ 50 GB
- [ ] 确认 T5-large 模型在 /data/140-0/users/liying/Sharing/Models/google-t5/t5-large 存在
- [ ] 运行 Order1，验证 checkpoint 和 eval log
- [ ] 运行 Order2，验证 checkpoint 和 eval log
- [ ] 运行 Order3，验证 checkpoint 和 eval log
- [ ] 汇总结果，生成 3×4×4 的比较矩阵
