#!/usr/bin/env python3
"""CPU-only paired analysis of the complete seven-arm ImageNet-A table.

Reads only the three public final_report files. Writes only
additional_paired_contrasts.csv and INTERPRETATION.md beside this script.
No third-party dependencies, training, checkpoint loading, or GPU access.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = (0, 1, 2)
METRICS = ("a21", "a22", "BWT", "average_accuracy", "d_QV_cont")
NUMERIC = ("a11", "B1_batch_proxy", "raw_trace_anchor_train256") + METRICS
CONFIG = {
    "vanilla": ("sgd", "sgd", 0, "C", "C"),
    "shaping_only": ("gam", "gam", 0, "C+F", "C+F"),
    "ewclora": ("sgd", "sgd", 2000, "C", "C+P"),
    "fs_ewclora": ("gam", "gam", 2000, "C+F", "C+F+P"),
    "task1_shaping_no_ewc": ("gam", "sgd", 0, "C+F", "C"),
    "task1_shaping": ("gam", "sgd", 2000, "C+F", "C+P"),
    "task2_shaping": ("sgd", "gam", 2000, "C", "C+F+P"),
}
CONTRASTS = (
    ("fs_ewclora", "shaping_only", "两任务GAM下第二任务P的增量"),
    ("task2_shaping", "ewclora", "SGD父起点且P开启时第二任务GAM的增量"),
    ("fs_ewclora", "task2_shaping", "未来GAM+P固定时第一任务GAM训练流程的增量"),
    ("shaping_only", "vanilla", "P关闭时两任务GAM训练流程的总差异"),
    ("fs_ewclora", "vanilla", "FS完整流程相对Vanilla的总差异"),
)
AUXILIARY = (
    ("ewclora", "vanilla", "两任务SGD下第二任务P的增量"),
    ("task1_shaping", "task1_shaping_no_ewc", "GAM父起点且未来SGD时第二任务P的增量"),
)
EPS = 1e-10


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_and_check(report):
    with (report / "individual.csv").open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    data = json.loads((report / "table_data.json").read_text())
    caption = (report / "TABLE_IMAGENETA.md").read_text()
    require(data["seeds"] == list(SEEDS), "Unexpected seed cohort")
    require(data["expected_main_cells"] == 21, "Unexpected expected count")
    require(data["validated_main_cells"] == 21, "Source ledger not complete")
    require(data["missing_main_cells"] == 0, "Source ledger has missing cells")
    require("d_QV_cont" in caption and "batch" in caption, "Metric caption missing")
    rows = {}
    for raw_row in raw:
        row = dict(raw_row)
        row["seed"] = int(row["seed"])
        row["ewc_lambda"] = int(row["ewc_lambda"])
        key = row["arm"], row["seed"]
        require(key not in rows, f"Duplicate CSV cell: {key}")
        require(row["status"] == "complete", f"Incomplete CSV cell: {key}")
        require(row["arm"] in CONFIG, f"Unknown arm: {key}")
        actual = tuple(row[k] for k in ("parent_method", "future_optimizer", "ewc_lambda", "task1_objective", "task2_objective"))
        require(actual == CONFIG[row["arm"]], f"Configuration mismatch: {key}")
        for metric in NUMERIC:
            row[metric] = float(row[metric])
            require(math.isfinite(row[metric]), f"Nonfinite metric: {key} {metric}")
        require(close(row["BWT"], row["a21"] - row["a11"]), f"BWT mismatch: {key}")
        require(close(row["average_accuracy"], (row["a21"] + row["a22"]) / 2), f"Average mismatch: {key}")
        rows[key] = row
    expected = {(arm, seed) for arm in CONFIG for seed in SEEDS}
    require(set(rows) == expected, "Missing or extra CSV cells")
    json_rows = {(r["arm"], r["seed"]): r for r in data["individual"]}
    require(len(json_rows) == len(data["individual"]) == 21, "Duplicate or missing JSON cells")
    require(set(json_rows) == expected, "JSON cell identities differ")
    for key, row in rows.items():
        for field, value in row.items():
            other = json_rows[key][field]
            ok = close(value, other) if field in NUMERIC else value == other
            require(ok, f"CSV / JSON mismatch: {key} {field}")
    for parent in ("sgd", "gam"):
        for seed in SEEDS:
            group = [r for r in rows.values() if r["parent_method"] == parent and r["seed"] == seed]
            for field in ("parent_checkpoint", "anchor_metrics") + NUMERIC[:3]:
                require(len({r[field] for r in group}) == 1, f"Parent mismatch: {parent} {seed} {field}")
    aggregates = {r["arm"]: r["statistics"] for r in data["aggregates"]}
    require(set(aggregates) == set(CONFIG), "Missing aggregate arm")
    for arm in CONFIG:
        for metric in NUMERIC:
            vals = [rows[arm, seed][metric] for seed in SEEDS]
            saved = aggregates[arm][metric]
            require(saved["n"] == 3 and close(saved["mean"], statistics.mean(vals)) and close(saved["sample_sd"], statistics.stdev(vals)), f"Aggregate mismatch: {arm} {metric}")
    extension = data["extension_validation"]
    require(len(extension) == 6, "Unexpected extension count")
    require({(r["arm"], r["seed"]) for r in extension} == {(a, s) for a in ("shaping_only", "task2_shaping") for s in SEEDS}, "Extension identities differ")
    for result in extension:
        key = result["arm"], result["seed"]
        require(result["status"] == "complete" and result["actual_future_optimizer"] == "gam", f"Extension status / optimizer mismatch: {key}")
        require(result["epochs"] == 50 and result["training_batches_per_epoch"] == 8, f"Extension budget mismatch: {key}")
        require(result["paired_task1_rng_passed"] and result["old_head_unchanged"] and result["frozen_backbone_unchanged"], f"Extension validator reports failure: {key}")
        require(result["ewc_lambda"] == rows[key]["ewc_lambda"], f"Extension lambda mismatch: {key}")
        require(close(result["drift_cpu_l2"], rows[key]["d_QV_cont"]), f"Extension drift mismatch: {key}")
        require(close(result["drift_cpu_l2"] ** 2, result["drift_cpu_l2_squared"]), f"Extension drift norm invariant failed: {key}")
    return rows, data


def contrast_rows(rows, definitions):
    output = []
    for left, right, estimand in definitions:
        for metric in METRICS:
            deltas = [rows[left, seed][metric] - rows[right, seed][metric] for seed in SEEDS]
            positive = sum(d > EPS for d in deltas)
            negative = sum(d < -EPS for d in deltas)
            drift = metric == "d_QV_cont"
            output.append({
                "contrast": f"{left}-{right}", "estimand": estimand,
                "left_arm": left, "right_arm": right, "metric": metric,
                "unit": "effective_weight_Frobenius_norm" if drift else "percentage_points",
                "direction": "lower_displacement_descriptive_only" if drift else "higher_is_better",
                "n": len(deltas), "seed0_delta": deltas[0], "seed1_delta": deltas[1], "seed2_delta": deltas[2],
                "mean": statistics.mean(deltas), "sample_sd": statistics.stdev(deltas),
                "positive_count": positive, "negative_count": negative,
                "tie_count": 3 - positive - negative,
                "win_count": negative if drift else positive,
            })
    return output


def fmt(value, places=3, signed=True):
    if abs(value) < 0.5 * 10 ** (-places):
        value = 0.0
    return format(value, f"{'+' if signed else ''}.{places}f")


def paired_table(records):
    lines = ["| 对照（左减右） | 指标 | 配对均值±样本SD | seed0 / seed1 / seed2 | win / tie / loss |",
             "|---|---|---:|---|---:|"]
    for row in records:
        places = 6 if row["metric"] == "d_QV_cont" else 3
        mean_sd = f"{fmt(row['mean'], places)} ± {fmt(row['sample_sd'], places, False)}"
        deltas = " / ".join(fmt(row[f"seed{s}_delta"], places) for s in SEEDS)
        losses = 3 - row["win_count"] - row["tie_count"]
        lines.append(f"| {row['contrast']} | {row['metric']} | {mean_sd} | {deltas} | {row['win_count']} / {row['tie_count']} / {losses} |")
    return lines


def write_interpretation(output, report, rows, data, primary, auxiliary):
    lines = [
        "# 七行 ImageNet-A 完整结果的独立配对解读",
        "",
        "本报告采用 analyze-results 技能的同 seed 配对统计规则，只在 CPU 上读取已完成结果。准确率和 BWT 的差值单位为百分点；SD 是三个配对差值的样本标准差（ddof=1），不是两组 SD 相减，也不是标准误。所有差值按左行减右行计算。",
        "",
        "## 数据完整性与复核范围",
        "",
        "- 预期与观察：7 行 × seed {0,1,2} = 21/21；缺失 0，重复 0，非 complete 0，非有限观测指标 0。",
        "- 本脚本核对 CSV 与 JSON 的全部逐 seed 字段、显式方法配置、同类型同 seed 父起点路径及第一任务指标、七行均值与样本 SD、BWT=a21−a11、average=(a21+a22)/2。全部通过。",
        "- table_data.json 声明 validated_main_cells=21、missing_main_cells=0；六个新增结果的内嵌验证记录均报告 50 轮、每轮 8 batch、实际 GAM、配对 RNG 通过、旧分类头及冻结骨干不变，并满足 CPU 位移范数与平方的一致性。",
        "- 上述运行状态与 checkpoint 验证结论来自报告元数据；本脚本没有重做底层日志、checkpoint SHA、退出码或权重审计。它核对报表一致性和统计，不替代独立实验完整性审计。主表没有声明需要终点 loss 的统计契约，本分析没有用缺失 loss 填零。",
        "- 输入准确率 a21/a22 已保留两位小数；精确相等只表示已报告指标相等，不表示模型权重、概率输出或优化轨迹相同。",
        "",
        "## 核心判断",
        "",
        "1. **本配置下 EWC 的准确率边际收益很小。** FS−shaping-only 的平均准确率为 −0.001667±0.197505，三个 seed 为 −0.200、0、+0.195；a21 和 BWT 三 seed 全部相等，a22 为 −0.003333±0.395011。EWC 仍使持续 Q/V 位移缩小 −0.001412±0.000974（三 seed 都缩小），约为 shaping-only 平均位移的 0.28%。因此可以说“当前 λ=2000、三 seed 的准确率未见一致增益”，不能说 EWC 严格无效、被统计证明等价，或实现没有生效。另两组 P 开关对照同样显示准确率收益有限，见下表。",
        "2. **第二任务 GAM 带来新任务学习与旧任务保留的权衡。** 从 EWCLoRA 改为 Task-2 shaping（同一 SGD 父起点，P 开启），a22 增加 +5.773±0.817，a21/BWT 降低 −2.280±1.135，平均准确率增加 +1.747±0.412，Q/V 位移增加 +0.202149±0.008892；这些方向在三个 seed 一致。这支持新任务适应更强、旧任务遗忘更多的描述，不能把平均准确率上升写成所有维度同时改善。",
        "3. **在未来 GAM+P 固定时，第一任务 GAM 流程改善了保留，却略牺牲新任务表现。** FS−Task-2 shaping 的 a21 为 +3.257±0.325，BWT 为 +3.040±0.819，a22 为 −1.310±0.606，平均准确率为 +0.973±0.234，位移为 −0.076140±0.005822；三个 seed 方向一致。这是第一任务训练流程的条件效应：它同时改变父 checkpoint、保护中心和由父状态得到的 Fisher 信息，不能归因于 B1 单一标量。",
        "4. **两任务 GAM 的收益在不加 EWC 时仍然存在。** Shaping-only−Vanilla 的平均准确率为 +2.722±0.309（3/3 获胜）、a22 为 +4.467±0.817（3/3），a21 为 +0.977±1.175（2 胜 1 平），BWT 为 +0.760±0.995（2 胜 1 负）。FS−Vanilla 相应为平均 +2.720±0.374、a22 +4.463±0.453；a21/BWT 与 shaping-only 相同。当前准确率提升主要由 GAM 流程差异支持；没有证据声称加 EWC 后有额外稳定提升。FS−Vanilla 与 FS−EWCLoRA 的均值虽同为 +2.720，配对 SD 不同（分别 0.374 与 0.215），不可混写。",
        "5. **最少遗忘的行不是平均准确率最高的行。** Task-1 shaping 的平均 a21=86.643、BWT=−0.653，但 a22=72.963、平均=79.803；FS 的平均 a21=83.387、BWT=−3.910、a22=80.840、平均=82.113。只比较 BWT 会忽略新任务学习不足；七行体现不同保留/适应折中，不能给出跨所有指标的统一优胜结论。Shaping-only 与 FS 的平均准确率处于同一水平，三 seed 不支持把小数末位差异作为方法排名证据。",
        "",
        "## 五组要求的同 seed 差值",
        "",
        "每组 n=3。win 对准确率与 BWT 表示差值为正；对 d_QV_cont 仅表示左行位移更小，是按方向计数，不意味着预测质量或稳定性必然更好。CSV 同时提供正、负、平数量，避免把位移方向误当性能胜负。判平容差为 1e−10，报告显示值不用于重新判平。",
        "",
    ]
    lines += paired_table(primary)
    lines += [
        "", "## 支撑 EWC 判断的另外两组 P 开关", "",
        "两任务 SGD 时，EWCLoRA−Vanilla 的平均准确率为 0.000±0.160；第一任务 GAM、未来 SGD 时，加 EWC 的平均增量为 +0.053±0.092（只有 seed1 改善）。三种已观察 P 开关都使位移略减，但准确率增量很小或方向混合。它们不支持将完整 FS 的总收益解释为 P 的贡献。", "",
    ]
    lines += paired_table(auxiliary)
    lines += [
        "", "## 指标与可解释范围", "",
        "- **F 的含义：** F 代表 GAM 实现的 shaping 更新。结果不证明代码直接最小化显式 Fisher-trace 惩罚，也不能仅凭效果把 GAM 的全部优化差异解释为 Fisher 机制。",
        "- **B1 的含义：** B1 是第一任务正式 anchor 训练日志中的 batch-gradient 平方 proxy，未归一化、无路径平均；不是逐样本梯度平方求和的 raw Fisher。SGD/GAM 的 B1 平均值为 31.440922/21.932823，而固定旧 train256 的逐样本 raw Fisher 平均为 1451.954128/634.510132。这两种量各有口径，不能互换或把 batchproxy 差异等同于总体 Fisher 真值变化。",
        "- **B1 不是独立消融因素：** 同一第一任务 GAM 起点在后续 SGD 与 GAM 下共享相同 B1，却产生明显不同的新任务准确率及遗忘；较低 B1 不是好结果的充分条件。报告中的关联不能建立低 Fisher → 少位移 → 少遗忘的因果链。",
        "- **位移的含义：** d_QV_cont 是延续同一 LoRA 分支时第一任务终点至第二任务终点的有效 Q/V 权重 Frobenius 位移；排除分类头，不是 A/B 因子距离，也不是 fresh K/V adapter 的范数。它不是 Fisher 加权位移；不同方向、分类头和任务间干扰均可能影响准确率。Shaping-only/FS 相比 Vanilla 位移更大而平均准确率更好，本表不支持“位移越小性能必然越好”。",
        "- **BWT 的基线：** BWT 按每个分支实际未舍入 a11 计算。跨 SGD/GAM 父起点时 ΔBWT=Δa21−Δa11；FS−Task-2 shaping 的 +3.040 不等于 Δa21 的 +3.257。FS/shaping-only 相比 Vanilla 的 BWT 在 seed1 反而差 −0.325733，不能称三个 seed 遗忘均减少。",
        "- **七行不是完整八格析因设计：** 以 F1、F2、P2 为三个二值因素，缺少 (F1=0,F2=1,P2=0)，即 SGD→GAM 且第二任务无 EWC。这里估计的是实际存在分支之间的条件简单效应及完整流程差异；没有在缺失 cell 上插值。P2=1 层的四格齐全，但这不能代替跨 P 的完整三因子主效应或三阶交互，更不能由当前五组差值宣称完整析因因果分解。",
        "- **证据强度：** 三 seed 均值、SD 与 win count 是描述性结果。未做显著性或等效性检验；3/3 同向不等于统计显著。现有配置也不覆盖更多 seed、不同 λ、其它数据集、full model 或严格等 FLOP 的优化器比较。",
        "",
        "## 完整逐 seed 观测值", "",
        "以下为全部 21 个有效报表行，未排除任何 seed。表内为了阅读而舍入；脚本与 CSV 用输入浮点值计算。B1 和 a11 的完整值及原始来源路径见 ../final_report/individual.csv。", "",
        "| arm | seed | a11 | B1 batch proxy | a21 | a22 | BWT | average_accuracy | d_QV_cont |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in CONFIG:
        for seed in SEEDS:
            row = rows[arm, seed]
            values = [fmt(row[m], 6 if m == "d_QV_cont" else 3, False) for m in ("a11", "B1_batch_proxy") + METRICS[:3] + ("average_accuracy", "d_QV_cont")]
            lines.append(f"| {arm} | {seed} | " + " | ".join(values) + " |")
    lines += [
        "", "## 可检验的后续问题（本次未启动实验）", "",
        "若要把“当前边际收益小”提升为 EWC 机制结论，应检查 EWC 损失/梯度相对任务损失的量级与归一化，并在更多 seed 和预先规定的 λ 范围检验效应；若要完整三因素分解，应补唯一缺失的无 EWC SGD→GAM cell。若要验证 Fisher 机制，应在统一样本与估计口径下比较逐样本 Fisher、Fisher 加权位移与遗忘，并设计能分开 GAM 更新规则及 Fisher 标量变化的控制实验。", "",
        "## 复现", "",
        "执行 `python compare_complete.py` 可重新生成本报告及 `additional_paired_contrasts.csv`。仅使用 Python 标准库；不导入 torch、不加载 checkpoint、不启动训练或模型评估。CSV 收录五组要求的 25 条指标对比，另外两组 P 开关只展示在本报告。", "",
        "输入文件 SHA-256：", "",
    ]
    for name in ("individual.csv", "table_data.json", "TABLE_IMAGENETA.md"):
        digest = hashlib.sha256((report / name).read_bytes()).hexdigest()
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", f"来源目录：`{report}`", ""]
    (output / "INTERPRETATION.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent.parent / "final_report")
    args = parser.parse_args()
    report = args.report_dir.resolve()
    output = Path(__file__).resolve().parent
    rows, data = load_and_check(report)
    primary = contrast_rows(rows, CONTRASTS)
    auxiliary = contrast_rows(rows, AUXILIARY)
    with (output / "additional_paired_contrasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(primary[0]))
        writer.writeheader()
        writer.writerows(primary)
    write_interpretation(output, report, rows, data, primary, auxiliary)
    print(f"PASS: {len(rows)}/21 complete unique cells; CSV/JSON/aggregates/configurations/invariants consistent.")
    print(f"Wrote {len(primary)} requested paired contrasts and INTERPRETATION.md to {output}")


if __name__ == "__main__":
    main()
