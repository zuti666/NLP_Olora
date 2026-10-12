"""Recheck raw ImageNet-R curves and summarize historical cohorts; no training."""
import ast
import csv
import hashlib
import json
import math
import re
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value


def write_json(name, value):
    (OUT / name).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n")


def literal(text):
    text = re.sub(r"np\.float(?:32|64)\(([^()]*)\)", r"\1", text)
    return ast.literal_eval(re.sub(r"\bnan\b", "None", text))


def curves(text):
    return [literal(v) for v in re.findall(r"(?:CNN top1 curve|\(curve\) top1 Acc): (\[[^\n]+\])", text)]


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


inventory_path = OUT / "flatness_r10_t10_runs.json"
if inventory_path.exists():
    flat = json.loads(inventory_path.read_text())
else:
    # Initial discovery was read-only over Flatness_CV JSON and companion logs.
    discovered = json.loads(Path("/tmp/imagenetr_flatness_inventory.json").read_text())
    flat = [r for r in discovered if r["tasks"] == 10
            and r["config"].get("lora_rank") == "10"
            and r["curve_faa_sample"] is not None]

flat_csv = []
for r in flat:
    raw = Path(r["log_path"]).read_bytes()
    text = raw.decode()
    r["log_sha256"] = hashlib.sha256(raw).hexdigest()
    raw_json = Path(r["json_path"]).read_bytes()
    r["metrics_sha256"] = hashlib.sha256(raw_json).hexdigest()
    matrix = json.loads(raw_json)["cnn"]["matrices"]["final"]
    assert clean(matrix) == clean(r["final_matrix"])
    assert len(r["curve"]) == 10 and r["curve"] in curves(text)
    assert r["curve_matrix_final_match"] and r["matrix_matched_curve_stages"] == list(range(1, 11))
    assert r["curve_faa_sample"] == r["curve"][-1]
    assert math.isclose(r["curve_aaa_stage_mean"], st.mean(r["curve"]))
    r["stage_task_macro_AAA"] = st.mean(st.mean(matrix[t][:t + 1]) for t in range(10))
    r["eligible_for_cohort"] = not bool(r.get("scientific_header_conflicts"))
    c = r["config"]
    config_path = c.get("config", "").replace("/data/140-0/", "/data/140-1/")
    r["current_config_path"] = config_path if Path(config_path).is_file() else None
    flat_csv.append(dict(
        method=r["method"], optimizer=c.get("optimizer_type"), seed=r["seed_path"],
        FAA=r["curve_faa_sample"], AAA=r["curve_aaa_stage_mean"],
        FAA_task_macro=r["json_faa_task_macro"], AAA_stage_task_macro=r["stage_task_macro_AAA"],
        AAA_native_triangle=r["json_aaa_triangle"], rank=10, tasks=10,
        epochs=c.get("epochs"), init_lr=c.get("init_lr"), lr=c.get("lrate"),
        scheduler=c.get("scheduler"), actual_fixed_class_order=r["logged_order_identity"],
        eligible_for_cohort=r["eligible_for_cohort"],
        header_conflicts=json.dumps(r.get("scientific_header_conflicts", {})),
        config=r["current_config_path"], log=r["log_path"], metrics=r["json_path"],
        log_sha256=r["log_sha256"], metrics_sha256=r["metrics_sha256"]))

write_json("flatness_r10_t10_runs.json", flat)
write_csv("flatness_r10_t10_runs.csv", flat_csv)

old = json.loads((OUT / "olora_inventory.json").read_text())
rows = []
for r in old["runs"]:
    if not r["included"]:
        continue
    raw = Path(r["log"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == r["sha256"]
    assert r["sample_weighted_curve"] in curves(raw.decode())
    assert len(r["sample_weighted_curve"]) == 10
    assert math.isclose(r["FAA_sample"], r["sample_weighted_curve"][-1])
    assert math.isclose(r["AAA_sample"], st.mean(r["sample_weighted_curve"]))
    rows.append(dict(group=r["method"], seed=r["seed"], FAA=r["FAA_sample"],
                     AAA=r["AAA_sample"], source="old_figure_sources",
                     log=r["log"], config=r["logged_config"]))

baseline_names = {"seqlora": "SeqLoRA", "inclora": "IncLoRA", "olora": "O-LoRA", "sdlora": "SD-LoRA"}
for r in flat:
    if r["method"] in baseline_names and r["eligible_for_cohort"]:
        group = baseline_names[r["method"]] + " + " + r["config"]["optimizer_type"].upper() + " (Flatness_CV)"
    elif r["curve_faa_sample"] == 79.92 and r["eligible_for_cohort"]:
        group = "Historical GAM + normalized EWC (selected single-seed candidate)"
    else:
        continue
    rows.append(dict(group=group, seed=int(r["seed_path"]), FAA=r["curve_faa_sample"],
                     AAA=r["curve_aaa_stage_mean"], source="Flatness_CV",
                     log=r["log_path"], config=r["config"]))

summary = []
for group in dict.fromkeys(r["group"] for r in rows):
    cohort = sorted((r for r in rows if r["group"] == group), key=lambda r: r["seed"])
    assert len({r["seed"] for r in cohort}) == len(cohort)
    c = cohort[0]["config"]
    row = dict(method=group, n=len(cohort), seeds=";".join(str(r["seed"]) for r in cohort))
    for metric in ["FAA", "AAA"]:
        values = [r[metric] for r in cohort]
        row[metric + "_mean"] = st.mean(values)
        row[metric + "_sample_sd"] = st.stdev(values) if len(values) > 1 else None
    row.update(optimizer=c.get("optimizer_type", c.get("optimizer")),
               epochs=c.get("epochs"), init_lr=c.get("init_lr", c.get("lrate")),
               lr=c.get("lrate"), config_note="Actual runtime settings are in per_seed_evidence.json")
    summary.append(row)

current = next(r for r in summary if r["method"] == "GAM-EWC+Merge r=1.0")
for row in summary:
    for metric in ["FAA", "AAA"]:
        delta = current[metric + "_mean"] - row[metric + "_mean"]
        row["current_minus_row_" + metric + "_pp"] = delta
        row["current_relative_to_row_" + metric + "_percent"] = delta / row[metric + "_mean"] * 100

write_csv("comparison.csv", summary)
write_json("per_seed_evidence.json", rows)
write_csv("per_seed.csv", [{k: r[k] for k in ["group", "seed", "FAA", "AAA", "source", "log"]}
                           for r in rows])
write_json("comparison.json", dict(
    metric="FAA = final global sample accuracy; AAA = mean of ten global stage accuracies; sample SD ddof=1",
    rounding="Source stage accuracies are rounded to 2 decimals; aggregates recomputed from those values.",
    comparisons="Historical descriptive comparisons. Different training budgets, class orders, checkpoints, and test distributions exist.",
    single_seed_selection="Historical 79.92/83.986 row is selected from a hyperparameter sweep, not a multiseed confirmed winner.",
    invalid_header_runs="Three old ewclora_youyue_fitarchitecture runs have conflicting scientific log headers and are omitted from comparison groups.",
    summary=summary))
print(json.dumps({"flatness_raw_runs": len(flat), "eligible_raw_runs": sum(r["eligible_for_cohort"] for r in flat),
                  "comparison_groups": len(summary), "comparison_runs": len(rows)}, indent=2))
for row in summary:
    print(row["method"], row["n"], f'{row["FAA_mean"]:.3f}', f'{row["AAA_mean"]:.3f}')
