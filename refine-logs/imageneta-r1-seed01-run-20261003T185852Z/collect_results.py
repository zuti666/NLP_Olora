"""Collect only receipted complete10 results, preserving native/probe protocols.

Historical seed2 is the predeclared catalog entry (including audited N03/S),
never the best of several repeats. Incomplete seeds contribute no accuracy.
Standard deviations use independent seed runs and ddof=1, blank for n < 2.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics

from common import HERE, atomic_json, atomic_text, load_suite, now, read_json, sha256
from experiment_contract import NATIVE, PROBE, require

METRICS = ("AAA", "FAA", "LA", "BWT", "forgetting")


def matrix_metrics(matrix):
    """All reported primary accuracies use equal-task, then equal-stage weights."""
    n = len(matrix)
    require(n == 10 and all(len(row) in (i + 1, n) for i, row in enumerate(matrix)),
            "A complete ten-task triangular or square matrix is required")
    observed = [list(row[:i + 1]) for i, row in enumerate(matrix)]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and 0 <= v <= 100 for row in observed for v in row),
            "Observed task accuracies must be finite percentages")
    return {
        "AAA": statistics.mean(statistics.mean(row) for row in observed),
        "FAA": statistics.mean(observed[-1]),
        "LA": statistics.mean(observed[i][i] for i in range(n)),
        "BWT": statistics.mean(observed[-1][i] - observed[i][i] for i in range(n - 1)),
        "forgetting": statistics.mean(max(observed[t][i] for t in range(i, n)) - observed[-1][i]
                                      for i in range(n - 1)),
    }


def read_matrix(path):
    data = read_json(path)
    if "cnn" in data:
        return data["cnn"]["matrices"]["final"]
    return data["matrix"]


def _receipt_result(suite, job, receipt_path):
    """The queue's validator receipt, not a log line or partial summary, is authority."""
    receipt = read_json(receipt_path)
    require(receipt.get("passed") is True, job["id"] + ": no passed completion receipt")
    require(receipt.get("job_id", receipt.get("job")) == job["id"], job["id"] + ": completion identity differs")
    require(receipt.get("status", "completed") == "completed", job["id"] + ": receipt is not completed")
    if "config_sha256" in receipt:
        require(receipt["config_sha256"] == job["config_sha256"], job["id"] + ": receipted config differs")
    require(sha256(job["config"]) == job["config_sha256"], job["id"] + ": current config changed")
    if "suite_sha256" in receipt:
        require(receipt["suite_sha256"] == sha256(HERE / "suite.json"), job["id"] + ": suite changed since validation")
    if "run_pins_sha256" in receipt:
        require(receipt["run_pins_sha256"] == sha256(HERE / "RUN_PINS.json"), job["id"] + ": launch pins changed since validation")
    summary_path = Path(job["output_dir"]) / "final_summary.json"
    require(receipt["summary_sha256"] == sha256(summary_path), job["id"] + ": summary receipt hash mismatch")
    summary = read_json(summary_path)
    require(summary["status"] == "completed" and summary["seed"] == job["seed"],
            job["id"] + ": summary status/seed differs")
    require(summary["completed_tasks"] == 10 and summary["completed_steps"] == 2550
            and summary["epochs_per_task"] == 50, job["id"] + ": full10 training budget missing")
    if job["family"] == NATIVE:
        result = summary["result"]
        metric_path = Path(result["metrics_path"])
        digest = result["metrics_sha256"]
    else:
        metric_path = Path(job["output_dir"]) / "task_matrix.json"
        digest = summary["task_matrix_sha256"]
    require(metric_path.resolve().is_relative_to(Path(job["output_dir"]).resolve()),
            job["id"] + ": metrics path escaped the job output")
    require(sha256(metric_path) == digest, job["id"] + ": completed task matrix changed")
    metrics = matrix_metrics(read_matrix(metric_path))
    return dict(metrics=metrics, metric_path=str(metric_path), metric_sha256=digest,
                summary_path=str(summary_path), summary_sha256=receipt["summary_sha256"],
                receipt_path=str(receipt_path), receipt_sha256=sha256(receipt_path),
                completed_at=receipt.get("completed_at"))


def _historical_rows(catalog):
    """Recompute catalog seed2 metrics from their immutable original matrices."""
    index_path = HERE / "plan_snapshot/COMPLETED_SEED2_INDEX.csv"
    with open(index_path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    require(len(records) == 30 and {r["condition"] for r in records} == set(catalog),
            "Historical seed2 index must contain exactly the predeclared 30 conditions")
    rows = []
    for record in records:
        condition = record["condition"]
        expected = catalog[condition]
        for key in ("family", "source_seed2_id", "source_config", "source_config_sha256",
                    "source_metrics", "source_metrics_sha256"):
            require(record[key] == expected[key], condition + ": historical source provenance changed")
        require(sha256(record["source_config"]) == record["source_config_sha256"], condition + ": historical config changed")
        require(sha256(record["source_metrics"]) == record["source_metrics_sha256"], condition + ": historical matrix changed")
        metrics = matrix_metrics(read_matrix(record["source_metrics"]))
        for key in ("AAA", "FAA"):
            require(abs(metrics[key] - float(record[key])) <= 1e-9,
                    condition + ": index accuracy does not match original matrix")
        for key, value in expected["seed2_metrics"].items():
            if key in METRICS:
                require(abs(metrics[key] - value) <= 1e-9, condition + ": catalog score differs from original matrix")
        rows.append(dict(condition=condition, family=record["family"], label=record["label"],
                         job=record["source_seed2_id"], seed=2, status="validated_historical",
                         source="predeclared_seed2", **metrics,
                         metrics_path=record["source_metrics"], metrics_sha256=record["source_metrics_sha256"]))
    return rows


def aggregate_rows(rows, historical, catalog):
    all_rows = rows + historical
    lookup = {(r["condition"], r["family"], r["seed"]): r for r in all_rows}
    require(len(lookup) == len(all_rows), "Duplicate seed within scientific condition/protocol")
    result = []
    for condition, cat in sorted(catalog.items()):
        cells = [lookup[(condition, cat["family"], seed)] for seed in (0, 1, 2)]
        complete = [row for row in cells if row["status"] in ("completed", "validated_historical")]
        row = dict(condition=condition, family=cat["family"], label=cat["label"],
                   expected_seeds="0;1;2", completed_seeds=";".join(str(r["seed"]) for r in complete),
                   n_expected=3, n_complete=len(complete),
                   aggregate_status="complete_3_seeds" if len(complete) == 3 else "partial_seeds",
                   std_ddof=1)
        for cell in cells:
            row[f"seed{cell['seed']}_status"] = cell["status"]
            for metric in METRICS:
                row[f"{metric}_seed{cell['seed']}"] = cell.get(metric)
        for metric in METRICS:
            values = [cell[metric] for cell in complete]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else None
        result.append(row)
    return result


def _csv(path, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def collect(suite, write=True):
    catalog = {row["id"]: row for row in read_json(HERE / "plan_snapshot/CONDITION_CATALOG.json")["conditions"]}
    state_path = HERE / "runtime/queue_state.json"
    state = read_json(state_path) if state_path.exists() else {"jobs": {}}
    rows = []
    for job in sorted((j for j in suite["jobs"] if not j["smoke"]), key=lambda j: (j["condition"], j["seed"])):
        lifecycle = state.get("jobs", {}).get(job["id"], {"status": "pending"})
        status = lifecycle.get("status", "pending")
        receipt_path = HERE / "runtime/jobs" / job["id"] / "completed.json"
        row = dict(condition=job["condition"], family=job["family"], label=job["label"],
                   job=job["id"], seed=job["seed"], status=status, source="new_seed01",
                   tasks=0, steps=0, AAA=None, FAA=None, LA=None, BWT=None, forgetting=None,
                   gpu=lifecycle.get("gpu"), config=job["config"], config_sha256=job["config_sha256"],
                   output_dir=job["output_dir"], metrics_path=None, metrics_sha256=None,
                   summary_path=None, summary_sha256=None, receipt_path=None,
                   receipt_sha256=None, completed_at=None)
        if receipt_path.is_file():
            verified = _receipt_result(suite, job, receipt_path)
            row.update(verified.pop("metrics"))
            row.update(status="completed", tasks=10, steps=2550,
                       metrics_path=verified.pop("metric_path"), metrics_sha256=verified.pop("metric_sha256"))
            row.update(verified)
        else:
            # A stale or manually edited state file cannot promote an unvalidated
            # run. No task matrix, summary or live log is used for final accuracy.
            if status == "completed":
                row["status"] = "awaiting_completion_receipt"
        rows.append(row)
    require(len(rows) == 60, "Expected exactly 60 new full10 jobs")
    historical = _historical_rows(catalog)
    aggregates = aggregate_rows(rows, historical, catalog)
    updated = now()
    report = dict(updated_at=updated, metric_protocol="equal-task FAA and equal-stage task-macro AAA",
                  std_ddof=1, formal_jobs=60, completed_new=sum(r["status"] == "completed" for r in rows),
                  historical_seed2=30, completed_three_seed_conditions=sum(r["n_complete"] == 3 for r in aggregates),
                  rows=rows, historical=historical, aggregates=aggregates,
                  native_S_policy="Audited N03 seed2 retained by predeclared source, historical duplicate excluded",
                  comparison_policy="Never pool native global-RNG 5CE and exact-restoration four-probe conditions")
    if write:
        _csv(HERE / "summary.csv", rows)
        _csv(HERE / "THREE_SEED_SUMMARY.csv", aggregates)
        atomic_json(HERE / "results.json", report)
        lines = ["# ImageNet-A r=1：seed0/1 纯 GAM 重复实验", "", "更新：" + updated, "",
                 f"正式训练验收 {report['completed_new']}/60；已有 seed2 复用30项；三seed齐全 {report['completed_three_seed_conditions']}/30 条件。",
                 "AAA/FAA均按任务等权计算；仅纳入通过验收的完整十任务结果。std为seed间样本标准差（ddof=1），不足两个seed留空。",
                 "原生5次CE协议与精确恢复4探针协议分开统计；同名梯度不跨协议合并。", "",
                 "| 条件 | 协议 | 已完成seed | AAA均值 ± std | FAA均值 ± std | 状态 |",
                 "|---|---|---|---:|---:|---|"]
        def fmt(mean, std):
            return "—" if mean is None else f"{mean:.3f}" + (" ± —" if std is None else f" ± {std:.3f}")
        for row in aggregates:
            lines.append("| " + " | ".join([row["condition"] + " " + row["label"],
                          "原生5CE" if row["family"] == NATIVE else "恢复4探针",
                          row["completed_seeds"] or "—", fmt(row["AAA_mean"], row["AAA_std"]),
                          fmt(row["FAA_mean"], row["FAA_std"]), row["aggregate_status"]]) + " |")
        atomic_text(HERE / "RESULTS.md", "\n".join(lines) + "\n")
    return report


def self_test(suite):
    report = collect(suite, write=False)
    # A synthetic complete10 matrix with a known forget/transfer pattern proves
    # task weighting, rather than simply replaying the aggregation implementation.
    matrix = [[80. - 2*(i-j) for j in range(i + 1)] for i in range(10)]
    observed = matrix_metrics(matrix)
    expected = dict(AAA=75.5, FAA=71., LA=80., BWT=-10., forgetting=10.)
    require(observed == expected, "Synthetic metric oracle failed")
    try:
        matrix_metrics(matrix[:9])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete nine-task matrix accepted")
    # n=1 historical-only rows must expose unknown std, never zero uncertainty.
    for row in report["aggregates"]:
        if row["n_complete"] == 1:
            require(all(row[m + "_std"] is None for m in METRICS), "n=1 reported a numerical std")
    return {"status": "passed", "new_formal_rows": len(report["rows"]),
            "historical_rows": len(report["historical"]), "conditions": len(report["aggregates"]),
            "completed_new": report["completed_new"], "metric_oracle": observed,
            "incomplete_matrix_rejected": True, "n1_std_blank": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=HERE / "suite.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if args.self_test:
        result = self_test(suite)
    else:
        report = collect(suite, write=not args.check_only)
        result = {key: report[key] for key in ("updated_at", "formal_jobs", "completed_new", "historical_seed2", "completed_three_seed_conditions")}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
