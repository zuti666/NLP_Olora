"""CPU-only inventory and aggregation of existing stage-path receipts."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
EXECUTION = BASE / "execution"
FULL = EXECUTION / "full_rq2_20260920"
METHODS = {"sgd": "EWCLoRA", "gam": "FS-EWCLoRA"}


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    records, source_hashes = [], {}
    for seed in range(3):
        for method in METHODS:
            for target in (1, 5, 9):
                directory = FULL / f"formal_results/seed{seed}/{method}/path_t{target:02d}"
                summary_path = directory / "summary.json"
                manifest_path = directory / "manifest.json"
                summary = json.loads(summary_path.read_text())
                manifest = json.loads(manifest_path.read_text())
                assert summary["status"] == "complete"
                assert summary["target_task"] == target and summary["seed"] == seed
                assert summary["method"] == method and summary["samples"] == 1259
                assert summary["protocol_sha256"] == manifest["protocol_sha256"]
                assert all(summary[k] for k in ("numerical_gate_passed", "quadrature_converged", "bound_passed_with_numeric_tolerance"))
                assert summary["endpoint_recovery"]["passed"]
                assert all(math.isfinite(summary[k]) for k in ("S_path", "integral", "drift", "D_test", "D_train"))
                assert math.isclose(summary["S_path"], summary["integral"] / summary["drift"] ** 2, rel_tol=1e-12)
                nodes = summary["path"]
                n = len(nodes)
                assert n == summary["selected_nodes"] and n in (9, 17, 33)
                assert all(math.isclose(row["s"], i / (n - 1), abs_tol=1e-12) for i, row in enumerate(nodes))
                q = [row["q"] for row in nodes]
                assert all(math.isfinite(v) and v >= 0 for v in q)
                integral = (q[0] + q[-1] + 4 * sum(q[1:-1:2]) + 2 * sum(q[2:-1:2])) / (3 * (n - 1))
                assert math.isclose(integral, summary["integral"], rel_tol=1e-12)
                records.append({"cohort": "formal_fresh_training", "dataset": "ImageNet-A", "seed": seed,
                    "method": method, "display_method": METHODS[method], "target_task_code": target,
                    "T_total_learned_tasks": target + 1, "fisher_scope": "Task0_train1259", "samples": 1259,
                    "nodes": n, "S_path": summary["S_path"], "I": summary["integral"], "drift": summary["drift"],
                    "D_test": summary["D_test"], "D_train": summary["D_train"],
                    "summary_path": str(summary_path), "protocol_sha256": summary["protocol_sha256"]})
                for path in (summary_path, manifest_path):
                    source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_csv("FORMAL_PATH_VALUES.csv", records)
    aggregate = []
    for target in (1, 5, 9):
        for method in METHODS:
            selected = [r for r in records if r["target_task_code"] == target and r["method"] == method]
            assert {r["seed"] for r in selected} == {0, 1, 2}
            values = [r["S_path"] for r in selected]
            aggregate.append({"target_task_code": target, "T_total_learned_tasks": target + 1,
                "method": method, "display_method": METHODS[method], "training_seeds": "0,1,2",
                "n_runs": 3, "samples_per_run": 1259,
                "S_path_mean": statistics.mean(values), "S_path_sample_sd": statistics.stdev(values)})
    write_csv("FORMAL_PATH_MEAN_SD.csv", aggregate)
    legacy = []
    for method in METHODS:
        path = EXECUTION / f"results_seed0_recovery1/{method}/path/summary.json"
        row = json.loads(path.read_text())
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        legacy.append({"cohort": "historical_CopyFlatCV_seed0", "method": method,
            "target_task_code": 9, "T_total_learned_tasks": 10, "samples": 256,
            "S_path": row["S_path"], "I": row["integral"], "drift": row["drift"], "summary_path": str(path)})
    write_csv("LEGACY_PATH_VALUES_SEPARATE.csv", legacy)
    config = json.loads((EXECUTION / "diagnostic_config_recovery1.json").read_text())
    historical_checkpoint_availability = {m: {t: Path(p).is_file() for t, p in config["checkpoints"][m].items()} for m in METHODS}
    assert all(all(tasks.values()) and len(tasks) == 10 for tasks in historical_checkpoint_availability.values())
    inventory = {"formal_path_validated": len(records), "formal_path_expected": 18,
        "formal_measured_target_tasks": [1, 5, 9], "formal_missing_target_tasks": [2, 3, 4, 6, 7, 8],
        "formal_missing_jobs_seed0": 12, "formal_missing_jobs_all_three_seeds": 36,
        "legacy_measured_target_tasks": [9], "legacy_missing_target_tasks": list(range(1, 9)),
        "legacy_missing_jobs": 16, "legacy_checkpoint_paths_exist": historical_checkpoint_availability,
        "zero_drift_anchor_S_path": "undefined, not zero", "new_gpu_jobs_launched": 0,
        "check_scope": "summary/manifest protocol and finite values, all path gates, Simpson reconstruction; no checkpoint tensor re-evaluation",
        "source_sha256": source_hashes}
    (HERE / "INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"formal_path_validated": len(records), "legacy_missing_jobs": 16, "formal_missing_jobs_all_three_seeds": 36}))


if __name__ == "__main__":
    main()
