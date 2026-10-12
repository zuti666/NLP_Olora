"""Read-only validation of completed RQ2 receipts; writes analysis artifacts only."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nonfinite(value, prefix=""):
    if isinstance(value, dict):
        return sum((nonfinite(v, f"{prefix}.{k}") for k, v in value.items()), [])
    if isinstance(value, list):
        return sum((nonfinite(v, f"{prefix}[{i}]") for i, v in enumerate(value)), [])
    return [prefix] if isinstance(value, float) and not math.isfinite(value) else []


plan = read(ROOT / "campaign_plan.json")
state = read(Path(plan["run_dir"]) / "pipeline_state.json")
manifest = read(ROOT / "report_seed0/MANIFEST.json")
expected = [job["id"] for stage in plan["stages"] for job in stage["jobs"]]
assert len(expected) == len(set(expected)), "Duplicate planned job IDs"
ledger = []
for name in expected:
    job = state["jobs"].get(name, {})
    receipt = Path(job["receipt"]) if job.get("receipt") else None
    valid = bool(
        job.get("status") == "complete"
        and job.get("exit_code") == 0
        and job.get("receipt_valid") is True
        and receipt and receipt.is_file()
        and sha(receipt) == job.get("receipt_sha256")
    )
    ledger.append({"job": name, "validated": valid,
                   "status": job.get("status", "missing"),
                   "receipt": str(receipt) if receipt else None})

source_checks = []
for name, source in manifest["sources"].items():
    data = read(source["path"])
    bad = nonfinite(data)
    source_checks.append({"source": name,
                          "hash_matches": sha(source["path"]) == source["sha256"],
                          "status": data.get("status"),
                          "nonfinite_observed_fields": bad})

arms = {}
configs = {}
for method in ("sgd", "gam"):
    run = ROOT / "training" / f"A_{method}_to_sgd_seed0"
    rm = read(run / "run_manifest.json")
    cfg = rm["config"]
    configs[method] = cfg
    table = read(ROOT / f"results_seed0_recovery1/{method}/table_a/summary.json")
    local = [json.loads(x) for x in (run / "task0_local_metrics.jsonl").read_text().splitlines()]
    runtime = [json.loads(x) for x in (run / "runtime_receipts.jsonl").read_text().splitlines()]
    checks = {
        "seed0": cfg["seed"] == [0],
        "future_sgd": cfg["optimizer_type"] == "sgd",
        "no_task1_shaping": cfg["lambda_flat"] == 0,
        "budget_50_epochs": cfg["epochs"] == 50,
        "two_tasks": cfg["max_train_tasks"] == 2,
        "ewc_lambda_2000": cfg["ewc_lambda"] == 2000,
        "exact_anchor_restore": any(r.get("event") == "complete_resume" and r.get("verified_exact") is True for r in runtime),
        "old_head_unchanged": table["old_task_head_exactly_unchanged"],
        "frozen_backbone_unchanged": table["nonlora_backbone_exactly_unchanged"],
        "local_metrics_finite": not nonfinite(local),
        "effective_drift_consistent": math.isclose(local[-1]["effective_weight_displacement"], table["effective_weight_drift"], rel_tol=1e-8),
        "a11_same_environment": math.isclose(local[0]["test"]["local_accuracy"], table["accuracy_CIL"]["a11"], abs_tol=1e-9),
    }
    arms[method] = {"checks": checks, "validated": all(checks.values()),
                    "config_source": str(run / "run_manifest.json"),
                    "rng_sha256": sha(run / "rng_receipts.jsonl")}

diff = {k: {m: configs[m].get(k) for m in configs}
        for k in sorted(set(configs["sgd"]) | set(configs["gam"]))
        if configs["sgd"].get(k) != configs["gam"].get(k)}
result = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Completed seed0 campaign; not evidence of a multi-seed confirmation",
    "campaign_status": state["status"], "campaign_finished_utc": state["updated_utc"],
    "jobs_expected": len(expected), "jobs_validated": sum(x["validated"] for x in ledger),
    "job_status_counts": dict(Counter(x["status"] for x in ledger)), "jobs": ledger,
    "unplanned_job_ids": sorted(set(state["jobs"]) - set(expected)),
    "source_checks": source_checks,
    "arms_expected": 2, "arms_validated": sum(x["validated"] for x in arms.values()),
    "arms": arms, "paired_seeds": [0], "paired_n": 1,
    "sample_standard_deviation": None, "sd_reason": "N/A: paired n < 2",
    "config_differences": diff,
    "paired_rng_receipts_identical": arms["sgd"]["rng_sha256"] == arms["gam"]["rng_sha256"],
    "paired_rng_validator_passed": manifest["A_paired_rng"]["passed"],
    "path_numerical_gates_passed": manifest["path_numerical_gates_passed"],
    "null_semantics": "None in diagnostic estimator floor/regularizer means disabled; sd null means N/A, never zero. Historical future-task matrix NaNs are not observed metrics and are outside this scan.",
}
assert result["jobs_validated"] == result["jobs_expected"] == 15
assert result["arms_validated"] == 2
assert all(x["hash_matches"] and not x["nonfinite_observed_fields"] and x["status"] == "complete" for x in source_checks)
assert result["paired_rng_receipts_identical"] and result["paired_rng_validator_passed"]
(HERE / "VALIDITY_LEDGER.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: result[k] for k in ("jobs_validated", "jobs_expected", "arms_validated", "arms_expected", "paired_n", "paired_rng_receipts_identical")}, indent=2))
