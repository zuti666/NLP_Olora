"""Collect accepted fixed-endpoint results; smoke values never enter tables."""
import csv
import io
from pathlib import Path
from common import HERE, atomic_json, atomic_text, now, read_json, sha256, verify_reused

READS = {"sgd": 1, "sam": 2, "mirror": 2, "g2": 3, "gam": 4}

def reused_rows(suite):
    verify_reused(suite)
    rows = []
    for reference in suite["reused_results"]:
        path = Path(reference["summary_path"])
        if sha256(path) != reference["summary_sha256"]:
            raise RuntimeError("Reusable result summary changed: " + str(path))
        rows.append(dict(train_scope="lora", method=reference["method"], seed=2, status="completed_reused", test_accuracy=reference["sample_accuracy"], test_ce=reference.get("test_ce"), test_task_macro_accuracy=reference.get("task_macro_accuracy"), updates=2350, ordinary_gradient_reads_per_step=READS[reference["method"]], formal_training_ce_gradient_passes=2350*READS[reference["method"]], output_dir=reference["output_dir"], source_run=reference.get("source_run"), provenance="historical full single200 run"))
    if len(rows) != 3 or {r["method"] for r in rows} != {"sgd", "sam", "g2"}:
        raise RuntimeError("Expected exactly three attributed LoRA references")
    return rows

def collect():
    suite = read_json(HERE / "suite.json")
    statepath = HERE / "runtime/queue_state.json"
    state = read_json(statepath) if statepath.exists() else {"status": "not_launched", "jobs": {}}
    rows = reused_rows(suite)
    for job in suite["jobs"]:
        if job["kind"] != "training" or job["smoke"]:
            continue
        record = state["jobs"].get(job["id"], {"status": "pending"})
        out = Path(job["output_dir"])
        endpoint = {}
        steps = 0
        if record["status"] == "completed":
            from validate_job import validate_job
            accepted = validate_job(job)
            receipt = read_json(HERE / "runtime/jobs" / job["id"] / "completed.json")
            if accepted["summary_sha256"] != record["summary_sha256"] or receipt["summary_sha256"] != accepted["summary_sha256"]:
                raise RuntimeError("Formal completion receipt changed")
            a = read_json(out / "final_summary.json")
            endpoint = a["endpoint_test"]
            steps = a["completed_steps"]
        elif (out / "step_trace.jsonl").exists():
            with (out / "step_trace.jsonl").open() as f:
                steps = sum(bool(line.strip()) for line in f)
        method = job["method"]
        rows.append(dict(train_scope=job["train_scope"], method=method, seed=2, status=record["status"], test_accuracy=endpoint.get("test_accuracy"), test_ce=endpoint.get("test_ce"), test_task_macro_accuracy=endpoint.get("test_task_macro_accuracy"), updates=steps, ordinary_gradient_reads_per_step=READS[method], formal_training_ce_gradient_passes=steps*READS[method], output_dir=job["output_dir"], source_run=job["id"], provenance="new full single200 run"))
    order = {"sgd": 0, "sam": 1, "mirror": 2, "g2": 3, "gam": 4}
    rows.sort(key=lambda r: (r["train_scope"] != "lora", order[r["method"]]))
    results = dict(updated_at=now(), queue_status=state["status"], primary_metric="fixed epoch50 full1519 test sample accuracy (%)", seed=2, dataset="ImageNet-A single200", note="Natural method read costs differ; fixed per-scope SGD LR, no continual learning. Smoke results excluded. Cross-scope accuracy is descriptive because parameter scope and learning rate differ.", results=rows)
    atomic_json(HERE / "RESULTS.json", results)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
    atomic_text(HERE / "RESULTS.csv", buf.getvalue())
    lines = ["# ImageNet-A single200 LoRA / full-FT results", "", f"Queue: {state['status']}. Updated UTC: {now()}.", "", "Fixed epoch50 endpoint on all 1519 test images. Accuracies are sample percentages. Only accepted formal runs appear; smoke measurements are excluded.", "", "| Scope | Method | Status | Test accuracy | Test CE | Updates | Native CE gradient reads/step |", "|---|---|---|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for r in rows:
        lines.append(f"| {r['train_scope']} | {r['method']} | {r['status']} | {fmt(r['test_accuracy'])} | {fmt(r['test_ce'])} | {r['updates']}/2350 | {r['ordinary_gradient_reads_per_step']} |")
    lines += ["", "Within each scope all methods share the initialized ViT/classifier, data stream, 50 epochs, batch128 and cosine SGD. LoRA LR=.04→.004; full-FT LR=.001→.0001. rho=.05 and outer r=1. GAM alone retains its intrinsic gamma=.1 projection. No EWC/Fisher, weight merge, replay or final norm matching.", "", "g2 comparisons (percentage points; full-FT remains pending until accepted):"]
    for scope in ("lora", "full_ft"):
        conditions = {r["method"]: r for r in rows if r["train_scope"] == scope and r["status"].startswith("completed")}
        if "g2" not in conditions:
            continue
        for method in ("sgd", "sam", "mirror", "gam"):
            if method in conditions:
                delta = conditions["g2"]["test_accuracy"] - conditions[method]["test_accuracy"]
                lines.append(f"- {scope}: g2 minus {method} = {delta:+.3f} percentage points.")
    lines += ["", "Single seed: these comparisons assess this configuration and do not establish multi-seed statistical significance. Different native gradient-read costs are recorded, not equalized.", ""]
    atomic_text(HERE / "RESULTS.md", "\n".join(lines))
    return results

if __name__ == "__main__":
    result = collect()
    print({"queue_status": result["queue_status"], "formal_or_reused_rows": len(result["results"])})
