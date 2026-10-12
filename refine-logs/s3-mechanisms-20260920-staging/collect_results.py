"""Build strict JSON/CSV summaries without treating incomplete jobs as results."""
from __future__ import annotations

import csv
import io
import re

from common import (HERE, atomic_json, atomic_text, inspect_metrics, load_suite,
                    now, read_json, resolve, sha256, verify_source)


def collect(suite=None, states=None):
    suite = suite or load_suite()
    source_hash = sha256(resolve(suite["source_manifest"]))
    source_checked = False
    records = []
    for job in suite["jobs"]:
        run = HERE / "runtime" / "jobs" / job["id"]
        row = {key: job.get(key) for key in ("id", "axis", "value", "seed")}
        row.update({"status": "pending", "config_sha256": job["config_sha256"]})
        if (run / "failed.json").exists() or (run / "queue_failed.json").exists():
            failure = run / "failed.json" if (run / "failed.json").exists() else run / "queue_failed.json"
            row.update({"status": "failed", "error": read_json(failure).get("error")})
        elif (run / "completed.json").exists():
            receipt = read_json(run / "completed.json")
            try:
                if receipt["config_sha256"] != job["config_sha256"]:
                    raise ValueError("Completed receipt config hash mismatch")
                if receipt["source_manifest_sha256"] != source_hash:
                    raise ValueError("Completed receipt source manifest hash mismatch")
                if not source_checked:
                    verify_source(suite)
                    source_checked = True
                result = inspect_metrics(suite, job)
                if result["metrics_sha256"] != receipt["result"]["metrics_sha256"]:
                    raise ValueError("Metrics changed after completion")
                boundary = receipt["result"].get("boundary_validation")
                if boundary:
                    for path, expected_hash in boundary["files"].items():
                        if sha256(path) != expected_hash:
                            raise ValueError("Boundary sidecar changed after completion: " + path)
                    result["boundary_validation"] = boundary
                row.update({"status": "completed", "gpu": receipt["gpu"], **result})
            except Exception as exc:
                row.update({"status": "invalid", "error": repr(exc)})
        elif run.exists():
            row["status"] = (states or {}).get(job["id"], {}).get("status", "started_or_interrupted")
            identity = (states or {}).get(job["id"], {})
            if not identity.get("pid") and (run / "started.json").exists():
                identity = read_json(run / "started.json")
            row.update({key: identity[key] for key in ("gpu", "pid") if key in identity})
            log = run / "console.log"
            if log.exists():
                with open(log, "rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 32768))
                    tail = handle.read().decode("utf-8", errors="replace")
                progress = re.findall(r"Task\s+(\d+).*?Epoch\s+(\d+)/(\d+)", tail)
                if progress:
                    row.update(dict(zip(("current_task", "current_epoch", "total_epochs"),
                                        map(int, progress[-1]))))
            try:
                partial = inspect_metrics(suite, job, require_complete=False)
                row["tasks_completed"] = partial["tasks_completed"]
            except Exception:
                pass
        records.append(row)
    document = {"updated_at": now(), "jobs": records}
    atomic_json(HERE / "results.json", document)
    fields = ["id", "axis", "value", "seed", "status", "gpu", "pid", "current_task", "current_epoch",
              "total_epochs", "tasks_completed", "expected_tasks",
              "faa_task_macro", "aaa_task_macro", "forgetting", "bwt", "faa_sample_weighted",
              "aaa_sample_weighted", "config_sha256", "metrics_path", "metrics_sha256", "error"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    atomic_text(HERE / "results.csv", buffer.getvalue())
    lines = ["# Experiment queue", "", f"Updated: {document['updated_at']}", "",
             "| Job | Axis | Value | Seed | Status | GPU | PID | Task | Epoch | Tasks done | FAA macro | FAA sample |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in records:
        cells = [row.get(key, "") for key in ("id", "axis", "value", "seed", "status", "gpu", "pid",
                                              "current_task", "current_epoch", "tasks_completed",
                                              "faa_task_macro", "faa_sample_weighted")]
        lines.append("| " + " | ".join("" if x is None else str(x).replace("|", "\\|") for x in cells) + " |")
    atomic_text(HERE / "EXPERIMENT_TRACKER.md", "\n".join(lines) + "\n")
    return document


if __name__ == "__main__":
    result = collect()
    print("; ".join(f"{row['id']}: {row['status']}" for row in result["jobs"]))
