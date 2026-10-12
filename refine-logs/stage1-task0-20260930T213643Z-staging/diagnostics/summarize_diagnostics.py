#!/usr/bin/env python3
"""Read diagnostic JSONs; write tables without changing checkpoints or inputs.

No means are pooled across anchors or batches, and batches are never counted as
independent seeds. Only completed reports that pass consistency checks enter CSVs.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def flatten_fields(value, prefix=""):
    """CSV keeps scalars as columns; lists stay losslessly encoded JSON cells."""
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(flatten_fields(item, name + "."))
        elif isinstance(item, (list, tuple)):
            result[name] = json.dumps(item, ensure_ascii=False, allow_nan=False)
        else:
            result[name] = item
    return result


def validate(report):
    if report.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if not finite(report):
        raise ValueError("nonfinite fields")
    if report.get("finite_check", {}).get("ok") is not True:
        raise ValueError("finite_check did not pass")
    checks = report.get("final_restore_check", {})
    expected = ("parameters_exact", "buffers_exact", "modes_exact", "rng_exact", "gradients_exact")
    if any(checks.get(k) is not True for k in expected):
        raise ValueError("final restoration did not pass")
    batches = report.get("batches", [])
    if not batches or report.get("batch_count") != len(batches):
        raise ValueError("batch_count inconsistent or empty")
    if len({b["batch"] for b in batches}) != len(batches):
        raise ValueError("duplicate batch index")
    requested = report.get("arguments", {}).get("max_batches")
    if requested is not None and requested != len(batches):
        raise ValueError("fewer batches than requested")
    for batch in batches:
        if any(batch.get("restore_check", {}).get(k) is not True for k in expected):
            raise ValueError("batch restoration did not pass")
        for sanity in ("scale_only_sanity", "r0_identity_sanity"):
            if batch.get(sanity, {}).get("passed") is False:
                raise ValueError(sanity + " failed")
        labels = [item["label"] for item in batch["counterfactuals"]]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("empty or duplicated counterfactual labels")
        for item in batch["counterfactuals"]:
            if item["match"].get("ok"):
                if "loss_change" not in item or "matched_actual_step" not in item:
                    raise ValueError("successful match lacks actual step/loss measurement")
                actual_error = item["match"].get("actual_relative_error")
                limit = report.get("arguments", {}).get("match_rtol", 1e-3)
                if actual_error is None or actual_error > limit:
                    raise ValueError("successful match exceeds actual-distance tolerance")
    for field in ("source_config_sha256", "checkpoint_hashes", "training_seed", "task"):
        if report.get(field) is None:
            raise ValueError("missing provenance: " + field)


def anchor_metadata(report, path):
    args = report["arguments"]
    identity = {k: report[k] for k in ("source_config_sha256", "checkpoint_hashes", "training_seed", "task")}
    identity["checkpoint_dir"] = args["checkpoint_dir"]
    identity["source"] = report.get("source")
    anchor_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return {"anchor_id": anchor_id, "anchor_name": Path(args["base_config"]).stem,
            "task": report["task"], "training_seed": report["training_seed"],
            "diagnostic_seed": args.get("seed"), "checkpoint_dir": args["checkpoint_dir"],
            "base_config": args["base_config"], "source_config_sha256": report["source_config_sha256"],
            "report_path": str(path), "report_sha256": digest(path)}


def tables_for(report, meta):
    counterfactuals, radii, geometries = [], [], []
    for batch in report["batches"]:
        base = {**meta, **{k: batch.get(k) for k in
                ("batch", "sample_count", "inputs_sha256", "targets_sha256")}}
        sam = flatten_fields(batch.get("radius_matched_sam", {}), "sam.")
        geometry = {**base, **flatten_fields(batch["geometry"]), **sam}
        geometry.update(flatten_fields(batch.get("scale_only_sanity", {}), "scale_only_sanity."))
        geometry.update(flatten_fields(batch.get("r0_identity_sanity", {}), "r0_identity_sanity."))
        geometries.append(geometry)
        for radius, scan in batch["radius_scan"].items():
            row = {**base, "r": float(radius)}
            row.update(flatten_fields({k: v for k, v in scan.items() if k != "points"}))
            for index, point in enumerate(scan.get("points", [])):
                row.update(flatten_fields(point, "probe%d." % index))
            radii.append(row)
        for item in batch["counterfactuals"]:
            row = {**base, **sam}
            row.update(flatten_fields({k: v for k, v in item.items()
                                       if k not in ("loss_change", "loss_after")}))
            change = item.get("loss_change", {})
            row.update(flatten_fields(change.get("current", {}), "current_delta."))
            row.update(flatten_fields(change.get("old_task_macro") or {}, "old_task_macro_delta."))
            row["old_task_status"] = change.get("old_task_status", "NA_unmatched")
            row["per_task_delta_json"] = json.dumps(change.get("per_task", {}), allow_nan=False)
            row["loss_after_json"] = json.dumps(item.get("loss_after", {}), allow_nan=False)
            counterfactuals.append(row)
    return counterfactuals, radii, geometries


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_csv(path, rows):
    common = ["anchor_id", "anchor_name", "task", "training_seed", "batch"]
    fields = common + sorted({k for row in rows for k in row} - set(common))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", "--results-dir", dest="results_dir", type=Path, default=HERE / "results")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=HERE)
    parser.add_argument("--expected", type=int, help="optional required number of valid completed reports")
    args = parser.parse_args()
    if args.expected is not None and args.expected < 1:
        parser.error("--expected must be positive")
    counterfactuals, radii, geometries, anchors, ignored, invalid = [], [], [], [], [], []
    identities = set()
    for path in sorted(args.results_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text())
            if report.get("status") != "completed":
                ignored.append({"path": str(path), "status": report.get("status"), "error": report.get("error")})
                continue
            validate(report)
            meta = anchor_metadata(report, path)
            if meta["anchor_id"] in identities:
                raise ValueError("duplicate anchor report; refusing to count repeated execution")
            cf, rs, gg = tables_for(report, meta)
            identities.add(meta["anchor_id"])
            counterfactuals.extend(cf)
            radii.extend(rs)
            geometries.extend(gg)
            anchors.append({**meta, "batch_count": len(report["batches"]),
                            "checkpoint_hashes": report["checkpoint_hashes"],
                            "source_hashes": report.get("source_hashes", {}),
                            "arguments": report["arguments"], "loss_before": report["loss_before"],
                            "all_matches_successful": report.get("all_matches_successful"),
                            "failed_matches": [{"batch": b["batch"], "label": c["label"], "match": c["match"]}
                                               for b in report["batches"] for c in b["counterfactuals"] if not c["match"].get("ok")],
                            "final_restore_check": report["final_restore_check"],
                            "batches": report["batches"]})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    enough = bool(anchors) and (args.expected is None or len(anchors) == args.expected)
    status = "invalid" if invalid else ("completed" if enough and not ignored else "partial")
    summary = {"schema_version": 1, "status": status, "results_dir": str(args.results_dir.resolve()),
               "expected_reports": args.expected, "completed_reports": len(anchors),
               "training_seeds": sorted({a["training_seed"] for a in anchors}),
               "counterfactual_rows": len(counterfactuals), "radius_scan_rows": len(radii),
               "geometry_rows": len(geometries), "ignored_reports": ignored, "invalid_reports": invalid,
               "interpretation": ["All rows retain checkpoint anchor, task, and diagnostic batch identity.",
                                  "No averaging across anchors and no confidence intervals over batches are computed.",
                                  "Batches are paired observations within a single training seed, not independent training seeds.",
                                  "Delta CE is after minus before; positive values mean increased loss.",
                                  "Old-task macro CE gives each old task equal weight; all-seen and task-local CE stay separate.",
                                  "Unmatched candidates retain reasons and have no fabricated loss-change values.",
                                  "Positive scale-only and half-sum controls may collapse to the same matched update as their collinear directions."],
               "anchors": anchors}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "counterfactuals.csv", counterfactuals)
    write_csv(args.output_dir / "radius_scan.csv", radii)
    write_csv(args.output_dir / "geometry.csv", geometries)
    print(json.dumps({k: summary[k] for k in ("status", "completed_reports", "counterfactual_rows", "radius_scan_rows", "geometry_rows")}))
    return 1 if invalid else (0 if status == "completed" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
