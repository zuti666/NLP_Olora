#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, Iterable, List


def _infer_optimizer(path: str) -> str:
    lowered = path.lower().replace(os.sep, "/")
    for part in reversed(lowered.split("/")):
        if part == "sgd" or part.startswith("sgd_") or part.startswith("s1c_sgd_"):
            return "sgd"
        if part == "gam" or part.startswith("gam_") or part.startswith("s1c_gam_"):
            return "gam"
    if "/sgd_" in lowered or "_sgd_" in lowered:
        return "sgd"
    if "/gam_" in lowered or "_gam_" in lowered:
        return "gam"
    return ""


def _iter_rows(inputs: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for root in inputs:
        for path in sorted(glob.glob(os.path.join(root, "**", "*_cl_metrics.json"), recursive=True)):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            steps = data.get("task0_trajectory", {}).get("steps", {})
            if not isinstance(steps, dict):
                continue
            for step, metrics in sorted(steps.items(), key=lambda item: int(item[0])):
                if not isinstance(metrics, dict):
                    continue
                row: Dict[str, object] = {
                    "optimizer": _infer_optimizer(path),
                    "step": int(step),
                    "metrics_json": path,
                }
                row.update(metrics)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task0 trajectory metrics from CL metrics JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Root directory/directories to scan.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    rows = _iter_rows(args.inputs)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fields = ["optimizer", "step", "metrics_json"]
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {len(rows)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
