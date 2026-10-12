#!/usr/bin/env python3
"""Summarize CL metrics JSON files under outputs_logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRICS_GLOB = "*_cl_metrics.json"
PREFIX_KEY = "prefix:"
PREFERRED_CONFIG_PARTS = ("config_", "ref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect main config, config path, metrics JSON path, and final "
            "FAA/AAA from continual-learning metrics outputs."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs_logs"),
        help="Directory to search for *_cl_metrics.json files.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used for relative paths and YAML discovery.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "tsv"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output file. Defaults to stdout.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory to search for matching *.log files. May be passed more "
            "than once. Defaults to all top-level logs* directories."
        ),
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Print absolute config and metrics paths.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep JSON files that do not expose final FAA/AAA values.",
    )
    return parser.parse_args()


def read_prefix(config_path: Path) -> str | None:
    try:
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line.startswith(PREFIX_KEY):
                continue
            value = line[len(PREFIX_KEY) :].strip()
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            return value.strip("'\"")
    except OSError:
        return None
    return None


def discover_config_prefixes(repo_root: Path) -> dict[str, list[Path]]:
    prefix_to_paths: dict[str, list[Path]] = defaultdict(list)
    for config_path in sorted(repo_root.rglob("*.yaml")):
        if "outputs_logs" in config_path.parts:
            continue
        if ".git" in config_path.parts:
            continue
        prefix = read_prefix(config_path)
        if prefix:
            prefix_to_paths[prefix].append(config_path)
    return prefix_to_paths


def config_rank(path: Path) -> tuple[int, int, str]:
    text = path.as_posix()
    has_copy = " copy" in text
    preferred = any(part in text for part in PREFERRED_CONFIG_PARTS)
    return (1 if has_copy else 0, 0 if preferred else 1, text)


def choose_config_path(
    prefix_to_paths: dict[str, list[Path]], main_config: str
) -> Path | None:
    matches = prefix_to_paths.get(main_config, [])
    if not matches:
        return None
    return sorted(matches, key=config_rank)[0]


def strip_copy_suffix(text: str) -> str:
    return text.replace(" copy", "")


def strip_seed_suffix(text: str) -> str:
    return re.sub(r"_s\d+$", "", text)


def normalize_match_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def infer_main_config(metrics_path: Path) -> str:
    try:
        exp_run_dir = metrics_path.parents[2]
    except IndexError:
        exp_run_dir = None
    if exp_run_dir and exp_run_dir.name and exp_run_dir.name != "exp_run":
        return exp_run_dir.name

    stem = metrics_path.name.removesuffix("_cl_metrics.json")
    vit_marker = "_vit_"
    if vit_marker in stem:
        stem = stem.split(vit_marker, 1)[0]
    return stem


def discover_log_paths(repo_root: Path, logs_roots: list[Path]) -> list[Path]:
    if logs_roots:
        roots = [
            path if path.is_absolute() else repo_root / path
            for path in logs_roots
        ]
    else:
        roots = [
            path
            for path in repo_root.iterdir()
            if path.is_dir() and path.name.startswith("logs")
        ]

    log_paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for log_path in root.rglob("*.log"):
            if ".git" in log_path.parts:
                continue
            resolved = log_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            log_paths.append(log_path)
    return sorted(log_paths)


def log_hint_tokens(metrics_path: Path) -> set[str]:
    hints: set[str] = set()
    parts = [part for part in metrics_path.parts if part]
    for part in parts:
        lower = part.lower()
        if lower.startswith("config_"):
            suffix = lower[len("config_") :]
            hints.add(f"logs_{suffix}")
            if suffix == "newdata":
                hints.add("logs_newdata")
            if suffix == "expand":
                hints.add("logs_expand")
        if lower.startswith(("imageneta_", "imagenetr_", "domainnet_", "cifar")):
            hints.add(lower)
        if lower.startswith("ewclora_"):
            hints.add(lower)
    return hints


def log_candidates(
    metrics_path: Path,
    config_path: Path | None,
    main_config: str,
) -> list[str]:
    candidates: list[str] = []
    if config_path is not None:
        candidates.append(strip_copy_suffix(config_path.stem))

    metric_stem = metrics_path.name.removesuffix("_cl_metrics.json")
    if "_vit_" in metric_stem:
        metric_stem = metric_stem.split("_vit_", 1)[0]

    candidates.extend(
        [
            main_config,
            strip_seed_suffix(main_config),
            metric_stem,
            strip_seed_suffix(metric_stem),
        ]
    )

    expanded: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        expanded.append(candidate)
        expanded.append(candidate.replace("as1normfisher", "as1_normfisher"))
        expanded.append(candidate.replace("newdata_stage1_", ""))
        expanded.append(candidate.replace("newdata_stage2_", ""))
        expanded.append(candidate.replace("newdata_stage3_largeewc_", ""))
        expanded.append(candidate.replace("newdata_epoch30_anchor_faa_anchor_", ""))
        expanded.append(candidate.replace("newdata_epoch30_anchor_aaa_anchor_", ""))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in expanded:
        candidate = strip_copy_suffix(strip_seed_suffix(candidate))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def score_log_path(
    log_path: Path,
    candidates: list[str],
    hints: set[str],
) -> tuple[int, str] | None:
    log_stem = strip_copy_suffix(log_path.stem)
    log_base = re.sub(r"_gpu\d+$", "", log_stem)
    norm_log = normalize_match_key(log_base)
    best: int | None = None

    for candidate in candidates:
        candidate_base = strip_copy_suffix(strip_seed_suffix(candidate))
        norm_candidate = normalize_match_key(candidate_base)
        if not candidate_base or not norm_candidate:
            continue
        if log_base == candidate_base:
            score = 0
        elif log_stem.startswith(f"{candidate_base}_gpu"):
            score = 1
        elif log_base.endswith(candidate_base):
            score = 4
        elif candidate_base in log_base:
            score = 8
        elif norm_log == norm_candidate:
            score = 12
        elif norm_candidate in norm_log:
            score = 16
        else:
            continue
        best = score if best is None else min(best, score)

    if best is None:
        return None

    norm_path = log_path.as_posix().lower()
    hint_bonus = sum(1 for hint in hints if hint and hint in norm_path)
    return (best - hint_bonus, log_path.as_posix())


def choose_log_path(
    log_paths: list[Path],
    metrics_path: Path,
    config_path: Path | None,
    main_config: str,
) -> Path | None:
    if not log_paths:
        return None

    candidates = log_candidates(metrics_path, config_path, main_config)
    hints = log_hint_tokens(metrics_path)
    scored: list[tuple[tuple[int, str], Path]] = []
    for log_path in log_paths:
        score = score_log_path(log_path, candidates, hints)
        if score is not None:
            scored.append((score, log_path))

    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0])[0][1]


def read_metric(metrics: dict, evaluator: str, key: str) -> float | None:
    value = metrics.get(evaluator, {}).get("final", {}).get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "NaN"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def path_text(path: Path | None, repo_root: Path, absolute: bool) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    if absolute:
        return str(resolved)
    try:
        return str(resolved.relative_to(repo_root.resolve()))
    except ValueError:
        return str(resolved)


def collect_rows(
    outputs_root: Path,
    repo_root: Path,
    logs_roots: list[Path],
    absolute_paths: bool,
    include_incomplete: bool,
) -> list[dict[str, str]]:
    prefix_to_paths = discover_config_prefixes(repo_root)
    log_paths = discover_log_paths(repo_root, logs_roots)
    rows: list[dict[str, str]] = []

    for metrics_path in sorted(outputs_root.rglob(METRICS_GLOB)):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        main_config = infer_main_config(metrics_path)
        config_path = choose_config_path(prefix_to_paths, main_config)
        log_path = choose_log_path(log_paths, metrics_path, config_path, main_config)
        row = {
            "main_config": main_config,
            "config_path": path_text(config_path, repo_root, absolute_paths),
            "log_path": path_text(log_path, repo_root, absolute_paths),
            "metrics_json_path": path_text(metrics_path, repo_root, absolute_paths),
            "cnn_FAA": format_number(read_metric(metrics, "cnn", "FAA")),
            "cnn_AAA": format_number(read_metric(metrics, "cnn", "AAA")),
            "nme_FAA": format_number(read_metric(metrics, "nme", "FAA")),
            "nme_AAA": format_number(read_metric(metrics, "nme", "AAA")),
        }
        has_final_metrics = any(
            row[name] for name in ("cnn_FAA", "cnn_AAA", "nme_FAA", "nme_AAA")
        )
        if include_incomplete or has_final_metrics:
            rows.append(row)

    return rows


def write_delimited(
    rows: Iterable[dict[str, str]], fieldnames: list[str], delimiter: str, stream
) -> None:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], fieldnames: list[str], stream) -> None:
    stream.write("| " + " | ".join(fieldnames) + " |\n")
    stream.write("| " + " | ".join("---" for _ in fieldnames) + " |\n")
    for row in rows:
        stream.write("| " + " | ".join(row[name] for name in fieldnames) + " |\n")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    outputs_root = args.outputs_root
    if not outputs_root.is_absolute():
        outputs_root = repo_root / outputs_root

    rows = collect_rows(
        outputs_root,
        repo_root,
        args.logs_root,
        args.absolute_paths,
        args.include_incomplete,
    )
    fieldnames = [
        "main_config",
        "config_path",
        "log_path",
        "metrics_json_path",
        "cnn_FAA",
        "cnn_AAA",
        "nme_FAA",
        "nme_AAA",
    ]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", encoding="utf-8", newline="")
        should_close = True
    else:
        stream = sys.stdout
        should_close = False

    try:
        if args.format == "markdown":
            write_markdown(rows, fieldnames, stream)
        elif args.format == "csv":
            write_delimited(rows, fieldnames, ",", stream)
        else:
            write_delimited(rows, fieldnames, "\t", stream)
    finally:
        if should_close:
            stream.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
