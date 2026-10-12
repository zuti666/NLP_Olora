#!/usr/bin/env python3
"""Check a completed IncLoRA+EWC stage before accepting or reusing it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_torch(path: Path) -> dict:
    # These locally generated checkpoints contain tensors and primitive values.
    # weights_only is unavailable in some of the older training environments.
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected a nonempty checkpoint dictionary: {path}")
    return value


def normalize_prefix(prefix: str) -> str:
    while prefix.startswith("module."):
        prefix = prefix[len("module."):]
    return prefix


def tensor_finite(value) -> bool:
    return torch.is_tensor(value) and bool(torch.isfinite(value).all().item())


def numeric_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify(current: Path, stage: int, previous: Optional[Path]) -> dict:
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(stage >= 1, "stage must be at least 1.")
    check(stage != 1 or previous is None, "Stage 1 must not have a previous adapter.")
    config = load_json(current / "adapter_config.json")
    state = load_torch(current / "adapter_model.bin")
    sidecar = load_torch(current / "ewclora_fisher.pt")
    train = load_json(current.parent / "train_results.json")
    predict = load_json(current.parent / "predict_results.json")

    check(config.get("r") == 8, f"Expected r=8, got {config.get('r')!r}.")
    check(config.get("r_sum") == 8 * stage,
          f"Expected r_sum={8 * stage}, got {config.get('r_sum')!r}.")
    check(config.get("save_loranew") is False, "Expected save_loranew=false for concat persistence.")
    check(not any("loranew_" in name for name in state), "Saved state still contains loranew_* keys.")
    adapter_finite = all(tensor_finite(value) for value in state.values())
    check(adapter_finite, "Adapter contains non-tensor or non-finite values.")

    pairs = {}
    for name, value in state.items():
        for part in ("A", "B"):
            marker = f".lora_{part}."
            if marker not in name:
                continue
            prefix = normalize_prefix(name.split(marker, 1)[0] + ".")
            check(part not in pairs.setdefault(prefix, {}), f"Duplicate LoRA {part} for {prefix}.")
            pairs[prefix][part] = value
            check(torch.is_tensor(value) and value.ndim == 2, f"LoRA {part} is not a matrix: {name}.")
            if torch.is_tensor(value) and value.ndim == 2:
                rank = value.shape[0 if part == "A" else 1]
                check(rank == 8 * stage, f"Unexpected rank {rank} for {name}; expected {8 * stage}.")
    check(bool(pairs), "No saved LoRA A/B pairs found.")
    expected_shapes = {}
    for prefix, pair in pairs.items():
        check(set(pair) == {"A", "B"}, f"Incomplete LoRA pair for {prefix}.")
        if set(pair) == {"A", "B"} and all(torch.is_tensor(v) and v.ndim == 2 for v in pair.values()):
            expected_shapes[prefix] = (pair["B"].shape[0], pair["A"].shape[1])

    omega = sidecar.get("omega")
    if isinstance(omega, list):
        names = sidecar.get("names", [])
        check(len(names) == len(omega), "Named-list Fisher has mismatched names and tensors.")
        omega = dict(zip(names, omega))
    check(isinstance(omega, dict) and bool(omega), "Fisher omega must be nonempty.")
    omega = omega if isinstance(omega, dict) else {}
    normalized_omega = {normalize_prefix(str(name)): value for name, value in omega.items()}
    check(len(normalized_omega) == len(omega), "Duplicate Fisher prefixes after normalization.")
    check(set(normalized_omega) == set(expected_shapes),
          "Fisher/adapter coverage mismatch: "
          f"missing={sorted(set(expected_shapes) - set(normalized_omega))}, "
          f"extra={sorted(set(normalized_omega) - set(expected_shapes))}.")
    fisher_finite = all(tensor_finite(value) for value in normalized_omega.values())
    check(fisher_finite, "Fisher contains non-tensor or non-finite values.")
    for prefix, value in normalized_omega.items():
        if not torch.is_tensor(value):
            continue
        check(not value.is_complex() and bool((value >= 0).all().item()),
              f"Fisher must be real and nonnegative: {prefix}.")
        if prefix in expected_shapes:
            check(tuple(value.shape) == expected_shapes[prefix],
                  f"Fisher shape for {prefix} is {tuple(value.shape)}, expected {expected_shapes[prefix]} [out,in].")
    check(sidecar.get("format") in {"full_delta_w_fisher_v1", "full_delta_w_fisher_v2"},
          f"Expected v1/v2 full-Delta-W Fisher, got {sidecar.get('format')!r}.")
    check(sidecar.get("count_updates") == stage,
          f"Expected count_updates={stage}, got {sidecar.get('count_updates')!r}.")

    prefix_exact = None
    checked_prefix_count = 0
    if previous is not None:
        old_config = load_json(previous / "adapter_config.json")
        old_state = load_torch(previous / "adapter_model.bin")
        check(old_config.get("r") == 8 and old_config.get("r_sum") == 8 * (stage - 1),
              "Previous adapter rank does not correspond to stage-1.")
        prefix_exact = set(state) == set(old_state)
        check(prefix_exact, "Previous/current adapter keys differ.")
        for name, old_value in old_state.items():
            value = state.get(name)
            if not (torch.is_tensor(value) and torch.is_tensor(old_value)):
                prefix_exact = False
                check(False, f"Missing or non-tensor historical parameter: {name}.")
                continue
            if ".lora_A." in name and value.ndim == old_value.ndim == 2:
                prefix = value[:old_value.shape[0], :]
            elif ".lora_B." in name and value.ndim == old_value.ndim == 2:
                prefix = value[:, :old_value.shape[1]]
            else:
                prefix = value
            exact = prefix.dtype == old_value.dtype and torch.equal(prefix, old_value)
            prefix_exact = prefix_exact and exact
            check(exact, f"Historical parameter changed: {name}.")
            checked_prefix_count += 1

    check(numeric_finite(train.get("train_loss")), "Missing or non-finite train_loss.")
    check(numeric_finite(predict.get("predict_exact_match")), "Missing or non-finite predict_exact_match.")
    prediction_metrics = {name: value for name, value in predict.items() if name.startswith("predict_")}
    check(bool(prediction_metrics), "No prediction metrics present.")
    for name, value in prediction_metrics.items():
        check(numeric_finite(value), f"Prediction metric {name} is not finite numeric: {value!r}.")

    return {
        "status": "fail" if errors else "pass",
        "stage": stage,
        "current_adapter": str(current),
        "previous_adapter": None if previous is None else str(previous),
        "r": config.get("r"),
        "r_sum": config.get("r_sum"),
        "adapter_tensor_count": len(state),
        "adapter_finite": adapter_finite,
        "lora_pair_count": len(pairs),
        "historical_prefix_exact": prefix_exact,
        "historical_tensor_count_checked": checked_prefix_count,
        "fisher_format": sidecar.get("format"),
        "fisher_count_updates": sidecar.get("count_updates"),
        "fisher_tensor_count": len(omega),
        "fisher_finite": fisher_finite,
        "train_loss": train.get("train_loss") if numeric_finite(train.get("train_loss")) else None,
        "prediction_metrics": {name: value if numeric_finite(value) else None for name, value in prediction_metrics.items()},
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True, help="Completed stage's adapter directory")
    parser.add_argument("--stage", type=int, required=True, help="One-based task stage")
    parser.add_argument("--previous", type=Path, help="Previous stage's adapter for exact history checking")
    parser.add_argument("--output", type=Path, required=True, help="Write the validation report here")
    args = parser.parse_args()
    try:
        report = verify(args.current, args.stage, args.previous)
    except Exception as exc:
        report = {
            "status": "fail", "stage": args.stage, "current_adapter": str(args.current),
            "previous_adapter": None if args.previous is None else str(args.previous),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
