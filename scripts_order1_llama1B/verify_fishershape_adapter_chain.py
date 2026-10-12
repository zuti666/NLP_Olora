#!/usr/bin/env python3
"""Verify that a saved concat adapter is preserved by the next task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_state(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(adapter_dir / "adapter_model.bin", map_location="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Invalid or empty adapter state: {adapter_dir}")
    return state


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def _state_sha256(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _all_finite(state: Dict[str, torch.Tensor]) -> bool:
    return all(
        not (value.is_floating_point() or value.is_complex())
        or bool(torch.isfinite(value).all().item())
        for value in state.values()
    )


def _load_sidecar(adapter_dir: Path) -> dict:
    payload = torch.load(adapter_dir / "ewclora_fisher.pt", map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Fisher sidecar: {adapter_dir}")
    return payload


def verify(current_dir: Path, previous_dir: Optional[Path], mode: str) -> dict:
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    current_config = _load_json(current_dir / "adapter_config.json")
    current_state = _load_state(current_dir)
    current_sidecar = _load_sidecar(current_dir)
    current_omega = current_sidecar.get("omega", {})

    check(not any("loranew_" in key for key in current_state),
          "Saved adapter unexpectedly contains loranew_* keys.")
    check(_all_finite(current_state), "Current adapter contains non-finite tensors.")
    check(isinstance(current_omega, dict) and bool(current_omega),
          "Current Fisher sidecar has no omega tensors.")
    if isinstance(current_omega, dict):
        check(_all_finite(current_omega), "Current Fisher contains non-finite tensors.")
        check(all(bool((value >= 0).all().item()) for value in current_omega.values()),
              "Current Fisher contains negative entries.")

    current_rank = int(current_config.get("r", 0))
    current_rank_sum = int(current_config.get("r_sum", -1))
    current_count = int(current_sidecar.get("count_updates", -1))
    current_lambda = float(current_sidecar.get("lambda", math.nan))
    check(current_rank > 0, f"Invalid current LoRA rank: {current_rank}.")
    check(current_config.get("save_loranew") is False,
          "save_loranew must be false for concat persistence.")

    previous_hash = None
    loaded_prefix_hash = None
    prefix_exact = None
    previous_rank_sum = 0

    if previous_dir is None:
        check(current_rank_sum == current_rank,
              f"First adapter r_sum={current_rank_sum}, expected r={current_rank}.")
        check(current_count == 1,
              f"First Fisher count_updates={current_count}, expected 1.")
    else:
        previous_config = _load_json(previous_dir / "adapter_config.json")
        previous_state = _load_state(previous_dir)
        previous_sidecar = _load_sidecar(previous_dir)
        previous_rank_sum = int(previous_config.get("r_sum", -1))
        previous_count = int(previous_sidecar.get("count_updates", -1))

        check(set(previous_state) == set(current_state),
              "Previous/current adapter key sets differ.")
        check(current_rank_sum == previous_rank_sum + current_rank,
              f"r_sum transition {previous_rank_sum}->{current_rank_sum} does not add rank {current_rank}.")

        loaded_prefix = {}
        for name in sorted(previous_state):
            previous_value = previous_state[name]
            current_value = current_state.get(name)
            if current_value is None:
                continue
            if "lora_A" in name:
                check(current_value.ndim == 2 and previous_value.ndim == 2,
                      f"Unexpected A rank for {name}.")
                check(tuple(current_value.shape) ==
                      (previous_value.shape[0] + current_rank, previous_value.shape[1]),
                      f"Unexpected concatenated A shape for {name}: {tuple(current_value.shape)}.")
                prefix = current_value[: previous_value.shape[0], :]
            elif "lora_B" in name:
                check(current_value.ndim == 2 and previous_value.ndim == 2,
                      f"Unexpected B rank for {name}.")
                check(tuple(current_value.shape) ==
                      (previous_value.shape[0], previous_value.shape[1] + current_rank),
                      f"Unexpected concatenated B shape for {name}: {tuple(current_value.shape)}.")
                prefix = current_value[:, : previous_value.shape[1]]
            else:
                prefix = current_value
            loaded_prefix[name] = prefix.detach().clone()
            check(torch.equal(prefix, previous_value),
                  f"Loaded historical block changed for {name}.")

        previous_hash = _state_sha256(previous_state)
        loaded_prefix_hash = _state_sha256(loaded_prefix)
        prefix_exact = previous_hash == loaded_prefix_hash
        check(prefix_exact, "Historical-prefix SHA256 does not match previous adapter.")

        if mode in {"sam_ewc", "gamfs_sam"}:
            check(current_count == previous_count + 1,
                  f"EWC count transition {previous_count}->{current_count}, expected +1.")
        else:
            check(current_count == 1,
                  f"EWC-off sidecar count_updates={current_count}, expected 1.")

    if mode in {"sam", "gamfs_sam"}:
        check(current_lambda == 0.0,
              f"SAM-equivalent sidecar lambda={current_lambda}, expected 0.")
    else:
        check(math.isfinite(current_lambda) and current_lambda > 0.0,
              f"SAM+EWC sidecar lambda={current_lambda}, expected positive.")

    report = {
        "status": "pass" if not errors else "fail",
        "mode": mode,
        "current_adapter": str(current_dir),
        "previous_adapter": None if previous_dir is None else str(previous_dir),
        "r": current_rank,
        "previous_r_sum": previous_rank_sum,
        "current_r_sum": current_rank_sum,
        "state_tensor_count": len(current_state),
        "loranew_tensor_count": sum("loranew_" in key for key in current_state),
        "adapter_finite": _all_finite(current_state),
        "fisher_tensor_count": len(current_omega) if isinstance(current_omega, dict) else 0,
        "fisher_count_updates": current_count,
        "fisher_lambda": current_lambda,
        "previous_state_sha256": previous_hash,
        "loaded_prefix_sha256": loaded_prefix_hash,
        "historical_prefix_exact": prefix_exact,
        "current_state_sha256": _state_sha256(current_state),
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("sam", "sam_ewc", "gamfs_sam"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.current / "chain_verification.json"
    try:
        report = verify(args.current, args.previous, args.mode)
    except Exception as exc:
        report = {
            "status": "fail",
            "mode": args.mode,
            "current_adapter": str(args.current),
            "previous_adapter": None if args.previous is None else str(args.previous),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
