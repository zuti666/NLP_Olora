#!/usr/bin/env python3
"""Verify the distinct adapter/sidecar invariants of IncLoRA and SeqLoRA."""

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


def _load_sidecar(adapter_dir: Path) -> dict:
    sidecar = torch.load(adapter_dir / "ewclora_fisher.pt", map_location="cpu")
    if not isinstance(sidecar, dict):
        raise ValueError(f"Invalid Fisher sidecar: {adapter_dir}")
    return sidecar


def _all_finite(values) -> bool:
    return all(
        not (value.is_floating_point() or value.is_complex())
        or bool(torch.isfinite(value).all().item())
        for value in values
        if torch.is_tensor(value)
    )


def _state_sha256(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _saved_delta_shapes(state: Dict[str, torch.Tensor]) -> dict:
    a_shapes = {}
    b_shapes = {}
    for name, value in state.items():
        if name.endswith("lora_A.weight"):
            a_shapes[name[: -len("lora_A.weight")]] = tuple(value.shape)
        elif name.endswith("lora_B.weight"):
            b_shapes[name[: -len("lora_B.weight")]] = tuple(value.shape)
    if set(a_shapes) != set(b_shapes):
        raise ValueError(
            "Saved adapter has unmatched lora_A/lora_B modules: "
            f"A-only={sorted(set(a_shapes) - set(b_shapes))}, "
            f"B-only={sorted(set(b_shapes) - set(a_shapes))}."
        )
    expected = {}
    for prefix, a_shape in a_shapes.items():
        b_shape = b_shapes[prefix]
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
            raise ValueError(
                f"Invalid saved LoRA factor shapes for {prefix}: "
                f"A={a_shape}, B={b_shape}."
            )
        expected[prefix] = (b_shape[0], a_shape[1])
    if not expected:
        raise ValueError("Saved adapter contains no lora_A/lora_B factor pairs.")
    return expected


def _verify_inclora(
    check,
    current_config: dict,
    current_state: Dict[str, torch.Tensor],
    previous_dir: Optional[Path],
) -> dict:
    rank = int(current_config.get("r", 0))
    rank_sum = int(current_config.get("r_sum", -1))
    check(current_config.get("save_lora_only", False) is False,
          "IncLoRA must use concat persistence, not save_lora_only.")
    check(current_config.get("save_loranew") is False,
          "IncLoRA concat persistence requires save_loranew=false.")
    prefix_exact = None
    previous_rank_sum = 0
    if previous_dir is None:
        check(rank_sum == rank, f"First IncLoRA r_sum={rank_sum}, expected r={rank}.")
    else:
        previous_config = _load_json(previous_dir / "adapter_config.json")
        previous_state = _load_state(previous_dir)
        previous_rank_sum = int(previous_config.get("r_sum", -1))
        check(set(previous_state) == set(current_state),
              "IncLoRA previous/current adapter key sets differ.")
        check(rank_sum == previous_rank_sum + rank,
              f"IncLoRA r_sum transition {previous_rank_sum}->{rank_sum} does not add rank {rank}.")
        loaded_prefix = {}
        for name, old_value in previous_state.items():
            new_value = current_state[name]
            if "lora_A" in name:
                check(tuple(new_value.shape) ==
                      (old_value.shape[0] + rank, old_value.shape[1]),
                      f"Unexpected concatenated A shape for {name}: {tuple(new_value.shape)}.")
                prefix = new_value[: old_value.shape[0], :]
            elif "lora_B" in name:
                check(tuple(new_value.shape) ==
                      (old_value.shape[0], old_value.shape[1] + rank),
                      f"Unexpected concatenated B shape for {name}: {tuple(new_value.shape)}.")
                prefix = new_value[:, : old_value.shape[1]]
            else:
                prefix = new_value
            loaded_prefix[name] = prefix.detach().clone()
            check(torch.equal(prefix, old_value),
                  f"IncLoRA historical block changed for {name}.")
        prefix_exact = _state_sha256(previous_state) == _state_sha256(loaded_prefix)
        check(prefix_exact, "IncLoRA historical-prefix SHA256 mismatch.")
    return {
        "rank": rank,
        "previous_rank_sum": previous_rank_sum,
        "current_rank_sum": rank_sum,
        "historical_prefix_exact": prefix_exact,
    }


def _verify_seqlora(
    check,
    current_config: dict,
    current_state: Dict[str, torch.Tensor],
    current_sidecar: dict,
    previous_dir: Optional[Path],
) -> dict:
    rank = int(current_config.get("r", 0))
    rank_sum = int(current_config.get("r_sum", -1))
    check(current_config.get("save_lora_only") is True,
          "SeqLoRA fixed-rank persistence requires save_lora_only=true.")
    check(rank_sum == rank,
          f"SeqLoRA must keep fixed rank; got r={rank}, r_sum={rank_sum}.")
    check(current_sidecar.get("format") == "full_delta_w_fisher_v3",
          "SeqLoRA sidecar must use full_delta_w_fisher_v3.")
    check(current_sidecar.get("adapter_update_space") == "sequential_shared",
          "SeqLoRA sidecar has the wrong adapter_update_space.")
    check(current_sidecar.get("anchor_mode") == "task_start_effective_delta_w",
          "SeqLoRA sidecar has the wrong anchor_mode.")
    check(current_sidecar.get("reference_encoding") == "lora_factors_v1",
          "SeqLoRA sidecar has the wrong reference encoding.")

    reference = current_sidecar.get("reference", {})
    check(isinstance(reference, dict) and bool(reference),
          "SeqLoRA sidecar has no compact factor reference.")
    if isinstance(reference, dict):
        check(set(reference) == set(current_sidecar.get("omega", {})),
              "SeqLoRA reference and Fisher module sets differ.")
        for prefix, factors in reference.items():
            if not isinstance(factors, dict):
                check(False, f"Invalid SeqLoRA reference block {prefix}.")
                continue
            a_key = f"{prefix}lora_A.weight"
            b_key = f"{prefix}lora_B.weight"
            check(a_key in current_state and b_key in current_state,
                  f"SeqLoRA reference has no matching saved factors for {prefix}.")
            if a_key in current_state and b_key in current_state:
                a_ref = factors.get("a")
                b_ref = factors.get("b")
                check(torch.is_tensor(a_ref) and torch.is_tensor(b_ref),
                      f"SeqLoRA reference block {prefix} lacks tensor A/B factors.")
                if torch.is_tensor(a_ref) and torch.is_tensor(b_ref):
                    check(torch.equal(a_ref, current_state[a_key].float()),
                          f"SeqLoRA saved A reference does not match {a_key}.")
                    check(torch.equal(b_ref, current_state[b_key].float()),
                          f"SeqLoRA saved B reference does not match {b_key}.")

    shape_fixed = None
    if previous_dir is not None:
        previous_config = _load_json(previous_dir / "adapter_config.json")
        previous_state = _load_state(previous_dir)
        check(set(previous_state) == set(current_state),
              "SeqLoRA previous/current adapter key sets differ.")
        shape_fixed = all(
            tuple(previous_state[name].shape) == tuple(current_state[name].shape)
            for name in previous_state.keys() & current_state.keys()
        )
        check(shape_fixed, "SeqLoRA adapter rank/shape grew between tasks.")
        check(int(previous_config.get("r_sum", -1)) == rank_sum,
              "SeqLoRA r_sum changed between tasks.")
    return {
        "rank": rank,
        "current_rank_sum": rank_sum,
        "fixed_shapes_across_tasks": shape_fixed,
        "reference_block_count": len(reference) if isinstance(reference, dict) else 0,
    }


def verify(
    current_dir: Path,
    previous_dir: Optional[Path],
    method: str,
) -> dict:
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    current_config = _load_json(current_dir / "adapter_config.json")
    current_state = _load_state(current_dir)
    current_sidecar = _load_sidecar(current_dir)
    omega = current_sidecar.get("omega", {})
    expected_delta_shapes = _saved_delta_shapes(current_state)

    check(not any("loranew_" in key for key in current_state),
          "Saved adapter unexpectedly contains loranew_* keys.")
    check(_all_finite(current_state.values()), "Adapter contains non-finite tensors.")
    check(isinstance(omega, dict) and bool(omega), "Fisher sidecar has no omega blocks.")
    if isinstance(omega, dict):
        check(_all_finite(omega.values()), "Fisher contains non-finite tensors.")
        check(all(bool((value >= 0).all().item()) for value in omega.values()),
              "Fisher contains negative entries.")
        check(set(omega) == set(expected_delta_shapes),
              "Fisher module set does not exactly cover the saved adapter.")
        for prefix, expected_shape in expected_delta_shapes.items():
            value = omega.get(prefix)
            check(torch.is_tensor(value), f"Missing Fisher tensor for {prefix}.")
            if torch.is_tensor(value):
                check(tuple(value.shape) == expected_shape,
                      f"Fisher shape mismatch for {prefix}: "
                      f"got {tuple(value.shape)}, expected {expected_shape}.")

    current_count = int(current_sidecar.get("count_updates", -1))
    current_lambda = float(current_sidecar.get("lambda", math.nan))
    check(math.isfinite(current_lambda) and current_lambda >= 0.0,
          f"Invalid EWC lambda in sidecar: {current_lambda}.")
    if previous_dir is None:
        check(current_count == 1,
              f"First Fisher count_updates={current_count}, expected 1.")
    else:
        previous_sidecar = _load_sidecar(previous_dir)
        previous_count = int(previous_sidecar.get("count_updates", -1))
        check(current_count == previous_count + 1,
              f"Fisher count transition {previous_count}->{current_count}, expected +1.")

    if method == "inclora_sam_ewc":
        details = _verify_inclora(
            check, current_config, current_state, previous_dir
        )
        check(current_sidecar.get("format") == "full_delta_w_fisher_v2",
              "IncLoRA FisherShape sidecar must use full_delta_w_fisher_v2.")
    else:
        details = _verify_seqlora(
            check, current_config, current_state, current_sidecar, previous_dir
        )

    return {
        "status": "pass" if not errors else "fail",
        "method": method,
        "current_adapter": str(current_dir),
        "previous_adapter": None if previous_dir is None else str(previous_dir),
        "state_tensor_count": len(current_state),
        "loranew_tensor_count": sum("loranew_" in key for key in current_state),
        "adapter_sha256": _state_sha256(current_state),
        "fisher_block_count": len(omega) if isinstance(omega, dict) else 0,
        "fisher_count_updates": current_count,
        "fisher_lambda": current_lambda,
        **details,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument(
        "--method",
        required=True,
        choices=("inclora_sam_ewc", "seqlora_sam_ewc"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.current / "sam_ewc_chain_verification.json"
    try:
        report = verify(args.current, args.previous, args.method)
    except Exception as exc:
        report = {
            "status": "fail",
            "method": args.method,
            "current_adapter": str(args.current),
            "previous_adapter": None if args.previous is None else str(args.previous),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
