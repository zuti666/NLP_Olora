"""Pure helpers for the ImageNet-R endpoint-region diagnostic.

The experiment-facing runner lives in ``scripts/``.  This module contains the
small deterministic pieces that can be unit tested without loading a vision
model or touching a GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


_LORA_STATE_KEY = re.compile(
    r"^backbone\.lora_vit\.blocks\.(\d+)\.attn\.qkv\.linear_([ab])_([qv])\.weight$"
)


def file_sha256(path: str | Path) -> str:
    """Hash one immutable experiment artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_resume_artifact_paths(
    checkpoint_dir: str | Path, task_idx: int
) -> dict[str, Path]:
    """Return every Task checkpoint artifact used by RQ3 diagnostics/resume."""
    root = Path(checkpoint_dir)
    task_idx = int(task_idx)
    return {
        "task_checkpoint": root / f"task_{task_idx:02d}.pt",
        "fisher_sidecar": root / f"fisher_state_{task_idx}.pt",
        "lora_a_sidecar": root / f"lora_w_a_{task_idx}.pt",
        "lora_b_sidecar": root / f"lora_w_b_{task_idx}.pt",
        "classifier_sidecar": root / f"fc_state_{task_idx}.pt",
    }


def task_resume_artifact_sha256(
    checkpoint_dir: str | Path, task_idx: int
) -> dict[str, str]:
    """Hash the complete RQ3 endpoint, including the files Stage 2 loads."""
    paths = task_resume_artifact_paths(checkpoint_dir, task_idx)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required task resume artifacts are missing: {missing}")
    return {name: file_sha256(path) for name, path in paths.items()}


def _saved_tensor(value, *, source: str) -> torch.Tensor:
    tensor = value.weight if hasattr(value, "weight") else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{source} is not a tensor")
    return tensor.detach().cpu()


def _assert_same_tensor(actual, expected: torch.Tensor, *, source: str) -> None:
    actual_tensor = _saved_tensor(actual, source=source)
    expected_tensor = expected.detach().cpu()
    if (
        actual_tensor.dtype != expected_tensor.dtype
        or tuple(actual_tensor.shape) != tuple(expected_tensor.shape)
        or not torch.equal(actual_tensor, expected_tensor)
    ):
        raise ValueError(
            f"resume sidecar tensor differs from task checkpoint network_state: {source}"
        )


def validate_task_resume_sidecars(
    checkpoint_dir: str | Path, task_idx: int
) -> dict[str, str]:
    """Prove the actual Stage-2 restore files encode the diagnosed endpoint.

    The RQ3 continuation restores LoRA A/B factors and the classifier from
    sidecars, while the offline diagnostic restores ``task_XX.pt``.  This
    validation makes those two restore routes an exact, fail-closed identity.
    It is intentionally specific to the frozen ViT q/v LoRA architecture.
    """
    paths = task_resume_artifact_paths(checkpoint_dir, task_idx)
    hashes = task_resume_artifact_sha256(checkpoint_dir, task_idx)
    checkpoint = torch.load(
        paths["task_checkpoint"], map_location="cpu", weights_only=False
    )
    network_state = checkpoint.get("network_state")
    if not isinstance(network_state, Mapping):
        raise ValueError("task checkpoint has no network_state mapping")

    indexed: dict[str, list[tuple[tuple[int, int], str, torch.Tensor]]] = {
        "a": [],
        "b": [],
    }
    channel_order = {"q": 0, "v": 1}
    for name, value in network_state.items():
        match = _LORA_STATE_KEY.fullmatch(str(name))
        if match is None:
            continue
        family = match.group(2)
        indexed[family].append(
            ((int(match.group(1)), channel_order[match.group(3)]), str(name), value)
        )
    for family in indexed:
        indexed[family].sort(key=lambda item: item[0])

    saved_as = torch.load(
        paths["lora_a_sidecar"], map_location="cpu", weights_only=False
    )
    saved_bs = torch.load(
        paths["lora_b_sidecar"], map_location="cpu", weights_only=False
    )
    if not isinstance(saved_as, (list, tuple)) or not isinstance(
        saved_bs, (list, tuple)
    ):
        raise TypeError("LoRA resume sidecars must contain tensor sequences")
    if not indexed["a"] or len(indexed["a"]) != len(saved_as):
        raise ValueError(
            "LoRA-A sidecar/checkpoint branch count mismatch: "
            f"checkpoint={len(indexed['a'])}, sidecar={len(saved_as)}"
        )
    if not indexed["b"] or len(indexed["b"]) != len(saved_bs):
        raise ValueError(
            "LoRA-B sidecar/checkpoint branch count mismatch: "
            f"checkpoint={len(indexed['b'])}, sidecar={len(saved_bs)}"
        )
    if [item[0] for item in indexed["a"]] != [item[0] for item in indexed["b"]]:
        raise ValueError("LoRA A/B checkpoint branch identities differ")
    for position, (saved, (_, name, expected)) in enumerate(zip(saved_as, indexed["a"])):
        _assert_same_tensor(saved, expected, source=f"lora_a[{position}]/{name}")
    for position, (saved, (_, name, expected)) in enumerate(zip(saved_bs, indexed["b"])):
        _assert_same_tensor(saved, expected, source=f"lora_b[{position}]/{name}")

    fc_state = torch.load(
        paths["classifier_sidecar"], map_location="cpu", weights_only=False
    )
    if not isinstance(fc_state, Mapping):
        raise TypeError("classifier resume sidecar must contain a state mapping")
    expected_fc = {
        str(name)[len("fc.") :]: value
        for name, value in network_state.items()
        if str(name).startswith("fc.")
    }
    if not expected_fc or set(fc_state) != set(expected_fc):
        raise ValueError(
            "classifier sidecar/checkpoint keys mismatch: "
            f"checkpoint={sorted(expected_fc)}, sidecar={sorted(fc_state)}"
        )
    for name, expected in expected_fc.items():
        _assert_same_tensor(fc_state[name], expected, source=f"classifier/{name}")
    return hashes


def parse_positive_floats(value: str | Sequence[float]) -> list[float]:
    """Parse a strictly increasing sequence of finite positive floats."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        values = [float(item) for item in items]
    else:
        values = [float(item) for item in value]
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("expected one or more finite positive values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("values must be strictly increasing")
    return values


def parameter_schema(named_parameters: Sequence[tuple[str, torch.Tensor]]) -> list[dict]:
    return [
        {
            "name": str(name),
            "shape": [int(item) for item in parameter.shape],
            "dtype": str(parameter.dtype),
            "numel": int(parameter.numel()),
        }
        for name, parameter in named_parameters
    ]


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def balanced_prefix_indices(labels: Sequence[int], per_class: int) -> list[int]:
    """Return the first ``per_class`` positions for every sorted class."""
    per_class = int(per_class)
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(int(index))
    if not grouped:
        raise ValueError("labels must be non-empty")
    selected: list[int] = []
    for label in sorted(grouped):
        if len(grouped[label]) < per_class:
            raise ValueError(
                f"class {label} has {len(grouped[label])} samples, needs {per_class}"
            )
        selected.extend(grouped[label][:per_class])
    return selected


def fixed_batch_indices(
    sample_count: int,
    batch_count: int,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """Draw non-overlapping, reproducible mini-batches from a fixed dataset."""
    sample_count = int(sample_count)
    batch_count = int(batch_count)
    batch_size = int(batch_size)
    needed = batch_count * batch_size
    if sample_count <= 0 or batch_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count, batch_count, and batch_size must be positive")
    if needed > sample_count:
        raise ValueError(f"need {needed} samples but dataset has only {sample_count}")
    permutation = np.random.default_rng(int(seed)).permutation(sample_count)[:needed]
    return [
        [int(item) for item in permutation[start : start + batch_size]]
        for start in range(0, needed, batch_size)
    ]


def random_unit_directions(dim: int, count: int, seed: int) -> torch.Tensor:
    """Generate a deterministic CPU float32 direction bank."""
    dim = int(dim)
    count = int(count)
    if dim <= 0 or count <= 0:
        raise ValueError("dim and count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    directions = torch.randn(count, dim, generator=generator, dtype=torch.float32)
    norms = directions.norm(dim=1, keepdim=True)
    if bool((norms <= 0).any()):
        raise RuntimeError("sampled a zero random direction")
    return directions / norms


def proxy_radii(trace: float, grad_norm: float, taus: Iterable[float]) -> dict[str, dict]:
    """Return quadratic-only and non-stationary trace-bound radii."""
    trace = float(trace)
    grad_norm = float(grad_norm)
    if not math.isfinite(trace) or trace < 0.0:
        raise ValueError("trace must be finite and non-negative")
    if not math.isfinite(grad_norm) or grad_norm < 0.0:
        raise ValueError("grad_norm must be finite and non-negative")
    output: dict[str, dict] = {}
    for tau_value in taus:
        tau = float(tau_value)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if trace == 0.0:
            fisher_radius = math.inf
            operational_radius = math.inf if grad_norm == 0.0 else tau / grad_norm
        else:
            fisher_radius = math.sqrt(2.0 * tau / trace)
            # Stable form of (-g + sqrt(g^2 + 2*tau*trace)) / trace.
            discriminant = math.sqrt(grad_norm * grad_norm + 2.0 * tau * trace)
            operational_radius = (2.0 * tau) / (grad_norm + discriminant)
        output[f"{tau:g}"] = {
            "tau": tau,
            "trace_quadratic_radius": fisher_radius,
            "gradient_trace_radius": operational_radius,
        }
    return output


def prefix_valid_radius(
    rows: Sequence[Mapping[str, float]],
    tolerance: float,
    error_key: str = "absolute_residual",
) -> float:
    """Largest radius in the consecutive passing prefix, or zero.

    A larger radius is not accepted after a smaller one has failed.  This is
    intentionally stricter than selecting the largest passing point anywhere
    on a noisy curve.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    radius = 0.0
    previous = -math.inf
    for row in sorted(rows, key=lambda item: float(item["radius"])):
        current = float(row["radius"])
        error = float(row[error_key])
        if current <= previous:
            raise ValueError("radii must be unique within one direction/sign curve")
        previous = current
        if not math.isfinite(error) or error > tolerance:
            break
        radius = current
    return float(radius)


def summarize_validity(
    records: Sequence[Mapping[str, object]],
    tolerances: Sequence[float],
    *,
    error_key: str = "absolute_residual",
) -> dict[str, dict]:
    """Summarize prefix-valid radii by direction family and tolerance.

    ``error_key`` makes the same conservative prefix rule reusable for three
    distinct RQ3 quantities: Taylor residual, observed old-task loss increase,
    and old-task accuracy damage.  In every case a curve stops at its first
    value strictly larger than the supplied tolerance.
    """
    grouped: dict[tuple[str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["direction_family"]),
            int(record["direction_index"]),
            int(record["sign"]),
        )
        grouped[key].append(record)

    families = sorted({key[0] for key in grouped})
    output: dict[str, dict] = {}
    for family in families:
        family_out: dict[str, dict] = {}
        family_groups = [rows for key, rows in grouped.items() if key[0] == family]
        grid_maxima = []
        for rows in family_groups:
            observed = [float(row["radius"]) for row in rows]
            if not observed:
                raise ValueError(f"empty direction/sign curve for family {family}")
            grid_maxima.append(max(observed))
        if grid_maxima and not np.allclose(
            grid_maxima, grid_maxima[0], rtol=0.0, atol=0.0
        ):
            raise ValueError(f"inconsistent radius grids within family {family}")
        grid_max = float(grid_maxima[0]) if grid_maxima else 0.0
        for tolerance in tolerances:
            radii = np.asarray(
                [
                    prefix_valid_radius(rows, tolerance, error_key=error_key)
                    for rows in family_groups
                ],
                dtype=np.float64,
            )
            all_radius = float(radii.min()) if radii.size else 0.0
            q05_radius = float(np.quantile(radii, 0.05)) if radii.size else 0.0
            saturated = int(np.count_nonzero(radii == grid_max)) if radii.size else 0
            family_out[f"{float(tolerance):g}"] = {
                "tolerance": float(tolerance),
                "curves": int(radii.size),
                "all_direction_radius": all_radius,
                "q05_direction_radius": q05_radius,
                "median_direction_radius": float(np.median(radii)) if radii.size else 0.0,
                "mean_direction_radius": float(radii.mean()) if radii.size else 0.0,
                "max_direction_radius": float(radii.max()) if radii.size else 0.0,
                "grid_max_radius": grid_max,
                "saturated_curves": saturated,
                "saturation_fraction": (
                    float(saturated / radii.size) if radii.size else 0.0
                ),
                "all_direction_radius_censored": bool(
                    radii.size and all_radius == grid_max
                ),
                "q05_direction_radius_censored": bool(
                    radii.size and q05_radius == grid_max
                ),
            }
        output[family] = family_out
    return output
