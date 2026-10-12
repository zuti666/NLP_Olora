#!/usr/bin/env python3
"""CPU-only algebra for s3_boundary_controls_v1 sidecars; no model forwards.

Examples:
  python analyze_boundaries.py --self-test --output-prefix /tmp/boundary_test
  python analyze_boundaries.py --input RUN/mechanism_states --output-prefix OUT

Writes OUT.json, OUT.layers.csv, OUT.hashes.json with exclusive creation.
DeltaW means the unscaled B@A coordinates used by this S3 learner. Classifier,
Fisher, rank-truncated replacements, and prediction changes are not evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path

import torch

VERSION = "s3_boundary_algebra_v1"
FORMAT = "s3_boundary_controls_v1"
FACTOR = re.compile(r"^(.*)\.linear_([ab])_([qv])\.weight$")
NORM_KEYS = [
    "endpoint_factor_l2", "retained_factor_l2", "factor_retained_endpoint_drift_l2",
    "factor_reconstruction_error_l2", "endpoint_dw_l2", "retained_dw_l2",
    "dw_retained_endpoint_drift_l2", "path_factor_l2", "factor_path_endpoint_drift_l2",
    "path_dw_l2", "snapshot_linear_dw_l2", "dw_path_endpoint_drift_l2",
    "intrapath_gap_l2", "intrapath_cross_term_l2", "intrapath_identity_error_l2",
    "blend_linear_dw_l2", "blend_gap_l2", "blend_cross_term_l2",
    "blend_identity_error_l2", "retained_dw_rounding_error_l2",
    "total_snapshot_linear_dw_l2", "total_gap_l2", "total_identity_error_l2",
]
RATIOS = {
    "factor_retained_endpoint_drift_relative": ("factor_retained_endpoint_drift_l2", "endpoint_factor_l2"),
    "dw_retained_endpoint_drift_relative": ("dw_retained_endpoint_drift_l2", "endpoint_dw_l2"),
    "factor_path_endpoint_drift_relative": ("factor_path_endpoint_drift_l2", "endpoint_factor_l2"),
    "dw_path_endpoint_drift_relative": ("dw_path_endpoint_drift_l2", "endpoint_dw_l2"),
    "intrapath_gap_relative_to_linear_dw": ("intrapath_gap_l2", "snapshot_linear_dw_l2"),
    "blend_gap_relative_to_linear_dw": ("blend_gap_l2", "blend_linear_dw_l2"),
    "blend_gap_relative_to_retained_drift": ("blend_gap_l2", "dw_retained_endpoint_drift_l2"),
    "total_gap_relative_to_linear_dw": ("total_gap_l2", "total_snapshot_linear_dw_l2"),
    "total_gap_relative_to_retained_drift": ("total_gap_l2", "dw_retained_endpoint_drift_l2"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.double()).item())


def factors_norm(a, b):
    return math.hypot(norm(a), norm(b))


def add_ratios(row):
    for key, (num, den) in RATIOS.items():
        row[key] = row[num] / row[den] if row.get(num) is not None and row.get(den, 0) else None


def validate_factors(factors, expected=None):
    require(isinstance(factors, dict) and factors, "Nonempty factor dictionary required")
    if expected is not None:
        require(set(factors) == set(expected), "Inconsistent factor keys")
    for key, value in factors.items():
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu", f"CPU tensor required: {key}")
        require(value.is_floating_point() and value.ndim == 2, f"Floating matrix required: {key}")
        require(bool(torch.isfinite(value).all()), f"Nonfinite factor: {key}")
        require(FACTOR.fullmatch(key) is not None, f"Unsupported factor name: {key}")
        if expected is not None:
            require(value.shape == expected[key].shape and value.dtype == expected[key].dtype,
                    f"Factor shape/dtype differs: {key}")


def factor_pairs(factors):
    pairs = {}
    for name in factors:
        module, side, branch = FACTOR.fullmatch(name).groups()
        pairs.setdefault(f"{module}.{branch}", {})[side] = name
    for name, keys in pairs.items():
        require(set(keys) == {"a", "b"}, f"Missing A/B pair: {name}")
        a, b = factors[keys["a"]], factors[keys["b"]]
        require(a.shape[0] == b.shape[1], f"Rank dimensions differ: {name}")
    return sorted(pairs.items())


def native_path(snapshot_factors, weights, keys):
    """Reproduce native tensor order, dtype, and final weight-sum division."""
    total = sum(weights)
    result = {}
    for key in keys:
        value = snapshot_factors[0][key] * weights[0]
        for factors, weight in zip(snapshot_factors[1:], weights[1:]):
            value = value + factors[key] * weight
        result[key] = value / total
    return result


def first_nonnull(*values):
    return next((value for value in values if value is not None), None)


def analyze_payload(payload, source, input_sha=None):
    require(payload.get("format_version") == FORMAT, "Unsupported boundary format")
    endpoint, retained = payload["endpoint_factors"], payload["retained_factors"]
    validate_factors(endpoint)
    validate_factors(retained, endpoint)
    pairs = factor_pairs(endpoint)
    config, merge = payload.get("config", {}), payload.get("merge_metrics", {})
    applied = bool(merge.get("in_task_weight_merge_applied", False))
    omega = float(first_nonnull(merge.get("in_task_weight_merge_omega"),
                               config.get("in_task_weight_merge_omega"),
                               config.get("in_task_fisher_omega"), 0.0))
    require(math.isfinite(omega) and 0 <= omega <= 1, "Expected convex merge omega")
    selected = payload.get("selected_path_weights", [])
    require(isinstance(selected, list), "selected_path_weights must be a list")
    snapshots = {int(x["index"]): x for x in payload.get("snapshots", [])}
    require(len(snapshots) == len(payload.get("snapshots", [])), "Duplicate snapshot index")
    require(len({int(x["index"]) for x in selected}) == len(selected), "Duplicate selected snapshot")
    weights, factors, selected_meta = [], [], []
    if selected:
        require(payload.get("path_weights_computed") is True, "Weights present but not actually computed")
        for record in selected:
            snap = snapshots[int(record["index"])]
            require(int(snap["epoch"]) == int(record["epoch"]), "Selected epoch/index disagree")
            validate_factors(snap["factors"], endpoint)
            weight = float(record["weight"])
            require(math.isfinite(weight) and weight >= 0, "Invalid path weight")
            weights.append(weight)
            factors.append(snap["factors"])
            selected_meta.append({**record, "snapshot_train_ce": snap.get("metrics", {}).get("snapshot_train_ce")})
        require(sum(weights) > 0 and abs(sum(weights) - 1) < 1e-8, "Weights must sum to one")
    require(not applied or bool(selected), "Applied merge requires captured actual path weights")
    path_native = native_path(factors, weights, endpoint) if selected else None
    effective_omega = omega if applied else 0.0
    reconstructed = {key: value * (1 - effective_omega) + path_native[key] * effective_omega
                     for key, value in endpoint.items()} if applied else endpoint
    exact = all(torch.equal(reconstructed[key], retained[key]) for key in endpoint)
    normalized_weights = [w / sum(weights) for w in weights] if weights else []
    rows = []
    for layer, keys in pairs:
        ak, bk = keys["a"], keys["b"]
        ae, be = endpoint[ak].double(), endpoint[bk].double()
        ar, br = retained[ak].double(), retained[bk].double()
        de, dr = be @ ae, br @ ar
        row = {"source": str(source), "task": int(payload["task"]), "layer": layer,
               "a_key": ak, "b_key": bk, "rank": ae.shape[0],
               "in_features": ae.shape[1], "out_features": be.shape[0],
               "merge_applied": applied, "omega": omega, "effective_omega": effective_omega,
               "path_available": bool(selected), "selected_snapshots": len(selected),
               "factor_reconstruction_bitwise_exact": all(torch.equal(reconstructed[k], retained[k]) for k in (ak, bk)),
               **{key: None for key in NORM_KEYS}}
        row.update(endpoint_factor_l2=factors_norm(ae, be), retained_factor_l2=factors_norm(ar, br),
                   factor_retained_endpoint_drift_l2=factors_norm(ar - ae, br - be),
                   factor_reconstruction_error_l2=factors_norm(reconstructed[ak].double() - ar,
                                                               reconstructed[bk].double() - br),
                   endpoint_dw_l2=norm(de), retained_dw_l2=norm(dr),
                   dw_retained_endpoint_drift_l2=norm(dr - de))
        if selected:
            # Float64 algebra separates operator nonlinearities from native-float rounding.
            aps = [x[ak].double() for x in factors]
            bps = [x[bk].double() for x in factors]
            ap = sum(w * a for w, a in zip(normalized_weights, aps))
            bp = sum(w * b for w, b in zip(normalized_weights, bps))
            dp = bp @ ap
            linear_path = sum(w * (b @ a) for w, a, b in zip(normalized_weights, aps, bps))
            gap_path = dp - linear_path
            cross_path = torch.zeros_like(dp)
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    cross_path -= normalized_weights[i] * normalized_weights[j] * ((bps[i] - bps[j]) @ (aps[i] - aps[j]))
            o = effective_omega
            am, bm = (1 - o) * ae + o * ap, (1 - o) * be + o * bp
            dm = bm @ am
            linear_blend = (1 - o) * de + o * dp
            gap_blend = dm - linear_blend
            cross_blend = -o * (1 - o) * ((bp - be) @ (ap - ae))
            linear_total = (1 - o) * de + o * linear_path
            gap_total = dr - linear_total
            rounding = dr - dm
            row.update(path_factor_l2=factors_norm(ap, bp),
                       factor_path_endpoint_drift_l2=factors_norm(ap - ae, bp - be),
                       path_dw_l2=norm(dp), snapshot_linear_dw_l2=norm(linear_path),
                       dw_path_endpoint_drift_l2=norm(dp - de),
                       intrapath_gap_l2=norm(gap_path), intrapath_cross_term_l2=norm(cross_path),
                       intrapath_identity_error_l2=norm(gap_path - cross_path),
                       blend_linear_dw_l2=norm(linear_blend), blend_gap_l2=norm(gap_blend),
                       blend_cross_term_l2=norm(cross_blend),
                       blend_identity_error_l2=norm(gap_blend - cross_blend),
                       retained_dw_rounding_error_l2=norm(rounding),
                       total_snapshot_linear_dw_l2=norm(linear_total), total_gap_l2=norm(gap_total),
                       total_identity_error_l2=norm(gap_total - (cross_blend + o * cross_path + rounding)))
        add_ratios(row)
        rows.append(row)
    aggregate = {key: math.sqrt(sum(row[key] ** 2 for row in rows))
                 if all(row[key] is not None for row in rows) else None for key in NORM_KEYS}
    add_ratios(aggregate)
    identities = []
    for error, reference in [("intrapath_identity_error_l2", "snapshot_linear_dw_l2"),
                             ("blend_identity_error_l2", "blend_linear_dw_l2"),
                             ("total_identity_error_l2", "total_snapshot_linear_dw_l2")]:
        if aggregate[error] is not None:
            identities.append(aggregate[error] <= 1e-10 * max(1.0, aggregate[reference]))
    result = {
        "source": str(source), "input_sha256": input_sha, "task": int(payload["task"]),
        "format_version": payload["format_version"], "factor_pairs": len(pairs),
        "factor_tensors": len(endpoint), "merge_applied": applied, "omega": omega,
        "effective_omega": effective_omega, "path_algebra_available": bool(selected),
        "selected_path_weights": selected_meta, "weight_sum": sum(weights),
        "path_weight_entropy_effective_n": math.exp(-sum(w * math.log(w) for w in normalized_weights if w)) if weights else None,
        "factor_reconstruction_bitwise_exact": exact,
        "cross_term_identities_passed": all(identities),
        "passed": exact and all(identities), "aggregate": aggregate,
        "config": config, "merge_metrics": merge, "path_metrics": payload.get("path_metrics", {}),
        "rng_before_boundary_sha256": payload.get("rng_before_boundary_sha256"),
        "rng_after_boundary_sha256": payload.get("rng_after_boundary_sha256"),
        "s3_fisher_location": payload.get("s3_fisher_location"),
    }
    return result, rows


def synthetic_checks():
    """Exercise rank1 hand result, nonuniform multilayer path, no-op, and corruption."""
    gen = torch.Generator(device="cpu").manual_seed(9202026)
    def draw():
        data = {}
        for layer in range(2):
            for branch in ("q", "v"):
                for side, shape in [("a", (3, 7)), ("b", (5, 3))]:
                    data[f"blocks.{layer}.attn.qkv.linear_{side}_{branch}.weight"] = torch.randn(shape, generator=gen)
        return data
    endpoint, snaps = draw(), [draw(), draw(), draw()]
    weights = [.05, .15, .8]
    path = native_path(snaps, weights, endpoint)
    def payload(omega, applied=True):
        return {"format_version": FORMAT, "task": 0, "endpoint_factors": endpoint,
                "retained_factors": {k: v * (1 - omega) + path[k] * omega for k, v in endpoint.items()} if applied else endpoint,
                "snapshots": [{"index": i, "epoch": 10 * (i + 1), "factors": f, "metrics": {"snapshot_train_ce": 1 / (i + 1)}} for i, f in enumerate(snaps)],
                "selected_path_weights": [{"index": i, "epoch": 10 * (i + 1), "weight": w} for i, w in enumerate(weights)] if applied else [],
                "path_weights_computed": applied,
                "merge_metrics": {"in_task_weight_merge_applied": applied, "in_task_weight_merge_omega": omega}, "config": {}}
    results, layers = [], []
    for name, o, applied in [("nonuniform_path", .3, True), ("all_path", 1., True), ("merge_disabled", .3, False)]:
        result, rows = analyze_payload(payload(o, applied), f"synthetic:{name}")
        require(result["passed"], f"Synthetic case failed: {name}")
        if applied:
            require(result["aggregate"]["intrapath_gap_l2"] > 0, "Need nontrivial path gap")
        if o == 1:
            require(result["aggregate"]["blend_cross_term_l2"] == 0, "omega1 blend gap must vanish")
        results.append(result); layers.extend(rows)
    # Scalar endpoint A=1,B=2; path A=3,B=5; omega=.25.
    # merged BA=1.5*2.75=4.125, direct blend=.75*2+.25*15=5.25.
    scalar_e = {"block.linear_a_q.weight": torch.tensor([[1.]]), "block.linear_b_q.weight": torch.tensor([[2.]])}
    scalar_p = {"block.linear_a_q.weight": torch.tensor([[3.]]), "block.linear_b_q.weight": torch.tensor([[5.]])}
    scalar = {"format_version": FORMAT, "task": 0, "endpoint_factors": scalar_e,
              "retained_factors": {k: .75 * v + .25 * scalar_p[k] for k, v in scalar_e.items()},
              "snapshots": [{"index": 0, "epoch": 40, "factors": scalar_p}],
              "selected_path_weights": [{"index": 0, "epoch": 40, "weight": 1.}],
              "path_weights_computed": True, "merge_metrics": {"in_task_weight_merge_applied": True, "in_task_weight_merge_omega": .25}}
    result, rows = analyze_payload(scalar, "synthetic:hand_rank1")
    require(result["passed"] and result["aggregate"]["blend_gap_l2"] == 1.125, "Hand-computed scalar identity failed")
    require(result["aggregate"]["intrapath_gap_l2"] == 0, "One snapshot path must have zero gap")
    results.append(result); layers.extend(rows)
    corrupt = payload(.3)
    corrupt["retained_factors"] = {k: v.clone() for k, v in corrupt["retained_factors"].items()}
    next(iter(corrupt["retained_factors"].values()))[0, 0] += .01
    failed, _ = analyze_payload(corrupt, "synthetic:deliberate_corruption")
    require(not failed["passed"] and not failed["factor_reconstruction_bitwise_exact"], "Corruption was not detected")
    return results, layers, {"passed": True, "positive_cases": 4, "corruption_detected": True,
                             "hand_rank1_gap": 1.125, "local_cpu_rng_only": True}


def save_outputs(prefix, report, rows):
    prefix = Path(prefix)
    targets = [Path(str(prefix) + suffix) for suffix in (".json", ".layers.csv", ".hashes.json")]
    require(not any(x.exists() for x in targets), "Refusing to overwrite output path")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    # All computation/serialization succeeds before any exclusive output creation.
    with targets[0].open("x") as f:
        f.write(text)
    with targets[1].open("x", newline="") as f:
        f.write(stream.getvalue())
    manifest = {"utility": str(Path(__file__).resolve()), "utility_sha256": sha(__file__),
                "outputs": {str(x.resolve()): sha(x) for x in targets[:2]},
                "input_hashes": {x["source"]: x["input_sha256"] for x in report["boundaries"] if x.get("input_sha256")}}
    with targets[2].open("x") as f:
        json.dump(manifest, f, indent=2); f.write("\n")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, help="Sidecar files or directories containing task_*.pt")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    require(bool(args.input) != args.self_test, "Use exactly one of --input or --self-test")
    require(args.threads > 0, "threads must be positive")
    torch.set_num_threads(args.threads)
    if args.self_test:
        boundaries, rows, checks = synthetic_checks()
    else:
        paths = sorted({x.resolve() for p in args.input for x in (sorted(p.glob("task_*.pt")) if p.is_dir() else [p])})
        require(bool(paths), "No task sidecars found")
        boundaries, rows, checks = [], [], None
        for path in paths:
            before = sha(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            result, layers = analyze_payload(payload, path, before)
            require(before == sha(path), f"Input changed while reading: {path}")
            boundaries.append(result); rows.extend(layers)
    report = {"version": VERSION, "device": "cpu", "dtype_for_algebra": "float64",
              "native_reconstruction": "original factor dtype, original selected order and weight-sum division",
              "passed": all(x["passed"] for x in boundaries), "synthetic_checks": checks,
              "boundary_count": len(boundaries), "layer_row_count": len(rows),
              "identities": {
                  "intrapath": "Bbar@Abar - sum_i(w_i Bi@Ai) = -sum_{i<j} w_i w_j (Bi-Bj)@(Ai-Aj)",
                  "blend": "Bmerge@Amerge - ((1-o)Be@Ae + o Bbar@Abar) = -o(1-o)(Bbar-Be)@(Abar-Ae)",
                  "total": "retained_deltaW - ((1-o)endpoint_deltaW + o sum_i(w_i snapshot_deltaW_i)) = blend_cross + o*path_cross + native_rounding",
              }, "limits": ["Algebra only; no accuracy, loss, Fisher, or classifier assessment.",
                             "DeltaW is unscaled B@A, matching this learner; does not include backbone weights.",
                             "Ratios with zero denominator are null.",
                             "Absent actual selected weights yields null path metrics; no hypothetical weights are invented.",
                             "Aggregate norms concatenate all dense blocks; relative norms are ratios of aggregate norms."],
              "boundaries": boundaries}
    targets = save_outputs(args.output_prefix, report, rows)
    print(json.dumps({"passed": report["passed"], "boundaries": len(boundaries), "layers": len(rows), "outputs": [str(x) for x in targets]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"boundary algebra error: {error}", file=sys.stderr)
        sys.exit(2)
