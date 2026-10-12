#!/usr/bin/env python3
"""Pure offline evaluation of native ImageNet-R EWC / GAM-EWC 20+20 runs.

Example:
  python evaluate.py --run-dir RUN --device cuda --fisher-samples 256
  python evaluate.py --self-test

Consumes complete checkpoints/task_00.pt and task_01.pt. Never trains or
downloads pretrained weights. Outputs evaluation/summary.json by default.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from eval_common import (atomic_json, audit_checkpoint_pair, canonical_hash,
                         compare_hook_autograd, configure_precision, convert_effective,
                         evaluate_scopes, fisher, load_network, prepare_data, self_test,
                         sha_file, tensor_hash, unpack_batch)


DEFAULT_SOURCE = "/data/140-1/users/liying/CopyFlatCV"
DEFAULT_DATA_ROOT = "/data/115-2/users/liying/datasets"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source", type=Path, help=f"Default: manifest source, else {DEFAULT_SOURCE}")
    parser.add_argument("--data-root", type=Path, help=f"Default: config data_root, else {DEFAULT_DATA_ROOT}")
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N within visible devices")
    parser.add_argument("--fisher-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="JSON path, or directory containing summary.json")
    parser.add_argument("--self-test", action="store_true", help="Synthetic CPU test; no dataset/source required")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing evaluation output")
    return parser.parse_args()


def read_config(run_dir):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.update(manifest.get("config", {}))
    method = manifest.get("method", config.get("comparison_method"))
    if method not in ("ewc", "gam_ewc"):
        raise ValueError("Expected a native ewc/gam_ewc run manifest, not a Ratio-FS run")
    if config.get("comparison_method", method) != method:
        raise ValueError("Manifest method and resolved config disagree")
    if str(config.get("dataset", "")).replace("-", "").lower() != "imagenetr":
        raise ValueError("Only ImageNet-R runs are supported")
    for name, expected in (("init_cls", 20), ("increment", 20), ("lora_rank", 10)):
        if int(config.get(name, expected)) != expected:
            raise ValueError(f"Only {name}={expected} is supported")
    return config, manifest


def endpoint(task, checkpoint, values, delta, loaders, device, source):
    started = time.monotonic()
    model = load_network(checkpoint, source, device)
    _, sanity_x, sanity_y = unpack_batch(next(iter(loaders["old_fisher"])))
    sanity_x, sanity_y = sanity_x[:2].to(device), sanity_y[:2].to(device)
    with torch.no_grad():
        native_logits = model(sanity_x)
    collector, parameters = convert_effective(model, values)
    with torch.no_grad():
        effective_logits = model(sanity_x)
    torch.testing.assert_close(effective_logits, native_logits, rtol=1e-5, atol=3e-5)
    sanity = {"native_to_effective_logits_max_absolute_error":
              float((native_logits - effective_logits).abs().max())}
    directions = {"task0_to_task1_delta": [value.to(device) for value in delta]}
    sanity["old_local20"] = compare_hook_autograd(model, collector, parameters, sanity_x, sanity_y,
                                                  "old_local20", directions)
    before_hash = tensor_hash(model.state_dict().values())
    scopes = ["old_local20"] + (["seen40"] if task == 1 else [])
    metrics = {split: evaluate_scopes(model, loaders[split], scopes, device)
               for split in ("old_train", "old_test")}
    traces = {"old_local20": fisher(model, collector, loaders["old_fisher"], "old_local20",
                                   directions, device)}
    if task == 1:
        _, new_x, new_y = unpack_batch(next(iter(loaders["new_fisher"])))
        sanity["new_local20"] = compare_hook_autograd(model, collector, parameters,
                                                       new_x[:2].to(device), new_y[:2].to(device),
                                                       "new_local20", {})
        metrics["new_test"] = evaluate_scopes(model, loaders["new_test"],
                                               ["new_local20", "seen40"], device)
        traces["new_local20"] = fisher(model, collector, loaders["new_fisher"], "new_local20", {}, device)
    after_hash = tensor_hash(model.state_dict().values())
    if before_hash != after_hash:
        raise RuntimeError("Evaluation mutated model tensors")
    squared = sum(float(value.double().square().sum()) for value in delta)
    traces["old_local20"]["unit_displacement_sensitivity"] = (
        traces["old_local20"]["q"]["task0_to_task1_delta"] / squared if squared > 0 else None)
    result = {"task_index": task, "metrics": metrics, "fisher": traces, "sanity": sanity,
              "model_sha256_before": before_hash, "model_sha256_after": after_hash,
              "seconds": time.monotonic() - started}
    del model, parameters, collector, directions
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = arguments()
    configure_precision()
    if args.self_test:
        result = self_test()
        if args.output:
            output = args.output if args.output.suffix == ".json" else args.output / "self_test.json"
            atomic_json(output, result)
        print(json.dumps(result, indent=2))
        return
    if args.run_dir is None:
        raise ValueError("--run-dir is required unless --self-test is used")
    if args.batch_size < 1 or args.workers < 0 or args.fisher_samples < 1:
        raise ValueError("Invalid batch size, workers, or sample count")
    run_dir = args.run_dir.resolve()
    config, manifest = read_config(run_dir)
    source_value = args.source or manifest.get("source") or config.get("source") or DEFAULT_SOURCE
    if isinstance(source_value, dict):
        source_value = source_value.get("path") or source_value.get("root")
    if not source_value:
        raise ValueError("Cannot determine source path from manifest")
    source = Path(source_value).resolve()
    data_root = Path(args.data_root or config.get("data_root") or DEFAULT_DATA_ROOT).resolve()
    output = args.output or run_dir / "evaluation" / "summary.json"
    if output.suffix != ".json":
        output = output / "summary.json"
    output = output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace explicitly")
    if args.overwrite:
        # A failed replacement must not leave an earlier success receipt active.
        (output.parent / "COMPLETE.json").unlink(missing_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; --device cpu is supported")
    paths = [run_dir / "checkpoints" / f"task_{task:02d}.pt" for task in (0, 1)]
    started = time.monotonic()
    print("Auditing complete endpoint checkpoints", flush=True)
    values, delta, audit = audit_checkpoint_pair(paths)
    loaders, data_manifest = prepare_data(config, manifest, source, data_root,
                                          args.fisher_samples, args.sample_seed,
                                          args.batch_size, args.workers, device)
    atomic_json(output.parent / "data_manifest.json", data_manifest)
    inputs = {
        "run_dir": str(run_dir), "source": str(source), "data_root": str(data_root),
        "run_manifest_sha256": sha_file(run_dir / "run_manifest.json"),
        "config": config, "checkpoint_audit": audit,
        "data_manifest_sha256": canonical_hash(data_manifest),
        "code_sha256": {name: sha_file(Path(__file__).parent / name)
                        for name in ("evaluate.py", "eval_common.py")},
        "source_sha256": {name: sha_file(source / name) for name in
                          ("backbone/lora.py", "backbone/linears.py", "utils/data.py", "utils/data_manager.py")},
        "precision": {"forward_backward": "FP32", "reduction": "float64",
                      "autocast": False, "matmul_tf32": False, "cudnn_tf32": False},
        "device": str(device), "torch": torch.__version__, "argv": sys.argv,
    }
    atomic_json(output.parent / "evaluation_manifest.json", inputs)
    endpoints = []
    for task in (0, 1):
        print(f"Evaluating task{task:02d}: streamed losses, accuracy and per-example Fisher", flush=True)
        result = endpoint(task, paths[task], values[task], delta, loaders, device, source)
        atomic_json(output.parent / f"task_{task:02d}_evaluation.json", result)
        endpoints.append(result)
    anchor, final = endpoints
    if anchor["fisher"]["old_local20"]["input_tensors_sha256"] != final["fisher"]["old_local20"]["input_tensors_sha256"]:
        raise RuntimeError("Old Fisher sample tensors differ between endpoints")
    a11 = anchor["metrics"]["old_test"]["old_local20"]["accuracy"]
    a21 = final["metrics"]["old_test"]["seen40"]["accuracy"]
    a22 = final["metrics"]["new_test"]["seen40"]["accuracy"]
    damages = {split: final["metrics"][split]["old_local20"]["loss"] -
                      anchor["metrics"][split]["old_local20"]["loss"]
               for split in ("old_train", "old_test")}
    summary = {
        "status": "complete", "protocol": "imagenetr_native_gam_ewc_1x1_effective_qv_ef_v1",
        "method": manifest.get("method", config.get("comparison_method")),
        "seed": manifest.get("seed", config.get("seed")),
        "a11": a11, "a21": a21, "a22": a22, "bwt": a21 - a11,
        "old_train_loss_delta": damages["old_train"],
        "old_test_loss_delta": damages["old_test"],
        "r_eff": audit["delta_norm"],
        "old_trace_task0": anchor["fisher"]["old_local20"]["raw_trace"],
        "old_trace_task1": final["fisher"]["old_local20"]["raw_trace"],
        "new_trace_task1": final["fisher"]["new_local20"]["raw_trace"],
        "units": {"accuracy": "percent", "bwt": "percentage points", "loss": "CE nats per sample",
                  "r_eff": "effective Q/V Frobenius L2 norm; excludes classifier"},
        "inputs": inputs, "checkpoint_audit": audit, "endpoints": endpoints,
        "accuracy_CIL_percent": {"a11": a11, "a21": a21, "a22": a22,
                                  "BWT_percentage_points": a21 - a11},
        "old_local20_loss_increment": damages,
        "fisher_raw_trace": {"old_at_task0": anchor["fisher"]["old_local20"]["raw_trace"],
                             "old_at_task1": final["fisher"]["old_local20"]["raw_trace"],
                             "new_at_task1": final["fisher"]["new_local20"]["raw_trace"]},
        "delta_norm": audit["delta_norm"], "delta_squared_norm": audit["delta_squared_norm"],
        "seconds": time.monotonic() - started,
    }
    atomic_json(output, summary)
    atomic_json(output.parent / "COMPLETE.json", {"status": "complete", "summary": str(output),
                                                  "summary_sha256": sha_file(output),
                                                  "protocol": summary["protocol"]})
    print(json.dumps({"status": "complete", "summary": str(output),
                      "accuracy_CIL_percent": summary["accuracy_CIL_percent"],
                      "fisher_raw_trace": summary["fisher_raw_trace"]}, indent=2))


if __name__ == "__main__":
    main()
