#!/usr/bin/env python3
"""Real-ViT gate: native read schedules and five-read audits submit identically.

Runs eight isolated ordinary SGD updates from one historical Task0 entry state.
These updates belong only to this gate, never to a formal training trajectory.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn.functional as F
import audit_core as audit
from common import PROTOCOL, atomic_json, now, read_json, sha256, snapshot_rng_fingerprint
from direction_step import FourProbeStep, expected_budget
from effective_step import EffectiveStep
from offline_diagnostics import construct_learner
from state_transaction import State

DIRECTIONS = ("sum12", "base02", "two_midpoint", "sum13")


def next_random_draws():
    """Probe every RNG stream, restoring it after measuring the next draw."""
    state = audit.rng_get()
    try:
        draws = dict(python=[random.random() for _ in range(16)],
                     numpy=np.random.random(16).tolist(),
                     torch_cpu=audit.tensor_digest(torch.rand(16)),
                     torch_cuda=[audit.tensor_digest(torch.rand(16, device=f"cuda:{i}"))
                                 for i in range(torch.cuda.device_count())])
        return dict(sha256=hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest(),
                    streams=draws)
    finally:
        audit.rng_set(state)
        if not audit.rng_equal(state, audit.rng_get()):
            raise AssertionError("Measuring next random draw changed RNG state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "reference", "config", "diagnostic-output", "output-dir"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    for key in ("source", "reference", "config", "diagnostic_output", "output_dir"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    source, reference, diagnostic, out = map(Path, (args.source, args.reference, args.diagnostic_output, args.output_dir))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Real-model gate output must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, status="running", protocol=PROTOCOL,
                  gate_protocol="real_model_native_vs_audit_five_reads_v1", started_at=now(),
                  source=str(source), reference_output=str(reference), diagnostic_output=str(diagnostic),
                  formal_training_runs=0, ordinary_gradient_calls=0, sgd_updates=0, directions={})
    atomic_json(out/"model_gate.json", report)
    started = time.monotonic()
    try:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Pinned deterministic cuBLAS workspace required before CUDA")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError("This is a real GPU model gate")
        torch.cuda.set_device(0)
        raw = read_json(args.config)
        if (raw.get("protocol"), raw.get("update_direction"), raw.get("gam_grad_rho"),
                raw.get("gam_grad_norm_rho"), raw.get("lora_rank")) != (PROTOCOL, "sum12", .05, 1., 10):
            raise ValueError("Expected the new sum12 seed2/rank10/r1 configuration")
        if raw.get("seed") not in (2, [2]) or raw.get("ewc_lambda") != 0 or raw.get("ewc_skip_task_fisher") is not True:
            raise ValueError("Pure seed2 configuration required")
        if any(raw.get(key) for key in ("approximate_fisher_in_task", "in_task_weight_merge", "task_endpoint_weight_merge")):
            raise ValueError("Fisher and persistent weight merging must be disabled")
        diagnostic_summary = read_json(diagnostic/"diagnostics_summary.json")
        if diagnostic_summary.get("status") != "completed" or diagnostic_summary.get("accepted") is not True:
            raise AssertionError("Accepted diagnostic smoke is required")
        rows = [json.loads(line) for line in (diagnostic/"diagnostics.jsonl").read_text().splitlines()]
        if not rows or (rows[0]["task"], rows[0]["point"], rows[0]["batch_index"]) != (0, "entry", 0):
            raise AssertionError("First diagnostic coordinates must be Task0 entry batch0")
        row = rows[0]
        coordinate_path = Path(row["coordinate_file"])
        if coordinate_path.parent.resolve() != diagnostic or sha256(coordinate_path) != row["coordinate_file_sha256"]:
            raise AssertionError("Diagnostic coordinates changed or escaped the accepted output")
        observations = read_json(reference/"observations.json")
        entries = [item for item in observations if (item["task"], item["point"]) == (0, "entry")]
        if len(entries) != 1 or sha256(entries[0]["path"]) != entries[0]["sha256"]:
            raise AssertionError("Historical Task0 entry snapshot is ambiguous or changed")
        snapshot = torch.load(entries[0]["path"], map_location="cpu", weights_only=False)
        payload = torch.load(coordinate_path, map_location="cpu", weights_only=False)
        pinned = {args.config: sha256(args.config), str(coordinate_path): row["coordinate_file_sha256"],
                  entries[0]["path"]: entries[0]["sha256"],
                  str(diagnostic/"diagnostics_summary.json"): sha256(diagnostic/"diagnostics_summary.json"),
                  str(diagnostic/"diagnostics.jsonl"): sha256(diagnostic/"diagnostics.jsonl"),
                  str(reference/"observations.json"): sha256(reference/"observations.json")}
        for path,digest in diagnostic_summary.get("source_hashes", {}).items():
            if sha256(path) != digest:
                raise AssertionError("Diagnostic source evidence changed: "+path)
            pinned[path] = digest
        manifest_path = Path(__file__).resolve().parent.parent/"source_manifest.json"
        frozen = read_json(manifest_path)
        pinned[str(manifest_path)] = sha256(manifest_path)
        for relative, digest in frozen["files"].items():
            if sha256(source/relative) != digest:
                raise AssertionError("Frozen source changed: "+relative)
            pinned[str(source/relative)] = digest
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        sys.path.insert(0, str(source))
        os.chdir(source)
        learner, model, dm, checkpoint, guards = construct_learner(source, reference, 0, raw)
        named = dict(model.named_parameters())
        names = tuple(payload["parameter_names"])
        if len(names) != len(set(names)) or set(names) != {name for name, p in named.items() if p.requires_grad}:
            raise AssertionError("Coordinates omit or reorder unidentified trainable parameters")
        if set(names) != set(snapshot["trainable_state"]) or len(payload["base"]) != len(names):
            raise AssertionError("Historical and diagnostic trainable identities differ")
        with torch.no_grad():
            for name, value in zip(names, payload["base"]):
                p = named[name]
                if value.dtype != torch.float32 or p.dtype != torch.float32 or p.shape != value.shape:
                    raise AssertionError("Actual diagnostic base must preserve FP32 named coordinates")
                if not torch.equal(value, snapshot["trainable_state"][name]):
                    raise AssertionError("Diagnostic base differs from historical Task0 entry")
                p.copy_(value.to(p))
        params = tuple(named[name] for name in names)
        model.train()
        audit.rng_set(snapshot["rng"])
        anchor = State(model)
        geometry = EffectiveStep.from_model(model)
        batch = payload["current_train_batch"]
        x, y = batch["input"].to("cuda:0"), batch["target"].to("cuda:0")
        if y.dtype != torch.long or not bool(((y >= 0)&(y < 20)).all()):
            raise AssertionError("Task0 training target slice differs")
        eta = float(payload["eta"])
        if not math.isfinite(eta) or eta != .04:
            raise AssertionError("Task0 entry gate requires actual LR=.04")
        input_hash = audit.tensor_digest(x)
        target_hash = audit.tensor_digest(y)
        if input_hash != row["current_batch_identity"]["input_sha256"] or target_hash != row["current_batch_identity"]["target_sha256"]:
            raise AssertionError("Saved real augmented training batch changed")
        report.update(parameter_names=list(names), coordinate_file=str(coordinate_path),
                      coordinate_file_sha256=pinned[str(coordinate_path)],
                      input_sha256=input_hash, target_sha256=target_hash, lr=eta,
                      initial_rng_sha256=snapshot_rng_fingerprint(anchor.rng),
                      input_config_sha256=pinned[args.config], trainer_sha256=sha256(__file__))

        def closure(inputs, targets):
            logits = model(inputs)["logits"]
            return F.cross_entropy(logits[:, :20], targets), logits

        for direction in DIRECTIONS:
            cfg = dict(direction=direction, branch="P", backend="native", rho=.05, radius=1., eps=1e-12)
            anchor.restore(); anchor.check()
            native = FourProbeStep(model, params, geometry, cfg)
            native_result = native.compute(x, y, closure, eta)
            native_U = tuple(value.detach().clone() for value in native_result["U"])
            native_optimizer = torch.optim.SGD(params, lr=eta, momentum=0, weight_decay=0)
            native_update = native.apply_update(eta, optimizer=native_optimizer)
            # Capture the submitted state, not the pre-update anchor. Compare
            # named tensors directly and also every cache/buffer/mode/RNG stream.
            submitted = State(model)
            submitted_named = {name: p.detach().cpu().clone() for name,p in model.named_parameters()}
            submitted_grads = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                               for name,p in model.named_parameters()}
            native_rng = audit.rng_get()
            native_draw = next_random_draws()
            anchor.restore(); anchor.check()
            audited = FourProbeStep(model, params, geometry, cfg)
            audited_result = audited.compute(x, y, closure, eta, audit_all=True)
            audited_optimizer = torch.optim.SGD(params, lr=eta, momentum=0, weight_decay=0)
            audited_update = audited.apply_update(eta, optimizer=audited_optimizer)
            audited_rng = audit.rng_get()
            audited_draw = next_random_draws()
            named_equal = all(torch.equal(p.detach().cpu(), submitted_named[name]) for name,p in model.named_parameters())
            grads_equal = all((p.grad is None and submitted_grads[name] is None) or
                              (p.grad is not None and submitted_grads[name] is not None and
                               torch.equal(p.grad.detach().cpu(), submitted_grads[name])) for name,p in model.named_parameters())
            state_checks = submitted.check()
            checks = dict(U_exact=all(torch.equal(u,v) for u,v in zip(native_U,audited_result["U"])),
                          common_gradient_readouts_exact=all(all(torch.equal(a,b) for a,b in zip(values,audited_result["context"]["grads"][key]))
                              for key,values in native_result["context"]["grads"].items()),
                          common_probe_coordinates_exact=all(all(torch.equal(a,b) for a,b in zip(values,audited_result["context"]["points"][key]))
                              for key,values in native_result["context"]["points"].items()),
                          all_named_parameters_exact=named_equal, all_parameter_gradients_exact=grads_equal,
                          **state_checks, rng_all_streams_exact=audit.rng_equal(native_rng,audited_rng),
                          next_random_draws_exact=native_draw == audited_draw,
                          training_inputs_unchanged=audit.tensor_digest(x)==input_hash and audit.tensor_digest(y)==target_hash,
                          native_read_budget=native_result["record"]["ordinary_gradient_calls"]==expected_budget(cfg)["train_ce_per_update"],
                          audited_five_reads=audited_result["record"]["ordinary_gradient_calls"]==5,
                          actual_sgd_updates=native_update["sgd_updates"]==audited_update["sgd_updates"]==1,
                          natural_sgd_exact=native_update["natural_sgd_exact"] and audited_update["natural_sgd_exact"],
                          all_losses_finite=all(math.isfinite(value) for result in (native_result,audited_result)
                                               for key,value in result["record"].items() if key.startswith("loss") and value is not None),
                          forbidden_gradient_work_zero=all(result["record"][key]==0 for result in (native_result,audited_result)
                              for key in ("fisher_calls","weight_merge_calls","gradient_projection_calls","hvp_calls","create_graph_calls")))
            if not all(checks.values()):
                raise AssertionError("Native/audit real-model state differs: "+repr(checks))
            report["ordinary_gradient_calls"] += native_result["record"]["ordinary_gradient_calls"]+5
            report["sgd_updates"] += 2
            report["directions"][direction] = dict(direction=direction, checks=checks,
                native_gradient_calls=native_result["record"]["ordinary_gradient_calls"], audit_gradient_calls=5,
                native_loss0=native_result["loss0"], audit_loss0=audited_result["loss0"],
                update_direction_sha256=[audit.tensor_digest(value) for value in native_U],
                submitted_rng_sha256=snapshot_rng_fingerprint(native_rng), next_random_draw_sha256=native_draw["sha256"])
            atomic_json(out/"model_gate.json", report)
            anchor.restore(); anchor.check()
            del submitted,submitted_named,submitted_grads,native_U,native_result,audited_result,native,audited
            gc.collect()
        anchor.restore(); restoration = anchor.check()
        unchanged = all(sha256(path)==digest for path,digest in pinned.items())
        final_checks = dict(four_directions=len(report["directions"])==4,
                            all_direction_checks=all(all(item["checks"].values()) for item in report["directions"].values()),
                            native_reads_3_3_3_4=[report["directions"][key]["native_gradient_calls"] for key in DIRECTIONS]==[3,3,3,4],
                            audited_reads_all_five=all(item["audit_gradient_calls"]==5 for item in report["directions"].values()),
                            ordinary_gradient_calls_33=report["ordinary_gradient_calls"]==33,
                            isolated_sgd_updates_8=report["sgd_updates"]==8,
                            no_formal_training=report["formal_training_runs"]==0,
                            historical_source_artifacts_unchanged=unchanged,
                            forbidden_work_zero=guards["forbidden_work_attempts"]==0,
                            final_anchor_restored=all(restoration.values()))
        if not all(final_checks.values()):
            raise AssertionError("Real-model gate accounting failed: "+repr(final_checks))
        report.update(status="passed", checks=final_checks, source_hashes=pinned,
                      gradient_calls_total=report["ordinary_gradient_calls"], independent_sgd_updates=report["sgd_updates"],
                      elapsed_seconds=time.monotonic()-started, finished_at=now(),
                      diagnostic_gradient_reads_only=True, old_data_gradient_calls=0,
                      fisher_calls=0, weight_merge_calls=0, hvp_calls=0, create_graph_calls=0,
                      scope="One Task0-entry paired batch; isolated copy updates, never formal training")
    except Exception as exc:
        report.update(status="failed", error=repr(exc), elapsed_seconds=time.monotonic()-started, finished_at=now())
        atomic_json(out/"model_gate.json", report)
        raise
    atomic_json(out/"model_gate.json", report)
    print(json.dumps(dict(status=report["status"], ordinary_gradient_calls=report["ordinary_gradient_calls"],
                         sgd_updates=report["sgd_updates"], checks=report["checks"])))


if __name__ == "__main__":
    main()
