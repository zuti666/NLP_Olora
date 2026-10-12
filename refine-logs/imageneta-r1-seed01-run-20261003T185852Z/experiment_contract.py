"""Fail-closed CPU contract for the approved pure-GAM seed0/1 repetition.

This checks scientific identity against frozen seed2 sources, independently of
the launcher's filename conventions. It never imports torch or starts a GPU job.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from common import HERE, load_suite, read_json, sha256, verify_source

NATIVE = "native_global_rng_5CE"
PROBE = "four_probe_exact_restore_rng_replay"
CONDITIONS = {f"N{i:02d}" for i in range(1, 15)} | {f"P{i:02d}" for i in range(1, 17)}
BASELINE_RADIUS_INDEPENDENT = {"P01", "P02", "P03", "P16"}
DIRECT = {"N01", "N02", "N04", "N10"}
NATIVE_SELECTORS = dict(zip(
    [f"N{i:02d}" for i in range(1, 15)],
    ["native_beta", "native_beta", "g1_control", "native_beta", "g0", "g2",
     "ug", "c21_base", "c21_perp", "native_beta", "g1_control", "g1_control", "mean", "gmatch"]))
NATIVE_ARMS = dict(zip([f"N{i:02d}" for i in range(1, 15)],
                      ["GAM", "Sum", "S", "Graw", "g0", "g2", "ug", "A", "B", "Mean", "S15", "S2", "Mean2", "Gmatch"]))
TASK_BATCHES = [10, 8, 8, 5, 4, 3, 3, 3, 4, 3]
SOURCE_MANIFEST_HASH = "e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def _pinned_json(path, digest):
    require(sha256(path) == digest, "Pinned file changed: " + str(path))
    return read_json(path)


def _verify_config(job, catalog_row, run_dir):
    identifier, condition = job["id"], job["condition"]
    native, smoke = job["family"] == NATIVE, job["smoke"]
    for key in ("source_config", "source_config_sha256", "source_runner", "source_entry"):
        require(job[key] == catalog_row[key], f"{identifier}: source provenance {key} changed")
    source = _pinned_json(job["source_config"], job["source_config_sha256"])
    cfg = _pinned_json(job["config"], job["config_sha256"])
    require(Path(job["config"]).resolve() == run_dir / "configs" / (identifier + ".json"),
            identifier + ": config path redirected")
    require(Path(job["output_dir"]).resolve() == run_dir / "outputs" / identifier,
            identifier + ": output path redirected")
    expected = copy.deepcopy(source)
    expected.update(seed=[job["seed"]], prefix=identifier,
                    outputs_root=str(Path(job["output_dir"]) / "native") if not native else job["output_dir"])
    if condition in BASELINE_RADIUS_INDEPENDENT:
        require(source["gam_grad_norm_rho"] == .25 and catalog_row["radius_independent_final_update"] is True,
                identifier + ": historical baseline radius/proof changed")
        expected["gam_grad_norm_rho"] = 1.
    if native and smoke:
        expected["max_train_tasks"] = 2
    require(cfg == expected, identifier + ": unauthorized change from the seed2 scientific config")
    diff = {k: {"source": source.get(k), "current": cfg.get(k)}
            for k in source.keys() | cfg.keys() if source.get(k) != cfg.get(k)}
    require(job["config_diff"] == diff, identifier + ": declared config diff is inaccurate")
    fixed = dict(dataset="imageneta", class_shuffle=False, init_cls=20, increment=20,
                 nb_tasks=10, total_sessions=10, lora_rank=10, batch_size=128,
                 lambda_flat=1., gam_grad_norm_rho=1., gam_grad_rho=.05,
                 gam_adaptive=False, init_epoch=50, epochs=50, scheduler="cosine",
                 optimizer="sgd", gam_base_optimizer="sgd", init_momentum=0,
                 momentum=0, init_weight_decay=0., weight_decay=0., memory_size=0,
                 memory_per_class=0, ewc_lambda=0., ewc_eta=0., ewc_skip_task_fisher=True,
                 approximate_fisher_in_task=False, in_task_weight_merge=False,
                 task_endpoint_weight_merge=False, ewc_normalize_fisher=False,
                 ewc_normalize_accumulated_fisher=False, ewc_fisher_damping_alpha=0.,
                 compute_full_fisher=False, task0_fisher_recompute_eval=False,
                 as1_delta_w_penalty=False, ewc_dual_ascent=False)
    for key, value in fixed.items():
        require(cfg.get(key) == value, f"{identifier}: required {key}={value!r} differs")
    multiplier = {"N11": 1.5, "N12": 2., "N13": 2.}.get(condition, 1.)
    require(abs(cfg["init_lr"] - .04 * multiplier) < 1e-14
            and abs(cfg["lrate"] - .02 * multiplier) < 1e-14,
            identifier + ": nominal learning rates changed")
    if native:
        require(job["direction_selector"] == NATIVE_SELECTORS[condition]
                and job["arm"] == NATIVE_ARMS[condition]
                and job["lr_multiplier"] == multiplier,
                identifier + ": final direction/learning-rate selector changed")
        beta_gamma = tuple(cfg[k] for k in ("gam_grad_beta_1", "gam_grad_beta_2", "gam_grad_beta_3", "gam_grad_gamma"))
        expected_bg = {"N01": (1., 1., 1., .1), "N02": (1., 1., 1., 0.),
                       "N04": (0., 1., 1., 0.), "N10": (.5, 1., .5, 0.)}.get(condition, (1., 1., 0., 0.))
        require(beta_gamma == expected_bg, identifier + ": historical beta/gamma implementation changed")
        require(cfg["mechanism_eval"] is True and cfg["mechanism_grad_stats"] is True
                and cfg["eval_num_workers"] == 8, identifier + ": native RNG/lifecycle protocol changed")
    else:
        bindings = catalog_row["runtime_bindings"]
        pairs = {"branch": "probe_branch", "direction": "update_direction", "mode": "step_mode", "backend": "attention_backend"}
        for field, config_key in pairs.items():
            require(job[field] == bindings[config_key] == cfg[config_key],
                    f"{identifier}: four-probe {field} changed")
        require(job["backend"] == "native" and cfg["mechanism_eval"] is False
                and cfg["mechanism_grad_stats"] is False and cfg["eval_num_workers"] == 4,
                identifier + ": exact-restoration protocol changed")
    return cfg


def _verify_bindings(job, catalog_row):
    """Bindings may repeat validated metadata, never introduce an extra override."""
    old = catalog_row["runtime_bindings"]
    actual = job.get("runtime_bindings", {})
    require(job["runtime_bindings_source_template"] == old, job["id"] + ": historical binding template changed")
    if job["family"] == PROBE:
        require(actual == old, job["id"] + ": four-probe binding redirect")
        return
    if not old:
        require(actual == {}, job["id"] + ": unexpected runtime override")
        return
    expected = copy.deepcopy(old)
    for field in ("id", "arm", "direction_selector", "seed", "phase", "lr_multiplier",
                  "outer_radius", "config", "config_sha256", "expected_tasks", "expected_steps",
                  "expected_samples", "expected_task_steps", "depends_on", "reference_job"):
        expected[field] = job[field]
    require(actual == expected, job["id"] + ": bindings disagree with validated seed/config/condition")


def verify_contract(suite):
    require(suite["allowed_gpus"] == [4, 5, 6] and suite["max_parallel"] == 3,
            "Only the authorized physical GPU4/5/6 three-worker pool is allowed")
    require(suite["seeds"] == [0, 1] and suite["radius"] == 1. and suite["rho"] == .05
            and suite["expected_tasks"] == 10, "Scientific campaign scope changed")
    require("CUDA_VISIBLE_DEVICES" not in suite.get("environment", {}), "Worker GPU binding must come from the lease")
    run_dir = Path(suite["run_dir"]).resolve()
    require(run_dir == HERE, "Suite points at a different campaign")
    require(suite["source_manifest_sha256"] == SOURCE_MANIFEST_HASH
            and sha256(suite["source_manifest"]) == SOURCE_MANIFEST_HASH, "Shared source190 manifest changed")
    catalog_list = read_json(run_dir / "plan_snapshot/CONDITION_CATALOG.json")["conditions"]
    catalog = {row["id"]: row for row in catalog_list}
    require(len(catalog_list) == 30 and set(catalog) == CONDITIONS, "Frozen 30-condition catalog changed")
    jobs = suite["jobs"]
    triples = [(j["condition"], j["seed"], j["phase"]) for j in jobs]
    expected_triples = {(c, s, p) for c in CONDITIONS for s in (0, 1) for p in ("smoke", "formal")}
    require(len(jobs) == 120 and len(set(triples)) == 120 and set(triples) == expected_triples,
            "Scope must be exactly 30 conditions x 2 seeds x smoke/formal")
    smoke_ids = {j["id"] for j in jobs if j["phase"] == "smoke"}
    all_ids = {j["id"] for j in jobs}
    require(len(all_ids) == 120, "Repeated job IDs")
    for job in jobs:
        condition, seed, phase = job["condition"], job["seed"], job["phase"]
        smoke = phase == "smoke"
        prefix = "smoke_" if smoke else ""
        identifier = f"{prefix}{condition}_r1_k10_s{seed}"
        row = catalog[condition]
        require(job["id"] == identifier and job["smoke"] is smoke and job["kind"] == "training",
                identifier + ": job/phase identity differs")
        require(job["family"] == row["family"] and job["block"] == row["block"]
                and job["label"] == row["label"], identifier + ": protocol grouping changed")
        require(job["outer_radius"] == 1. and job["inner_radius"] == .05, identifier + ": actual probe radii changed")
        native = job["family"] == NATIVE
        budget = (2, 1, 18, 2256) if native and smoke else (2, 1, 2, 256) if smoke else (10, 50, 2550, 299050)
        require(tuple(job[k] for k in ("expected_tasks", "expected_epochs_per_task", "expected_steps", "expected_samples")) == budget,
                identifier + ": task/epoch/update/sample budget differs")
        task_steps = {"0": 10, "1": 8} if native and smoke else {"0": 1, "1": 1} if smoke else {str(t): 50*b for t, b in enumerate(TASK_BATCHES)}
        require(job["expected_task_steps"] == task_steps, identifier + ": task update counts differ")
        ref = f"{prefix}N03_r1_k10_s{seed}" if condition == "N14" else None
        require(job["reference_job"] == job["independent_reference_job"] == ref,
                identifier + ": Gmatch must use the independent same-seed same-phase S")
        dependencies = set() if smoke else smoke_ids.copy()
        if ref:
            dependencies.add(ref)
        require(len(job["depends_on"]) == len(dependencies) and set(job["depends_on"]) == dependencies,
                identifier + ": smoke gates/reference dependency changed")
        require(job["requires_same_seed_smoke_gate"] is True, identifier + ": smoke requirement removed")
        _verify_config(job, row, run_dir)
        _verify_bindings(job, row)
        runner = "native" if native else "probe"
        expected_command = [suite["python"], str(run_dir / runner / "run.py"),
                            "--suite", str(run_dir / "suite.json"), "--job", identifier]
        require(job["command"] == expected_command, identifier + ": runnable command redirected")
    # All old formulas/optimizer helpers remain exactly pinned, independently of
    # locally adapted seed-aware launch/validation wrappers in RUN_PINS.
    checked = set()
    for row in catalog_list:
        for item in row["source_implementation"]:
            pair = (item["path"], item["sha256"])
            if pair not in checked:
                require(sha256(item["path"]) == item["sha256"], "Historical implementation changed: " + item["path"])
                checked.add(pair)
    verify_source(suite)
    return True


def self_test(suite):
    """Meaningful negative tests: every dangerous metadata mutation is rejected."""
    verify_contract(suite)
    cases = []
    def rejected(name, change):
        candidate = copy.deepcopy(suite)
        change(candidate)
        try:
            verify_contract(candidate)
        except (RuntimeError, ValueError, KeyError):
            cases.append(name)
        else:
            raise AssertionError("Contract accepted corruption: " + name)
    rejected("unauthorized_gpu", lambda s: s.update(allowed_gpus=[0, 4, 5]))
    rejected("duplicate_condition_seed", lambda s: s["jobs"].__setitem__(1, copy.deepcopy(s["jobs"][0])))
    rejected("missing_all_smoke_gate", lambda s: next(j for j in s["jobs"] if not j["smoke"]).__setitem__("depends_on", []))
    rejected("cross_seed_gmatch", lambda s: next(j for j in s["jobs"] if j["condition"] == "N14").__setitem__("reference_job", "smoke_N03_r1_k10_s1"))
    rejected("short_training_budget", lambda s: next(j for j in s["jobs"] if not j["smoke"]).__setitem__("expected_steps", 18))
    rejected("direction_selector_changed", lambda s: next(j for j in s["jobs"] if j["condition"] == "N08").__setitem__("direction_selector", "g0"))
    rejected("runtime_seed_override", lambda s: next(j for j in s["jobs"] if j["condition"] == "N03")["runtime_bindings"].__setitem__("seed", 2))
    rejected("command_redirect", lambda s: s["jobs"][0]["command"].append("--seed=2"))
    return {"status": "passed", "jobs": 120, "formal": 60, "smoke": 60,
            "negative_cases_rejected": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=HERE / "suite.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    result = self_test(suite) if args.self_test else {"passed": verify_contract(suite), "jobs": len(suite["jobs"])}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
