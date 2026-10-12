"""CPU-only preparation checks; writes only this training directory's audit JSON."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import common
from common import HERE, atomic_json, load_suite, read_json, sha256, verify_config, verify_source
import summarize_g1g3 as summary


def main():
    paths = read_json(HERE / "PREP_PATHS.json")
    final = Path(paths["final_training"])
    previous = Path(paths["previous"])
    suite = load_suite(HERE / "suite.json")
    assert len(suite["jobs"]) == 5
    assert suite["max_parallel"] == 5 and suite["allowed_gpus"] == list(range(7))
    assert suite["source_root"] == paths["source_root"]
    assert suite["source_manifest"] == str(final / "source_manifest.json")
    assert suite["final_report_script"] == str(final / "summarize_g1g3.py")
    old_suite = read_json(previous / "suite.json")
    assert suite["environment"] == old_suite["environment"]
    assert suite["python"] == old_suite["python"]
    core_names = ("common.py", "run_job.py", "launch_queue.py", "collect_results.py")
    assert all(sha256(HERE / name) == sha256(previous / name) for name in core_names)
    assert sha256(HERE / "source_manifest.json") == sha256(previous / "source_manifest.json")
    local_suite = dict(suite, source_manifest=str(HERE / "source_manifest.json"))
    provenance = verify_source(local_suite)
    actual, configs, seen_outputs = [], {}, set()
    false_flags = ("in_task_weight_merge", "task_endpoint_weight_merge", "approximate_fisher_in_task",
                   "ewc_normalize_fisher", "ewc_normalize_accumulated_fisher", "ewc_dual_ascent",
                   "compute_full_fisher", "posterior_state_save", "mechanism_anchor_eval",
                   "mechanism_future_surrogate_eval", "as1_delta_w_penalty", "endpoint_eval",
                   "task0_trajectory_eval", "task0_fisher_recompute_eval", "fisher_split",
                   "flat_eval", "feature_flat_eval", "in_task_path_weighted")
    diff_document = read_json(HERE / "CONFIG_DIFFS.json")
    for job, recorded in zip(suite["jobs"], diff_document["config_diffs"]):
        assert job["id"] == recorded["id"]
        local = HERE / "configs" / Path(job["config"]).name
        assert job["config"] == str(final / "configs" / local.name)
        config = read_json(local)
        configs[job["id"]] = config
        assert sha256(local) == job["config_sha256"]
        assert config["prefix"] == job["id"]
        assert config["outputs_root"] == str(final / "outputs" / job["id"])
        assert config["outputs_root"] not in seen_outputs
        seen_outputs.add(config["outputs_root"])
        assert config["seed"] == [2] and config["lora_rank"] == 10
        assert config["lambda_flat"] == 1 and config["ewc_lambda"] == 0
        assert config["ewc_skip_task_fisher"] is True
        assert all(config[key] is False for key in false_flags)
        assert all(config[key] == 0 for key in ("in_task_weight_merge_omega", "in_task_fisher_omega",
                                               "ewc_fisher_damping_alpha", "ewc_eta"))
        assert config["gam_grad_gamma"] == 0 and config["gam_grad_beta_2"] == 1
        assert config["gam_grad_rho"] == .05
        assert not any(key in config for key in ("resume_task", "resume_checkpoint", "as1_rho", "as1_norm_rho"))
        for meta_key, config_key in (("beta1", "gam_grad_beta_1"), ("beta2", "gam_grad_beta_2"),
                                     ("beta3", "gam_grad_beta_3"), ("gamma", "gam_grad_gamma"),
                                     ("outer_radius", "gam_grad_norm_rho")):
            assert job[meta_key] == config[config_key]
        actual.append((config["gam_grad_beta_1"], config["gam_grad_beta_3"], config["gam_grad_norm_rho"]))
        base_path = Path(recorded["reference_config"])
        assert sha256(base_path) == recorded["reference_sha256"]
        base = read_json(base_path)
        changes = {key: {"before": base.get(key), "after": config.get(key)}
                   for key in sorted(set(base) | set(config)) if base.get(key) != config.get(key)}
        assert changes == recorded["changes"]
        allowed = {"prefix", "outputs_root", "gam_grad_beta_1", "gam_grad_beta_3"} if job["direction"] == "mean_sum" else {"prefix", "outputs_root", "gam_grad_norm_rho"}
        assert set(changes) <= allowed
        # Exercise the unchanged runtime config validator with staged file paths.
        original_here = common.HERE
        try:
            common.HERE = final
            verify_config(dict(job, config=str(local)))
        finally:
            common.HERE = original_here
    assert actual == [(.5, .5, 1.), (0., 1., 0.), (0., 1., .1), (0., 1., .25), (0., 1., .5)]
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))

    # Reuse the prior tested CPU harness definitions only. Its original cases,
    # report writes and module-level execution below `cases = ...` are excluded.
    audit_path = previous / "puregam_cpu_audit.py"
    tree = ast.parse(audit_path.read_text())
    cut = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "cases" for target in node.targets))
    namespace = {"__name__": "g1g3_cpu_harness", "__file__": str(audit_path)}
    exec(compile(ast.Module(body=tree.body[:cut], type_ignores=[]), str(audit_path), "exec"), namespace)
    torch = namespace["torch"]
    cases = []
    for job in suite["jobs"]:
        config = configs[job["id"]]
        namespace["BASE_CONFIG"] = HERE / "configs" / Path(job["config"]).name
        for dtype in (torch.float64, torch.float32):
            case = namespace["check"](job["id"], config["gam_grad_beta_1"], config["gam_grad_beta_3"], 0., dtype)
            case["outer_radius"] = config["gam_grad_norm_rho"]
            cases.append(case)

    pending = [{"id": job["id"], "value": job["value"], "seed": job["seed"], "status": "pending"}
               for job in suite["jobs"]]
    preview = summary.build_report(pending, configs)
    assert preview["completed_new"] == 0 and preview["reused"] == 4 and len(preview["rows"]) == 9
    assert next(row for row in preview["rows"] if row["value"] == "mean_sum_r1")["beta1"] == .5
    # Exercise all five final-summary rows using validated reference matrices,
    # explicitly as synthetic data in memory; no fake completion receipts exist.
    reference = next(row for row in preview["rows"] if row["status"] == "reused" and row["beta1"] == 0)
    synthetic = [{**copy.deepcopy(reference), "id": job["id"], "value": job["value"], "status": "completed"}
                 for job in suite["jobs"]]
    full_preview = summary.build_report(synthetic, configs)
    assert full_preview["completed_new"] == 5 and full_preview["reused"] == 4
    assert all("acquisition_task_macro" in row for row in full_preview["rows"])
    assert not (HERE / "runtime").exists() and not (HERE / "outputs").exists()
    report = {"status": "PASS", "scope": "CPU-only; staged path mapping; no GPU query or training launch",
              "suite_sha256": sha256(HERE / "suite.json"), "source_files_verified": len(provenance["source_files"]),
              "source_manifest_sha256": provenance["source_manifest_sha256"],
              "runtime_byte_identical": {name: sha256(HERE / name) for name in core_names},
              "new_jobs": 5, "reused_jobs_verified": 4, "old_harness_sha256": sha256(audit_path),
              "torch_version": torch.__version__, "cases": cases,
              "summary_checks": {"pending_plus_reused_rows": 9, "synthetic_complete_rows": 9,
                                 "synthetic_results_written": False},
              "max_direction_error": max(case["max_direction_error"] for case in cases),
              "max_update_error": max(case["max_update_error"] for case in cases),
              "training_modified": False, "training_launched": False}
    atomic_json(HERE / "CPU_PREFLIGHT.json", report)
    print(json.dumps({key: report[key] for key in ("status", "new_jobs", "reused_jobs_verified", "max_direction_error", "max_update_error")}))


if __name__ == "__main__":
    main()
