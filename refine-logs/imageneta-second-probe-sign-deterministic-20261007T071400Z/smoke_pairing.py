"""CPU comparison of accepted real-GPU smoke inputs and their common initial state."""
from collections import defaultdict
import json
from pathlib import Path

from common import HERE, PROTOCOL, VARIANTS, atomic_json, read_json, sha256
from validate_job import validate_job


def main():
    suite = read_json(HERE / "suite.json")
    state = read_json(HERE / "runtime/queue_state.json")
    smokes = [job for job in suite["jobs"] if job["kind"] == "training" and job["smoke"]]
    formals = [job for job in suite["jobs"] if job["kind"] == "training" and not job["smoke"]]
    items = []
    accepted = True
    for job in smokes:
        checked = validate_job(job)
        receipt = read_json(HERE / "runtime/jobs" / job["id"] / "completed.json")
        accepted &= state["jobs"][job["id"]]["status"] == "completed"
        accepted &= all(receipt[key] == checked[key] for key in ("summary_sha256", "trace_sha256", "matrix_sha256", "diagnostics_sha256", "updates", "tasks"))
        out = Path(job["output_dir"])
        trace = [json.loads(line) for line in (out / "step_trace.jsonl").read_text().splitlines()]
        identities = read_json(out / "task_identities.json")
        counters = read_json(out / "final_summary.json")["counters"]
        items.append(dict(job=job["id"], radius=job["radius"], variant=job["variant"],
                          summary_sha256=checked["summary_sha256"], trace_sha256=checked["trace_sha256"],
                          initial_parameter_hash=identities[0]["task_initial_parameter_sha256"],
                          initial_head_hash=identities[0]["new_fc_rows_sha256"],
                          frozen_parameter_hash=identities[0]["frozen_origin_sha256"],
                          batch_identities=[row["identity"] for row in trace],
                          task_class_slices=[row["class_slice"] for row in trace],
                          g012_fingerprints=trace[0]["g012_fingerprints"],
                          point012_fingerprints=trace[0]["point012_fingerprints"],
                          same_state_snapshot=checked["same_state_snapshot"],
                          diagnostic_gradient_calls=counters["diagnostic_gradient_calls"],
                          diagnostic_snapshots=counters["diagnostic_snapshots"],
                          steps=len(trace), training_ce_passes=counters["training_ce_gradient_passes"]))
    first = items[0]
    groups = defaultdict(list)
    for item in items:
        groups[item["radius"]].append(item)
    checks = dict(all_smokes_accepted=bool(accepted), sixteen_smokes=len(items) == 16,
                  sixteen_formals=len(formals) == 16,
                  initial_parameters_equal=all(item["initial_parameter_hash"] == first["initial_parameter_hash"] for item in items),
                  initial_head_equal=all(item["initial_head_hash"] == first["initial_head_hash"] for item in items),
                  initial_frozen_parameters_equal=all(item["frozen_parameter_hash"] == first["frozen_parameter_hash"] for item in items),
                  initial_rng_all_streams_equal=all(all(item["same_state_snapshot"][key] == first["same_state_snapshot"][key] for key in ("rng0_sha256", "rng1_sha256")) for item in items),
                  initial_parameter_layout_equal=all(item["same_state_snapshot"]["parameter_layout_sha256"] == first["same_state_snapshot"]["parameter_layout_sha256"] for item in items),
                  paired_realized_batches=all(item["batch_identities"] == first["batch_identities"] for item in items),
                  class_slices_equal=all(item["task_class_slices"] == [[0, 20], [20, 40]] for item in items),
                  four_training_ce_per_step=all(item["steps"] == 2 and item["training_ce_passes"] == 8 for item in items),
                  same_state_tensor_snapshots_verified=all(item["same_state_snapshot"]["verified"] is True for item in items),
                  one_opposite_diagnostic_ce_per_task=all(item["diagnostic_snapshots"] == 2 and item["diagnostic_gradient_calls"] == 2 for item in items),
                  same_radius_initial_g012=all(all(item["g012_fingerprints"] == group[0]["g012_fingerprints"] for item in group) for group in groups.values()),
                  same_radius_initial_points=all(all(item["point012_fingerprints"] == group[0]["point012_fingerprints"] for item in group) for group in groups.values()),
                  radius_factor_grid=set(groups) == {.25, 1.} and all({item["variant"] for item in group} == VARIANTS and len(group) == len(VARIANTS) for group in groups.values()))
    report = dict(status="passed" if all(checks.values()) else "failed", protocol=PROTOCOL, smoke_count=len(smokes),
                  formal_count=len(formals), checks=checks, smokes=items,
                  scope="Pair realized inputs and initial Task0 model/g0/g1/g2 within radius; later evolving gradients need not match.")
    atomic_json(HERE / "gates/smoke_pairing/smoke_pairing.json", report)
    if report["status"] != "passed":
        raise AssertionError("Actual-GPU smoke data/state pairing failed")
    print(json.dumps(dict(status=report["status"], checks=checks), ensure_ascii=False))


if __name__ == "__main__":
    main()
