"""Register 16 smokes, one CPU pairing gate, and 16 formals; never starts jobs."""
import argparse
import csv
import json
from pathlib import Path
import socket
import os

from common import HERE, PROTOCOL, atomic_json, now, read_json, sha256, suite_counts, verify_cpu_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="/home-local/liying/.conda/envs/Pilot_new_local/bin/python")
    ap.add_argument("--execution-hostname", default=socket.gethostname())
    ap.add_argument("--execution-pid-namespace", default=os.readlink("/proc/self/ns/pid"))
    ap.add_argument("--cpu-gate", default=str(HERE / "trainer/CPU_GATE.json"))
    ap.add_argument("--source", default="/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source")
    ap.add_argument("--max-parallel", type=int, default=5)
    args = ap.parse_args()
    if (HERE / "runtime/queue_state.json").exists():
        raise RuntimeError("Refusing to rebuild an active or previously started suite")
    cpu_gate = dict(path=str(Path(args.cpu_gate).resolve()), sha256=sha256(args.cpu_gate))
    verify_cpu_gate(dict(cpu_gate=cpu_gate))
    settings = list(csv.DictReader((HERE / "EXPERIMENT_SETTINGS.csv").open()))
    if len(settings) != 16:
        raise ValueError("Expected 16 settings")
    source_manifest = HERE / "source_manifest.json"
    environment_spec = HERE / "environment_spec.json"
    environment = dict(read_json(environment_spec)["env"])
    environment.update({key: "/tmp/" + HERE.name for key in ("TMPDIR", "TMP", "TEMP")})
    source = str(Path(args.source).resolve())
    formal_settings = settings[:]
    # Each radius starts with corrected descending probes, then its clean baseline.
    priority = {variant: index for index, variant in enumerate(("N5", "N6", "B0", "B1", "N1", "N2", "N3", "N4"))}
    formal_settings.sort(key=lambda row: (-float(row["radius"]), priority[row["variant"]]))
    jobs = []
    for smoke in (True, False):
        for row in formal_settings:
            condition = row["job_id"]
            job_id = ("smoke_" if smoke else "formal_") + condition
            config_path = Path(row["config"])
            config = read_json(config_path)
            if smoke:
                config_path = HERE / "configs" / (job_id + ".json")
                config.update(prefix=job_id, outputs_root=str(HERE / "outputs" / job_id / "native"))
                atomic_json(config_path, config)
            output_dir = HERE / "outputs" / job_id if smoke else HERE / "outputs" / condition
            if not Path(config["outputs_root"]).resolve().is_relative_to(output_dir.resolve()):
                raise ValueError("Formal outputs_root mismatch: " + condition)
            command = [args.python, str(HERE / "trainer/four_probe_train.py"),
                       "--source", source, "--config", str(config_path), "--output-dir", str(output_dir)]
            if smoke:
                command += ["--max-tasks", "2", "--max-epochs", "1", "--max-batches", "1"]
            jobs.append(dict(id=job_id, condition_id=condition, variant=row["variant"], kind="training",
                             smoke=smoke, radius=float(row["radius"]), rho=.05, seed=2, rank=10,
                             branch="P", direction=config["update_direction"], update_base=row["anchor"],
                             first_probe_sign=1, second_probe_sign=int(row["probe2_sign"]),
                             second_difference_sign=int(row["residual_sign"]), formula=row["formula"],
                             source_root=source,
                             config=str(config_path), config_sha256=sha256(config_path), output_dir=str(output_dir),
                             command=command, depends_on=[], expected_tasks=2 if smoke else 10,
                             expected_epochs_per_task=1 if smoke else 50, expected_steps=2 if smoke else 2550,
                             expected_diagnostic_gradient_calls=2 if smoke else 10,
                             selected_side_reused_from_training=True))
    smoke_ids = [job["id"] for job in jobs if job["smoke"]]
    for job in jobs:
        if not job["smoke"]:
            job["depends_on"] = ["smoke_pairing_gate"]
    pairing = dict(id="smoke_pairing_gate", kind="pairing", smoke=False,
                   output_dir=str(HERE / "gates/smoke_pairing"), report_file="smoke_pairing.json",
                   command=[args.python, str(HERE / "smoke_pairing.py")], depends_on=smoke_ids)
    jobs.insert(len(smoke_ids), pairing)
    suite = dict(schema_version=1, run_id=HERE.name, run_dir=str(HERE), generated_at=now(),
                 execution_hostname=args.execution_hostname, execution_pid_namespace=args.execution_pid_namespace,
                 python=args.python, source_root=source, source_manifest=str(source_manifest),
                 source_manifest_sha256=sha256(source_manifest), environment_spec=str(environment_spec),
                 environment_spec_sha256=sha256(environment_spec), environment=environment,
                 allowed_gpus=[2, 3, 4, 5, 6], max_parallel=args.max_parallel, poll_seconds=60,
                 gpu_lock_dir="/tmp/olora_gamfs_gpu_locks", idle_memory_mib_below=500,
                 idle_util_percent_below=5, require_no_compute_process=True,
                 expected_tasks=10, protocol=PROTOCOL,
                 diagnostic_protocol="selected_training_g3_reuse_opposite_only_v2",
                 diagnostic_gradients_per_task=1,
                 diagnostic_accounting=dict(per_task_new_gradient_calls=1, per_task_new_forward_calls=1,
                                          selected_side_source="training_g3", selected_side_new_gradient_calls=0,
                                          opposite_side_source="isolated_opposite_probe", opposite_side_new_gradient_calls=1),
                 no_external_notifications=True, failure_policy="stop_on_failure",
                 cpu_gate=cpu_gate, reused_results=[], jobs=jobs)
    atomic_json(HERE / "suite.json", suite)
    print(json.dumps(dict(status="registered", **suite_counts(suite),
                          max_parallel=args.max_parallel, suite_sha256=sha256(HERE / "suite.json"))))


if __name__ == "__main__":
    main()
