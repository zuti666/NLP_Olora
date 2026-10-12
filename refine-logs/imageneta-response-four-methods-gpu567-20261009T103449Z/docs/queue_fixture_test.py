"""Independent CPU-only scheduler fixtures. Never dispatch CUDA or training."""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H))
import common
import run_queue as q


def exercise(case):
    with tempfile.TemporaryDirectory(prefix="response567_fixture_") as d:
        root = Path(d)
        suite = json.loads((H / "suite.json").read_text())
        suite["run_dir"] = str(root)
        suite["gpu_lock_dir"] = str(root / "leases")
        for k in ("TMPDIR", "TMP", "TEMP"):
            suite["environment"][k] = str(root / "tmp")
        for job in suite["jobs"]:
            job["output_dir"] = str(root / "outputs" / job["id"])
        jobs = {j["id"]: j for j in suite["jobs"]}
        launch = []
        query_count = 0
        sleeps = 0

        def query():
            nonlocal query_count
            query_count += 1
            gpus = {i: dict(index=i, uuid=f"GPU-fixture-{i}", memory_used_mb=2,
                            utilization_percent=0, compute_pids=[]) for i in range(7)}
            if case == "occupied5":
                gpus[5].update(memory_used_mb=9000, utilization_percent=70, compute_pids=[12345])
            if case == "underlock_recheck" and query_count == 2:
                gpus[5].update(memory_used_mb=9000, utilization_percent=70, compute_pids=[12345])
            return gpus

        class FakeWorker:
            def __init__(self, command, *, cwd, env, stdout, stderr, start_new_session, pass_fds):
                jobfile = Path(command[command.index("--job") + 1])
                job = json.loads(jobfile.read_text())
                state = json.loads((root / "runtime/queue_state.json").read_text())
                record = state["jobs"][job["id"]]
                assert record["gpu"] in (5, 6)
                assert env["CUDA_VISIBLE_DEVICES"] == f"GPU-fixture-{record['gpu']}"
                assert len(pass_fds) == 2 and start_new_session is True
                names = {Path(os.readlink(f"/proc/self/fd/{fd}")).name for fd in pass_fds}
                assert names == {f"gpu{record['gpu']}.lock", env["CUDA_VISIBLE_DEVICES"] + ".lock"}
                for fd in pass_fds:
                    # A separate open-file description must fail to take the held lease.
                    with open(os.readlink(f"/proc/self/fd/{fd}"), "a+") as competing:
                        try:
                            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            raise AssertionError("Admission did not hold both exclusive GPU leases")
                assert all(state["jobs"][dep]["status"] == "completed" for dep in job["depends_on"])
                if case == "underlock_recheck" and not launch:
                    assert record["gpu"] == 6, "Under-lock busy GPU was admitted"
                launch.append(dict(job=job["id"], gpu=record["gpu"], phase=job["phase"]))
                out = Path(job["output_dir"])
                out.mkdir(parents=True)
                common.atomic_json(out / "final_summary.json", dict(fixture=job["id"]))
                common.atomic_json(jobfile.parent / "exit.json", dict(job_id=job["id"], returncode=0))
                self.pid = 100000 + len(launch)
                self.polls = 0

            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

        def acceptance(job):
            if case == "failed_validation" and job["id"] == suite["jobs"][0]["id"]:
                raise AssertionError("Injected acceptance failure")
            return dict(summary_sha256=common.sha256(Path(job["output_dir"]) / "final_summary.json"))

        def sleep(seconds):
            nonlocal sleeps
            sleeps += 1
            assert sleeps < 100, "Scheduler fixture stalled"

        fake_subprocess = types.SimpleNamespace(Popen=FakeWorker, STDOUT=-2,
                                                run=lambda *a, **kw: types.SimpleNamespace(returncode=0))
        fake_collector = types.ModuleType("collect_results")
        fake_collector.external_seed2_ready = lambda: True
        with patch.object(q, "H", root), patch.object(q, "load_suite", lambda p: suite), \
             patch.object(q, "assert_host", lambda s: None), \
             patch.object(q, "verify_pins", lambda s: None), \
             patch.object(q, "verify_source", lambda s: None), \
             patch.object(q, "verify_implementation", lambda s: None), \
             patch.object(q, "validate_job", acceptance), \
             patch.object(q, "query_gpus", query), \
             patch.object(q, "subprocess", fake_subprocess), \
             patch.object(q, "proc_identity", lambda pid: dict(pid=pid, proc_starttime=str(pid))), \
             patch.object(q, "time", types.SimpleNamespace(sleep=sleep)), \
             patch.object(sys, "argv", ["run_queue.py", "--suite", str(H / "suite.json")]), \
             patch.dict(sys.modules, {"collect_results": fake_collector}), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = q.main()
        state = json.loads((root / "runtime/queue_state.json").read_text())
        if case == "failed_validation":
            assert rc == 1 and state["status"] == "halted_failure"
            assert len(launch) == 2, "Accepted pending jobs after a validation failure"
        else:
            assert rc == 0 and state["status"] == "completed" and len(launch) == 7
            assert all(r["status"] == "completed" for r in state["jobs"].values())
            assert [j["phase"] for j in launch] == ["single200"] * 4 + ["cl10"] * 3
            if case == "occupied5":
                assert {x["gpu"] for x in launch} == {6}
        return dict(case=case, status="passed", launched=launch, scheduler_queries=query_count)


def main():
    configured = common.load_suite(H / "suite.json")
    assert configured["allowed_gpus"] == [5, 6, 7] and configured["max_parallel"] == 2
    assert len(configured["jobs"]) == 7 and not any(j["smoke"] for j in configured["jobs"])
    for job in configured["jobs"]:
        assert not ({"--max-tasks", "--max-epochs", "--max-batches"} & set(job["command"]))
    rejects = []
    with tempfile.TemporaryDirectory(prefix="response567_allow_") as d:
        for gpus in ([0], [1], [2], [3], [4], [0, 5, 6], [5, 5], [7, 8]):
            s = copy.deepcopy(configured)
            s["allowed_gpus"] = gpus
            s["max_parallel"] = 1
            p = Path(d) / "suite.json"
            p.write_text(json.dumps(s))
            try:
                common.load_suite(p)
            except ValueError:
                rejects.append(gpus)
            else:
                raise AssertionError(f"Unauthorized/invalid GPUs accepted: {gpus}")
    return dict(status="passed", gpu_orchestration_only=True,
                no_cuda_or_training_dispatched=True, invalid_gpu_sets_rejected=rejects,
                cases=[exercise(c) for c in ("healthy_missing7", "occupied5",
                                           "underlock_recheck", "failed_validation")])


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
