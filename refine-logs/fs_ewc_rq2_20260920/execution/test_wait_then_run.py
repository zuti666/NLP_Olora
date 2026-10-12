"""CPU-only gate tests: no test queries or allocates an actual GPU."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import wait_then_run as queue


class StopPolling(Exception):
    pass


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.predecessor = self.root / "predecessor.json"
        self.receipt = self.root / "final_results.json"
        self.source = self.root / "source.py"
        self.source.write_text("# pinned source\n")
        self.predecessor.write_text(json.dumps({"status": "completed"}))
        self.receipt.write_text(json.dumps({"status": "complete"}))
        self.plan = {
            "run_dir": str(self.root / "run"), "gpus": [0, 1], "code_files": [str(self.source)],
            "pipeline_command": [sys.executable, "-c", "print('pipeline ran')"],
            "gpu_lock_dir": str(self.root / "locks"), "cwd": str(self.root),
            "start_after": {
                "state_file": str(self.predecessor), "final_receipt": str(self.receipt),
                "required_status": "completed", "required_receipt_status": "complete",
                "gpus": [0, 1, 2, 3], "idle_seconds": 60, "poll_seconds": 30,
                "supervisor_identity": {"pid": 99999999, "start_ticks": 1, "boot_id": "test-boot"},
                "gpu_uuids": {str(i): f"GPU-test-{i}" for i in range(4)},
            },
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.cards = {f"GPU-test-{i}": {"index": i, "memory_mib": 0, "utilization_pct": 0, "compute_pids": []} for i in range(4)}
        self.telemetry = mock.patch.object(queue, "nvidia_snapshot", return_value=self.cards).start()
        self.addCleanup(mock.patch.stopall)

    def state(self):
        return queue.read_json(Path(self.plan["run_dir"]) / "queue_state.json")

    def assert_does_not_launch(self):
        with mock.patch.object(queue.time, "sleep", side_effect=StopPolling), \
             mock.patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaises(StopPolling):
                queue.execute(self.plan_path, self.plan)
        launch.assert_not_called()
        state = self.state()
        self.assertFalse(state["gate"]["ready"])
        self.assertNotIn("pipeline_started_at", state)
        # The injected polling exception is persisted as a failure after the gate check.
        self.assertEqual(state["status"], "failed")

    def test_running_predecessor_and_idle_gpus_do_not_start(self):
        self.predecessor.write_text(json.dumps({"status": "running"}))
        self.assert_does_not_launch()
        self.assertIn("predecessor status", self.state()["gate"]["reasons"][0])

    def test_failed_predecessor_is_blocked_and_does_not_start(self):
        self.predecessor.write_text(json.dumps({"status": "failed"}))
        observed = []
        original_save = queue.save_state
        def capture(run_dir, state):
            observed.append(state["status"])
            original_save(run_dir, state)
        with mock.patch.object(queue, "save_state", side_effect=capture):
            self.assert_does_not_launch()
        self.assertIn("blocked", observed)

    def test_completed_predecessor_alive_supervisor_does_not_start(self):
        self.plan["start_after"]["supervisor_identity"] = queue.process_identity(os.getpid())
        self.assert_does_not_launch()
        self.assertTrue(self.state()["gate"]["predecessor_supervisor_alive"])

    def test_busy_memory_utilization_or_compute_process_each_blocks(self):
        for field, value in (("memory_mib", 500), ("utilization_pct", 5), ("compute_pids", [1234])):
            with self.subTest(field=field):
                old = self.cards["GPU-test-3"][field]
                self.cards["GPU-test-3"][field] = value
                gate = queue.inspect_gate(self.plan)
                self.assertFalse(gate["ready"])
                self.assertIn("GPU 3 is busy", gate["reasons"])
                self.cards["GPU-test-3"][field] = old

    def test_gpu_query_error_does_not_start(self):
        self.telemetry.side_effect = subprocess.TimeoutExpired("nvidia-smi", 20)
        self.assert_does_not_launch()
        self.assertTrue(self.state()["gate"]["errors"])

    def test_missing_or_malformed_final_receipt_blocks(self):
        self.receipt.unlink()
        self.assertFalse(queue.inspect_gate(self.plan)["ready"])
        self.receipt.write_text("{broken")
        self.assertFalse(queue.inspect_gate(self.plan)["ready"])

    def test_uuid_index_mismatch_blocks(self):
        self.cards["GPU-test-2"]["index"] = 7
        self.assertFalse(queue.inspect_gate(self.plan)["ready"])

    def test_process_identity_rejects_pid_reuse(self):
        identity = queue.process_identity(os.getpid())
        self.assertTrue(queue.identity_alive(identity))
        identity["start_ticks"] += 1
        self.assertFalse(queue.identity_alive(identity))

    def test_shared_gpu_lock_blocks(self):
        streams = queue.acquire_gpu_locks(self.plan)
        try:
            gate = queue.inspect_gate(self.plan)
            self.assertFalse(gate["ready"])
            self.assertEqual(set(gate["blocked_gpu_locks"]), {"GPU-test-0", "GPU-test-1"})
        finally:
            for stream in streams.values():
                stream.close()

    def test_check_once_does_not_create_state_or_lock_directory(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = queue.main(["--plan", str(self.plan_path), "--check-once"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["ready"])
        self.assertFalse(Path(self.plan["run_dir"]).exists())
        self.assertFalse(Path(self.plan["gpu_lock_dir"]).exists())

    def test_source_change_during_wait_fails_without_launch(self):
        self.predecessor.write_text(json.dumps({"status": "running"}))
        def change(_):
            self.source.write_text("# unexpected new source\n")
        with mock.patch.object(queue.time, "sleep", side_effect=change), mock.patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "changed while queued"):
                queue.execute(self.plan_path, self.plan)
        launch.assert_not_called()
        self.assertEqual(self.state()["status"], "failed")

    def test_stable_idle_period_required_and_resets_on_busy(self):
        run_dir = Path(self.plan["run_dir"])
        run_dir.mkdir()
        state = {"status": "waiting", "plan_path": str(self.plan_path), "queue_identity": queue.process_identity(os.getpid())}
        ready = {"ready": True, "reasons": [], "errors": []}
        busy = {"ready": False, "reasons": ["GPU busy"], "errors": []}
        # Initial idle 0,30; busy at60 resets; next idle90,120,150 succeeds.
        with mock.patch.object(queue, "inspect_gate", side_effect=[ready, ready, busy, ready, ready, ready, ready]), \
             mock.patch.object(queue.time, "monotonic", side_effect=[0, 30, 60, 90, 120, 150]), \
             mock.patch.object(queue.time, "sleep") as sleep:
            locks = queue.wait_until_ready(self.plan_path, self.plan, queue.snapshot_files(self.plan_path, self.plan), state, run_dir)
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual(state["idle_elapsed_seconds"], 60)
        for stream in locks.values():
            stream.close()

    def test_pipeline_inherits_gpu_and_queue_locks_and_records_completion(self):
        script = """import fcntl,json,os,pathlib
locks=json.loads(os.environ['RQ2_GPU_LOCK_FDS'])
assert len(locks)==2
for uuid,fd in locks.items():
    os.fstat(fd)
    with open(pathlib.Path('locks')/(uuid+'.lock'),'a+') as other:
        try:
            fcntl.flock(other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('GPU lock did not survive exec')
os.fstat(int(os.environ['RQ2_QUEUE_LOCK_FD']))
print('inherited locks verified')
"""
        self.plan["pipeline_command"] = [sys.executable, "-c", script]
        with mock.patch.object(queue, "wait_until_ready", side_effect=lambda *args: queue.acquire_gpu_locks(self.plan)):
            self.assertEqual(queue.execute(self.plan_path, self.plan), 0)
        self.assertEqual(self.state()["status"], "completed")
        self.assertEqual(self.state()["returncode"], 0)
        self.assertIn("inherited locks verified", (Path(self.plan["run_dir"]) / "pipeline.log").read_text())
        with self.assertRaisesRegex(RuntimeError, "already launched"):
            queue.execute(self.plan_path, self.plan)

    def test_live_child_identity_rejects_duplicate_start(self):
        run_dir = Path(self.plan["run_dir"])
        run_dir.mkdir()
        (run_dir / "queue_state.json").write_text(json.dumps({"status": "running", "child_identity": queue.process_identity(os.getpid())}))
        with mock.patch.object(queue.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                queue.execute(self.plan_path, self.plan)
        launch.assert_not_called()

    def test_four_gpu_pipeline_inherits_all_shared_reservations(self):
        self.plan['gpus'] = [0, 1, 2, 3]
        script = """import fcntl,json,os,pathlib
locks=json.loads(os.environ['RQ2_GPU_LOCK_FDS'])
assert set(locks)=={'GPU-test-0','GPU-test-1','GPU-test-2','GPU-test-3'}
for uuid,fd in locks.items():
    os.fstat(fd)
    with open(pathlib.Path('locks')/(uuid+'.lock'),'a+') as other:
        try:
            fcntl.flock(other.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('A GPU reservation was released before pipeline completion')
os.fstat(int(os.environ['RQ2_QUEUE_LOCK_FD']))
"""
        self.plan['pipeline_command'] = [sys.executable, '-c', script]
        self.plan_path.write_text(json.dumps(self.plan))
        queue.validate_plan(self.plan)
        with mock.patch.object(queue, 'wait_until_ready', side_effect=lambda *args: queue.acquire_gpu_locks(self.plan)):
            self.assertEqual(queue.execute(self.plan_path, self.plan), 0)
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(queue.gpu_locks_available(self.plan), [])

    def test_held_queue_lock_rejects_second_supervisor(self):
        run_dir = Path(self.plan["run_dir"])
        run_dir.mkdir()
        with (run_dir / "queue.lock").open("a+") as held:
            queue.fcntl.flock(held.fileno(), queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
            with mock.patch.object(queue.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "already holds queue.lock"):
                    queue.execute(self.plan_path, self.plan)
            launch.assert_not_called()

    def test_gpu_csv_unavailable_values_fail_closed(self):
        with mock.patch.object(queue.subprocess, "run", return_value=mock.Mock(stdout="0, GPU-test-0, [N/A], 0\n")):
            with self.assertRaises(ValueError):
                # Bypass the per-test nvidia_snapshot mock.
                self.original_nvidia_snapshot()


# Preserve the real parser before per-test telemetry mocking.
QueueTests.original_nvidia_snapshot = staticmethod(queue.nvidia_snapshot)

if __name__ == "__main__":
    unittest.main()
