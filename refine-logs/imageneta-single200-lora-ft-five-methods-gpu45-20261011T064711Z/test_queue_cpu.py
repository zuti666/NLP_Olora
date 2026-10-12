"""Deterministic host-independent admission/lease and manifest regression checks."""
import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common
import run_queue

class AdmissionTests(unittest.TestCase):
    def test_idle_thresholds_are_strict_and_compute_pid_blocks(self):
        g = dict(memory_used_mb=499, utilization_percent=4, compute_pids=[])
        self.assertTrue(common.gpu_idle(g))
        for key, value in (("memory_used_mb", 500), ("utilization_percent", 5), ("compute_pids", [123])):
            self.assertFalse(common.gpu_idle({**g, key: value}))

    def test_physical_index_and_uuid_leases_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = dict(gpu_lock_dir=tmp, allowed_gpus=[4, 5])
            g4 = dict(index=4, uuid="GPU-test4", memory_used_mb=0, utilization_percent=0, compute_pids=[])
            g5 = dict(index=5, uuid="GPU-test5", memory_used_mb=0, utilization_percent=0, compute_pids=[])
            with open(Path(tmp) / "GPU-test4.lock", "a+") as block:
                fcntl.flock(block, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(run_queue, "query_gpus", return_value={4: g4, 5: g5}):
                    chosen = run_queue.acquire_idle_gpu(suite, set())
                self.assertEqual(chosen[0], 5)
                self.assertEqual(len(chosen[2]), 2)
                for lease in chosen[2]: lease.close()

    def test_gpu_becoming_busy_after_lock_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = dict(gpu_lock_dir=tmp, allowed_gpus=[4])
            free = dict(index=4, uuid="GPU-test4", memory_used_mb=0, utilization_percent=0, compute_pids=[])
            busy = {**free, "compute_pids": [456]}
            with patch.object(run_queue, "query_gpus", side_effect=[{4: free}, {4: busy}]):
                self.assertIsNone(run_queue.acquire_idle_gpu(suite, set()))

    def test_paused_gpu_is_excluded_before_nvml_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(run_queue, "query_gpus") as query:
                self.assertIsNone(run_queue.acquire_idle_gpu(dict(gpu_lock_dir=tmp, allowed_gpus=[4, 5]), {4, 5}))
                query.assert_not_called()

    def test_actual_suite_has_required_dag_and_only_authorized_gpu(self):
        suite = common.load_suite(common.HERE / "suite.json")
        self.assertEqual(len(suite["jobs"]), 18)
        self.assertEqual(suite["allowed_gpus"], [4, 5])

    def test_manifest_rejects_unapproved_gpu_and_missing_gate(self):
        original = common.read_json(common.HERE / "suite.json")
        for change in ("gpu", "gate", "scope"):
            suite = copy.deepcopy(original)
            if change == "gpu": suite["allowed_gpus"] = [0, 1]
            elif change == "gate":
                formal = next(j for j in suite["jobs"] if j["kind"] == "training" and not j["smoke"])
                formal["depends_on"] = []
            else:
                next(j for j in suite["jobs"] if j["kind"] == "training")["train_scope"] = "illegal"
            with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
                json.dump(suite, f); f.flush()
                with self.assertRaises((ValueError, KeyError)):
                    common.load_suite(f.name)

    def test_nonfinite_evidence_is_rejected(self):
        for x in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(AssertionError): common.finite_tree({"nested": [x]})

if __name__ == "__main__":
    unittest.main(verbosity=2)
