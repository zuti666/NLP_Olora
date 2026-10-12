#!/usr/bin/env python3
"""CPU-only queue admission, failure isolation, source freezing and metric tests."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


queue = load("run_g2_orders_queue")
collector = load("collect_g2_orders_results")


class Handle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRunner:
    class GPUUnavailableError(RuntimeError):
        pass

    def __init__(self, busy_first=False):
        self.busy_first = busy_first
        self.claim_calls = 0
        self.handles = []
        self.verified = []

    def claim_gpus(self, gpus):
        if gpus != "0,1,2,3":
            raise AssertionError("Unauthorized physical GPU selection")
        self.claim_calls += 1
        if self.busy_first and self.claim_calls == 1:
            raise self.GPUUnavailableError("GPU 2 busy")
        handles = [Handle() for _ in range(4)]
        self.handles.extend(handles)
        return [{"index": i, "uuid": f"GPU-{i}"} for i in range(4)], handles

    def environment(self, source, root, gpus, spec):
        return {"CUDA_VISIBLE_DEVICES": ",".join(g["uuid"] for g in gpus)}

    def verify_stage(self, spec):
        self.verified.append((spec["smoke"], spec["stage"]))
        return {"status": "pass", "stage": spec["stage"], "smoke": spec["smoke"]}


class QueueTests(unittest.TestCase):
    def fixture(self, root):
        artifact = root / "bulk"
        artifact.mkdir()
        job_id = "inclora_g2_llama1b_order1"
        stages = []
        for smoke, count in ((True, 2), (False, 4)):
            for stage in range(1, count + 1):
                output = artifact / ("smoke" if smoke else "formal") / str(stage)
                previous = output.parent / str(stage - 1) / "adapter" if stage > 1 else None
                stages.append({"seed": 0, "stage": stage, "smoke": smoke, "previous_adapter": str(previous) if previous else None,
                               "output_dir": str(output), "command": ["fake-stage", str(output), str(stage)]})
        source = root / "source"
        source.mkdir()
        job = {"id": job_id, "seed": 0, "model_size": "1B", "order": 1,
               "config": {"python_executable": "/fake/python"}, "plan": {"stages": stages}}
        manifest = {"schema": queue.SCHEMA, "queue_root": str(root), "artifact_root": str(artifact),
                    "source": str(source), "repository": str(root), "jobs": [job]}
        queue.write_json(root / "runtime/queue_state.json", {"status": "prepared", "completed_stages": []})
        return manifest

    def stage_child(self, command, logfile, env, cwd, handles, heartbeat, **kwargs):
        heartbeat(12345)
        output = Path(command[1])
        (output / "adapter").mkdir(parents=True)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text("fake completed\n")
        heartbeat(None)

    def patches(self, manifest, runner, child=None, witness=None):
        return [mock.patch.object(queue, "verify_manifest", return_value=manifest),
                mock.patch.object(queue, "import_runner", return_value=runner),
                mock.patch.object(queue, "run_witnesses", side_effect=witness),
                mock.patch.object(queue, "run_child", side_effect=child or self.stage_child),
                mock.patch.object(queue.subprocess, "run", return_value=types.SimpleNamespace(returncode=0, stderr=""))]

    def invoke(self, patches, root, sleep=None):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(queue.time, "sleep", side_effect=sleep))
            return queue.run(root, wait_seconds=.001)

    def test_waits_for_all_four_and_smoke_admits_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            runner = FakeRunner(busy_first=True)
            self.assertEqual(self.invoke(self.patches(manifest, runner), root), 0)
            state = json.loads((root / "runtime/queue_state.json").read_text())
            self.assertEqual(state["status"], "completed")
            self.assertEqual((state["formal_completed"], state["smoke_completed"]), (4, 2))
            self.assertEqual(runner.claim_calls, 2)
            self.assertEqual(runner.verified, [(True, 1), (True, 2), (False, 1), (False, 2), (False, 3), (False, 4)])
            self.assertTrue(all(handle.closed for handle in runner.handles))

    def test_smoke_failure_stops_before_any_formal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            runner = FakeRunner()
            calls = []

            def fail_first(command, *args, **kwargs):
                calls.append(command)
                raise subprocess.CalledProcessError(17, command)

            with self.assertRaises(subprocess.CalledProcessError):
                self.invoke(self.patches(manifest, runner, child=fail_first), root)
            state = json.loads((root / "runtime/queue_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["formal_completed"], 0)
            self.assertEqual(len(calls), 1)
            self.assertTrue(all(handle.closed for handle in runner.handles))

    def test_existing_interrupted_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            output = Path(manifest["jobs"][0]["plan"]["stages"][0]["output_dir"])
            output.mkdir(parents=True)
            sentinel = output / "preserve.txt"
            sentinel.write_text("user data")
            with self.assertRaises(FileExistsError):
                self.invoke(self.patches(manifest, FakeRunner()), root)
            self.assertEqual(sentinel.read_text(), "user data")

    def test_pause_after_stage_releases_only_own_leases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            runner = FakeRunner()
            calls = []

            def stage_then_pause(*args, **kwargs):
                self.stage_child(*args, **kwargs)
                calls.append(args[0])
                if len(calls) == 1:
                    (root / "runtime/PAUSE").touch()

            def release_pause(_):
                self.assertTrue(all(handle.closed for handle in runner.handles))
                (root / "runtime/PAUSE").unlink()

            self.assertEqual(self.invoke(self.patches(manifest, runner, child=stage_then_pause), root, sleep=release_pause), 0)
            self.assertEqual(runner.claim_calls, 2)

    def test_manifest_or_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            frozen = source / "implementation.py"
            frozen.write_text("x=1\n")
            original = root / "data.json"
            original.write_text("[]")
            manifest = {"schema": queue.SCHEMA, "queue_root": str(root), "source": str(source),
                "gpus": "0,1,2,3", "seeds": [0], "source_sha256": {"implementation.py": queue.digest(frozen)},
                "input_sha256": {str(original): {"sha256": queue.digest(original), "bytes": original.stat().st_size}}}
            queue.write_json(root / "manifest.json", manifest)
            (root / "manifest.sha256").write_text(queue.digest(root / "manifest.json"))
            self.assertEqual(queue.verify_manifest(root)["schema"], queue.SCHEMA)
            frozen.write_text("x=2\n")
            with self.assertRaisesRegex(RuntimeError, "Frozen source changed"):
                queue.verify_manifest(root)
            frozen.write_text("x=1\n")
            original.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Frozen local input changed"):
                queue.verify_manifest(root)
            original.write_text("[]")
            (root / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "prepared manifest changed"):
                queue.verify_manifest(root)

    def test_completed_queue_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue.write_json(root / "runtime/queue_state.json", {"status": "completed", "completed_stages": []})
            with mock.patch.object(queue, "verify_manifest", side_effect=AssertionError("No GPU or source work expected")):
                self.assertEqual(queue.run(root), 0)

    def test_witness_is_exact_seed0_sentinel_and_bf16_backward(self):
        code = queue.witness_code()
        self.assertIn("torch.manual_seed(0)", code)
        self.assertIn('torch.randn(8,8,device="cuda")', code)
        self.assertIn("y=x@x", code)
        self.assertIn("z.backward()", code)
        self.assertIn("torch.isfinite(w.grad).all()", code)

    def test_progress_ignores_incomplete_appended_trace_row(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "g2_step_trace.jsonl").write_text('{"step":1}\n{"step":2}\n{"step":3')
            self.assertEqual(queue.latest_trace_step(str(root)), 2)
            self.assertIsNone(queue.latest_trace_step(str(root / "absent")))

    def test_stop_while_waiting_never_admits_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            runner = FakeRunner(busy_first=True)
            self.assertEqual(self.invoke(self.patches(manifest, runner), root,
                sleep=lambda _: (root / "runtime/STOP_AFTER_STAGE").touch()), 0)
            state = json.loads((root / "runtime/queue_state.json").read_text())
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(runner.claim_calls, 1)
            self.assertEqual(runner.verified, [])

    def test_stop_while_paused_never_claims_gpus(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            runner = FakeRunner()
            (root / "runtime/PAUSE").touch()
            self.assertEqual(self.invoke(self.patches(manifest, runner), root,
                sleep=lambda _: (root / "runtime/STOP_AFTER_STAGE").touch()), 0)
            state = json.loads((root / "runtime/queue_state.json").read_text())
            self.assertEqual(state["status"], "stopped")
            self.assertEqual(runner.claim_calls, 0)


class MetricTests(unittest.TestCase):
    def test_exact_match_normalization_does_not_remove_articles(self):
        self.assertEqual(collector.normalize("  OFFICE-Holder!  "), "officeholder")
        self.assertNotEqual(collector.normalize("a business"), collector.normalize("business"))

    def test_raw_prediction_population_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            rows = [{"Dataset": "dbpedia", "Instance": {"id": str(i), "label": "Office Holder"}, "Prediction": p}
                    for i, p in enumerate(("office holder!", "Album"))]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            metrics = collector.prediction_metrics(path, ["dbpedia"], expected_per_task=2)
            self.assertEqual(metrics["accuracy"], {"dbpedia": 50.0})
            path.write_text("\n".join(json.dumps(row) for row in (rows[0], rows[0])))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collector.prediction_metrics(path, ["dbpedia"], expected_per_task=2)

    def test_frozen_gold_labels_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            row = {"Dataset": "dbpedia", "Instance": {"id": "0", "label": "Wrong Label"}, "Prediction": "Wrong Label"}
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "label differs"):
                collector.prediction_metrics(path, ["dbpedia"], expected_per_task=1, gold={"dbpedia": [{"label": "Office Holder"}]})

    def test_alias_id_cannot_replace_a_missing_canonical_row(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            rows = [{"Dataset": "dbpedia", "Instance": {"id": identity, "label": "Office Holder"}, "Prediction": "Office Holder"}
                    for identity in ("0", "00")]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            for gold in (None, {"dbpedia": [{"label": "Office Holder"}, {"label": "Album"}]}):
                with self.assertRaisesRegex(ValueError, "canonical test-set index"):
                    collector.prediction_metrics(path, ["dbpedia"], expected_per_task=2, gold=gold)

    def test_missing_final_test_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.jsonl"
            path.write_text(json.dumps({"Dataset": "dbpedia", "Instance": {"id": "0", "label": "Album"}, "Prediction": "Album"}))
            with self.assertRaisesRegex(ValueError, "population differs"):
                collector.prediction_metrics(path, ["dbpedia"], expected_per_task=2)

    def test_manual_collection_rejects_manifest_and_input_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline.csv"
            baseline.write_text("model,method,seed,order,faa,run_dir\nLlama-3.2-1B,IncLoRA,0,1,61.2,seed0\n")
            files = [baseline]
            tasks = list(collector.TASK_RELATIVE)
            for task in tasks:
                path = root / "data" / collector.TASK_RELATIVE[task] / "test.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([{"label": "label"}] * 7600))
                files.append(path)
            manifest = {"schema": queue.SCHEMA, "repository": str(root), "input_sha256":
                {str(path.resolve()): {"sha256": collector.sha256(path), "bytes": path.stat().st_size} for path in files},
                "jobs": [{"id": "g2", "model_size": "1B", "order": 1, "seed": 0,
                    "config": {"tasks": tasks, "data_dir": str(root / "data")}, "plan": {"stages": []}}]}
            queue.write_json(root / "manifest.json", manifest)
            (root / "manifest.sha256").write_text(collector.sha256(root / "manifest.json"))
            queue.write_json(root / "runtime/queue_state.json", {"status": "prepared", "completed_stages": []})
            result = collector.collect(root, baseline_path=baseline)
            self.assertEqual(result["order_results"][0]["FAA"], "")
            original = baseline.read_text()
            baseline.write_text(original.replace("61.2", "99.2"))
            with self.assertRaisesRegex(ValueError, "Frozen collector input changed"):
                collector.collect(root, baseline_path=baseline)
            baseline.write_text(original)
            test_path = files[1]
            original_test = test_path.read_text()
            test_path.write_text(original_test.replace("label", "LABEL", 1))
            with self.assertRaisesRegex(ValueError, "Frozen collector input changed"):
                collector.collect(root, baseline_path=baseline)
            test_path.write_text(original_test)
            (root / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "collector manifest changed"):
                collector.collect(root, baseline_path=baseline)

    def test_stage_cost_reports_trace_counts_fallback_and_timings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"step": i, "g2_ce_gradient_phases": phases, "g2_replay_closure_calls": phases - 1,
                     "g2_final_direction": "g2" if phases == 3 else "g0_fallback",
                     "g2_fallback_reason": None if phases == 3 else "zero_response"}
                    for i, phases in enumerate((3, 2, 3), 1)]
            trace = root / "g2_step_trace.jsonl"
            trace.write_text("\n".join(json.dumps(row) for row in rows))
            queue.write_json(root / "train_results.json", {"train_runtime": 10.0})
            queue.write_json(root / "predict_results.json", {"predict_runtime": 3.2})
            receipt = {"global_step": 3, "g2_fallback_steps": 1, "g2_trace_sha256": collector.sha256(trace)}
            spec = {"output_dir": str(root), "expected_updates": 3}
            result = collector.stage_cost_metrics(spec, receipt)
            self.assertEqual(result["global_updates"], 3)
            self.assertEqual(result["ce_gradient_phases_total"], 8)
            self.assertEqual(result["replay_closure_calls_total"], 5)
            self.assertEqual(result["fallback_fraction"], 1 / 3)
            self.assertEqual(result["train_runtime_seconds"], 10.0)
            self.assertEqual(result["predict_runtime_seconds"], 3.2)
            queue.write_json(root / "train_results.json", {"train_runtime": "NaN"})
            with self.assertRaisesRegex(ValueError, "invalid or nonfinite"):
                collector.stage_cost_metrics(spec, receipt)
            queue.write_json(root / "train_results.json", {"train_runtime": 10.0})
            trace.write_text(trace.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "trace changed"):
                collector.stage_cost_metrics(spec, receipt)

    def test_triangular_metrics_closed_form(self):
        tasks = ["dbpedia", "amazon", "yahoo", "agnews"]
        accuracies = [[80], [70, 60], [65, 55, 75], [60, 50, 70, 90]]
        stages = {stage: {"accuracy": dict(zip(tasks[:stage], values)), "seen_task_accuracy": sum(values) / stage,
                          "counts": {task: 7600 for task in tasks[:stage]}}
                  for stage, values in enumerate(accuracies, 1)}
        metrics = collector.continual_metrics(tasks, stages)
        self.assertEqual(metrics["FAA"], 67.5)
        self.assertAlmostEqual(metrics["AAA"], (80 + 65 + 65 + 67.5) / 4)
        self.assertAlmostEqual(metrics["BWT"], (-20 - 10 - 5) / 3)
        self.assertEqual(metrics["test_samples"], 30400)
        with self.assertRaisesRegex(ValueError, "all four"):
            collector.continual_metrics(tasks, {1: stages[1]})

    def test_baseline_uses_matching_seed_instead_of_multiseed_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.csv"
            path.write_text("model,method,seed,order,faa,run_dir\nLlama-3.2-1B,IncLoRA,0,1,61.2,seed0\n"
                            "Llama-3.2-1B,IncLoRA,42,1,79.5,seed42\nLlama-3.2-1B,FS-EWCLoRA,0,1,90.1,other\n")
            baseline = collector.read_baseline(path)
            self.assertEqual(float(baseline[("Llama-3.2-1B", 1, 0)]["faa"]), 61.2)
            self.assertEqual(len(baseline), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
