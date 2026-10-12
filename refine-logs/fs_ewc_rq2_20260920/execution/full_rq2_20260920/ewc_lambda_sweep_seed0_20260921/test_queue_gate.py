"""Fake-GPU tests: no CUDA or nvidia-smi is used by this suite."""
import contextlib
from datetime import datetime
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('lambda_sweep_queue', Path(__file__).with_name('campaign_runner.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def idle(gpu):
    return dict(index=gpu, uuid=f'GPU-{gpu}', memory_mib=2,
                utilization=0, has_compute_process=False)


class AllGpuStartGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.locks = self.root / 'locks'
        self.locks.mkdir()
        self.receipt = self.root / 'TRIAL_RESULT.json'
        worker = "import json,sys;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'status':'observed','outcome':'diverged'}))"
        job = dict(id='trial', gpu=0, gpu_pool=list(range(7)),
                   command=[sys.executable, '-c', worker, str(self.receipt)],
                   receipt=str(self.receipt), receipt_require={'status': 'observed'})
        self.plan = dict(gpus=list(range(7)), max_parallel=2, run_dir=str(self.run),
                         gpu_lock_dir=str(self.locks), code_files=[__file__],
                         start_after={'gpu_uuids': {str(g): f'GPU-{g}' for g in range(7)}},
                         all_gpu_start_gate=dict(required=True, gpus=list(range(7)), poll_seconds=900),
                         stages=[dict(id='sweep', depends_on=[], jobs=[job])])
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def runner(self):
        jobs = queue.validate_plan(self.plan)
        frozen = queue.fingerprint(self.plan['code_files'] + [str(self.plan_path)])
        state = dict(fingerprints=frozen, jobs={j['id']: dict(status='queued', gpu=j['gpu']) for j in jobs})
        return queue.PipelineRunner(self.plan, frozen, state, self.run / 'pipeline_state.json', ())

    def assert_locks_available(self, gpus=range(7)):
        for gpu in gpus:
            with (self.locks / f'GPU-{gpu}.lock').open('a') as lane:
                fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_each_idle_requirement_blocks_and_persists_every_gpu(self):
        cases = [('memory_mib', 500, 'memory_mib_at_least_500'),
                 ('utilization', 5, 'utilization_at_least_5'),
                 ('has_compute_process', True, 'compute_process_present'),
                 ('uuid', 'GPU-wrong', 'uuid_mismatch'),
                 ('error', 'temporary query failure', 'gpu_query_error')]
        for field, value, reason in cases:
            with self.subTest(field=field):
                runner = self.runner()
                def status(gpu):
                    if gpu == 6 and field == 'error':
                        raise RuntimeError(value)
                    return dict(idle(gpu), **({field: value} if gpu == 6 else {}))
                with patch.object(queue, 'gpu_status', side_effect=status) as query:
                    self.assertFalse(runner._check_all_gpu_gate_once())
                self.assertEqual(query.call_count, 7)
                state = json.loads((self.run / 'pipeline_state.json').read_text())
                gate = state['all_gpu_start_gate']
                self.assertEqual(state['status'], 'waiting_for_all_gpus')
                self.assertEqual(gate['blocked_gpus'], [6])
                self.assertEqual(set(gate['gpu_statuses']), {str(g) for g in range(7)})
                self.assertIn(reason, gate['gpu_statuses']['6']['blocked_reasons'])
                self.assertEqual((datetime.fromisoformat(gate['next_poll_utc']) -
                                  datetime.fromisoformat(gate['checked_utc'])).total_seconds(), 900)
                self.assertFalse((self.run / 'ADMISSION.json').exists())
                self.assertFalse(self.receipt.exists())
                self.assert_locks_available()

    def test_idle_gpu_with_busy_shared_lock_blocks_without_reserving_others(self):
        runner = self.runner()
        with (self.locks / 'GPU-6.lock').open('a') as external:
            fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(queue, 'gpu_status', side_effect=idle):
                self.assertFalse(runner._check_all_gpu_gate_once())
            gate = runner.state['all_gpu_start_gate']
            self.assertEqual(gate['blocked_gpus'], [6])
            self.assertEqual(gate['gpu_statuses']['6']['shared_lock'], 'blocked')
            self.assert_locks_available(range(6))
            self.assertFalse((self.run / 'ADMISSION.json').exists())

    def test_second_observation_holds_all_locks_and_detects_new_activity(self):
        runner = self.runner()
        calls = 0
        def status(gpu):
            nonlocal calls
            calls += 1
            if calls > 7:
                for locked_gpu in range(7):
                    with (self.locks / f'GPU-{locked_gpu}.lock').open('a') as lane:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return dict(idle(gpu), has_compute_process=(calls > 7 and gpu == 6))
        with patch.object(queue, 'gpu_status', side_effect=status):
            self.assertFalse(runner._check_all_gpu_gate_once())
        self.assertEqual(calls, 14)
        self.assertEqual(runner.state['all_gpu_start_gate']['blocked_gpus'], [6])
        self.assertFalse((self.run / 'ADMISSION.json').exists())
        self.assert_locks_available()

    def test_admission_receipt_precedes_job_and_observed_divergence_is_accepted(self):
        busy = True
        waited = []
        original_barrier = queue.PipelineRunner.wait_for_all_gpus
        original_run_job = queue.PipelineRunner.run_job
        def status(gpu):
            return dict(idle(gpu), memory_mib=900 if busy and gpu == 6 else 2)
        def fake_wait(seconds):
            nonlocal busy
            self.assertFalse(self.receipt.exists())
            self.assertFalse((self.run / 'ADMISSION.json').exists())
            waited.append(seconds)
            busy = False
            return False
        def barrier(runner):
            with patch.object(runner.stop, 'wait', side_effect=fake_wait):
                return original_barrier(runner)
        def launch(runner, job):
            self.assertFalse(busy)
            self.assertEqual(runner.state['all_gpu_start_gate']['status'], 'admitted')
            admission = json.loads((self.run / 'ADMISSION.json').read_text())
            self.assertEqual(admission['shared_locks_acquired'], list(range(7)))
            self.assert_locks_available(range(1, 7))
            return original_run_job(runner, job)
        with patch.object(queue, 'gpu_status', side_effect=status), \
             patch.object(queue.PipelineRunner, 'wait_for_all_gpus', new=barrier), \
             patch.object(queue.PipelineRunner, 'run_job', new=launch):
            state = queue.execute_plan(self.plan, self.plan_path)
        self.assertEqual(len(waited), 1)
        self.assertGreater(waited[0], 895)
        self.assertLessEqual(waited[0], 900)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(json.loads(self.receipt.read_text())['outcome'], 'diverged')
        self.assert_locks_available()
        # Completed resume validates admission and trial receipts without polling.
        with patch.object(queue, 'gpu_status', side_effect=AssertionError('one-time gate queried again')):
            resumed = queue.execute_plan(self.plan, self.plan_path, resume=True)
        self.assertEqual(resumed['status'], 'complete')
        with (self.run / 'ADMISSION.json').open('a') as receipt:
            receipt.write(' ')
        with self.assertRaisesRegex(RuntimeError, 'admission receipt changed or invalid'):
            queue.execute_plan(self.plan, self.plan_path, resume=True)

    def test_pending_poll_time_survives_resume_and_lock_pass_releases_all_cards(self):
        runner = self.runner()
        with patch.object(queue, 'gpu_status', side_effect=lambda g: dict(idle(g), utilization=10)):
            runner._check_all_gpu_gate_once()
        previous = json.loads((self.run / 'pipeline_state.json').read_text())
        resumed = queue.PipelineRunner(self.plan, runner.frozen, previous, runner.state_file, ())
        with patch.object(resumed.stop, 'wait', return_value=False) as wait, \
             patch.object(queue, 'gpu_status', side_effect=idle) as query:
            resumed.wait_for_all_gpus()
        wait.assert_called_once()
        self.assertGreater(wait.call_args.args[0], 895)
        self.assertEqual(query.call_count, 14)
        self.assert_locks_available()

    def test_dry_run_does_not_query_or_create_admission(self):
        with patch.object(sys, 'argv', ['campaign_runner.py', '--plan', str(self.plan_path)]), \
             patch.object(queue, 'gpu_status', side_effect=AssertionError('GPU query in dry run')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            queue.main()
        self.assertTrue(json.loads(output.getvalue())['dry_run'])
        self.assertFalse((self.run / 'pipeline_state.json').exists())
        self.assertFalse((self.run / 'ADMISSION.json').exists())

    def test_required_gate_configuration_is_validated(self):
        self.plan['all_gpu_start_gate']['gpus'] = list(range(6))
        with self.assertRaisesRegex(ValueError, 'GPUs 0 through 6'):
            queue.validate_plan(self.plan)
        self.plan['all_gpu_start_gate']['gpus'] = list(range(7))
        self.plan['all_gpu_start_gate']['poll_seconds'] = 60
        with self.assertRaisesRegex(ValueError, 'must be 900'):
            queue.validate_plan(self.plan)
        self.plan['all_gpu_start_gate']['poll_seconds'] = 900
        del self.plan['start_after']['gpu_uuids']['6']
        with self.assertRaisesRegex(ValueError, 'Expected GPU UUID required'):
            queue.validate_plan(self.plan)


if __name__ == '__main__':
    unittest.main()
