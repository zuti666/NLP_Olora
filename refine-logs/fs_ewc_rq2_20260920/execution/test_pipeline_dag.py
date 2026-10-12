"""CPU-only integration checks for the RQ2 dependency scheduler.

The jobs are short real Python subprocesses. GPU observations are mocked and
CUDA_VISIBLE_DEVICES is checked as an ordinary environment string; no CUDA or
nvidia-smi is used by these tests.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rq2_pipeline', HERE / 'pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

WORKER = r'''
import json, os, pathlib, sys, time
cfg = json.loads(sys.argv[1])
root = pathlib.Path(cfg['root'])
start = time.monotonic_ns()
(root / (cfg['id'] + '.started')).write_text(str(start))
if cfg.get('expected_cuda') is not None:
    assert os.environ['CUDA_VISIBLE_DEVICES'] == cfg['expected_cuda']
for fd in cfg.get('check_fds', []):
    os.fstat(fd)
if cfg.get('barrier'):
    deadline = time.monotonic() + 5
    while not all((root / (jid + '.started')).exists() for jid in cfg['barrier']):
        if time.monotonic() > deadline:
            raise RuntimeError('Independent branches did not overlap')
        time.sleep(.01)
time.sleep(cfg.get('delay', .03))
result = dict(cfg.get('result', {'status': 'complete', 'passed': True}))
result.update(start_ns=start, end_ns=time.monotonic_ns())
pathlib.Path(cfg['receipt']).write_text(json.dumps(result))
sys.exit(cfg.get('exit', 0))
'''


class PipelineDAGTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rq2-dag-test-')
        self.root = Path(self.temp.name)
        self.script = self.root / 'worker.py'
        self.script.write_text(WORKER)
        self.plan_path = self.root / 'plan.json'
        self.plan = dict(gpus=[0, 1, 2, 3], max_parallel=4,
                         run_dir=str(self.root / 'run'), code_files=[str(self.script)], stages=[])
        self.environment = mock.patch.dict(os.environ, {'RQ2_GPU_LOCK_FDS': '{}', 'RQ2_QUEUE_LOCK_FD': ''})
        self.environment.start()
        self.gpus = mock.patch.object(pipeline, 'gpu_status', side_effect=self.idle_gpu)
        self.gpu_mock = self.gpus.start()

    def tearDown(self):
        self.gpus.stop()
        self.environment.stop()
        self.temp.cleanup()

    @staticmethod
    def idle_gpu(gpu):
        return dict(index=gpu, uuid=f'GPU-mock-{gpu}', memory_mib=0,
                    utilization=0, has_compute_process=False)

    def job(self, jid, gpu, **options):
        receipt = self.root / f'{jid}.json'
        requirements = options.pop('requirements', {'status': 'complete'})
        cfg = dict(root=str(self.root), id=jid, receipt=str(receipt),
                   expected_cuda=f'GPU-mock-{gpu}' if gpu is not None else '', **options)
        return dict(id=jid, gpu=gpu, command=[sys.executable, str(self.script), json.dumps(cfg)],
                    receipt=str(receipt), receipt_require=requirements)

    def stage(self, sid, deps, jobs):
        self.plan['stages'].append(dict(id=sid, depends_on=deps, jobs=jobs))

    def run_plan(self, resume=False):
        if not resume:
            self.plan_path.write_text(json.dumps(self.plan))
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.execute_plan(self.plan, self.plan_path, resume=resume)

    def state(self):
        return json.loads((self.root / 'run' / 'pipeline_state.json').read_text())

    def receipt(self, jid):
        return json.loads((self.root / f'{jid}.json').read_text())

    def test_two_lines_overlap_after_both_sanity_jobs_and_report_joins(self):
        self.stage('sanity', [], [self.job('sanity_sgd', 0, delay=.03),
                                  self.job('sanity_gam', 1, delay=.15)])
        branch_ids = ['neighborhood_sgd', 'neighborhood_gam', 'training_sgd', 'training_gam']
        for sid, deps, gpus in [('neighborhood', ['sanity'], [0, 1]),
                                ('training', ['sanity'], [2, 3]),
                                ('bridge', ['neighborhood'], [0, 1]),
                                ('table_a', ['training'], [2, 3]),
                                ('trajectory', ['table_a'], [2, 3]),
                                ('path', ['trajectory'], [2, 3])]:
            options = {'barrier': branch_ids, 'delay': .08} if sid in ('neighborhood', 'training') else {}
            self.stage(sid, deps, [self.job(f'{sid}_{method}', gpu, **options)
                                   for method, gpu in zip(('sgd', 'gam'), gpus)])
        self.stage('report', ['bridge', 'path'], [self.job('report', None)])
        snapshots = []
        real_write = pipeline.write_json

        def capture(path, state):
            snapshots.append(copy.deepcopy(state))
            real_write(path, state)

        with mock.patch.object(pipeline, 'write_json', side_effect=capture):
            state = self.run_plan()
        self.assertEqual(state['status'], 'complete')
        sanity_end = max(self.receipt(j)['end_ns'] for j in ('sanity_sgd', 'sanity_gam'))
        starts = [self.receipt(j)['start_ns'] for j in branch_ids]
        ends = [self.receipt(j)['end_ns'] for j in branch_ids]
        self.assertGreater(min(starts), sanity_end)
        self.assertLess(max(starts), min(ends), 'All four branch workers must overlap')
        self.assertTrue(any({'neighborhood', 'training'} <= set(s['active_stages']) for s in snapshots))
        by_stage = {s['id']: s for s in self.plan['stages']}
        for stage in self.plan['stages']:
            for dependency in stage['depends_on']:
                predecessor_end = max(self.receipt(j['id'])['end_ns'] for j in by_stage[dependency]['jobs'])
                for job in stage['jobs']:
                    self.assertGreater(self.receipt(job['id'])['start_ns'], predecessor_end)
        self.assertEqual(state['active_stages'], [])

    def test_ready_stages_sharing_one_gpu_never_overlap(self):
        self.stage('one', [], [self.job('one', 0, delay=.12)])
        self.stage('two', [], [self.job('two', 0)])
        self.run_plan()
        self.assertGreater(self.receipt('two')['start_ns'], self.receipt('one')['end_ns'])

    def test_failure_drains_running_job_and_blocks_successors(self):
        self.stage('root', [], [self.job('bad', 0, exit=4, delay=.1), self.job('survivor', 1, delay=.3)])
        self.stage('child', ['root'], [self.job('never', 2)])
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.run_plan()
        state = self.state()
        self.assertEqual(state['jobs']['bad']['status'], 'failed')
        self.assertEqual(state['jobs']['survivor']['status'], 'complete')
        self.assertEqual(state['jobs']['never']['status'], 'blocked')
        self.assertFalse((self.root / 'never.started').exists())
        self.assertGreater(self.receipt('survivor')['end_ns'], self.receipt('bad')['end_ns'])

    def test_failure_interrupts_waiting_for_busy_gpu(self):
        def gpu_status(gpu):
            row = self.idle_gpu(gpu)
            if gpu == 1:
                row.update(memory_mib=5000, utilization=100, has_compute_process=True)
            return row
        self.gpu_mock.side_effect = gpu_status
        self.stage('root', [], [self.job('bad', 0, exit=1, delay=.15), self.job('waiter', 1)])
        started = time.monotonic()
        with self.assertRaises(RuntimeError):
            self.run_plan()
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(self.state()['jobs']['waiter']['status'], 'cancelled')
        self.assertFalse((self.root / 'waiter.started').exists())

    def test_failed_receipt_gates_block_followup_even_with_zero_exit(self):
        gates = [({'status': 'complete', 'passed': True}, {'status': 'complete', 'passed': False}),
                 ({'status': 'complete', 'quadrature_converged': True,
                   'bound_passed_with_2pct_numeric_tolerance': True},
                  {'status': 'complete', 'quadrature_converged': True,
                   'bound_passed_with_2pct_numeric_tolerance': False})]
        for i, (required, result) in enumerate(gates):
            with self.subTest(gate=i):
                self.plan['run_dir'] = str(self.root / f'run{i}')
                self.plan['stages'] = []
                self.stage('gate', [], [self.job(f'gate{i}', 0, requirements=required, result=result)])
                self.stage('next', ['gate'], [self.job(f'next{i}', 1)])
                with self.assertRaises(RuntimeError):
                    self.run_plan()
                state = json.loads((Path(self.plan['run_dir']) / 'pipeline_state.json').read_text())
                self.assertFalse(state['jobs'][f'gate{i}']['receipt_valid'])
                self.assertFalse((self.root / f'next{i}.started').exists())

    def test_invalid_dependency_graph_rejected(self):
        self.stage('a', [], [self.job('a', None)])
        self.stage('b', ['a'], [self.job('b', None)])
        for kind in ('unknown', 'cycle', 'duplicate_stage', 'duplicate_dependency', 'duplicate_job'):
            with self.subTest(kind=kind):
                broken = copy.deepcopy(self.plan)
                if kind == 'unknown':
                    broken['stages'][1]['depends_on'] = ['missing']
                elif kind == 'cycle':
                    broken['stages'][0]['depends_on'] = ['b']
                elif kind == 'duplicate_stage':
                    broken['stages'][1]['id'] = 'a'
                elif kind == 'duplicate_dependency':
                    broken['stages'][1]['depends_on'] = ['a', 'a']
                else:
                    broken['stages'][1]['jobs'][0]['id'] = 'a'
                with self.assertRaises(ValueError):
                    pipeline.validate_plan(broken)

    def test_resume_checks_hash_and_gate_before_launch(self):
        self.stage('gate', [], [self.job('gate', None, requirements={'status': 'complete', 'passed': True})])
        first = self.run_plan()
        original = (self.root / 'gate.json').read_bytes()
        resumed = self.run_plan(resume=True)
        self.assertEqual(resumed['jobs']['gate']['attempt'], 1)
        (self.root / 'gate.json').write_bytes(original + b'\n')
        with self.assertRaisesRegex(RuntimeError, 'receipt changed or invalid'):
            self.run_plan(resume=True)
        invalid = json.loads(original)
        invalid['passed'] = False
        payload = json.dumps(invalid).encode()
        (self.root / 'gate.json').write_bytes(payload)
        first['jobs']['gate']['receipt_sha256'] = hashlib.sha256(payload).hexdigest()
        (self.root / 'run' / 'pipeline_state.json').write_text(json.dumps(first))
        with self.assertRaisesRegex(RuntimeError, 'receipt changed or invalid'):
            self.run_plan(resume=True)

    def test_resume_rejects_matching_live_process_identity(self):
        self.stage('a', [], [self.job('a', None)])
        self.plan_path.write_text(json.dumps(self.plan))
        run = Path(self.plan['run_dir'])
        run.mkdir()
        state = dict(fingerprints=pipeline.fingerprint(self.plan['code_files'] + [str(self.plan_path)]),
                     jobs={'a': dict(status='running', gpu=None, pid=os.getpid(),
                                     process_identity=pipeline.process_identity(os.getpid()))})
        (run / 'pipeline_state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, 'prior job is alive'):
            self.run_plan(resume=True)
        self.assertFalse((self.root / 'a.started').exists())

    def test_preflight_failure_never_launches_experiment(self):
        job = self.job('a', 0)
        job['preflight_command'] = [sys.executable, '-c', 'raise SystemExit(7)']
        self.stage('a', [], [job])
        with self.assertRaises(RuntimeError):
            self.run_plan()
        self.assertEqual(self.state()['jobs']['a']['exit_code'], 7)
        self.assertFalse((self.root / 'a.started').exists())

    def test_lock_descriptors_reach_preflight_and_job(self):
        with (self.root / 'gpu.lock').open('a') as gpu_lock, (self.root / 'queue.lock').open('a') as queue_lock:
            fds = [gpu_lock.fileno(), queue_lock.fileno()]
            os.environ['RQ2_GPU_LOCK_FDS'] = json.dumps({'GPU-mock-0': fds[0]})
            os.environ['RQ2_QUEUE_LOCK_FD'] = str(fds[1])
            job = self.job('a', 0, check_fds=fds)
            job['preflight_command'] = [sys.executable, '-c', f'import os; [os.fstat(fd) for fd in {fds!r}]']
            self.stage('a', [], [job])
            self.assertEqual(self.run_plan()['status'], 'complete')

    def test_dryrun_is_cpu_only_and_does_not_create_run_directory(self):
        self.stage('a', [], [self.job('a', 0)])
        self.plan_path.write_text(json.dumps(self.plan))
        self.gpu_mock.side_effect = AssertionError('Dryrun queried GPU')
        output = io.StringIO()
        with mock.patch.object(sys, 'argv', ['pipeline.py', '--plan', str(self.plan_path)]), contextlib.redirect_stdout(output):
            pipeline.main()
        self.assertTrue(json.loads(output.getvalue())['dry_run'])
        self.assertFalse(Path(self.plan['run_dir']).exists())
        self.gpu_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
