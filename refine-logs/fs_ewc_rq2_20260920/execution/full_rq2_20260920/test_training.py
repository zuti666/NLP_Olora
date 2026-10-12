"""CPU contract checks for the full RQ2 training runner."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('rq2_full_train', Path(__file__).with_name('train.py'))
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class TrainingContract(unittest.TestCase):
    def args(self, *args):
        return train.parser().parse_args(['--run-name', 'unit_run', *args])

    def test_four_cell_optimizer_control(self):
        for method in ('sgd', 'gam'):
            for lam in (0, 2000):
                cfg, _ = train.build_config(self.args('--mode', 'continuation', '--method', method,
                                                     '--ewc-lambda', str(lam)))
                self.assertEqual(cfg['optimizer_type'], 'sgd')
                self.assertEqual(cfg['lambda_flat'], 0)
                self.assertEqual(cfg['ewc_lambda'], lam)
                self.assertEqual(cfg['max_train_tasks'], 2)

    def test_cold_start_has_no_task_zero_resume(self):
        for mode in ('anchor', 'sequence'):
            cfg, _ = train.build_config(self.args('--mode', mode, '--method', 'gam', '--seed', '1'))
            self.assertNotIn('resume_from_checkpoint_dir', cfg)
            self.assertEqual(cfg['optimizer_type'], 'gam')
            self.assertEqual(cfg['lambda_flat'], 1)
            self.assertEqual(cfg['max_train_tasks'], 1 if mode == 'anchor' else 10)

    def test_sequence_own_anchor_preserves_method(self):
        cfg, _ = train.build_config(self.args('--mode', 'sequence', '--method', 'gam', '--seed', '1',
                                             '--anchor-dir', '/tmp/anchor/checkpoints',
                                             '--anchor-metrics', '/tmp/anchor/metrics.json'))
        self.assertEqual(cfg['resume_from_task'], 0)
        self.assertEqual(cfg['optimizer_type'], 'gam')

    def test_invalid_output_name(self):
        with self.assertRaises(ValueError):
            train.build_config(self.args('--mode', 'anchor', '--method', 'sgd', '--run-name', '../bad'))

    def make_run(self, root, final=True, epochs=True):
        out = root / 'trainer_outputs'
        out.mkdir()
        matrix = [[90, None], [85, 80]]
        (out / 'run_cl_metrics.json').write_text(json.dumps({'cnn': {'matrices': {'final' if final else 't01': matrix}}}))
        (out / 'run.log').write_text('Task 1, Epoch 50/50 => Loss 0.1' if epochs else 'Task 1, Epoch 49/50 => Loss 0.1')
        train.append_json(root / 'runtime_receipts.jsonl', {'event': 'task_complete', 'task': 1})
        for idx in range(50):
            train.append_json(root / 'rng_receipts.jsonl', {'event': 'loader_iteration', 'task': 1,
                              'shuffle': True, 'iteration': idx, 'samples': 100, 'sample_count': 100})
        (root / 'checkpoints').mkdir()
        for name in ('task_01.pt', 'fisher_state_1.pt', 'rng_state_01.pt'):
            (root / 'checkpoints' / name).write_bytes(b'fixture')

    def test_checkpoint_without_final_matrix_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root, final=False)
            with self.assertRaises(ValueError):
                train.validate_completion(root, {'max_train_tasks': 2, 'resume_from_checkpoint_dir': 'anchor'}, 'continuation')

    def test_missing_epoch_fifty_is_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root, epochs=False)
            with self.assertRaises(ValueError):
                train.validate_completion(root, {'max_train_tasks': 2, 'resume_from_checkpoint_dir': 'anchor'}, 'continuation')

    def test_completion_records_assets_and_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_run(root)
            result = train.validate_completion(root, {'max_train_tasks': 2, 'resume_from_checkpoint_dir': 'anchor'}, 'continuation')
            self.assertEqual(result['validated_tasks'], [1])
            self.assertEqual(result['epochs_per_new_task'], 50)
            self.assertEqual(len(result['checkpoint_assets']), 3)


if __name__ == '__main__':
    unittest.main()
