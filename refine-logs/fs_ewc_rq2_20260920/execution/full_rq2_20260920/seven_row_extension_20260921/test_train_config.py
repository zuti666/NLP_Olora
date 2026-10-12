"""CPU-only configuration/provenance and real-receipt regression tests."""
import copy
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('extension_train', HERE / 'train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def args(arm='shaping_only', seed=0, smoke=False):
    anchor, future, lam = train.ARMS[arm]
    root = train.FORMAL / 'formal_training' / f'anchor_{anchor}_seed{seed}'
    argv = ['--arm', arm, '--anchor-method', anchor, '--future-optimizer', future,
            '--seed', str(seed), '--ewc-lambda', str(lam), '--run-name', f'{arm}_seed{seed}',
            '--anchor-dir', str(root / 'checkpoints'), '--anchor-metrics', str(root / 'trainer_outputs')]
    return train.parser().parse_args(argv + (['--smoke'] if smoke else []))


class ExtensionContract(unittest.TestCase):
    def test_all_six_formal_configs_have_only_task1_and_gam(self):
        for arm in train.ARMS:
            for seed in (0, 1, 2):
                ns = args(arm, seed); cfg, out = train.build_config(ns)
                self.assertEqual((cfg['epochs'], cfg['max_train_tasks'], cfg['resume_from_task']), (50, 2, 0))
                self.assertEqual((cfg['optimizer_type'], cfg['lambda_flat'], cfg['train_num_workers']), ('gam', 1., 4))
                self.assertEqual(cfg['ewc_lambda'], train.ARMS[arm][2])
                self.assertFalse(cfg['rq2_smoke_only'])
                self.assertEqual(out, HERE / 'training' / f'{arm}_seed{seed}')

    def test_arm_identity_cannot_be_mislabeled(self):
        ns = args('task2_shaping'); ns.anchor_method = 'gam'
        with self.assertRaises(ValueError): train.build_config(ns)
        ns = args(); ns.future_optimizer = 'sgd'
        with self.assertRaises(ValueError): train.build_config(ns)
        ns = args(); ns.ewc_lambda = 2000
        with self.assertRaises(ValueError): train.build_config(ns)

    def test_explicit_smoke_does_not_change_formal_config_or_outputs(self):
        cfg, out = train.build_config(args(smoke=True))
        formal, formal_out = train.build_config(args())
        self.assertEqual(cfg['epochs'], 1)
        self.assertTrue(out.is_relative_to(HERE / 'smoke'))
        self.assertEqual(formal['epochs'], 50)
        self.assertNotEqual(out, formal_out)

    def test_wrong_parent_seed_is_rejected_before_training(self):
        ns = args(seed=1); ns.anchor_dir = train.FORMAL / 'formal_training/anchor_gam_seed0/checkpoints'
        cfg, _ = train.build_config(ns)
        with self.assertRaises(ValueError): train.validate_parent(ns, cfg, {})

    def test_real_fifty_epoch_plus_fit_fisher_receipts(self):
        path = train.FORMAL / 'formal_training/sequence_gam_seed0/rng_receipts.jsonl'
        rows = [r for r in train.read_jsonl(path) if r.get('event') == 'loader_iteration'
                and r.get('task') == 1 and r.get('shuffle')]
        self.assertEqual(len(rows), 51)
        train.validate_loader_passes(rows, 50)
        train.validate_loader_passes(rows[:2], 1)
        with self.assertRaises(ValueError): train.validate_loader_passes(rows[:50], 50)
        broken = copy.deepcopy(rows); broken[-1]['samples'] -= 1
        with self.assertRaises(ValueError): train.validate_loader_passes(broken, 50)

    def test_frozen_file_logger_final_epoch_only_is_valid(self):
        root = train.FORMAL / 'formal_training/sequence_gam_seed0/trainer_outputs'
        logs = '\n'.join(p.read_text() for p in root.rglob('*.log'))
        train.validate_final_epoch_log(logs, 1, 50)
        with self.assertRaises(ValueError): train.validate_final_epoch_log('Task 1, Epoch 49/50', 1, 50)

    def test_real_all_six_parent_identities(self):
        for arm in train.ARMS:
            for seed in (0, 1, 2):
                ns = args(arm, seed); cfg, _ = train.build_config(ns)
                identity = train.experiment_identity(ns, cfg)
                self.assertTrue(identity['parent_validation']['passed'])
                self.assertEqual(identity['anchor_method'], train.ARMS[arm][0])
                self.assertEqual(identity['future_optimizer'], 'gam')
                self.assertEqual(identity['arm'], arm)
                self.assertEqual(identity['parent_validation']['seed'], seed)


if __name__ == '__main__':
    unittest.main()
