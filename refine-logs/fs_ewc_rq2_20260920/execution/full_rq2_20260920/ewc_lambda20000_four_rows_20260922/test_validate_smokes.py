"""CPU regressions for smoke-only release, provenance, and protection evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load('train')
gate = load('validate_smokes')
contract = load('test_train')


class SmokeGate(unittest.TestCase):
    def fixture(self, arm):
        cfg, _ = train.build_config(contract.args(smoke=True, arm=arm))
        parent, optimizer = train.ARMS[arm]
        manifest = {'requested_config': cfg, 'arm': arm, 'seed': 0, 'anchor_method': parent,
                    'future_optimizer': optimizer, 'mode': 'continuation', 'smoke_only': True,
                    'identity_sha256': 'test', 'matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
                    'runner_sha256': train.sha256(HERE / 'train.py')}
        receipt = {'status': 'smoke_complete', 'completed_tasks': 2, 'identity_sha256': 'test'}
        return manifest, receipt

    def test_four_fixed_arms_have_valid_smoke_contract(self):
        for arm in train.ARMS:
            manifest, receipt = self.fixture(arm)
            self.assertEqual(gate.validate_smoke_contract(train, arm, manifest, receipt)['epochs'], 1)

    def test_formal_epoch_count_normalization_and_identity_cannot_pass_smoke(self):
        for key, value in [('epochs', 50), ('ewc_normalize_fisher', True),
                           ('ewc_normalize_accumulated_fisher', True), ('ewc_lambda', 2000),
                           ('rq2_smoke_only', False), ('optimizer_type', 'gam')]:
            manifest, receipt = self.fixture('ewclora')
            manifest['requested_config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate_smoke_contract(train, 'ewclora', manifest, receipt)
        for key, value in [('seed', 1), ('runner_sha256', 'stale'), ('anchor_method', 'gam'),
                           ('smoke_only', False), ('matmul_allow_tf32', True)]:
            manifest, receipt = self.fixture('ewclora'); manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate_smoke_contract(train, 'ewclora', manifest, receipt)
        manifest, receipt = self.fixture('ewclora'); receipt['status'] = 'complete'
        with self.assertRaises(ValueError): gate.validate_smoke_contract(train, 'ewclora', manifest, receipt)

    def rows(self, optimizer):
        return [{'task': 1, 'step': step, 'parameters_finite_after_step': True,
                 'loss': 1. + step, 'ce_clean': 1., 'ewc': float(step), 'grad_combined_norm': 1.,
                 'native_ewc_evaluations': 1, 'single_ewc_injection_verified': True,
                 'gradient_source': f'actual_native_{optimizer}_base_optimizer_gradients_immediately_before_step',
                 'gradient_components_sampled': True, 'component_source': train.gradient_component_source(optimizer),
                 'grad_clean_norm': 1., 'grad_fisher_norm': float(step), 'grad_gam_direction_norm': 2.,
                 'fisher_to_clean_norm_ratio': float(step), 'cos_clean_fisher': 0.}
                for step in range(2)]

    def test_nonzero_step1_guard_for_both_optimizers(self):
        for optimizer in ('sgd', 'gam'):
            cfg = {'optimizer_type': optimizer}
            self.assertEqual(len(train.validate_batch_receipts(self.rows(optimizer), cfg)), 2)
            for key, value in [('grad_fisher_norm', 0.), ('ewc', 0.), ('grad_fisher_norm', float('inf')),
                               ('native_ewc_evaluations', 2), ('single_ewc_injection_verified', False),
                               ('gradient_source', 'proxy'), ('component_source', 'combined_minus_EWC_is_CE')]:
                rows = self.rows(optimizer); rows[1][key] = value
                with self.subTest(optimizer=optimizer, key=key), self.assertRaises((ValueError, FloatingPointError)):
                    train.validate_batch_receipts(rows, cfg)


if __name__ == '__main__':
    unittest.main()
