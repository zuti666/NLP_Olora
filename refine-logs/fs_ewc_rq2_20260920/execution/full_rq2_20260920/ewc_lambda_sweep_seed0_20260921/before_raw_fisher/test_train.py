"""CPU regression tests for the sweep contract and nonmutating diagnostics."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lambda_train', HERE / 'train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def args(lam=1e5, smoke=False):
    argv = ['--seed', '0', '--ewc-lambda', str(lam), '--run-name', 'contract_seed0']
    return train.parser().parse_args(argv + (['--smoke'] if smoke else []))


class SweepContract(unittest.TestCase):
    def test_arbitrary_finite_lambda_without_grid_constraint(self):
        for lam in (0., 17.3, 2000., 3e5, 1e8, 1e9):
            ns = args(lam)
            cfg, out = train.build_config(ns)
            self.assertEqual(cfg['ewc_lambda'], lam)
            self.assertEqual((cfg['epochs'], cfg['max_train_tasks'], cfg['resume_from_task']), (50, 2, 0))
            self.assertEqual((cfg['optimizer_type'], cfg['lambda_flat'], cfg['seed']), ('sgd', 0., [0]))
            self.assertTrue(cfg['ewc_normalize_fisher'])
            self.assertTrue(cfg['ewc_normalize_accumulated_fisher'])
            self.assertEqual(out, HERE / 'training/contract_seed0')

    def test_invalid_lambda_rejected_even_when_parser_bypassed(self):
        for lam in (-1, math.nan, math.inf, -math.inf, 1e9 + 1):
            with self.assertRaises(Exception): train.finite_ewc_lambda(lam)
            ns = args(); ns.ewc_lambda = lam
            with self.assertRaises(Exception): train.build_config(ns)

    def test_no_other_arm_method_or_seed(self):
        for key, value in [('arm', 'task2_shaping'), ('seed', 1), ('anchor_method', 'gam'),
                           ('future_optimizer', 'gam')]:
            ns = args(); setattr(ns, key, value)
            with self.assertRaises(ValueError): train.build_config(ns)
        self.assertEqual(train.parser().parse_args(['--seed0', '--ewc-lambda', '2000', '--run-name', 'alias']).seed, 0)

    def test_smoke_and_formal_paths_do_not_overlap(self):
        cfg, out = train.build_config(args(smoke=True))
        formal, formal_out = train.build_config(args())
        self.assertEqual(cfg['epochs'], 1)
        self.assertTrue(out.is_relative_to(HERE / 'smoke'))
        self.assertEqual(formal['epochs'], 50)
        self.assertNotEqual(out, formal_out)
        ns = args(); ns.output_root = HERE / 'smoke/training'
        with self.assertRaises(ValueError): train.build_config(ns)

    def test_parent_and_lambda_identity(self):
        identities = []
        for lam in (0, 1e8):
            ns = args(lam); cfg, _ = train.build_config(ns)
            identity = train.experiment_identity(ns, cfg)
            self.assertTrue(identity['parent_validation']['passed'])
            self.assertEqual(identity['anchor_method'], 'sgd')
            self.assertEqual(identity['future_optimizer'], 'sgd')
            self.assertEqual(identity['seed'], 0)
            identities.append(identity)
        self.assertEqual(identities[0]['parent'], identities[1]['parent'])
        self.assertNotEqual(identities[0]['identity_sha256'], identities[1]['identity_sha256'])
        ns = args(); ns.anchor_dir = train.FORMAL / 'formal_training/anchor_gam_seed0/checkpoints'
        cfg, _ = train.build_config(ns)
        with self.assertRaises(ValueError): train.validate_parent(ns, cfg, {})

    def test_real_sgd_fifty_plus_fisher_passes(self):
        root = train.FORMAL / 'formal_training/sequence_sgd_seed0'
        rows = [r for r in train.read_jsonl(root / 'rng_receipts.jsonl')
                if r.get('event') == 'loader_iteration' and r.get('task') == 1 and r.get('shuffle')]
        self.assertEqual(len(rows), 51)
        train.validate_loader_passes(rows, 50)
        train.validate_loader_passes(rows[:2], 1)
        with self.assertRaises(ValueError): train.validate_loader_passes(rows[:50], 50)
        broken = copy.deepcopy(rows); broken[-1]['samples'] -= 1
        with self.assertRaises(ValueError): train.validate_loader_passes(broken, 50)
        logs = '\n'.join(p.read_text() for p in (root / 'trainer_outputs').rglob('*.log'))
        train.validate_final_epoch_log(logs, 1, 50)


class GradientObservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        sys.path.insert(0, str(train.EXECUTION / 'source'))
        from models_LoRAbasedCL.ewclora_youyue_fitArchitechture import Learner
        cls.native_penalty = staticmethod(Learner._past_fisher_penalty)

    def tiny_model(self, lam):
        torch = self.torch
        net = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad(): net.weight.copy_(torch.tensor([[.2, .3], [.4, -.1]]))
        model = types.SimpleNamespace(_ewc_lambda=lam, _ewc_eta=0., _device=torch.device('cpu'),
            _fisher_past_delta={'q': torch.tensor([[.1, .2], [.3, .4]])},
            _delta_reference={'q': torch.zeros(2, 2)})
        model._unwrap_network = lambda: net
        model._iter_delta_terms = lambda unused: [('q', net.weight @ net.weight.T)]
        model._past_fisher_penalty = types.MethodType(self.native_penalty, model)
        return model, net

    def test_native_penalty_components_preserve_grad_rng_and_update(self):
        torch = self.torch
        for lam in (0., 2000.):
            model, net = self.tiny_model(lam)
            optimizer = torch.optim.SGD(net.parameters(), lr=.02, momentum=0, weight_decay=0)
            clean_loss = net.weight.square().sum()
            clean_grad, = torch.autograd.grad(clean_loss, [net.weight], retain_graph=True)
            total = clean_loss + model._past_fisher_penalty(net)
            total.backward()
            weight, grad, rng = net.weight.detach().clone(), net.weight.grad.clone(), torch.get_rng_state().clone()
            expected = weight.clone(); expected.add_(grad, alpha=-.02)
            record = train.observe_sgd_gradients(model, optimizer.param_groups, True)
            self.assertTrue(torch.equal(weight, net.weight))
            self.assertTrue(torch.equal(grad, net.weight.grad))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertAlmostEqual(record['grad_clean_norm'], clean_grad.double().norm().item(), places=4)
            if lam == 0: self.assertEqual(record['grad_fisher_norm'], 0.)
            optimizer.step()
            self.assertTrue(torch.equal(expected, net.weight))

    def test_nonfinite_actual_gradient_stops_before_update(self):
        torch = self.torch
        model, net = self.tiny_model(1e8)
        optimizer = torch.optim.SGD(net.parameters(), lr=.02)
        net.weight.grad = torch.full_like(net.weight, math.inf)
        before = net.weight.detach().clone()
        with self.assertRaises(FloatingPointError):
            train.observe_sgd_gradients(model, optimizer.param_groups, True)
        self.assertTrue(torch.equal(before, net.weight))


if __name__ == '__main__':
    unittest.main()
