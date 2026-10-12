"""CPU regression tests for the four-row raw-Fisher contract and nonmutating diagnostics."""
import copy
import importlib.util
import math
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lambda_train', HERE / 'train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def args(lam=20000, smoke=False, arm='ewclora', seed=0):
    anchor, future = train.ARMS[arm]
    parent = train.FORMAL / 'formal_training' / f'anchor_{anchor}_seed{seed}'
    fisher = HERE / 'prepared_fisher' / f'{anchor}_seed{seed}' / 'fisher_state_raw.pt'
    argv = ['--seed', str(seed), '--ewc-lambda', str(lam), '--run-name', f'{arm}_seed{seed}',
            '--arm', arm, '--anchor-method', anchor, '--future-optimizer', future,
            '--anchor-dir', str(parent / 'checkpoints'), '--anchor-metrics', str(parent / 'trainer_outputs'),
            '--prepared-fisher', str(fisher)]
    return train.parser().parse_args(argv + (['--smoke'] if smoke else []))


class FourRowContract(unittest.TestCase):
    def test_all_four_arms_three_seeds(self):
        for arm, (anchor, future) in train.ARMS.items():
            for seed in range(3):
                with self.subTest(arm=arm, seed=seed):
                    cfg, out = train.build_config(args(arm=arm, seed=seed))
                    self.assertEqual(cfg['ewc_lambda'], 20000.)
                    self.assertEqual((cfg['epochs'], cfg['max_train_tasks'], cfg['resume_from_task']), (50, 2, 0))
                    self.assertEqual((cfg['optimizer_type'], cfg['lambda_flat'], cfg['seed']), (future, float(future == 'gam'), [seed]))
                    self.assertEqual(cfg['rq2_anchor_method'], anchor)
                    self.assertFalse(cfg['ewc_normalize_fisher'])
                    self.assertFalse(cfg['ewc_normalize_accumulated_fisher'])
                    self.assertEqual(out, HERE / 'training' / f'{arm}_seed{seed}')
                    self.assertEqual((cfg['lrate'], cfg['batch_size'], cfg['train_num_workers']), (.02, 128, 4))

    def test_only_manual_lambda_even_when_parser_bypassed(self):
        for lam in (-1, 0, 19999, 20001, math.nan, math.inf, -math.inf):
            with self.assertRaises(Exception): train.finite_ewc_lambda(lam)
            ns = args(); ns.ewc_lambda = lam
            with self.assertRaises(Exception): train.build_config(ns)

    def test_boundary_mutations_rejected(self):
        for key, value in [('arm', 'unknown'), ('seed', 3), ('anchor_method', 'gam'),
                           ('future_optimizer', 'gam'), ('source', train.FORMAL),
                           ('workers', 0), ('max_train_tasks', 3), ('run_name', '../escape'),
                           ('anchor_dir', train.FORMAL / 'formal_training/anchor_sgd_seed1/checkpoints'),
                           ('prepared_fisher', HERE / 'prepared_fisher/gam_seed0/fisher_state_raw.pt')]:
            ns = args(); setattr(ns, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError): train.build_config(ns)
        for gpu in range(7):
            self.assertIn(str(gpu), next(a for a in train.parser()._actions if a.dest == 'gpu').choices)

    def test_smoke_and_formal_paths_do_not_overlap(self):
        cfg, out = train.build_config(args(smoke=True))
        formal, formal_out = train.build_config(args())
        self.assertEqual(cfg['epochs'], 1)
        self.assertTrue(out.is_relative_to(HERE / 'smoke'))
        self.assertEqual(formal['epochs'], 50)
        self.assertNotEqual(out, formal_out)
        for folder in ('smoke/training', 'cpu_validation'):
            ns = args(); ns.output_root = HERE / folder
            with self.assertRaises(ValueError): train.build_config(ns)

    def test_actual_six_parent_identities_and_source(self):
        for arm in ('ewclora', 'fs_ewclora'):
            for seed in range(3):
                ns = args(arm=arm, seed=seed); cfg, _ = train.build_config(ns)
                identity = train.experiment_identity(ns, cfg, allow_missing_prepared=True)
                self.assertTrue(identity['parent_validation']['passed'])
                self.assertEqual(identity['anchor_method'], ns.anchor_method)
                self.assertEqual(identity['seed'], seed)
                self.assertEqual(identity['parent_validation']['formal_parent'], str(ns.anchor_dir.parent))

    def test_real_sgd_fifty_plus_fisher_passes(self):
        root = train.FORMAL / 'formal_training/sequence_sgd_seed0'
        rows = [r for r in train.read_jsonl(root / 'rng_receipts.jsonl')
                if r.get('event') == 'loader_iteration' and r.get('task') == 1 and r.get('shuffle')]
        train.validate_loader_passes(rows, 50)
        train.validate_loader_passes(rows[:2], 1)
        with self.assertRaises(ValueError): train.validate_loader_passes(rows[:50], 50)
        broken = copy.deepcopy(rows); broken[-1]['samples'] -= 1
        with self.assertRaises(ValueError): train.validate_loader_passes(broken, 50)
        logs = '\n'.join(p.read_text() for p in (root / 'trainer_outputs').rglob('*.log'))
        train.validate_final_epoch_log(logs, 1, 50)


class NativeSeedMutation(unittest.TestCase):
    def test_requested_and_runtime_seed_forms(self):
        for seed in range(3):
            for value in (seed, [seed], (seed,)):
                cfg = {'seed': value}
                self.assertEqual(train.config_seed(cfg), seed)
                self.assertEqual(cfg['seed'], value)

    def test_ambiguous_or_invalid_seed_forms_rejected(self):
        for value in ([], [0, 1], [[0]], '0', 0., True, None, -1, 3, [True], [3]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train.config_seed({'seed': value})

    def test_six_real_prepared_payloads_after_native_seed_mutation(self):
        # The native trainer mutates the SAME dictionary from [seed] to seed.
        # Exercise the actual resume-time identity and raw-payload load with it.
        import torch
        old_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        self.addCleanup(torch.set_num_threads, old_threads)
        for arm in ('ewclora', 'fs_ewclora'):
            for seed in range(3):
                with self.subTest(arm=arm, seed=seed):
                    ns = args(arm=arm, seed=seed)
                    cfg, _ = train.build_config(ns)
                    requested = copy.deepcopy(cfg)
                    before = train.prepared_fisher_identity(cfg)
                    cfg['seed'] = cfg['seed'][0]
                    parent_fisher = torch.load(ns.anchor_dir / 'fisher_state_0.pt',
                                               map_location='cpu', weights_only=True, mmap=True)['fisher_past_delta']
                    payload, after = train.load_prepared_fisher(cfg, {key: value.shape for key, value in parent_fisher.items()})
                    self.assertEqual(before, after)
                    self.assertEqual(after['seed'], seed)
                    self.assertEqual(after['anchor_method'], ns.anchor_method)
                    self.assertEqual(len(payload['fisher_past_delta']), 24)
                    self.assertGreater(after['raw_trace'], 0.)
                    self.assertEqual(requested['seed'], [seed])
                    self.assertEqual(cfg['seed'], seed)


class PreparedRawFisher(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.directory = tempfile.TemporaryDirectory(prefix='raw_fisher_contract_')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fisher_state_raw.pt'
        self.cfg, _ = train.build_config(args())
        self.cfg['rq2_prepared_fisher'] = str(self.path)

    def prepare_fixture(self, negative=False, wrong_shape=False):
        torch = self.torch
        values = {f'layer{i}': torch.full((2, 2), .25) for i in range(24)}
        if negative: values['layer0'][0, 0] = -.25
        if wrong_shape: values['layer0'] = torch.ones(3, 2)
        trace = sum(float(v.double().sum()) for v in values.values())
        torch.save({'fisher_past_delta': values, 'raw_trace': trace, 'stats': {'n': 24}}, self.path)
        receipt = {'status': 'complete', 'seed': self.cfg['seed'][0], 'anchor_method': self.cfg['rq2_anchor_method'], 'task': 0,
            'ewc_normalize_fisher': False, 'ewc_normalize_accumulated_fisher': False,
            'fisher_mode': 'batch_empirical', 'fisher_source': 'fit', 'tensor_count': 24,
            'weights_unchanged': True, 'exact_inputs_receipt': {'matched': True},
            'raw_trace': trace, 'numel': sum(v.numel() for v in values.values()),
            'parent_checkpoint_sha256': train.sha256(Path(self.cfg['resume_from_checkpoint_dir']) / 'task_00.pt'),
            'source_manifest_sha256': train.sha256(train.EXECUTION / 'source_frozen.json'),
            'payload_sha256': train.sha256(self.path)}
        train.write_json(self.path.parent / 'PREPARED.json', receipt)
        return receipt

    def test_missing_preparation_allows_preview_but_blocks_execution(self):
        identity = train.prepared_fisher_identity(self.cfg, allow_missing=True)
        self.assertEqual(identity['status'], 'pending')
        self.assertFalse(identity['execution_ready'])
        with self.assertRaises(FileNotFoundError): train.prepared_fisher_identity(self.cfg)

    def test_native_raw_tensor_payload_checked_without_rescaling(self):
        self.prepare_fixture()
        payload, identity = train.load_prepared_fisher(self.cfg, {f'layer{i}': (2, 2) for i in range(24)})
        self.assertEqual(identity['raw_trace'], 24.)
        self.assertEqual(float(payload['fisher_past_delta']['layer0'].sum()), 1.)
        self.assertEqual(sum(float(v.sum()) for v in payload['fisher_past_delta'].values()), 24.)

    def test_normalized_or_unpaired_receipt_and_tampering_rejected(self):
        for key, value in [('ewc_normalize_fisher', True), ('ewc_normalize_accumulated_fisher', True),
                           ('parent_checkpoint_sha256', 'wrong'), ('source_manifest_sha256', 'wrong'),
                           ('payload_sha256', 'wrong'), ('exact_inputs_receipt', {'matched': False}),
                           ('seed', 1), ('anchor_method', 'gam')]:
            receipt = self.prepare_fixture(); receipt[key] = value
            train.write_json(self.path.parent / 'PREPARED.json', receipt)
            with self.assertRaises(ValueError): train.prepared_fisher_identity(self.cfg)

    def test_negative_and_wrong_shape_fisher_rejected(self):
        for kwargs in ({'negative': True}, {'wrong_shape': True}):
            self.prepare_fixture(**kwargs)
            with self.assertRaises(ValueError):
                train.load_prepared_fisher(self.cfg, {f'layer{i}': (2, 2) for i in range(24)})


class GradientObservation(unittest.TestCase):
    def test_one_epoch_smoke_samples_after_leaving_anchor(self):
        self.assertEqual([i for i in range(8) if train.sample_gradient_components(i)], [0, 1])
        self.assertEqual([i for i in range(17) if train.sample_gradient_components(i)], [0, 1, 8, 16])

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
        for lam in (0., 20000.):
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

    def test_gam_uses_clean_cache_not_combined_minus_ewc(self):
        torch = self.torch
        model, net = self.tiny_model(20000.)
        optimizer = torch.optim.SGD(net.parameters(), lr=.02)
        clean = torch.ones_like(net.weight)
        fisher = torch.full_like(net.weight, 2.)
        gam = torch.full_like(net.weight, 4.)
        net.weight.grad = gam + fisher
        old_weight, old_grad, rng = net.weight.detach().clone(), net.weight.grad.clone(), torch.get_rng_state().clone()
        record = train.observe_gam_gradients(optimizer.param_groups, {id(net.weight): clean}, {id(net.weight): fisher}, True)
        self.assertEqual(record['grad_clean_norm'], 2.)
        self.assertEqual(record['grad_gam_direction_norm'], 8.)
        self.assertEqual(record['grad_fisher_norm'], 4.)
        self.assertTrue(torch.equal(old_weight, net.weight))
        self.assertTrue(torch.equal(old_grad, net.weight.grad))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        fisher[0, 0] = math.inf
        with self.assertRaises(FloatingPointError):
            train.observe_gam_gradients(optimizer.param_groups, {id(net.weight): clean}, {id(net.weight): fisher}, False)

    def test_two_native_gam_updates_with_raw_fisher_and_cached_ce(self):
        # Execute the real frozen learner/GAM optimizer on a tiny CPU network.
        # This exercises four native closures, cached EWC, and the native
        # direction projector without constructing or running the ViT.
        torch = self.torch
        from models_LoRAbasedCL.as1_normfisher_gam_intask import Learner as Native
        from optimer_PerturabtionType.optimer_gam_filiter import GAM_Filiter

        class TinyNet(torch.nn.Linear):
            def forward(self, x):
                return {'logits': super().forward(x)}

        class TinyLearner(Native):
            def __init__(self):
                self._network = TinyNet(2, 2, bias=False)
                with torch.no_grad():
                    self._network.weight.copy_(torch.tensor([[.2, .3], [.4, -.1]]))
                self._optimizer_type = 'gam'
                self._ewc_lambda = self._lambda_ewc = 20000.
                self._lambda_flat, self._ewc_eta, self._mechanism_eps = 1., 0., 1e-12
                self._mechanism_grad_stats = self._mechanism_eval = True
                self._mechanism_grad_sums, self._mechanism_grad_count = {}, 0
                self._mechanism_loss_sums, self._mechanism_loss_count = {}, 0
                self._device = torch.device('cpu')
                self._fisher_past_delta = {'q': torch.full((2, 2), 1e-4)}
                self._delta_reference = {'q': self._network.weight.detach().clone()}
                self.clean_cache = self.fisher_cache = None
                self.penalty_calls = 0
            def _unwrap_network(self): return self._network
            def _iter_delta_terms(self, net): return [('q', net.weight)]
            def _task_loss(self, logits, targets, class_offset):
                return torch.nn.functional.cross_entropy(logits, targets), logits, targets
            def _delta_w_penalty_loss(self, net): return torch.tensor(0.), {}
            def _past_fisher_penalty(self, net):
                self.penalty_calls += 1
                return super()._past_fisher_penalty(net)
            def _build_direction_projector(self, clean, fisher, delta=None):
                self.clean_cache, self.fisher_cache = clean, fisher
                return super()._build_direction_projector(clean, fisher, delta)

        model = TinyLearner()
        base = torch.optim.SGD(model._network.parameters(), lr=.02)
        optimizer = GAM_Filiter(model._network.parameters(), base, model._network,
                               grad_rho=.05, grad_norm_rho=.2,
                               args=types.SimpleNamespace(grad_beta_1=1., grad_beta_2=1., grad_beta_3=1., grad_gamma=.1))
        records = []
        base_step = base.step
        def step(*a, **kw):
            record = train.observe_gam_gradients(base.param_groups, model.clean_cache, model.fisher_cache, True)
            records.append(record)
            return base_step(*a, **kw)
        base.step = step
        inputs, targets = torch.tensor([[1., .5], [-.5, 1.], [.4, -.8]]), torch.tensor([0, 1, 0])
        with tempfile.TemporaryDirectory(prefix='native_gam_cpu_') as directory, ExitStack() as stack:
            for name in ('_task_loss', '_past_fisher_penalty', '_build_direction_projector', '_step_batch_with_log'):
                stack.enter_context(patch.object(Native, name, getattr(Native, name)))
            context = {'task': 1, 'train_step': 0, 'batch_active': False, 'diagnostic': False}
            train.install_batch_audit({'optimizer_type': 'gam', 'ewc_lambda': 20000.}, Path(directory), context, Native)
            for _ in range(2):
                _, loss, logs = model._step_batch_with_log(optimizer, inputs, targets, 0)
                self.assertTrue(math.isfinite(loss))
            emitted = train.read_jsonl(Path(directory) / 'training_batch_metrics.jsonl')
            self.assertEqual(len(emitted), 2)
            train.validate_batch_receipts(emitted, {'optimizer_type': 'gam'})
            self.assertTrue(all(row['single_ewc_injection_verified'] for row in emitted))
        self.assertEqual(model.penalty_calls, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['grad_fisher_norm'], 0.)
        self.assertGreater(records[1]['grad_fisher_norm'], 0.)
        self.assertGreater(logs['ewc'], 0.)
        native = model._summarize_mechanism_grad_stats()
        for record_key, native_key in [('grad_clean_norm', 'grad_clean_norm_mean'),
                                       ('grad_fisher_norm', 'grad_fisher_norm_mean'),
                                       ('grad_combined_norm', 'grad_combined_norm_mean')]:
            self.assertAlmostEqual(sum(row[record_key] for row in records) / 2, native[native_key], places=5)


if __name__ == '__main__':
    unittest.main()
