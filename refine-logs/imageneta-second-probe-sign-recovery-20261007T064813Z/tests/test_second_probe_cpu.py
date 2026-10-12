"""Independent analytic and stochastic acceptance tests; CUDA is disabled."""
import copy
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.dont_write_bytecode = True
TRAINER = Path(__file__).resolve().parents[1]/'trainer'
sys.path.insert(0, str(TRAINER))

from audit_core import digest, rng_equal, rng_get, rng_set
from direction_step import FourProbeStep, direction_name
from effective_step import EffectiveStep
from state_transaction import State
from four_probe_train import Controller, resolve_config

def flat(values):
    return torch.cat([v.reshape(-1) for v in values])


class Toy(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        gen = torch.Generator().manual_seed(913)
        self.a = torch.nn.Parameter(torch.randn(2, 3, generator=gen, dtype=dtype)*.2)
        self.b = torch.nn.Parameter(torch.randn(4, 2, generator=gen, dtype=dtype)*.2)
        self.fc = torch.nn.Linear(4, 3, dtype=dtype)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(3, 4, generator=gen, dtype=dtype)*.2)
            self.fc.bias.copy_(torch.randn(3, generator=gen, dtype=dtype)*.2)
        self.frozen = torch.nn.Parameter(torch.ones(2, dtype=dtype), requires_grad=False)
        self.register_buffer('probe_counter', torch.zeros((), dtype=dtype))
        self.dropout = torch.nn.Dropout(.4)
        self._register_delta_hook = False
        self.delta_w_q_new_grad = None
        self.external_delta_w_q = None
        self.forward_cache = {'entry': torch.ones(1, dtype=dtype)}


def make_case(dtype=torch.float64, reverse=False):
    model = Toy(dtype)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if reverse:
        params = params[::-1]
    geometry = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
    n = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(177)
    matrix = torch.randn(n, n, generator=gen, dtype=dtype)
    H = matrix.T@matrix/n + torch.diag(torch.linspace(.3, 2., n, dtype=dtype))
    linear = torch.linspace(-.2, .4, n, dtype=dtype)
    calls = []

    def closure(_x, _y):
        theta = flat(params)
        calls.append(theta.detach().clone())
        return .5*theta@(H@theta)-linear@theta+.08*theta.pow(4).sum()+.07*(1-theta.cos()).sum(), theta[:3].unsqueeze(0)

    return model, params, geometry, H, linear, closure, calls


class Acceptance(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-11, rtol=1e-11):
        actual = flat(actual) if isinstance(actual, (list, tuple)) else actual
        expected = flat(expected) if isinstance(expected, (list, tuple)) else expected
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        'max error='+str(float((actual-expected).abs().max())))

    def engine(self, model, params, geom, radius, anchor='g1', probe=1, difference=1):
        return FourProbeStep(model, params, geom, dict(branch='P',
            direction=direction_name(anchor, probe, difference), rho=.05, radius=radius,
            first_probe_sign=1, second_probe_sign=probe, second_difference_sign=difference,
            update_base=anchor, second_difference_weight=1.))

    def test_nonlinear_analytic_locations_direction_and_four_pass_one_sgd(self):
        for radius in (.25, 1.):
            for reverse in (False, True):
                model, params, geom, H, b, closure, calls = make_case(reverse=reverse)
                base = tuple(p.detach().clone() for p in params)
                theta = flat(base)
                def grad(t):
                    return H@t-b+.32*t.pow(3)+.07*t.sin()
                def normalize(t):
                    return t/(torch.linalg.vector_norm(t)+1e-12)
                g0 = grad(theta)
                point1 = theta+.05*normalize(g0)
                g1 = grad(point1)
                point2 = theta+radius*normalize(g1-g0)
                g2 = grad(point2)
                point3 = point2+.05*normalize(g2)
                g3 = grad(point3)
                expected = g1+(g3-g2)
                engine = self.engine(model,params,geom,radius)
                optimizer = torch.optim.SGD(params,lr=.04)
                with patch('torch.autograd.grad',wraps=torch.autograd.grad) as gradients, \
                        patch.object(optimizer,'step',wraps=optimizer.step) as updates:
                    result = engine.compute(None,None,closure,.04)
                    self.assertEqual(gradients.call_count,4)
                    self.assertTrue(all(c.kwargs['create_graph'] is False for c in gradients.call_args_list))
                    self.assertEqual(updates.call_count,0)
                    self.assertEqual(len(calls),4)
                    for actual, reference in zip(calls,(theta,point1,point2,point3)):
                        self.assert_close(actual,reference)
                    self.assert_close(result['context']['grads']['g3'],g3)
                    self.assert_close(result['context']['delta32'],g3-g2)
                    self.assert_close(result['U'],expected)
                    self.assertGreater(float(torch.linalg.vector_norm(expected-(g1+g3))),.1)
                    self.assertGreater(float(torch.linalg.vector_norm(point3-(theta+.05*normalize(g2)))),.2)
                    self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
                    record = result['record']
                    self.assertEqual(record['ordinary_gradient_calls'],4)
                    self.assertEqual(record['g3_gradient_calls'],1)
                    self.assertEqual(record['opposite_g1_gradient_calls'],0)
                    self.assertEqual(record['gradient_projection_calls'],0)
                    self.assertEqual(record['radius'],radius)
                    self.assertIn('phi3',record['probe_displacements'])
                    self.assertIn('delta32',record['norms'])
                    update = engine.apply_update(optimizer=optimizer)
                    self.assertEqual(updates.call_count,1)
                    self.assertTrue(update['natural_sgd_exact'])
                    self.assert_close(tuple(p.detach() for p in params),theta-.04*expected)
                    self.assertEqual(len(optimizer.state),0)

    def stochastic(self, model, params, draws):
        def closure(_x, _y):
            theta = flat(params)
            mask = model.dropout(torch.ones_like(theta))
            draws.append((random.random(), float(np.random.rand()), torch.rand(5), mask.detach().clone()))
            model.probe_counter.add_(1)
            model.dropout.training = False
            model.delta_w_q_new_grad = torch.ones(2)
            model.external_delta_w_q = torch.ones(3)
            model._register_delta_hook = True
            model.forward_cache['entry'].add_(2)
            model.forward_cache['created'] = torch.ones(2)
            model.created_cache = [torch.ones(1)]
            # Nonlinear curvature distinguishes positive and negative first probes.
            loss = ((theta*mask).square()+.2*(theta*mask).pow(4)).sum()
            return loss, theta[:3].unsqueeze(0)
        return closure

    def test_rng_replay_full_state_rollback_and_only_one_rng_advance(self):
        model, params, geom, *_ = make_case()
        model.train()
        state = State(model)
        reference_draws = []
        self.stochastic(model, params, reference_draws)(None, None)
        expected_rng1 = rng_get()
        state.restore()
        draws = []
        engine = self.engine(model,params,geom,1.)
        out = engine.compute(None, None, self.stochastic(model, params, draws), .04)
        self.assertEqual(len(draws), 4)
        for draw in draws[1:]:
            self.assertEqual(draw[:2], draws[0][:2])
            self.assertTrue(torch.equal(draw[2], draws[0][2]))
            self.assertTrue(torch.equal(draw[3], draws[0][3]))
        self.assertTrue(rng_equal(rng_get(), out['context']['rng1']))
        self.assertTrue(rng_equal(rng_get(), expected_rng1))
        self.assertTrue(all(torch.equal(p.grad, u) for p, u in zip(params, out['U'])))
        for p, grad in state.grads:
            p.grad = None if grad is None else grad.clone()
        rng_set(state.rng)
        self.assertTrue(all(state.check().values()))
        self.assertFalse(hasattr(model, 'created_cache'))
        engine.discard()

    def test_exception_restores_entry_parameters_gradients_buffers_caches_and_rng(self):
        for failure_probe in (1, 2, 3, 4):
            model, params, geom, *_ = make_case()
            for p in params:
                p.grad = torch.ones_like(p)*3
            state = State(model)
            draws = []
            stochastic = self.stochastic(model, params, draws)
            def failing(x, y):
                result = stochastic(x, y)
                if len(draws) == failure_probe:
                    raise RuntimeError('intentional probe failure')
                return result
            engine = self.engine(model,params,geom,.25)
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                engine.compute(None, None, failing, .04)
            self.assertTrue(all(state.check().values()))
            with self.assertRaisesRegex(RuntimeError, 'compute must precede'):
                engine.apply_update()

    def test_update_exception_rolls_back_and_duplicate_step_is_rejected(self):
        model, params, geom, _, _, closure, _ = make_case()
        base = tuple(p.detach().clone() for p in params)
        engine = self.engine(model,params,geom,1.)
        optimizer = torch.optim.SGD(params, lr=.04)
        native = optimizer.step
        def failing_step(*args, **kwargs):
            native(*args, **kwargs)
            raise RuntimeError('intentional SGD failure')
        engine.compute(None, None, closure, .04)
        with patch.object(optimizer, 'step', side_effect=failing_step):
            with self.assertRaisesRegex(RuntimeError, 'intentional SGD'):
                engine.apply_update(optimizer=optimizer)
        self.assertTrue(all(torch.equal(p, v) for p, v in zip(params, base)))
        with self.assertRaisesRegex(RuntimeError, 'compute must precede'):
            engine.apply_update(optimizer=optimizer)

    def test_effective_config_radius_reaches_actual_controller_stepper(self):
        config_path=TRAINER.parent/'configs/B1_g1_probep1_residualp1_r1_k10_s2.json'
        raw=json.loads(config_path.read_text())
        for radius in (.25,1.):
            with tempfile.TemporaryDirectory() as output:
                cli=SimpleNamespace(max_tasks=1,max_epochs=1,max_batches=1,output_dir=output)
                candidate=copy.deepcopy(raw)
                candidate.update(gam_grad_norm_rho=radius)
                cfg=resolve_config(candidate,cli)
                self.assertEqual(cfg['gam_grad_norm_rho'],radius)
                cfg['seed']=2
                model,params,geom,_,_,closure,calls=make_case()
                dummy=lambda *args,**kwargs:None
                learner=SimpleNamespace(_unwrap_network=lambda:model,_run_regularized_train=dummy,
                                        incremental_train=dummy,eval_task=dummy,after_task=dummy)
                controller=Controller(learner,cfg,cli)
                controller.task=0;controller.old_fc={}
                try:
                    with patch('four_probe_train.native.install_forbidden_work_guards'), \
                            patch.object(EffectiveStep,'from_model',return_value=geom), \
                            patch.object(controller,'save_observation'):
                        controller.install()
                        learner._build_optimizer(params,'init')
                    self.assertEqual(controller.stepper.radius,radius)
                    result=controller.stepper.compute(None,None,closure,.04)
                    self.assertAlmostEqual(float(torch.linalg.vector_norm(calls[2]-calls[0])),radius,places=9)
                    self.assertEqual(result['record']['radius'],radius)
                    controller.stepper.discard()
                finally:
                    controller.handle.close()

    def test_fp32_natural_step_and_zero_momentum_guard(self):
        for radius in (.25,1.):
            model,params,geom,_,_,closure,_=make_case(torch.float32,reverse=True)
            engine=self.engine(model,params,geom,radius)
            out=engine.compute(None,None,closure,.04)
            self.assertTrue(all(torch.isfinite(v).all() for v in out['U']))
            bad=torch.optim.SGD(params,lr=.04,momentum=.9)
            with self.assertRaisesRegex(ValueError,'zero momentum'):
                engine.apply_update(optimizer=bad)
            good=torch.optim.SGD(params,lr=.04)
            self.assertTrue(engine.apply_update(optimizer=good)['natural_sgd_exact'])


    def test_all_eight_formulas_first_three_points_are_identical(self):
        for radius in (.25, 1.):
            common = None
            for anchor in ('g0', 'g1'):
                for probe in (-1, 1):
                    for difference in (-1, 1):
                        model, params, geom, H, b, closure, calls = make_case()
                        theta = flat(params).detach().clone()
                        def grad(t):
                            return H@t-b+.32*t.pow(3)+.07*t.sin()
                        normalize = lambda t: t/(torch.linalg.vector_norm(t)+1e-12)
                        g0 = grad(theta)
                        p1 = theta+.05*normalize(g0)
                        g1 = grad(p1)
                        p2 = theta+radius*normalize(g1-g0)
                        g2 = grad(p2)
                        p3 = p2+probe*.05*normalize(g2)
                        g3 = grad(p3)
                        reference = (g0 if anchor == 'g0' else g1)+difference*(g3-g2)
                        engine = self.engine(model, params, geom, radius, anchor, probe, difference)
                        with patch('torch.autograd.grad', wraps=torch.autograd.grad) as gradients:
                            output = engine.compute(None, None, closure, .04)
                        self.assertEqual(gradients.call_count, 4)
                        self.assertEqual(len(calls), 4)
                        self.assert_close(output['U'], reference)
                        self.assert_close(calls[3], p3)
                        current = [output['context']['grads'][label] for label in ('g0', 'g1', 'g2')]
                        if common is None:
                            common = current
                        else:
                            for actual, previous in zip(current, common):
                                self.assertTrue(all(torch.equal(a, v) for a, v in zip(actual, previous)))
                        geometry_audit = output['record']['second_probe_parameter_audit']
                        self.assertTrue(geometry_audit['passed'])
                        self.assertAlmostEqual(geometry_audit['alignment_cosine'], probe, places=11)
                        self.assertAlmostEqual(geometry_audit['parameter_length'], .05, places=10)
                        engine.apply_update()
                        self.assert_close(flat(params), theta-.04*reference)

    def test_side_diagnostics_preserve_pending_update_complete_state_and_rng(self):
        for probe in (-1, 1):
            for anchor in ('g0', 'g1'):
                model, params, geom, *_ = make_case(torch.float32)
                draws = []
                closure = self.stochastic(model, params, draws)
                engine = self.engine(model, params, geom, 1., anchor, probe, -1)
                output = engine.compute(None, None, closure, .04)
                state = State(model)
                pending = engine._pending
                expected = tuple(v.add(u, alpha=-.04) for v, u in zip(output['context']['points']['phi0'], output['U']))
                diagnostic = engine.diagnose_sides(None, None, closure, output, .04)
                self.assertEqual(diagnostic['diagnostic_gradient_calls'], 1)
                self.assertEqual(diagnostic['diagnostic_sgd_updates'], 0)
                self.assertEqual(len(draws), 5)
                self.assertTrue(all(state.check().values()))
                self.assertIs(engine._pending, pending)
                self.assertTrue(diagnostic['selected_side_exact'])
                self.assertTrue(diagnostic['selected_side_reused_from_training'])
                self.assertTrue(diagnostic['selected_side_reference_exact'])
                self.assertEqual(diagnostic['selected_side_new_gradient_calls'],0)
                self.assertEqual(diagnostic['opposite_side_new_gradient_calls'],1)
                side = 'plus' if probe == 1 else 'minus'
                other = 'minus' if probe == 1 else 'plus'
                self.assertEqual(diagnostic['measurement_sources'][side],'training_g3')
                self.assertEqual(diagnostic['measurement_sources'][other],'isolated_opposite_probe')
                self.assertTrue(all(diagnostic['state_restoration'].values()))
                self.assertTrue(engine.apply_update()['natural_sgd_exact'])
                self.assertTrue(all(torch.equal(p, v) for p, v in zip(params, expected)))

    def test_old_b1_exact_equivalence_two_radii_fp32_and_stochastic(self):
        old_path = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/trainer/direction_step.py')
        spec = importlib.util.spec_from_file_location('old_g1_d32_step', old_path)
        old_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old_module)
        for radius in (.25, 1.):
            model, params, geom, *_ = make_case(torch.float32)
            entry = State(model)
            old_engine = old_module.FourProbeStep(model, params, geom, dict(branch='P', direction='g1_plus_second_difference', rho=.05, radius=radius))
            old_result = old_engine.compute(None, None, self.stochastic(model, params, []), .04)
            old_rng1 = rng_get()
            old_update = old_engine.apply_update()
            old_parameters = tuple(p.detach().clone() for p in params)
            entry.restore()
            engine = self.engine(model, params, geom, radius)
            result = engine.compute(None, None, self.stochastic(model, params, []), .04)
            self.assertTrue(rng_equal(rng_get(), old_rng1))
            for label in ('g0', 'g1', 'g2', 'g3'):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(result['context']['grads'][label], old_result['context']['grads'][label])))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(result['U'], old_result['U'])))
            diagnostic = engine.diagnose_sides(None, None, self.stochastic(model, params, []), result, .04)
            self.assertTrue(diagnostic['selected_side_exact'])
            self.assertTrue(rng_equal(rng_get(), old_rng1))
            update = engine.apply_update()
            self.assertEqual(update, old_update)
            self.assertTrue(all(torch.equal(p, b) for p, b in zip(params, old_parameters)))

    def test_strict_factors_and_all_16_registered_configurations(self):
        cli = SimpleNamespace(max_tasks=2, max_epochs=1, max_batches=1, output_dir='/tmp/second-probe-dry-test')
        for path in sorted((TRAINER.parent/'configs').glob('*.json')):
            raw = json.loads(path.read_text())
            cfg = resolve_config(raw, cli)
            self.assertEqual(cfg['update_direction'], raw['update_direction'])
            self.assertTrue(cfg['compute_g3_probe'])
            self.assertEqual(cfg['first_probe_sign'], 1)
            self.assertEqual(cfg['second_probe_sign'], raw['second_probe_sign'])
            for field, value in [('first_probe_sign', -1), ('second_probe_sign', 0),
                                 ('second_difference_sign', True), ('second_difference_weight', .5),
                                 ('update_base', 'g2'), ('component_rule', 'legacy')]:
                broken = dict(raw, **{field:value})
                with self.assertRaises(ValueError):
                    resolve_config(broken, cli)
            broken = dict(raw)
            del broken['second_probe_sign']
            with self.assertRaises(KeyError):
                resolve_config(broken, cli)

    def test_controller_training_and_diagnostic_accounting_are_separate(self):
        class TrainToy(Toy):
            def __init__(self):
                super().__init__(torch.float32)
                self.fc = torch.nn.Linear(4, 20)

            def forward(self, x):
                self.probe_counter.add_(1)
                return {'logits': self.fc(self.dropout(x@(self.b@self.a).T))}

        raw = json.loads((TRAINER.parent/'configs/N5_g0_probem1_residualm1_r1_k10_s2.json').read_text())
        with tempfile.TemporaryDirectory() as output:
            cli = SimpleNamespace(max_tasks=2, max_epochs=1, max_batches=1, output_dir=output)
            cfg = resolve_config(raw, cli)
            cfg['seed'] = 2
            model = TrainToy()
            params = tuple(p for p in model.parameters() if p.requires_grad)
            geom = EffectiveStep([(model.a, model.b)], tuple(model.fc.parameters()), params)
            dummy = lambda *args, **kwargs: None
            learner = SimpleNamespace(_unwrap_network=lambda:model, _run_regularized_train=dummy,
                incremental_train=dummy, eval_task=dummy, after_task=dummy, _total_classes=20,
                _task_loss=lambda logits, targets, offset: (torch.nn.functional.cross_entropy(logits[:,offset:offset+20],targets-offset),None,None))
            controller = Controller(learner, cfg, cli)
            controller.task=0; controller.old_fc={}; controller.task_start_step=0
            controller.epoch=1; controller.batch=1; controller.batch_identity={'test':True}
            controller.samples=0; controller.loss_sum=0.; controller.correct=0; controller.epoch_batches=0
            try:
                with patch('four_probe_train.native.install_forbidden_work_guards'), \
                        patch.object(EffectiveStep,'from_model',return_value=geom), \
                        patch.object(controller,'save_observation'):
                    controller.install()
                    optimizer = learner._build_optimizer(params,'init')
                for _ in range(2):
                    learner._step_batch_with_log(optimizer, torch.ones(2,3), torch.tensor([0,1]), 0)
                self.assertEqual(controller.steps, 2)
                self.assertEqual(controller.counters['training_forward_count'],8)
                self.assertEqual(controller.counters['training_ce_gradient_passes'],8)
                self.assertEqual(controller.counters['training_sgd_updates'],2)
                self.assertEqual(controller.counters['training_g3_negative_gradient_passes'],2)
                self.assertEqual(controller.counters['training_g3_positive_gradient_passes'],0)
                self.assertEqual(controller.counters['diagnostic_gradient_calls'],1)
                self.assertEqual(controller.counters['diagnostic_snapshots'],1)
                trace = [json.loads(line) for line in controller.trace.read_text().splitlines()]
                self.assertIn('g012_fingerprints',trace[0])
                self.assertNotIn('g012_fingerprints',trace[1])
                self.assertTrue(all(row['second_probe_parameter_audit']['passed'] for row in trace))
                snapshot_info = trace[0]['same_state_snapshot']
                self.assertEqual(digest(snapshot_info['path']),snapshot_info['sha256'])
                snapshot = torch.load(snapshot_info['path'],map_location='cpu',weights_only=False)
                from audit_core import tensor_digest
                for label in ('g0','g1','g2'):
                    self.assertEqual([tensor_digest(v) for v in snapshot['grads'][label]],trace[0]['g012_fingerprints'][label])
                for label in ('phi0','phi1','phi2'):
                    self.assertEqual([tensor_digest(v) for v in snapshot['points'][label]],trace[0]['point012_fingerprints'][label])
                self.assertEqual(float(model.probe_counter),0.)
            finally:
                controller.handle.close()

    def test_failed_opposite_diagnostic_restores_pending_u_and_full_state(self):
        for probe in (-1,1):
            model, params, geom, *_ = make_case(torch.float32)
            engine = self.engine(model,params,geom,1.,'g0',probe,-1)
            draws = []
            closure = self.stochastic(model,params,draws)
            result = engine.compute(None,None,closure,.04)
            entry = State(model)
            pending = engine._pending
            def fails(x,y):
                closure(x,y)
                raise RuntimeError('intentional opposite diagnostic failure')
            with self.assertRaisesRegex(RuntimeError,'opposite diagnostic'):
                engine.diagnose_sides(None,None,fails,result,.04)
            self.assertEqual(len(draws),5)
            self.assertTrue(all(entry.check().values()))
            self.assertIs(engine._pending,pending)
            self.assertTrue(engine.apply_update()['natural_sgd_exact'])

    def test_recovery_does_not_edit_training_compute_or_sgd(self):
        old = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-second-probe-sign-plan-20261006T185119Z/trainer/direction_step.py')
        def methods(path):
            source=path.read_text()
            lines=source.splitlines()
            cls=next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=='FourProbeStep')
            return {node.name:'\n'.join(lines[node.lineno-1:node.end_lineno]) for node in cls.body if isinstance(node,ast.FunctionDef)}
        previous,current=methods(old),methods(TRAINER/'direction_step.py')
        for name in ('_evaluate','_point','_third_probe_audit','compute','apply_update'):
            self.assertEqual(previous[name],current[name],name+' changed')


if __name__=='__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run CPU acceptance with CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(1)
    random.seed(2); np.random.seed(2); torch.manual_seed(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    accepted = result.wasSuccessful()
    old_path = Path('/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-g1-plus-g3-minus-g2-20261005T170708Z/trainer/direction_step.py')
    output = dict(status='passed' if accepted else 'failed', tests_total=result.testsRun,
                  tests_passed=result.testsRun-len(result.failures)-len(result.errors),
                  failures=len(result.failures), errors=len(result.errors), device='cpu',
                  checks={key:accepted for key in ('old_b1_same_state_equivalence',
                      'reuse_g1_positive_same_state', 'all_eight_formulas', 'same_g012',
                      'exact_state_rng_restoration', 'four_ce_per_step', 'side_diagnostics_isolated',
                      'selected_side_reused_opposite_only', 'smoke_snapshot_hashes_exact',
                      'training_compute_and_sgd_source_unchanged')},
                  equivalence_by_radius={str(r):accepted for r in (.25, 1.)},
                  old_executor=str(old_path), old_executor_sha256=digest(old_path),
                  executor_sha256=digest(TRAINER/'direction_step.py'),
                  entry_sha256=digest(TRAINER/'four_probe_train.py'),
                  command='CUDA_VISIBLE_DEVICES="" /home-local/liying/.conda/envs/Pilot_new_local/bin/python '+str(Path(__file__).resolve()))
    (TRAINER/'CPU_GATE.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))
    raise SystemExit(0 if accepted else 1)
