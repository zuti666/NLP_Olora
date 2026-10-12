"""CPU acceptance for the copied r=1 ordinary four-probe campaign."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from audit_core import rng_equal, rng_get, rng_set
from direction_step import FourProbeStep, vector_norm
from four_probe_train import resolve_config
from test_direction_cpu import DirectionAcceptance, make_case, flat


HERE = Path(__file__).resolve().parent
OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
BASELINE_EVIDENCE = []
BRANCH_EVIDENCE = []


def dense_delta(before, after):
    a0,b0 = (v.double() for v in before[:2])
    a1,b1 = (v.double() for v in after[:2])
    squared = float((b1@a1-b0@a0).square().sum())
    squared += sum(float((v.double()-u.double()).square().sum())
                   for u,v in zip(before[2:],after[2:]))
    return math.sqrt(squared)


class RadiusOneAcceptance(unittest.TestCase):
    def test_radius_one_branch_signs_base_points_and_side_selection(self):
        model,params,geometry,H,b,closure = make_case()
        base = tuple(p.detach().clone() for p in params)
        expected_g0 = H@flat(base)-b
        u0 = expected_g0/(torch.linalg.vector_norm(expected_g0)+1e-12)
        values = {}
        for branch in ('P','D','C','R'):
            engine = FourProbeStep(model,params,geometry,
                                   {'branch':branch,'direction':'A','radius':1.,'rho':.05})
            out = engine.compute(None,None,closure,.04)
            ctx = out['context']
            side = 1 if branch in ('P','R') else -1
            expected_first = H@(flat(base)+side*.05*u0)-b
            expected_q = expected_first-expected_g0
            if branch in ('C','R'):expected_q = -expected_q
            expected_outer = flat(base)+expected_q/(torch.linalg.vector_norm(expected_q)+1e-12)
            self.assertTrue(torch.allclose(flat(ctx['q']),expected_q,atol=1e-13,rtol=1e-12))
            self.assertTrue(torch.allclose(flat(ctx['points']['phi2']),expected_outer,atol=1e-12,rtol=1e-12))
            self.assertTrue(torch.allclose(flat(ctx['grads']['g1']),expected_first,atol=1e-13,rtol=1e-12))
            expected_u = expected_g0+(H@expected_outer-b)-expected_first
            self.assertTrue(torch.allclose(flat(out['U']),expected_u,atol=1e-12,rtol=1e-12))
            distance = float(torch.linalg.vector_norm(flat(ctx['points']['phi2'])-flat(base)))
            self.assertAlmostEqual(distance,1.,places=10)
            self.assertTrue(all(torch.equal(p,v) for p,v in zip(params,base)))
            self.assertTrue(all(out['record']['state_restoration'].values()))
            self.assertEqual(out['record']['ordinary_gradient_calls'],4)
            self.assertEqual(out['record']['gradient_projection_calls'],0)
            values[branch] = ctx
            engine.discard()
        self.assertTrue(torch.allclose(flat(values['P']['points']['phi2']),flat(values['C']['points']['phi2']),atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.allclose(flat(values['D']['points']['phi2']),flat(values['R']['points']['phi2']),atol=1e-12,rtol=1e-12))
        self.assertGreater(float(torch.linalg.vector_norm(flat(values['P']['U'])-flat(values['C']['U']))),1e-3)

    def test_radius_one_fp32_all_branch_updates_and_positive_reference_dense_oracle(self):
        # O matched is a helper check only; the manifest preserves the OLD16 arms.
        for branch in ('P','D','C','R'):
            for direction in ('g2','A'):
                for matched in (False,True):
                    model,params,geometry,_,_,closure = make_case(torch.float32)
                    before = geometry.capture()
                    engine = FourProbeStep(model,params,geometry,
                                           {'branch':branch,'direction':direction,'radius':1.})
                    optimizer = torch.optim.SGD(params,lr=.04)
                    out = engine.compute(None,None,closure,.04)
                    ctx = out['context']
                    selected = ctx['grads']['g2'] if direction=='g2' else tuple(
                        g0+(g2-g1) for g0,g2,g1 in zip(ctx['grads']['g0'],ctx['grads']['g2'],ctx['grads']['g1']))
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(selected,out['U'])))
                    positive = tuple(v.clone().add_(g,alpha=-.04) for v,g in zip(before,ctx['grads']['g1_plus']))
                    target = dense_delta(before,positive)
                    update = engine.apply_update(matched=matched,optimizer=optimizer)
                    actual = dense_delta(before,geometry.capture())
                    self.assertAlmostEqual(update['reference']['joint_norm'],target,places=11)
                    self.assertAlmostEqual(update['actual']['joint_norm'],actual,places=11)
                    self.assertEqual(update['sgd_updates'],1)
                    self.assertEqual(optimizer.param_groups[0]['lr'],.04)
                    self.assertEqual(len(optimizer.state),0)
                    if matched:self.assertLessEqual(abs(actual-target)/target,1e-3)
                    else:self.assertEqual(update['step_scale'],.04)
                    if branch in ('D','C'):
                        negative = tuple(v.clone().add_(g,alpha=-.04) for v,g in zip(before,ctx['grads']['g1_minus']))
                        self.assertGreater(abs(target-dense_delta(before,negative)),1e-5)
                    BRANCH_EVIDENCE.append({'branch':branch,'direction':direction,'matched':matched,
                                            'target_is_same_state_raw_g1_plus':True,
                                            'actual_match_relative_error':update['actual_match_relative_error']})

    def test_four_baselines_multistep_radius_invariance_with_stochastic_probe_restoration(self):
        # Replay the exact same entry RNG in two independent CPU runs. Each run
        # itself advances RNG through g0 only, as the unchanged protocol requires.
        for direction,matched in [('g0',False),('g1_plus',False),('g1_minus',False),('g0',True)]:
            initial_rng = rng_get()
            runs = {}
            for radius in (.25,1.):
                rng_set(initial_rng)
                model,params,geometry,H,b,_ = make_case(torch.float32)
                optimizer = torch.optim.SGD(params,lr=.04)
                records = []
                def stochastic_closure(_x,_y):
                    theta = flat(params)
                    mask = model.dropout(torch.ones_like(theta))
                    coefficient = .8+.1*random.random()+.1*float(np.random.rand())
                    model.probe_counter.add_(1)
                    model.forward_cache['entry'].add_(1)
                    value = theta*mask
                    loss = coefficient*(.5*value@(H@value)-b@value)
                    return loss,theta[:3].unsqueeze(0)
                for step in range(5):
                    lr = .004+.036*(1+math.cos(math.pi*step/50))/2
                    optimizer.param_groups[0]['lr'] = lr
                    before = geometry.capture()
                    engine = FourProbeStep(model,params,geometry,
                                           {'branch':'P','direction':direction,'radius':radius})
                    out = engine.compute(None,None,stochastic_closure,lr)
                    self.assertTrue(all(out['record']['state_restoration'].values()))
                    self.assertEqual(out['record']['ordinary_gradient_calls'],4)
                    self.assertEqual(float(model.probe_counter),0.)
                    self.assertTrue(torch.equal(model.forward_cache['entry'],torch.tensor([1.])))
                    update = engine.apply_update(matched=matched,optimizer=optimizer)
                    self.assertEqual(optimizer.param_groups[0]['lr'],lr)
                    self.assertAlmostEqual(update['actual']['joint_norm'],dense_delta(before,geometry.capture()),places=11)
                    records.append({'parameters':tuple(p.detach().clone() for p in params),
                                    'rng':rng_get(),'U':out['U'],'actual':update['actual'],
                                    'reference':update['reference'],'scale':update['step_scale'],
                                    'outer':out['context']['points']['phi2'],
                                    'g2':out['context']['grads']['g2']})
                runs[radius] = records
            diagnostics_differ = False
            for a,c in zip(runs[.25],runs[1.]):
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['parameters'],c['parameters'])))
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(a['U'],c['U'])))
                self.assertTrue(rng_equal(a['rng'],c['rng']))
                self.assertEqual(a['actual'],c['actual'])
                self.assertEqual(a['reference'],c['reference'])
                self.assertEqual(a['scale'],c['scale'])
                diagnostics_differ |= any(not torch.equal(x,y) for x,y in zip(a['outer'],c['outer']))
                diagnostics_differ |= any(not torch.equal(x,y) for x,y in zip(a['g2'],c['g2']))
            self.assertTrue(diagnostics_differ,'Radius-invariance fixture must actually change the unused outer probe')
            BASELINE_EVIDENCE.append({'direction':direction,'matched':matched,'steps_per_radius':5,
                                      'parameters_selected_gradient_final_rng_effective_step_exact':True,
                                      'unused_outer_probe_diagnostics_differ':True,
                                      'reuse_scope':'final_update_and_rng_only; original_r025_probe_trace_retained'})

    def test_campaign_config_reads_and_enforces_explicit_radius_one(self):
        raw = json.loads((OLD/'configs/S1-12_A_P_matched_s2.json').read_text())
        cli = SimpleNamespace(max_tasks=0,max_epochs=0,max_batches=0,output_dir='/tmp/g012_r1_cpu_config_only')
        with self.assertRaises(ValueError):resolve_config(raw,cli)
        raw['gam_grad_norm_rho'] = 1.
        cfg = resolve_config(raw,cli)
        self.assertEqual(cfg['gam_grad_norm_rho'],1.)
        self.assertEqual(cfg['gam_grad_rho'],.05)
        self.assertEqual((cfg['init_epoch'],cfg['epochs'],cfg['max_train_tasks']),(50,50,10))
        self.assertEqual((cfg['init_lr'],cfg['lrate'],cfg['cosine_min_lr_ratio']),(.04,.02,.1))
        self.assertEqual((cfg['step_mode'],cfg['probe_branch'],cfg['update_direction']),('matched','P','A'))
        missing = copy.deepcopy(raw);missing.pop('gam_grad_norm_rho')
        with self.assertRaises(KeyError):resolve_config(missing,cli)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    names = ['test_quadratic_signs_locations_and_final_residual_reference',
             'test_all_direction_selectors_and_one_actual_optimizer_step',
             'test_rng_replay_buffers_modes_and_advancement_once',
             'test_exception_restores_gradients_state_and_rng',
             'test_matching_fp32_and_geometry_parameter_order',
             'test_degenerate_normalization_and_match_failure_no_update']
    suite = unittest.TestSuite([DirectionAcceptance(name) for name in names])
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RadiusOneAcceptance))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = sorted(HERE.glob('*.py'))
    unchanged = ['direction_step.py','effective_step.py','native_lifecycle.py',
                 'task0_trainer_base_v2.py','audit_core.py','response_hvp.py','cuda_witness.py']
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    old_hashes = {name:hashlib.sha256((OLD/'trainer'/name).read_bytes()).hexdigest() for name in unchanged}
    source_identity = all(hashes[name]==old_hashes[name] for name in unchanged)
    passed = result.wasSuccessful() and source_identity
    report = {'status':'passed' if passed else 'failed','passed':passed,'device':'cpu',
              'outer_radius':1.,'inner_rho':.05,'tests_run':result.testsRun,
              'ordinary_only_tests':True,'GPU_or_full_ViT_not_exercised':True,
              'failures':[name.id() for name,_ in result.failures],
              'errors':[name.id() for name,_ in result.errors],
              'four_baseline_update_radius_invariance':BASELINE_EVIDENCE,
              'branch_fp32_dense_oracle_cases':BRANCH_EVIDENCE,
              'algorithm_helpers_bytes_unchanged':source_identity,
              'trainer_sha256':hashes,'unchanged_helper_original_sha256':old_hashes,
              'original_trainer_sha256':{p.name:hashlib.sha256((OLD/'trainer'/p.name).read_bytes()).hexdigest()
                                         for p in files if (OLD/'trainer'/p.name).exists()},
              'baseline_radius_invariance_passed':passed and len(BASELINE_EVIDENCE)==4,
              'baseline_reuse_does_not_claim_radius_one_outer_probe_trace':True}
    output = Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'tests_run':result.testsRun,'report':str(output)}))
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
