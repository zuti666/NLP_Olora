"""Driver integration and fail-closed CLI contract for independent probes.

Uses the actual driver resolver, old frozen driver, and quadratic stepper API.
These are mathematical/configuration checks, not ImageNet training scores.
"""
import copy
import importlib.util
import math
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import response_train as driver
from test_response_lifecycle_cpu import raw_config, cli, SOURCE
from test_response_step_cpu import VectorModel, PlainGeometry, quadratic

OLD = Path(__file__).parents[2] / 'imageneta-response-four-methods-gpu567-20261009T103449Z/trainer/response_train.py'
spec = importlib.util.spec_from_file_location('frozen_response_four_driver', OLD)
old_driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_driver)


def independent_raw(label='CurrentG2-FixedBoth'):
    cfg = raw_config('cl10', 'current_g2')
    inner_mode, outer_mode, short_name = driver.PROBE_VARIANTS[label]
    cfg.update(response_radius_mode=driver.INDEPENDENT_MODE,
               response_probe_variant=label,
               inner_probe_radius_mode=inner_mode,
               outer_probe_radius_mode=outer_mode,
               inner_probe_radius=.05 if inner_mode=='fixed_l2' else None,
               outer_probe_radius=.05 if outer_mode=='fixed_l2' else None,
               response_algorithm_family='current_g2')
    return cfg


class IndependentDriverCPU(unittest.TestCase):
    def test_all_original_method_layout_resolutions_are_exact(self):
        for layout in ('single200','cl10'):
            for method in driver.DIRECTIONS:
                raw = raw_config(layout, method)
                self.assertEqual(driver.resolve_config(raw, cli()), old_driver.resolve_config(raw, cli()))

    def test_three_registered_variants_preserve_cl_training_contract(self):
        for label, (inner_mode, outer_mode, short_name) in driver.PROBE_VARIANTS.items():
            cfg = driver.resolve_config(independent_raw(label), cli())
            self.assertEqual(cfg['response_method'], 'current_g2')
            self.assertEqual(cfg['response_layout'], 'cl10')
            self.assertEqual(cfg['task_increments'], [20]*10)
            self.assertEqual(cfg['probe_gradient_count'], 3)
            self.assertEqual(cfg['radius_variant'], short_name)
            self.assertEqual(cfg['inner_probe_radius_mode'], inner_mode)
            self.assertEqual(cfg['outer_probe_radius_mode'], outer_mode)
            self.assertEqual((cfg['init_epoch'],cfg['epochs']), (50,50))
            self.assertEqual((cfg['init_lr'],cfg['lrate'],cfg['cosine_min_lr_ratio']), (.04,.02,.1))
            self.assertEqual(cfg['history_weight'], 0.)
            self.assertIsNone(cfg['history_source'])
            self.assertFalse(cfg['gam_radii_active'])
            self.assertEqual(cfg['protocol'], driver.INDEPENDENT_PROTOCOL)

    def test_original_route_is_original_class_and_bitwise_trajectory(self):
        cfg = driver.resolve_config(raw_config('cl10','current_g2'), cli())
        new_model, old_model = VectorModel(), VectorModel()
        new = driver.make_stepper(new_model, list(new_model.parameters()), PlainGeometry(new_model), cfg)
        old = old_driver.ResponseStep(old_model, list(old_model.parameters()), PlainGeometry(old_model),
                dict(method='current_g2',history_weight=0.,eps=1e-12,backend='native',radius_mode='sgd_step',radius_scale=1.))
        self.assertIs(type(new), old_driver.ResponseStep)
        new_opt, old_opt = torch.optim.SGD(new_model.parameters(),lr=.04), torch.optim.SGD(old_model.parameters(),lr=.04)
        for eta in (.04,.02,.004):
            new_opt.param_groups[0]['lr'] = old_opt.param_groups[0]['lr'] = eta
            a = new.compute(None,None,quadratic(new_model),eta)
            b = old.compute(None,None,quadratic(old_model),eta)
            for key in ('g0','g1','g2','U','future','relocation'):
                self.assertTrue(torch.equal(a['context'][key][0], b['context'][key][0]))
            new.apply_update(eta,new_opt);old.apply_update(eta,old_opt)
            self.assertTrue(torch.equal(new_model.w,old_model.w))

    def test_new_route_passes_exact_config_and_one_actual_sgd_update(self):
        for label in driver.PROBE_VARIANTS:
            cfg = driver.resolve_config(independent_raw(label), cli())
            model = VectorModel()
            step = driver.make_stepper(model,list(model.parameters()),PlainGeometry(model),cfg)
            self.assertEqual(type(step).__name__, 'IndependentResponseStep')
            before = model.w.detach().clone()
            out = step.compute(None,None,quadratic(model),.04)
            self.assertEqual(out['record']['ordinary_gradient_calls'],3)
            self.assertFalse(out['record']['history_used'])
            self.assertTrue(torch.equal(model.w,before))
            opt = torch.optim.SGD(model.parameters(),lr=.04)
            update = step.apply_update(.04,opt)
            self.assertEqual(update['sgd_updates'],1)
            self.assertTrue(torch.equal(model.w,before.clone().add_(out['U'][0],alpha=-.04)))
            state = step.state_dict()
            self.assertEqual(state['format'], 'IndependentResponseStep_v1')
            self.assertIn('independent_radius_config',state)

    def test_metadata_distinguishes_reference_and_actual_configured_radii(self):
        for label,(inner_mode,outer_mode,_) in driver.PROBE_VARIANTS.items():
            cfg=driver.resolve_config(independent_raw(label),cli())
            meta=driver.radius_metadata(cfg)
            self.assertEqual(meta['probe_mode'],driver.INDEPENDENT_MODE)
            self.assertEqual(meta['response_probe_variant'],label)
            self.assertEqual(meta['inner_rho'],.05 if inner_mode=='fixed_l2' else None)
            self.assertEqual(meta['outer_R'],.05 if outer_mode=='fixed_l2' else None)
            self.assertEqual(meta['normal_gradient_calls'],3)
            self.assertFalse(meta['gam_radii_active'])

    def test_old_mode_rejects_independent_fields_instead_of_ignoring(self):
        for key in driver.PROBE_CONFIG_KEYS:
            raw=raw_config('cl10','current_g2');raw[key]=None
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_independent_mode_rejects_history_and_sgd_methods(self):
        for method in ('sgd','history_fusion','history_g2'):
            raw=independent_raw();raw.update(response_method=method,
                history_source=driver.HISTORY_SOURCE[method],
                history_weight=.5 if method.startswith('history') else 0.)
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_missing_explicit_radius_fields_rejected(self):
        for key in driver.PROBE_CONFIG_KEYS:
            raw=independent_raw();raw.pop(key)
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_fixed_radius_invalid_or_outside_registered_contrast_rejected(self):
        for value in (None,True,False,0.,-.05,.02,.1,'0.05',float('nan'),float('inf')):
            raw=independent_raw();raw['inner_probe_radius']=value
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_variant_mode_or_dynamic_radius_conflict_rejected(self):
        for change in ({'response_probe_variant':'unknown'}, {'radius_variant':'fixed_outer'},
                       {'inner_probe_radius_mode':'sgd_step'}, {'response_radius_mode':'independent_inner_outer_proposal_v1'},
                       {'response_algorithm_family':'history_g2'}):
            raw=independent_raw();raw.update(change)
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())
        raw=independent_raw('CurrentG2-FixedInner');raw['outer_probe_radius']=.05
        with self.assertRaises(ValueError):driver.resolve_config(raw,cli())
        raw=raw_config('cl10','current_g2');raw['radius_variant']='fixed_inner'
        with self.assertRaises(ValueError):driver.resolve_config(raw,cli())
        for key in ('fixed_inner_probe_radius','fixed_outer_probe_radius',
                    'inner_radius_mode','outer_radius_mode','inner_radius','outer_radius'):
            raw=independent_raw();raw[key]=None
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_original_label_only_accepted_on_original_current_g2(self):
        raw=raw_config('cl10','current_g2');raw['response_probe_variant']='CurrentG2-Original'
        cfg=driver.resolve_config(raw,cli())
        self.assertEqual(driver.radius_metadata(cfg)['response_probe_variant'],'CurrentG2-Original')
        for label in driver.PROBE_VARIANTS:
            raw['response_probe_variant']=label
            with self.assertRaises(ValueError):driver.resolve_config(raw,cli())

    def test_smoke_budget_does_not_change_partition_or_cosine_protocol(self):
        cfg=driver.resolve_config(independent_raw(),cli(tasks=2,epochs=1,batches=2))
        self.assertEqual(cfg['task_increments'],[20]*10)
        self.assertEqual(cfg['max_train_tasks'],2)
        self.assertEqual((cfg['init_epoch'],cfg['epochs']),(50,50))
        self.assertEqual((cfg['init_lr'],cfg['lrate'],cfg['cosine_min_lr_ratio']),(.04,.02,.1))

    def test_actual_cli_dry_runs_four_variants_without_output_mutation(self):
        raws=[raw_config('cl10','current_g2')]+[independent_raw(label) for label in driver.PROBE_VARIANTS]
        with tempfile.TemporaryDirectory(prefix='radius_driver_cpu_') as directory:
            root=Path(directory)
            for index,raw in enumerate(raws):
                path=root/(str(index)+'.json');path.write_text(json.dumps(raw))
                output=root/('output'+str(index))
                command=[sys.executable,'-B',str(Path(driver.__file__)),
                    '--source',str(SOURCE), '--config',str(path),
                    '--output-dir',str(output),'--max-tasks','2','--max-epochs','1','--max-batches','2','--dry-run']
                run=subprocess.run(command,text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
                self.assertEqual(run.returncode,0,run.stderr)
                receipt=json.loads(run.stdout)
                self.assertEqual(receipt['status'],'valid')
                self.assertEqual(receipt['tasks'],2)
                self.assertEqual(receipt['full_partition_tasks'],10)
                self.assertEqual(receipt['gradients_per_normal_step'],3)
                self.assertEqual(receipt['normal_gradient_calls'],3)
                self.assertEqual(receipt['updates_per_step'],1)
                self.assertEqual(receipt['response_probe_variant'],raw.get('response_probe_variant','CurrentG2-Original'))
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main(verbosity=2)
