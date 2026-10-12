#!/usr/bin/env python3
"""CPU checks against the real frozen DataManager/evaluator and driver contract."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
import numpy as np
import torch

SOURCE=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0,str(SOURCE))
from utils.data_manager import DataManager
from models_CL.baseLearner import BaseLearner
from layout_lifecycle import TaskLayout,seen_accuracy_row,matrix_metrics
import response_train as driver

OLD_CONFIG=Path(__file__).parents[2]/'imageneta-history-response-execution-20261008T175403Z/configs/formal_history_fused_k10_s2.json'


class SyntheticDataManager(DataManager):
    """Use the exact source partition algorithm without opening real datasets."""
    def _setup_data(self,dataset_name,shuffle,seed):
        self._class_order=list(range(200));self._task_splits=None


def raw_config(layout,method):
    cfg=json.loads(OLD_CONFIG.read_text())
    cfg.update(response_method=method,response_layout=layout,response_radius_mode='sgd_step',
               radius_scale=1.,history_weight=.5 if method.startswith('history_') else 0.,
               history_source=driver.HISTORY_SOURCE[method],history_reset='task',post_update_loss=True,
               strict_deterministic=True,init_cls=200 if layout=='single200' else 20,
               increment=200 if layout=='single200' else 20)
    for key in ('task_increments','eval_task_splits','task_splits'):
        cfg.pop(key,None)
    return cfg


def cli(tasks=0,epochs=0,batches=0):
    return SimpleNamespace(max_tasks=tasks,max_epochs=epochs,max_batches=batches,
                           output_dir='/tmp/response_lifecycle_cpu_not_created')


class LifecycleTests(unittest.TestCase):
    def test_single200_is_source_one_task_all_classes(self):
        dm=SyntheticDataManager('imageneta',False,2,200,200,{})
        layout=TaskLayout.from_name('single200');layout.validate_data_manager(dm)
        self.assertEqual(dm._increments,[200]);self.assertEqual(dm.nb_tasks,1)
        self.assertEqual(dm.get_task_class_range(0),(0,200))
        # A one-stage truncation of the old split is a different condition.
        old_dm=SyntheticDataManager('imageneta',False,2,20,20,{})
        with self.assertRaises(AssertionError):layout.validate_data_manager(old_dm)

    def test_cl10_exact_source_boundaries(self):
        dm=SyntheticDataManager('imageneta',False,2,20,20,{})
        layout=TaskLayout.from_name('cl10');layout.validate_data_manager(dm)
        self.assertEqual(layout.boundaries,tuple((20*t,20*(t+1)) for t in range(10)))
        self.assertEqual(dm.nb_tasks,10)

    def test_single200_source_evaluator_uses_one_bucket(self):
        learner=BaseLearner.__new__(BaseLearner)
        learner.args={'init_cls':200,'increment':200,'task_increments':[200],'eval_task_splits':[200]}
        learner._total_classes=200;learner._known_classes=0;learner.topk=2
        y=np.arange(200);pred=np.stack([y,(y+1)%200],axis=1)
        cnn=learner._evaluate(pred,y)
        self.assertEqual({k for k in cnn['grouped'] if '-' in k},{'00-199'})
        self.assertEqual(seen_accuracy_row(cnn,0,200,0,TaskLayout.from_name('single200')),[100.])

    def test_cl10_source_rounded_matrix_matches_legacy(self):
        learner=BaseLearner.__new__(BaseLearner)
        learner.args={'init_cls':20,'increment':20,'task_increments':[20]*10}
        learner._total_classes=40;learner._known_classes=20;learner.topk=2
        y=np.repeat(np.arange(40),3)
        guess=y.copy();guess[0]=1
        pred=np.stack([guess,(guess+1)%40],axis=1)
        cnn=learner._evaluate(pred,y)
        actual=seen_accuracy_row(cnn,1,40,1,TaskLayout.from_name('cl10'))
        expected=driver.native.seen_accuracy_row(cnn,1,40,1)
        self.assertEqual(actual,expected);self.assertEqual(actual,[98.33,100.])

    def test_driver_resolves_all_eight_methods_layouts(self):
        for layout in ('single200','cl10'):
            for method in driver.DIRECTIONS:
                cfg=driver.resolve_config(raw_config(layout,method),cli())
                counts=[200] if layout=='single200' else [20]*10
                self.assertEqual(cfg['task_increments'],counts)
                self.assertEqual(cfg['eval_task_splits'],counts)
                self.assertEqual(cfg['init_cls'],counts[0])
                self.assertEqual(cfg['max_train_tasks'],len(counts))
                self.assertEqual(cfg['probe_gradient_count'],driver.NORMAL_GRADIENTS[method])
                self.assertEqual(cfg['history_source'],driver.HISTORY_SOURCE[method])
                self.assertFalse(cfg['gam_radii_active'])
                self.assertEqual((cfg['init_epoch'],cfg['epochs']),(50,50))
                self.assertEqual((cfg['init_lr'],cfg['lrate']),(.04,.02))

    def test_mismatched_partition_cache_and_budget_rejected(self):
        cfg=raw_config('single200','history_g2')
        for change in ({'init_cls':20},{'increment':20},{'history_source':'base'},
                       {'radius_scale':2.},{'ewc_skip_task_fisher':False},
                       {'task_increments':[20]*10}):
            bad=copy.deepcopy(cfg);bad.update(change)
            with self.assertRaises(ValueError):driver.resolve_config(bad,cli())
        with self.assertRaises(ValueError):driver.resolve_config(cfg,cli(tasks=2))

    def test_current_ce_and_head_cover_all200(self):
        layout=TaskLayout.from_name('single200');targets=torch.tensor([0,19,20,99,199])
        self.assertEqual(layout.validate_batch(0,0,targets,200,200),(0,200))
        logits=torch.randn(5,200,requires_grad=True)
        ce=torch.nn.functional.cross_entropy(logits,targets)
        grad=torch.autograd.grad(ce,logits)[0]
        self.assertNotEqual(float(grad[:,199].abs().sum()),0.)
        for seen,width in ((20,20),(200,20)):
            with self.assertRaises(AssertionError):layout.validate_batch(0,0,targets,seen,width)

    def test_cl_later_task_ce_and_full_seen_fc(self):
        layout=TaskLayout.from_name('cl10');targets=torch.tensor([20,39])
        self.assertEqual(layout.validate_batch(1,20,targets,40,40),(20,40))
        with self.assertRaises(AssertionError):layout.validate_batch(1,0,targets,40,40)
        with self.assertRaises(AssertionError):layout.validate_batch(1,20,targets,40,20)

    def test_single_stage_metrics_are_not_epoch_mean(self):
        metrics=matrix_metrics([[70.]], [70.],TaskLayout.from_name('single200'))
        self.assertEqual(metrics['AAA_task_macro'],70.)
        self.assertEqual(metrics['FAA_task_macro'],70.)
        self.assertEqual(metrics['final_test_accuracy'],70.)
        self.assertIsNone(metrics['BWT']);self.assertIsNone(metrics['forgetting'])
        self.assertIn('not the epoch',metrics['AAA_definition'])

    def test_cl_metrics_match_native_definitions(self):
        result=matrix_metrics([[80.],[70.,90.],[60.,75.,85.]],[80.,78.,74.],TaskLayout.from_name('cl10'))
        self.assertAlmostEqual(result['FAA_task_macro'],220/3)
        self.assertAlmostEqual(result['AAA_task_macro'],(80+80+220/3)/3)
        self.assertEqual(result['BWT'],-17.5)
        self.assertEqual(result['forgetting'],17.5)

    def test_epoch_cosine_does_not_become_probe_schedule(self):
        for initial in (.04,.02):
            parameter=torch.nn.Parameter(torch.tensor(1.))
            opt=torch.optim.SGD([parameter],lr=initial,momentum=0.,weight_decay=0.)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50,eta_min=initial*.1)
            trace=[]
            for epoch in range(50):
                used=float(opt.param_groups[0]['lr']);trace.append(used)
                # Three probe reads do not advance the optimizer/scheduler.
                for _ in range(3):self.assertEqual(float(opt.param_groups[0]['lr']),used)
                parameter.grad=torch.ones_like(parameter);opt.step();scheduler.step()
            self.assertEqual(trace[0],initial)
            self.assertAlmostEqual(float(opt.param_groups[0]['lr']),initial*.1)
            self.assertAlmostEqual(trace[-1],initial*.1+(initial-initial*.1)*(1+np.cos(np.pi*49/50))/2)
            self.assertEqual(scheduler.last_epoch,50)


if __name__=='__main__':unittest.main(verbosity=2)
