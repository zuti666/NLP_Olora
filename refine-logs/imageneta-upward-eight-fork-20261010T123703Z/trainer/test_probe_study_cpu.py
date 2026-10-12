"""CPU checks for holdout isolation, exact snapshot restore and fork admission."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import audit_core as audit
import task0_trainer_base_v2 as base
import probe_study_train as study
from probe_study_data import select_indices

ROOT=Path(__file__).resolve().parents[3]
BASECFG=ROOT/'imageneta-response-four-methods-gpu567-20261009T103449Z/configs/formal_cl10_sgd_s2.json'
# Resolve against the actual neighboring campaign independently of cwd.
BASECFG=Path(__file__).resolve().parents[2]/'imageneta-response-four-methods-gpu567-20261009T103449Z/configs/formal_cl10_sgd_s2.json'

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(3,3);self.fc=torch.nn.Linear(3,20)
        self.register_buffer('scale',torch.tensor(2.));self.drop=torch.nn.Dropout(.2)
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.forward_cache={'tensor':torch.tensor([4.])}
    def forward(self,x):return {'logits':self.fc(self.drop(self.backbone(x)))}

class StudyCPU(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.split=self.root/'split.json';self.split.write_text('{}')
        self.cli=SimpleNamespace(max_epochs=0,max_batches=0,max_tasks=0,output_dir=str(self.root/'output'))
    def tearDown(self):self.tmp.cleanup()
    def raw(self,condition='parent_sgd'):
        raw=json.loads(BASECFG.read_text());method,im,ir,om,orr=study.CONDITIONS[condition]
        raw.update(study_condition_id=condition,study_role='parent' if condition=='parent_sgd' else 'branch',
            response_method=method,study_inner_mode=im,study_inner_radius=ir,study_outer_mode=om,study_outer_radius=orr,
            study_dev_split_path=str(self.split),study_parent_output_dir=str(self.root/'parent'),
            study_parent_checkpoint=str(self.root/'parent/checkpoints/task_000'),study_parent_receipt=str(self.root/'completed.json'))
        return raw
    def test_all_registered_cells_resolve_without_ignored_radii(self):
        for condition,spec in study.CONDITIONS.items():
            cfg=study.resolve_config(self.raw(condition),self.cli)
            self.assertEqual(cfg['response_method'],spec[0]);self.assertEqual(cfg['study_inner_radius'],spec[2]);self.assertEqual(cfg['study_outer_radius'],spec[4])
            self.assertEqual(cfg['max_train_tasks'],1 if condition=='parent_sgd' else 2)
            self.assertEqual(cfg['component_scale'],'natural');self.assertEqual(cfg['seed'],2)
    def test_conflicting_method_or_radius_is_rejected(self):
        for key,value in [('study_inner_radius',.1),('study_outer_radius',1.),('response_method','sgd')]:
            raw=self.raw('R25');raw[key]=value
            with self.assertRaises(ValueError):study.resolve_config(raw,self.cli)
    def test_parent_role_and_explicit_parent_receipt(self):
        raw=self.raw('B0');raw['study_parent_receipt']=None
        with self.assertRaises(ValueError):study.resolve_config(raw,self.cli)
        raw=self.raw('parent_sgd');raw['study_role']='branch'
        with self.assertRaises(ValueError):study.resolve_config(raw,self.cli)
    def test_unregistered_smoke_budget_rejected(self):
        cli=copy.copy(self.cli);cli.max_epochs=1
        with self.assertRaises(ValueError):study.resolve_config(self.raw(),cli)
        cli.max_batches=2;self.assertEqual(study.resolve_config(self.raw(),cli)['study_role'],'parent')
        cli.max_epochs=2
        with self.assertRaises(ValueError):study.resolve_config(self.raw(),cli)
    def test_split_reproducible_disjoint_complete_and_per_class(self):
        labels=np.repeat(np.arange(200),np.arange(200)%18+2)
        train,dev,counts=select_indices(labels)
        self.assertEqual((train,dev,counts),select_indices(labels))
        self.assertFalse(set(train)&set(dev));self.assertEqual(sorted(train+dev),list(range(len(labels))))
        for row in counts:self.assertGreaterEqual(row['dev'],1);self.assertGreaterEqual(row['train'],1)
        self.assertNotEqual(dev,select_indices(labels,seed=20261011)[1])
    def test_singleton_class_rejected(self):
        with self.assertRaises(ValueError):select_indices(np.array([0,1,1]))
    def test_complete_snapshot_preserves_grads_modes_buffers_caches_rng(self):
        model=Toy();model.train();model.drop.eval()
        model.fc.weight.grad=torch.ones_like(model.fc.weight);model.backbone.weight.grad=torch.zeros_like(model.backbone.weight)
        cfg=study.resolve_config(self.raw(),self.cli);cfg['study_source']=str(self.root)
        before=base.parameter_hash(model);rng=audit.rng_get()
        payload=study._complete_snapshot(model,cfg,0,50,.004,{'parent_after_task_completed':True})
        self.assertTrue(audit.rng_equal(rng,audit.rng_get()));self.assertEqual(before,base.parameter_hash(model))
        self.assertIsNot(payload['model'],model);self.assertTrue(torch.equal(payload['model'].fc.weight.grad,model.fc.weight.grad))
        self.assertTrue(torch.equal(payload['model'].backbone.weight.grad,model.backbone.weight.grad))
        self.assertIsNone(payload['model'].fc.bias.grad);self.assertFalse(payload['model'].drop.training)
        self.assertTrue(torch.equal(payload['model'].forward_cache['tensor'],model.forward_cache['tensor']))
        path=self.root/'snap.pt';torch.save(payload,path)
        torch.rand(10);loaded,state=study.load_snapshot(path,'cpu')
        self.assertTrue(audit.rng_equal(rng,audit.rng_get()));self.assertEqual(base.parameter_hash(loaded),before)
        self.assertTrue(torch.equal(loaded.fc.weight.grad,model.fc.weight.grad))
        self.assertEqual(state['current_optimizer_lr'],.004)
    def test_tampered_snapshot_parameter_identity_rejected(self):
        model=Toy();cfg=study.resolve_config(self.raw(),self.cli);cfg['study_source']=str(self.root)
        p=study._complete_snapshot(model,cfg,0,50,.004)
        with torch.no_grad():p['model'].fc.weight.add_(1.)
        path=self.root/'bad.pt';torch.save(p,path)
        with self.assertRaises(AssertionError):study.load_snapshot(path,'cpu')

if __name__=='__main__':unittest.main(verbosity=2)
