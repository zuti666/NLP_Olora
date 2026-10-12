"""Exercise resolver and actual installed training batch, without GPU/data."""
from pathlib import Path
import copy
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'trainer'))
sys.path.insert(0, str(ROOT))
from common import PROTOCOL, expected_budget
import four_probe_train as trainer
from direction_step import expected_budget as executor_budget
from validate_job import validate_second_probe, validate_reuse


def raw(variant):
    value=json.loads((ROOT/'plan_snapshot/configs'/f'{variant}.draft.json').read_text())
    value.update(protocol=PROTOCOL, component_rule=value['update_direction'], registered=True, plan_only=False)
    return value


def cli(output):
    return SimpleNamespace(max_tasks=0, max_epochs=0, max_batches=0, output_dir=str(output), source='unused')


def test_registered_configs_and_rejections():
    for variant in ('CPLUS','CMINUS','BF','BB'):
        cfg=trainer.resolve_config(raw(variant),cli('/tmp/cj_owned_resolve'))
        assert cfg['protocol']==PROTOCOL
        assert expected_budget(cfg)==executor_budget(cfg)
        assert expected_budget(cfg)['train_ce_per_update']==(5 if variant.startswith('C') else 4)
        assert cfg['probe_branch']==('P' if variant.startswith('C') else 'C')
        assert cfg['extra_diagnostic_gradient_calls']==0 and not cfg['second_probe_side_diagnostics']
        assert cfg['gam_grad_norm_rho']==1. and cfg['lora_rank']==10
        assert all(key not in cfg for key in ('first_probe_sign','second_probe_sign','second_difference_sign'))
    for field,value in [('seed',[3]),('lora_rank',20),('gam_grad_norm_rho',.25),('gam_grad_rho',.1),
                         ('extra_diagnostic_gradient_calls',1),('protocol','old'),
                         ('cuda_deterministic_warn_only',True),('component_rule','u0'),('residual_coefficient',0)]:
        bad=raw('BB');bad[field]=value
        try:trainer.resolve_config(bad,cli('/tmp/cj_owned_reject'))
        except ValueError:pass
        else:raise AssertionError('Accepted incompatible factor '+field)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(3,20,dtype=torch.float64)
    def forward(self,inputs):return {'logits':self.fc(inputs)}


class Geometry:
    def __init__(self,model):self.params=tuple(model.parameters())
    def actual_delta(self,before,after=None):
        after=tuple(p.detach() for p in self.params) if after is None else after
        return {'norm':float(torch.linalg.vector_norm(torch.cat([(a-b).flatten() for a,b in zip(after,before)])))}


class Learner:
    def __init__(self,model):self.model=model;self._total_classes=20;self._cur_task=0
    def _unwrap_network(self):return self.model
    def _task_loss(self,output,targets,offset):return F.cross_entropy(output[:,offset:offset+20],targets-offset),None,None
    def _run_regularized_train(self,*a,**kw):raise AssertionError('Lifecycle is outside batch unit test')
    def incremental_train(self,*a,**kw):pass
    def eval_task(self,*a,**kw):pass
    def after_task(self,*a,**kw):pass


class Controller(trainer.Controller):
    def save_observation(self,point):pass


def test_actual_installed_batch_dynamic_budget_no_diagnostics():
    old_det=torch.are_deterministic_algorithms_enabled()
    old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    old_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    try:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        trainer.configure_deterministic_execution()
        for variant in ('CPLUS','CMINUS','BF','BB'):
            with tempfile.TemporaryDirectory(prefix='cj_batch_') as directory:
                cfg=trainer.resolve_config(raw(variant),cli(directory));cfg['seed']=2
                torch.manual_seed(13);model=Model();learner=Learner(model)
                controller=Controller(learner,cfg,cli(directory))
                controller.task=0;controller.epoch=1;controller.batch=0
                controller.task_start_step=0;controller.old_fc={};controller.batch_identity={'unit_test_batch':True}
                controller.samples=controller.loss_sum=controller.correct=controller.epoch_batches=0
                controller.install()
                with patch.object(trainer.EffectiveStep,'from_model',side_effect=lambda m:Geometry(m)):
                    opt=learner._build_optimizer(model.parameters(),'init')
                before=tuple(p.detach().clone() for p in model.parameters())
                inputs=torch.tensor([[.2,.4,-.1],[-.3,.2,.5],[.4,-.1,.2]],dtype=torch.float64)
                targets=torch.tensor([0,1,2])
                with patch.object(trainer.FourProbeStep,'diagnose_sides',side_effect=AssertionError('Extra diagnostic forbidden')):
                    output,loss,log=learner._step_batch_with_log(opt,inputs,targets,0)
                controller.handle.close()
                record=json.loads((Path(directory)/'step_trace.jsonl').read_text())
                budget=expected_budget(cfg)
                assert record['ordinary_gradient_calls']==record['training_forward_count']==budget['train_ce_per_update']
                assert record['sgd_updates']==1 and controller.steps==1
                assert controller.counters['training_ce_gradient_passes']==budget['train_ce_per_update']
                assert controller.counters['training_first_positive_gradient_passes']==budget['first_plus_ce_per_update']
                assert controller.counters['training_g3_positive_gradient_passes']==budget['g3_plus_ce_per_update']
                assert not (Path(directory)/'side_diagnostics.jsonl').read_text()
                assert controller.counters['diagnostic_gradient_calls']==0
                for side,audit in record['second_probe_parameter_audits'].items():
                    validate_second_probe(record,1 if side=='plus' else -1,side)
                if variant=='BB':
                    assert controller.counters['training_local_positive_gradient_passes']==0
                    assert record['first_gradient_label']=='g1_minus'
                if variant.startswith('C'):
                    assert record['centered_response']['measured_from_training_context'] is True
                    assert record['centered_response']['additional_gradient_calls']==0
                assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
                assert torch.isfinite(output).all() and log['ewc']==0
    finally:
        torch.use_deterministic_algorithms(old_det,warn_only=old_warn)
        if old_workspace is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        else:os.environ['CUBLAS_WORKSPACE_CONFIG']=old_workspace


def test_second_probe_audit_rejects_wrong_sign():
    record={'rho':.05,'eps':1e-12,'norms':{'g2':1.},'second_probe_parameter_audits':{'minus':{
        'passed':True,'requested_sign':-1,'target_radius':.05,'expected_parameter_length':.05/(1.+1e-12),
        'absolute_tolerance':.05*1e-4,'cosine_tolerance':1e-4,'degenerate':False,
        'parameter_length':.05/(1.+1e-12),'length_error':0.,'alignment_cosine':-1.}}}
    validate_second_probe(record,-1,'minus')
    changed=copy.deepcopy(record);changed['second_probe_parameter_audits']['minus']['alignment_cosine']=1.
    try:validate_second_probe(changed,-1,'minus')
    except AssertionError:pass
    else:raise AssertionError('Wrong actual side accepted')


if __name__=='__main__':
    for test in (test_registered_configs_and_rejections,test_actual_installed_batch_dynamic_budget_no_diagnostics,
                 test_second_probe_audit_rejects_wrong_sign):
        test();print(test.__name__+': PASS')
