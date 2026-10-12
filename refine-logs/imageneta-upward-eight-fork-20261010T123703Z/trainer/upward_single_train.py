#!/usr/bin/env python3
"""Full original 200-class ImageNet-A, one task, six upward-side rules."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
import audit_core as audit
import task0_trainer_base_v2 as base
import response_train as inherited
from upward_residual_step import READS, FORMULAS
from upward_fork_train import make_study_stepper

PROTOCOL='upward_six_single200_full_original_train_exact_restore_native_skip_v1'
DIRECTIONS=('g0','g1','g2_minus_s','g2','g1_plus_d','g3')

def metadata(cfg):
    return dict(upward_direction=cfg['upward_direction'],gradient_formula=FORMULAS[cfg['upward_direction']],
        native_probe_skipping=True,normal_gradient_calls=READS[cfg['upward_direction']],
        probe_family='upward_sam_response_outer_point',inner_rho=.05,outer_R=1.,
        gam_radii_active=True,probe_radius_coordinates='global trainable LoRA+FC parameter L2')

inherited.make_stepper=make_study_stepper
inherited.radius_metadata=metadata

def resolve_config(raw,cli):
    direction=raw.get('upward_direction');rawseed=raw.get('seed')
    seed=rawseed[0] if isinstance(rawseed,list) and len(rawseed)==1 else rawseed
    if direction not in DIRECTIONS or seed not in (0,1,2) or isinstance(seed,bool):
        raise ValueError('Registered six directions and seed0/1/2 required')
    if raw.get('response_layout')!='single200' or raw.get('init_cls')!=200 or raw.get('increment')!=200:
        raise ValueError('One task must include all 200 classes')
    if raw.get('gam_grad_rho')!=.05 or raw.get('gam_grad_norm_rho')!=1.:
        raise ValueError('Frozen single200 upward radii required')
    if cli.max_epochs not in (0,1) or cli.max_batches not in (0,2) or bool(cli.max_epochs)!=bool(cli.max_batches):
        raise ValueError('Full50epochs or 1epoch/2batch smoke only')
    clean=copy.deepcopy(raw)
    for key in inherited.PROBE_CONFIG_KEYS+('response_probe_variant','radius_variant'):clean.pop(key,None)
    method='sgd' if direction=='g0' else 'current_g2'
    clean.update(seed=2,response_method=method,response_radius_mode='sgd_step',radius_scale=1.,
        history_weight=0.,history_source=None,history_reset='task',post_update_loss=True,strict_deterministic=True)
    surrogate=copy.copy(cli);surrogate.max_tasks=0
    cfg=inherited.resolve_config(clean,surrogate)
    cfg.update(seed=seed,device=[0],upward_direction=direction,protocol=PROTOCOL,
        optimizer_tag_override='upward_single200_'+direction,nb_tasks=1,nb_classes=200,total_sessions=1,
        max_train_tasks=1,task_increments=[200],eval_task_splits=[200],probe_gradient_count=READS[direction],
        compute_g3_probe=READS[direction]==4,gam_radii_active=True)
    inherited.NORMAL_GRADIENTS[method]=READS[direction]
    return cfg

class Controller(inherited.Controller):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.full_test_curve=[];self.passive_eval_seconds=0.
    def finish_epoch(self):
        inherited.native.Controller.finish_epoch(self)
        started=time.perf_counter();model=self.learner._unwrap_network()
        count=correct=0;loss=0.;group_count=[0]*10;group_correct=[0]*10
        with inherited.observational(model):
            model.eval()
            with torch.no_grad():
                for _,inputs,targets in self.test_loader:
                    inputs=inputs.to(self.learner._device);targets=targets.to(self.learner._device)
                    logits=model(inputs)['logits'];pred=logits.argmax(-1)
                    if logits.shape[-1]!=200:raise AssertionError('Single200 classifier is incomplete')
                    loss+=float(F.cross_entropy(logits,targets,reduction='sum'))
                    hit=pred.eq(targets);count+=targets.numel();correct+=int(hit.sum())
                    for group in range(10):
                        mask=(targets>=20*group)&(targets<20*(group+1))
                        group_count[group]+=int(mask.sum());group_correct[group]+=int((mask&hit).sum())
        self.passive_eval_seconds+=time.perf_counter()-started
        if count!=1519 or any(n<=0 for n in group_count) or not math.isfinite(loss):
            raise AssertionError('Full original single200 test budget incomplete')
        sample_accuracy=100.*correct/count
        groups=[100.*c/n for c,n in zip(group_correct,group_count)]
        row=dict(epoch=self.epoch,test_ce=loss/count,test_accuracy=sample_accuracy,
            test_task_macro_accuracy=sum(groups)/10,test_group_accuracies=groups,
            test_group_counts=group_count,test_samples=count,train_ce=self.rows[-1]['train_ce'],
            classification_classes=200,passive_state_and_rng_restored=True,test_used_for_selection=False)
        self.full_test_curve.append(row);self.epoch_test_curve.append(dict(epoch=self.epoch,
            test_accuracy=sample_accuracy,train_ce=self.rows[-1]['train_ce']))
        self.rows[-1].update(test_accuracy_all_200_classes=sample_accuracy,
            test_ce_all_200_classes=loss/count,test_accuracy_original10group_macro=sum(groups)/10)
        audit.atomic_json(self.out/'metrics.json',dict(epochs=self.rows,counters=self.counters))
        audit.atomic_json(self.out/'epoch_test_curve.json',dict(response_layout='single200',
            evaluation_scope='all original1519 test images; all200 classifier',
            selection_rule='fixed endpoint epoch50; no best-epoch selection',epochs=self.full_test_curve))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','config','output-dir'):parser.add_argument('--'+key,required=True)
    for key in ('max-epochs','max-batches'):parser.add_argument('--'+key,type=int,default=0)
    parser.add_argument('--dry-run',action='store_true');cli=parser.parse_args();cli.max_tasks=0
    for key in ('source','config','output_dir'):setattr(cli,key,str(Path(getattr(cli,key)).resolve()))
    cfg=resolve_config(json.loads(Path(cli.config).read_text()),cli)
    if cli.dry_run:
        print(json.dumps(dict(status='valid',protocol=PROTOCOL,seed=cfg['seed'],response_layout='single200',
            classes=200,full_train_samples=5981,full_test_samples=1519,epochs=cli.max_epochs or 50,**metadata(cfg))))
        return
    out=Path(cli.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Strict CUBLAS workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    os.environ.setdefault('HF_HUB_OFFLINE','1');sys.path.insert(0,cli.source);os.chdir(cli.source)
    from utils import factory
    from utils.random_reproduce import set_device,set_random,json_safe
    from utils.data_manager import DataManager
    cfg['filepath']=str(out/'native'/'checkpoints')+os.sep;Path(cfg['filepath']).mkdir(parents=True)
    set_random(cfg['seed']);set_device(cfg)
    dm=DataManager('imageneta',False,cfg['seed'],200,200,cfg)
    learner=factory.get_model(cfg['model_name'],cfg);learner._network.backbone.to(learner._device)
    c=Controller(learner,cfg,cli);c.dm=dm;c.install()
    audit.atomic_json(out/'effective_config.json',json_safe(cfg))
    summary=dict(status='running',protocol=PROTOCOL,seed=cfg['seed'],response_method=cfg['response_method'],
        response_layout='single200',upward_direction=cfg['upward_direction'],smoke=bool(cli.max_epochs),
        epochs_per_task=cli.max_epochs or 50,full_partition_tasks=1,total_dataset_classes=200,
        r=1.,rho=.05,input_config_sha256=audit.digest(cli.config),trainer_sha256=audit.digest(__file__),
        algorithm_sha256=audit.digest(Path(__file__).with_name('upward_residual_step.py')),completed_tasks=0,
        BWT=None,forgetting=None,selection_rule='fixed epoch50; no best-epoch/test selection')
    audit.atomic_json(out/'final_summary.json',summary)
    try:
        learner.incremental_train(dm);learner.eval_task();learner.after_task()
        if len(c.matrix)!=1 or len(c.rows)!=(cli.max_epochs or 50):raise AssertionError('Incomplete single200 budget')
        if any(c.counters[k] for k in ('forbidden_work_attempts','weight_merge_updates','fisher_estimation_passes')):
            raise AssertionError('Forbidden work occurred')
        rows=[json.loads(line) for line in c.trace.read_text().splitlines()]
        compute=sum(row['native_probe_compute_seconds'] for row in rows)
        commit=sum(row['native_sgd_commit_seconds'] for row in rows)
        summary.update(status='completed',completed_tasks=1,completed_epochs=len(c.rows),completed_steps=c.steps,
            counters=c.counters,task_details=c.task_rows,checkpoint_dirs=c.checkpoints,
            dataset_partition_receipt=c.dataset_receipt,initial_identity=c.task_identities[0],history_used_steps=0,
            elapsed_seconds=time.time()-c.started,step_trace_sha256=audit.digest(c.trace),
            task_matrix_sha256=audit.digest(out/'task_matrix.json'),final_metrics=json.loads((out/'task_matrix.json').read_text()),
            endpoint_test=c.full_test_curve[-1],native_training_compute_seconds=compute+commit,
            native_probe_compute_seconds=compute,native_sgd_commit_seconds=commit,
            native_training_peak_allocated_bytes=max(row['native_training_peak_allocated_bytes'] for row in rows),
            passive_extra_eval_seconds=c.passive_eval_seconds,
            final_model_parameter_sha256=base.parameter_hash(learner._unwrap_network()),
            checks=dict(finite=True,exact_snapshot_restore=True,one_sgd_per_batch=True,no_history=True,
                fisher_empty=True,actual_weight_merges_zero=True,full_original_train_test_used=True,test_not_used_for_selection=True),
            **metadata(cfg))
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc),completed_steps=c.steps);raise
    finally:
        c.handle.close();audit.atomic_json(out/'final_summary.json',summary)

if __name__=='__main__':main()
