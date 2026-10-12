"""Immutable completed independent-S targets for native full-task Gmatch."""
import hashlib
import json
import math
from pathlib import Path

import torch
from effective_step import solve_positive_scale

MATCH_RTOL = 1e-3
FORMAL_TASK_STEPS = {'0':500,'1':400,'2':400,'3':250,'4':200,
                     '5':150,'6':150,'7':150,'8':200,'9':150}
SMOKE_TASK_STEPS = {'0':10,'1':8}
CONFIG_PROVENANCE = {'outputs_root','prefix','config','filepath','output_dir',
                     'direction_selector','formation_arm','control_arm','arm',
                     'step_reference_dir','step_reference_job','step_reference_job_id'}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite independent-S evidence')
    if isinstance(value,dict):
        for item in value.values():finite_json(item)
    if isinstance(value,list):
        for item in value:finite_json(item)


class StepReference:
    def __init__(self,spec,source_root,helper_path):
        job=spec['job'];self.job_id=job['id'];self.phase=job['phase']
        self.directory=Path(spec['run_dir']).resolve()
        self.paths={key:self.directory/name for key,name in
                    [('audit','direction_audit.jsonl'),('summary','direction_audit_summary.json'),
                     ('receipt','completed.json'),('config','config.json')]}
        self.hashes={key:sha_file(path) for key,path in self.paths.items()}
        for key,digest in spec.get('pinned_sha256',{}).items():
            if self.hashes.get(key)!=digest:raise ValueError('Pinned S file changed: '+key)
        summary=json.loads(self.paths['summary'].read_text())
        receipt=json.loads(self.paths['receipt'].read_text())
        config=json.loads(self.paths['config'].read_text())
        finite_json(summary);finite_json(receipt);finite_json(config)
        candidate=spec['candidate_config']
        if self.phase not in ('smoke','formal') or spec['candidate_phase']!=self.phase:
            raise ValueError('S and Gmatch smoke/formal phases must match')
        tasks=FORMAL_TASK_STEPS if self.phase=='formal' else SMOKE_TASK_STEPS
        steps=2550 if self.phase=='formal' else 18
        samples=299050 if self.phase=='formal' else 2256
        if (job.get('direction_selector')!='g1_control' or job.get('seed') not in (0,1,2) or job.get('seed') != (candidate.get('seed')[0] if isinstance(candidate.get('seed'),list) else candidate.get('seed'))
                or job.get('expected_steps')!=steps or job.get('expected_samples')!=samples
                or job.get('expected_task_steps')!=tasks):
            raise ValueError('Independent S job has wrong selector/seed/native budget')
        if (summary.get('status')!='completed' or summary.get('selector')!='g1_control'
                or summary.get('steps')!=steps or summary.get('samples')!=samples
                or summary.get('task_steps')!=tasks
                or receipt.get('status')!='completed' or receipt.get('id')!=self.job_id):
            raise ValueError('Independent S is not fully completed at its declared budget')
        if (summary.get('native_ce_forward_passes')!=5*steps
                or summary.get('native_gradient_closures')!=4*steps
                or summary.get('actual_sgd_steps')!=steps
                or summary.get('extra_training_forwards')!=0
                or summary.get('RNG_reseeds_or_restores')!=0
                or summary.get('all_recorded_steps_passed') is not True
                or summary.get('factory_and_bound_methods_restored') is not True):
            raise ValueError('Independent S execution contract differs')
        proof=receipt['direction_proof']
        if (summary.get('audit_sha256')!=self.hashes['audit']
                or proof.get('audit_sha256')!=self.hashes['audit']
                or proof.get('audit_summary_sha256')!=self.hashes['summary']
                or proof.get('steps')!=steps
                or receipt.get('config_sha256')!=self.hashes['config']
                or job.get('config_sha256')!=self.hashes['config']):
            raise ValueError('Independent S receipt does not bind its evidence')
        helper_path=Path(helper_path)
        expected={'legacy_direction_sha256':sha_file(helper_path),
                  'effective_step_sha256':sha_file(helper_path.with_name('effective_step.py')),
                  'step_reference_sha256':sha_file(__file__),
                  'source_gam_sha256':sha_file(Path(source_root)/'optimer_PerturabtionType/gam.py')}
        if any(summary.get(key)!=value for key,value in expected.items()):
            raise ValueError('Independent S code/source identity differs')
        reference_config={key:value for key,value in config.items() if key not in CONFIG_PROVENANCE}
        candidate_config={key:value for key,value in candidate.items() if key not in CONFIG_PROVENANCE}
        if reference_config!=candidate_config:
            raise ValueError('Independent S scientific configuration differs')
        if config.get('seed') not in ([job['seed']],job['seed']):raise ValueError('Independent S configuration seed differs')
        self.rows={};count_samples=0;task_counts={};previous_task=0
        for line in self.paths['audit'].read_text().splitlines():
            row=json.loads(line);finite_json(row)
            task=row['task'];key=(task,row['task_step_index'])
            if (row.get('step_index')!=len(self.rows) or key in self.rows
                    or task not in (previous_task,previous_task+1)
                    or task<0 or str(task) not in tasks
                    or row['task_step_index']!=task_counts.get(str(task),0)
                    or row.get('selector')!='g1_control' or row.get('class_offset')!=20*task
                    or row.get('seen_classes')!=20*(task+1)):
                raise ValueError('Independent S step/task sequence differs')
            if (row.get('forward_count'),row.get('closure_count'),row.get('sgd_count'),
                    row.get('native_decomposition_called'))!=(5,4,1,1):
                raise ValueError('Independent S native call counts differ')
            for flag in ('selected_exact','four_native_gradient_caches_exact','projector_exact','sgd_exact',
                         'fisher_cache_empty','delta_cache_empty','selection_rng_unchanged',
                         'projector_rng_unchanged','sgd_rng_unchanged'):
                if row.get(flag) is not True:raise ValueError('Independent S proof failed: '+flag)
            if (row.get('component_scale')!='natural' or row.get('final_norm_matching') is not False
                    or row.get('step_scale')!=row.get('lr_nominal')
                    or row.get('scale_over_nominal_lr')!=1.):
                raise ValueError('Independent S has altered update scale')
            if row['sgd_expected_parameter_sha256']!=row['sgd_actual_parameter_sha256']:
                raise ValueError('Independent S SGD tensors differ from proof')
            target=row['actual_effective_delta']['joint_norm']
            if not math.isfinite(target) or target<=0:
                raise ValueError('Independent S target must be finite and positive')
            if row.get('effective_geometry_scope')!='unscaled_LoRA_BA_and_all_seen_FC':
                raise ValueError('Independent S effective scope differs')
            task_counts[str(task)]=task_counts.get(str(task),0)+1
            count_samples+=row['batch_samples'];previous_task=task;self.rows[key]=row
        if len(self.rows)!=steps or count_samples!=samples or task_counts!=tasks:
            raise ValueError('Independent S trace budget is incomplete')
        self.identity={'reference_kind':'independent_completed_S_trace',
                       'reference_job_id':self.job_id,'reference_phase':self.phase,
                       'reference_trace_sha256':self.hashes['audit'],
                       'reference_summary_sha256':self.hashes['summary'],
                       'reference_receipt_sha256':self.hashes['receipt'],
                       'reference_config_sha256':self.hashes['config']}

    def get(self,current,nominal_lr):
        key=(current['task'],current['task_step_index'])
        row=self.rows.get(key)
        if row is None:raise ValueError('Missing independent S step')
        for field in ('step_index','task','task_step_index','input_sha256','target_sha256',
                      'rng_before_batch_sha256','seen_classes','class_offset','batch_samples',
                      'parameter_names','parameter_shapes'):
            if current.get(field)!=row.get(field):
                raise ValueError('Independent S native batch identity mismatch: '+field)
        if nominal_lr!=row['lr_nominal']:
            raise ValueError('Independent S nominal LR differs')
        return row

    def verify_unchanged(self):
        if any(sha_file(path)!=self.hashes[key] for key,path in self.paths.items()):
            raise ValueError('Independent S evidence changed during Gmatch')


@torch.no_grad()
def solve_fp32_scale(geometry,before,target,nominal_lr,max_scale=10.):
    """Pure cloned FP32 candidates; no trial optimizer/model update."""
    root=solve_positive_scale(geometry.polynomial(snapshot=before),target,
                              initial_scale=nominal_lr,max_scale=max_scale)
    def measure(scale):
        after=[value.clone().add_(p.grad,alpha=-scale) for p,value in zip(geometry.params,before)]
        delta=geometry.actual_delta(before,after)
        error=abs(delta['joint_norm']-target)/target
        return after,delta,error
    scale=root['scale'];after,delta,error=measure(scale);evaluations=1
    # Usually one candidate suffices. If SGD rounding is material, search only
    # the same first polynomial crossing interval, never a live trial step.
    if error>MATCH_RTOL:
        lo,hi=root['bracket']
        _,low,_=measure(lo);_,high,_=measure(hi);evaluations+=2
        if not low['joint_norm']<=target<=high['joint_norm']:
            raise ValueError('FP32 target is not bracketed in first feasible interval')
        for _ in range(60):
            scale=(lo+hi)/2;after,delta,error=measure(scale);evaluations+=1
            if error<=MATCH_RTOL:break
            if delta['joint_norm']<target:lo=scale
            else:hi=scale
        if error>MATCH_RTOL:raise ValueError('No FP32 step meets matching tolerance')
    if not math.isfinite(scale) or not 0.<scale<=max_scale:
        raise ValueError('Invalid matched SGD scale')
    return scale,after,{'polynomial_root':root,'fp32_predicted_effective_delta':delta,
                        'fp32_predicted_relative_error':error,'fp32_candidate_evaluations':evaluations,
                        'matching_tolerance':MATCH_RTOL,'no_live_trial_updates':True}
