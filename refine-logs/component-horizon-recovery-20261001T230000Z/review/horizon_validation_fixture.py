#!/usr/bin/env python3
"""Strict validator fixtures, using real CPU native-GAM step records.

These are synthetic completion envelopes, not ViT experiment results. The
positive must be accepted first; malformed fixtures must be rejected for their
specific corruption (not an unrelated KeyError in an incomplete positive).
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

STAGE=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(STAGE/'trainer'))
sys.path.insert(0,str(STAGE))
import torch
import component_horizon as component
import test_component_cpu as tiny


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def build_fixture(root,rule='us',scale='matched'):
    root=Path(root);out=root/'outputs'/'fixture';out.mkdir(parents=True)
    for name in ('trainer','diagnostics'):(root/name).symlink_to(STAGE/name,target_is_directory=True)
    inventory=read(STAGE/'data_inventory.json')
    write(root/'data_inventory.json',inventory)
    write(root/'source_manifest.json',read(STAGE/'source_manifest.json'))
    write(root/'suite.json',{'expected_tasks':2,'source_root':str(tiny.SOURCE),'source_manifest':str(root/'source_manifest.json'),'jobs':[]})
    raw=read(STAGE/'configs'/'smoke_us_matched_r025_s2.json')
    raw.update(component_rule=rule,component_scale=scale,outputs_root=str(out),prefix='cpu_fixture')
    config=root/'input_config.json';write(config,raw)
    cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=str(out))
    cfg=component.resolve_config(raw,cli);cfg['seed']=2;write(out/'effective_config.json',cfg)
    job={'id':'cpu_fixture','kind':'horizon','config':str(config),'config_sha256':sha(config),
         'output_dir':str(out),'seed':2,'rule':rule,'scale':scale,'radius':.25,
         'smoke':True,'expected_tasks':2,'expected_epochs_per_task':1,'expected_steps':2}
    rows=[];identities=[];details=[];after=[];checkpoints=[]
    torch.manual_seed(521);model=tiny.Toy();x=torch.randn(128,7);ids=torch.arange(128);y=ids%20
    for task in range(2):
        if task:model.expand(40)
        row=tiny.run_step(tiny.make_step(model,rule,scale),x,y+task*20)
        row.update(step_index=task,task_step_index=0,task=task,epoch=1,batch=0,seed=2,
                   identity={'input_sha256':component.audit.tensor_digest(x),'target_sha256':component.audit.tensor_digest(y+task*20),
                             'indices_sha256':component.audit.tensor_digest(ids),'indices':ids.tolist(),'data_seed':200000+task*1000+1},
                   seen_classes=20*(task+1),fc_shape=list(model.fc.weight.shape),class_slice=[20*task,20*(task+1)],
                   training_forward_count=5,training_ce_gradient_passes=5,batch_samples=128,train_ce=2.)
        rows.append(row)
        cp=out/'checkpoints'/('task_%03d'%task);cp.mkdir(parents=True)
        names=['lora_w_a_%d.pt'%task,'lora_w_b_%d.pt'%task,'lora_meta_%d.json'%task,'fc_state_%d.pt'%task]
        torch.save(list(model.backbone.w_As),cp/names[0]);torch.save(list(model.backbone.w_Bs),cp/names[1])
        write(cp/names[2],model.backbone.w_meta);torch.save(model.fc.state_dict(),cp/names[3])
        torch.save({'task':task,'epoch':1,'fixture':True},cp/'training_state.pt')
        write(cp/'manifest.json',{'task':task,'epoch':1,'seed':2,'component_rule':rule,'component_scale':scale,
            'class_range':[0,20*(task+1)],'current_task_class_range':[20*task,20*(task+1)],
            'model_sha256':component.base.parameter_hash(model),'source_gam_sha256':sha(tiny.SOURCE/'optimer_PerturabtionType/gam.py'),
            'fc_shape':list(model.fc.weight.shape),'backbone_task_id':task+1,'native_checkpoint_dir':str(out/'native'),
            'files_sha256':{name:sha(cp/name) for name in names},'loader':'load_task_checkpoint(directory, task, data_manager)'})
        checkpoints.append(str(cp))
        ident={'task':task,'seed':2,'old_fc_rows_inherited_exact':True,'fc_shape':list(model.fc.weight.shape),
               'component_rule':rule,'component_scale':scale,'class_slice':[20*task,20*(task+1)],
               'new_fc_rows_sha256':{k:component.audit.tensor_digest(v[20*task:]) for k,v in model.fc.state_dict().items()},
               'task_initial_parameter_sha256':component.base.parameter_hash(model),'frozen_origin_sha256':component.base.parameter_hash(model,True),
               'data_seed_rule':'seed*100000+task*1000+epoch (epoch 1-based)','native_global_rng_protocol_preserved':False}
        identities.append(ident)
        details.append({'task':task,'train_dataset_samples':inventory['tasks'][task]['train_n'],
            'batches_per_epoch':1,'epochs':1,'steps':1,'checkpoint':str(cp)})
        after.append({'task':task,'parameters_exact':True,'parameter_references_exact':True,
                      'fisher_empty':True,'known_classes':20*(task+1)})
    write(out/'task_identities.json',identities);write(out/'tasks.json',details)
    write(out/'after_task_checks.json',after)
    write(out/'task_checkpoints.json',{'loader':'load_task_checkpoint(directory, task, data_manager)',
        'all_tasks':{str(i):p for i,p in enumerate(checkpoints)},'representative_tasks':{'0':checkpoints[0]}})
    (out/'step_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    matrix={'matrix':[[50.],[40.,60.]],'matrix_scope':'native CNN argmax over all seen classes; rows=time, columns=task',
            'AAA_task_macro':50.,'FAA_task_macro':50.,'LA':55.,'BWT':-10.,'forgetting':10.,
            'sample_accuracy_curve':[50.,51.],'FAA_sample_weighted':51.,'AAA_sample_weighted':50.5}
    write(out/'task_matrix.json',matrix)
    native={'cnn':{'matrices':{'t00':[[50.]],'final':[[50.,None],[40.,60.]]},
                   'final':{'FAA':50.,'AAA':50.,'FWT':None,'AFG':None}}}
    write(out/'native'/'fixture_cl_metrics.json',native)
    log=root/'runtime'/'jobs'/job['id']/'console.log';log.parent.mkdir(parents=True)
    log.write_text('CPU synthetic fixture (not an experiment result)\nCNN top1 curve: [50.0, 51.0]\n')
    ctr={'training_batches':2,'training_samples':256,'training_ce_gradient_passes':10,
         'fisher_estimation_passes':0,'weight_merge_updates':0,'forbidden_work_attempts':0,'disabled_merge_hook_calls':4}
    write(out/'metrics.json',{'epochs':[{'task':t,'epoch':1,'samples':128,'batches':1,'train_ce':2.,
        'train_accuracy_local':50.,'lr_used':[.04 if t==0 else .02],
        'lr_after_scheduler':[(.04 if t==0 else .02)*(.1+.45*(1+__import__('math').cos(__import__('math').pi/50)))], 'sampler_order_sha256':'fixture',
        'seconds':1.,'samples_per_second':128.,'peak_gpu_allocated_bytes':0} for t in range(2)],'counters':ctr})
    s={'status':'completed','schema_version':1,'seed':2,'component_rule':rule,'component_scale':scale,
       'r':.25,'rho':.05,'smoke':True,'requested_tasks':2,'epochs_per_task':1,'scheduler_T_max':50,
       'completed_tasks':2,'task_count':2,'completed_epochs':2,'completed_steps':2,
       'reference_definition':'same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD',
       'trainer_sha256':sha(STAGE/'trainer/component_horizon.py'),'effective_step_sha256':sha(STAGE/'trainer/effective_step.py'),
       'source_gam_sha256':sha(tiny.SOURCE/'optimer_PerturabtionType/gam.py'),
       'input_config_sha256':sha(config),'source':str(tiny.SOURCE),'input_config':str(config),
       'effective_config_sha256':sha(out/'effective_config.json'),'counters':ctr,'task_details':details,
       'checkpoint_dirs':checkpoints,'after_task_checks':after,'initial_identity':identities[0],
       'checks':{'finite':True,'all_steps_self_spec':True,'all_matched_steps':True if scale=='matched' else None,
                 'fisher_empty_all_tasks':True,'actual_weight_merges_zero':True,'complete_task_lifecycle':True},
       'task_identities_sha256':sha(out/'task_identities.json'),'task_checkpoints_sha256':sha(out/'task_checkpoints.json'),
       'matched_steps':2 if scale=='matched' else 0,'max_match_relative_error':max(r['actual_match_relative_error'] for r in rows) if scale=='matched' else None,
       'step_trace_sha256':sha(out/'step_trace.jsonl'),'task_matrix_sha256':sha(out/'task_matrix.json'),
       'final_metrics':matrix,'finite_check':{'passed':True},'elapsed_seconds':2.}
    write(out/'final_summary.json',s)
    suite=read(root/'suite.json');suite['jobs']=[job];write(root/'suite.json',suite)
    return job


def patch_trace(out,fn):
    path=out/'step_trace.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    fn(rows);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    s=read(out/'final_summary.json');s['step_trace_sha256']=sha(path);write(out/'final_summary.json',s)


def patch_json(path,fn):
    d=read(path);fn(d);write(path,d)


def mutations():
    """Rehash semantic corruptions so rejection is not merely a stale hash."""
    def summary(key,value):return lambda o,j:patch_json(o/'final_summary.json',lambda s:s.__setitem__(key,value))
    def trace(key,value):return lambda o,j:patch_trace(o,lambda rs:rs[0].__setitem__(key,value))
    def checkpoint_omit(o,j):
        cp=o/'checkpoints/task_000';name='lora_w_b_0.pt';(cp/name).unlink()
        patch_json(cp/'manifest.json',lambda m:m['files_sha256'].pop(name))
    def checkpoint_source(o,j):patch_json(o/'checkpoints/task_000/manifest.json',lambda m:m.__setitem__('source_gam_sha256','0'*64))
    def sample_curve(o,j):
        p=o/'task_matrix.json';patch_json(p,lambda m:m.__setitem__('sample_accuracy_curve',[50.,52.]))
        patch_json(o/'final_summary.json',lambda s:s.update(task_matrix_sha256=sha(p),final_metrics=read(p)))
    def native_nan(o,j):patch_json(o/'native/fixture_cl_metrics.json',lambda m:m['cnn']['matrices']['final'][1].__setitem__(0,float('nan')))
    def config_lr(o,j):
        p=o/'effective_config.json';patch_json(p,lambda c:c.__setitem__('init_lr',.041))
        patch_json(o/'final_summary.json',lambda s:s.__setitem__('effective_config_sha256',sha(p)))
    def config_beta(o,j):
        p=o/'effective_config.json';patch_json(p,lambda c:c.__setitem__('gam_grad_beta_3',1.))
        patch_json(o/'final_summary.json',lambda s:s.__setitem__('effective_config_sha256',sha(p)))
    def norm_error(o,j):
        def mutate(rows):
            r=rows[0];r['actual_effective_delta']['joint_norm']=r['reference_target_norm']*1.0011
            r['actual_match_relative_error']=.0011;r['actual_to_reference_norm_ratio']=1.0011
        patch_trace(o,mutate)
    def fisher(o,j):patch_json(o/'final_summary.json',lambda s:s['counters'].__setitem__('fisher_estimation_passes',1))
    def closure(o,j):patch_trace(o,lambda rs:rs[0].__setitem__('closure_count',3))
    return {
        'wrong_status':summary('status','running'),'wrong_completed_tasks':summary('completed_tasks',1),
        'missing_trace_row':lambda o,j:patch_trace(o,lambda rs:rs.pop()),
        'wrong_row_lr':trace('lr_nominal',.041),'wrong_effective_config_lr':config_lr,
        'wrong_native_beta':config_beta,
        'projector_false':trace('projector_spec_exact',False),'projector_not_bool':trace('projector_spec_exact',1),
        'fisher_pass_performed':fisher,'closure_count_wrong':closure,
        'nonfinite_trace_value':trace('reference_target_norm',float('nan')),
        'zero_reference_target':trace('reference_target_norm',0.),
        'us_not_nominal':trace('step_scale',.06),'us_reference_after_not_exact':trace('us_nominal_reference_after_tensors_exact',False),
        'wrong_step_order':trace('step_index',1),'wrong_class_slice':trace('class_slice',[0,40]),
        'matched_error_exceeds_bound':norm_error,
        'wrong_summary_reference_definition':summary('reference_definition','independent_S_trajectory'),
        'wrong_row_reference_definition':trace('reference_definition','independent_S_trajectory'),
        'stale_trace_hash':summary('step_trace_sha256','0'*64),
        'stale_input_config_hash':summary('input_config_sha256','0'*64),
        'wrong_source_gam_hash':summary('source_gam_sha256','0'*64),
        'checkpoint_native_file_missing':checkpoint_omit,'checkpoint_wrong_source':checkpoint_source,
        'wrapper_sample_curve_disagrees_log':sample_curve,'native_matrix_nan':native_nan,
    }


def run_tests(validator=None):
    torch.set_num_threads(1)
    if validator is None:
        spec=importlib.util.spec_from_file_location('component_fixture_validator',STAGE/'validate_job.py')
        validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    common=sys.modules[validator.inspect_metrics.__module__]
    previous=(validator.HERE,common.HERE)
    positives=[];rejected=[];unexpected=[]
    try:
        for rule in component.RULES:
            for scale in ('natural','matched'):
                with tempfile.TemporaryDirectory(prefix='horizon_validator_positive_') as tmp:
                    root=Path(tmp);job=build_fixture(root,rule,scale);validator.HERE=common.HERE=root
                    try:result=validator.validate_job(job)
                    except Exception as exc:raise AssertionError('Well-formed '+rule+'/'+scale+' rejected: '+repr(exc)) from exc
                    assert result['summary']['status']=='completed' and len(result['checkpoints'])==2
                    positives.append(rule+'_'+scale)
        for label,mutate in mutations().items():
            with tempfile.TemporaryDirectory(prefix='horizon_validator_negative_') as tmp:
                root=Path(tmp);job=build_fixture(root);validator.HERE=common.HERE=root
                # Establish this fixture is complete before corrupting it.
                result=validator.validate_job(job);assert result['summary']['status']=='completed'
                mutate(Path(job['output_dir']),job)
                try:validator.validate_job(job)
                except (ValueError,AssertionError,FileNotFoundError) as exc:rejected.append({'case':label,'error':repr(exc)})
                except Exception as exc:raise AssertionError('Negative '+label+' failed for an unexpected/incomplete-fixture reason: '+repr(exc)) from exc
                else:unexpected.append(label)
    finally:validator.HERE,common.HERE=previous
    return {'status':'passed' if not unexpected else 'failed','fixture_scope':'synthetic schema envelopes with actual CPU frozen-GAM step records; not ViT experiments',
            'positive_cases':positives,'positive_count':len(positives),'negative_cases_rejected':rejected,
            'negative_count':len(rejected),'unexpectedly_accepted':unexpected,'validator_sha256':sha(STAGE/'validate_job.py'),
            'common_sha256':sha(STAGE/'common.py'),'fixture_sha256':sha(__file__)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    result=run_tests()
    if a.output:write(a.output,result)
    print(json.dumps(result,indent=2))
    assert result['status']=='passed', 'Validator accepted malformed envelopes: '+str(result['unexpectedly_accepted'])
