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


def actual_step(model,rule,scale,protocol,x,y):
    """Real native GAM plus real autograd HVP, no pre-baked response flags."""
    from types import MethodType
    import torch.nn.functional as F
    learner=SimpleNamespace(args=protocol,_unwrap_network=lambda:model,_lambda_flat=1.,_mechanism_eps=1e-12,_mechanism_grad_stats=False)
    projector=MethodType(tiny.native_projector(),learner)
    sgd=torch.optim.SGD(model.parameters(),lr=.04 if model.fc.out_features==20 else .02,momentum=0.,weight_decay=0.)
    opt=tiny.GAM(model.parameters(),sgd,model,grad_rho=.05,grad_norm_rho=1.,adaptive=False,
        args=SimpleNamespace(grad_beta_1=1.,grad_beta_2=1.,grad_beta_3=0.,grad_gamma=0.,use_sgv=False))
    stepper=component.ComponentStepAudit(learner,opt,rule,scale);lo=model.fc.out_features-20
    stepper.set_batch(x,y,lo,lo+20);count=[0]
    hook=model.register_forward_hook(lambda *_:count.__setitem__(0,count[0]+1))
    try:
        with component.hvp.backend_context(protocol['component_attention_backend']):
            loss=F.cross_entropy(model(x)['logits'][:,lo:lo+20],y-lo)
            gs=torch.autograd.grad(loss,list(model.parameters()))
            clean={id(p):g.detach().clone() for p,g in zip(model.parameters(),gs)}
            opt.set_direction_projector(stepper.projector(projector(clean,{},{}),clean,{},{}))
            def closure():
                opt.zero_grad();values=model(x);ce=F.cross_entropy(values['logits'][:,lo:lo+20],y-lo);ce.backward()
                return values,ce.detach()
            opt.step(closure)
    finally:hook.remove()
    row=copy.deepcopy(stepper.record)
    expected_hvp=int(protocol['component_first_response']=='hvp')+int(protocol['component_second_response']=='hvp')
    assert count[0]==5+expected_hvp and row['closure_count']==4
    row.update(training_forward_count=count[0],native_training_forward_count=5,train_ce=float(loss.detach()))
    return row


def build_fixture(root,rule='us',scale='matched',first='fd',second='fd',backend='native',coefficient=1.):
    root=Path(root);out=root/'outputs'/'fixture';out.mkdir(parents=True)
    for name in ('trainer','diagnostics'):(root/name).symlink_to(STAGE/name,target_is_directory=True)
    inventory=read(STAGE/'data_inventory.json')
    write(root/'data_inventory.json',inventory)
    write(root/'source_manifest.json',read(STAGE/'source_manifest.json'))
    write(root/'suite.json',{'expected_tasks':2,'source_root':str(tiny.SOURCE),'source_manifest':str(root/'source_manifest.json'),'jobs':[]})
    raw=read(STAGE/'review/fixture_base_config.json')
    protocol={'component_first_response':first,'component_second_response':second,'component_lambda_corr':coefficient,'component_attention_backend':backend}
    raw.update(protocol);raw['gam_grad_norm_rho']=1.
    raw.update(component_rule=rule,component_scale=scale,outputs_root=str(out),prefix='cpu_fixture')
    config=root/'input_config.json';write(config,raw)
    cli=SimpleNamespace(max_tasks=2,max_epochs=1,max_batches=1,output_dir=str(out))
    cfg=component.resolve_config(raw,cli);cfg['seed']=2;write(out/'effective_config.json',cfg)
    job={'id':'cpu_fixture','kind':'horizon','config':str(config),'config_sha256':sha(config),
         'output_dir':str(out),'seed':2,'rule':rule,'scale':scale,'radius':1.,'first_response':first,'second_response':second,'attention_backend':backend,'lambda_corr':coefficient,
         'smoke':True,'expected_tasks':2,'expected_epochs_per_task':1,'expected_steps':2}
    rows=[];identities=[];details=[];after=[];checkpoints=[]
    torch.manual_seed(521);model=tiny.Toy();x=torch.randn(128,7);ids=torch.arange(128);y=ids%20
    for task in range(2):
        if task:model.expand(40)
        row=actual_step(model,rule,scale,protocol,x,y+task*20)
        row.update(step_index=task,task_step_index=0,task=task,epoch=1,batch=0,seed=2,
                   identity={'input_sha256':component.audit.tensor_digest(x),'target_sha256':component.audit.tensor_digest(y+task*20),
                             'indices_sha256':component.audit.tensor_digest(ids),'indices':ids.tolist(),'data_seed':200000+task*1000+1},
                   seen_classes=20*(task+1),fc_shape=list(model.fc.weight.shape),class_slice=[20*task,20*(task+1)],
                   training_ce_gradient_passes=5,batch_samples=128)
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
            'response_config':protocol,**protocol,'files_sha256':{name:sha(cp/name) for name in names},'loader':'load_task_checkpoint(directory, task, data_manager)'})
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
         'fisher_estimation_passes':0,'weight_merge_updates':0,'forbidden_work_attempts':0,'disabled_merge_hook_calls':4,
         **{k:sum(r['response_audit']['hvp_forward_count'] for r in rows) for k in ('hvp_forward_passes','hvp_first_derivative_passes','hvp_second_derivative_passes')},
         'total_training_forward_passes':sum(r['training_forward_count'] for r in rows)}
    write(out/'metrics.json',{'epochs':[{'task':t,'epoch':1,'samples':128,'batches':1,'train_ce':2.,
        'train_accuracy_local':50.,'lr_used':[.04 if t==0 else .02],
        'lr_after_scheduler':[(.04 if t==0 else .02)*(.1+.45*(1+__import__('math').cos(__import__('math').pi/50)))], 'sampler_order_sha256':'fixture',
        'seconds':1.,'samples_per_second':128.,'peak_gpu_allocated_bytes':0} for t in range(2)],'counters':ctr})
    s={'status':'completed','schema_version':1,'seed':2,'component_rule':rule,'component_scale':scale,
       'r':1.,'rho':.05,'smoke':True,'requested_tasks':2,'epochs_per_task':1,'scheduler_T_max':50,
       'completed_tasks':2,'task_count':2,'completed_epochs':2,'completed_steps':2,
       'reference_definition':'same_candidate_state_same_batch_native_projected_g1_after_native_unperturb_actual_FP32_SGD',
       'trainer_sha256':sha(STAGE/'trainer/component_horizon.py'),'effective_step_sha256':sha(STAGE/'trainer/effective_step.py'),
       'source_gam_sha256':sha(tiny.SOURCE/'optimer_PerturabtionType/gam.py'),
       'response_config':protocol,**protocol,'hvp_helpers_sha256':sha(STAGE/'trainer/hvp_helpers.py'),
       'curvature_primitives_sha256':sha(STAGE/'diagnostics/curvature_audit.py'),
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


def response_mutations():
    def row(fn):return lambda o,j:patch_trace(o,lambda rs:fn(rs[0]))
    def field(key,value):return row(lambda r:r['response_audit'].__setitem__(key,value))
    def hvp(key,value):return row(lambda r:r['response_audit']['hvp_reports']['second'].__setitem__(key,value))
    def cfg(o,j):
        p=o/'effective_config.json';patch_json(p,lambda c:c.__setitem__('component_second_response','fd'))
        patch_json(o/'final_summary.json',lambda s:s.__setitem__('effective_config_sha256',sha(p)))
    def counter(o,j):patch_json(o/'final_summary.json',lambda s:s['counters'].__setitem__('hvp_forward_passes',0))
    def checkpoint(o,j):patch_json(o/'checkpoints/task_000/manifest.json',lambda m:m.__setitem__('component_second_response','fd'))
    def rho_norm(o,j):patch_trace(o,lambda rs:rs[0]['response_audit']['hvp_reports']['second'].__setitem__('rho_hvp_norm',rs[0]['response_audit']['hvp_reports']['second']['hvp_norm']))
    return {
        'effective_response_changed':cfg,'checkpoint_response_changed':checkpoint,
        'summary_hvp_counter_zero':counter,'response_backend_native':field('backend','native'),
        'response_lambda_altered':field('lambda_corr',.5),'response_rho_altered':field('rho',1.),
        'response_r_altered':field('r',.25),'selected_formula_FD_alias':field('selected_formula','g0 + lambda_corr * (g3 - g2)'),
        'response_fallback_true':field('fallback_used',True),'hvp_forward_not_executed':field('hvp_forward_count',0),
        'hvp_second_derivative_not_executed':field('hvp_second_derivative_passes',0),
        'hvp_status_unsupported':hvp('status','unsupported'),'hvp_backend_native':hvp('backend','native'),
        'hvp_wrong_original_point':hvp('stage','original'),'hvp_wrong_closure':hvp('native_closure_index',0),
        'hvp_direction_not_detached':hvp('direction_detached',False),'hvp_rho_not_exact':hvp('rho_multiplication_exact',False),
        'hvp_missing_rho_multiplier':rho_norm,'hvp_rho_wrong':hvp('rho',1.),
        'hvp_false_fallback':hvp('fallback_used',True),'hvp_nonfinite':hvp('hvp_norm',float('nan')),
        'hvp_samepoint_ce_wrong':hvp('same_point_ce',20.),
        'hvp_samepoint_gradient_wrong':hvp('same_point_gradient',{'relative_l2':.01,'max_abs':.1}),
        'hvp_restore_empty':hvp('entry_state_restore',{}),
        'hvp_restore_rng_false':row(lambda r:r['response_audit']['hvp_reports']['second']['entry_state_restore'].__setitem__('rng_exact',False)),
        'outer_point_false':row(lambda r:r['response_audit']['native_point_checks'].__setitem__('outer_point_exact',False)),
        'hvp_report_missing':field('hvp_reports',{}),'selected_correction_wrong':field('correction_norm',200.),
        'selected_gradient_did_not_reach_SGD':row(lambda r:r['gradient_flow'].__setitem__('actual_sgd_gradient_sha256','0'*64)),
        'candidate_gradient_hash_wrong':row(lambda r:r['gradient_flow'].__setitem__('selected_actual_sha256','0'*64)),
        'flow_gate_not_boolean':row(lambda r:r['gradient_flow'].__setitem__('selected_to_projector_to_sgd_exact',1)),
        'total_forward_undercounted':row(lambda r:r.__setitem__('training_forward_count',5)),
        'native_forward_overcounted':row(lambda r:r.__setitem__('native_training_forward_count',6)),
        'wrong_hvp_helper_source_pin':lambda o,j:patch_json(o/'final_summary.json',lambda s:s.__setitem__('hvp_helpers_sha256','0'*64)),
        'wrong_hvp_primitive_source_pin':lambda o,j:patch_json(o/'final_summary.json',lambda s:s.__setitem__('curvature_primitives_sha256','0'*64)),
        'math_control_flash_enabled':field('backend_flags',{'flash':True,'math':True,'mem_efficient':False,'cudnn':False}),
        'hvp_cudnn_backend_enabled':hvp('backend_flags',{'flash':False,'math':True,'mem_efficient':False,'cudnn':True}),
        'hvp_detached_but_wrong_direction':hvp('direction_matches_native_unit_exact',False),
        'hvp_direction_norm_nonunit':hvp('direction_norm',2.),
    }


def run_tests(validator=None):
    torch.set_num_threads(1)
    if validator is None:
        spec=importlib.util.spec_from_file_location('fdhvp_fixture_validator',STAGE/'validate_job.py')
        validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    common=sys.modules[validator.inspect_metrics.__module__];previous=(validator.HERE,common.HERE)
    positives=[];rejected=[];unexpected=[];paired=None
    cases=[(rule,scale,'fd','fd','native',1.) for rule in component.RULES for scale in ('natural','matched')]
    cases += [('ug',scale,first,second,'math',1.) for first,second in (('fd','fd'),('fd','hvp'),('hvp','fd'),('hvp','hvp')) for scale in ('natural','matched')]
    cases += [('us',scale,'fd','fd','native',.5) for scale in ('natural','matched')]
    pins={str(p.relative_to(STAGE)):sha(p) for p in [STAGE/'validate_job.py',STAGE/'trainer/component_horizon.py',STAGE/'trainer/hvp_helpers.py',STAGE/'trainer/effective_step.py',STAGE/'diagnostics/curvature_audit.py',STAGE/'common.py',STAGE/'review/fixture_base_config.json']}
    try:
        for case in cases:
            label='/'.join(map(str,case))
            with tempfile.TemporaryDirectory(prefix='fdhvp_validator_positive_') as tmp:
                root=Path(tmp);job=build_fixture(root,*case);validator.HERE=common.HERE=root
                try:result=validator.validate_job(job)
                except Exception as exc:raise AssertionError('Well-formed '+label+' rejected: '+repr(exc)) from exc
                assert result['summary']['status']=='completed' and len(result['checkpoints'])==2
                positives.append(label)
        for group,case in ((mutations(),('us','matched','fd','fd','native',1.)),(response_mutations(),('ug','matched','fd','hvp','math',1.))):
            for label,mutate in group.items():
                with tempfile.TemporaryDirectory(prefix='fdhvp_validator_negative_') as tmp:
                    root=Path(tmp);job=build_fixture(root,*case);validator.HERE=common.HERE=root
                    accepted=validator.validate_job(job);assert accepted['summary']['status']=='completed'
                    mutate(Path(job['output_dir']),job)
                    try:validator.validate_job(job)
                    except (ValueError,AssertionError,FileNotFoundError) as exc:rejected.append({'case':label,'error':repr(exc)})
                    except Exception as exc:raise AssertionError('Negative '+label+' failed for incomplete-fixture/schema reasons: '+repr(exc)) from exc
                    else:unexpected.append(label)
        with tempfile.TemporaryDirectory(prefix='fdhvp_validator_paired_') as tmp:
            root=Path(tmp);control=build_fixture(root,'ug','natural','fd','fd','math')
            candidate=build_fixture(root/'candidate','ug','natural','fd','hvp','math');candidate['id']='cpu_hvp_candidate'
            log=root/'runtime/jobs'/candidate['id']/'console.log';log.parent.mkdir(parents=True)
            log.write_text((root/'candidate/runtime/jobs/cpu_fixture/console.log').read_text())
            validator.HERE=common.HERE=root
            paired=validator.validate_paired_inputs([control,candidate],required_backend='math')
            assert paired['status']=='PASS' and paired['steps_per_arm']==2
            cr=[json.loads(line) for line in (Path(control['output_dir'])/'step_trace.jsonl').read_text().splitlines()]
            hr=[json.loads(line) for line in (Path(candidate['output_dir'])/'step_trace.jsonl').read_text().splitlines()]
            # Same initial toy, same actual data, same native four probes. The
            # replacement must change what SGD consumes; an FD alias fails.
            assert cr[0]['gradient_flow']['actual_sgd_gradient_sha256']!=hr[0]['gradient_flow']['actual_sgd_gradient_sha256']
            paired['real_HVP_changed_actual_SGD_gradient_on_same_start']=True
            patch_trace(Path(candidate['output_dir']),lambda rs:rs[0]['identity'].__setitem__('input_sha256','0'*64))
            # Both jobs remain individually valid: only the paired comparison
            # can detect changed augmentation content with a valid digest.
            assert validator.validate_job(candidate)['summary']['status']=='completed'
            try:validator.validate_paired_inputs([control,candidate],required_backend='math')
            except ValueError as exc:rejected.append({'case':'paired_augmentation_mismatch','error':repr(exc)})
            else:unexpected.append('paired_augmentation_mismatch')
    finally:validator.HERE,common.HERE=previous
    after={name:sha(STAGE/name) for name in pins}
    assert pins==after,'Executable files changed during verification; rerun after freeze'
    return {'status':'PASS' if not unexpected else 'FAIL','fixture_scope':'Synthetic completion envelopes containing real two-task CPU native GAM and actual autograd HVP steps; not ViT results.',
        'positive_cases':positives,'positive_count':len(positives),'negative_cases_rejected':rejected,'negative_count':len(rejected),
        'unexpectedly_accepted':unexpected,'paired_input_audit':paired,'source_pins':pins,'source_pins_unchanged':True,'test_sha256':sha(__file__)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');args=p.parse_args();result=run_tests()
    if args.output:write(args.output,result)
    print(json.dumps(result,indent=2));assert result['status']=='PASS'
