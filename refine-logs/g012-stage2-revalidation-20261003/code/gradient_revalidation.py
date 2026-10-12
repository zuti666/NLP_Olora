#!/usr/bin/env python3
"""Read-only exhaustive same-state validation at all 24 registered cases.

Uses the original frozen FourProbeStep without editing it. No optimizer is
constructed or step called. A task shard contains 3 observations x 2 batches;
only the union of passing shards for tasks 0,1,4,9 covers the full acceptance
budget. Math backend is the stage2 prerequisite. Native deterministic controls
must be launched with CUBLAS_WORKSPACE_CONFIG=:4096:8 before importing torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
RUN = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
sys.path.insert(0, str(RUN/'trainer'))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from direction_step import FourProbeStep
from effective_step import EffectiveStep
from hvp_gate import model_sha, pure_config, seed_all, sha, tensor_sha, write_json
from mechanism_diagnostics import (BRANCHES, POINTS, TASKS, _pinned_assets,
                                   _validate_config, overlay_observation, trainable_sha)
from response_hvp import State, compare, rng_get, rng_set


def vector_stats(values, reference, names):
    if len(values) != len(reference) or len(values) != len(names):
        raise ValueError('Vector tensor count mismatch')
    row = compare(values, reference)
    row['bitwise_exact'] = all(torch.equal(a,b) for a,b in zip(values,reference))
    row['differing_elements'] = sum(int(torch.count_nonzero(a!=b)) for a,b in zip(values,reference))
    row['tensor_metrics'] = {
        name: dict(compare((a,), (b,)), bitwise_exact=torch.equal(a,b),
                   shape=list(a.shape), dtype=str(a.dtype),
                   differing_elements=int(torch.count_nonzero(a!=b)))
        for name,a,b in zip(names,values,reference)}
    blocks = {}
    for name,a,b in zip(names,values,reference):
        key = 'block_'+name.split('.blocks.')[1].split('.')[0] if '.blocks.' in name else 'classifier_or_other'
        if key not in blocks: blocks[key] = ([],[])
        blocks[key][0].append(a); blocks[key][1].append(b)
    row['blocks'] = {key:dict(compare(a,b),bitwise_exact=all(torch.equal(x,y) for x,y in zip(a,b)))
                     for key,(a,b) in blocks.items()}
    return row


def rng_sha(state):
    digest = hashlib.sha256()
    digest.update(repr(state[0]).encode())
    digest.update(state[1][0].encode()); digest.update(state[1][1].tobytes())
    digest.update(repr(state[1][2:]).encode())
    digest.update(state[2].cpu().numpy().tobytes())
    for item in state[3]: digest.update(item.cpu().numpy().tobytes())
    return digest.hexdigest()


def audit_case(model, x, y, closure, geometry, backend, lr, case_path):
    live = State(model)
    names, params = zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    entry_model = model_sha(model)
    entry_trainable = trainable_sha(model)
    entry_rng = rng_sha(live.rng)
    entry_ids = tuple(id(p) for p in params)
    entry_inputs = tensor_sha(x)
    entry_targets = tensor_sha(y)
    report = {'status':'running','backend':backend,'optimizer_updates':0,'fisher_calls':0,
              'weight_merge_calls':0,'gradient_projection_calls':0,'hvp_calls':0,
              'create_graph':False,'entry_model_sha256':entry_model,
              'entry_trainable_sha256':entry_trainable,'entry_rng_sha256':entry_rng,
              'entry_inputs_sha256':entry_inputs,'entry_targets_sha256':entry_targets,
              'parameter_count':sum(p.numel() for p in params),'tensor_count':len(params),
              'engine_results':[],'criteria':{'bitwise_exact_required':True,
                 'repeat_branch':'P','shared_gradients':['g0','g1_plus','g1_minus'],
                 'shared_points':['phi0','phi1_plus','phi1_minus']}}
    write_json(case_path,report)
    reference = None
    all_checks = True
    started = time.perf_counter()
    try:
        for index,branch in enumerate(('P','P','D','C','R')):
            live.restore(); live.check()
            engine = FourProbeStep(model,params,geometry,
                       {'branch':branch,'direction':'A','backend':backend,
                        'rho':.05,'radius':.25,'eps':1e-12})
            try:
                result = engine.compute(x,y,closure,lr)
                ctx = result['context']; record = result['record']
                if any(record[k]!=0 for k in ('sgd_updates','fisher_calls','weight_merge_calls','gradient_projection_calls')):
                    raise AssertionError('Read-only probe submitted forbidden work')
                if record['ordinary_gradient_calls']!=4:
                    raise AssertionError('Unexpected probe count')
                if reference is None:
                    reference = ctx
                gradient_labels = ('g0','g1_plus','g1_minus','g2','g1') if branch=='P' else ('g0','g1_plus','g1_minus')
                point_labels = tuple(ctx['points']) if branch=='P' else ('phi0','phi1_plus','phi1_minus')
                metrics = {label:vector_stats(ctx['grads'][label],reference['grads'][label],names)
                           for label in gradient_labels}
                point_metrics = {label:vector_stats(ctx['points'][label],reference['points'][label],names)
                                 for label in point_labels}
                residual_metrics = {label:vector_stats(ctx[label],reference[label],names)
                                    for label in ('q','c10','c21','d12','U')} if branch=='P' else {}
                exact = all(row['bitwise_exact'] for group in (metrics,point_metrics,residual_metrics) for row in group.values())
                all_checks &= exact
                row = {'repeat_index':index,'branch':branch,'status':'passed' if exact else 'failed',
                       'record':record,'shared_or_repeated_gradients':metrics,
                       'shared_or_repeated_points':point_metrics,'repeat_residuals_and_direction':residual_metrics}
            finally:
                engine.discard(); live.restore()
                restore = live.check()
            row['restoration'] = restore
            report['engine_results'].append(row)
            write_json(case_path,report)
        report.update(status='passed' if all_checks else 'failed',all_comparisons_bitwise_exact=all_checks,
                      ordinary_gradient_calls=20,repeat_comparisons=1,cross_branch_comparisons=3)
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        live.restore()
        report['final_restoration'] = live.check()
        report.update(final_model_sha256=model_sha(model),final_trainable_sha256=trainable_sha(model),
                      final_rng_sha256=rng_sha(rng_get()),
                      parameter_identities_preserved=entry_ids==tuple(id(p) for p in model.parameters() if p.requires_grad),
                      seconds=time.perf_counter()-started)
        report['hashes_restored'] = (entry_model==report['final_model_sha256'] and
                                    entry_trainable==report['final_trainable_sha256'] and
                                    entry_rng==report['final_rng_sha256'])
        report['inputs_targets_unchanged'] = (tensor_sha(x)==entry_inputs and tensor_sha(y)==entry_targets)
        if not report['hashes_restored'] or not report['parameter_identities_preserved'] or not report['inputs_targets_unchanged']:
            report['status'] = 'failed'
        write_json(case_path,report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--task',type=int,choices=TASKS,help='One six-case shard; default all 24 cases')
    parser.add_argument('--backend',choices=('native','math'),default='math')
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    if args.backend=='native' and args.deterministic and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('Native deterministic launch requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before torch import')
    tasks = [args.task] if args.task is not None else list(TASKS)
    out = Path(args.output_dir).resolve()
    if out.exists(): raise FileExistsError('Refusing to overwrite diagnostic output: '+str(out))
    out.mkdir(parents=True)
    target = out/'gradient_revalidation_report.json'
    original_report_path = RUN/'outputs/Stage1_mechanism_diagnostics/stage1_mechanism_report.json'
    original_registry_path = RUN/'outputs/Stage1_mechanism_diagnostics/batch_registry.json'
    prior = json.loads(original_report_path.read_text())
    old_registry = json.loads(original_registry_path.read_text())
    source = Path(prior['source'])
    reference = Path(prior['reference_output'])
    raw_path = reference/'effective_config.json'
    raw = json.loads(raw_path.read_text()); _validate_config(raw)
    assets, manifests = _pinned_assets(reference,tasks)
    if sha(raw_path)!=prior['config_sha256'] or sha(original_registry_path)!=prior['batch_registry_sha256']:
        raise AssertionError('Configuration/registry differs from original registered report')
    for category,values in assets.items():
        for name,digest in values.items():
            if prior['reference_assets_sha256'][category].get(name)!=digest:
                raise AssertionError('Reference asset differs from original report: '+name)
    source_pins = {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}
    trainer_pins = {str(p.name):sha(p) for p in sorted((RUN/'trainer').glob('*.py'))}
    if source_pins!=prior['source_python_sha256']:
        raise AssertionError('Frozen source differs from original mechanism report')
    for name,digest in prior['diagnostic_code_sha256'].items():
        if trainer_pins.get(name)!=digest: raise AssertionError('Frozen trainer pin mismatch: '+name)
    old_rng = rng_get()
    old_det = (torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    old_flags = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    started = time.perf_counter()
    model = origin = None
    report = {'schema_version':1,'status':'preparing','backend':args.backend,'deterministic_algorithms':args.deterministic,
              'tasks':tasks,'points':list(POINTS),'batches_per_state':2,'batch_size':128,
              'expected_case_count':6*len(tasks),'complete_24_case_budget':tasks==list(TASKS),
              'seed':2,'rho':.05,'radius':.25,'run':str(RUN),'source':str(source),
              'reference_output':str(reference),'reference_training_backend':raw.get('attention_backend','native'),
              'original_report_sha256':sha(original_report_path),'original_registry_sha256':sha(original_registry_path),
              'config_sha256':sha(raw_path),'reference_assets_sha256':assets,
              'source_python_sha256':source_pins,'trainer_python_sha256':trainer_pins,
              'revalidation_code_sha256':sha(Path(__file__)),
              'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,
                  **{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED')}},
              'optimizer_updates':0,'fisher_calls':0,'weight_merge_calls':0,'hvp_calls':0,
              'cases':[],'shard_passed':False,'stage2_launch_allowed':False,
              'note':'Gradient prerequisite only. Stage2 additionally requires separate HVP acceptance.'}
    write_json(target,report)
    try:
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(args.deterministic,warn_only=False)
        if device.type=='cuda': torch.cuda.set_device(device)
        sys.path.insert(0,str(source)); os.environ.setdefault('HF_HUB_OFFLINE','1'); os.chdir(source)
        from utils.data_manager import DataManager
        from utils import factory
        cfg = pure_config(raw,device,out); cfg['attention_backend']=args.backend
        seed_all(2)
        dm = DataManager(cfg['dataset'],cfg.get('class_shuffle',False),2,cfg['init_cls'],cfg['increment'],cfg)
        cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
        if dm.nb_tasks!=10 or dm.nb_classes!=200 or list(dm._increments)!=[20]*10:
            raise AssertionError('Reference dataset differs from frozen ImageNet-A 10x20')
        batches = {}
        registry = {'schema_version':1,'registered_before_gradient_comparisons':True,
                    'original_registry_sha256':sha(original_registry_path),'tasks':{}}
        for task in tasks:
            start,end = dm.get_task_class_range(task)
            if (start,end)!=(20*task,20*(task+1)): raise AssertionError('Unexpected task range')
            data_seed = 281000+task
            seed_all(data_seed)
            dataset = dm.get_dataset(np.arange(start,end),source='train',mode='train')
            loader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0,
                     generator=torch.Generator().manual_seed(data_seed),drop_last=False)
            entries = []; batches[task]=[]
            for index,(ids,x,y) in enumerate(loader):
                if index>=2: break
                identity = {'batch':index,'samples':int(y.numel()),'indices':ids.tolist(),
                    'input_sha256':tensor_sha(x),'target_sha256':tensor_sha(y),'indices_sha256':tensor_sha(ids),
                    'data_seed':data_seed,'forward_seed':282000+task*10+index}
                old_identity = old_registry['tasks'][str(task)]['batches'][index]
                if identity!=old_identity: raise AssertionError('Original registered batch mismatch: '+str((task,index)))
                if int(y.numel())!=128 or bool(((y<start)|(y>=end)).any()):
                    raise AssertionError('Frozen batch size/current-task labels mismatch')
                archive = out/('batch_task%03d_%d.pt'%(task,index))
                torch.save({'indices':ids,'inputs':x,'targets':y,'identity':identity},archive)
                entries.append(dict(identity,archive_path=str(archive),archive_sha256=sha(archive)))
                batches[task].append((ids,x,y,identity))
            if len(entries)!=2: raise AssertionError('Missing registered batch')
            registry['tasks'][str(task)]={'class_range':[start,end],'dataset_samples':len(dataset),'batches':entries}
            del loader,dataset
        registry_path = out/'batch_registry.json'
        write_json(registry_path,registry)
        report.update(status='running',batch_registry_sha256=sha(registry_path),registered_batches_exact=True)
        write_json(target,report)
        all_passed = True
        for task in tasks:
            seed_all(2)
            learner = factory.get_model(cfg['model_name'],cfg)
            learner._unwrap_network().backbone.to(device)
            checkpoint = reference/'checkpoints'/('task_%03d'%task)
            learner.load_task_checkpoint(str(checkpoint),task,dm)
            model = learner._unwrap_network().to(device); model.train()
            if model_sha(model)!=manifests[task]['model_sha256']:
                raise AssertionError('Native checkpoint full model hash mismatch')
            origin = State(model)
            geometry = EffectiveStep.from_model(model)
            start,end = dm.get_task_class_range(task)
            for point in POINTS:
                origin.restore(); origin.check()
                obs = reference/'observations'/('task_%03d_%s.pt'%(task,point))
                payload = torch.load(obs,map_location='cpu',weights_only=False)
                _validate_config(payload['config'])
                overlay = overlay_observation(model,payload,task,point)
                if point=='end' and model_sha(model)!=manifests[task]['model_sha256']:
                    raise AssertionError('End observation does not match native end checkpoint')
                for ids,x,y,identity in batches[task]:
                    for value,key in ((ids,'indices_sha256'),(x,'input_sha256'),(y,'target_sha256')):
                        if tensor_sha(value)!=identity[key]: raise AssertionError('Registered input mutated')
                    batch_rng = rng_get()
                    seed_all(identity['forward_seed'])
                    case_path = out/('case_task%03d_%s_batch%d.json'%(task,point,identity['batch']))
                    def closure(inputs,targets):
                        output = model(inputs)
                        logits = output['logits'] if isinstance(output,dict) else output
                        if logits.ndim!=2 or logits.shape[1]!=end:
                            raise AssertionError('Current-task seen-classifier shape mismatch')
                        return F.cross_entropy(logits[:,start:end],targets-start),logits
                    try:
                        result = audit_case(model,x.to(device),y.to(device),closure,geometry,args.backend,
                                            .04 if task==0 else .02,case_path)
                    finally: rng_set(batch_rng)
                    all_passed &= result['status']=='passed'
                    report['cases'].append({'task':task,'point':point,'batch':identity['batch'],
                        'identity':identity,'observation_sha256':sha(obs),'overlay':overlay,
                        'case_report':str(case_path),'case_report_sha256':sha(case_path),
                        'status':result['status'],'seconds':result['seconds'],
                        'all_comparisons_bitwise_exact':result.get('all_comparisons_bitwise_exact',False),
                        'hashes_restored':result['hashes_restored']})
                    write_json(target,report)
                    print(json.dumps({'event':'case_complete','task':task,'point':point,
                                      'batch':identity['batch'],'status':result['status']}),flush=True)
            origin.restore(); origin.check()
            if model_sha(model)!=manifests[task]['model_sha256']:
                raise AssertionError('Native origin hash not restored')
            origin=None; model=None
            del learner,geometry
            if device.type=='cuda': torch.cuda.empty_cache()
        after_assets,_ = _pinned_assets(reference,tasks)
        if after_assets!=assets or sha(raw_path)!=report['config_sha256']:
            raise AssertionError('Reference assets changed')
        if source_pins!={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}:
            raise AssertionError('Frozen source changed')
        if trainer_pins!={str(p.name):sha(p) for p in sorted((RUN/'trainer').glob('*.py'))}:
            raise AssertionError('Frozen trainer changed')
        if sha(original_report_path)!=report['original_report_sha256'] or sha(original_registry_path)!=report['original_registry_sha256']:
            raise AssertionError('Original mechanism report/registry changed')
        if sha(Path(__file__))!=report['revalidation_code_sha256']:
            raise AssertionError('Revalidation implementation changed')
        for row in registry['tasks'].values():
            for entry in row['batches']:
                if sha(entry['archive_path'])!=entry['archive_sha256']:
                    raise AssertionError('Preregistered input archive changed')
        if len(report['cases'])!=report['expected_case_count']:
            raise AssertionError('Incomplete registered case budget')
        report.update(status='passed' if all_passed else 'failed',shard_passed=all_passed,
                      gradient_prerequisite_passed=all_passed and tasks==list(TASKS) and args.backend=='math',
                      source_unchanged=True,trainer_unchanged=True,reference_assets_unchanged=True,
                      native_origins_restored=True,case_count=len(report['cases']))
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        raise
    finally:
        if origin is not None:
            origin.restore(); report['exception_origin_restoration']=origin.check()
        torch.use_deterministic_algorithms(old_det[0],warn_only=old_det[1])
        (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)=old_flags
        rng_set(old_rng)
        report['seconds']=time.perf_counter()-started
        write_json(target,report)
    print(json.dumps({'status':report['status'],'report':str(target),
                      'stage2_launch_allowed':False}),flush=True)
    if report['status']!='passed': raise SystemExit(2)


if __name__=='__main__':
    main()
