#!/usr/bin/env python3
"""Read-only task0-entry, registered-batch0 gradient repeat controls."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

RUN = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
sys.path.insert(0, str(RUN/'trainer'))
from direction_step import FourProbeStep, normalized
from effective_step import EffectiveStep
from hvp_gate import model_sha, pure_config, seed_all, sha, tensor_sha, write_json
from mechanism_diagnostics import overlay_observation, trainable_sha
from response_hvp import State, compare, gradient_at, rng_get, rng_set, vector_norm

def stats(values, reference, names):
    result = compare(values, reference)
    result['bitwise_exact'] = all(torch.equal(a,b) for a,b in zip(values, reference))
    result['differing_names'] = [n for n,a,b in zip(names,values,reference) if not torch.equal(a,b)]
    result['differing_elements'] = sum(int(torch.count_nonzero(a!=b)) for a,b in zip(values,reference))
    blocks = {}
    for n,a,b in zip(names,values,reference):
        key = n.split('.blocks.')[1].split('.')[0] if '.blocks.' in n else 'fc'
        blocks.setdefault(key, [[],[]]); blocks[key][0].append(a); blocks[key][1].append(b)
    result['blocks'] = {k:compare(a,b) for k,(a,b) in blocks.items()}
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backends', nargs='+', choices=['native','math'], default=['native','math'])
    p.add_argument('--native-deterministic', action='store_true')
    args = p.parse_args(); out = Path(args.output_dir).resolve()
    if out.exists(): raise FileExistsError('Diagnostic output must be new: '+str(out))
    out.mkdir(parents=True); target = out/'repeat_report.json'; started = time.perf_counter()
    old_rng = rng_get(); old_det = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_tf = (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    old_cudnn = (torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    origin = live = model = None
    report = {'status':'preparing','optimizer_updates':0,'fisher_calls':0,'weight_merge_calls':0,
              'run':str(RUN),'code_sha256':sha(Path(__file__)),'backends':{},'state':{'task':0,'point':'entry','batch':0}}
    write_json(target,report)
    try:
        previous = json.loads((RUN/'outputs/Stage1_mechanism_diagnostics/stage1_mechanism_report.json').read_text())
        source = Path(previous['source']); reference = Path(previous['reference_output'])
        source_pins = {str(q.relative_to(source)):sha(q) for q in sorted(source.rglob('*.py'))}
        if source_pins != previous['source_python_sha256']: raise AssertionError('Frozen source pin mismatch')
        report.update(source=str(source),reference=str(reference),source_python_sha256=source_pins,
                      trainer_sha256={n:sha(RUN/'trainer'/n) for n in ('direction_step.py','response_hvp.py','mechanism_diagnostics.py','hvp_gate.py')},
                      environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','PYTHONHASHSEED')},
                      torch=torch.__version__,cuda=torch.version.cuda,initial_deterministic_algorithms=old_det)
        sys.path.insert(0,str(source)); os.environ.setdefault('HF_HUB_OFFLINE','1'); os.chdir(source)
        from utils.data_manager import DataManager
        from utils import factory
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        if device.type=='cuda': torch.cuda.set_device(device)
        raw = json.loads((reference/'effective_config.json').read_text()); cfg = pure_config(raw,device,out)
        seed_all(2); dm = DataManager(cfg['dataset'],cfg.get('class_shuffle',False),2,cfg['init_cls'],cfg['increment'],cfg)
        cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
        identity = json.loads((RUN/'outputs/Stage1_mechanism_diagnostics/batch_registry.json').read_text())['tasks']['0']['batches'][0]
        seed_all(identity['data_seed']); dataset = dm.get_dataset(np.arange(0,20),source='train',mode='train')
        loader = DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0,
                            generator=torch.Generator().manual_seed(identity['data_seed']),drop_last=False)
        ids,x,y = next(iter(loader))
        for key,value in [('input_sha256',x),('target_sha256',y),('indices_sha256',ids)]:
            if tensor_sha(value)!=identity[key]: raise AssertionError('Registered batch mismatch: '+key)
        if ids.tolist()!=identity['indices'] or int(y.numel())!=128: raise AssertionError('Registered ids/sample count mismatch')
        report.update(batch_identity=identity,input_shape=list(x.shape),batch_verified=True)
        seed_all(2); learner = factory.get_model(cfg['model_name'],cfg)
        learner._unwrap_network().backbone.to(device); ckpt = reference/'checkpoints/task_000'
        learner.load_task_checkpoint(str(ckpt),0,dm); model = learner._unwrap_network().to(device); model.train()
        manifest = json.loads((ckpt/'manifest.json').read_text()); checkpoint_hash = model_sha(model)
        if checkpoint_hash!=manifest['model_sha256']: raise AssertionError('Native checkpoint model hash mismatch')
        origin = State(model); obs = reference/'observations/task_000_entry.pt'
        if sha(obs)!=previous['state_results'][0]['observation_sha256']: raise AssertionError('Observation file hash mismatch')
        overlay = overlay_observation(model,torch.load(obs,map_location='cpu',weights_only=False),0,'entry')
        seed_all(identity['forward_seed']); live = State(model); entry_hash = model_sha(model)
        report.update(checkpoint_hash=checkpoint_hash,entry_full_model_hash=entry_hash,entry_overlay=overlay)
        named = [(n,q) for n,q in model.named_parameters() if q.requires_grad]; names,params = zip(*named)
        point0 = tuple(q.detach().clone() for q in params); geometry = EffectiveStep.from_model(model)
        x=x.to(device); y=y.to(device); input_sha=tensor_sha(x); target_sha=tensor_sha(y)
        def closure(): return F.cross_entropy(model(x)['logits'][:,:20],y)
        def loss_closure(a,b):
            logits = model(a)['logits']; return F.cross_entropy(logits[:,:20],b),logits
        def emit(event):
            write_json(target,report); print(json.dumps(event),flush=True)
        for backend in args.backends:
            live.restore(); torch.use_deterministic_algorithms(old_det,warn_only=old_warn)
            row = {'point_repeats':{},'engine_repeats':[]}; report['backends'][backend]=row
            _,g0,_,ginfo = gradient_at(model,params,point0,closure,live.rng,backend)
            direction = normalized(g0); points = {'g0':point0,
                'g1_plus':tuple((q+.05*d).detach() for q,d in zip(point0,direction)),
                'g1_minus':tuple((q-.05*d).detach() for q,d in zip(point0,direction))}
            for label,point in points.items():
                values = []; losses = []; records = []
                for index in range(3):
                    live.restore(); live.check(); loss,g,_,info = gradient_at(model,params,point,closure,live.rng,backend)
                    records.append({'repeat':index,'loss':loss,'comparison':stats(g,values[0] if values else g,names),'restoration':info['entry_state_restore']})
                    values.append(g); losses.append(loss)
                row['point_repeats'][label]={'point_norm':vector_norm(point),'repeats':records,'loss_span':max(losses)-min(losses)}
                emit({'event':'point_complete','backend':backend,'point':label,'exact':[r['comparison']['bitwise_exact'] for r in records]})
            first = None
            for index,branch in enumerate(('P','P','D')):
                live.restore(); live.check(); engine = FourProbeStep(model,params,geometry,{'branch':branch,'direction':'A','backend':backend,'rho':.05,'radius':.25,'eps':1e-12})
                try:
                    result = engine.compute(x,y,loss_closure,.04); ctx=result['context']
                    if first is None: first=ctx
                    comparisons = {label:stats(ctx['grads'][label],first['grads'][label],names) for label in ('g0','g1_plus','g1_minus','g2')}
                    comparisons.update({label:stats(ctx[label],first[label],names) for label in ('q','c21','U')})
                    row['engine_repeats'].append({'repeat':index,'branch':branch,'record':result['record'],'comparisons':comparisons})
                finally:
                    engine.discard(); live.restore(); live.check()
                emit({'event':'engine_complete','backend':backend,'branch':branch,'repeat':index})
            live.restore(); row['final_state_restoration']=live.check()
        if args.native_deterministic:
            live.restore(); torch.use_deterministic_algorithms(True,warn_only=False)
            try:
                fixed=[]; records=[]
                for index in range(3):
                    live.restore(); live.check()
                    _,g,_,info=gradient_at(model,params,point0,closure,live.rng,'native')
                    records.append({'repeat':index,'comparison':stats(g,fixed[0] if fixed else g,names),'info':info})
                    fixed.append(g)
                report['native_deterministic']={'status':'passed','gradient_norm':vector_norm(g),'repeats':records}
            except RuntimeError as error: report['native_deterministic']={'status':'operator_error','error':repr(error)}
            finally: live.restore(); live.check()
        live.restore(); report.update(post_entry_model_hash=model_sha(model),post_entry_trainable_hash=trainable_sha(model))
        if report['post_entry_model_hash']!=entry_hash: raise AssertionError('Entry full parameter/buffer hash changed')
        if tensor_sha(x)!=input_sha or tensor_sha(y)!=target_sha: raise AssertionError('Input/target mutation')
        report.update(status='passed',inputs_targets_unchanged=True,source_unchanged=source_pins=={str(q.relative_to(source)):sha(q) for q in sorted(source.rglob('*.py'))})
    except BaseException as error:
        report.update(status='failed',error=repr(error)); raise
    finally:
        if live is not None: live.restore(); report['entry_state_restoration']=live.check()
        if origin is not None:
            origin.restore(); report['native_origin_restoration']=origin.check(); report['post_native_origin_model_hash']=model_sha(model)
        torch.use_deterministic_algorithms(old_det,warn_only=old_warn)
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old_tf
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=old_cudnn
        rng_set(old_rng); report['seconds']=time.perf_counter()-started; write_json(target,report)
    print(json.dumps({'status':report['status'],'report':str(target)}),flush=True)

if __name__=='__main__': main()
