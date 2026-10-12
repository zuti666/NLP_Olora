#!/usr/bin/env python3
"""Exploratory Task9/batch0 Hstart; full FP32 and weighted 64-sample FP64 chunks."""
import argparse, copy, json, os, re, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
    parser.add_argument('--out', '--output-dir', dest='out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--double', action='store_true')
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out/'probe.json'
    if target.exists(): raise FileExistsError(target)
    original = json.loads((run/'outputs/HVP_full_gate/hvp_gate_report.json').read_text())
    oldtask, oldbatch = original['tasks']['9'], original['tasks']['9']['batches'][0]
    source, reference = Path(original['source']), Path(original['reference_output'])
    checkpoint = reference/'checkpoints/task_009'
    sys.path.insert(0, str(run/'trainer'))
    from hvp_gate import sha, tensor_sha, model_sha, pure_config, seed_all, normalized, write_json
    from response_hvp import State, rng_get, gradient_at, compare, vector_norm
    def source_pins(): return {str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}
    def checkpoint_pins(): return {p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}
    assert source_pins() == original['source_python_sha256'], 'Frozen source mismatch'
    assert checkpoint_pins() == oldtask['checkpoint_sha256'], 'Checkpoint mismatch'
    assert sha(run/'trainer/response_hvp.py') == original['hvp_code_sha256']
    assert sha(run/'trainer/hvp_gate.py') == original['gate_code_sha256']
    sys.path.insert(0, str(source)); os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.chdir(source)
    from utils.data_manager import DataManager
    from utils import factory
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.set_device(device); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    raw = json.loads((reference/'effective_config.json').read_text())
    cfg = pure_config(raw, device, out)
    seed_all(cfg['seed'])
    dm = DataManager(cfg['dataset'],cfg.get('class_shuffle',False),cfg['seed'],cfg['init_cls'],cfg['increment'],cfg)
    cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
    learner = factory.get_model(cfg['model_name'],cfg)
    learner._unwrap_network().backbone.to(device)
    learner.load_task_checkpoint(str(checkpoint),9,dm)
    model = learner._unwrap_network().to(device); model.train()
    initial_hash = model_sha(model)
    assert initial_hash == oldtask['model_sha256'], 'Original model reconstruction mismatch'
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules()), 'Training BatchNorm prevents chunk equivalence'
    assert not any((isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p>0) or float(getattr(m,'drop_prob',0.) or 0.)>0 for m in model.modules())
    start,end = dm.get_task_class_range(9)
    dataset = dm.get_dataset(np.arange(start,end),source='train',mode='train')
    data_seed = cfg['seed']*100000+70000+9; seed_all(data_seed)
    loader = DataLoader(dataset,batch_size=oldbatch['samples'],shuffle=True,num_workers=0,
                        generator=torch.Generator().manual_seed(data_seed),drop_last=False)
    ids,x,y = next(iter(loader))
    assert data_seed == 270009 and ids.tolist() == oldbatch['indices'], 'Original batch indices mismatch'
    assert tensor_sha(x) == oldbatch['input_sha256'] and tensor_sha(y) == oldbatch['target_sha256'], 'Original batch hash mismatch'
    x,y = x.to(device),y.to(device)
    names,params = zip(*[(n,p) for n,p in model.named_parameters() if p.requires_grad])
    assert all(p.dtype == torch.float32 for p in params), 'Expected original FP32 coordinates'
    entry = State(model); R0 = rng_get(); point0 = tuple(p.detach().clone() for p in params)
    def closure_for(m,inputs,targets=None):
        targets = y if targets is None else targets
        def closure():
            logits=m(inputs); logits=logits['logits'] if isinstance(logits,dict) else logits
            assert logits.ndim==2 and logits.shape[1]==end
            return F.cross_entropy(logits[:,start:end],targets-start)
        return closure
    closure = closure_for(model,x)
    report = {'status':'running','exploratory':True,'task':9,'batch':0,'response':'hstart',
              'full_stage2_allowed':False,'criteria':{'relative_l2_max':.005,'cosine_min':.999,'two_adjacent_required':True},
              'model_sha256':initial_hash,'input_sha256':oldbatch['input_sha256'],'target_sha256':oldbatch['target_sha256'],
              'data_seed':data_seed,'indices':ids.tolist(),'class_range':[start,end],
              'source':str(source),'source_python_sha256':original['source_python_sha256'],
              'checkpoint':str(checkpoint),'checkpoint_sha256':oldtask['checkpoint_sha256'],
              'probe_script_sha256':sha(Path(__file__)),'original_gate':oldbatch['responses']['hstart'],'precisions':{},'optimizer_updates':0,
              'fp64_chunking':{'chunk_size':64,'total_samples':len(y),'loss':'sample weighted mean of chunk CE means',
              'mathematical_objective_preserved':True,'numerical_path_changed':True,
              'proof_limit':'FP64 also changes batch GEMM shapes and reduction order; it is not a bitwise fullbatch precision-only comparison.'}}
    write_json(target,report)
    def save_tensors(filename,values):
        torch.save({k:({n:v.detach().cpu() for n,v in zip(names,vs)} if isinstance(vs,(list,tuple)) else vs) for k,vs in values.items()},out/filename)
    groups = {}
    for i,name in enumerate(names):
        match = re.search(r'\.blocks\.(\d+)\.',name)
        key = 'block_%02d'%int(match.group(1)) if match else 'fc' if name.startswith('fc.') else 'other'
        groups.setdefault(key,[]).append(i)
    def blocks(a,b): return {k:compare(tuple(a[i] for i in idx),tuple(b[i] for i in idx)) for k,idx in groups.items()}
    def probe(label,m,ps,point,direction,fun,hs,inputs=None):
        def evaluate(at,need_hvp=False):
            if label != 'float64': return gradient_at(m,ps,at,fun,R0,'math',need_hvp,direction if need_hvp else None)
            gradients = [torch.zeros_like(p) for p in ps]
            hvps = [torch.zeros_like(p) for p in ps] if need_hvp else None
            loss_sum,details = 0.,[]
            for lo in range(0,len(y),64):
                hi = min(lo+64,len(y)); weight = (hi-lo)/len(y)
                chunk_closure = closure_for(m,inputs[lo:hi],y[lo:hi])
                loss,gg,hh,meta = gradient_at(m,ps,at,chunk_closure,R0,'math',need_hvp,direction if need_hvp else None)
                for result,value in zip(gradients,gg): result.add_(value,alpha=weight)
                if need_hvp:
                    for result,value in zip(hvps,hh): result.add_(value,alpha=weight)
                loss_sum += weight*loss
                meta.update(chunk_slice=[lo,hi],samples=hi-lo,weight=weight); details.append(meta)
            info = {'backend':'math','chunk_size':64,'total_samples':len(y),'chunks':details,'ce':loss_sum,
                    'ce_forward_count':len(details),'first_derivative_passes':len(details),
                    'second_derivative_passes':len(details) if need_hvp else 0,
                    'gradient_norm':vector_norm(gradients),'hvp_norm':vector_norm(hvps) if need_hvp else None,
                    'direction_norm':vector_norm(direction) if need_hvp else None,
                    'mathematical_loss':'mean over the same fixed 128 samples','fallback_used':False}
            return loss_sum,tuple(gradients),tuple(hvps) if need_hvp else None,info
        _,gref,_,refinfo = evaluate(point)
        _,g,hv,hinfo = evaluate(point,True)
        item = {'dtype':str(ps[0].dtype),'hvp':hinfo,'ordinary_reference':refinfo,
                'same_point_gradient':compare(g,gref),'scan':[]}
        report['precisions'][label] = item
        save_tensors(label+'_base.pt',{'point':point,'direction':direction,'ordinary_gradient':gref,'graph_gradient':g,'hvp':hv})
        if label == 'float64':
            item['hvp_vs_original_fp32'] = compare(hv,tuple(v.double() for v in original_hvp))
            item['hvp_vs_original_fp32_blocks'] = blocks(hv,tuple(v.double() for v in original_hvp))
        write_json(target,report)
        for h in hs:
            plus = tuple((p+h*d).detach() for p,d in zip(point,direction))
            minus = tuple((p-h*d).detach() for p,d in zip(point,direction))
            _,gp,_,gpinfo = evaluate(plus)
            _,gm,_,gminfo = evaluate(minus)
            fd = tuple((a-b)/(2*h) for a,b in zip(gp,gm))
            desired = tuple(h*d.double() for d in direction)
            ap = tuple(p.double()-v.double() for p,v in zip(plus,point))
            am = tuple(v.double()-p.double() for p,v in zip(minus,point))
            effective = tuple((p.double()-q.double())/(2*h) for p,q in zip(plus,minus))
            drift = tuple((p.double()+q.double())*.5-v.double() for p,q,v in zip(plus,minus,point))
            stats = compare(fd,hv)
            row = {'h':h,'central_vs_hvp':stats,'per_block':blocks(fd,hv),
                   'accepted':stats['relative_l2'] is not None and stats['relative_l2']<=.005 and stats['cosine'] is not None and stats['cosine']>=.999,
                   'actual_plus_delta_vs_h_d':compare(ap,desired),'actual_minus_delta_vs_h_d':compare(am,desired),
                   'effective_direction_vs_d':compare(effective,tuple(d.double() for d in direction)),
                   'midpoint_drift_norm':vector_norm(drift),'nominal_offset_norm':vector_norm(desired),
                   'gp_report':gpinfo,'gm_report':gminfo}
            row['plus_unchanged_fraction'] = sum(int((p==v).sum()) for p,v in zip(plus,point))/sum(v.numel() for v in point)
            filename = label+'_h'+format(h,'.0e')+'.pt'
            save_tensors(filename,{'plus':plus,'minus':minus,'gp':gp,'gm':gm,'fd':fd,'hvp':hv})
            row['tensor_file'] = filename; item['scan'].append(row); write_json(target,report)
            print(json.dumps({'precision':label,'h':h,'relative_l2':stats['relative_l2'],'cosine':stats['cosine'],'accepted':row['accepted']}),flush=True)
        item['two_adjacent_h_accepted'] = any(a['accepted'] and b['accepted'] for a,b in zip(item['scan'],item['scan'][1:]))
        write_json(target,report)
        return hv
    started = time.perf_counter()
    try:
        _,g0,_,_ = gradient_at(model,params,point0,closure,R0,'math')
        point = tuple((p+.05*u).detach() for p,u in zip(point0,normalized(g0)))
        _,g1,_,_ = gradient_at(model,params,point,closure,R0,'math')
        c10 = tuple(a-b for a,b in zip(g1,g0))
        phi2 = tuple((p+.25*u).detach() for p,u in zip(point0,normalized(c10)))
        direction = tuple((b-a).detach() for a,b in zip(point,phi2))
        report['d12_norm'] = vector_norm(direction)
        assert abs(report['d12_norm']-oldbatch['d12_norm']) <= 1e-12, 'Original displacement mismatch'
        original_hvp = probe('float32',model,params,point,direction,closure,[.001,.0003,.0001,.00003,.00001])
        report['original_hvp_norm_match'] = abs(vector_norm(original_hvp)-oldbatch['responses']['hstart']['hvp']['hvp_norm'])
        original_fd_error = oldbatch['responses']['hstart']['scan'][-1]['central_vs_hvp']['relative_l2']
        report['original_fp32_fd_error_match'] = abs(report['precisions']['float32']['scan'][0]['central_vs_hvp']['relative_l2']-original_fd_error)
        assert report['original_hvp_norm_match'] < 1e-8 and report['original_fp32_fd_error_match'] < 1e-8, 'Fullbatch FP32 reproduction mismatch'
        if args.double:
            if device.type == 'cuda': torch.cuda.empty_cache()
            double_model = copy.deepcopy(model).double()
            double_params = tuple(p for p in double_model.parameters() if p.requires_grad)
            double_inputs = x.double()
            probe('float64',double_model,double_params,tuple(p.double() for p in point),tuple(d.double() for d in direction),closure_for(double_model,double_inputs),[.003,.001,.0003,.0001,.00003,.00001],double_inputs)
            del double_model
        report['status'] = 'completed_exploratory'
    except Exception as exc:
        report.update(status='failed_exploratory',error=repr(exc)); raise
    finally:
        entry.restore(); report['entry_state_restore'] = entry.check()
        report['model_unchanged'] = model_sha(model)==initial_hash
        report['source_unchanged'] = source_pins()==original['source_python_sha256']
        report['checkpoint_unchanged'] = checkpoint_pins()==oldtask['checkpoint_sha256']
        report['seconds'] = time.perf_counter()-started; write_json(target,report)
        assert report['model_unchanged'] and report['source_unchanged'] and report['checkpoint_unchanged']
    print(json.dumps({'status':report['status'],'report':str(target),'full_stage2_allowed':False}),flush=True)

if __name__ == '__main__': main()
