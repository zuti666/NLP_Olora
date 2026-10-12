#!/usr/bin/env python3
import argparse, importlib.util, json, os, sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
from r0_probe_audit import audited_probes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--original',required=True);p.add_argument('--source',required=True)
    p.add_argument('--base-config',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('original_audit',args.original)
    A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
    sys.path.insert(0,args.source);os.chdir(args.source)
    from utils.data_manager import DataManager
    from utils import factory
    from optimer_PerturabtionType.gam import GAM
    cfg=json.load(open(args.base_config));seed=cfg['seed'][0]
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    cfg.update(seed=seed,device=[device],filepath=args.checkpoint_dir+os.sep,train_num_workers=0,eval_num_workers=0,
       resume_require_experiment_identity=False,resume_require_model_sidecars=False,resume_require_fisher_state=False)
    A.seed_all(seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    dm=DataManager(cfg['dataset'],cfg.get('class_shuffle',cfg.get('shuffle',True)),seed,cfg['init_cls'],cfg['increment'],cfg)
    cfg.update(nb_classes=dm.nb_classes,task_increments=list(dm._increments),nb_tasks=dm.nb_tasks)
    learner=factory.get_model(cfg['model_name'],cfg)
    learner._unwrap_network().backbone.to(device);learner.load_task_checkpoint(args.checkpoint_dir,0,dm)
    model=learner._unwrap_network().to(device);geo=A.Geometry(model)
    train=A.cache_batches(dm,0,'train','train',2,128,20260930)
    ev=A.cache_batches(dm,0,'test','test',1,128,20360930)
    A.seed_all(20260930);model.train();anchor=A.Snapshot(model)
    metric_x=ev[0][0].to(device);ref=A.isolated_logits(model,metric_x)
    A.eval_losses(model,{0:ev},{0:dm.get_task_class_range(0)},device);anchor.restore()
    report=dict(status='running',arguments=vars(args),original_sha256=A.digest(args.original),
       source_gam_sha256=A.digest(Path(args.source)/'optimer_PerturabtionType/gam.py'),batches=[],
       stochastic_layers=[n for n,m in model.named_modules() if (isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p>0) or float(getattr(m,'drop_prob',0) or 0)>0])
    assert report['stochastic_layers']==[]
    A.atomic_json(args.output,report)
    def difference(x,y):
        return dict(relative_error=A.norm(x-y)/max(A.norm(y),1e-20),absolute_norm=A.norm(x-y),
                    max_abs=float((x-y).abs().max()),left_norm=A.norm(x),right_norm=A.norm(y),cosine=A.cosine(x,y),exact=bool(torch.equal(x,y)))
    try:
        for i,(xx,yy,idx) in enumerate(train):
            x,y=xx.to(device),yy.to(device)
            entry=dict(batch=i,inputs_sha256=A.tensor_digest(xx),targets_sha256=A.tensor_digest(yy),
                       sample_indices=idx.tolist(),native={},instrumented={})
            raw={};checked={}
            for r in (0.,1.):
                raw[r]=A.gam_probes(model,geo,anchor,x,y,0,20,.05,r,metric_x,ref,GAM)
                entry['native'][str(r)]=raw[r][1]
            entry['native_cross_measurements']={
                'r0_g3_vs_r1_g1':difference(raw[0.][0]['g3'],raw[1.][0]['g1']),
                'r0_g3_vs_r0_g1':difference(raw[0.][0]['g3'],raw[0.][0]['g1']),
                'r0_g1_vs_r1_g1':difference(raw[0.][0]['g1'],raw[1.][0]['g1']),
                'r0_g2_vs_g0':difference(raw[0.][0]['g2'],raw[0.][0]['g0']),
                'r0_point3_vs_point1':difference(raw[0.][2][3],raw[0.][2][1]),
                'r0_point1_vs_r1_point1':difference(raw[0.][2][1],raw[1.][2][1])}
            report['current_batch']=entry;A.atomic_json(args.output,report)
            for r in (0.,1.):
                checked[r]=audited_probes(A,model,geo,anchor,x,y,0,20,.05,r,metric_x,ref,GAM)
                entry['instrumented'][str(r)]=checked[r][1]
                entry['instrumented'][str(r)]['vs_uninstrumented']={k:difference(checked[r][0][k],raw[r][0][k]) for k in raw[r][0]}
                A.atomic_json(args.output,report)
            entry['restore']=anchor.check()
            report['batches'].append(entry);report.pop('current_batch',None)
            A.atomic_json(args.output,report)
            print(json.dumps({'batch':i,'cross':entry['native_cross_measurements'],'self_spec_passed':True}),flush=True)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        anchor.restore();report['final_restore']=anchor.check();A.atomic_json(args.output,report)

if __name__=='__main__':main()
