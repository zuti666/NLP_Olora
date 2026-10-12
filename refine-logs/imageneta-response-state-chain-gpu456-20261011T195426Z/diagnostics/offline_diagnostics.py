#!/usr/bin/env python3
"""Frozen-checkpoint Hessian response and exact effective-step diagnostics.

Examples: --run-root ROOT --checkpoint S/epoch_001.pt --seed 2
          --output-dir ROOT/diagnostics/results/pilot --smoke
Repeat --checkpoint for both branches' epochs 1/10/25. No training updates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.dont_write_bytecode=True


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--seed',type=int,required=True,choices=(0,1,2))
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--microbatch',type=int,default=16)
    cli=parser.parse_args()
    root=Path(cli.run_root).resolve();out=Path(cli.output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Offline output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'trainer'))
    import torch
    from response_chain_train import load_diagnostic_context
    from state_transaction import State,backend_context
    from effective_step import EffectiveStep,solve_positive_scale
    from offline_core import DiagnosticEngine,correction_diagnostic,matched_effective_diagnostic
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('Inherited strict deterministic CUBLAS workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.set_num_threads(max(1,int(os.environ.get('OMP_NUM_THREADS','2'))))
    checkpoint_paths=[Path(p).resolve() for p in cli.checkpoint]
    if cli.smoke:checkpoint_paths=checkpoint_paths[:1]
    aggregate=dict(protocol='response_state_chain_offline_v1',seed=cli.seed,
        smoke=cli.smoke,status='running',submitted_training_updates=0,results=[],
        independent_repeat_unit='paired training seed, never checkpoints or diagnostic minibatches',
        theory='exact loss Hessian H at common base; H may be indefinite; second HVP uses detached H.g0',
        startup_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    atomic_json(out/'diagnostic_results.json',aggregate)
    try:
        for checkpoint_index,checkpoint in enumerate(checkpoint_paths):
            data=load_diagnostic_context(str(root),str(checkpoint),cli.seed)
            if not isinstance(data,dict):raise TypeError('Trainer context must be a dictionary')
            model=data['model'];payload=data['payload'];lr=float(payload['next_lr'])
            actual_lrs=[float(g['lr']) for g in payload['optimizer_state']['param_groups']]
            if not actual_lrs or any(value!=lr for value in actual_lrs):
                raise AssertionError('Offline next-update LR differs from saved actual optimizer state')
            epoch=int(payload.get('epoch',payload.get('completed_epoch',0)))
            if epoch not in (1,10,25) and not cli.smoke:
                raise ValueError('Formal offline epochs fixed at 1/10/25')
            geometry=EffectiveStep.from_model(model)
            native_loss=lambda x,y: data['loss_fn'](model,x,y)
            context=SimpleNamespace(model=model,geometry=geometry,loss_fn=native_loss,
                                    State=State,backend_context=backend_context)
            batches=data['batches'][:1] if cli.smoke else data['batches']
            if not cli.smoke and len(batches)!=2:raise AssertionError('Exactly two fixed diagnostic batches required')
            for batch_index,batch in enumerate(batches):
                inputs,targets=batch['inputs'],batch['targets']
                if targets.numel()!=128:raise AssertionError('Diagnostic batch must contain 128 fixed train samples')
                if not bool(((targets>=20)&(targets<40)).all()):raise AssertionError('Offline gradients must use Task1 train labels')
                if next(model.parameters()).device.type!='cuda':raise RuntimeError('Full ViT offline diagnostics require scheduled CUDA device')
                base=dict(seed=cli.seed,epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                    checkpoint_index=checkpoint_index,batch_index=batch_index,batch_identity=batch.get('identity'),
                    epoch_lr_used=float(payload['epoch_lr_used']),probe_lr=lr,
                    probe_lr_rule='actual next-update LR from checkpoint optimizer state after scheduler',
                    train_input_source='fixed current-task TRAIN minibatch',
                    class_slice=[20,40],training_ce_reduction='mean',training_updates=0,
                    branch=payload.get('branch',payload.get('chain_branch',payload.get('method'))))
                print(json.dumps(dict(event='diagnostic_start',**base)),flush=True)
                engine=DiagnosticEngine(context,cli.microbatch)
                try:correction=correction_diagnostic(engine,inputs,targets,lr)
                finally:engine.close()
                row=dict(**base,correction=correction)
                accepted=correction['numerical_gate'].get('accepted',False)
                if accepted and epoch in (10,25):
                    match_engine=DiagnosticEngine(context,cli.microbatch)
                    try:row['matched_effective_step']=matched_effective_diagnostic(match_engine,inputs,targets,lr,solve_positive_scale)
                    finally:match_engine.close()
                aggregate['results'].append(row)
                atomic_json(out/('checkpoint%02d_batch%02d.json'%(checkpoint_index,batch_index)),row)
                atomic_json(out/'diagnostic_results.json',aggregate)
                print(json.dumps(dict(event='diagnostic_complete',seed=cli.seed,epoch=epoch,batch=batch_index,
                    numerical_gate=correction['numerical_gate']['status'],scales=correction['scales'])),flush=True)
                if correction['numerical_gate'].get('numerical_implementation_failed',True):
                    aggregate.update(status='numerical_gate_failed',failed_checkpoint=str(checkpoint),
                        failed_batch=batch_index,claim_boundary='No C^2g/H^2g correspondence claim; formal training can remain valid')
                    atomic_json(out/'diagnostic_results.json',aggregate)
                    atomic_json(out/'completion_receipt.json',dict(status='numerical_gate_failed',seed=cli.seed,
                        submitted_training_updates=0,completed_combinations=len(aggregate['results']),
                        final_results_sha256=digest(out/'diagnostic_results.json')))
                    return 2
            del context,geometry,model,data
            torch.cuda.empty_cache()
        any_valid=any(r['correction']['numerical_gate'].get('accepted',False) for r in aggregate['results'])
        aggregate.update(status='completed' if any_valid else 'inconclusive_all_degenerate',completed_combinations=len(aggregate['results']),
            valid_numerical_combinations=sum(r['correction']['numerical_gate'].get('accepted',False) for r in aggregate['results']),
            valid_correction_scalings=sum(s['status']=='accepted' and not s['degenerate']
                for r in aggregate['results'] for s in r['correction']['scales']),
            total_correction_scalings=sum(len(r['correction']['scales']) for r in aggregate['results']),
            completion_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        atomic_json(out/'diagnostic_results.json',aggregate)
        atomic_json(out/'completion_receipt.json',dict(status='accepted' if any_valid else 'inconclusive_all_degenerate',seed=cli.seed,
            submitted_training_updates=0,completed_combinations=len(aggregate['results']),
            numerical_gate_passed=any_valid,state_restoration_passed=True,
            final_results_sha256=digest(out/'diagnostic_results.json'),checkpoint_sha256={str(p):digest(p) for p in checkpoint_paths}))
        return 0 if any_valid else 3
    except Exception as error:
        aggregate.update(status='failed',exception_type=type(error).__name__,exception_message=str(error))
        atomic_json(out/'diagnostic_results.json',aggregate)
        raise


if __name__=='__main__':raise SystemExit(main())
