#!/usr/bin/env python3
"""HVP-independent same-state SAM/S_F finite effective-step matching.

Uses frozen original native_candidates/matched_effective_diagnostic functions.
No Hessian gate is consulted and no optimizer/training updates are submitted.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
from checkpoint_eval import digest,write_json,configure_runtime


def vector_part_comparisons(native, matched, sam, lora_count, comparison, reference):
    return dict(all= dict(native=comparison(native,sam,reference),matched=comparison(matched,sam,reference)),
        lora_only=dict(native=comparison(native[:lora_count],sam[:lora_count]),
                       matched=comparison(matched[:lora_count],sam[:lora_count])),
        fc_only=dict(native=comparison(native[lora_count:],sam[lora_count:]),
                     matched=comparison(matched[lora_count:],sam[lora_count:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--seed',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--pilot',action='store_true')
    cli = parser.parse_args()
    root,output = Path(cli.campaign).resolve(),Path(cli.output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Fresh output directory required')
    output.mkdir(parents=True,exist_ok=True)
    configure_runtime()
    import torch
    sys.path.insert(0,str(root/'trainer'))
    sys.path.insert(0,str(root/'diagnostics'))
    from response_chain_train import load_diagnostic_context
    from state_transaction import State,backend_context
    from effective_step import solve_positive_scale
    from offline_core import DiagnosticEngine,matched_effective_diagnostic,finite_lora_parts,comparison
    epochs = (10,) if cli.pilot else (10,25)
    branches = ('S',) if cli.pilot else ('S','R')
    aggregate=dict(protocol='three_gaps_hvp_independent_effective_match_v1',seed=cli.seed,pilot=cli.pilot,
        status='running',submitted_training_updates=0,HVP_calls=0,
        mechanism_boundary='Matched instantaneous effective norm does not rule out long-term learning-rate effects',
        independent_repeat_unit='paired training seed, never checkpoints/layers/minibatches',results=[])
    write_json(output/'effective_match_results.json',aggregate)
    try:
        for branch in branches:
            for epoch in epochs:
                checkpoint=root/'outputs'/('formal_%s_s%d'%(branch,cli.seed))/'checkpoints'/('epoch_%03d.pt'%epoch)
                data=load_diagnostic_context(str(root),str(checkpoint),cli.seed)
                model,geometry,payload=data['model'],data['geometry'],data['payload']
                lr=float(payload['next_lr'])
                if any(float(g['lr'])!=lr for g in payload['optimizer_state']['param_groups']):
                    raise AssertionError('Native next-update optimizer LR mismatch')
                context=SimpleNamespace(model=model,geometry=geometry,loss_fn=lambda x,y:data['loss_fn'](model,x,y),
                    State=State,backend_context=backend_context)
                for batch_index,batch in enumerate(data['batches'][:1] if cli.pilot else data['batches']):
                    if len(batch['targets'])!=128:raise AssertionError('Native fixed128 batch required')
                    engine=DiagnosticEngine(context,microbatch=16)
                    try:
                        match=matched_effective_diagnostic(engine,batch['inputs'],batch['targets'],lr,solve_positive_scale)
                        if engine.hvp_calls!=0:raise AssertionError('No HVP permitted in this diagnostic')
                        if match['status']=='accepted':
                            # Repeat the deterministic native candidates, validate unchanged state,
                            # then decompose LoRA and full FC directions separately.
                            candidates=engine.native_candidates(batch['inputs'],batch['targets'],lr)
                            if candidates['status']!='accepted':raise AssertionError('Native candidate repeat changed status')
                            sam,ds=finite_lora_parts(geometry,candidates['g1'],lr)
                            raw,dr=finite_lora_parts(geometry,candidates['g2'],lr)
                            matched,dm=finite_lora_parts(geometry,candidates['g2'],match['scalar_solver']['scale'])
                            if sam['effective_norm']!=match['sam']['effective_norm'] or raw['effective_norm']!=match['g2_native']['effective_norm']:
                                raise AssertionError('Native repeat effective norms not exact')
                            match['direction_parts']=vector_part_comparisons(dr,dm,ds,len(geometry.pairs),comparison,sam['effective_norm'])
                            match['native_repeat_exact_effective_norms']=True
                    finally:
                        restoration=engine.close()
                    row=dict(seed=cli.seed,branch=branch,epoch=epoch,batch_index=batch_index,
                        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),batch_identity=batch['identity'],
                        epoch_lr_used=payload['epoch_lr_used'],probe_lr=lr,match=match,
                        submitted_training_updates=0,HVP_calls=engine.hvp_calls,restoration=restoration)
                    aggregate['results'].append(row)
                    write_json(output/('branch%s_epoch%03d_batch%d.json'%(branch,epoch,batch_index)),row)
                    write_json(output/'effective_match_results.json',aggregate)
                    print(json.dumps(dict(event='effective_match_complete',seed=cli.seed,branch=branch,epoch=epoch,batch=batch_index,status=match['status'])),flush=True)
                del model,data,context,geometry,payload
                torch.cuda.empty_cache()
        aggregate.update(status='completed',completed_combinations=len(aggregate['results']),
            accepted_matches=sum(r['match']['status']=='accepted' for r in aggregate['results']),
            unavailable_matches=sum(r['match']['status']!='accepted' for r in aggregate['results']))
        write_json(output/'effective_match_results.json',aggregate)
        write_json(output/'completion_receipt.json',dict(status='accepted',submitted_training_updates=0,HVP_calls=0,
            completed_combinations=len(aggregate['results']),accepted_matches=aggregate['accepted_matches'],
            state_rng_restored=True,results_sha256=digest(output/'effective_match_results.json'),script_sha256=digest(__file__)))
        return 0
    except Exception as error:
        aggregate.update(status='failed',exception_type=type(error).__name__,exception=str(error))
        write_json(output/'effective_match_results.json',aggregate)
        raise


if __name__=='__main__':raise SystemExit(main())
