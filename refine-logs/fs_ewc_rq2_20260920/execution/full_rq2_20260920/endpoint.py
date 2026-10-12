#!/usr/bin/env python3
"""Evaluate the paired Task1 continuation endpoints without training."""
from __future__ import annotations

import argparse
import json
import math
import time
import sys
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from rq2_common import (ROOT, atomic_json, canonical_hash, checkpoint_path,
                        convert_effective, effective_weights, evaluate, fisher,
                        load_network, norm_sq, prepare_data, read_state, sha_file,
                        tensor_hash)


def resolve_metrics(path):
    path = Path(path).resolve()
    if path.is_file():
        return path
    matches = sorted(path.rglob('*_cl_metrics.json'))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one cl_metrics JSON under {path}: {matches}')
    return matches[0]


def cil_accuracy(path):
    data = json.loads(Path(path).read_text())
    matrices = data['cnn']['matrices']
    matrix = matrices.get('final')
    if matrix is None or len(matrix) != 2:
        matrix = matrices.get('t01')
    if matrix is None or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError('A requires the two-task CNN accuracy matrix (final 2x2 or t01).')
    result = {'a11':float(matrix[0][0]),'a21':float(matrix[1][0]),'a22':float(matrix[1][1]),
              'definition':'CNN class-incremental accuracy (%), classifier covers all classes seen at that stage'}
    if not all(math.isfinite(result[k]) for k in ('a11','a21','a22')):
        raise ValueError(f'Incomplete accuracy matrix: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--method',choices=('sgd','gam'),required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    run_dir = Path(args.run_dir).resolve()
    checkpoint = run_dir / 'checkpoints/task_01.pt'
    metrics = resolve_metrics(run_dir / 'trainer_outputs')
    training_manifest_path=run_dir/'run_manifest.json'
    if not training_manifest_path.exists():
        raise ValueError('Expected the continuation run_manifest.json beside checkpoints; optimizer provenance is required')
    training_manifest=json.loads(training_manifest_path.read_text())
    train_config=training_manifest['config']
    assert training_manifest['mode'] in ('continuation', 'sequence') and training_manifest['anchor_method']==args.method
    assert (run_dir/'COMPLETE.json').is_file(), 'Training run is not complete'
    assert train_config['optimizer_type']=='sgd' and float(train_config['lambda_flat'])==0
    assert float(train_config['ewc_lambda']) in (0.,2000.)
    assert train_config.get('optimizer_type_by_task') in (None,'','sgd')
    assert not train_config.get('approximate_fisher_in_task',False)
    origin_path = checkpoint_path(args.method,0,config)
    parent_config = training_manifest.get('requested_config', train_config)
    inherited = parent_config.get('resume_from_checkpoint_dir')
    if inherited is None:
        actual_origin = run_dir / 'checkpoints/task_00.pt'
    else:
        actual_origin = Path(inherited) / 'task_00.pt'
    assert sha_file(actual_origin) == sha_file(origin_path), 'Diagnostic anchor differs from actual training parent'
    assert int(config.get('training_seed',0)) == int(training_manifest['seed']), 'Diagnostic/training seed mismatch'
    paths_sha = {'0':sha_file(origin_path),'1':sha_file(checkpoint)}
    torch.set_num_threads(int(config.get('cpu_threads',4)))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    print(json.dumps({'event':'table_a_loading','method':args.method,'checkpoint':str(checkpoint)}),flush=True)
    x,y,data=prepare_data(config,output)
    s0,s1=read_state(origin_path),read_state(checkpoint)
    assert s0['fc.weight'].shape==(20,768) and s1['fc.weight'].shape==(40,768)
    assert torch.equal(s0['fc.weight'],s1['fc.weight'][:20])
    assert torch.equal(s0['fc.bias'],s1['fc.bias'][:20])
    for key,value in s0.items():
        if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
            assert torch.equal(value,s1[key]), f'Frozen backbone changed: {key}'
    delta=[b-a for a,b in zip(effective_weights(s0),effective_weights(s1))]
    drift=norm_sq(delta)**.5
    delta_hash=tensor_hash(delta)
    del s0,s1,delta
    signature={'method':args.method,'checkpoint_sha256':paths_sha,'metrics_sha256':sha_file(metrics),
               'training_manifest_sha256':sha_file(training_manifest_path),
               'data_manifest_sha256':data['manifest_sha256'],'code_sha256':sha_file(__file__),
               'common_code_sha256':sha_file(ROOT/'rq2_common.py'),
               'config_sha256':canonical_hash(config), 'training_complete_sha256':sha_file(run_dir/'COMPLETE.json')}
    signature_sha=canonical_hash(signature)
    existing=output/'summary.json'
    if existing.exists():
        prior=json.loads(existing.read_text())
        if prior.get('status')=='complete':
            assert prior['signature_sha256']==signature_sha, 'Existing A summary belongs to different assets/protocol'
            print(json.dumps({'event':'already_complete','path':str(existing)}),flush=True)
            return
    started=time.monotonic()
    conversion_errors={}
    def measure(path, label):
        model=load_network(path,Path(config.get('source_path',ROOT/'source')),device)
        with torch.no_grad():
            source_logits=model(x['test'][:8].to(device)).cpu()
        collector,params=convert_effective(model)
        with torch.no_grad():
            logits=model(x['test'][:8].to(device)).cpu()
        error=float((source_logits-logits).abs().max())
        assert math.isfinite(error) and error<3e-5, error
        conversion_errors[label]=error
        value={'metrics':{s:evaluate(model,x[s],y[s],device,int(config.get('eval_batch_size',64))) for s in ('train','test')},
               'fisher':fisher(model,collector,params,x['subset'],y['subset'],{},device,int(config.get('fisher_batch_size',8))),
               'provenance':{'reused':False,'checkpoint':str(path),'sha256':sha_file(path)}}
        assert value['fisher']['normalized'] is False and value['fisher']['floor'] is None
        assert math.isfinite(value['fisher']['raw_trace']) and value['fisher']['raw_trace'] > 0
        assert value['fisher']['samples'] == len(y['subset'])
        del collector,params,model
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
        return value
    anchor=measure(origin_path,'anchor')
    endpoint=measure(checkpoint,'task1_endpoint')
    logit_error=max(conversion_errors.values())
    matrix_acc=cil_accuracy(metrics)
    # A resumed matrix may inherit historical rounded a11. Use the actual FP32
    # anchor evaluation for a11, while retaining the training matrix separately.
    old_metrics_path=checkpoint.parent.parent/'task0_local_metrics.jsonl'
    old_rows=[json.loads(line) for line in old_metrics_path.read_text().splitlines() if line.strip()]
    anchor_rows=[r for r in old_rows if r['task']==0]
    assert anchor_rows, 'Continuation is missing its current-FP32 Task0 evaluation'
    training_anchor=anchor_rows[-1]['test']
    current_a11=anchor['metrics']['test']['accuracy']
    assert abs(training_anchor['local_accuracy']-current_a11)<1e-9
    assert abs(training_anchor['seen_accuracy']-current_a11)<1e-9
    assert abs(training_anchor['local_ce']-anchor['metrics']['test']['loss'])<3e-5
    acc={**matrix_acc,'a11':current_a11,
         'a11_source':'current full-FP32 anchor, matched to continuation task0_local_metrics.jsonl',
         'a21_a22_source':'current continuation CNN class-incremental matrix'}
    result={'status':'complete','method':args.method,'seed':int(config.get('training_seed',0)),
            'run_name':training_manifest.get('run_name',run_dir.name),
            'training_mode':training_manifest['mode'],'ewc_lambda':float(train_config['ewc_lambda']),
            'task0_shaping':args.method=='gam',
            'future_optimizer':'sgd','shaping_in_task1':False,'signature':signature,'signature_sha256':signature_sha,
            'checkpoints':{'0':str(origin_path),'1':str(checkpoint)},'metrics_json':str(metrics),
            'training_manifest':str(training_manifest_path),'training_rng_protocol':training_manifest.get('protocol'),
            'data_manifest_sha256':data['manifest_sha256'],'source_to_effective_max_logit_error':logit_error,
            'conversion_checks':conversion_errors,
            'sanity':{'fixed_old_head_equal':True,'frozen_backbone_equal':True,
                      'raw_empirical_fisher_finite_positive':True,'common_code_sha256':sha_file(ROOT/'rq2_common.py')},
            'accuracy_CIL':acc,'a11_historical_inherited':matrix_acc['a11'],
            'a11_current_minus_historical_pp':current_a11-matrix_acc['a11'],
            'accuracy_CIL_inherited_matrix':matrix_acc,
            'current_anchor_training_metrics':{'path':str(old_metrics_path),'sha256':sha_file(old_metrics_path),'test':training_anchor},
            'production_precision':'full FP32; inherited training-matrix a11 retained separately',
            'old_task_loss_definition':'CE on fixed Task0 head, 20 logits; deterministic eval transforms for train and test',
            'old_task_head_exactly_unchanged':True,'nonlora_backbone_exactly_unchanged':True,
            'effective_weight_drift':drift,'delta_sha256':delta_hash,'anchor':anchor,'endpoint':endpoint,
            'D0':{s:endpoint['metrics'][s]['loss']-anchor['metrics'][s]['loss'] for s in ('train','test')},
            'D0_subset':endpoint['fisher']['subset_loss']-anchor['fisher']['subset_loss'],
            'retention_scope':('Shared future SGD without protection; inherited Fisher is inactive.'
                               if float(train_config['ewc_lambda']) == 0. else
                               'Shared future SGD plus EWC; inherited normalized Fisher matrices may differ.'),
            'seconds_new_endpoint':time.monotonic()-started}
    atomic_json(existing,result)
    print(json.dumps({'event':'table_a_complete','method':args.method,'D0':result['D0'],
                      'a11':acc['a11'],'a21':acc['a21'],'a22':acc['a22'],'drift':drift}),flush=True)


if __name__=='__main__':
    main()
