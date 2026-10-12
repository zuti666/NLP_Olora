#!/usr/bin/env python3
"""Evaluate the paired Task1 continuation endpoints without training."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

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
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--metrics',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    metrics = resolve_metrics(args.metrics)
    training_manifest_path=checkpoint.parent.parent/'run_manifest.json'
    if not training_manifest_path.exists():
        raise ValueError('Expected the continuation run_manifest.json beside checkpoints; optimizer provenance is required')
    training_manifest=json.loads(training_manifest_path.read_text())
    train_config=training_manifest['config']
    assert training_manifest['mode']=='continuation' and training_manifest['anchor_method']==args.method
    assert train_config['optimizer_type']=='sgd' and float(train_config['lambda_flat'])==0
    assert float(train_config['ewc_lambda'])==2000
    assert train_config.get('optimizer_type_by_task') in (None,'','sgd')
    assert not train_config.get('approximate_fisher_in_task',False)
    origin_path = checkpoint_path(args.method,0,config)
    paths_sha = {'0':sha_file(origin_path),'1':sha_file(checkpoint)}
    base_output = Path(config.get('output_dir',ROOT/'results'))/args.method
    sanity = json.loads((base_output/'sanity/summary.json').read_text())
    sanity_manifest = json.loads((base_output/'sanity/manifest.json').read_text())
    assert sanity['status']=='complete' and sanity['passed'] is True
    assert sanity_manifest['common_code_sha256']==sha_file(ROOT/'rq2_common.py'), 'Fisher engine changed since sanity'
    assert sanity_manifest['checkpoint_sha256'][args.method]['0']==paths_sha['0']
    torch.set_num_threads(int(config.get('cpu_threads',4)))
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    print(json.dumps({'event':'table_a_loading','method':args.method,'checkpoint':str(checkpoint)}),flush=True)
    x,y,data=prepare_data(config,output)
    assert data['manifest_sha256']==sanity_manifest['data_manifest_sha256'], 'A and B/C use different data'
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
               'common_code_sha256':sha_file(ROOT/'rq2_common.py'),'sanity_protocol_sha256':sanity['protocol_sha256']}
    signature_sha=canonical_hash(signature)
    existing=output/'summary.json'
    if existing.exists():
        prior=json.loads(existing.read_text())
        if prior.get('status')=='complete':
            assert prior['signature_sha256']==signature_sha, 'Existing A summary belongs to different assets/protocol'
            print(json.dumps({'event':'already_complete','path':str(existing)}),flush=True)
            return
    origin=json.loads((base_output/'anchor_measurement.json').read_text())
    assert origin['protocol_sha256']==sanity['protocol_sha256']
    assert origin['data_manifest_sha256']==data['manifest_sha256']
    assert origin['checkpoint_sha256']['0']==paths_sha['0']
    assert origin['fisher']['normalized'] is False and origin['fisher']['floor'] is None
    # Origin q values refer to old Task9 directions, so reuse only trace and CE.
    anchor={'metrics':origin['metrics'],'fisher':{k:origin['fisher'][k] for k in
            ('raw_trace','subset_loss','samples','estimator','coordinate','normalized','floor','regularizer')},
            'provenance':{'reused':True,'path':str(base_output/'anchor_measurement.json'),
                          'sha256':sha_file(base_output/'anchor_measurement.json'),
                          'protocol_sha256':origin['protocol_sha256'],'reused_q':False}}
    model=load_network(checkpoint,Path(config.get('source_path',ROOT/'source')),device)
    with torch.no_grad():
        source_logits=model(x['test'][:8].to(device)).cpu()
    collector,params=convert_effective(model)
    with torch.no_grad():
        logits=model(x['test'][:8].to(device)).cpu()
    logit_error=float((source_logits-logits).abs().max())
    assert logit_error<3e-5,logit_error
    started=time.monotonic()
    endpoint={'metrics':{s:evaluate(model,x[s],y[s],device,int(config.get('eval_batch_size',64))) for s in ('train','test')},
              'fisher':fisher(model,collector,params,x['subset'],y['subset'],{},device,int(config.get('fisher_batch_size',8))),
              'provenance':{'reused':False,'checkpoint':str(checkpoint),'sha256':paths_sha['1']}}
    matrix_acc=cil_accuracy(metrics)
    # The resumed matrix inherits historical a11 (TF32 convolution). Production
    # uses full FP32; use the actual current anchor evaluation for a11 instead.
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
            'future_optimizer':'sgd','shaping_in_task1':False,'signature':signature,'signature_sha256':signature_sha,
            'checkpoints':{'0':str(origin_path),'1':str(checkpoint)},'metrics_json':str(metrics),
            'training_manifest':str(training_manifest_path),'training_rng_protocol':training_manifest.get('protocol'),
            'data_manifest_sha256':data['manifest_sha256'],'source_to_effective_max_logit_error':logit_error,
            'accuracy_CIL':acc,'a11_historical_inherited':matrix_acc['a11'],
            'a11_current_minus_historical_pp':current_a11-matrix_acc['a11'],
            'accuracy_CIL_inherited_matrix':matrix_acc,
            'current_anchor_training_metrics':{'path':str(old_metrics_path),'sha256':sha_file(old_metrics_path),'test':training_anchor},
            'production_precision':'full FP32; historical inherited a11 retained separately',
            'old_task_loss_definition':'CE on fixed Task0 head, 20 logits; deterministic eval transforms for train and test',
            'old_task_head_exactly_unchanged':True,'nonlora_backbone_exactly_unchanged':True,
            'effective_weight_drift':drift,'delta_sha256':delta_hash,'anchor':anchor,'endpoint':endpoint,
            'D0':{s:endpoint['metrics'][s]['loss']-anchor['metrics'][s]['loss'] for s in ('train','test')},
            'D0_subset':endpoint['fisher']['subset_loss']-anchor['fisher']['subset_loss'],
            'retention_scope':'Same future training rule; inherited EWC Fisher matrices may differ.',
            'seconds_new_endpoint':time.monotonic()-started}
    atomic_json(existing,result)
    print(json.dumps({'event':'table_a_complete','method':args.method,'D0':result['D0'],
                      'a11':acc['a11'],'a21':acc['a21'],'a22':acc['a22'],'drift':drift}),flush=True)


if __name__=='__main__':
    main()
