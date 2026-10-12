#!/usr/bin/env python3
"""Artifact acceptance: recorded native updates, paired start, switch and metrics."""
import argparse,json,math
from pathlib import Path
import audit_core as audit
from chain_step import active_method,READS,cosine_lr

def validate_output(config_path,out):
    raw=json.loads(Path(config_path).read_text());out=Path(out);s=json.loads((out/'final_summary.json').read_text());checks=[]
    def check(value,label):
        if not value:raise AssertionError(label)
        checks.append(label)
    check(s['status']=='completed','workercompleted');check(s['input_config_sha256']==audit.digest(config_path),'registeredconfigunchanged')
    seed=raw['seed'][0] if isinstance(raw['seed'],list) else raw['seed'];check(s['seed']==seed,'registeredseedexact');check(s['role']==raw['chain_role'],'registeredroleexact')
    if raw['chain_role']=='start':
        receipt=json.loads((out/'common_start_receipt.json').read_text());check(receipt['sha256']==audit.digest(out/'common_start.pt'),'expandedstartfilepinned');check(receipt['head_expansion_count']==1 and receipt['head_shape']==[40,768],'singleheadexpansion40');check(s['completed_steps']==0,'zerotrainingupdatesduringstart');check(all(s['checks'].values()),'startchecksalltrue')
    else:
        epochs=2 if raw['chain_smoke'] else 50;batches=2 if raw['chain_smoke'] else 8
        check(s['completed_epochs']==epochs and s['completed_steps']==epochs*batches,'fullregisteredtrainingbudget')
        check(s['common_start_sha256']==audit.digest(s['common_start_snapshot']),'commonexpandedstartpinned');check(all(s['checks'].values()),'branchchecksalltrue')
        records=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines() if line.strip()]
        check(len(records)==epochs*batches,'exactsteptracecount')
        expected_reads=0
        for i,r in enumerate(records):
            e=i//batches+1;method=active_method(raw['chain_branch'],e,raw['chain_switch_after_epoch']);reads=READS[method];expected_reads+=reads
            check(r['task']==1 and r['epoch']==e and r['batch']==i%batches,'taskepochbatchindexexact_'+str(i))
            check(r['ordinary_gradient_calls']==reads and r['training_ce_gradient_passes']==reads,'nativereadbudget_'+str(i));check(r['sgd_updates']==1,'oneSGDcommit_'+str(i))
            check(r['active_method']==method,'registeredmethod_'+str(i));check(math.isclose(r['nominal_lr'],cosine_lr(.02,e),rel_tol=1e-12),'continuous50epochLR_'+str(i))
            check(all(r[k]==0 for k in ('projection_calls','fisher_estimation_passes','weight_merge_updates','hvp_calls','create_graph_calls','final_norm_rescale_calls')),'forbiddenworkzero_'+str(i));check(not r.get('history_used') and not r.get('fallback_reason'),'nativecleanwithoutfallback_'+str(i));check(all(r['diagnostic_state_restoration'].values()),'passivediagnosticrestored_'+str(i))
        check(s['counters']['training_sgd_updates']==epochs*batches and s['counters']['training_ce_gradient_passes']==expected_reads,'countertotalsmatchtrace')
        check(s['step_trace_sha256']==audit.digest(out/'step_trace.jsonl'),'tracepinned');check(len(s['switch_receipts'])==1,'onecontinuityswitchreceipt')
        switch=s['switch_receipts'][0];check(switch['epoch']==raw['chain_switch_after_epoch']+1 and switch['method_after']=='sgd','correctswitchboundary');check(all(switch[k] for k in ('optimizer_same','scheduler_same','parameters_exact','parameter_references_exact','rng_exact')),'switchpreserveslivestate')
        curve=json.loads((out/'retention_curve.json').read_text())['epochs'];check(len(curve)==epochs+1 and curve[0]['epoch']==0,'postexpansionbaselineandallepochs');check(all(r['old_test_allseen40']['samples']==307 and r['current_test_allseen40']['samples']==254 for r in curve),'officialtestsizesunchanged')
        check(all(r['clean_training']['samples']==906 for r in curve[1:]),'cleanfitusesallcurrenttrainonly')
        factors=json.loads((out/'factor_state_curve.json').read_text())['epochs'];check(len(factors)==epochs and all(len(r['layers'])==24 for r in factors),'all24QVfactorsallepochs')
        decomps=json.loads((out/'factor_update_decomposition.json').read_text())['steps'];check(len(decomps)==epochs*2,'fixedfirstmiddleupdateobservations');check(all(len(r['layers'])==24 for r in decomps),'alllayersupdateobservations');check(all(x['delta_q_identity_state_scale_error']<1e-10 for r in decomps for x in r['layers']),'exactactualdeltaQidentity')
        snapshots=json.loads((out/'snapshots.json').read_text())['snapshots'];check(len(snapshots)==epochs+1,'epoch0andallepochcheckpoints')
        for r in snapshots:check(r['sha256']==audit.digest(r['path']),'snapshotimmutable_'+str(r['epoch']))
    return dict(status='accepted',checks=len(checks),check_labels=checks,summary_sha256=audit.digest(out/'final_summary.json'),registered_config_sha256=audit.digest(config_path),output_dir=str(out.resolve()),seed=seed,role=raw['chain_role'])

def validate_pair(left,right):
    outputs=[Path(left),Path(right)];summaries=[json.loads((p/'final_summary.json').read_text()) for p in outputs]
    a,b=summaries
    if a['seed']!=b['seed'] or a['common_start_sha256']!=b['common_start_sha256']:raise AssertionError('Paircommonstateor seedmismatch')
    ia,ib=a['initial_identity'],b['initial_identity']
    for key in ('task_initial_parameter_sha256','new_fc_rows_sha256','frozen_origin_sha256','common_start_sha256','common_start_rng_sha256'):
        if ia[key]!=ib[key]:raise AssertionError('Pairstartidentitymismatch:'+key)
    traces=[[json.loads(line) for line in (p/'step_trace.jsonl').read_text().splitlines() if line.strip()] for p in outputs]
    if len(traces[0])!=len(traces[1]):raise AssertionError('Pairtraininglengthmismatch')
    for i,(ra,rb) in enumerate(zip(*traces)):
        if ra['identity']!=rb['identity']:raise AssertionError('Pairinputoraugmentationmismatch:'+str(i))
    if traces[0][0]['g0_sha256']!=traces[1][0]['g0_sha256']:raise AssertionError('Pairstartbasegradientmismatch')
    return dict(status='accepted',seed=a['seed'],common_start_sha256=a['common_start_sha256'],all_augmented_minibatches_paired=len(traces[0]),first_g0_bitwise_equal=True,new_head_rows_bitwise_equal=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output-dir');p.add_argument('--pair-left');p.add_argument('--pair-right');a=p.parse_args()
    result=validate_pair(a.pair_left,a.pair_right) if a.pair_left else validate_output(a.config,a.output_dir)
    print(json.dumps(result,allow_nan=False))
