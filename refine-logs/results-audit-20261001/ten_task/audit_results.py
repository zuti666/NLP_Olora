#!/usr/bin/env python3
"""Read-only audit of completed training and paired diagnostics; writes only HERE."""
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z')
TRAIN = CAMPAIGN / 'training'
DIAG = CAMPAIGN / 'diagnostics'
sys.path.insert(0, str(TRAIN))
import common
import summarize_g1g3 as train_report


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvwrite(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()} for row in rows)


def stats(values):
    values = [v for v in values if v is not None]
    return {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)} if values else None


def nonfinite(value, prefix=''):
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    if isinstance(value, dict):
        return [p for k,v in value.items() for p in nonfinite(v,prefix+'/'+str(k))]
    if isinstance(value, list):
        return [p for k,v in enumerate(value) for p in nonfinite(v,prefix+'/'+str(k))]
    return []


def label(config):
    b1,b3,g,r = [config[k] for k in ('gam_grad_beta_1','gam_grad_beta_3','gam_grad_gamma','gam_grad_norm_rho')]
    if g != 0: return 'original_gamma_%g' % g
    if b1 == b3 == .5: return 'mean_sum_r%g' % r
    if b1 == b3 == 1: return 'sum_r%g' % r
    if b1 == 1 and b3 == 0: return 'g1_r%g' % r
    if b1 == 0 and b3 == 1: return 'g3_r%g' % r
    raise ValueError('Unknown explicit arm metadata')


def main():
    suite = common.load_suite()
    source = common.verify_source(suite)
    ledger = {'status':'PASS','training_expected':9,'training_validated':0,
              'diagnostics_expected':9,'diagnostics_validated':0,'training':[], 'diagnostics':[],
              'source_manifest_sha256':source['source_manifest_sha256'],
              'source_files_verified':len(source['source_files']), 'missing':[], 'invalid':[],
              'training_seed_count':1,'training_seeds':[2],
              'historical_diagnostic_failure':'g1_task9 requested 4 train batches but only 3 existed; recovered without changing script or data'}
    rows=[]
    for job in suite['jobs']:
        run=TRAIN/'runtime/jobs'/job['id']
        receipt=read(run/'completed.json')
        assert receipt['status']=='completed'
        assert not (run/'failed.json').exists() and not (run/'queue_failed.json').exists()
        _,cfg,_=common.verify_config(job)
        assert common.sha256(run/'config.json')==job['config_sha256']==receipt['config_sha256']
        assert receipt['suite_sha256']==common.sha256(TRAIN/'suite.json')
        assert receipt['source_manifest_sha256']==source['source_manifest_sha256']
        for name,wanted in receipt['runtime_sha256'].items(): assert common.sha256(TRAIN/name)==wanted
        result=common.inspect_metrics(suite,job)
        assert result['metrics_sha256']==receipt['result']['metrics_sha256']
        row=train_report.decorate({'id':job['id'],'status':'completed','seed':2,**result},cfg,'new_training')
        row['config_path']=str(common.resolve(job['config']))
        row['config_sha256']=job['config_sha256']
        row['completed_receipt']=str(run/'completed.json')
        row['pure_gam_audit']=train_report.audit_pure_gam(run/'console.log')
        rows.append(row)
    rows += train_report.load_reused()
    refs={r['id']:r for r in read(TRAIN/'REUSED_RESULTS.json')['rows']}
    matrix_long=[];summary=[]
    keys=['aaa_task_macro','faa_task_macro','acquisition_task_macro','bwt','forgetting','aaa_sample_weighted','faa_sample_weighted']
    for row in rows:
        if row['id'] in refs: row['config_path']=refs[row['id']]['config']
        cfg=read(row['config_path']);row['arm']=label(cfg)
        a=row['cnn_matrix']; assert len(a)==10
        stage_means=[sum(a[t][:t+1])/(t+1) for t in range(10)]
        diagonal=[a[t][t] for t in range(10)]
        computed={'aaa_task_macro':sum(stage_means)/10,'faa_task_macro':sum(a[-1])/10,
                  'acquisition_task_macro':sum(diagonal)/10,
                  'bwt':sum(a[-1][j]-a[j][j] for j in range(9))/9,
                  'forgetting':sum(max(a[i][j] for i in range(j,10))-a[-1][j] for j in range(9))/9,
                  'aaa_sample_weighted':sum(row['cnn_top1_curve'])/10,'faa_sample_weighted':row['cnn_top1_curve'][-1]}
        assert all(math.isclose(row[k],v,rel_tol=0,abs_tol=1e-10) for k,v in computed.items())
        assert math.isclose(row['native_final']['FAA'],computed['faa_task_macro'],abs_tol=1e-10)
        row['stage_task_macro']=stage_means
        row['bwt_per_old_task']=[a[-1][j]-a[j][j] for j in range(9)]
        row['forgetting_per_old_task']=[max(a[i][j] for i in range(j,10))-a[-1][j] for j in range(9)]
        missing=nonfinite(read(row['metrics_path']))
        upper=[p for p in missing if re.fullmatch(r'/(cnn|nme)/matrices/[^/]+/\d+/\d+',p)
               and int(p.split('/')[-2]) < int(p.split('/')[-1])]
        unavailable=[p for p in missing if re.fullmatch(r'/(cnn|nme)/final/(AFG|FWT|BWT/0|FG/\d+|vectors/BWT_t/0|vectors/FG_t/\d+)',p)]
        assert len(upper)+len(unavailable)==len(missing), 'Unexpected nonfinite fields'
        log=(Path(row['completed_receipt']).parent/'console.log').read_text(errors='replace')
        epoch_records={(int(t),int(e)) for t,e in re.findall(r'Task\s+(\d+), Epoch\s+(\d+)/50',log)}
        assert epoch_records=={(t,e) for t in range(10) for e in range(1,51)}
        assert 'Traceback (most recent call last)' not in log
        assert not re.search(r'\b(?:Loss|CE|GAM) (?:nan|inf|-inf)\b',log,re.I)
        ledger['training'].append({'id':row['id'],'arm':row['arm'],'status':'validated','origin':row['origin'],
            'tasks':10,'matrix_observed_cells':55,'observed_cells_finite':True,
            'raw_nonfinite_count':len(missing),'raw_nonfinite_paths':missing,
            'structural_upper_triangle_nonfinite':len(upper),'unavailable_metadata_nonfinite':len(unavailable),
            'unique_epoch_records':len(epoch_records),'training_loss_nonfinite_or_traceback':False,
            'raw_nonfinite_interpretation':'Only schema-unobserved upper matrix and unavailable forward-transfer/first-task metadata; strict observed-cell validator passed.',
            'metrics_path':row['metrics_path'],'metrics_sha256':row['metrics_sha256'],
            'config_sha256':row['config_sha256'],'receipt':row['completed_receipt'],
            'receipt_sha256':common.sha256(row['completed_receipt']),'pure_gam_audit':row['pure_gam_audit']})
        summary.append({k:row[k] for k in ['arm','id','origin','seed','outer_radius','beta1','beta3','gamma']+keys})
        matrix_long += [{'arm':row['arm'],'stage':i,'evaluated_task':j,'accuracy':a[i][j],
                         'acquisition':diagonal[j],'final_accuracy':a[-1][j]} for i in range(10) for j in range(i+1)]
    ledger['training_validated']=len(rows)
    byarm={r['arm']:r for r in summary};contrasts=[]
    for lhs,rhs in [('mean_sum_r1','sum_r1'),('mean_sum_r1','g1_r1'),('mean_sum_r1','g3_r1'),
                    ('g3_r0','g1_r1')]+[('g3_r%g'%r,'g3_r1') for r in (0,.1,.25,.5)]:
        contrasts.append({'arm':lhs,'baseline':rhs,**{k:byarm[lhs][k]-byarm[rhs][k] for k in keys}})
    write('training_results.json',rows);csvwrite('training_summary.csv',summary)
    csvwrite('training_matrix_long.csv',matrix_long);csvwrite('training_contrasts_pp.csv',contrasts)
    spec=importlib.util.spec_from_file_location('diag_summary',DIAG/'summarize_diagnostics.py')
    ds=importlib.util.module_from_spec(spec);spec.loader.exec_module(ds)
    original=read(DIAG/'suite.json');recovery=read(DIAG/'suite_recovery_20260930T214003Z.json')
    effective=[j for j in original['jobs'] if j['id'] in ('g1_task0','g1_task4')]+recovery['jobs']
    assert len(effective)==len({j['id'] for j in effective})==9
    for manifest in (original,recovery):
        for path,wanted in manifest['file_pins'].items():assert common.sha256(path)==wanted,path
    assert read(Path(recovery['runtime'])/'state.json')['status']=='completed'
    raw_cf=[];raw_radius=[];raw_geo=[];anchor_rows=[];candidate_rows=[];sam_rows=[];scan_rows=[]
    for job in effective:
        runtime=original['runtime'] if job['id'] in ('g1_task0','g1_task4') else recovery['runtime']
        rp=Path(runtime)/'jobs'/job['id']/'completed.json';receipt=read(rp)
        assert receipt['returncode']==0 and receipt['validation_errors']==[]
        assert receipt['output']==job['output'] and receipt['batches']==job['max_batches']
        d=read(job['output']);ds.validate(d)
        assert d['training_seed']==2 and d['task']==job['task'] and d['arguments']['max_batches']==job['max_batches']
        assert d['source_config_sha256']==common.sha256(job['config'])
        assert d['arguments']['checkpoint_dir']==job['checkpoint_dir']
        for name,want in d['checkpoint_hashes'].items():assert common.sha256(Path(job['checkpoint_dir'])/name)==want
        for name,want in d['source_hashes'].items():assert common.sha256(Path(d['source'])/name)==want
        cfg=read(job['config']);arm=label(cfg)
        meta=ds.anchor_metadata(d,Path(job['output']));meta['source_arm']=arm
        cf,rs,gg=ds.tables_for(d,meta);raw_cf+=cf;raw_radius+=rs;raw_geo+=gg
        batches=d['batches'];n=len(batches)
        for b in batches:assert Path(b['vectors_file']).is_file()
        anchor={'anchor':job['id'],'source_arm':arm,'task':job['task'],'n_probe_batches':n,
                'probe_batch_sizes':[b['sample_count'] for b in batches],
                'reference_eta':d['arguments']['eta_ref'],'inner_rho':d['arguments']['rho'],
                'a':stats([b['geometry']['a'] for b in batches]),
                'R_perp':stats([b['geometry']['r_perp_over_g3'] for b in batches]),
                'cos_g1_g3':stats([b['geometry']['cosines']['g1_g3'] for b in batches]),
                'r0_relative_error_max':max(b['r0_identity_sanity']['relative_error'] for b in batches),
                'scale_only_step_relative_error_max':max(b['scale_only_sanity'].get('matched_parameter_step_relative_error',0) for b in batches),
                'reference_effective_step':stats([b['reference_update']['effective_joint_norm'] for b in batches]),
                'matched_candidates':sum(c['match']['ok'] for b in batches for c in b['counterfactuals']),
                'candidate_count':sum(len(b['counterfactuals']) for b in batches),
                'matching_error_max':max(c['match'].get('actual_relative_error',0) for b in batches for c in b['counterfactuals']),
                'eval_samples_per_task':{k:v['n'] for k,v in d['loss_before'].items()},
                'report':job['output'],'report_sha256':common.sha256(job['output'])}
        anchor_rows.append(anchor)
        for candidate in batches[0]['counterfactuals']:
            name=candidate['label'];items=[next(c for c in b['counterfactuals'] if c['label']==name) for b in batches]
            row={'anchor':job['id'],'source_arm':arm,'task':job['task'],'candidate':name,'n_probe_batches':n,
                 'n_matched':sum(c['match']['ok'] for c in items)}
            for region in ('current','old_task_macro'):
                for metric in ('all_seen_ce','task_local_ce'):
                    vals=[];vs=[]
                    for b,c in zip(batches,items):
                        if not c['match']['ok']:continue
                        baseline=next(z for z in b['counterfactuals'] if z['label']=='g1')
                        block=c['loss_change'].get(region)
                        if block is None:continue
                        value=block[metric];vals.append(value)
                        vs.append(value-baseline['loss_change'][region][metric])
                    key=region+'_'+metric
                    row[key+'_delta_mean']=statistics.mean(vals) if vals else None
                    row[key+'_delta_min']=min(vals) if vals else None
                    row[key+'_delta_max']=max(vals) if vals else None
                    row[key+'_vs_g1_mean']=statistics.mean(vs) if vs else None
                    row[key+'_vs_g1_negative_batches']=sum(v < -1e-7 for v in vs) if vs else None
            row['matched_scale_mean']=statistics.mean(c['match']['scale'] for c in items if c['match']['ok'])
            row['effective_step_error_max']=max(c['match'].get('actual_relative_error',0) for c in items)
            candidate_rows.append(row)
        for kind in ('parameter','kl'):
            ss=[b['radius_matched_sam'][kind] for b in batches]
            good=[s for s in ss if s['ok']]
            sam_rows.append({'anchor':job['id'],'source_arm':arm,'task':job['task'],'match_kind':kind,
                             'n_probe_batches':n,'successful':len(good),'rho':stats([s.get('sam_rho') for s in good]),
                             'rho_over_original_005':stats([s['sam_rho']/.05 for s in good]),
                             'cos_g3':stats([s.get('cos_g3') for s in good]),
                             'match_relative_error_max':max(s['match'].get('actual_relative_error',s['match'].get('relative_error',0)) for s in good),
                             'failure_reasons':[s for s in ss if not s['ok']]})
        for r in ('0.0','0.1','0.25','0.5','1.0'):
            ss=[b['radius_scan'][r] for b in batches]
            scan_rows.append({'anchor':job['id'],'source_arm':arm,'task':job['task'],'r':float(r),'n_probe_batches':n,
                              **{key:stats([s[key] for s in ss]) for key in
                                 ('g3_gradient_norm','cos_g1_g3','effective_update_cos_g1','g3_minus_g1_relative_norm')},
                              'g3_probe_parameter_norm':stats([s['points'][3]['parameter_norm'] for s in ss]),
                              'g3_probe_kl':stats([s['points'][3]['kl_all_seen'] for s in ss])})
        ledger['diagnostics'].append({'id':job['id'],'status':'validated','source_arm':arm,'task':job['task'],
            'batch_count':n,'batch_sizes':anchor['probe_batch_sizes'],'receipt':str(rp),'receipt_sha256':common.sha256(rp),
            'result_sha256':anchor['report_sha256'],'all_matches_successful':d['all_matches_successful'],
            'finite_pass':True,'restore_pass':True,'checkpoint_source_config_hashes_pass':True})
    ledger['diagnostics_validated']=len(anchor_rows)
    ledger['diagnostic_probe_batches']=sum(a['n_probe_batches'] for a in anchor_rows)
    ledger['matched_counterfactuals']=sum(a['matched_candidates'] for a in anchor_rows)
    ledger['counterfactuals_total']=sum(a['candidate_count'] for a in anchor_rows)
    ledger['training_validated_expected']='9/9 (5 new + 4 reused)'
    ledger['diagnostics_validated_expected']='9/9 (2 original + 7 recovered)'
    write('VALIDITY_LEDGER.json',ledger)
    write('diagnostic_anchor_summary.json',anchor_rows);csvwrite('diagnostic_anchor_summary.csv',anchor_rows)
    write('diagnostic_candidate_summary.json',candidate_rows);csvwrite('diagnostic_candidate_summary.csv',candidate_rows)
    write('diagnostic_sam_summary.json',sam_rows);csvwrite('diagnostic_sam_summary.csv',sam_rows)
    write('diagnostic_radius_summary.json',scan_rows);csvwrite('diagnostic_radius_summary.csv',scan_rows)
    csvwrite('diagnostic_counterfactual_batches.csv',raw_cf);csvwrite('diagnostic_radius_batches.csv',raw_radius);csvwrite('diagnostic_geometry_batches.csv',raw_geo)
    print(json.dumps({'status':'PASS','training':summary,'contrasts':contrasts,
                      'diagnostic_anchor_summaries':anchor_rows,'sam':sam_rows},indent=2))


if __name__=='__main__': main()
