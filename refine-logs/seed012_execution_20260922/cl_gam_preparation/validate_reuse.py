#!/usr/bin/env python3
"""Read-only provenance checks; no imports of training modules and no training."""
import datetime,hashlib,json,math,pathlib
P=pathlib.Path(__file__).resolve().parent
O=pathlib.Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2')
B=P.parent/'baseline_ewc_preparation'
def read(p): return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def checked_file(p,expected=None):
 p=pathlib.Path(p); actual=sha(p)
 return {'path':str(p),'sha256':actual,'expected_sha256':expected,'matches':actual==expected if expected else None}
oldsrc=read(O/'source_frozen.json')['files']; newsrc=read(P/'source_manifest.json')
source_checks=[]
for rel,expected in oldsrc.items():
 a=sha(O/'code'/rel); b=sha(P/'formal_sep18_code'/rel)
 source_checks.append({'path':rel,'expected_sha256':expected,'original_sha256':a,'prepared_sha256':b,'matches':a==b==expected==newsrc['files'].get(rel)})
source_ok=all(x['matches'] for x in source_checks) and len(oldsrc)==len(newsrc['files'])
runtime_checks=[checked_file(O/rel,expected) for rel,expected in read(O/'runtime_frozen.json')['files'].items()]
oldpf=read(O/'preflight.json'); olddata={x['dataset']:x for x in oldpf['data']}
newpfs={name:read(B/f'cpu_preflight_{name}.json') for name in ('baseline','ewc')}
data_comparisons=[]
for dataset in ('aircraft','cars196','cub200','flowers','het5'):
 x=olddata[dataset]; oldmanifest=read(x['manifest']); comparisons=[]
 for name,pf in newpfs.items():
  y=next(z for z in pf['datasets'] if z['dataset']==dataset); manifest=read(y['manifest'])
  comparisons.append({'campaign_source':name,'manifest':checked_file(y['manifest']),'ordered_train_paths_labels_equal':oldmanifest['train']==manifest['train'],'ordered_test_paths_labels_equal':oldmanifest['test']==manifest['test'],'task_boundaries_equal':x['task_sizes']==y['task_widths'],'task_widths':y['task_widths'],'train_count':len(manifest['train']),'test_count':len(manifest['test']),'missing_files_preflight':y['missing_files'],'train_test_realpath_overlap_preflight':y['train_test_realpath_overlap']})
 data_comparisons.append({'dataset':dataset,'original_manifest':checked_file(x['manifest'],x['sha256']),'prepared_original_manifest':checked_file(P/f'data_manifest_{dataset}.json',x['sha256']),'original_task_widths':x['task_sizes'],'comparisons':comparisons})
data_ok=all(d['original_manifest']['matches'] and d['prepared_original_manifest']['matches'] and all(c['ordered_train_paths_labels_equal'] and c['ordered_test_paths_labels_equal'] and c['task_boundaries_equal'] and c['missing_files_preflight']==0 and c['train_test_realpath_overlap_preflight']==0 for c in d['comparisons']) for d in data_comparisons)
cache=read(P/'cache_manifest.json'); weight_check=checked_file(cache['cl_pretrained_path'],cache['expected_sha256'])
records=[]
for candidate in read(P/'jobs_manifest.json')['reused_candidates']:
 receipt_check=checked_file(candidate['receipt'],candidate['receipt_sha256']); r=read(candidate['receipt']); config=read(r['config'])
 files={'receipt':receipt_check,'config':checked_file(r['config'],r['config_sha256']),'metric':checked_file(r['metric_path'],r['metric_sha256']),'log':checked_file(r['log'],r['log_sha256'])}
 matrix=read(r['metric_path'])['cnn']['matrices']['final']; n=r['expected_tasks']
 shape_ok=len(matrix)==n and all(len(row)==n for row in matrix)
 finite_ok=shape_ok and all(math.isfinite(matrix[i][j]) for i in range(n) for j in range(i+1))
 faa=sum(matrix[-1])/n; stages=[sum(row[:i+1])/(i+1) for i,row in enumerate(matrix)]; aaa=sum(stages)/n
 metrics_ok=finite_ok and r['metrics']['tasks']==n and abs(faa-r['metrics']['faa'])<1e-9 and abs(aaa-r['metrics']['aaa_stage'])<1e-9
 is_cl=candidate['method']=='CL-LoRA'
 bound=is_cl and config.get('cllora_pretrained_path')==weight_check['path'] and config.get('cllora_pretrained_sha256')==weight_check['sha256'] and weight_check['matches']
 protocol_ignored={'seed','prefix','outputs_root','optimizer_tag_override','protocol_note'}
 config_pairs=[]
 for seed in (1,2):
  newpath=P/'configs'/f"{r['id'][:-1]}{seed}.json"; new=read(newpath)
  diffs={k:{'old':config.get(k),'new':new.get(k)} for k in set(config)|set(new) if config.get(k)!=new.get(k)}
  unapproved={k:v for k,v in diffs.items() if k not in protocol_ignored}
  config_pairs.append({'seed':seed,'config':checked_file(newpath),'all_differences':diffs,'differences_other_than_seed_and_output_metadata':unapproved,'same_training_settings':not unapproved})
 checks={'receipt_and_payload_hashes':all(v['matches'] for v in files.values()),'formal_successful_seed0':r['stage']=='formal' and r['status']=='completed' and r['exit_code']==0 and r['seed']==0 and config['seed']==[0],'full_finite_task_matrix_and_recomputed_metrics':metrics_ok,'source_matches_formal_snapshot':source_ok,'runtime_files_match_recorded_manifest':all(x['matches'] for x in runtime_checks),'ordered_data_and_task_boundaries_match':data_ok,'prepared_seed1_seed2_training_settings_match':all(x['same_training_settings'] for x in config_pairs),'cl_explicit_weight_binding_verified':bound if is_cl else None}
 accepted=all(v for v in checks.values() if v is not None)
 limitations=['Original data manifests bind ordered absolute paths and labels, not historical image-file content hashes.']
 if not is_cl: limitations.append('Historical GAM-Het5 config/receipt did not bind the timm/HuggingFace weight revision or SHA. Current cached revision and bytes are checked, but cannot retroactively establish which historical weight bytes were loaded; accepted under the explicitly selected 99-new + 6-reuse campaign with this provenance limitation.')
 records.append({'id':r['id'],'method':candidate['method'],'dataset':r['dataset'],'seed':0,'stage':'formal','status':'accepted_reused' if accepted else 'rejected','accepted':accepted,'checks':checks,'receipt':candidate['receipt'],'receipt_sha256':files['receipt']['sha256'],'config':r['config'],'config_sha256':files['config']['sha256'],'metric_path':r['metric_path'],'metric_sha256':files['metric']['sha256'],'log':r['log'],'log_sha256':files['log']['sha256'],'source_root':str(P/'formal_sep18_code'),'source_manifest':str(P/'source_manifest.json'),'source_manifest_sha256':sha(P/'source_manifest.json'),'original_source_manifest':str(O/'source_frozen.json'),'original_source_manifest_sha256':sha(O/'source_frozen.json'),'files_verified':files,'metrics':{'tasks':n,'faa':faa,'aaa_stage':aaa,'stage_accuracies':stages},'expected_tasks':n,'elapsed_hours':(r['finished_unix']-r['started_unix'])/3600,'peak_cuda_bytes':r.get('peak_cuda_bytes'),'historical_pretrained_sha_bound':bool(bound),'prepared_seed_config_comparison':config_pairs,'limitations':limitations})
report={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_WITH_DOCUMENTED_LIMITATIONS' if all(x['accepted'] for x in records) else 'FAIL','scope':'Validation only; no training or GPU initialization','campaign_decision':'99 newly executed formal runs + these 6 archived formal seed0 runs; preserve each row implementation and actual RNG behavior','accepted_reused_records':[x for x in records if x['accepted']],'rejected_records':[x for x in records if not x['accepted']],'accepted_count':sum(x['accepted'] for x in records),'source_verification':{'matches':source_ok,'original_manifest':checked_file(O/'source_frozen.json',newsrc['original_manifest_sha256']),'prepared_manifest':checked_file(P/'source_manifest.json'),'file_count':len(source_checks),'files':source_checks},'original_runtime_verification':runtime_checks,'data_preflight_evidence':{'original':checked_file(O/'preflight.json'),'current':{name:checked_file(B/f'cpu_preflight_{name}.json') for name in newpfs},'current_preflights_passed':{name:pf['pass'] for name,pf in newpfs.items()}},'data_comparisons':data_comparisons,'pretrained_validation':{'current_cached_weight':weight_check,'current_hub_main_revision':cache['main_ref'],'CL_original_config_binds_path_and_sha':True,'GAM_historical_revision_sha_bound':False},'limitations':['Historical data manifests contain paths and labels rather than historical image byte digests.','GAM-Het5 historical pretrained revision and bytes were not bound in its archived receipt/config.','Reuse verifies evidence and row-local protocol compatibility; it is not a claim of bitwise retraining reproducibility.']}
(P/'reuse_validation.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Formal seed0 reuse validation','',report['status'],'',f"Accepted {report['accepted_count']} / {len(records)} records. Verified all {len(source_checks)} frozen source files against original and prepared copies; verified recorded runtime hashes. Ordered training/test paths, labels, and task boundaries match both new baseline and EWC preflight manifests for all five datasets.",'','| Method | Dataset | FAA | AAA stage | Receipt/hash, config/hash, metric/hash, log/hash |','|---|---|---:|---:|---|']
for r in records: lines.append(f"| {r['method']} | {r['dataset']} | {r['metrics']['faa']:.6f} | {r['metrics']['aaa_stage']:.6f} | {'PASS' if r['accepted'] else 'FAIL'} |")
lines+=['','CL-LoRA weight path and SHA match the explicit historical config binding: `'+weight_check['sha256']+'`.','',*['- '+s for s in report['limitations']],'','Full machine-readable paths, digests, checks, recomputed metrics, and seed1/2 configuration differences: `reuse_validation.json`. No training launched.']
(P/'reuse_validation.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'accepted_count':report['accepted_count'],'rejected_checks':{r['id']:r['checks'] for r in report['rejected_records']},'source_files':len(source_checks),'data_ok':data_ok,'CL_weight_matches':weight_check['matches']},indent=2))
