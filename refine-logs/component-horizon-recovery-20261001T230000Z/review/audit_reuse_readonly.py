"""Read original results; emit new reuse audit JSON only in this review directory."""
import json,hashlib,math,re,csv,datetime
from pathlib import Path
OUT=Path(__file__).parent
ROOT=Path('/data/140-1/users/liying/CopyFlatCV')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(a):
 assert len(a)==10 and all(len(r)==10 for r in a)
 assert all(isinstance(a[i][j],(int,float)) and math.isfinite(a[i][j]) for i in range(10) for j in range(i+1))
 return dict(AAA=sum(sum(a[t][:t+1])/(t+1) for t in range(10))/10,FAA=sum(a[-1])/10,LA=sum(a[t][t] for t in range(10))/10,BWT=sum(a[-1][j]-a[j][j] for j in range(9))/9,F=sum(max(a[t][j] for t in range(j,10))-a[-1][j] for j in range(9))/9)
FIELDS=['dataset','model_name','optimizer_type','optimizer','gam_base_optimizer','init_epoch','epochs','init_lr','lrate','scheduler','lora_rank','seed','batch_size','gam_grad_rho','gam_grad_norm_rho','gam_grad_beta_1','gam_grad_beta_2','gam_grad_beta_3','gam_grad_gamma','lambda_flat','ewc_lambda','ewc_skip_task_fisher','fisher_mode','ewc_normalize_fisher','ewc_normalize_accumulated_fisher','ewc_fisher_damping_alpha','approximate_fisher_in_task','in_task_fisher_omega','in_task_weight_merge','in_task_weight_merge_omega','task_endpoint_weight_merge','init_momentum','momentum','init_weight_decay','weight_decay','class_shuffle','init_cls','increment','train_num_workers','eval_num_workers']
current_manifest=read(ROOT/'refine-logs/s3-formation-run-20261001T180712Z/source_manifest.json')
rows=[];verified_sources={}
for old in read('/data/140-1/users/liying/O-LoRA/refine-logs/results-audit-20261001/ten_task/training_results.json'):
 p=Path(old['metrics_path']); cpath=Path(old['config_path']); rp=Path(old['completed_receipt']);r=read(rp);c=read(cpath);raw=read(p);a=raw['cnn']['matrices']['final'];res=metrics(a)
 assert r['status']=='completed' and r['id']==old['id'] and r['result']['tasks_completed']==10
 assert sha(p)==r['result']['metrics_sha256']==old['metrics_sha256']
 assert sha(cpath)==r['config_sha256']==old['config_sha256']
 started=read(rp.parent/'started.json');assert started['id']==r['id'] and started['config_sha256']==r['config_sha256']
 assert sha(rp.parent/'config.json')==r['config_sha256']
 assert not any((rp.parent/n).exists() for n in ['failed.json','queue_failed.json'])
 campaign=rp.parents[3];assert sha(campaign/'suite.json')==r['suite_sha256']
 assert all(sha(campaign/n)==v for n,v in r['runtime_sha256'].items())
 assert sha(r['source_manifest'])==r['source_manifest_sha256']
 skey=(r['source_root'],r['source_manifest_sha256'])
 if skey not in verified_sources:
  bad=[k for k,v in r['source_files'].items() if sha(Path(r['source_root'])/k)!=v];assert not bad,bad
  verified_sources[skey]=len(r['source_files'])
 same_native_pins=r['source_files']==current_manifest['files']
 console=rp.parent/'console.log';log=console.read_text(errors='replace');lines=log.splitlines()
 epochs={(int(t),int(e)) for t,e in re.findall(r'Task\s+(\d+), Epoch\s+(\d+)/50',log)};assert epochs=={(t,e) for t in range(10) for e in range(1,51)}
 assert 'Traceback (most recent call last)' not in log and not re.search(r'\b(?:Loss|CE|GAM) (?:nan|inf|-inf)\b',log,re.I)
 curves=re.findall(r'CNN top1 curve: \[([^\n]+)\]',log);curve=[float(v) for v in re.findall(r'np\.float64\(([0-9.eE+-]+)\)',curves[-1])];assert len(curve)==10 and curve==r['result']['cnn_top1_curve']
 res.update(sampleAAA=sum(curve)/10,sampleFAA=curve[-1])
 for x,y in [('AAA','aaa_task_macro'),('FAA','faa_task_macro'),('LA','acquisition_task_macro'),('BWT','bwt'),('F','forgetting')]:assert abs(res[x]-old[y])<1e-9
 assert abs(res['FAA']-raw['cnn']['final']['FAA'])<1e-9
 tasks=raw['mechanism']['final']['tasks'];assert len(tasks)==10
 for t in tasks:
  assert all(t[k]==0 for k in ['lambda_ewc','train_fisher_loss_mean','grad_fisher_norm_mean','ewc_penalty_value','fisher_past_trace_mass','in_task_weight_merge_snapshots','train_ewc_mean'])
  assert not t['in_task_weight_merge_applied'] and not t['weight_merge_applied']
 skips=[i+1 for i,s in enumerate(lines) if 'ewc_skip_task_fisher=true: skipped task-end' in s];assert len(skips)==10
 starts=[i+1 for i,s in enumerate(lines) if '[TaskStart]' in s and 'has_fisher=False' in s];assert len(starts)==10
 match='historical_same_LR_g3_r025' if old['arm']=='g3_r0.25' else 'historical_auxiliary_only'
 rows.append(dict(id=old['id'],arm=old['arm'],metrics=res,protocol={k:c.get(k) for k in FIELDS},scheduler_source_verified=dict(T_max=50,Task0_start=.04,Task0_floor=.004,Task1to9_start=.02,Task1to9_floor=.002),config_path=str(cpath),config_sha256=sha(cpath),metrics_path=str(p),metrics_sha256=sha(p),completed_receipt=str(rp),receipt_sha256=sha(rp),completed_utc=r['ended_at'],source_root=r['source_root'],source_manifest=r['source_manifest'],source_manifest_sha256=r['source_manifest_sha256'],source_files_verified=verified_sources[skey],native_source_files_equal_current_formation=same_native_pins,native_flags_before_main=r['native_flags_before_main'],console_log=str(console),console_sha256=sha(console),unique_epoch_records=len(epochs),observed_matrix_cells=55,finite_observed_cells=True,purity=dict(taskstart_no_fisher_lines=starts,task_fisher_skip_lines=skips,mechanism_all10_ewc_zero=True,mechanism_all10_fisher_empty=True,mechanism_all10_merge_false=True,mechanism_all10_snapshots_zero=True),reuse_class=match,strict_new_cohort_paired=False))
# Historical clean SGD runs located via the GAM-EWC all-results inventory; effective log values override nominal config templates.
history=read(OUT/'SEARCH_HISTORICAL_CANDIDATES.json')['noewc_sgd_or_fishershape_10task'];old_sgd=[]
for x in history:
 row=x['index'];c=x['log_protocol']
 if 'sgdplain_oldlearner' not in row['source']:continue
 p=Path(row['source']);m=read(p);a=m['cnn']['matrices']['final'];assert sha(p)==row['sha256']
 log=Path(row['log']);text=log.read_text(errors='replace');epochs={(int(t),int(e)) for t,e in re.findall(r'Task\s+(\d+), Epoch\s+(\d+)/'+c['epochs'],text)}
 assert {(t,int(c['epochs'])) for t in range(10)}.issubset(epochs)
 cp=Path(c['config']);cp=cp if cp.is_absolute() else ROOT/cp
 old_sgd.append(dict(id=c['prefix'],metrics=metrics(a),protocol={k:c.get(k) for k in FIELDS},metrics_path=str(p),metrics_sha256=sha(p),log_path=str(log),log_sha256=sha(log),nominal_config_path=str(cp),config_is_runtime_effective=False,completed_receipt=None,completion_evidence=f'10x10 matrix plus {len(epochs)} task/epoch records; no hash-bound completed receipt located',reuse=False,exclusion_reasons=['10/20 epochs instead of 50','seed0 instead of seed2','old learner ewclora_youyue_fitarchitechture','single SGD path; not new four-probe/post-unperturb component wrapper']))
# Superficially matching lambda-zero controls retained Fisher/merge or used a development split.
confounded=[]
for mode in ['diagonal','kronecker']:
 rid=f's3f_{mode}_sgd_s2_l0';cp=ROOT/f'refine-logs/s3-fisher-lambda-grid-20260921/configs/{rid}.json';rp=cp.parent.parent/f'runtime/jobs/{rid}/completed.json';r=read(rp);mp=Path(r['result']['metrics_path']);c=read(cp);raw=read(mp)
 assert r['status']=='completed' and sha(cp)==r['config_sha256'] and sha(mp)==r['result']['metrics_sha256']
 confounded.append(dict(id=rid,metrics=metrics(raw['cnn']['matrices']['final']),protocol={k:c.get(k) for k in FIELDS},config_path=str(cp),metrics_path=str(mp),metrics_sha256=sha(mp),completed_receipt=str(rp),receipt_sha256=sha(rp),reuse=False,exclusion_reasons=['task-end Fisher active','in-task weight merge active','different native source cohort']))
for mode in ['diagonal','kronecker']:
 rid=f'coarse__{mode}__sgd__seed42__lambda0';p=ROOT/f'refine-logs/imageneta-batched-fisher-lambda-plan-20260919/execution/runs/{rid}';m=read(p/'metrics.json');r=read(p/'receipt.json');c=read(p/'resolved_config.json')
 assert r['status']=='completed' and r['exit_code']==0 and sha(p/'metrics.json')==r['metrics_sha256']
 confounded.append(dict(id=rid,metrics=metrics(m['matrix']),protocol={k:c.get(k) for k in FIELDS},evaluation_source=m['evaluation_source'],evaluation_description=m['evaluation_description'],config_path=str(p/'resolved_config.json'),metrics_path=str(p/'metrics.json'),metrics_sha256=sha(p/'metrics.json'),completed_receipt=str(p/'receipt.json'),receipt_sha256=sha(p/'receipt.json'),reuse=False,exclusion_reasons=['seed42','train_holdout_10pct development evaluation; not official test','batched Fisher-history learner/passes']))
report=dict(audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Read-only originals; metadata/config/log/metrics scan with checkpoint payloads excluded',attachment='/home-local/liying/.codex/attachments/32fbd065-7a52-41f1-acf6-7f2cd4f9a51f/Pasted text.txt',metric_definitions=dict(AAA='mean over 10 stage means over seen tasks',FAA='mean final row over 10 tasks',LA='mean diagonal over 10 tasks',BWT='mean(final-diagonal) over first9tasks',F='mean(max stage>=task accuracy - final) over first9tasks',sample_metrics='native CNN top1 curve mean / last; separate weighting'),verified_nine_pure_gam_runs=rows,historical_clean_sgd_different_protocol=old_sgd,lambda_zero_controls_not_reusable=confounded,g0_matched_full10_found=False,g1_r025_full10_found=False,reuse_decision=dict(fresh_stage1_jobs=9,matched=['u0','us','ug','uo','ufull'],natural=['u0','ug','uo','ufull'],natural_us='Reuse fresh matchedUS only if c=eta exact and same four-probe/base-point path',old_g3_r025='Historical practical result only; not paired against new matched cohort',old_g1_r1='Historical auxiliary only; r difference may change FP32 perturb/unperturb roundtrip',matched_target='Virtual g1 step evaluated at each candidate own current state; NOT current formation independent-S trace target',r1_robustness='Conditional future plan; no configs auto-added',stage2='Conditional candidate selection after StageI; SS/XS/SX/XX paired seeds0/1/2',stage3='Conditional after clear StageII signal',launch_barrier='Do not modify old queue or launch; wait until current campaign all120 jobs (12smoke+24formation+84future) verified completed'))
(OUT/'REUSE_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'pure_validated':len(rows),'source_file_counts':list(verified_sources.values()),'native_pins_equal_current':all(r['native_source_files_equal_current_formation'] for r in rows),'old_clean_sgd':len(old_sgd),'confounded_controls':len(confounded),'path':str(OUT/'REUSE_AUDIT.json')},indent=2))
