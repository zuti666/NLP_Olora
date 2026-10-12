"""Prepare immutable candidate configs/source only; never launch training."""
from pathlib import Path
import hashlib,json,copy,ast,shutil
HERE=Path(__file__).resolve().parent
FORMAL=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2')
DATASETS=['aircraft','cars196','cub200','flowers','het5']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
source=json.load(open(FORMAL/'source_frozen.json'));code=HERE/'formal_sep18_code'
for rel,h in source['files'].items():
 old=FORMAL/'code'/rel;assert sha(old)==h,rel
 new=code/rel;new.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,new);assert sha(new)==h
save(HERE/'source_manifest.json',dict(source_root=str(code),original_root=str(FORMAL/'code'),original_manifest_sha256=sha(FORMAL/'source_frozen.json'),files=source['files'],count=len(source['files'])))
# Copy immutable evidence, never mutate original run files.
for name in ['preflight.json','source_frozen.json','env_witness.json']:
 shutil.copy2(FORMAL/name,HERE/('original_'+name))
for name in ['run_job.py','launch_queue.py']:
 shutil.copy2(FORMAL/name,HERE/('original_'+name+'.reference'))
for d in DATASETS:shutil.copy2(FORMAL/'data_manifests'/f'{d}.json',HERE/f'data_manifest_{d}.json')
common_environment=json.load(open(FORMAL/'suite.json'))['environment'];python=json.load(open(FORMAL/'suite.json'))['python']
rows=json.load(open('/data/140-1/users/liying/O-LoRA/refine-logs/cllora_comparison_20260918/results.json'))
historical={(r['method'],r['dataset']):r for r in rows if r.get('method')=='GAM-EWCLoRA'}
jobs=[];reused=[]
for method in ['CL-LoRA','GAM-EWCLoRA']:
 for d in DATASETS:
  cl=method=='CL-LoRA'
  name=f'cllora_{d}' if cl else f'gam_ewc_{d}'
  old_name=f'cllora_{d}_s0' if cl else 'gam_ewc_het5_s0'
  reuse=cl or d=='het5'
  if reuse:
   oldcfg=FORMAL/'configs'/f'{old_name}.json';cfg=json.load(open(oldcfg));recpath=FORMAL/'receipts'/f'{old_name}.json';rec=json.load(open(recpath))
   assert rec['status']=='completed' and rec['exit_code']==0
   assert sha(rec['config'])==rec['config_sha256'] and sha(rec['metric_path'])==rec['metric_sha256']
   reused.append(dict(method=method,dataset=d,seed=0,config=str(oldcfg),config_sha256=sha(oldcfg),receipt=str(recpath),receipt_sha256=sha(recpath),source_root=str(code),metric_path=rec['metric_path'],metric_sha256=rec['metric_sha256'],status='conditional_existing_complete',conditions=['same frozen source and actual flags','recheck ordered data paths/labels','same cached pretrained bytes/environment']))
   desired=[1,2]
  else:
   # Recover logged training settings, leave provenance honest for absent defaults.
   r=historical[(method,d)]; cfg={}
   exclude={'config','filepath','prefix','device','seed','CNN','NME','feature_flat_save_path'}
   for k,v in r['params'].items():
    if k in exclude:continue
    if isinstance(v,str):
     try:v=ast.literal_eval(v)
     except (ValueError,SyntaxError):pass
    cfg[k]=v
   cfg.update(device=[0],data_root='/data/115-2/users/liying/datasets',train_num_workers=8,eval_num_workers=8,fisher_num_workers=8,cosine_min_lr_ratio=.1,save_task_checkpoints=False,save_final_model=False,ewc_preserve_fisher_rng=False,ewc_isolate_clean_rng=False,deterministic_loader_rng=False,fisher_mode='batch_empirical',ewc_normalize_accumulated_fisher=False,ewc_max_batches=100,ewc_skip_task_fisher=False,compute_full_fisher=False,approximate_fisher_in_task=False,in_task_weight_merge=False,ewc_project_grad=False,ewc_fisher_damping_alpha=0.,ewc_fisher_damping_beta=0.,gam_base_optimizer='sgd',gam_adaptive=False,gam_grad_beta_1=1.,gam_grad_beta_2=1.,gam_grad_beta_3=1.)
   cfg['optimizer_type']='gam';cfg['optimizer']='sgd';cfg['model_name']='ewclora_normfisher_gam'
   rho={'aircraft':(.01,.1,.02),'cars196':(.05,.2,.03),'cub200':(.2,.2,.1),'flowers':(.2,.2,.1)}[d]
   cfg.update(gam_grad_rho=rho[0],gam_grad_norm_rho=rho[1],gam_grad_gamma=rho[2],as1_rho=rho[0],as1_norm_rho=rho[1])
   desired=[0,1,2]
  cfg.pop('max_train_tasks',None)
  cfg['outputs_root']=str(HERE/'outputs');cfg['optimizer_tag_override']='seed012_20260922'
  for seed in desired:
   x=copy.deepcopy(cfg);id=f'{name}_s{seed}';x.update(seed=[seed],prefix='seed012_'+id,protocol_note=('Reuse Sep18 actual execution; only seed and output identity differ' if reuse else 'Candidate recovery: logged historical settings, explicit documented defaults; source is Sep18 snapshot, not certified historical source; seed0 parity/source review required before replication'))
   p=HERE/'configs'/f'{id}.json';save(p,x)
   jobs.append(dict(id=id,method=method,dataset=d,seed=seed,stage='formal',expected_tasks=int(x['nb_tasks']),config=str(p),config_sha256=sha(p),source_root=str(code),source_manifest=str(HERE/'source_manifest.json'),entrypoint='src.main.main(["--config", CONFIG, "--override", "device=[0]"])',depends_on=[f'smoke_{name}_s0']+([f'{name}_s0'] if not reuse and seed>0 else []),candidate_seed0=not reuse and seed==0,source_status='formal_source_exact' if reuse else 'historical_source_unbound_candidate',selection_pending='joost inventory and root protocol decision'))
  # One smoke per method/data. max_train_tasks=2 controls trainer without altering task mapping.
  smoke=copy.deepcopy(cfg);sid=f'smoke_{name}_s0';smoke.update(seed=[0],prefix='seed012_'+sid,outputs_root=str(HERE/'smoke_outputs'),init_epoch=1,epochs=1,max_train_tasks=2,protocol_note='Smoke only: two original tasks, one epoch per task; not a result replicate')
  p=HERE/'configs'/f'{sid}.json';save(p,smoke)
  jobs.append(dict(id=sid,method=method,dataset=d,seed=0,stage='smoke',expected_tasks=2,config=str(p),config_sha256=sha(p),source_root=str(code),source_manifest=str(HERE/'source_manifest.json'),entrypoint='src.main.main(["--config", CONFIG, "--override", "device=[0]"])',depends_on=[],source_status='formal_source_exact' if reuse else 'historical_source_unbound_candidate',selection_pending='not queued'))
manifest=dict(status='PREPARED_NOT_LAUNCHED',python=python,environment=common_environment,gpu_assignment='root scheduler only; user GPUs 2,3,4,5',jobs=jobs,reused_candidates=reused,job_counts={'formal':sum(j['stage']=='formal' for j in jobs),'smoke':sum(j['stage']=='smoke' for j in jobs)},notes=['No scheduler or GPU job launched.','First four GAM groups are candidates; existing-source inventory may replace/drop them.','Original fixed-source finite guard can be reused; never run original queue against old state.','CL has minLR0 and direct loader; GAM minLR .1LR.','Formal GAM-Het5 config clean RNG true is intentionally preserved but ignored by this source, not an actual enabled feature.','Aircraft GAM source DataManager uses persistent workers true from iAircraft; top-level false does not override.'])
save(HERE/'jobs_manifest.json',manifest)
print(json.dumps({'root':str(HERE),'source_files':len(source['files']),'formal':manifest['job_counts']['formal'],'smoke':manifest['job_counts']['smoke'],'reuse':len(reused)},indent=2))
