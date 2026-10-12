from pathlib import Path
import json,hashlib,shutil,socket
import yaml
HERE=Path(__file__).resolve().parent
DEPLOY=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-reference-hyperparams-20260920')
ROOT=Path('/data/140-1/users/liying/CopyFlatCV')
ORIGIN=ROOT/'refine-logs/gam-ewc-paper-gapfill-20260918/attempt2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
manifest=json.loads((ORIGIN/'source_frozen.json').read_text())
for name,digest in manifest['files'].items():
 src=ORIGIN/'code'/name
 assert sha(src)==digest,name
 dst=HERE/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
save(HERE/'source_manifest.json',dict(origin=str(ORIGIN/'code'),original_manifest_sha256=sha(ORIGIN/'source_frozen.json'),files=manifest['files'],note='September18 frozen snapshot, used natively WITHOUT runtime_v3/protocol.install; June29 source identity unavailable. New center rerun required.'))
base=yaml.safe_load((ROOT/'config_save/s3_damp_c3_alpha02_int20_s2.yaml').read_text())
shutil.copy2(ROOT/'config_save/s3_damp_c3_alpha02_int20_s2.yaml',HERE/'reference_original.yaml')
# Freeze defaults explicitly; identical values across the native reference campaign.
base.update(fisher_mode='batch_empirical',ewc_preserve_fisher_rng=False,fisher_split=False,ewc_skip_task_fisher=False,train_num_workers=8,eval_num_workers=8,device=[0],data_root='/data/140-1/datasets')
points=[('center','center',None,{}),('lambda0','ewc_lambda',0,{'ewc_lambda':0}),('lambda200','ewc_lambda',200,{'ewc_lambda':200}),('lambda20000','ewc_lambda',20000,{'ewc_lambda':20000}),('strength0','lambda_flat',0,{'lambda_flat':0.0}),('strength05','lambda_flat',0.5,{'lambda_flat':0.5}),('radius01','gam_grad_norm_rho',0.1,{'gam_grad_norm_rho':0.1}),('radius05','gam_grad_norm_rho',0.5,{'gam_grad_norm_rho':0.5})]
jobs=[]; audits=[]
for name,axis,value,patch in points:
 jobid='s3_'+name+'_s2';cfg=dict(base);cfg.update(patch)
 cfg.update(prefix=jobid,outputs_root=str(DEPLOY/'outputs'/jobid),optimizer_tag_override='s3_reference_native_'+name)
 path=HERE/'configs'/f'{jobid}.json';save(path,cfg)
 jobs.append(dict(id=jobid,config=str(DEPLOY/'configs'/path.name),config_sha256=sha(path),axis=axis,value=value,seed=2,depends_on=[]))
 ignored={'prefix','outputs_root','optimizer_tag_override'}
 differences={k:[base.get(k),v]for k,v in cfg.items()if base.get(k)!=v and k not in ignored}
 assert differences=={k:[base[k],v]for k,v in patch.items()},differences
 audits.append(dict(id=jobid,scientific_differences=differences))
# Short integration check: two tasks and two epochs, interval=1 to exercise real weight merge then Fisher.
smoke=dict(base,prefix='s3_smoke_s2',outputs_root=str(DEPLOY/'outputs'/'s3_smoke_s2'),optimizer_tag_override='s3_reference_smoke',max_train_tasks=2,init_epoch=2,epochs=2,in_task_fisher_interval=1)
p=HERE/'configs/s3_smoke_s2.json';save(p,smoke)
smoke_job=dict(id='s3_smoke_s2',config=str(DEPLOY/'configs'/p.name),config_sha256=sha(p),axis='smoke',value=None,seed=2,depends_on=[],expected_tasks=2)
suite=dict(python='/home-local/liying/.conda/envs/Pilot_new_local/bin/python',execution_hostname=socket.gethostname(),allowed_gpus=[4,5,6],max_parallel=3,source_root=str(DEPLOY/'source'),source_manifest=str(DEPLOY/'source_manifest.json'),environment=dict(PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',FLATNESS_CV_DATA_ROOT='/data/140-1/datasets',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/home-local/liying/.cache/huggingface',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),expected_tasks=10,jobs=jobs)
save(HERE/'suite.json',suite)
smoke_suite=dict(suite,jobs=[smoke_job]);save(HERE/'smoke_suite.json',smoke_suite)
save(HERE/'CONFIG_AUDIT.json',dict(passed=True,reference=str(ROOT/'config_save/s3_damp_c3_alpha02_int20_s2.yaml'),source_hash=sha(HERE/'source_manifest.json'),explicit_defaults={k:v for k,v in base.items()if k not in yaml.safe_load((HERE/'reference_original.yaml').read_text())},jobs=audits,single_seed_exploratory=True,A_retained=True,protocol_install=False))
print('Prepared',len(jobs),'formal jobs + 1 separate smoke;',len(manifest['files']),'frozen source files')
