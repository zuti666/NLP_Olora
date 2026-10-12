"""Recover candidate source trees and materialize configs, without launching training."""
import copy
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
AUDIT=HERE.parent.parent
FLAT=Path('/data/140-1/users/liying/Flatness_CV')
COPY=Path('/data/140-1/users/liying/CopyFlatCV')
COMMIT='9c4439b397ec749d906f980cbda46632e06b08f3'
FROZEN=COPY/'refine-logs/gam-ewc-fine-gapfill-20260918-attempt2'
PYTHON='/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
DATA='/data/115-2/users/liying/datasets'
PACKAGES=['src','utils','backbone','models_LoRAbasedCL','models_CL','optimer_PerturabtionType',
          'evaluation','evaluation_performance','evaluation_weight_sharpness','evaluation_feature',
          'evaluation_attention','evaluation_ood','eval_flat','sharpness_evaluation_core']
baseline_root=HERE/'source_baselines_flatness_aug11'
ewc_root=HERE/'source_ewc_sep18'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

# Preserve exact git-archive source bytes, including auxiliary files in relevant packages.
archive=subprocess.check_output(['git','archive',COMMIT,*PACKAGES],cwd=FLAT)
baseline_root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
    for member in tar.getmembers():
        if not member.isfile():continue
        target=baseline_root/member.name
        assert target.resolve().is_relative_to(baseline_root.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(tar.extractfile(member).read())

# Existing formal source tree already includes evaluation dependency packages.
ewc_root.mkdir(parents=True,exist_ok=True)
source_frozen=json.loads((FROZEN/'source_frozen.json').read_text())
for p in sorted((FROZEN/'code').rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
    rel=p.relative_to(FROZEN/'code')
    if str(rel) in source_frozen['files']:
        assert sha(p)==source_frozen['files'][str(rel)],rel
    dst=ewc_root/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())

source_trees=[]
for name,root,origin in [('baseline',baseline_root,{'repository':str(FLAT),'git_commit':COMMIT}),
                         ('ewc',ewc_root,{'snapshot':str(FROZEN/'code'),'source_frozen_sha256':sha(FROZEN/'source_frozen.json')})]:
    entries={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    source_trees.append({'id':name,'root':str(root),'origin':origin,'files':entries,
                         'aggregate_sha256':hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()})
write_json(HERE/'source_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'trees':source_trees})

baseline_plan=json.loads((AUDIT/'seed012_plan_20260922/baseline_plan.json').read_text())
ewc_plan=json.loads((AUDIT/'seed012_plan_20260922/ewc_plan.json').read_text())
groups=[];jobs=[]
def materialize(method,dataset,model_name,source_config,config,source_id,extra_metadata):
    root=baseline_root if source_id=='baseline' else ewc_root
    group_id=f'{model_name}_{dataset}' if source_id=='baseline' else f'ewclora_{dataset}'
    original=copy.deepcopy(config)
    config['data_root']=DATA
    config['train_num_workers']=8
    config['min_lr']=float(config['lrate'])*.1
    config['device']=['0']
    if model_name=='inflora':config.update(lamb=.7,lame=.99)
    if model_name=='olora':config.update(olora_lamda_1=1.,olora_lamda_2=0.)
    if source_id=='ewc' and dataset in ('aircraft','cub200'):
        # Explicit resolution of the exact frozen candidate defaults; not a historical fact claim.
        config.update(fs_direction_mode='ratio',fs_preserve_probe_rng=False,ewc_preserve_fisher_rng=False)
    group_configs=[]
    for seed in [0,1,2]:
        cfg=copy.deepcopy(config)
        job_id=f'{group_id}_s{seed}'
        cfg.update(seed=[seed],prefix=f'seed012_{job_id}',outputs_root=str(HERE/'outputs'/'formal'/job_id))
        path=HERE/'configs'/'formal'/f'{job_id}.json';write_json(path,cfg)
        job={'id':job_id,'group_id':group_id,'method':method,'dataset':dataset,'seed':seed,
             'stage':'formal_candidate','source_id':source_id,'cwd':str(root),'config':str(path),'config_sha256':sha(path),
             'expected_tasks':int(config['nb_tasks']),'command':[PYTHON,'-m','src.main','--config',str(path)],
             'activation':'pending_root_and_joost_reuse_decision','depends_on':[f'smoke_{group_id}_s0']}
        jobs.append(job);group_configs.append(job_id)
    smoke=copy.deepcopy(config)
    smoke_id=f'smoke_{group_id}_s0'
    smoke.update(seed=[0],prefix=smoke_id,outputs_root=str(HERE/'outputs'/'smoke'/smoke_id),max_train_tasks=2,init_epoch=1,epochs=1)
    path=HERE/'configs'/'smoke'/f'{smoke_id}.json';write_json(path,smoke)
    jobs.append({'id':smoke_id,'group_id':group_id,'method':method,'dataset':dataset,'seed':0,'stage':'smoke_candidate',
                 'source_id':source_id,'cwd':str(root),'config':str(path),'config_sha256':sha(path),'expected_tasks':2,
                 'command':[PYTHON,'-m','src.main','--config',str(path)],'activation':'pending_root','depends_on':[]})
    changes={k:{'source':original.get(k,'<absent>'),'candidate':v} for k,v in config.items() if original.get(k,'<absent>')!=v}
    groups.append({'id':group_id,'method':method,'dataset':dataset,'source_id':source_id,'source_config':str(source_config),
                   'source_config_sha256':sha(source_config),'formal_candidate_ids':group_configs,'smoke_candidate_id':smoke_id,
                   'resolved_default_and_location_changes':changes,**extra_metadata})

for group in baseline_plan['groups']:
    source=Path(group['source_config']);rel=str(source.relative_to(FLAT))
    original_bytes=subprocess.check_output(['git','show',COMMIT+':'+rel],cwd=FLAT)
    assert hashlib.sha256(original_bytes).hexdigest()==group['source_config_sha256']
    cfg=yaml.safe_load(original_bytes)
    materialize(group['method'],group['dataset'],cfg['model_name'],source,cfg,'baseline',
                {'historical_existing_seeds':[0],'old_result_reuse':'not_decided_by_this_preparation',
                 'historical_metrics':group['source_metrics'],'task_widths':group['actual_task_widths'],
                 'candidate_provenance':'Recovered Aug11 Git content, not proven run-bound March source.'})
for group in ewc_plan['rows']:
    source=Path(group['source_config']);cfg=yaml.safe_load(source.read_text())
    materialize('EWCLoRA',group['dataset'],cfg['model_name'],source,cfg,'ewc',
                {'historical_existing_seeds':group['existing_eligible_seeds'],'old_result_reuse':'not_decided_by_this_preparation',
                 'ewc_lambda':cfg['ewc_lambda'],'candidate_provenance':'Sep18 formal source candidate; not proven run-bound Aug EWC source.',
                 'rng_semantics':'legacy per-forward CPU RNG consumption; historical sweep uses candidate ratio/false/false defaults, dualRNG groups preserve both, Het5 native_sgd.'})

environment=json.loads((FROZEN/'suite.json').read_text())['environment']
environment.update(FLATNESS_CV_DATA_ROOT=DATA,PYTHONDONTWRITEBYTECODE='1')
write_json(HERE/'environment.json',environment)
write_json(HERE/'candidate_jobs.json',{'status':'prepared_not_launched','authorized_gpu_pool':[2,3,4,5],
          'python':PYTHON,'environment':environment,'jobs':jobs})
write_json(HERE/'metadata.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'training_launched':False,
          'formal_candidate_count':sum(j['stage']=='formal_candidate' for j in jobs),
          'smoke_candidate_count':sum(j['stage']=='smoke_candidate' for j in jobs),'groups':groups,
          'provenance_boundary':'Candidate trees only. Historical reuse/new seed0 decision pending Joost/root; no queue edits.',
          'data_root':DATA,'smoke_rule':'JSON retains full nb_tasks/total_sessions, but baseline trainer overwrites effective args.nb_tasks with2 after task cap. Smoke validates functionality/finite outputs only. Formal configs have no cap. Only smoke max_train_tasks=2 and epochs/init_epoch=1; no train-subset or Fisher cap changes.',
          'checkpoint_boundary':'Offline cached timm ID expected; effective pretrained weight bytes must be verified by root preflight before activation.'})
print('Prepared',len(groups),'groups,',len(jobs),'candidate jobs; no training launched')
