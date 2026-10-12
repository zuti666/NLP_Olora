"""Assemble the authorized cohort without editing either source repository."""
import json
from pathlib import Path
from collections import Counter,defaultdict
from run_job import sha,save
ROOT=Path(__file__).resolve().parent
B=ROOT/'baseline_ewc_preparation';C=ROOT/'cl_gam_preparation'
assert not (ROOT/'state.json').exists(), 'Never rebuild a launched suite'
b=json.loads((B/'candidate_jobs.json').read_text());c=json.loads((C/'jobs_manifest.json').read_text())
(ROOT/'source_manifests').mkdir(exist_ok=True)
source_manifests={}
for tree in json.loads((B/'source_manifest.json').read_text())['trees']:
    p=ROOT/'source_manifests'/f"{tree['id']}.json";save(p,tree);source_manifests[tree['id']]=str(p)
jobs=[]
for old in b['jobs']+c['jobs']:
    j=dict(old);j['stage']=j['stage'].replace('_candidate','');j.pop('selection_pending',None);j.pop('activation',None)
    if 'cwd'in j:j['source_root']=j['cwd'];j['source_manifest']=source_manifests[j['source_id']]
    j['source_manifest_sha256']=sha(j['source_manifest'])
    if 'group_id'not in j:j['group_id']=j['id'].removeprefix('smoke_').rsplit('_s',1)[0]
    manifest=B/'data_manifests'/f"{'baseline' if j.get('source_id')=='baseline' else 'ewc'}_{j['dataset']}.json"
    j['data_manifest']=str(manifest);j['data_manifest_sha256']=sha(manifest)
    jobs.append(j)
smoke_order=['smoke_seqlora_flowers_s0','smoke_inflora_flowers_s0','smoke_olora_flowers_s0','smoke_sdlora_flowers_s0',
 'smoke_cllora_flowers_s0','smoke_ewclora_flowers_s0','smoke_gam_ewc_flowers_s0','smoke_ewclora_het5_s0','smoke_gam_ewc_het5_s0']
smoke_by_method={j['method']:j['id']for j in jobs if j['id']in smoke_order and j['dataset']=='flowers'}
formal=[j for j in jobs if j['stage']=='formal']
for j in formal:
    smoke=smoke_by_method[j['method']]
    if j['dataset']=='het5'and j['method']in ['EWCLoRA','GAM-EWCLoRA']:
        smoke='smoke_'+j['group_id']+'_s0'
    j['depends_on']=[smoke]
    zero=next((x for x in formal if x['group_id']==j['group_id']and x['seed']==0),None)
    if j['seed']!=0 and zero:j['depends_on'].append(zero['id'])
    j['activation']='selected_20260922_fixed_protocol'
dataset_order=['flowers','het5','aircraft','cars196','cub200']
method_order=['Vanilla LoRA','InfLoRA','OLoRA','SDLoRA','CL-LoRA','EWCLoRA','GAM-EWCLoRA']
formal.sort(key=lambda j:(j['seed']>0 and j['method']!='CL-LoRA'and not(j['method']=='GAM-EWCLoRA'and j['dataset']=='het5'),dataset_order.index(j['dataset']),j['seed'],method_order.index(j['method'])))
selected=[next(j for j in jobs if j['id']==k)for k in smoke_order]+formal
reuse=json.loads((C/'reuse_validation.json').read_text());assert reuse['accepted_count']==6 and not reuse['rejected_records'],reuse
reused=reuse['accepted_reused_records']
for j in reused:
    j['expected_tasks']=5 if j['dataset']=='het5'else 10
    j['reuse_validation_status']=j['status'];j['status']='completed'
groups=defaultdict(list)
for j in formal+reused:groups[(j['method'],j['dataset'])].append(j['seed'])
assert len(groups)==35 and all(sorted(v)==[0,1,2]for v in groups.values())
assert len(formal)==99 and len(reused)==6 and len(selected)==108
env=dict(b['environment']);env['PYTHONDONTWRITEBYTECODE']='1'
save(ROOT/'suite.json',dict(version=1,allowed_gpus=[2,3,4,5],max_parallel=4,python=b['python'],environment=env,
 policy='Conservative 99 new + 6 verified formal seed0 reuse. New seed0 is cohort establishment, not an accuracy gate. Historical source gaps do not establish that historical scores are invalid.',
 jobs=selected,reused=reused,counts=dict(formal=99,smoke=9,reused=6,groups=35,seeds=[0,1,2]),
 metrics=dict(faa='mean last CNN task accuracy row',aaa_stage='mean of seen-task row means',std='sample SD across seeds, ddof=1'),
 limitations=['Archived per-dataset settings are retained; methods are not asserted to have identical compute budgets.',
 'Data identity binds ordered paths and labels, not historical image bytes.',
 'Reused GAM Het5 has formal source/environment binding but no historical checkpoint byte SHA; current offline cache is verified.',
 'Smoke task cap also changes trainer effective nb_tasks; smoke numbers are not formal results.']))
for d in ['logs','receipts']:(ROOT/d).mkdir(exist_ok=True)
print(json.dumps(dict(jobs=len(selected),formal=len(formal),reused=len(reused),methods=dict(Counter(j['method']for j in formal))),indent=2))
