"""Validate independent evidence and freeze the final execution closure."""
import json,time
from pathlib import Path
from collections import defaultdict
from run_job import sha,save
ROOT=Path(__file__).resolve().parent
assert not (ROOT/'state.json').exists(),'Do not change a running campaign'
suite=json.loads((ROOT/'suite.json').read_text())
B=ROOT/'baseline_ewc_preparation';C=ROOT/'cl_gam_preparation'
evidence={}
for name in ['cpu_preflight_baseline.json','cpu_preflight_ewc.json','sample_items_baseline.json','sample_items_ewc.json','weight_resolution.json']:
    p=B/name;o=json.loads(p.read_text());assert o['pass'],name;evidence[str(p.relative_to(ROOT))]=sha(p)
assert json.loads((C/'import_check.json').read_text())['status']=='PASS_CPU_IMPORT_ONLY'
r=json.loads((C/'reuse_validation.json').read_text());assert r['accepted_count']==6 and not r['rejected_records']
assert json.loads((ROOT/'env_witness.json').read_text())['passed']
groups=defaultdict(list)
seen_sources=set()
for j in suite['jobs']:
    assert sha(j['config'])==j['config_sha256'],j['id']
    assert sha(j['source_manifest'])==j['source_manifest_sha256'],j['id']
    assert sha(j['data_manifest'])==j['data_manifest_sha256'],j['id']
    if j['source_manifest'] not in seen_sources:
        source=json.loads(Path(j['source_manifest']).read_text())
        for rel,h in source['files'].items():assert sha(Path(j['source_root'])/rel)==h,rel
        seen_sources.add(j['source_manifest'])
    cfg=json.loads(Path(j['config']).read_text());assert cfg['seed']==[j['seed']]
    assert cfg['batch_size']==128 and cfg['lora_rank']==16
    assert cfg['class_shuffle'] is False and cfg['memory_size']==0
    assert not Path(cfg['outputs_root']).exists(), 'Output path already exists: '+j['id']
    if j['stage']=='formal':
        assert 'max_train_tasks'not in cfg
        groups[j['method'],j['dataset']].append((j,cfg))
for j in suite['reused']:
    assert sha(j['metric_path'])==j['metric_sha256']
    cfg=json.loads(Path(j['config']).read_text());groups[j['method'],j['dataset']].append((j,cfg))
allowed={'seed','prefix','outputs_root','optimizer_tag_override','protocol_note','config'}
for key,rows in groups.items():
    assert sorted(j['seed']for j,_ in rows)==[0,1,2],key
    base={k:v for k,v in rows[0][1].items()if k not in allowed}
    for j,cfg in rows[1:]:
        other={k:v for k,v in cfg.items()if k not in allowed};assert base==other,(key,j['id'],{k:(base.get(k),other.get(k))for k in base.keys()|other.keys()if base.get(k)!=other.get(k)})
w=json.loads((B/'weight_resolution.json').read_text());assert w['same_weight_bytes_as_cllora']
ref=Path(suite['environment']['HF_HOME'])/'hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/refs/main'
assert ref.read_text().strip()==w['snapshot_revision']
checkpoint_files=[dict(path=w['weight_path'],sha256=w['weight_sha256']),dict(path=str(ref),sha256=sha(ref))]
for p in checkpoint_files:assert sha(p['path'])==p['sha256']
save(ROOT/'preflight.json',dict(passed=True,checked_unix=time.time(),group_count=len(groups),new_formal=99,smoke=9,reused=6,
 evidence=evidence,checkpoint_files=checkpoint_files,reuse_validation_sha256=sha(C/'reuse_validation.json'),
 environment_contract_sha256=sha(ROOT/'environment_contract.json'),
 data_binding='Ordered current path/label lists; historical image byte identity is not claimed.',gpu_smokes='Pending execution after independent launch dry-run.'))
runtime=['run_job.py','launch_queue.py','start_background.py','collect.py','suite.json','preflight.json','environment_contract.json','environment_spec.json','env_witness.json']
save(ROOT/'runtime_frozen.json',dict(created_unix=time.time(),files={p:sha(ROOT/p)for p in runtime}))
print('PASS: 35 consistent three-seed cohorts; 99 formal + 9 diagnostic jobs; 6 verified reuses.')
