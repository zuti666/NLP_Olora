#!/usr/bin/env python3
"""Cross-arm real-GPU pairing and Sum/Mean2 equivalence gate before formal jobs."""
import json
from pathlib import Path
import sys
import torch
from common import atomic_json,now,sha256
from validate_job import validate_job,receipt_matches,require

HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text());sys.path.insert(0,suite['source_root'])
smokes=[j for j in suite['jobs'] if j['smoke']]
for job in smokes:
    receipt=json.loads((HERE/'runtime/jobs'/job['id']/'completed.json').read_text())
    require(receipt_matches(receipt,validate_job(job)),'unverified GPU short test '+job['id'])
formation=[j for j in smokes if j['kind']=='formation' and j['seed']==2]
base=None
for job in formation:
    out=Path(job['output_dir']);identity=json.loads((out/'identity.json').read_text())
    row=json.loads((out/'step_trace.jsonl').read_text())
    paired={'initial_model_sha256':identity['initial_model_sha256'],'frozen_origin_sha256':identity['frozen_origin_sha256'],
            'class_order':identity['class_order'],'task_range':identity['task_range'],'batch':row['identity']}
    if base is None: base=paired
    require(paired==base,'different initial model/data/augmentation '+job['id'])

def state(cp):
    aa=torch.load(cp/'lora_w_a_0.pt',map_location='cpu',weights_only=False)
    bb=torch.load(cp/'lora_w_b_0.pt',map_location='cpu',weights_only=False)
    fc=torch.load(cp/'fc_state_0.pt',map_location='cpu',weights_only=False)
    factors=[tensor.detach().double() for pair in zip(aa,bb) for module in pair for tensor in module.parameters()]
    eff=[b.weight.detach().double()@a.weight.detach().double() for a,b in zip(aa,bb)]
    return factors+list(v.double() for v in fc.values()),eff+list(v.double() for v in fc.values())
origin=Path(next(j['output_dir'] for j in formation if j['arm']=='sum'))/'checkpoints'
mean=Path(next(j['output_dir'] for j in formation if j['arm']=='mean_lr2'))/'checkpoints'
sumpar,sumeff=state(origin/'epoch_001');meanpar,meaneff=state(mean/'epoch_001');_,baseeff=state(origin/'epoch_000')
diff=(sum(float(((a-b)**2).sum()) for a,b in zip(sumeff,meaneff)))**.5
delta=(sum(float(((a-b)**2).sum()) for a,b in zip(sumeff,baseeff)))**.5
relative=diff/max(delta,1e-30)
require(relative<=1e-3,'Sum/Mean2 actual effective model update mismatch')
future=[j for j in smokes if j['kind']=='future']
identities=[json.loads((Path(j['output_dir'])/'identity.json').read_text()) for j in future]
require(identities[0]['origin_parameter_sha256']==identities[1]['origin_parameter_sha256'],'future origins differ')
require(identities[0]['expansion']['new_rows_sha256']==identities[1]['expansion']['new_rows_sha256'],'future head initial rows differ')
traces=[json.loads((Path(j['output_dir'])/'step_trace.jsonl').read_text()) for j in future]
require(traces[0]['identity']==traces[1]['identity'] and traces[0]['lr_nominal']==traces[1]['lr_nominal'],'future data/augmentation/LR differ')
result={'status':'passed','at':now(),'verified_smokes':len(smokes),'paired_formation_seed2_arms':len(formation),
    'Sum_vs_Mean2_effective_model_difference':diff,'Sum_effective_delta_from_initial':delta,
    'Sum_vs_Mean2_relative_to_actual_update':relative,
    'Sum_vs_Mean2_max_parameter_abs_difference':max(float((a-b).abs().max()) for a,b in zip(sumpar,meanpar)),
    'future_head_and_data_pairing_passed':True,
    'suite_sha256':sha256(HERE/'suite.json'),'receipt_sha256':{j['id']:sha256(HERE/'runtime/jobs'/j['id']/'completed.json') for j in smokes}}
atomic_json(HERE/'runtime/GPU_SMOKES_VERIFIED.json',result)
print(json.dumps(result,indent=2))
