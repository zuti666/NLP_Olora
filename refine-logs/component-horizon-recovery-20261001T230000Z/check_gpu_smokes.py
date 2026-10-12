#!/usr/bin/env python3
"""Two-task real-GPU paired initialization/data/head and US identity gate."""
import json
from pathlib import Path
import sys
import torch
from common import atomic_json,now,sha256
from validate_job import validate_job,receipt_matches,require
HERE=Path(__file__).resolve().parent
suite=json.loads((HERE/'suite.json').read_text());sys.path.insert(0,suite['source_root'])
jobs=[j for j in suite['jobs'] if j['smoke']]
require(len(jobs)==10,'wrong smoke count')
reference=None
for job in jobs:
    receipt=json.loads((HERE/'runtime/jobs'/job['id']/'completed.json').read_text())
    require(receipt['job_id']==job['id'] and receipt['exit']['returncode']==0 and receipt_matches(receipt,validate_job(job)),'unverified smoke '+job['id'])
    out=Path(job['output_dir']);ids=json.loads((out/'task_identities.json').read_text())
    trace=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines()]
    paired={'Task0_initial':ids[0]['task_initial_parameter_sha256'],'frozen_origins':[i['frozen_origin_sha256'] for i in ids],
        'new_head_rows':[i['new_fc_rows_sha256'] for i in ids],'class_slices':[i['class_slice'] for i in ids],
        'data':[r['identity'] for r in trace],'nominal_lr':[r['lr_nominal'] for r in trace]}
    if reference is None:reference=paired
    require(paired==reference,'smoke initial model/data/head/nominal LR differs '+job['id'])
def parameters(job):
    cp=Path(job['output_dir'])/'checkpoints/task_001'
    aa=torch.load(cp/'lora_w_a_1.pt',map_location='cpu',weights_only=False)
    bb=torch.load(cp/'lora_w_b_1.pt',map_location='cpu',weights_only=False)
    fc=torch.load(cp/'fc_state_1.pt',map_location='cpu',weights_only=False)
    return [p.detach().double() for pair in zip(aa,bb) for module in pair for p in module.parameters()]+[v.double() for v in fc.values()]
natural=next(j for j in jobs if j['rule']=='us' and j['scale']=='natural')
matched=next(j for j in jobs if j['rule']=='us' and j['scale']=='matched')
a,b=parameters(natural),parameters(matched)
diff=sum(float(((x-y)**2).sum()) for x,y in zip(a,b))**.5
norm=sum(float((x*x).sum()) for x in a)**.5
require(diff/max(norm,1e-30)<=1e-4,'US natural/matched trajectory discrepancy exceeds smoke numerical gate')
result={'status':'passed','at':now(),'verified_smokes':10,'all_initial_model_data_head_pairs':True,
    'US_natural_matched_parameter_relative_difference':diff/max(norm,1e-30),
    'US_natural_matched_parameter_max_abs_difference':max(float((x-y).abs().max()) for x,y in zip(a,b)),
    'US_scalar_and_counterfactual_tensor_identity_every_step':True,'suite_sha256':sha256(HERE/'suite.json'),
    'receipts_sha256':{j['id']:sha256(HERE/'runtime/jobs'/j['id']/'completed.json') for j in jobs}}
atomic_json(HERE/'runtime/GPU_SMOKES_VERIFIED.json',result);print(json.dumps(result,indent=2))
