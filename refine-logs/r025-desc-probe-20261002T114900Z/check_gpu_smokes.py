#!/usr/bin/env python3
"""Three paired two-task GPU smokes, including a frozen old-US compatibility gate."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from common import atomic_json,now,sha256
from validate_job import validate_job,receipt_matches,require

HERE=Path(__file__).resolve().parent
OLD_ROOT=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-component-horizon-recovery-20261001T230000Z')
OLD_US_ID='smoke_us_natural_r025_s2'


def paired_identity(job):
    out=Path(job['output_dir'])
    identities=json.loads((out/'task_identities.json').read_text())
    trace=[json.loads(line) for line in (out/'step_trace.jsonl').read_text().splitlines()]
    return {
        'Task0_initial':identities[0]['task_initial_parameter_sha256'],
        'frozen_origins':[i['frozen_origin_sha256'] for i in identities],
        'new_head_rows':[i['new_fc_rows_sha256'] for i in identities],
        'class_slices':[i['class_slice'] for i in identities],
        'fc_shapes':[i['fc_shape'] for i in identities],
        'data':[r['identity'] for r in trace],
        'nominal_lr':[r['lr_nominal'] for r in trace],
        'task_epoch_batch':[[r['task'],r['epoch'],r['batch']] for r in trace],
    }


def tensors(job,task):
    """Compare every checkpoint tensor by a stable structural name, never truncating zip."""
    cp=Path(job['output_dir'])/'checkpoints'/f'task_{task:03d}'
    result={}
    def visit(value,name):
        if isinstance(value,torch.Tensor):
            result[name]=value.detach().double().cpu()
        elif isinstance(value,torch.nn.Module):
            visit(value.state_dict(),name+'.state_dict')
        elif isinstance(value,dict):
            for key in sorted(value):visit(value[key],name+'.'+str(key))
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):visit(item,name+'.'+str(index))
        else:
            raise TypeError(f'Unexpected learned checkpoint object {name}: {type(value).__name__}')
    for stem in ('lora_w_a','lora_w_b','fc_state'):
        visit(torch.load(cp/f'{stem}_{task}.pt',map_location='cpu',weights_only=False),stem)
    require(bool(result),'empty learned checkpoint')
    return result


def main():
    suite=json.loads((HERE/'suite.json').read_text())
    sys.path.insert(0,suite['source_root'])
    jobs=[j for j in suite['jobs'] if j['smoke']]
    require(len(jobs)==3 and {(j['rule'],j['scale']) for j in jobs}=={('c21_base','natural'),('c21_perp','natural'),('us','natural')},'wrong A/B/US smoke set')
    require(all(j['expected_tasks']==2 and j['expected_epochs_per_task']==1 and j['expected_steps']==2 for j in jobs),'smokes must include exactly Task0 and Task1, one native batch each')
    old_suite=json.loads((OLD_ROOT/'suite.json').read_text())
    old_us=next(j for j in old_suite['jobs'] if j['id']==OLD_US_ID)
    old_receipt_path=OLD_ROOT/'runtime/jobs'/OLD_US_ID/'completed.json'
    # The new validator deliberately accepts only this campaign's A/B/US jobs.
    # Revalidate the old smoke with its own frozen validator in a fresh process.
    code="""import json,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
from validate_job import validate_job,receipt_matches,require
suite=json.loads((root/'suite.json').read_text());job=next(j for j in suite['jobs'] if j['id']==sys.argv[2])
r=json.loads((root/'runtime/jobs'/job['id']/'completed.json').read_text())
require(r['job_id']==job['id'] and r['exit']['returncode']==0 and receipt_matches(r,validate_job(job)),'old US smoke is not independently validated')
print('OLD_US_VALIDATED')
"""
    validation=subprocess.run([suite['python'],'-c',code,str(OLD_ROOT),OLD_US_ID],cwd=OLD_ROOT,text=True,capture_output=True,timeout=120)
    require(validation.returncode==0 and 'OLD_US_VALIDATED' in validation.stdout,'old US validation failed: '+validation.stderr[-3000:])
    old_config=json.loads(Path(old_us['config']).read_text())
    excluded={'outputs_root','prefix','component_rule'}
    fixed_old={k:v for k,v in old_config.items() if k not in excluded}
    reference=paired_identity(old_us)
    for job in jobs:
        receipt=json.loads((HERE/'runtime/jobs'/job['id']/'completed.json').read_text())
        require(receipt['job_id']==job['id'] and receipt['exit']['returncode']==0 and receipt_matches(receipt,validate_job(job)),'unverified smoke '+job['id'])
        config=json.loads(Path(job['config']).read_text())
        require({k:v for k,v in config.items() if k not in excluded}==fixed_old,'configuration differs from frozen old cohort beyond rule/output '+job['id'])
        require(paired_identity(job)==reference,'initial model/data/head/nominal LR differs from frozen old US '+job['id'])
    new_us=next(j for j in jobs if j['rule']=='us')
    comparisons=[]
    for task in range(2):
        old,new=tensors(old_us,task),tensors(new_us,task)
        require(set(old)==set(new),'US checkpoint tensor names/counts differ')
        require(all(old[k].shape==new[k].shape for k in old),'US checkpoint tensor shapes differ')
        diff_sq=sum(float(((old[k]-new[k])**2).sum()) for k in old)
        norm_sq=sum(float((old[k]**2).sum()) for k in old)
        relative=diff_sq**.5/max(norm_sq**.5,1e-30)
        require(relative<=1e-4,'new/old US trajectory discrepancy exceeds frozen numerical gate')
        comparisons.append({'task':task,'tensor_count':len(old),'relative_l2_difference':relative,
                            'max_abs_difference':max(float((old[k]-new[k]).abs().max()) for k in old)})
    result={
        'status':'passed','at':now(),'verified_smokes':3,
        'all_initial_model_data_head_pairs':True,
        'frozen_old_US_initial_data_head_LR_identity':True,
        'fixed_config_identical_except_output_prefix_rule':True,
        'US_scalar_and_counterfactual_tensor_identity_every_step':True,
        'US_old_new_parameter_comparisons':comparisons,
        'US_old_new_parameter_relative_tolerance':1e-4,
        'suite_sha256':sha256(HERE/'suite.json'),
        'receipts_sha256':{j['id']:sha256(HERE/'runtime/jobs'/j['id']/'completed.json') for j in jobs},
        'old_US_provenance':{'root':str(OLD_ROOT),'job_id':OLD_US_ID,
            'suite_sha256':sha256(OLD_ROOT/'suite.json'),
            'validator_sha256':sha256(OLD_ROOT/'validate_job.py'),
            'receipt_sha256':sha256(old_receipt_path)},
    }
    atomic_json(HERE/'runtime/GPU_SMOKES_VERIFIED.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
