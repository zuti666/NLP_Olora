#!/usr/bin/env python3
"""CPU-only negative admission tests; never run main or lease a GPU."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

H=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(H))
import common
import run_queue

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    suite=common.load_suite(H/'suite.json')
    for j in suite['jobs']:
        run_queue.verify_output_path(suite,j)
    checks={'all66_registered_output_paths_valid':True}
    with tempfile.TemporaryDirectory(prefix='fresh_admission_',dir=H/'gates') as d:
        d=Path(d)
        def bad_suite(name,modify):
            s=copy.deepcopy(suite);modify(s)
            p=d/(name+'.json');p.write_text(json.dumps(s))
            try:common.load_suite(p)
            except (ValueError,RuntimeError,AssertionError):checks[name]=True
            else:raise AssertionError('Invalid suite was accepted: '+name)
        bad_suite('reject_gpu0',lambda s:s.update(allowed_gpus=[0,2,3,4,5]))
        bad_suite('reject_gpu1',lambda s:s.update(allowed_gpus=[1,2,3,4,5]))
        bad_suite('reject_duplicate_gpu',lambda s:s.update(allowed_gpus=[2,2,3,4,5]))
        bad_suite('reject_parallel_over_gpu_count',lambda s:s.update(max_parallel=6))
        bad_suite('reject_duplicate_jobid',lambda s:s['jobs'].append(copy.deepcopy(s['jobs'][0])))
        bad_suite('reject_unknown_dependency',lambda s:s['jobs'][0].update(depends_on=['missing']))
        bad_suite('reject_dependency_cycle',lambda s:(s['jobs'][0].update(depends_on=[s['jobs'][1]['id']]),s['jobs'][1].update(depends_on=[s['jobs'][0]['id']])))
        bad_suite('reject_manifest_cuda_override',lambda s:s['environment'].update(CUDA_VISIBLE_DEVICES='0'))
        def bad_path(name,s,j):
            try:run_queue.verify_output_path(s,j)
            except (ValueError,RuntimeError,AssertionError):checks[name]=True
            else:raise AssertionError('Escaped output was accepted: '+name)
        s=copy.deepcopy(suite);s['outputs_root']='/data/115-2/users/liying/cache/tmp'
        bad_path('reject_broad_cache_root',s,copy.deepcopy(suite['jobs'][0]))
        j=copy.deepcopy(suite['jobs'][0]);j['output_dir']='/tmp/invalid_upward_output'
        j['command'][j['command'].index('--output-dir')+1]=j['output_dir']
        bad_path('reject_output_escape',suite,j)
        j=copy.deepcopy(suite['jobs'][0]);j['command'][j['command'].index('--output-dir')+1]='/tmp/invalid_upward_output'
        bad_path('reject_cli_output_mismatch',suite,j)
        j=copy.deepcopy(suite['jobs'][0]);cfg=json.loads(Path(j['config']).read_text())
        cfg['outputs_root']=str(Path(j['output_dir'])/'other_native')
        p=d/'native_mismatch.json';p.write_text(json.dumps(cfg));j['config']=str(p);j['config_sha256']=sha(p)
        bad_path('reject_native_output_mismatch',suite,j)
        j=copy.deepcopy(suite['jobs'][0]);j['config_sha256']='0'*64
        bad_path('reject_config_sha_mismatch',suite,j)
    evidence={str(H/n):sha(H/n) for n in ('suite.json','common.py','run_queue.py','validate_job.py')}
    result=dict(status='passed',checked_at_utc=datetime.now(timezone.utc).isoformat(),
                checks=checks,evidence_hashes=evidence,
                gpu_admission=False,gpu_training=False,
                scope='Direct pure admission helpers only; no run_queue.main, query_gpus, locks or CUDA witness')
    target=H/'gates/FRESH_ADMISSION_CPU.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='passed',checks=len(checks),output=str(target))))

if __name__=='__main__':main()
