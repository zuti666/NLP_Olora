import ast, copy, json, sys
from pathlib import Path
from unittest.mock import patch
H=Path('/data/140-1/users/liying/O-LoRA/refine-logs/odd-even-four-seed01-20261012T104100Z')
sys.path.insert(0,str(H))
import common, verify_pairing, validate_job
checks=[]
def check(name, fn):
    try: fn(); checks.append({'name':name,'passed':True})
    except Exception as e: checks.append({'name':name,'passed':False,'error':repr(e)}); raise

def assert_true(value):
    if not value: raise AssertionError('False check')

def reject(call):
    try:call()
    except (ValueError,AssertionError,RuntimeError,KeyError):return
    raise AssertionError('Mutation was accepted')

files=['common.py','build_suite.py','verify_pairing.py','validate_job.py','run_queue.py','job_worker.py','suite.json']
for name in files[:-1]:check('syntax_'+name,lambda name=name:ast.parse((H/name).read_text()))
suite=common.load_suite()
check('exact_17_jobs_8_formal_8_smoke_1_pairing',lambda:assert_true(len(suite['jobs'])==17 and sum(j['kind']=='training' and not j['smoke'] for j in suite['jobs'])==8 and sum(j['smoke'] for j in suite['jobs'])==8))
check('source_190_files_unchanged',lambda:assert_true(len(common.verify_source(suite)['source_files'])==190))
for job in suite['jobs']:
    if job['kind']=='training':check('exact_parent_config_'+job['id'],lambda job=job:common.verify_config(job))

mutations={
 'reject_old_gpu123':lambda s:s.update(allowed_gpus=[1,2,3]),
 'reject_extra_gpu':lambda s:s.update(allowed_gpus=[4,5,6,0]),
 'reject_parallel_4':lambda s:s.update(max_parallel=4),
 'reject_changed_idle_memory':lambda s:s.update(gpu_free_threshold_mib=501),
 'reject_changed_idle_util':lambda s:s.update(gpu_free_util_below_percent=11),
 'reject_private_lock_directory':lambda s:s.update(gpu_lock_dir='/tmp/private'),
 'reject_seed2_supplement':lambda s:s.update(seeds=[0,2]),
 'reject_unregistered_job':lambda s:s['jobs'].pop(),
 'reject_wrong_job_seed':lambda s:s['jobs'][0].update(seed=1),
 'reject_missing_smoke_dependency':lambda s:s['jobs'][8]['depends_on'].pop(),
 'reject_formal_bypasses_pairing':lambda s:s['jobs'][9].update(depends_on=[]),
 'reject_formal_alternate_command':lambda s:s['jobs'][9]['command'].append('--other'),
}
for name, mutation in mutations.items():
    def run(mutation=mutation):
        changed=copy.deepcopy(suite);mutation(changed)
        with patch.object(common,'read_json',return_value=changed):reject(lambda:common.load_suite())
    check(name,run)

# Runtime scientific settings must remain exact to each corresponding historical config.
job=suite['jobs'][9];real_read=common.read_json
for key,value in [('rho',.1),('seed',[1]),('lora_rank',20),('gam_grad_norm_rho',1.),('extra_hidden_option',True)]:
    def run(key=key,value=value):
        cfg=real_read(job['config']);cfg[key]=value
        def mocked(path):return cfg if Path(path)==Path(job['config']) else real_read(path)
        with patch.object(common,'read_json',side_effect=mocked):reject(lambda:common.verify_config(job))
    check('reject_config_mutation_'+key,run)

# Synthetic pairing checks exercise seed-group boundaries only; these are NOT GPU smoke results.
smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
identity_data={};trace_data={}
for j in smokes:
    seed=j['seed'];out=Path(j['output_dir'])
    identity_data[str(out/'task_identities.json')]=[dict(task_initial_parameter_sha256=f'initial{seed}',old_fc_rows_inherited_exact=True,frozen_origin_sha256=f'origin{seed}',new_fc_rows_sha256=f'newrows{seed}_{t}',class_slice=[20*t,20*(t+1)]) for t in [0,1]]
    trace_data[str(out/'step_trace.jsonl')]=[dict(task=t,epoch=1,batch=0,seed=seed,identity={'input':f'input{seed}_{t}'},probe_tensor_sha256={'g0':f'g0{seed}','g1_plus':f'plus{seed}','g1_minus':f'minus{seed}'},rng0_sha256=f'rng0{seed}',rng1_sha256=f'rng1{seed}') for t in [0,1]]
def run_pairing():
    with patch.object(validate_job,'validate_training',return_value={}),patch.object(verify_pairing,'sha256',return_value='synthetic-only'),patch.object(verify_pairing,'read_json',side_effect=lambda path:identity_data[str(path)]),patch.object(Path,'read_text',autospec=True,side_effect=lambda path,*a,**kw:'\n'.join(json.dumps(r) for r in trace_data[str(path)])):
        return verify_pairing.compute_pairing(suite)
check('synthetic_cross_seed_differences_allowed',lambda:assert_true(run_pairing()['status']=='passed'))
changed_path=str(Path(smokes[0]['output_dir'])/'step_trace.jsonl')
original=copy.deepcopy(trace_data[changed_path])
trace_data[changed_path][1]['identity']['input']='unpaired'
check('synthetic_within_seed_task1_batch_mismatch_rejected',lambda:reject(run_pairing))
trace_data[changed_path]=copy.deepcopy(original)
trace_data[changed_path][0]['probe_tensor_sha256']['g0']='unpaired'
check('synthetic_within_seed_initial_probe_mismatch_rejected',lambda:reject(run_pairing))
trace_data[changed_path]=copy.deepcopy(original)

# Only witness seed selection changed in the actual scheduler/worker control paths.
old=H.parent/'odd-even-rho005-gpu123-20261003T185629Z'
check('queue_gpu_dual_lock_and_failure_logic_unchanged',lambda:assert_true((H/'run_queue.py').read_text()==(old/'run_queue.py').read_text().replace("witness['seed']!=2", "witness['seed']!=jobs[jid]['seed']")))
check('worker_dual_lease_and_exit_logic_unchanged',lambda:assert_true((H/'job_worker.py').read_text()==(old/'job_worker.py').read_text().replace("'--output',job['cuda_witness_output']]", "'--output',job['cuda_witness_output'],'--seed',str(job['seed'])]")))
failed=[x for x in checks if not x['passed']]
report=dict(status='passed' if not failed else 'failed',created_at=common.now(),cpu_only=True,gpu_acceptance=False,gpu_calls=0,checks=checks,checks_run=len(checks),failed_checks=failed,files_sha256={name:common.sha256(H/name) for name in files},allowed_gpus=suite['allowed_gpus'],max_parallel=3,synthetic_pairing_only=True,scope='Manifest/config mutation rejection and synthetic seed-group pairing boundaries; actual trainer and smoke gates remain mandatory',launched=False)
report['tests_script_sha256']=common.sha256(Path(__file__))
common.atomic_json(H/'gates/QUEUE_CPU_GATE.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('checks','files_sha256')}))
