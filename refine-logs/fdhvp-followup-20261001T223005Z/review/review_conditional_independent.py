"""Independent CPU replay and manifest/interface checks; no queue invocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
NAMES=('conditional_gate.py','collect_results.py','check_gpu_smokes.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
pins={name:sha(ROOT/name) for name in NAMES}
ns=runpy.run_path(str(ROOT/'review/test_conditional_cpu.py'))
replay=read(ROOT/'review/CONDITIONAL_CPU_REPORT.json')
assert replay['status']=='PASS' and replay['check_count']==39 and replay['negative_count']==13
suite=read(ROOT/'suite.json');gates=read(ROOT/'FROZEN_GATES.json');jobs={j['id']:j for j in suite['jobs']}
d=gates['definitions'];signal=d['correction_signal_with_hvp'];fallback=d['no_correction_signal']
for key in ('baseline_id','candidate_id','min_AAA_gain_pp','min_FAA_gain_pp'):assert signal[key]==fallback[key]
assert signal['min_AAA_gain_pp']==1. and signal['min_FAA_gain_pp']==-1.
assert gates['HVP_alignment_is_not_a_release_gate'] is True
assert gates['conditional_skip_is_not_proof_of_no_effect'] is True
assert gates['matching_relative_tolerance']==.001 and gates['US_smoke_relative_parameter_tolerance']==.0001
pairs=d['second_hvp_effect']['comparisons'];assert len(pairs)==2
ids=[pair[key] for pair in pairs for key in ('fd_id','hvp_id')];assert len(set(ids))==4
assert d['second_hvp_effect']['min_abs_AAA_difference_pp']==1.
assert {jobs[p['fd_id']]['scale'] for p in pairs}=={'natural','matched'}
for p in pairs:
 a,b=jobs[p['fd_id']],jobs[p['hvp_id']]
 assert all(a[k]==b[k] for k in ('seed','radius','rule','scale','lambda_corr','first_response','attention_backend','expected_tasks','expected_epochs_per_task','expected_steps'))
 assert a['rule']=='ug' and a['first_response']=='fd' and a['second_response']=='fd' and b['second_response']=='hvp'
 assert a['attention_backend']=='math' and a['seed']==2 and a['radius']==1. and a['lambda_corr']==1.
 assert a['expected_tasks']==10 and not a['smoke'] and not b['smoke']
assert [jobs[j]['task'] for j in signal['diagnostic_ids']]==[0,4,9]
assert all(jobs[j]['radius']==1. and jobs[j]['rule']=='ug' and jobs[j]['scale']=='natural' for j in signal['diagnostic_ids'])
for name,count in {'base':10,'second_hvp':4,'first_hvp':4,'lambda':4}.items():
 selected={j['id'] for j in suite['jobs'] if j['smoke'] and j['smoke_group']==name}
 assert len(selected)==count and selected==set(suite['smoke_groups'][name])
 assert all(jobs[j]['expected_tasks']==2 for j in selected)
 for j in suite['jobs']:
  if j['kind']=='horizon' and not j['smoke'] and j['smoke_group']==name:assert set(j['depends_on'])==selected
# Exercise exactly the root queue's canonical envelope builder/verifier in a
# temporary synthetic fixture. Never call main(), query GPUs, or launch jobs.
spec=importlib.util.spec_from_file_location('independent_queue_skip_interface',ROOT/'run_queue.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
with tempfile.TemporaryDirectory(prefix='fdhvp-independent-skip-') as temp:
 f=ns['Fixture'](Path(temp));f.use();f.publish(f.base);f.publish(f.ug,aaa=60.5)
 job=f.jobs[f.second];decision=f.decide(job['id']);assert decision['decision']=='skip'
 queue.HERE=f.root;receipt=queue.skipped_receipt(job,decision)
 assert receipt['status']=='skipped_by_gate' and 'summary' not in receipt and 'final_metrics' not in receipt
 path=f.root/'runtime/jobs'/job['id']/'skipped.json';ns['write'](path,receipt)
 queue.verify_skip(job,{'skip_sha256':sha(path)})
 bad=dict(receipt);bad['gate_result']=dict(decision,decision='ready');ns['write'](path,bad)
 try:queue.verify_skip(job,{'skip_sha256':sha(path)})
 except ValueError:canonical_wrong_decision_rejected=True
 else:raise AssertionError('canonical skip wrong decision accepted')
assert pins=={name:sha(ROOT/name) for name in NAMES}
report={'status':'PASS','blocking_findings':[],
 'replayed_checks':39,'replayed_negative_cases':13,'canonical_queue_skip_roundtrip':True,
 'canonical_wrong_decision_rejected':canonical_wrong_decision_rejected,
 'manifest_checks':{'exact_frozen_signal_AAA_FAA':True,'low_alignment_does_not_reject':True,
 'all_four_second_response_arms_completed_before_effect_gate':True,'paired_same_math_backend':True,
 'absolute_effect_not_improvement_test':True,'r1_native_UG_checkpoints_0_4_9':True,
 'base10_second4_first4_lambda4_smoke_groups_and_full_dependencies':True,
 'skipped_or_pending_numeric_results_omitted_not_zero':True},
 'implementation_sha256':pins,'suite_sha256':sha(ROOT/'suite.json'),'frozen_gates_sha256':sha(ROOT/'FROZEN_GATES.json'),
 'queue_interface_sha256_at_review':sha(ROOT/'run_queue.py'),'test_sha256':sha(__file__),
 'scope':'Read-only static review plus CPU synthetic gates/receipts/tensors; no real-GPU smoke or queue launch.',
 'scientific_limits':['Seed2 branch thresholds are exploratory pilot decisions, not significance or equivalence tests.',
 'An absolute negative HVP effect may release the next investigation; it is not an improvement claim.',
 'Explicit unsupported HVP may skip its conditional branch; low alignment alone never does.',
 'The conditional tests stub full completion validation to isolate branch logic; separate actual-GAM/HVP validator CPU tests provide that layer.']}
(ROOT/'review/CONDITIONAL_INDEPENDENT_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
