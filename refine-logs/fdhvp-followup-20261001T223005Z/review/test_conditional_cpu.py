"""Synthetic decision/receipt/pairing tests; never invokes a queue or GPU.

The full native-result validator has separate tests. Here its API is replaced
with a small file/hash validator, so these tests isolate branch logic, report
omission and the smoke checker's actual pairing/tensor comparisons.
"""
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import conditional_gate as gate
import collect_results as collector
import check_gpu_smokes as smoke

PINS = {n: gate.sha(ROOT / n) for n in ('conditional_gate.py', 'collect_results.py', 'check_gpu_smokes.py')}
CHECKS = {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def check(name, condition):
    assert condition, name
    CHECKS[name] = True


def rejected(name, fn, expected):
    try:
        fn()
    except ValueError as exc:
        assert expected in str(exc), (name, str(exc), expected)
        CHECKS[name] = {'rejected': True, 'reason': str(exc)}
    else:
        raise AssertionError('Invalid data accepted: ' + name)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.suite = gate.read(ROOT / 'suite.json')
        self.frozen = gate.read(ROOT / 'FROZEN_GATES.json')
        self.suite['source_root'] = str(root / 'source')
        for j in self.suite['jobs']:
            j['output_dir'] = str(root / 'outputs' / j['id'])
            j['config'] = str(root / 'configs' / (j['id'] + '.json'))
            write(Path(j['config']), {'id': j['id']})
            j['config_sha256'] = gate.sha(j['config'])
        self.jobs = {j['id']: j for j in self.suite['jobs']}
        write(root / 'suite.json', self.suite)
        write(root / 'FROZEN_GATES.json', self.frozen)
        self.base = 'screen_r1_u0_natural_s2'
        self.ug = 'screen_r1_ug_natural_s2'
        self.second = 'second_hvp_natural_s2'
        self.coeff = 'coeff_us_l05_s2'
        self.first = 'first_hvp_second_fd_natural_s2'

    def validate(self, job):
        assert gate.sha(job['config']) == job['config_sha256']
        out = Path(job['output_dir'])
        summary = gate.read(out / 'summary.json')
        result = {'config_sha256': gate.sha(job['config']), 'summary_sha256': gate.sha(out / 'summary.json'),
                  'summary': summary, 'step_trace_sha256': None, 'identity_sha256': None,
                  'evaluations_sha256': None, 'checkpoints': []}
        if job['kind']=='horizon':
            result.update(step_trace_sha256=gate.sha(out / 'step_trace.jsonl'),
                          identity_sha256=gate.sha(out / 'task_identities.json'))
        if job['smoke']:
            cp = out / 'checkpoints/task_001'
            result['checkpoints'] = {p.name: gate.sha(p) for p in cp.iterdir()}
        return result

    @staticmethod
    def matches(receipt, actual):
        return all(receipt.get(k) == v for k, v in actual.items() if k != 'summary')

    def use(self):
        gate._validation_api = lambda: (self.validate, self.matches)
        gate._pairing_api = lambda: self.pair

    def pair(self,jobs,required_backend=None):
        if required_backend is not None and any(j['attention_backend']!=required_backend for j in jobs):
            raise ValueError('paired backends differ')
        traces=[(Path(j['output_dir'])/'step_trace.jsonl').read_text() for j in jobs]
        if any(x!=traces[0] for x in traces):raise ValueError('paired actual inputs differ')
        return {'status':'PASS','required_backend':required_backend,'arms':[j['id'] for j in jobs]}

    def publish(self, jid, aaa=60., faa=50., available=True):
        j = self.jobs[jid]
        out = Path(j['output_dir'])
        summary = {'status': 'completed', 'completed_tasks': 10, 'max_match_relative_error': 0.,
                   'final_metrics': dict(zip(collector.METRICS, [aaa, faa, 55., -5., 5., aaa+1., faa+1.])),
                   'hvp_all_available': available, 'batches': [{}] * j.get('max_batches', 0),
                   'measured_alignment': -0.9, 'measured_FD_relative_error': 200.}
        write(out / 'summary.json', summary)
        if j['kind']=='horizon':
            ids = [{'task_initial_parameter_sha256': 'same_init' if t == 0 else jid,
                    'frozen_origin_sha256': 'same_frozen', 'new_fc_rows_sha256': {'weight': 'same_new_'+str(t)},
                    'class_slice': [20*t, 20*(t+1)]} for t in range(j['expected_tasks'])]
            write(out / 'task_identities.json', ids)
            trace = [{'identity': {'data_seed': 200001+1000*t, 'input_sha256': 'same_input',
                      'indices': [1, 2]}, 'lr_nominal': .04 if t == 0 else .02} for t in range(j['expected_tasks'])]
            (out / 'step_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
        if j['smoke']:
            cp = out / 'checkpoints/task_001'
            cp.mkdir(parents=True, exist_ok=True)
            aa, bb = torch.nn.Linear(2, 1, bias=False), torch.nn.Linear(1, 2, bias=False)
            # Same actual US identity control; other rules deliberately differ.
            value = 1. if j['rule'] == 'us' else float(len(jid))
            with torch.no_grad():
                aa.weight.fill_(value)
                bb.weight.fill_(value)
            torch.save([aa], cp / 'lora_w_a_1.pt')
            torch.save([bb], cp / 'lora_w_b_1.pt')
            torch.save({'weight': torch.full((40, 2), value), 'bias': torch.zeros(40)}, cp / 'fc_state_1.pt')
        self.refresh(jid)

    def refresh(self, jid):
        record = {'job_id': jid, 'exit': {'returncode': 0}, **self.validate(self.jobs[jid])}
        record.pop('summary')
        write(self.root / 'runtime/jobs' / jid / 'completed.json', record)

    def decide(self, jid):
        return gate.evaluate(self.jobs[jid], self.suite, self.root)

    def diagnostics(self, available=True):
        for jid in self.frozen['definitions']['correction_signal_with_hvp']['diagnostic_ids']:
            self.publish(jid, available=available)

    def skip(self, jid):
        decision = self.decide(jid)
        assert decision['decision'] == 'skip'
        j = self.jobs[jid]
        receipt = {'job_id': jid, 'status': 'skipped_by_gate', 'config_sha256': j['config_sha256'],
                   'suite_sha256': gate.sha(self.root/'suite.json'),
                   'frozen_gates_sha256': gate.sha(self.root/'FROZEN_GATES.json'), 'gate_result': decision}
        write(self.root / 'runtime/jobs' / jid / 'skipped.json', receipt)
        return receipt


with tempfile.TemporaryDirectory(prefix='fdhvp-conditional-cpu-') as temp:
    f = Fixture(Path(temp)); f.use()
    check('unconditional_core_ready', f.decide(f.base)['decision'] == 'ready')
    check('absent_core_waits', f.decide(f.second)['decision'] == 'wait')
    f.publish(f.base); f.publish(f.ug, aaa=60.999, faa=50.)
    check('below_signal_second_skips', f.decide(f.second)['decision'] == 'skip')
    check('below_signal_coefficient_ready', f.decide(f.coeff)['decision'] == 'ready')
    f.publish(f.ug, aaa=62., faa=48.999)
    check('FAA_guard_applies', f.decide(f.second)['decision'] == 'skip')
    f.publish(f.ug, aaa=61., faa=49.)
    check('exact_signal_boundary_waits_for_diagnostics', f.decide(f.second)['decision'] == 'wait')
    check('signal_true_no_lambda_fallback', f.decide(f.coeff)['decision'] == 'skip')
    f.diagnostics()
    check('exact_boundary_and_low_alignment_release', f.decide(f.second)['decision'] == 'ready')
    f.diagnostics(available=False)
    check('unsupported_HVP_skips_second', f.decide(f.second)['decision'] == 'skip')
    check('unsupported_HVP_does_not_trigger_lambda', f.decide(f.coeff)['decision'] == 'skip')
    # First response cannot wait forever on a validly skipped second branch.
    for pair in f.frozen['definitions']['second_hvp_effect']['comparisons']:
        for jid in pair.values(): f.skip(jid)
    check('valid_second_skips_propagate_to_first', f.decide(f.first)['decision'] == 'skip')
    rec = f.skip(f.first)
    check('current_skip_validates', gate.verify_skipped(f.jobs[f.first], rec, f.suite, f.root)['decision']=='skip')
    wrong = copy.deepcopy(rec); wrong['frozen_gates_sha256'] = 'changed'
    rejected('stale_skip_rejected', lambda: gate.verify_skipped(f.jobs[f.first],wrong,f.suite,f.root), 'frozen gates changed')
    result = collector.collect(f.root)
    check('collector_separates_skips', result['validated_full_horizons']==2 and result['skipped_count']==5)
    with (f.root/'results/horizon_metrics.csv').open() as h:
        rows = list(csv.DictReader(h))
    check('collector_absent_not_zero_filled', len(rows)==2 and {r['job_id'] for r in rows}=={f.base,f.ug})

with tempfile.TemporaryDirectory(prefix='fdhvp-effect-cpu-') as temp:
    f = Fixture(Path(temp)); f.use()
    f.publish(f.base); f.publish(f.ug, aaa=61., faa=49.); f.diagnostics()
    check('first_waits_for_four_trajectories', f.decide(f.first)['decision']=='wait')
    pairs=f.frozen['definitions']['second_hvp_effect']['comparisons']
    f.publish(pairs[0]['fd_id']); f.publish(pairs[0]['hvp_id'],aaa=65.)
    check('one_large_effect_still_waits_for_all_four', f.decide(f.first)['decision']=='wait')
    f.publish(pairs[1]['fd_id']); f.publish(pairs[1]['hvp_id'])
    check('first_any_scale_effect_releases', f.decide(f.first)['decision']=='ready')
    f.publish(pairs[0]['hvp_id'],aaa=59.)
    check('negative_exact_effect_boundary_releases', f.decide(f.first)['decision']=='ready')
    f.publish(pairs[0]['hvp_id'],aaa=59.001)
    check('both_below_absolute_threshold_skip_first', f.decide(f.first)['decision']=='skip')
    f.frozen['definitions']['second_hvp_effect']['min_abs_AAA_difference_pp']=.5
    write(f.root/'FROZEN_GATES.json', f.frozen)
    check('effect_threshold_read_from_frozen_input', f.decide(f.first)['decision']=='ready')
    f.frozen['definitions']['correction_signal_with_hvp']['min_AAA_gain_pp']=2.
    write(f.root/'FROZEN_GATES.json', f.frozen)
    check('signal_threshold_read_from_frozen_input', f.decide(f.second)['decision']=='skip')
    f.publish(f.ug,aaa=float('nan'))
    rejected('nonfinite_metric_rejected',lambda:f.decide(f.second),'invalid finite number')
    f.publish(f.ug,aaa=62.)
    ip=Path(f.jobs[f.ug]['output_dir'])/'task_identities.json';ids=gate.read(ip);ids[0]['task_initial_parameter_sha256']='wrong';write(ip,ids);f.refresh(f.ug)
    rejected('core_initialization_mismatch_rejected',lambda:f.decide(f.second),'initial model')
    f.publish(f.ug,aaa=62.)
    ip=Path(f.jobs[f.ug]['output_dir'])/'step_trace.jsonl';ip.write_text(ip.read_text().replace('same_input','wrong_input'));f.refresh(f.ug)
    rejected('core_actual_input_mismatch_rejected',lambda:f.decide(f.second),'actual inputs differ')
    f.publish(f.ug,aaa=62.)
    rp=f.root/'runtime/jobs'/f.ug/'completed.json';r=gate.read(rp);r['exit']['returncode']=1;write(rp,r)
    rejected('nonzero_completed_exit_rejected',lambda:f.decide(f.second),'identity/exit')
    f.refresh(f.ug);r=gate.read(rp);r['summary_sha256']='changed';write(rp,r)
    rejected('tampered_completed_hash_rejected',lambda:f.decide(f.second),'receipt changed')
    f.refresh(f.ug);r=gate.read(rp);r['job_id']='other';write(rp,r)
    rejected('wrong_completed_identity_rejected',lambda:f.decide(f.second),'identity/exit')

with tempfile.TemporaryDirectory(prefix='fdhvp-smoke-cpu-') as temp:
    f=Fixture(Path(temp));f.use()
    base=[j for j in f.suite['jobs'] if j['smoke'] and j['smoke_group']=='base']
    for j in base:f.publish(j['id'])
    proof=smoke.check_group('base',f.root)
    check('base10_pairing_and_real_CPU_tensor_identity_pass', proof['verified_smokes']==10 and proof['US_lambda1_identity']['parameter_relative_l2']==0.)
    check('different_rule_endstate_not_compared',proof['different_rule_endpoint_equality_required'] is False)
    f.publish(f.base);f.publish(f.ug,aaa=61.,faa=49.);f.diagnostics()
    second=[j for j in f.suite['jobs'] if j['smoke'] and j['smoke_group']=='second_hvp']
    for j in second:f.publish(j['id'])
    proof=smoke.check_group('second_hvp',f.root)
    check('second4_cross_group_pairs_without_US_identity_requirement',proof['verified_smokes']==4 and proof['US_lambda1_identity'] is None and proof['base_group_binding'] is not None)
    for pair in f.frozen['definitions']['second_hvp_effect']['comparisons']:
        f.publish(pair['fd_id']);f.publish(pair['hvp_id'],aaa=61.)
    first=[j for j in f.suite['jobs'] if j['smoke'] and j['smoke_group']=='first_hvp']
    for j in first:f.publish(j['id'])
    proof=smoke.check_group('first_hvp',f.root)
    check('first4_pairing_pass',proof['verified_smokes']==4 and proof['US_lambda1_identity'] is None)
    bad=second[0];path=Path(bad['output_dir'])/'task_identities.json';ids=gate.read(path);ids[1]['new_fc_rows_sha256']['weight']='wrong';write(path,ids);f.refresh(bad['id'])
    rejected('changed_second_task_head_pair_rejected',lambda:smoke.check_group('second_hvp',f.root),'pairing differs')
    f.publish(bad['id'])
    path=Path(bad['output_dir'])/'step_trace.jsonl';trace=[json.loads(v) for v in path.read_text().splitlines()];trace[0]['identity']['input_sha256']='wrong';path.write_text(''.join(json.dumps(v)+'\n' for v in trace));f.refresh(bad['id'])
    rejected('changed_input_pair_rejected',lambda:smoke.check_group('second_hvp',f.root),'pairing differs')
    f.publish(bad['id'])
    bp=f.root/'runtime/GPU_SMOKES_base_VERIFIED.json';b=gate.read(bp);b['suite_sha256']='changed';write(bp,b)
    rejected('stale_base_proof_rejected',lambda:smoke.check_group('second_hvp',f.root),'base smoke proof stale')
    smoke.check_group('base',f.root)
    us=next(j for j in base if j['rule']=='us' and j['scale']=='matched')
    cp=Path(us['output_dir'])/'checkpoints/task_001/fc_state_1.pt';state=torch.load(cp,weights_only=False);state['weight'].add_(1.);torch.save(state,cp);f.refresh(us['id'])
    rejected('US_lambda1_tensor_mismatch_rejected',lambda:smoke.check_group('base',f.root),'numerical gate failed')
    rejected('missing_or_unknown_smoke_group_rejected',lambda:smoke.check_group('missing',f.root),'unknown or empty')
    # Shape checks prohibit silent zip truncation in the identity comparison.
    rejected('US_tensor_count_mismatch_rejected',lambda:smoke.compare_parameters([('x',torch.ones(2))],[],.1),'structures differ')
    f.publish(us['id']);smoke.check_group('base',f.root)
    f.publish(f.ug,aaa=60.)
    coeff=[j for j in f.suite['jobs'] if j['smoke'] and j['smoke_group']=='lambda']
    for j in coeff:f.publish(j['id'])
    proof=smoke.check_group('lambda',f.root)
    check('lambda4_pairs_without_false_US_lambda1_identity',proof['verified_smokes']==4 and proof['US_lambda1_identity'] is None)

assert PINS == {n:gate.sha(ROOT/n) for n in PINS}, 'implementation changed during tests'
result={'status':'PASS','checks':CHECKS,'check_count':len(CHECKS),
        'negative_count':sum(isinstance(v,dict) for v in CHECKS.values()),
        'implementation_sha256':PINS,'test_sha256':gate.sha(__file__),
        'scope':'CPU-only synthetic branch and report fixtures, actual CPU checkpoint tensors; full validator tested separately; no queue/GPU invoked.'}
write(ROOT/'review/CONDITIONAL_CPU_REPORT.json',result)
print(json.dumps({'status':result['status'],'checks':result['check_count'],'negative_count':result['negative_count'],'pins':PINS},indent=2))
