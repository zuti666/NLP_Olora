#!/usr/bin/env python3
"""Read-only, receipt-bound decisions for the preregistered seed-2 branches."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'invalid finite number: ' + name)
    return float(value)


def _validation_api():
    # This module only validates this campaign. Cross-parent validation belongs
    # in a fresh subprocess with that parent's cwd/PYTHONPATH, never this import.
    from validate_job import validate_job, receipt_matches
    return validate_job, receipt_matches


def _pairing_api():
    from validate_job import validate_paired_inputs
    return validate_paired_inputs


def paired_inputs(jobs, required_backend=None):
    """Validate actual input/LR pairing plus exogenous initialization/head pairing."""
    audit = _pairing_api()(jobs, required_backend=required_backend)
    require(audit['status'] == 'PASS', 'full trajectory pairing did not pass')
    reference = None
    for job in jobs:
        identities = read(Path(job['output_dir']) / 'task_identities.json')
        require(len(identities) == job['expected_tasks'], 'paired task identities incomplete')
        value = {'task0_initial': identities[0]['task_initial_parameter_sha256'],
                 'frozen_origins': [v['frozen_origin_sha256'] for v in identities],
                 'new_head_rows': [v['new_fc_rows_sha256'] for v in identities],
                 'class_slices': [v['class_slice'] for v in identities]}
        if reference is None:
            reference = value
        require(value == reference, 'paired initial model/frozen origins/new head differs')
    return {**audit, 'same_initial_model_frozen_origins_new_heads': True}


def validated_record(job, root=HERE):
    """Return None for absent completion; invalid/failed data never becomes skip."""
    root = Path(root)
    directory = root / 'runtime/jobs' / job['id']
    complete = directory / 'completed.json'
    skipped = directory / 'skipped.json'
    require(not (complete.exists() and skipped.exists()), 'both completed and skipped: ' + job['id'])
    if not complete.exists():
        return None
    receipt = read(complete)
    validate, matches = _validation_api()
    actual = validate(job)
    require(receipt['job_id'] == job['id'] and receipt['exit']['returncode'] == 0,
            'invalid completion identity/exit: ' + job['id'])
    require(matches(receipt, actual), 'completion receipt changed: ' + job['id'])
    return {'summary': actual['summary'], 'actual': actual,
            'input': {'job_id': job['id'], 'receipt_sha256': sha(complete),
                      'summary_sha256': actual['summary_sha256'],
                      'config_sha256': actual['config_sha256']}}


def evaluate(job, suite=None, root=HERE):
    """Return wait/ready/skip. Thresholds and source IDs come only from frozen JSON."""
    root = Path(root)
    disk_suite = read(root / 'suite.json')
    if suite is None:
        suite = disk_suite
    require(suite == disk_suite, 'suite argument differs from current frozen manifest')
    jobs = {j['id']: j for j in suite['jobs']}
    require(len(jobs) == len(suite['jobs']), 'duplicate job IDs')
    require(job['id'] in jobs and jobs[job['id']] == job, 'job not in frozen suite')
    frozen = read(root / 'FROZEN_GATES.json')
    gate = job.get('gate')
    evidence = {'suite_sha256': sha(root / 'suite.json'),
                'frozen_gates_sha256': sha(root / 'FROZEN_GATES.json'), 'inputs': []}
    def result(decision, reason):
        return {'decision': decision, 'gate': gate, 'reason': reason, 'evidence': evidence}
    if gate is None:
        return result('ready', 'Unconditional core or diagnostic job; queue dependencies still apply.')
    definitions = frozen['definitions']
    require(gate in definitions, 'undefined conditional gate: ' + str(gate))
    definition = definitions[gate]
    require(definition['kind'] == gate, 'conditional definition kind mismatch')
    evidence['definition'] = definition
    cache = {}
    def get(jid):
        require(jid in jobs, 'gate references unknown job: ' + jid)
        require(not jobs[jid]['smoke'], 'gate cannot use a smoke result: ' + jid)
        if jid not in cache:
            cache[jid] = validated_record(jobs[jid], root)
            if cache[jid] is not None:
                evidence['inputs'].append(cache[jid]['input'])
        return cache[jid]
    def signal(d):
        aa, bb = get(d['baseline_id']), get(d['candidate_id'])
        if aa is None or bb is None:
            return None
        require(jobs[d['baseline_id']]['kind'] == jobs[d['candidate_id']]['kind'] == 'horizon',
                'signal gate needs full horizon results')
        evidence['core_pairing'] = paired_inputs([jobs[d['baseline_id']], jobs[d['candidate_id']]],
                                                jobs[d['baseline_id']]['attention_backend'])
        a, b = aa['summary']['final_metrics'], bb['summary']['final_metrics']
        delta = {k: number(b[k], k) - number(a[k], k) for k in ('AAA_task_macro', 'FAA_task_macro')}
        passed = (delta['AAA_task_macro'] >= number(d['min_AAA_gain_pp'], 'AAA threshold')
                  and delta['FAA_task_macro'] >= number(d['min_FAA_gain_pp'], 'FAA threshold'))
        evidence['correction_signal'] = {'baseline_id': d['baseline_id'], 'candidate_id': d['candidate_id'],
                                         'AAA_gain_pp': delta['AAA_task_macro'],
                                         'FAA_gain_pp': delta['FAA_task_macro'], 'passed': passed}
        return passed

    if gate in ('correction_signal_with_hvp', 'no_correction_signal'):
        passed = signal(definition)
        if passed is None:
            return result('wait', 'Awaiting validated core baseline/candidate full-horizon results.')
        if gate == 'no_correction_signal':
            return result('skip' if passed else 'ready',
                          'Core correction signal passed; coefficient fallback not selected.' if passed
                          else 'Core correction signal absent at lambda=1; coefficient fallback selected.')
        if not passed:
            return result('skip', 'Core correction signal absent; second-HVP branch not selected.')
        diagnostics = [get(jid) for jid in definition['diagnostic_ids']]
        if any(x is None for x in diagnostics):
            return result('wait', 'Awaiting validated mandatory UG checkpoint diagnostics.')
        require(len(diagnostics) > 0 and len(set(definition['diagnostic_ids'])) == len(diagnostics),
                'empty or duplicate diagnostic gate inputs')
        available = []
        for jid, record in zip(definition['diagnostic_ids'], diagnostics):
            require(jobs[jid]['kind'] == 'curvature', 'HVP gate input is not a curvature diagnostic')
            value = record['summary']['hvp_all_available']
            require(type(value) is bool, 'HVP availability must be an explicit boolean')
            available.append({'job_id': jid, 'hvp_all_available': value})
        evidence['HVP_availability'] = available
        # Finite/point/restore checks are part of validate_job. Alignment and
        # FD/HVP error are observations, deliberately absent from this decision.
        ok = all(x['hvp_all_available'] for x in available)
        return result('ready' if ok else 'skip', 'Correction signal and HVP availability passed.' if ok
                      else 'HVP explicitly unsupported on mandatory UG diagnostics; no substitute training.')

    if gate == 'second_hvp_effect':
        pairs = definition['comparisons']
        require(len(pairs) > 0, 'empty second-HVP effect comparisons')
        comparisons = []
        missing = []
        skipped = []
        for pair in pairs:
            for key in ('fd_id', 'hvp_id'):
                jid = pair[key]
                rec = get(jid)
                if rec is None:
                    sp = root / 'runtime/jobs' / jid / 'skipped.json'
                    if sp.exists():
                        verify_skipped(jobs[jid], read(sp), suite, root)
                        skipped.append({'job_id': jid, 'skipped_receipt_sha256': sha(sp)})
                    else:
                        missing.append(jid)
            fd, hvp = cache[pair['fd_id']], cache[pair['hvp_id']]
            if fd is not None and hvp is not None:
                difference = (number(hvp['summary']['final_metrics']['AAA_task_macro'], 'HVP AAA')
                              - number(fd['summary']['final_metrics']['AAA_task_macro'], 'FD AAA'))
                comparisons.append({**pair, 'HVP_minus_FD_AAA_pp': difference})
        evidence['second_HVP_comparisons'] = comparisons
        if skipped:
            evidence['skipped_inputs'] = skipped
            return result('skip', 'Required second-HVP branch was validly skipped.')
        if missing:
            evidence['pending_inputs'] = missing
            return result('wait', 'Awaiting all four validated FD/HVP natural/matched full trajectories.')
        evidence['second_HVP_pairing'] = paired_inputs(
            [jobs[p[key]] for p in pairs for key in ('fd_id', 'hvp_id')], required_backend='math')
        threshold = number(definition['min_abs_AAA_difference_pp'], 'second-HVP effect threshold')
        require(threshold >= 0, 'negative absolute-effect threshold')
        passed = any(abs(x['HVP_minus_FD_AAA_pp']) >= threshold for x in comparisons)
        return result('ready' if passed else 'skip', 'Second-HVP absolute AAA effect met threshold.' if passed
                      else 'Neither scale met the second-HVP absolute AAA effect threshold.')
    raise ValueError('unsupported frozen gate kind: ' + str(gate))


def verify_skipped(job, receipt, suite=None, root=HERE):
    root = Path(root)
    require(receipt['job_id'] == job['id'] and receipt['status'] in ('skipped', 'skipped_by_gate'),
            'invalid skip identity/status')
    require(receipt['config_sha256'] == job['config_sha256'] == sha(job['config']), 'skipped config changed')
    require(receipt['suite_sha256'] == sha(root / 'suite.json'), 'skipped suite changed')
    require(receipt['frozen_gates_sha256'] == sha(root / 'FROZEN_GATES.json'), 'skipped frozen gates changed')
    current = evaluate(job, suite, root)
    require(current['decision'] == 'skip' and receipt['gate_result'] == current, 'skip evidence is not current')
    return current


def main():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--job')
    selection.add_argument('--group')
    args = parser.parse_args()
    suite = read(HERE / 'suite.json')
    if args.job:
        job = next(j for j in suite['jobs'] if j['id'] == args.job)
    else:
        selected = [j for j in suite['jobs'] if j.get('smoke_group') == args.group]
        require(bool(selected), 'unknown smoke group')
        require(len({j.get('gate') for j in selected}) == 1, 'group contains different gates')
        job = selected[0]
    print(json.dumps(evaluate(job, suite), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
