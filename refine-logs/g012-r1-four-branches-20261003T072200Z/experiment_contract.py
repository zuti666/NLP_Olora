"""The only scientific intervention relative to completed Stage1 is outer r."""
import json
from pathlib import Path
from common import HERE, read_json, sha256, verify_source

BASELINES = {'S1-01_CE_natural_s2', 'S1-02_Splus_natural_s2',
             'S1-03_Sminus_natural_s2', 'S1-16_CE_matched_s2'}


def verify_reuse(suite):
    report = read_json(suite['baseline_reuse'])
    if not report['passed'] or {j['id'] for j in report['jobs']} != BASELINES:
        raise RuntimeError('Four validated radius-independent baselines are required')
    cpu = read_json(HERE / 'gates/R1_CPU_ACCEPTANCE.json')
    if cpu['status'] != 'passed' or not cpu['baseline_radius_invariance_passed']:
        raise RuntimeError('CPU radius-invariance proof is required for reuse')
    for name, digest in cpu['trainer_sha256'].items():
        if sha256(HERE / 'trainer' / name) != digest:
            raise RuntimeError('CPU proof implementation binding changed: ' + name)
    for row in report['jobs']:
        for key in ('config', 'completed', 'summary', 'trace', 'matrix'):
            if sha256(row[key]) != row[key + '_sha256']:
                raise RuntimeError('Historical baseline evidence changed: ' + row['id'] + '/' + key)
    return report


def verify_contract(suite):
    if (suite['allowed_gpus'] != [3, 2] or suite['max_parallel'] != 2
            or suite['radius'] != 1. or suite['rho'] != .05):
        raise RuntimeError('GPU or radius policy differs from the declared campaign')
    old_path = Path(suite['old_campaign']) / 'suite.json'
    if sha256(old_path) != suite['old_suite_sha256']:
        raise RuntimeError('Original experiment design changed')
    old = read_json(old_path)
    original = {j['id']: j for j in old['jobs'] if j['kind'] == 'training'
                and not j['smoke'] and j['id'].startswith('S1-')}
    if len(original) != 16:
        raise RuntimeError('Unexpected original Stage1 scope')
    formals = [j for j in suite['jobs'] if not j['smoke']]
    smokes = [j for j in suite['jobs'] if j['smoke']]
    if (len(formals) != 12 or len(smokes) != 16
            or {j['original_job_id'] for j in formals} != set(original) - BASELINES
            or {j['original_job_id'] for j in smokes} != set(original)):
        raise RuntimeError('Scope must be 16 smokes and 12 new full10 cells')
    for job in suite['jobs']:
        old_job = original[job['original_job_id']]
        cfg = read_json(job['config'])
        if sha256(job['config']) != job['config_sha256']:
            raise RuntimeError('Config identity changed')
        reference = read_json(old_job['config'])
        reference.update(prefix=job['id'], outputs_root=str(Path(job['output_dir']) / 'native'),
                         gam_grad_norm_rho=1.)
        if cfg != reference:
            raise RuntimeError('More than the outer radius/provenance changed: ' + job['id'])
        if (job['branch'], job['direction'], job['mode'], job['backend']) != (
                old_job['branch'], old_job['direction'], old_job['mode'], 'native'):
            raise RuntimeError('Direction or probe condition changed')
        if job['kind'] != 'training' or job['direction'] not in ('g0', 'g1_plus', 'g1_minus', 'g2', 'A'):
            raise RuntimeError('Unexpected diagnostics/HVP/g3 task')
        expected = (2, 1, 2, 256) if job['smoke'] else (10, 50, 2550, 299050)
        if tuple(job[k] for k in ('expected_tasks', 'expected_epochs_per_task', 'expected_steps', 'expected_samples')) != expected:
            raise RuntimeError('Training budget changed')
        if job['depends_on'] != ([] if job['smoke'] else [j['id'] for j in smokes]):
            raise RuntimeError('Formal runs must wait for all 16 smokes')
    verify_reuse(suite)
    verify_source(suite)
    return True
