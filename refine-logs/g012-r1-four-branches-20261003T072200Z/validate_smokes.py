"""Release formal runs after all actual r=1 probe/step tests pass."""
import json
import math
from pathlib import Path
from common import HERE, atomic_json, read_json, sha256, now
from validate_job import validate_job


def validate_smokes(suite):
    jobs = [j for j in suite['jobs'] if j['smoke']]
    if len(jobs) != 16:
        raise RuntimeError('All sixteen original conditions require a short GPU run')
    baseline = None
    proofs = {}
    maximum_error = 0.
    for job in jobs:
        checked = validate_job(job)
        run = HERE / 'runtime/jobs' / job['id']
        receipt = read_json(run / 'completed.json')
        if receipt['summary_sha256'] != checked['summary_sha256']:
            raise RuntimeError('GPU smoke receipt changed')
        output = Path(job['output_dir'])
        summary = read_json(output / 'final_summary.json')
        rows = [json.loads(line) for line in (output / 'step_trace.jsonl').read_text().splitlines()]
        initial = summary['initial_identity']
        identity = {'batch_identities': [r['identity'] for r in rows],
                    'coordinates': [(r['task'], r['epoch'], r['batch']) for r in rows],
                    'nominal_lrs': [r['nominal_lr'] for r in rows],
                    'task0_initial_parameter_sha256': initial['task_initial_parameter_sha256'],
                    'initial_fc_rows': initial['new_fc_rows_sha256'],
                    'frozen_origin_sha256': initial['frozen_origin_sha256']}
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise RuntimeError('Smoke initialization/data pairing failed: ' + job['id'])
        for row in rows:
            # r is the global trainable-factor/head perturbation norm, not the BA norm.
            distance = row['probe_displacements']['phi2']['parameter_norm']
            if row['norms']['q'] > 1e-10 and not math.isclose(distance, 1., rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError('Actual outer point is not one unit from the base')
            if row['forward_rng_rule'] != 'R0 replay for every probe; final RNG=R1 from g0':
                raise RuntimeError('Probe RNG contract differs')
            if job['mode'] == 'matched':
                maximum_error = max(maximum_error, row['actual_match_relative_error'])
        proofs[job['id']] = checked
    result = dict(status='passed', checked_at=now(), smoke_conditions=16,
                  actual_updates=32, paired_initial_parameters_and_batch_identities=True,
                  actual_outer_radius=1., rho=.05, no_g3_no_projection=True,
                  matching_relative_error_max=maximum_error, tolerance=1e-3,
                  scope='execution/configuration/pairing; no strict cross-GPU gradient equality claim',
                  suite_sha256=sha256(HERE / 'suite.json'), proofs=proofs)
    atomic_json(HERE / 'gates/R1_GPU_SMOKE_PAIRING.json', result)
    return result
