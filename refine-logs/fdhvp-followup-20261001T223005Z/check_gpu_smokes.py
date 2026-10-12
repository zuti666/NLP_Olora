#!/usr/bin/env python3
"""CPU inspection of completed real-GPU smokes; group-specific release proofs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from common import atomic_json, now
from conditional_gate import read, sha, require, number, evaluate, validated_record, paired_inputs

HERE = Path(__file__).resolve().parent


def pairing(job):
    out = Path(job['output_dir'])
    identities = read(out / 'task_identities.json')
    trace = [json.loads(line) for line in (out / 'step_trace.jsonl').read_text().splitlines()]
    require(len(identities) == job['expected_tasks'] and bool(trace), 'missing smoke pairing records')
    return {'task0_initial': identities[0]['task_initial_parameter_sha256'],
            'frozen_origins': [v['frozen_origin_sha256'] for v in identities],
            'new_head_rows': [v['new_fc_rows_sha256'] for v in identities],
            'class_slices': [v['class_slice'] for v in identities],
            'data': [v['identity'] for v in trace],
            'nominal_lr': [v['lr_nominal'] for v in trace]}


def pairing_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def parameters(job):
    import torch
    cp = Path(job['output_dir']) / 'checkpoints' / ('task_%03d' % (job['expected_tasks'] - 1))
    task = job['expected_tasks'] - 1
    aa = torch.load(cp / f'lora_w_a_{task}.pt', map_location='cpu', weights_only=False)
    bb = torch.load(cp / f'lora_w_b_{task}.pt', map_location='cpu', weights_only=False)
    fc = torch.load(cp / f'fc_state_{task}.pt', map_location='cpu', weights_only=False)
    require(len(aa) == len(bb) and len(aa) > 0, 'invalid LoRA checkpoint pair counts')
    result = []
    for i, (a, b) in enumerate(zip(aa, bb)):
        for side, module in (('A', a), ('B', b)):
            for name, tensor in sorted(module.named_parameters()):
                result.append((f'{i}/{side}/{name}', tensor.detach().double()))
    result += [('fc/' + key, tensor.detach().double()) for key, tensor in sorted(fc.items())]
    require(bool(result) and all(torch.isfinite(t).all().item() for _, t in result), 'nonfinite/empty smoke tensors')
    return result


def compare_parameters(a, b, tolerance):
    require(len(a) == len(b) and all(ka == kb and x.shape == y.shape
                                   for (ka, x), (kb, y) in zip(a, b)), 'US checkpoint structures differ')
    require(bool(a), 'empty US parameter comparison')
    diff = sum(float(((x - y) ** 2).sum()) for (_, x), (_, y) in zip(a, b)) ** .5
    norm = sum(float((x * x).sum()) for _, x in a) ** .5
    relative = diff / max(norm, 1e-30)
    maximum = max(float((x - y).abs().max()) for (_, x), (_, y) in zip(a, b))
    require(math.isfinite(relative) and math.isfinite(maximum) and relative <= tolerance,
            'US natural/matched checkpoint numerical gate failed')
    return {'parameter_relative_l2': relative, 'parameter_max_abs': maximum,
            'relative_tolerance': tolerance, 'bitwise_cross_run_required': False}


def check_group(group, root=HERE):
    root = Path(root)
    suite = read(root / 'suite.json')
    sys.path.insert(0, suite['source_root'])  # Native checkpoint class unpickling, CPU only.
    jobs = [j for j in suite['jobs'] if j['smoke'] and j['smoke_group'] == group]
    require(bool(jobs), 'unknown or empty smoke group: ' + group)
    require(len({j.get('gate') for j in jobs}) == 1, 'group mixes conditional gates')
    gate = evaluate(jobs[0], suite, root)
    require(gate['decision'] == 'ready', 'smoke group not scientifically/technically released')
    frozen = read(root / 'FROZEN_GATES.json')
    reference = None
    receipt_hashes = {}
    for job in jobs:
        checked = validated_record(job, root)
        require(checked is not None, 'smoke completion absent: ' + job['id'])
        require(job['expected_tasks'] == 2, 'smoke must cross Task0-to-Task1')
        current = pairing(job)
        if reference is None:
            reference = current
        require(current == reference, 'smoke initialization/data/new-head/LR pairing differs: ' + job['id'])
        receipt_hashes[job['id']] = checked['input']['receipt_sha256']
    observed_pairing = paired_inputs(jobs, required_backend=jobs[0]['attention_backend'])
    base_binding = None
    if group != 'base':
        base_path = root / 'runtime/GPU_SMOKES_base_VERIFIED.json'
        proof = read(base_path)
        base_jobs = [j for j in suite['jobs'] if j['smoke'] and j['smoke_group'] == 'base']
        require(proof['status'] == 'passed' and proof['group'] == 'base'
                and proof['suite_sha256'] == sha(root / 'suite.json')
                and proof['frozen_gates_sha256'] == sha(root / 'FROZEN_GATES.json')
                and set(proof['receipts_sha256']) == {j['id'] for j in base_jobs}, 'base smoke proof stale')
        for job in base_jobs:
            checked = validated_record(job, root)
            require(checked is not None and checked['input']['receipt_sha256'] == proof['receipts_sha256'][job['id']],
                    'base smoke receipt changed')
            require(pairing(job) == reference, 'conditional smoke differs from base exogenous pairing')
        require(proof['pairing_sha256'] == pairing_sha(reference), 'base pairing proof differs')
        paired_inputs([base_jobs[0], jobs[0]])
        base_binding = {'proof_sha256': sha(base_path), 'receipts_sha256': proof['receipts_sha256']}
    identities = [j for j in jobs if j['rule'] == 'us' and j['lambda_corr'] == 1.]
    identity_check = None
    if identities:
        require(len(identities) == 2 and {j['scale'] for j in identities} == {'natural', 'matched'},
                'US lambda1 smoke identity pair incomplete')
        natural = next(j for j in identities if j['scale'] == 'natural')
        matched = next(j for j in identities if j['scale'] == 'matched')
        tolerance = number(frozen['US_smoke_relative_parameter_tolerance'], 'US smoke tolerance')
        require(tolerance >= 0, 'negative US smoke tolerance')
        identity_check = compare_parameters(parameters(natural), parameters(matched), tolerance)
        identity_check['jobs'] = [natural['id'], matched['id']]
        identity_check['per_step_scalar_and_own_counterfactual_exact'] = True
    require(group != 'base' or identity_check is not None, 'base lacks required US lambda1 identity control')
    result = {'status': 'passed', 'group': group, 'at': now(), 'verified_smokes': len(jobs),
              'suite_sha256': sha(root / 'suite.json'), 'frozen_gates_sha256': sha(root / 'FROZEN_GATES.json'),
              'receipts_sha256': receipt_hashes, 'pairing_sha256': pairing_sha(reference),
              'all_initial_model_data_head_pairs': True, 'gate_result': gate,
              'base_group_binding': base_binding, 'US_lambda1_identity': identity_check,
              'observed_input_pairing': observed_pairing,
              'different_rule_endpoint_equality_required': False}
    atomic_json(root / f'runtime/GPU_SMOKES_{group}_VERIFIED.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True)
    print(json.dumps(check_group(parser.parse_args().group), indent=2, allow_nan=False))
