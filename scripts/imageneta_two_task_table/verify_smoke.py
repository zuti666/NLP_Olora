"""Fail formal deployment unless the real two-task smoke validates its protocol."""
import argparse
import json
import math
from pathlib import Path


def verify(root):
    from train import RULES
    checks = []
    for method, (_, optimizer, protected) in RULES.items():
        run = root / method / 'seed0'
        summary = json.loads((run / 'evaluation/summary.json').read_text())
        if not summary['smoke']:
            raise ValueError('Smoke verifier received a formal experiment.')
        for task in (0, 1):
            path = run / 'trajectory' / f'task_{task:02d}'
            merge = json.loads((path / 'merge.json').read_text())
            fisher = json.loads((path / 'current_fisher.json').read_text())
            if merge['enabled'] and (merge['epochs'] != [1, 2] or not math.isclose(sum(merge['weights']), 1.)):
                raise ValueError(f'Missing/invalid smoke trajectory snapshots: {path}')
            if merge['post_factor_sha256'] != fisher['factor_sha256']:
                raise ValueError('Fisher was not computed at the mixed endpoint.')
            raw_training = fisher['raw_trace']
            raw_offline = summary['B1' if task == 0 else 'B2']
            if not math.isclose(raw_training, raw_offline, rel_tol=5e-3, abs_tol=1e-6):
                raise ValueError(f'Training and offline batch proxy differ: {method}/task{task}: {raw_training}, {raw_offline}')
            identity = json.loads((run / 'checkpoints' / f'experiment_identity_{task}.json').read_text())
            if identity['lr'] != (.04 if task == 0 else .02):
                raise ValueError('Stage learning-rate metadata mismatch.')
        counts = merge['step_counts']
        if counts['ce_backward_count'] != counts['updates'] * (4 if optimizer == 'gam' else 1):
            raise ValueError('Incorrect GAM/SGD backward budget.')
        if protected != (counts['ewc_penalty_max'] > 0):
            raise ValueError('EWC protection was not enabled/disabled as required.')
        checks.append({'variant': method, 'merge_before_F': True, 'proxy_crosscheck': True,
                       'stage_lr': True, 'CE_backward_budget': True, 'EWC_activation': True})
    result = {'status': 'passed', 'scope': 'real ImageNet-A GPU smoke, 3 epochs per task only',
              'paper_evidence_allowed': False, 'checks': checks}
    (root / 'SMOKE_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.output_root.resolve()), indent=2))
