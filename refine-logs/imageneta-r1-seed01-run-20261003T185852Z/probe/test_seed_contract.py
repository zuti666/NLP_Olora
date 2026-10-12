"""CPU checks for seed dispatch and unchanged four-probe scientific helpers."""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from contract import FAMILY, HERE, check_job, read_json, sha256

sys.path.insert(0, str(HERE / 'trainer'))
from four_probe_train import resolve_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    suite = read_json(args.suite)
    jobs = [job for job in suite['jobs'] if job['family'] == FAMILY]
    expected = {(phase, seed, 'P%02d' % i) for phase in ('smoke', 'formal')
                for seed in (0, 1) for i in range(1, 17)}
    assert len(jobs) == 64
    assert {(j['phase'], j['seed'], j['condition']) for j in jobs} == expected
    receipts = []
    rejected_wrong_seed = rejected_wrong_budget = 0
    for job in jobs:
        raw, resolved = check_job(suite, job)
        cli = SimpleNamespace(max_tasks=2 if resolved['smoke'] else 0,
                              max_epochs=1 if resolved['smoke'] else 0,
                              max_batches=1 if resolved['smoke'] else 0,
                              output_dir=job['output_dir'])
        cfg = resolve_config(raw, cli)
        assert cfg['seed'] == [job['seed']]
        assert cfg['max_train_tasks'] == (2 if resolved['smoke'] else 10)
        assert cfg['init_epoch'] == cfg['epochs'] == 50
        assert cfg['init_lr'] == .04 and cfg['lrate'] == .02 and cfg['cosine_min_lr_ratio'] == .1
        assert cfg['in_task_path_normalize_snapshots'] is False
        assert cfg['ewc_skip_task_fisher'] is True and cfg['ewc_lambda'] == 0.
        assert cfg['attention_backend'] == 'native'
        assert all(cfg[key] == raw[key] for key in ('probe_branch', 'update_direction', 'step_mode'))
        for mutation in ('seed', 'expected_steps'):
            wrong = copy.deepcopy(job)
            wrong[mutation] = 1 - job['seed'] if mutation == 'seed' else job['expected_steps'] + 1
            try:
                check_job(suite, wrong)
            except ValueError:
                if mutation == 'seed': rejected_wrong_seed += 1
                else: rejected_wrong_budget += 1
            else:
                raise AssertionError('Contract accepted a wrong ' + mutation)
        receipts.append(dict(id=job['id'], seed=cfg['seed'][0], phase=job['phase'],
                             tasks=cfg['max_train_tasks'], scheduler_T_max=cfg['epochs'],
                             branch=cfg['probe_branch'], direction=cfg['update_direction'],
                             mode=cfg['step_mode']))
    original = HERE.parent.parent / 'g012-r1-four-branches-20261003T072200Z' / 'trainer'
    helpers = ('direction_step.py', 'effective_step.py', 'native_lifecycle.py',
               'task0_trainer_base_v2.py', 'audit_core.py', 'response_hvp.py', 'cuda_witness.py')
    unchanged = {name: sha256(HERE / 'trainer' / name) == sha256(original / name) for name in helpers}
    assert all(unchanged.values())
    old = (original / 'four_probe_train.py').read_text()
    expected_trainer = old.replace('epochs=cli.max_epochs or 50,seed=2,',
                                   'epochs=cli.max_epochs or 50,seed=cfg["seed"][0],').replace(
                                   'schema_version=1,seed=2,protocol=',
                                   'schema_version=1,seed=cfg["seed"][0],protocol=')
    assert (HERE / 'trainer/four_probe_train.py').read_text() == expected_trainer
    report = dict(status='passed', passed=True, jobs_checked=len(jobs), seeds=[0, 1],
                  wrong_seed_rejections=rejected_wrong_seed, wrong_budget_rejections=rejected_wrong_budget,
                  unchanged_scientific_helpers=unchanged,
                  trainer_changes_are_only_actual_seed_metadata=True,
                  seed2_base_validator_surrogate_preserved=True,
                  actual_GPU_training_not_exercised=True, receipts=receipts)
    path = Path(args.output)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'receipts'}))


if __name__ == '__main__':
    main()
