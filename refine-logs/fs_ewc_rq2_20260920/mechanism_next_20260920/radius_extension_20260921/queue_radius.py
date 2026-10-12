#!/usr/bin/env python3
"""CPU-only plan validation by default; pass --execute to run the shared scheduler."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parents[1] / 'execution'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=HERE / 'campaign/plan.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    module_path = EXECUTION / 'full_rq2_20260920/campaign_runner.py'
    spec = importlib.util.spec_from_file_location('historical_campaign_runner', module_path)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    plan = json.loads(args.plan.read_text())
    jobs = scheduler.validate_plan(plan)
    base = json.loads((EXECUTION / 'diagnostic_config_recovery1.json').read_text())
    config = json.loads((HERE / 'configs/radius_seed0.json').read_text())
    assert config == dict(base, output_dir=str(HERE / 'results'), radii=[0, *range(11, 20)])
    assert plan['gpus'] == [0, 3] and plan['max_parallel'] == 2
    for job in jobs:
        if job['gpu'] is not None:
            assert job['gpu'] == {'sgd': 0, 'gam': 3}[job['id'].split('_')[0]]
            assert not job.get('gpu_pool'), 'Radius jobs must stay on the pinned method GPU'
    fingerprints = scheduler.fingerprint(plan['code_files'] + [str(args.plan.resolve())])
    if not args.execute:
        print(json.dumps(dict(dry_run=True, stages=len(plan['stages']), jobs=len(jobs),
            max_parallel=plan['max_parallel'], gpus=plan['gpus'],
            gpu_uuids=plan['start_after']['gpu_uuids'],
            added_radii=list(range(11, 20)), directions=16,
            new_perturbations_per_method=144, production_rows_per_method=145,
            dependencies={stage['id']: stage['depends_on'] for stage in plan['stages']},
            config_changes=['output_dir', 'radii'], fingerprints=fingerprints), indent=2))
        return
    scheduler.execute_plan(plan, args.plan, args.resume)


if __name__ == '__main__':
    main()
