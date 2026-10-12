"""CPU plan validation, predecessor gate and shared GPU 0/3 queue entrypoint."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import campaign_runner as scheduler


def read(path):
    return json.loads(Path(path).read_text())


def wait_prior():
    spec = read(HERE / 'extension_spec.json')
    prior = Path(spec['prior_pipeline_state'])
    while True:
        state = read(prior)
        if state['status'] == 'complete':
            assert all(r['status'] == 'complete' for r in state['jobs'].values())
            for row in state['jobs'].values():
                receipt = Path(row['receipt'])
                assert hashlib.sha256(receipt.read_bytes()).hexdigest() == row['receipt_sha256']
            scheduler.write_json(HERE / 'campaign/prior_complete.json', dict(status='complete',
                prior_state=str(prior), prior_state_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                completed_jobs=len(state['jobs']), checked_utc=scheduler.now()))
            return
        if state['status'] not in ('running', 'waiting_for_gpu'):
            raise RuntimeError(f'Prior queue has not succeeded: {state["status"]}; review before proceeding')
        identity = state.get('process_identity')
        if not identity or scheduler.process_identity(identity['pid']) != identity:
            raise RuntimeError('Prior controller is not live; do not silently bypass its earlier queue position')
        scheduler.write_json(HERE / 'campaign/waiting_for_prior.json', dict(status='waiting',
            checked_utc=scheduler.now(), prior_state=str(prior), prior_controller=identity,
            prior_jobs={k:v['status'] for k,v in state['jobs'].items()}))
        time.sleep(15)


def validate(plan):
    jobs = scheduler.validate_plan(plan)
    spec = read(HERE / 'extension_spec.json')
    assert plan['gpus'] == spec['gpus'] == [0, 3] and plan['max_parallel'] == 2
    expected = {(s,m,t) for s in range(3) for m in ('sgd','gam') for t in spec['new_path_tasks']}
    found = set()
    for job in jobs:
        if job['gpu'] is None:
            continue
        assert not job.get('gpu_pool')
        command = job['command']; method = command[command.index('--method')+1]
        assert job['gpu'] == {'sgd':0,'gam':3}[method]
        config_path = Path(command[command.index('--config')+1])
        config = read(config_path); seed = config['training_seed']
        reference = read(BASE / f'configs/seed{seed}.json')
        assert config['output_dir'] == str(HERE / f'results/seed{seed}')
        job_kind = command[command.index('--job')+1]
        allowed = {'output_dir'} if job_kind == 'path' else {'output_dir','sparse_train_radii'}
        assert {k for k in config.keys() | reference.keys() if config.get(k) != reference.get(k)} == allowed
        if job_kind == 'path':
            t = int(command[command.index('--path-task')+1]); found.add((seed,method,t))
            assert t in spec['new_path_tasks']
            assert not (BASE / f'formal_results/seed{seed}/{method}/path_t{t:02d}/summary.json').exists(), 'Already measured; avoid duplicate jobs'
        else:
            assert config['sparse_train_radii'] == [0,*range(11,20)]
        for path in config['checkpoints'][method].values():
            assert Path(path).is_file(), path
    assert found == expected and len(found) == 36
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--execute', action='store_true')
    group.add_argument('--wait-prior', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.wait_prior:
        wait_prior(); return
    path = HERE / 'campaign/plan.json'; plan = read(path)
    jobs = validate(plan)
    if not args.execute:
        print(json.dumps(dict(status='passed', dry_run=True, jobs=len(jobs),
            new_path_jobs=36, new_radius_jobs=sum('radius_' in j['id'] for j in jobs if j['gpu'] is not None),
            reused_paths=18, final_paths=54, gpus=[0,3], max_parallel=2,
            prior_pipeline=read(HERE/'extension_spec.json')['prior_pipeline_state'],
            fingerprints=scheduler.fingerprint(plan['code_files']+[str(path)])), indent=2))
        return
    scheduler.execute_plan(plan, path, args.resume)


if __name__ == '__main__':
    main()
