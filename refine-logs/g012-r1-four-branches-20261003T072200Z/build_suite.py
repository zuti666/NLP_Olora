"""Materialize the original Stage1 design at r=1; no GPU launch."""
from pathlib import Path
import copy
import json
import os
import socket
from common import atomic_json, sha256, load_suite, now

H = Path(__file__).resolve().parent
OLD = Path('/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z')
BASELINES = {'S1-01_CE_natural_s2', 'S1-02_Splus_natural_s2',
             'S1-03_Sminus_natural_s2', 'S1-16_CE_matched_s2'}


def main():
    if (H / 'suite.json').exists():
        raise RuntimeError('Refuse to regenerate an existing manifest')
    previous = json.loads((OLD / 'suite.json').read_text())
    original = [j for j in previous['jobs'] if j['kind'] == 'training'
                and not j['smoke'] and j['id'].startswith('S1-')]
    if len(original) != 16 or BASELINES - {j['id'] for j in original}:
        raise RuntimeError('Original Stage1 design is not the expected 16 conditions')
    env = copy.deepcopy(previous['environment'])
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        env[key] = '/tmp/' + H.name
    suite = dict(schema_version=1, run_id=H.name, run_dir=str(H),
                 execution_hostname=socket.gethostname(),
                 execution_pid_namespace='pid:[4026531836]',
                 python=previous['python'], source_root=previous['source_root'],
                 source_manifest=str(H / 'source_manifest.json'),
                 source_manifest_sha256=sha256(H / 'source_manifest.json'),
                 allowed_gpus=[3, 2], max_parallel=2, poll_seconds=15,
                 gpu_lock_dir=previous['gpu_lock_dir'], gpu_free_threshold_mib=500,
                 gpu_free_max_util_percent=4, expected_tasks=10,
                 environment=env, no_external_notifications=True,
                 failure_policy='halt new launches; preserve artifacts; active jobs finish',
                 old_campaign=str(OLD), old_suite_sha256=sha256(OLD / 'suite.json'),
                 scope='Stage1 only: 12 new full10 + 4 radius-independent historical baselines',
                 radius=1., rho=.05, baseline_reuse=str(H / 'REUSE_BASELINES.json'), jobs=[])
    smokes = ['smoke_' + j['id'] for j in original]
    for smoke in (True, False):
        for old_job in original:
            if not smoke and old_job['id'] in BASELINES:
                continue
            jid = ('smoke_' if smoke else '') + old_job['id']
            output = H / 'outputs' / jid
            cfg = json.loads(Path(old_job['config']).read_text())
            cfg.update(prefix=jid, outputs_root=str(output / 'native'), gam_grad_norm_rho=1.)
            path = H / 'configs' / (jid + '.json')
            atomic_json(path, cfg)
            command = [suite['python'], str(H / 'trainer/four_probe_train.py'),
                       '--source', suite['source_root'], '--config', str(path),
                       '--output-dir', str(output)]
            if smoke:
                command += ['--max-tasks', '2', '--max-epochs', '1', '--max-batches', '1']
            suite['jobs'].append(dict(id=jid, kind='training',
                phase='smoke' if smoke else 'formal', original_job_id=old_job['id'],
                branch=old_job['branch'], direction=old_job['direction'],
                mode=old_job['mode'], backend='native', smoke=smoke,
                command=command, config=str(path), config_sha256=sha256(path),
                output_dir=str(output), depends_on=[] if smoke else smokes,
                outer_radius=1., inner_radius=.05,
                expected_tasks=2 if smoke else 10,
                expected_epochs_per_task=1 if smoke else 50,
                expected_steps=2 if smoke else 2550,
                expected_samples=256 if smoke else 299050))
    atomic_json(H / 'suite.json', suite)
    load_suite(H / 'suite.json')
    print(json.dumps(dict(materialized_at=now(), smokes=16, new_formal=12,
                          reused_baselines=4, allowed_gpus=suite['allowed_gpus'])))


if __name__ == '__main__':
    main()
