"""Build an isolated six-anchor queue; this script never starts GPU work."""
from pathlib import Path
import importlib.util
import json
import hashlib
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
PYTHON = '/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python'
ARMS = [('probe005_norm1', .05, 1.), ('probe1_norm1', 1., 1.)]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('new_radius_trainer', HERE / 'train.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def main():
    campaign = HERE / 'training_campaign'
    if (campaign / 'pipeline_state.json').exists():
        raise RuntimeError('Never regenerate a launched/frozen plan.')
    campaign.mkdir(parents=True, exist_ok=True)
    source = json.loads((EXECUTION / 'source_frozen.json').read_text())
    code = [EXECUTION / 'source' / f['path'] for f in source['files']]
    code += [EXECUTION / 'source_frozen.json', BASE / 'train.py', BASE / 'campaign_runner.py',
             HERE / 'train.py', HERE / 'prepare_training.py', HERE / 'RUN_TRAINING.md',
             HERE / 'launch_training.py',
             BASE / 'stage_path_completion_20260921/env_witness.py']
    jobs, changes = [], []
    for arm, probe, norm in ARMS:
        for seed in range(3):
            name = f'{arm}_seed{seed}'
            argv = ['--mode', 'anchor', '--method', 'gam', '--seed', str(seed),
                    '--run-name', name, '--probe-radius', str(probe), '--norm-radius', str(norm),
                    '--workers', '4']
            ns = trainer.parser().parse_args(argv)
            cfg, run_dir = trainer.build_config(ns)
            old_manifest = BASE / f'formal_training/anchor_gam_seed{seed}/run_manifest.json'
            old_cfg = json.loads(old_manifest.read_text())['requested_config']
            ignored = {'prefix', 'outputs_root', 'save_task_checkpoint_dir', 'optimizer_tag_override',
                       'rq2_radius_intervention'}
            different = {k: {'old': old_cfg.get(k), 'new': cfg.get(k)}
                         for k in set(old_cfg) | set(cfg)
                         if k not in ignored and old_cfg.get(k) != cfg.get(k)}
            expected = {'gam_grad_norm_rho', 'as1_norm_rho'}
            if probe != .05:
                expected |= {'gam_grad_rho', 'as1_rho'}
            assert set(different) == expected, different
            cfg_path = HERE / 'training_configs' / f'{name}.json'
            cfg_path.parent.mkdir(exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
            code += [cfg_path, old_manifest]
            changes.append(dict(arm=arm, seed=seed, changes=different,
                                same_other_scientific_config=True, previous_manifest=str(old_manifest)))
            jobs.append(dict(id=name, gpu=0, gpu_pool=[0, 1], cwd=str(HERE),
                             command=[PYTHON, '-u', str(HERE / 'train.py'), *argv, '--gpu', '{gpu}'],
                             env={'TMPDIR': '/tmp', 'TMP': '/tmp', 'TEMP': '/tmp'},
                             preflight_command=[PYTHON, str(BASE / 'stage_path_completion_20260921/env_witness.py'),
                                                '--output', str(campaign / f'{name}_env.json')],
                             receipt=str(run_dir / 'COMPLETE.json'),
                             receipt_require={'status': 'complete', 'completed_tasks': 1}))
    plan = dict(schema='rq2_two_training_radii_v1', gpus=[0, 1], max_parallel=2,
                run_dir=str(campaign), gpu_lock_dir='/data/140-1/users/liying/O-LoRA/.cache/gpu_locks',
                start_after={'gpu_uuids': {'0': 'GPU-fd7e2955-3733-e30a-dff1-5da992423839',
                                          '1': 'GPU-d59d54d6-e68a-ffb7-4b2b-5e8d69d4b78a'}},
                code_files=sorted({str(p.resolve()) for p in code}),
                stages=[dict(id='train_task0_anchors', depends_on=[], jobs=jobs)])
    (campaign / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (HERE / 'CONFIG_DIFFERENCES.json').write_text(json.dumps(changes, indent=2) + '\n')
    print(json.dumps(dict(status='prepared', jobs=len(jobs), gpus=plan['gpus'],
                         config_differences='only requested radii and output/provenance fields')))


if __name__ == '__main__':
    main()
