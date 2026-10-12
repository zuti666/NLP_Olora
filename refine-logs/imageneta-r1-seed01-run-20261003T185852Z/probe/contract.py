"""Thin seed-aware contract around the frozen four-probe training implementation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = 'four_probe_exact_restore_rng_replay'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def select_job(suite, identifier):
    selected = [job for job in suite['jobs'] if job['id'] == identifier]
    if len(selected) != 1:
        raise ValueError('Expected one exact suite job: ' + identifier)
    return selected[0]


def check_job(suite, job):
    if job['family'] != FAMILY or job['condition'] not in {'P%02d' % i for i in range(1, 17)}:
        raise ValueError('Only the sixteen approved four-probe conditions are supported')
    if type(job['seed']) is not int or job['seed'] not in (0, 1):
        raise ValueError('This campaign repeats seed 0 and seed 1 only')
    if job['phase'] not in ('smoke', 'formal'):
        raise ValueError('Only smoke or formal phase is supported')
    raw = read_json(job['config'])
    if sha256(job['config']) != job['config_sha256']:
        raise ValueError('Job configuration hash changed')
    required = dict(seed=[job['seed']], dataset='imageneta', class_shuffle=False,
                    init_cls=20, increment=20, nb_tasks=10, total_sessions=10,
                    lora_rank=10, batch_size=128, gam_grad_norm_rho=1., gam_grad_rho=.05,
                    init_epoch=50, epochs=50, init_lr=.04, lrate=.02,
                    cosine_min_lr_ratio=.1, init_momentum=0., momentum=0.,
                    init_weight_decay=0., weight_decay=0., lambda_flat=1.,
                    ewc_lambda=0., ewc_skip_task_fisher=True,
                    approximate_fisher_in_task=False, in_task_weight_merge=False,
                    task_endpoint_weight_merge=False, memory_size=0, memory_per_class=0,
                    gam_adaptive=False, gam_grad_gamma=0.)
    for key, value in required.items():
        if raw.get(key) != value:
            raise ValueError('Approved configuration mismatch: ' + key)
    for key in ('ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher',
                'ewc_dual_ascent', 'compute_full_fisher', 'posterior_state_save',
                'as1_delta_w_penalty', 'fisher_split', 'use_sgv', 'gam_use_sgv',
                'mechanism_eval', 'mechanism_grad_stats'):
        if raw.get(key, False) is not False:
            raise ValueError('Disabled module is enabled: ' + key)
    bindings = job.get('runtime_bindings', {})
    for key in ('probe_branch', 'update_direction', 'step_mode', 'attention_backend'):
        if key in bindings and bindings[key] != raw[key]:
            raise ValueError('Runtime selector/configuration mismatch: ' + key)
    approved = {
        'P01': ('P', 'g0', 'natural'), 'P02': ('P', 'g1_plus', 'natural'),
        'P03': ('P', 'g1_minus', 'natural'), 'P04': ('P', 'g2', 'natural'),
        'P05': ('D', 'g2', 'natural'), 'P06': ('C', 'g2', 'natural'),
        'P07': ('P', 'A', 'natural'), 'P08': ('D', 'A', 'natural'),
        'P09': ('C', 'A', 'natural'), 'P10': ('R', 'g2', 'natural'),
        'P11': ('R', 'A', 'natural'), 'P12': ('P', 'A', 'matched'),
        'P13': ('D', 'A', 'matched'), 'P14': ('C', 'A', 'matched'),
        'P15': ('R', 'A', 'matched'), 'P16': ('P', 'g0', 'matched')}
    actual = tuple(raw[key] for key in ('probe_branch', 'update_direction', 'step_mode'))
    if actual != approved[job['condition']] or raw['attention_backend'] != 'native':
        raise ValueError('Condition selector is outside the approved sixteen cells')
    smoke = job['phase'] == 'smoke'
    budget = (2, 1, 2, 256) if smoke else (10, 50, 2550, 299050)
    if tuple(job[key] for key in ('expected_tasks', 'expected_epochs_per_task',
                                 'expected_steps', 'expected_samples')) != budget:
        raise ValueError('Approved independent training budget changed')
    source = Path(suite['source_root']).resolve()
    if not (source / 'src/trainer.py').is_file():
        raise FileNotFoundError('Frozen source layout is unavailable')
    resolved = dict(job, smoke=smoke, branch=raw['probe_branch'],
                    direction=raw['update_direction'], mode=raw['step_mode'],
                    backend=raw['attention_backend'], outer_radius=1., inner_radius=.05)
    return raw, resolved
