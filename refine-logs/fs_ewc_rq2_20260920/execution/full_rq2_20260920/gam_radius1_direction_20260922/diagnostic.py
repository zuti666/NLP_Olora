#!/usr/bin/env python3
"""Fixed Task0 ascent / Task1 descent rays at newly trained GAM anchors.

Only the configured new Task0 checkpoint is loaded. Historical Task1 RNG
receipts supply the paired, untrained classifier, never trained Task1 weights.
Each of 11 radii independently restores the new anchor before measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EXECUTION = BASE.parent
RADII = [i / 2 for i in range(11)]
ARMS = {'probe005_norm1': (.05, 1.), 'probe1_norm1': (1., 1.)}
JOBS = {0: 'task0_positive', 1: 'task1_negative'}
HELPERS = {
    'positive': ('fixed_gradient_extension_20260921/diagnostic_fixed_gradient.py',
                 '54135942d552bf17676d41014f51d6f2045f6d7cd64c952c304539b3b54b9c7c'),
    'negative': ('task1_gradient_extension_20260921/diagnostic_task1_gradient.py',
                 '72245b04065a67f0dd5a418f8fd872d3a555703c93f79cb4554ccd8cc61c118d'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load_helper(name):
    relative, expected = HELPERS[name]
    path = BASE / relative
    require(sha(path) == expected, f'Frozen gradient helper changed: {path}')
    spec = importlib.util.spec_from_file_location('_gam_intervention_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


positive = load_helper('positive')
negative = load_helper('negative')
close = positive.close
WITNESS_STEPS = positive.WITNESS_STEPS


def helper_for(task):
    require(task in JOBS, 'Only gradient tasks 0 and 1 are supported')
    return positive if task == 0 else negative


def validate_configuration(config):
    seed, arm = config['training_seed'], config['intervention_arm']
    require(type(seed) is int and seed in range(3), 'Only formal seeds 0, 1, 2 are supported')
    require(arm in ARMS, 'Unknown intervention arm')
    require((config['training_probe_radius'], config['training_norm_radius']) == ARMS[arm], 'Arm/radius mismatch')
    require(config['radii'] == RADII, 'Explicit 11-point diagnostic radius grid changed')
    formal = read(BASE / f'configs/seed{seed}.json')
    for key in ('source_path', 'data_root', 'workers', 'fisher_samples', 'fisher_batch_size',
                'eval_batch_size', 'subset_seed'):
        require(config[key] == formal[key], f'Formal diagnostic setting changed: {key}')
    require(config['fisher_samples'] == 256 and config['fisher_batch_size'] == 8 and config['eval_batch_size'] == 64, 'Fisher/evaluation settings changed')
    expected_dir = HERE / f'training/{arm}_seed{seed}'
    checkpoint = Path(config['checkpoints']['gam']['0']).resolve()
    require(checkpoint == expected_dir / 'checkpoints/task_00.pt', 'Diagnostic must load the new arm/seed Task0 anchor')
    require(Path(config['training_receipt']).resolve() == expected_dir / 'COMPLETE.json', 'Wrong new training receipt')
    require(Path(config['output_dir']).resolve() == HERE / f'results/{arm}/seed{seed}', 'Output directory must isolate arm and seed')
    return expected_dir, checkpoint


def validate_training_manifest(config, manifest):
    require(manifest['seed'] == config['training_seed'] and manifest['anchor_method'] == 'gam' and manifest['mode'] == 'anchor', 'Wrong new anchor identity')
    require(manifest.get('parent') is None, 'New anchor must be a cold start')
    original = read(BASE / f'formal_training/anchor_gam_seed{config["training_seed"]}/run_manifest.json')
    allowed = {'gam_grad_rho', 'gam_grad_norm_rho', 'as1_rho', 'as1_norm_rho',
               'outputs_root', 'save_task_checkpoint_dir', 'optimizer_tag_override', 'prefix', 'rq2_radius_intervention'}
    for field in ('config', 'requested_config'):
        trained = manifest[field]
        require(set(trained) == set(original[field]) | {'rq2_radius_intervention'}, f'Training configuration keys changed: {field}')
        for key in trained:
            if key not in allowed:
                require(trained[key] == original[field][key], f'Non-radius training setting changed: {field}/{key}')
        for key in ('gam_grad_rho', 'as1_rho'):
            require(trained[key] == config['training_probe_radius'], f'Probe radius/alias mismatch: {key}')
        for key in ('gam_grad_norm_rho', 'as1_norm_rho'):
            require(trained[key] == config['training_norm_radius'], f'Norm radius/alias mismatch: {key}')
        require(trained['max_train_tasks'] == 1 and trained['init_epoch'] == 50, 'Task0 training schedule changed')
        intervention = trained['rq2_radius_intervention']
        require(intervention['arm'] == config['intervention_arm'] and
                intervention['probe_radius_after'] == config['training_probe_radius'] and
                intervention['norm_radius_after'] == config['training_norm_radius'], 'Training intervention metadata mismatch')
        require(intervention['wrapper_sha256'] == sha(HERE / 'train.py') and
                intervention['formal_runner_sha256'] == sha(BASE / 'train.py'), 'Training wrapper/formal runner changed')
    require(manifest['source_manifest_sha256'] == sha(EXECUTION / 'source_frozen.json'), 'Frozen training sources changed')
    require(manifest['matmul_allow_tf32'] is False and manifest['cudnn_allow_tf32'] is False, 'Training precision changed')


def validate_training(config, allow_pending=False):
    directory, checkpoint = validate_configuration(config)
    receipt_path, manifest_path = directory / 'COMPLETE.json', directory / 'run_manifest.json'
    if manifest_path.is_file():
        validate_training_manifest(config, read(manifest_path))
    if not all(path.is_file() for path in (receipt_path, manifest_path, checkpoint)):
        require(allow_pending, 'New anchor training is not complete')
        return dict(status='pending', expected_checkpoint=str(checkpoint), expected_receipt=str(receipt_path),
                    running_manifest_checked=manifest_path.is_file())
    receipt, manifest = read(receipt_path), read(manifest_path)
    require(receipt['status'] == 'complete' and receipt['completed_tasks'] == 1, 'New Task0 training not complete')
    require(receipt['identity_sha256'] == manifest['identity_sha256'], 'Training receipt identity mismatch')
    checkpoint_hash = sha(checkpoint)
    require(receipt['validation']['checkpoint_assets']['checkpoints/task_00.pt']['sha256'] == checkpoint_hash, 'New checkpoint differs from training receipt')
    old_receipt = read(BASE / f'formal_training/anchor_gam_seed{config["training_seed"]}/COMPLETE.json')
    require(checkpoint_hash != old_receipt['validation']['checkpoint_assets']['checkpoints/task_00.pt']['sha256'], 'Historical GAM anchor reused as new anchor')
    for name, field in (('rng_receipts.jsonl', 'rng_receipts_sha256'), ('runtime_receipts.jsonl', 'runtime_receipts_sha256')):
        require(sha(directory / name) == receipt['validation'][field], f'Training {name} changed')
    def initial_head(path):
        heads = [row for row in (json.loads(line) for line in Path(path).read_text().splitlines())
                 if row.get('event') == 'head_initialized' and row.get('task') == 0]
        require(len(heads) == 1, 'Missing/duplicate Task0 initial-head receipt')
        return heads[0]
    original_rng = BASE / f'formal_training/anchor_gam_seed{config["training_seed"]}/rng_receipts.jsonl'
    require(initial_head(directory / 'rng_receipts.jsonl') == initial_head(original_rng), 'Paired fresh Task0 head initialization changed')
    def training_iterations(path):
        return [row for row in (json.loads(line) for line in Path(path).read_text().splitlines())
                if row.get('event') == 'loader_iteration' and row.get('task') == 0 and
                row.get('shuffle') is True and row.get('loader_serial') == 0 and row['iteration'] < 50]
    paired_iterations = training_iterations(directory / 'rng_receipts.jsonl')
    require(len(paired_iterations) == 50 and paired_iterations == training_iterations(original_rng),
            'The 50 paired training loader iterations differ from the formal seed')
    return dict(status='passed', intervention_arm=config['intervention_arm'],
                training_probe_radius=config['training_probe_radius'], training_norm_radius=config['training_norm_radius'],
                checkpoint=file_record(checkpoint), receipt=file_record(receipt_path),
                manifest=file_record(manifest_path), identity_sha256=manifest['identity_sha256'],
                fresh_task0_training=True, only_training_radii_changed=True,
                paired_initial_head_and_50_loader_iterations=True,
                paired_rng_reference=file_record(original_rng), new_rng_receipt=file_record(directory / 'rng_receipts.jsonl'))


def preflight(args):
    config_path = Path(args.config).resolve()
    config = read(config_path)
    task = args.gradient_task
    require(getattr(args, 'method', 'gam') == 'gam', 'Only new GAM anchors are scanned')
    helper_for(task)
    training = validate_training(config, allow_pending=getattr(args, 'allow_pending_training', False))
    reference_path = BASE / f'formal_results/seed{config["training_seed"]}/gam/sparse_train/manifest.json'
    reference = read(reference_path)
    require(reference['counts'] == {'train': 1259, 'test': 307, 'subset': 256}, 'Formal dataset size changed')
    for field, path in (('diagnostic_code_sha256', BASE / 'diagnostic_full.py'),
                        ('common_code_sha256', EXECUTION / 'rq2_common.py'),
                        ('pilot_runner_code_sha256', EXECUTION / 'diagnostic.py')):
        require(sha(path) == reference[field], f'Canonical diagnostic code changed: {path}')
    task1_formal = negative.validate_task1_formal(config['training_seed'], 'gam', config) if task == 1 else None
    return config, dict(status='passed' if training['status'] == 'passed' else 'pending_training',
        intervention_arm=config['intervention_arm'], seed=config['training_seed'], method='gam',
        job=JOBS[task], gradient_task=task, evaluation_task=0, direction_sign=1 if task == 0 else -1,
        gradient_samples=1259 if task == 0 else 997, fisher_samples=256,
        radii=RADII, expected_states=11, training=training, task1_formal=task1_formal,
        task1_provenance_scope='Historical paired untrained head/data initialization only; effective anchor is the newly trained Task0 checkpoint',
        config=file_record(config_path), wrapper=file_record(__file__),
        helper_sources=[file_record(BASE / relative) for relative, _ in HELPERS.values()],
        formal_data_reference=file_record(reference_path),
        trained_checkpoint_weight_reads=[config['checkpoints']['gam']['0']])


def validate_loaded_identity(runner, evidence):
    reference = read(evidence['formal_data_reference']['path'])
    for key in ('counts', 'coordinate_order', 'coordinate_numel', 'coordinate_sha256',
                'data_manifest_sha256', 'source_files', 'source_path', 'production_precision', 'fisher_definition'):
        require(runner.meta[key] == reference[key], f'Formal measurement identity changed: {key}')
    expected = evidence['training']['checkpoint']
    require(runner.meta['checkpoints'] == {'gam': {'0': expected['path']}}, 'Loader read checkpoints beyond the new Task0 anchor')
    require(runner.meta['checkpoint_sha256'] == {'gam': {'0': expected['sha256']}}, 'Loaded new checkpoint hash mismatch')
    require(runner.meta['anchor_effective_sha256'] != reference['anchor_effective_sha256'], 'Historical effective anchor reused')
    from rq2_common import tensor_hash
    require(runner.meta['head_sha256'] == tensor_hash([runner.anchor_state['fc.weight'][:20], runner.anchor_state['fc.bias'][:20]]), 'New checkpoint Task0 head was not preserved')
    require(not runner.journal.rows, 'Loader-only journal unexpectedly contains measurements')
    return dict(passed=True, new_anchor_checkpoint=expected, formal_data_and_coordinates=True,
                own_checkpoint_task0_head=True, original_task0_weights_not_loaded=True,
                loaded_anchor_effective_sha256=runner.meta['anchor_effective_sha256'])


def validate_finished(summary, meta, rows, gradient_meta, witness):
    task = meta['gradient_task']
    helper = helper_for(task)
    helper.support.verify_protocol(meta)
    helper.finite_tree(summary, 'summary')
    require(summary['status'] == 'complete', 'Summary incomplete')
    for key in ('intervention_arm', 'seed', 'method', 'job', 'gradient_task', 'evaluation_task',
                'direction_sign', 'training_probe_radius', 'training_norm_radius', 'protocol_sha256'):
        require(summary[key] == meta[key], f'Summary identity mismatch: {key}')
    require(meta['job'] == JOBS[task] and meta['direction_sign'] == (1 if task == 0 else -1), 'Direction protocol changed')
    require(meta['evaluation_task'] == 0 and meta['method'] == 'gam', 'Evaluation anchor/task changed')
    require(meta['radii'] == summary['radii'] == RADII, 'Radius grid changed')
    require(len(rows) == summary['expected_states'] == summary['states_completed'] == 11, 'Incomplete radius coverage')
    expected = {'r0'} | {f'r{r:.12g}_gradient' for r in RADII if r}
    require({row['state_id'] for row in rows} == expected, 'Missing or duplicate radius states')
    require(summary['gradient_scope'] == ['subset'] and summary['directions_per_radius'] == 1, 'Fisher/direction scope changed')
    for key in ('anchor_effective_sha256', 'coordinate_sha256', 'head_sha256', 'data_manifest_sha256'):
        require(gradient_meta[key] == meta[key], f'Gradient new-anchor identity mismatch: {key}')
    require(gradient_meta['checkpoint_sha256'] == meta['checkpoint_sha256']['gam'], 'Gradient checkpoint mismatch')
    require(summary['direction_sha256'] == meta['direction_record']['sha256'] == gradient_meta['direction_sha256'], 'Fixed direction hash mismatch')
    require(meta['direction_record']['task'] == task and meta['direction_record']['direction_sign'] == meta['direction_sign'] and
            meta['direction_record']['anchor_effective_sha256'] == meta['anchor_effective_sha256'], 'Direction record scope changed')
    require(meta['gradient'] == meta['fixed_gradient' if task == 0 else 'task1_gradient'] == gradient_meta, 'Embedded gradient manifest changed')
    direction_check = helper.validate_direction_scalars(gradient_meta)
    witness_check = helper.validate_witness(witness, gradient_meta)
    by_id = {row['state_id']: row for row in rows}
    origin = by_id['r0']
    for row in rows:
        helper.finite_tree(row, row['state_id'])
        for key in ('seed', 'method', 'protocol_sha256', 'coordinate_sha256', 'data_manifest_sha256'):
            require(row[key] == meta[key], f'Raw row identity mismatch: {key}')
        require(row['checkpoint_sha256'] == meta['checkpoint_sha256']['gam'], 'Raw row used another checkpoint')
        radius = row['point']['radius']
        require(radius in RADII and row['amount'] == radius and row['anchor_restored_exactly'] is True, 'Radius/restoration changed')
        close(row['effective_weight_drift'], radius, 'Realized effective-Q/V radius', atol=1e-5, rtol=2e-6)
        helper.numerical_equal(row['effective_weight_drift_squared'], row['effective_weight_drift'] ** 2, 'Squared drift')
        require(row['state_id'] == ('r0' if radius == 0 else f'r{radius:.12g}_gradient'), 'State/radius mismatch')
        require(row['point']['kind'] == ('anchor' if radius == 0 else JOBS[task]), 'Point kind changed')
        if radius:
            require(row['point']['direction'] == meta['direction_record'], 'Direction changed across radii')
        for split, count in (('train', 1259), ('test', 307)):
            metrics = row['metrics'][split]
            require(metrics['samples'] == count and metrics['loss'] >= 0 and 0 <= metrics['accuracy'] <= 100, 'Invalid Task0 metrics')
        require(set(row['fisher_by_scope']) == {'subset'}, 'Fisher scope changed')
        fisher = row['fisher_by_scope']['subset']
        require(fisher['samples'] == 256 and fisher['normalized'] is False and fisher['floor'] is None and
                fisher['regularizer'] is None and fisher['raw_trace'] >= 0 and fisher['q'] == {}, 'Invalid raw Fisher')
        for key in ('samples', 'normalized', 'floor', 'regularizer', 'estimator', 'coordinate', 'gradient_batch_size'):
            require(fisher[key] == origin['fisher_by_scope']['subset'][key], f'Fisher metadata changed: {key}')
    recomputed = [positive.summarize_point(by_id['r0' if r == 0 else f'r{r:.12g}_gradient'], origin) for r in RADII]
    require(summary['radius_summary'] == recomputed, 'Summary differs from raw measurements')
    require(summary['end_restoration']['effective_sha256'] == meta['anchor_effective_sha256'], 'Final anchor changed')
    require(summary['frozen_state_sha256_before'] == summary['frozen_state_sha256_after'] == meta['frozen_state_sha256'] and summary['fixed_state_restored'] is True, 'Fixed backbone/old head changed')
    require(summary['direction_sha256_after'] == gradient_meta['direction_sha256'], 'Direction tensors changed')
    if task == 0:
        close(origin['metrics']['train']['loss'], gradient_meta['objective_at_anchor'], 'Task0 gradient objective at new anchor', atol=3e-6, rtol=1e-7)
    else:
        require(summary['task1_head_sha256_before'] == summary['task1_head_sha256_after'] == gradient_meta['task1_initial_head_sha256'], 'Untrained Task1 head changed')
    return dict(passed=True, measured_states=11, all_statistics_recomputed=True,
                direction_check=direction_check, witness_check=witness_check, new_anchor_bound=True,
                raw_states_finite=True, fixed_backbone_and_head=True, direction_constant=True)


ARTIFACT_FIELDS = {
    'summary_sha256': 'summary.json', 'manifest_sha256': 'manifest.json', 'states_sha256': 'states.jsonl',
    'gradient_artifact_sha256': 'gradient_direction.pt', 'gradient_manifest_sha256': 'gradient_manifest.json',
    'sample_manifest_sha256': 'gradient_sample_manifest.json', 'witness_sha256': 'finite_difference_witness.json',
}
TASK1_FIELDS = {
    'task1_data_manifest_sha256': 'task1_data_manifest.json', 'task1_head_manifest_sha256': 'task1_head_manifest.json',
    'task1_head_artifact_sha256': 'task1_initial_head.pt', 'task1_labels_artifact_sha256': 'task1_labels.pt',
}


def validate_artifacts(directory, load_tensors=True):
    directory = Path(directory)
    receipt = read(directory / 'extension_receipt.json')
    require(receipt['status'] == 'complete' and receipt['wrapper_sha256'] == sha(__file__), 'Incomplete/stale worker receipt')
    fields = dict(ARTIFACT_FIELDS)
    if receipt['gradient_task'] == 1:
        fields.update(TASK1_FIELDS)
    for field, filename in fields.items():
        require(receipt[field] == sha(directory / filename), f'Artifact hash mismatch: {filename}')
    summary, meta = read(directory / 'summary.json'), read(directory / 'manifest.json')
    rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
    gradient_meta, witness = read(directory / 'gradient_manifest.json'), read(directory / 'finite_difference_witness.json')
    require(meta['wrapper_sha256'] == sha(__file__), 'Manifest worker changed')
    for field, filename in (('gradient_manifest_sha256', 'gradient_manifest.json'),
                            ('gradient_artifact_sha256', 'gradient_direction.pt'),
                            ('witness_sha256', 'finite_difference_witness.json')):
        require(meta[field] == sha(directory / filename), f'Manifest artifact binding mismatch: {filename}')
    require(gradient_meta['gradient_artifact_sha256'] == sha(directory / 'gradient_direction.pt') and
            gradient_meta['sample_manifest_sha256'] == sha(directory / 'gradient_sample_manifest.json'), 'Gradient artifact/sample binding mismatch')
    for key in ('seed', 'method', 'job', 'gradient_task', 'evaluation_task', 'direction_sign', 'intervention_arm', 'training_probe_radius', 'training_norm_radius', 'protocol_sha256'):
        require(receipt[key] == meta[key], f'Receipt identity mismatch: {key}')
    require(receipt['expected_states'] == 11, 'Receipt state count changed')
    training = validate_training(meta['config'])
    require(training == meta['training'], 'New training provenance changed')
    require(witness['gradient_manifest_sha256'] == sha(directory / 'gradient_manifest.json'), 'Witness bound to another gradient')
    result = validate_finished(summary, meta, rows, gradient_meta, witness)
    if load_tensors:
        import torch
        sys.path.insert(0, str(EXECUTION))
        from rq2_common import tensor_hash
        artifact = torch.load(directory / 'gradient_direction.pt', map_location='cpu', weights_only=True)
        task = meta['gradient_task']
        gradient = artifact['gradient_mean_fp64']
        direction = artifact['unit_positive_gradient_fp32' if task == 0 else 'unit_negative_gradient_fp32']
        require(artifact['coordinate_order'] == meta['coordinate_order'], 'Saved gradient coordinate order changed')
        require(len(gradient) == len(direction) == 24, 'Saved tensor count changed')
        require(all(tuple(v.shape) == (768, 768) and v.dtype == torch.float64 and bool(torch.isfinite(v).all()) for v in gradient), 'Invalid saved gradient')
        require(all(tuple(v.shape) == (768, 768) and v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in direction), 'Invalid saved direction')
        norm, normu, dot = helper_for(task).validate_gradient_tensors(gradient, direction)
        for key, value in (('raw_grad_norm', norm), ('direction_norm', normu), ('gradient_dot_direction', dot)):
            close(gradient_meta[key], value, f'Saved {key}', atol=1e-10, rtol=1e-10)
        require(tensor_hash(gradient) == gradient_meta['gradient_sha256'] and tensor_hash(direction) == gradient_meta['direction_sha256'], 'Saved gradient/direction hash mismatch')
        sample = read(directory / 'gradient_sample_manifest.json')
        require(len(sample['records']) == gradient_meta['samples'], 'Saved gradient sample count changed')
        require(canonical(sample['records']) == gradient_meta['gradient_samples_sha256'], 'Saved gradient sample membership changed')
        if task == 1:
            head = torch.load(directory / 'task1_initial_head.pt', map_location='cpu', weights_only=True)
            formal = negative.validate_task1_formal(meta['seed'], 'gam', meta['config'])
            expected_head = negative.reconstruct_task1_head(Path(meta['config']['source_path']), meta['seed'])
            require(set(head) == {'weight', 'bias'} and all(torch.equal(head[k], expected_head[k]) for k in head), 'Saved Task1 classifier is not the exact paired untrained head')
            require(negative.named_tensor_hash(head.items()) == gradient_meta['task1_initial_head_sha256'] == formal['head_initialization']['new_head_sha256'], 'Task1 head identity mismatch')
            head_meta = read(directory / 'task1_head_manifest.json')
            require(head_meta == meta['task1_head'] and head_meta['trained'] is False and
                    head_meta['future_trained_checkpoint_used'] is False and head_meta['old_task0_head_sha256'] == meta['head_sha256'], 'Task1/Task0 classifier binding mismatch')
            require(sample == read(directory / 'task1_data_manifest.json'), 'Task1 sample manifests differ')
            labels = torch.load(directory / 'task1_labels.pt', map_location='cpu', weights_only=True)
            require(len(labels['local_labels']) == 997 and torch.equal(labels['global_labels'] - 20, labels['local_labels']) and
                    int(labels['local_labels'].min()) == 0 and int(labels['local_labels'].max()) == 19, 'Task1 label mapping changed')
            require(tensor_hash([labels['local_labels']]) == gradient_meta['training_labels_sha256'], 'Task1 saved label hash mismatch')
            require([row['label'] for row in sample['records']] == labels['global_labels'].tolist() and
                    [row['local_label'] for row in sample['records']] == labels['local_labels'].tolist(), 'Task1 sample/label order mismatch')
        result['actual_gradient_tensors_validated'] = True
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', default='gam', choices=('gam',))
    parser.add_argument('--gradient-task', type=int, required=True, choices=(0, 1))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--path-task', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--allow-pending-training', action='store_true', help='Only with --validate-only; never permits measurement before training completion')
    return parser


def main():
    args = argument_parser().parse_args()
    require(not args.allow_pending_training or args.validate_only, 'Pending training allowed only for read-only preflight')
    config, evidence = preflight(args)
    if args.validate_only:
        print(json.dumps(evidence, indent=2, allow_nan=False))
        return
    require(evidence['status'] == 'passed', 'Training must complete before measurement')
    sys.path.insert(0, str(BASE))
    from diagnostic_full import FullRunner
    import diagnostic as canonical_loader
    require(Path(canonical_loader.__file__).resolve() == EXECUTION / 'diagnostic.py', 'Canonical diagnostic loader was shadowed; import this worker under a unique name')
    from rq2_common import atomic_json, tensor_hash
    loader_args = argparse.Namespace(**vars(args), job='sparse_train')
    runner = FullRunner(loader_args)
    identity = validate_loaded_identity(runner, evidence)
    task, helper = args.gradient_task, helper_for(args.gradient_task)
    job = JOBS[task]
    directory = runner.output / job
    frozen = [value for name, value in runner.model.state_dict().items() if not name.endswith(('.effective_q', '.effective_v'))]
    frozen_hash = tensor_hash(frozen)
    if task == 1:
        negative.prepare_task1_context(runner, directory, evidence['task1_formal'])
    direction, gradient_meta = helper.prepare_gradient(runner, directory)
    witness, witness_check = helper.finite_difference_witness(runner, direction, gradient_meta, directory)
    direction_record = dict(kind='positive_full_task0_training_ce_gradient' if task == 0 else 'negative_full_task1_training_ce_gradient',
        task=task, direction_sign=1 if task == 0 else -1, sha256=gradient_meta['direction_sha256'],
        gradient_sha256=gradient_meta['gradient_sha256'], held_fixed_across_radii=True,
        anchor_effective_sha256=runner.meta['anchor_effective_sha256'])
    runner.args.job, runner.directory = job, directory
    runner.meta.pop('protocol_sha256')
    runner.meta.update(schema='rq2_gam_radius_intervention_v1', job=job, gradient_task=task, evaluation_task=0,
        direction_sign=1 if task == 0 else -1, radii=RADII, intervention_arm=config['intervention_arm'],
        training_probe_radius=config['training_probe_radius'], training_norm_radius=config['training_norm_radius'],
        training=evidence['training'], preflight=evidence, gradient=gradient_meta, direction_record=direction_record,
        wrapper_sha256=sha(__file__), gradient_manifest_sha256=sha(directory / 'gradient_manifest.json'),
        gradient_artifact_sha256=sha(directory / 'gradient_direction.pt'), witness_sha256=sha(directory / 'finite_difference_witness.json'),
        frozen_state_sha256=frozen_hash, task1_head_provenance_scope=evidence['task1_provenance_scope'])
    runner.meta['fixed_gradient' if task == 0 else 'task1_gradient'] = gradient_meta
    if task == 1:
        runner.meta['task1_head'] = runner.task1_head
        runner.meta['task1_data_manifest_sha256'] = runner.task1_data['manifest_sha256']
    runner.protocol = canonical(runner.meta)
    runner.meta['protocol_sha256'] = runner.protocol
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        require(read(manifest_path)['protocol_sha256'] == runner.protocol, 'Ray protocol changed; cannot resume')
    atomic_json(manifest_path, runner.meta)
    runner.journal = canonical_loader.Journal(directory, runner.protocol)
    origin = runner.origin_full(['subset'])
    for radius in RADII:
        if radius:
            runner.measure_full(f'r{radius:.12g}_gradient', direction, radius, fisher_scopes=['subset'],
                                kind=job, radius=radius, direction=direction_record)
    direction_after, frozen_after = tensor_hash(direction), tensor_hash(frozen)
    require(direction_after == gradient_meta['direction_sha256'] and frozen_after == frozen_hash, 'Direction or fixed state changed')
    extra = {}
    if task == 1:
        head_after = negative.named_tensor_hash([('weight', runner.task1_objective_model.new20_weight), ('bias', runner.task1_objective_model.new20_bias)])
        require(head_after == runner.task1_head['named_tensor_sha256'], 'Fresh Task1 head changed')
        extra = dict(task1_head_sha256_before=runner.task1_head['named_tensor_sha256'], task1_head_sha256_after=head_after)
    runner.finish(dict(radius_summary=[positive.summarize_point(runner.journal.rows['r0' if r == 0 else f'r{r:.12g}_gradient'], origin) for r in RADII],
        radii=RADII, expected_states=11, directions_per_radius=1, gradient_scope=['subset'], gradient_task=task,
        evaluation_task=0, direction_sign=1 if task == 0 else -1, intervention_arm=config['intervention_arm'],
        training_probe_radius=config['training_probe_radius'], training_norm_radius=config['training_norm_radius'],
        direction_sha256=gradient_meta['direction_sha256'], direction_sha256_after=direction_after,
        signed_damage_primary=True, positive_damage_also_saved=True, frozen_state_sha256_before=frozen_hash,
        frozen_state_sha256_after=frozen_after, fixed_state_restored=True, **extra))
    summary = read(directory / 'summary.json')
    rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
    result_check = validate_finished(summary, runner.meta, rows, gradient_meta, witness)
    fields = dict(ARTIFACT_FIELDS)
    if task == 1:
        fields.update(TASK1_FIELDS)
    receipt = dict(status='complete', schema='rq2_gam_radius_intervention_v1', job=job, seed=runner.seed, method='gam',
        intervention_arm=config['intervention_arm'], training_probe_radius=config['training_probe_radius'],
        training_norm_radius=config['training_norm_radius'], gradient_task=task, evaluation_task=0,
        direction_sign=1 if task == 0 else -1, expected_states=11, samples=256,
        protocol_sha256=runner.protocol, summary=str(directory / 'summary.json'),
        wrapper_sha256=sha(__file__), preflight=evidence, identity_check=identity,
        origin_check=dict(passed=True, source='direct measurement of this new Task0 anchor; no historical anchor metric substitution'),
        witness_check=witness_check, result_check=result_check, endpoint_check=result_check,
        **{field: sha(directory / filename) for field, filename in fields.items()})
    atomic_json(directory / 'extension_receipt.json', receipt)
    print(json.dumps(dict(event='gam_intervention_diagnostic_complete', receipt=str(directory / 'extension_receipt.json'))), flush=True)


if __name__ == '__main__':
    main()
