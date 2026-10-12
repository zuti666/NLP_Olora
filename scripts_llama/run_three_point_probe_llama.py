#!/usr/bin/env python3
"""Versioned three-point NLP probes; inspection is CPU/stdlib-only.

The actual queue uses the corrected decoder pipeline, four explicit GPUs,
three identical CE phases (BASE,+probe,-probe), one HF AdamW commit and fresh controls.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location('_three_point_shared_runner', Path(__file__).with_name('run_g2_llama.py'))
shared = importlib.util.module_from_spec(_s)
_s.loader.exec_module(shared)
TASK_ORDERS = shared.TASK_ORDERS
SEEDS = shared.SEEDS
GPUUnavailableError = shared.GPUUnavailableError
MODES = {'g0': 'g0', 'g1_plus': 'gplus', 'backward_reflected': '2g0_minus_gminus',
         'central': 'g0_plus_half_gplus_minus_gminus'}
write_json = shared.write_json
sha256 = shared.sha256


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if cfg.get('schema_version') != 1 or cfg.get('method') != 'inclora_three_point_probe':
        raise ValueError('Unknown three-point-probe configuration.')
    validate_execution_profile(cfg)
    if cfg.get('probe_protocol') != 'paired_three_point' or cfg.get('no_g2_g3') is not True:
        raise ValueError('Three-point runs must declare paired_three_point and no g2/g3.')
    opts = cfg['training']
    if opts.get('optimizer_type') != 'threepoint-adamw_hf' or opts.get('probe_mode') not in MODES:
        raise ValueError('Use a named three-point mode and HF AdamW backend.')
    if opts.get('probe_rho') != .05 or opts.get('probe_eps') != 1e-12:
        raise ValueError('This registered comparison fixes rho=.05 and epsilon=1e-12.')
    interval = opts.get('probe_quantization_diagnostics_interval')
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError('Probe cast diagnostic cadence must be a nonnegative integer.')
    if any(k.startswith('g2_') for k in opts):
        raise ValueError('This method does not compute g2/g3 or accept G2 settings.')
    # Write-free protocol checks; the same shared task/seed/budget constants
    # also drive the actual native stage commands below.
    if cfg.get('model_size') not in ('1B', '3B') or cfg.get('order') not in TASK_ORDERS:
        raise ValueError('Unknown model size or task order.')
    if cfg.get('tasks') != TASK_ORDERS[cfg['order']]:
        raise ValueError('Task ordering differs from the registered Order.')
    if cfg.get('seeds') != list(SEEDS) or cfg.get('default_seed') != 0:
        raise ValueError('Seeds must be [0,42,1993], default0.')
    if cfg.get('execution') != {'world_size': 4, 'global_batch': 64,
            'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 16}:
        raise ValueError('The matched NLP protocol fixes DDP4/batch1/GA16.')
    if cfg.get('optimizer_class') != 'transformers.AdamW':
        raise ValueError('This experiment uses Transformers AdamW.')
    if cfg.get('formal_stage_updates') != [shared.TASK_UPDATES[t] for t in cfg['tasks']]:
        raise ValueError('Task update budgets differ from the matched protocol.')
    if cfg.get('stage_learning_rates') != [.001, .0001, .0001, .0001]:
        raise ValueError('Stage learning rates differ.')
    if cfg.get('adapter_geometry') != {'rank': 8, 'alpha': 32, 'dropout': .1,
                                       'target_modules': ['q_proj', 'v_proj']}:
        raise ValueError('Adapter geometry differs.')
    fixed = {'train_method': 'lora', 'lora_strategy': 'inclora', 'lora_dim': 8,
        'num_train_epochs': 1, 'bf16': True, 'fp16': False,
        'bf16_full_eval': False, 'fp16_full_eval': False,
        'ddp_find_unused_parameters': False, 'lr_scheduler_type': 'constant', 'warmup_steps': 0,
        'adam_beta1': .9, 'adam_beta2': .999, 'adam_epsilon': 1e-6, 'weight_decay': 0.,
        'max_grad_norm': 1., 'max_source_length': 512, 'max_target_length': 50,
        'generation_max_length': 50, 'ewclora_lambda': 0., 'olora_lamda_1': 0.,
        'olora_lamda_2': 0., 'add_task_name': True, 'add_dataset_name': True}
    for k, value in fixed.items():
        if opts.get(k) != value:
            raise ValueError(f'The protocol fixes {k}={value!r}.')
    for key in ('python_executable', 'model_name_or_path', 'data_dir', 'instruction_file'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute local path.')
    return cfg


def validate_execution_profile(cfg: dict) -> None:
    if cfg.get('decoder_pipeline_version') != 'decoder_token_ids_v2':
        raise ValueError('Fresh three-point runs require corrected decoder_token_ids_v2.')


def build_plan(cfg: dict, seeds: list[int], run_root: Path, *, source: Path | None = None,
               max_stages: int = 4, smoke_only: bool = False) -> dict:
    shared.REPO = REPO
    plan = shared.build_plan(cfg, seeds, run_root, source=source, max_stages=max_stages, smoke_only=smoke_only)
    plan['schema'] = 'ThreePointProbePlan_v1'
    plan['mode'] = cfg['training']['probe_mode']
    plan['probe_protocol'] = 'paired_three_point'
    plan['formal_ce_phases_per_complete_order'] = 3 * sum(cfg['formal_stage_updates'])
    plan['formal_replay_calls_per_complete_order'] = 2 * sum(cfg['formal_stage_updates'])
    plan['ce_phases_per_update'] = 3
    plan['replay_calls_per_update'] = 2
    for stage in plan['stages']:
        stage['mode'] = cfg['training']['probe_mode']
        stage['probe_protocol'] = 'paired_three_point'
        stage['expected_ce_phases'] = 3 * stage['expected_updates']
        stage['expected_replay_calls'] = 2 * stage['expected_updates']
    return plan


def claim_gpus(requested: str):
    shared.REPO = REPO
    selected, handles = shared.claim_gpus(requested)
    # The shared lock mechanics are reused; annotate only leases held by this controller.
    for handle in handles:
        handle.write(json.dumps({'method': 'ThreePointProbe-Llama', 'protocol': 'paired_three_point'}) + '\n')
        handle.flush()
    return selected, handles


def environment(source, run_root, gpus, spec_file):
    shared.REPO = REPO
    env = shared.environment(source, run_root, gpus, spec_file)
    env['NCCL_P2P_DISABLE'] = '1'
    env['THREE_POINT_PROBE_SPEC'] = env.pop('G2_LLAMA_SPEC')
    return env


def freeze_source(root):
    shared.REPO = REPO
    return shared.freeze_source(root)


def verify_stage(spec: dict) -> dict:
    output = Path(spec['output_dir'])
    expected = spec['expected_updates']
    mode = spec['mode']
    state = json.loads((output / 'trainer_state.json').read_text())
    if state.get('global_step') != expected:
        raise RuntimeError('Native optimizer update count differs.')
    trace_file = output / 'three_point_probe_step_trace.jsonl'
    rows = [json.loads(s) for s in trace_file.read_text().splitlines() if s.strip()]
    if len(rows) != expected or [r.get('step') for r in rows] != list(range(1, expected + 1)):
        raise RuntimeError('Probe trace is missing a submitted optimizer step.')
    phases = 3
    for row in rows:
        checks = {'world_size': 4, 'actual_micro_batches': 16,
            'configured_gradient_accumulation_steps': 16,
            'effective_window_mean_denominator': 16,
            'probe_schema': 'ThreePointProbe_v1', 'probe_protocol': 'paired_three_point',
            'probe_mode': mode, 'probe_base_optimizer_steps': 1,
            'probe_restore_base_exact': True, 'probe_ce_gradient_phases': phases,
            'probe_replay_closure_calls': 2,
            'probe_final_direction': MODES[mode],
            'probe_rho': .05, 'probe_eps': 1e-12, 'probe_sign': 0, 'probe_signs': [1, -1],
            'probe_plus_sign': 1, 'probe_minus_sign': -1,
            'probe_symmetric_response_definition': '(gplus-gminus)/2',
            'probe_even_response_definition': '(gplus+gminus-2g0)/2',
            'probe_commit_semantics': 'virtual_sgd_probe_then_adamw_commit', 'probe_fallback_reason': None,
            'probe_outer_probe_calls': 0, 'probe_g2_calls': 0, 'probe_g3_calls': 0,
            'probe_fisher_probe_calls': 0, 'probe_projection_calls': 0, 'probe_hvp_calls': 0,
            'probe_history_used': False, 'probe_regularization_hook_used': False,
            'probe_core_explicit_mean_reductions': 0,
            'probe_gradient_reduction_owner': 'trainer',
            'configured_clip_grad_norm': 1., 'probe_final_clip_enabled': True,
            'base_optimizer': 'transformers.optimization.AdamW',
            'strategy': 'inclora', 'history_enabled': False,
            'clean_regularization_enabled': False}
        for key, value in checks.items():
            if row.get(key) != value:
                raise RuntimeError(f'Three-point execution differs at step{row.get("step")}: {key}={row.get(key)!r}, expected{value!r}')
        if any(key.startswith('g2_') or key.startswith('g3_') for key in row):
            raise RuntimeError('Three-point trace unexpectedly contains an outer-gradient method field.')
        for key in ('probe_g0_norm', 'probe_submitted_preclip_norm',
                    'probe_submitted_postclip_norm', 'probe_actual_update_norm',
                    'probe_submitted_minus_g0_norm', 'probe_radius_actual',
                    'probe_plus_radius_actual', 'probe_minus_radius_actual',
                    'probe_gplus_norm', 'probe_gminus_norm',
                    'probe_symmetric_response_norm', 'probe_even_response_norm',
                    'probe_side_difference_norm'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError(f'Invalid observed three-point diagnostic: {key}')
        if row.get('schema') != 'ThreePointProbeTrainerStep_v1':
            raise RuntimeError('Three-point trainer trace schema differs.')
        if row.get('probe_lr') != spec['options']['learning_rate']:
            raise RuntimeError('Observed learning rate differs from native stage.')
    audit_file = output / 'three_point_probe_training_audit.json'
    audit = json.loads(audit_file.read_text())
    checks = {'schema': 'ThreePointProbeTrainerAudit_v1', 'status': 'passed',
        'mode': mode, 'global_step': expected, 'num_records': expected,
        'radius': .05, 'trainable_precision': 'torch.float32', 'bf16_forward': True,
        'world_size': 4, 'actual_micro_batch_counts': [16], 'strategy': 'inclora',
        'fisher_state_entries': 0, 'regularization_enabled': False,
        'history_enabled': False, 'projection_enabled': False, 'tail_windows': 0,
        'g2_enabled': False, 'g3_enabled': False,
        'total_ce_gradient_phases': phases * expected,
        'total_replay_closure_calls': (2) * expected,
        'total_base_optimizer_steps': expected,
        'decoder_pipeline_version': 'decoder_token_ids_v2', 'trace_sha256': sha256(trace_file)}
    checks['probe_quantization_diagnostics_interval'] = spec['options']['probe_quantization_diagnostics_interval']
    for key, value in checks.items():
        if audit.get(key) != value:
            raise RuntimeError(f'Three-point final audit differs: {key}={audit.get(key)!r}, expected{value!r}')
    pipeline = json.loads((output / 'data_pipeline_audit.json').read_text())
    if (pipeline.get('schema') != 'UIEDataPipelineAudit_v1'
            or pipeline.get('decoder_version') != 'decoder_token_ids_v2'):
        raise RuntimeError('Actual tokenization pipeline differs.')
    if audit.get('base_optimizer') != 'transformers.optimization.AdamW':
        raise RuntimeError('Actual AdamW backend differs.')
    for group in audit.get('base_optimizer_param_groups', []):
        if (group.get('eps') != 1e-6 or group.get('weight_decay') != 0
                or group.get('betas') != [.9, .999]
                or group.get('lr') != spec['options']['learning_rate']):
            raise RuntimeError('Actual AdamW parameter-group hyperparameters differ.')
    if not audit.get('base_optimizer_param_groups') or not audit.get('trainable_elements', 0):
        raise RuntimeError('The model did not record trained optimizer parameters.')
    adapter = output / 'adapter'
    cfg = json.loads((adapter / 'adapter_config.json').read_text())
    if (cfg.get('r') != 8 or cfg.get('r_sum') != spec['expected_saved_rank']
        or cfg.get('lora_alpha') != 32 or cfg.get('lora_dropout') != .1
        or set(cfg.get('target_modules', [])) != {'q_proj', 'v_proj'}):
        raise RuntimeError('Exported adapter geometry differs.')
    weight = next((adapter / name for name in ['adapter_model.bin', 'adapter_model.safetensors']
                   if (adapter / name).is_file()), None)
    if weight is None or (adapter / 'ewclora_fisher.pt').exists():
        raise RuntimeError('Adapter is missing or unexpectedly includes Fisher.')
    geometry = shared.verify_adapter_tensors(spec, weight)
    predictions = output / 'predict_eval_predictions.jsonl'
    counts, identities = {}, set()
    gold = {task: json.loads((Path(spec['options']['data_dir']) /
        {'dbpedia': 'TC/dbpedia', 'amazon': 'SC/amazon', 'yahoo': 'TC/yahoo', 'agnews': 'TC/agnews'}[task]
        / 'test.json').read_text()) for task in spec['seen_tasks']}
    for line in predictions.read_text().splitlines():
        pred = json.loads(line)
        dataset = pred.get('Dataset')
        instance = pred.get('Instance', {})
        raw_id = str(instance.get('id'))
        identity = (dataset, raw_id)
        if dataset not in spec['seen_tasks'] or identity in identities or 'Prediction' not in pred:
            raise RuntimeError('Prediction identity/coverage differs.')
        if (not raw_id.isdecimal() or raw_id != str(int(raw_id))
                or not 0 <= int(raw_id) < len(gold[dataset])
                or instance.get('label') != gold[dataset][int(raw_id)]['label']):
            raise RuntimeError('Smoke/formal prediction ID or gold label differs from the pinned dataset.')
        identities.add(identity)
        counts[dataset] = counts.get(dataset, 0) + 1
    if sum(counts.values()) != spec['expected_test_samples']:
        raise RuntimeError('Prediction population differs.')
    if not spec['smoke'] and counts != {task: 7600 for task in spec['seen_tasks']}:
        raise RuntimeError('Evaluation must include all seen-task examples.')
    results = json.loads((output / 'predict_results.json').read_text())
    if not results or any(isinstance(v, float) and not math.isfinite(v) for v in results.values()):
        raise RuntimeError('Prediction results are nonfinite/empty.')
    train_results = json.loads((output / 'train_results.json').read_text())
    for values, field in [(train_results, 'train_runtime'), (results, 'predict_runtime')]:
        value = values.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f'Actual stage timing is invalid: {field}')
    receipt = {'status': 'pass', 'method': 'ThreePointProbe-Llama', 'mode': mode, 'probe_protocol': 'paired_three_point',
        'seed': spec['seed'], 'stage': spec['stage'], 'task': spec['task'],
        'smoke': spec['smoke'], 'order': spec['order'], 'model_size': spec['model_size'],
        'global_step': expected, 'trace_sha256': sha256(trace_file),
        'prediction_sha256': sha256(predictions), 'prediction_samples': sum(counts.values()),
        'adapter_sha256': sha256(weight), 'adapter_config_sha256': sha256(adapter / 'adapter_config.json'),
        'audit_sha256': sha256(audit_file),
        'train_results_sha256': sha256(output / 'train_results.json'),
        'predict_results_sha256': sha256(output / 'predict_results.json'),
        'data_pipeline_audit_sha256': sha256(output / 'data_pipeline_audit.json'),
        'trainer_state_sha256': sha256(output / 'trainer_state.json'), **geometry}
    receipt_file = output / 'three_point_probe_script_verification.json'
    if receipt_file.exists():
        if json.loads(receipt_file.read_text()) != receipt:
            raise RuntimeError('An accepted stage receipt differs; refusing to re-validate altered artifacts.')
    else:
        write_json(receipt_file, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--run-root', type=Path, default=Path('/tmp/three_point_probe_plan'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    plan = build_plan(cfg, args.seeds, args.run_root.resolve())
    if not args.dry_run:
        parser.error('Execute versioned chains through run_three_point_probe_queue.py; this entry point inspects plans only.')
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
