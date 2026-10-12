#!/usr/bin/env python3
"""No-commit BF16/FP32 G2 inner-probe diagnostic on one parameter point/batch.

The full model must have FP32 master weights. BF16 means an autocast forward,
not a BF16 master model. This helper never steps an optimizer and restores
parameter values, existing gradients, buffers, and RNG even on exceptions.
The optional CLI loads only local model/tokenizer files; it does not train.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _rng_snapshot(device):
    return {
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': (torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None),
        'cuda_device': device if device.type == 'cuda' else None,
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }


def _rng_restore(state):
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(state['torch_cuda'], device=state['cuda_device'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def _norm(vector):
    return float(torch.linalg.vector_norm(vector.double()).item())


def _cos(a, b):
    a_norm, b_norm = _norm(a), _norm(b)
    if a_norm == 0 or b_norm == 0:
        return None
    return max(-1.0, min(1.0, float(torch.dot(a.double(), b.double()).item()) / (a_norm * b_norm)))


def _floating_inputs_fp32(value):
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _floating_inputs_fp32(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_floating_inputs_fp32(item) for item in value)
    return value


def diagnose_probe_precision(model, trainable_params: Iterable[torch.nn.Parameter], batch: dict,
                             lr: float, *, fixed_radii=(), eps=1e-12):
    """Measure actual gradient responses and a separate BF16 rounding proxy.

    All probes use the same direction ``-N(g0_bf16)`` and BASE/RNG. The SGD
    radius ``h=lr*||g0_bf16||`` is not claimed to equal an AdamW update length.
    The report contains no raw gradients/model data and is JSON serializable.
    This is a single-device diagnostic, not a distributed trainer closure.
    """
    params = list(trainable_params)
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError('Provide a nonempty list of distinct trainable parameters')
    model_param_ids = {id(p) for p in model.parameters()}
    if any(id(p) not in model_param_ids or not p.requires_grad for p in params):
        raise ValueError('Every probe parameter must be a trainable parameter of this model')
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('lr must be positive and finite')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be positive and finite')
    fixed_radii = tuple(float(r) for r in fixed_radii)
    if any(not math.isfinite(r) or r <= 0 for r in fixed_radii):
        raise ValueError('fixed radii must be positive and finite')
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise ValueError('Use the unwrapped model in this single-device no-commit diagnostic')
    non_fp32 = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                if value.is_floating_point() and value.dtype != torch.float32]
    if non_fp32:
        raise ValueError('Full FP32 comparison requires FP32 master weights and buffers; '
                         'reload the model with torch_dtype=float32. Non-FP32: ' + ', '.join(non_fp32[:4]))
    device = params[0].device
    if device.type not in ('cpu', 'cuda') or any(p.device != device for p in params):
        raise ValueError('Probe parameters must be on one CPU or CUDA device')
    mixed_devices = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                     if value.device != device]
    if mixed_devices:
        raise ValueError('The diagnostic requires the entire model on one device; mixed-device values: '
                         + ', '.join(mixed_devices[:4]))
    if device.type == 'cuda':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError('BF16 autocast is unavailable on the selected CUDA device')

    bases = [p.detach().clone() for p in params]
    initial_rng = _rng_snapshot(device)
    initial_grads = [(p, p.grad, None if p.grad is None else p.grad.detach().clone())
                     for p in model.parameters()]
    initial_buffers = [(module, dict(module._buffers),
                        {key: None if value is None else value.detach().clone()
                         for key, value in module._buffers.items()},
                        set(module._non_persistent_buffers_set)) for module in model.modules()]
    initial_training = [(module, module.training) for module in model.modules()]

    def restore_base():
        with torch.no_grad():
            for p, base in zip(params, bases):
                p.copy_(base)
            for p, grad, contents in initial_grads:
                p.grad = grad
                if grad is not None:
                    grad.copy_(contents)
            for module, refs, contents, nonpersistent in initial_buffers:
                module._buffers.clear()
                module._buffers.update(refs)
                module._non_persistent_buffers_set = set(nonpersistent)
                for name, value in refs.items():
                    if value is not None:
                        value.copy_(contents[name])
            for module, training in initial_training:
                module.training = training

    direction = None

    def measure(mode, radius):
        restore_base()
        _rng_restore(initial_rng)
        with torch.no_grad():
            if radius:
                for p, base, normalized in zip(params, bases, direction):
                    p.copy_(base - radius * normalized)
        context = (torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False)
                   if mode == 'bf16' else torch.autocast(device_type=device.type, enabled=False, cache_enabled=False))
        with context:
            output = model(**_floating_inputs_fp32(batch))
            loss = output['loss'] if isinstance(output, dict) else output.loss
        if loss is None or loss.numel() != 1 or not torch.isfinite(loss).all():
            raise ValueError('The diagnostic requires a finite scalar model loss')
        # A caller may already be inside an autocast region. Disable both ambient
        # autocast and its weight cache for backward as well as for the FP32 control.
        with torch.autocast(device_type=device.type, enabled=False, cache_enabled=False):
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                          for p, g in zip(params, gradients)])
        if not torch.isfinite(flat).all():
            raise ValueError('Nonfinite probe gradients')
        return flat, float(loss.detach().float().item())

    try:
        g0_bf16, loss_bf16 = measure('bf16', 0.0)
        g0_fp32, loss_fp32 = measure('fp32', 0.0)
        g0_norm = _norm(g0_bf16)
        if g0_norm == 0:
            raise ValueError('A zero BF16 base gradient has no shared descent direction')
        h = lr * g0_norm
        if not math.isfinite(h) or h <= 0:
            raise ValueError('The SGD probe radius must be finite and positive')
        flat_direction = g0_bf16 / g0_norm
        direction, start = [], 0
        for p in params:
            end = start + p.numel()
            direction.append(flat_direction[start:end].reshape_as(p))
            start = end
        points = [('zero', 0.0), ('half_h', h / 2), ('h', h), ('double_h', 2 * h)]
        points.extend(('fixed_' + repr(r), r) for r in fixed_radii)
        rows, responses = [], {'bf16': {}, 'fp32': {}}
        for name, radius in points:
            row = {'name': name, 'radius': radius}
            for mode, g0 in [('bf16', g0_bf16), ('fp32', g0_fp32)]:
                probe, loss = measure(mode, radius)
                response = g0 - probe
                responses[mode][name] = response
                response_norm = _norm(response)
                row[mode] = {
                    'probe_loss': loss, 'probe_gradient_norm': _norm(probe),
                    'response_norm': response_norm,
                    'response_relative_to_g0': response_norm / max(_norm(g0), eps),
                    'response_norm_per_radius': response_norm / radius if radius else None,
                    'cos_g0_probe_gradient': _cos(g0, probe),
                }
            row['cos_response_bf16_fp32'] = _cos(responses['bf16'][name], responses['fp32'][name])
            with torch.no_grad():
                represented, actual, intended = [], [], []
                for base, normalized in zip(bases, direction):
                    perturbed = base - radius * normalized
                    represented.append((perturbed.to(torch.bfloat16).float()
                                        - base.to(torch.bfloat16).float()).reshape(-1))
                    actual.append((perturbed - base).reshape(-1))
                    intended.append((-radius * normalized).reshape(-1))
                represented, actual, intended = map(torch.cat, (represented, actual, intended))
                intended_norm = _norm(intended)
                row['fp32_parameter_displacement'] = {
                    'actual_norm': _norm(actual), 'intended_norm': intended_norm,
                    'cos_actual_intended': _cos(actual, intended),
                }
                row['bf16_weight_rounding_proxy'] = {
                    'kind': 'FP32 trainable-parameter casts to BF16; not a forward measurement',
                    'proves_forward_visibility': False,
                    'changed_coordinate_fraction': float((represented != 0).float().mean().item()),
                    'represented_displacement_norm': _norm(represented),
                    'represented_over_intended_norm': _norm(represented) / intended_norm if intended_norm else None,
                    'cos_represented_intended': _cos(represented, intended),
                }
            rows.append(row)
        for row in rows:
            for mode in ('bf16', 'fp32'):
                response = responses[mode][row['name']] - responses[mode]['zero']
                reference = responses[mode]['h'] - responses[mode]['zero']
                prediction = (row['radius'] / h) * reference
                row[mode]['noise_corrected_response_norm'] = _norm(response)
                row[mode]['relative_linear_residual_against_h'] = (
                    _norm(response - prediction) / max(_norm(prediction), eps) if row['radius'] else None)
                row[mode]['cos_noise_corrected_response_h'] = _cos(response, reference)
        report = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'diagnostic': 'same_base_same_rng_inner_probe_bf16_vs_full_fp32',
            'device': str(device), 'cuda_rng_scope': str(device) if device.type == 'cuda' else None,
            'autocast_weight_cache': False, 'lr': lr, 'h': h,
            'direction': '-normalize(g0_bf16)', 'master_weights': 'full_model_FP32',
            'precision_modes': {'bf16': 'BF16 autocast over FP32 masters; weight cache disabled',
                                'fp32': 'autocast disabled over FP32 masters'},
            'optimizer_commits': 0, 'batch_count': 1, 'g0_is_unclipped': True,
            'gradient_evaluations': 2 + 2 * len(points),
            'trainable_coordinates': sum(p.numel() for p in params),
            'base': {'bf16_loss': loss_bf16, 'fp32_loss': loss_fp32,
                     'bf16_g0_norm': g0_norm, 'fp32_g0_norm': _norm(g0_fp32),
                     'cos_g0_bf16_fp32': _cos(g0_bf16, g0_fp32),
                     'g0_bf16_minus_fp32_relative_norm': _norm(g0_bf16 - g0_fp32) / max(_norm(g0_fp32), eps)},
            'points': rows,
            'limitations': ['One batch and parameter point, not trained performance.',
                            'The rounding proxy does not prove that a forward sees or loses a probe.',
                            'Shared direction isolates precision; FP32 does not choose its own g0 direction.',
                            'The virtual SGD radius is distinct from an AdamW commit length.'],
        }
        json.dumps(report, allow_nan=False)
        return report
    finally:
        restore_base()
        _rng_restore(initial_rng)


def load_real_training_example(data_dir, dataset, sample_index, instruction_file):
    """Reconstruct a canonical TC/SC example without changing sampling state."""
    task = {'dbpedia': 'TC', 'amazon': 'SC', 'yahoo': 'TC', 'agnews': 'TC'}[dataset]
    directory = Path(data_dir) / task / dataset
    rows = json.loads((directory / 'train.json').read_text())
    if sample_index < 0 or sample_index >= len(rows):
        raise ValueError('sample index outside train.json')
    raw = rows[sample_index]
    labels = json.loads((directory / 'labels.json').read_text())
    instructions = json.loads(Path(instruction_file).read_text())
    instruction = next(item['instruction'] for item in instructions[task]
                       if item['instruction_type'] == 'zero-shot')
    instruction += 'Option: ' + ', '.join(labels) + ' \n{0}\nAnswer:'
    return {'Task': task, 'Dataset': dataset, 'Samples': [], 'subset': 'train',
            'Instance': {'id': str(sample_index), 'sentence': raw['sentence'],
                         'label': raw['label'], 'ground_truth': raw['label'],
                         'instruction': instruction}}


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--dataset', choices=['dbpedia', 'amazon', 'yahoo', 'agnews'], default='dbpedia')
    parser.add_argument('--data-dir', type=Path, default=REPO / 'CL_Benchmark')
    parser.add_argument('--instruction-file', type=Path, default=REPO / 'configs/instruction_config.json')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fixed-radii', type=float, nargs='*', default=[0.05, 0.2])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--new-rank', type=int, default=8, help='Fresh-base-only LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='Fresh-base-only LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='Fresh-base-only LoRA dropout')
    parser.add_argument('--trainable-scope', choices=['new', 'saved'], default='new',
                        help='new: current IncLoRA task factors; saved: existing checkpoint factors')
    parser.add_argument('--add-task-name', action=argparse.BooleanOptionalAction, default=True,
                        help='Match the formal profile Task prefix (default: true)')
    parser.add_argument('--add-dataset-name', action=argparse.BooleanOptionalAction, default=True,
                        help='Match the formal profile Dataset prefix (default: true)')
    parser.add_argument('--max-source-length', type=int, default=512)
    parser.add_argument('--max-target-length', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    return parser


def validate_cli_arguments(args, parser):
    if args.adapter is not None:
        fresh_defaults = {'new_rank': 8, 'lora_alpha': 32, 'lora_dropout': 0.1}
        nondefaults = [name.replace('_', '-') for name, default in fresh_defaults.items()
                       if getattr(args, name) != default]
        if nondefaults:
            parser.error('--' + ', --'.join(nondefaults) + ' are fresh-base-only; '
                         'with --adapter, preserve the saved adapter settings')
    if args.trainable_scope == 'saved' and args.adapter is None:
        parser.error('--trainable-scope=saved requires --adapter')
    if not args.base_model.is_dir() or (args.adapter is not None and not args.adapter.is_dir()):
        parser.error('model and optional adapter must be existing local directories')
    return args


def main():
    parser = build_argument_parser()
    args = validate_cli_arguments(parser.parse_args(), parser)
    sys.path.insert(0, str(REPO / 'src'))
    from transformers import AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from model.llama import LlamaForCausalLM_with_lossmask
    from uie_collator import DataCollatorForUIE
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = LlamaForCausalLM_with_lossmask.from_pretrained(
        str(args.base_model), local_files_only=True, torch_dtype=torch.float32)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=args.new_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=['q_proj', 'v_proj']))
    for name, param in model.named_parameters():
        param.requires_grad_(('loranew_' in name) if args.trainable_scope == 'new' else ('lora_' in name))
    model = model.float().to(args.device)
    model.train()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('Loaded adapter has no trainable parameters')
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    collator = DataCollatorForUIE(tokenizer, model, max_source_length=args.max_source_length,
                                  max_target_length=args.max_target_length,
                                  add_task_name=args.add_task_name, add_dataset_name=args.add_dataset_name)
    example = load_real_training_example(args.data_dir, args.dataset, args.sample_index, args.instruction_file)
    batch = {key: value.to(args.device) for key, value in collator([example]).items()}
    report = diagnose_probe_precision(model, params, batch, args.lr, fixed_radii=args.fixed_radii)
    report['source'] = {'base_model': str(args.base_model.resolve()),
                        'adapter': str(args.adapter.resolve()) if args.adapter else None,
                        'dataset': args.dataset, 'train_sample_index': args.sample_index,
                        'decoder_pipeline_version': collator.decoder_pipeline_version,
                        'seed': args.seed, 'trainable_scope': args.trainable_scope,
                        'add_task_name': args.add_task_name, 'add_dataset_name': args.add_dataset_name,
                        'lora_settings_source': 'saved_adapter_config' if args.adapter else 'fresh_base_cli',
                        'parameter_point_note': ('Loaded historical adapter plus fresh current-task factors'
                                                  if args.adapter and args.trainable_scope == 'new'
                                                  else 'Loaded saved factors' if args.adapter else 'Fresh task factors')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete_no_commit', 'output': str(args.output), 'h': report['h']}))


if __name__ == '__main__':
    main()
