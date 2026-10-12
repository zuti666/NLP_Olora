"""No-commit precision diagnostics, including real custom Llama and local LoRA."""
import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts_llama'))
from diagnose_g2_probe_precision import (diagnose_probe_precision, load_real_training_example,
                                        _rng_snapshot, _rng_restore, build_argument_parser,
                                        validate_cli_arguments)


class Quadratic(torch.nn.Module):
    def __init__(self, fail_on=None, dropout=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.7, -0.4]))
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('seen', torch.zeros((), dtype=torch.float32))
        self.calls = 0
        self.fail_on = fail_on
        self._failure_calls = 0

    def forward(self, features, targets):
        self.calls += 1
        self._failure_calls += 1
        self.seen.add_(1)
        random.random()
        np.random.rand()
        if self.fail_on == self._failure_calls:
            raise RuntimeError('injected probe failure')
        x = self.dropout(features)
        return {'loss': ((x * self.weight - targets).square()).mean()}


BATCH = {'features': torch.tensor([[1.0, 2.0], [2.0, -1.0]]),
         'targets': torch.tensor([[0.2, -0.1], [-0.2, 0.3]])}


def snapshot(model):
    return {'state': {name: value.detach().clone() for name, value in model.state_dict().items()},
            'grads': [(p, p.grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()],
            'torch_rng': torch.get_rng_state().clone(), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'training': [m.training for m in model.modules()]}


class ProbePrecisionTests(unittest.TestCase):
    def assert_preserved(self, model, before):
        self.assertEqual(set(model.state_dict()), set(before['state']))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before['state'][name]), name)
        for parameter, reference, contents in before['grads']:
            self.assertIs(parameter.grad, reference)
            if contents is not None:
                self.assertTrue(torch.equal(parameter.grad, contents))
        self.assertTrue(torch.equal(torch.get_rng_state(), before['torch_rng']))
        self.assertEqual(random.getstate(), before['python_rng'])
        now = np.random.get_state()
        self.assertEqual(now[0], before['numpy_rng'][0])
        self.assertTrue(np.array_equal(now[1], before['numpy_rng'][1]))
        self.assertEqual(now[2:], before['numpy_rng'][2:])
        self.assertEqual([m.training for m in model.modules()], before['training'])

    def test_exact_zero_radius_floor_and_fp32_linear_response(self):
        model = Quadratic()
        before = snapshot(model)
        result = diagnose_probe_precision(model, [model.weight], BATCH, 1e-3, fixed_radii=[0.05, 0.2])
        rows = {row['name']: row for row in result['points']}
        self.assertEqual(result['optimizer_commits'], 0)
        self.assertEqual(result['master_weights'], 'full_model_FP32')
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(result['h'], 1e-3 * result['base']['bf16_g0_norm'])
        for precision in ['bf16', 'fp32']:
            self.assertEqual(rows['zero'][precision]['response_norm'], 0)
            self.assertIsNone(rows['zero'][precision]['response_norm_per_radius'])
        for name in ['half_h', 'double_h', 'fixed_0.05', 'fixed_0.2']:
            self.assertLess(rows[name]['fp32']['relative_linear_residual_against_h'], 1e-4)
            self.assertGreater(rows[name]['fp32']['cos_noise_corrected_response_h'], .99999)
        json.dumps(result, allow_nan=False)
        self.assert_preserved(model, before)

    def test_existing_gradients_buffers_and_all_rng_are_preserved(self):
        torch.manual_seed(109)
        model = Quadratic(dropout=0.3)
        model.weight.grad = torch.tensor([9., -7.])
        before = snapshot(model)
        batch_before = {key: value.clone() for key, value in BATCH.items()}
        result = diagnose_probe_precision(model, [model.weight], BATCH, 1e-3)
        self.assertEqual(result['points'][0]['bf16']['response_norm'], 0)
        self.assertEqual(result['points'][0]['fp32']['response_norm'], 0)
        self.assert_preserved(model, before)
        for key, value in BATCH.items():
            self.assertTrue(torch.equal(value, batch_before[key]))

    def test_exception_restores_base_gradient_buffers_and_rng(self):
        # Calls 1/2 measure base gradients; call 5 is the first nonzero probe.
        model = Quadratic(fail_on=5, dropout=0.3)
        model.weight.grad = torch.tensor([2., 4.])
        before = snapshot(model)
        with self.assertRaisesRegex(RuntimeError, 'injected probe failure'):
            diagnose_probe_precision(model, [model.weight], BATCH, 1e-3)
        self.assert_preserved(model, before)

    def test_bf16_master_model_is_rejected_as_fp32_control(self):
        model = Quadratic().to(torch.bfloat16)
        before = snapshot(model)
        with self.assertRaisesRegex(ValueError, 'FP32 master weights'):
            diagnose_probe_precision(model, [model.weight], BATCH, 1e-3)
        self.assert_preserved(model, before)

    def test_invalid_parameters_and_radii_are_rejected(self):
        model = Quadratic()
        for params, lr, fixed in [([], .01, []), ([model.weight] * 2, .01, []),
                                  ([model.weight], 0, []), ([model.weight], float('nan'), []),
                                  ([model.weight], .01, [-.2])]:
            with self.assertRaises(ValueError):
                diagnose_probe_precision(model, params, BATCH, lr, fixed_radii=fixed)
        with self.assertRaisesRegex(ValueError, 'parameter of this model'):
            diagnose_probe_precision(model, [torch.nn.Parameter(torch.ones(2))], BATCH, .01)

    def test_proxy_is_separate_from_actual_forward_gradient_evidence(self):
        model = Quadratic()
        result = diagnose_probe_precision(model, [model.weight], BATCH, 1e-5, fixed_radii=[.05])
        for point in result['points']:
            proxy = point['bf16_weight_rounding_proxy']
            self.assertFalse(proxy['proves_forward_visibility'])
            self.assertIn('response_norm', point['bf16'])
            self.assertIn('response_norm', point['fp32'])
            self.assertIn('cos_response_bf16_fp32', point)
            self.assertGreaterEqual(proxy['changed_coordinate_fraction'], 0)
            self.assertLessEqual(proxy['changed_coordinate_fraction'], 1)

    def test_actual_custom_llama_local_lora_no_commit(self):
        from transformers import LlamaConfig
        from model.llama import LlamaForCausalLM_with_lossmask
        from peft import LoraConfig, TaskType, get_peft_model
        torch.set_num_threads(1)
        torch.manual_seed(33)
        model = LlamaForCausalLM_with_lossmask(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
            use_cache=False))
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=.1,
            target_modules=['q_proj', 'v_proj']))
        for name, parameter in model.named_parameters():
            parameter.requires_grad_('loranew_' in name)
            if 'loranew_B' in name:
                with torch.no_grad():
                    parameter.normal_(std=.03)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        params[0].grad = torch.full_like(params[0], 7)
        tokens = torch.tensor([[1, 3, 7, 5, 11, 2]])
        labels = torch.full_like(tokens, -100)
        labels[:, 3:] = tokens[:, 3:]
        loss_mask = torch.zeros_like(tokens, dtype=torch.float32)
        loss_mask[:, 2:5] = 1
        batch = {'input_ids': tokens, 'attention_mask': torch.ones_like(tokens),
                 'labels': labels, 'loss_mask': loss_mask}
        before = snapshot(model)
        with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('No optimizer commits')):
            result = diagnose_probe_precision(model, params, batch, 1e-3, fixed_radii=[.05, .2])
        self.assertEqual(result['points'][0]['bf16']['response_norm'], 0)
        self.assertEqual(result['points'][0]['fp32']['response_norm'], 0)
        self.assertGreater(result['base']['bf16_g0_norm'], 0)
        self.assertEqual(result['optimizer_commits'], 0)
        self.assert_preserved(model, before)
        json.dumps(result, allow_nan=False)

    def test_nested_autocast_does_not_reuse_stale_weight_casts(self):
        class LinearLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1, bias=False)
            def forward(self, features):
                return {'loss': self.layer(features).float().square().mean()}
        torch.manual_seed(2)
        model = LinearLoss()
        params = list(model.parameters())
        batch = {'features': torch.tensor([[1., -.5]])}
        before = snapshot(model)
        standalone = diagnose_probe_precision(model, params, batch, .01, fixed_radii=[.05])
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
            # Populate the ambient context's cache before parameter probes.
            model(**batch)
            nested = diagnose_probe_precision(model, params, batch, .01, fixed_radii=[.05])
        standalone_h = next(row for row in standalone['points'] if row['name'] == 'h')
        nested_h = next(row for row in nested['points'] if row['name'] == 'h')
        self.assertGreater(standalone_h['bf16']['response_norm'], .01)
        self.assertEqual(nested_h['bf16']['response_norm'], standalone_h['bf16']['response_norm'])
        for a, b in zip(standalone['points'], nested['points']):
            self.assertEqual(a['bf16']['response_norm'], b['bf16']['response_norm'])
            self.assertEqual(a['fp32']['response_norm'], b['fp32']['response_norm'])
        self.assertFalse(nested['autocast_weight_cache'])
        self.assert_preserved(model, before)

    def test_cuda_rng_scope_uses_only_selected_device_without_real_cuda(self):
        device = torch.device('cuda:3')
        cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch.object(torch.cuda, 'get_rng_state', return_value=cuda_state) as get_one, \
             patch.object(torch.cuda, 'set_rng_state') as set_one, \
             patch.object(torch.cuda, 'get_rng_state_all', side_effect=AssertionError('all devices')), \
             patch.object(torch.cuda, 'set_rng_state_all', side_effect=AssertionError('all devices')):
            state = _rng_snapshot(device)
            _rng_restore(state)
            get_one.assert_called_once_with(device)
            self.assertEqual(set_one.call_count, 1)
            self.assertEqual(set_one.call_args.kwargs['device'], device)
            self.assertTrue(torch.equal(set_one.call_args.args[0], cuda_state))
        with patch.object(torch.cuda, 'get_rng_state', side_effect=AssertionError('CPU touches CUDA')):
            self.assertIsNone(_rng_snapshot(torch.device('cpu'))['torch_cuda'])

    def test_mixed_device_model_is_rejected_before_snapshot(self):
        model = Quadratic()
        model.register_buffer('off_device', torch.empty((), device='meta', dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, 'entire model on one device'):
            diagnose_probe_precision(model, [model.weight], BATCH, .01)

    def test_adapter_cli_rejects_fresh_base_nondefault_lora_options(self):
        parser = build_argument_parser()
        base_args = ['--base-model', str(ROOT), '--adapter', str(ROOT), '--output', '/tmp/probe.json']
        defaults = parser.parse_args(base_args)
        self.assertIs(validate_cli_arguments(defaults, parser), defaults)
        for option, value in [('--new-rank', '16'), ('--lora-alpha', '64'), ('--lora-dropout', '.2')]:
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    validate_cli_arguments(parser.parse_args(base_args + [option, value]), parser)
        # Fresh bases explicitly retain all three configurable values.
        fresh = parser.parse_args(['--base-model', str(ROOT), '--new-rank', '16',
                                   '--lora-alpha', '64', '--lora-dropout', '.2', '--output', '/tmp/probe.json'])
        self.assertIs(validate_cli_arguments(fresh, parser), fresh)

    def test_cli_prompt_prefix_defaults_match_formal_training_profile(self):
        parser = build_argument_parser()
        args = parser.parse_args(['--base-model', str(ROOT), '--output', '/tmp/probe.json'])
        formal = json.loads((ROOT / 'configs/g2_llama/inclora_g2_llama1b_order1.json').read_text())
        self.assertEqual(args.add_task_name, formal['training']['add_task_name'])
        self.assertEqual(args.add_dataset_name, formal['training']['add_dataset_name'])
        self.assertTrue(args.add_task_name)
        self.assertTrue(args.add_dataset_name)
        altered = parser.parse_args(['--base-model', str(ROOT), '--output', '/tmp/probe.json',
                                     '--no-add-task-name', '--no-add-dataset-name'])
        self.assertFalse(altered.add_task_name)
        self.assertFalse(altered.add_dataset_name)

    def test_distinct_fixed_radii_keep_distinct_linearity_references(self):
        model = Quadratic()
        radii = [.05000000001, .05000000002]
        result = diagnose_probe_precision(model, [model.weight], BATCH, .001, fixed_radii=radii)
        points = result['points'][-2:]
        self.assertNotEqual(points[0]['name'], points[1]['name'])
        self.assertEqual([point['radius'] for point in points], radii)

    def test_real_canonical_training_example_matches_instruction_pipeline(self):
        row = load_real_training_example(ROOT / 'CL_Benchmark', 'amazon', 0,
                                         ROOT / 'configs/instruction_config.json')
        raw = json.loads((ROOT / 'CL_Benchmark/SC/amazon/train.json').read_text())[0]
        self.assertEqual(row['Instance']['sentence'], raw['sentence'])
        self.assertEqual(row['Instance']['label'], raw['label'])
        self.assertTrue(row['Instance']['instruction'].endswith('\nAnswer:'))
        self.assertEqual(row['subset'], 'train')
        with self.assertRaisesRegex(ValueError, 'sample index'):
            load_real_training_example(ROOT / 'CL_Benchmark', 'amazon', -1,
                                       ROOT / 'configs/instruction_config.json')


if __name__ == '__main__':
    unittest.main()
