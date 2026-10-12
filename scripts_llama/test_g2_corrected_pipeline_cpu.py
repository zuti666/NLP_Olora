"""Corrected tokenizer protocol, fixed-radius presets and native G2 integration."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
spec = importlib.util.spec_from_file_location('corrected_g2_runner', ROOT / 'scripts_llama/run_g2_llama.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
PRESETS = sorted((ROOT / 'configs/g2_llama/corrected_v2').glob('*.json'))
cspec = importlib.util.spec_from_file_location('corrected_g2_collector', ROOT / 'scripts_llama/collect_g2_orders_results.py')
collector = importlib.util.module_from_spec(cspec)
cspec.loader.exec_module(collector)


class CorrectedG2PipelineTests(unittest.TestCase):
    def test_registered_fixed_radius_protocols_expand_all_three_seeds(self):
        self.assertEqual(len(PRESETS), 12)
        cells = set()
        for path in PRESETS:
            cfg = runner.load_config(path)
            cells.add((cfg['model_size'], cfg['order'], cfg['training']['g2_outer_radius']))
            plan = runner.build_plan(cfg, [0, 42, 1993], Path('/tmp/g2_tokv2_plan_only'))
            self.assertEqual(plan['formal_orders'], 3)
            self.assertEqual(plan['formal_stages'], 12)
            self.assertEqual(sum(s['smoke'] for s in plan['stages']), 6)
            for stage in plan['stages']:
                self.assertEqual(stage['decoder_pipeline_version'], 'decoder_token_ids_v2')
                self.assertTrue(stage['require_gradient_diagnostics'])
                for side in ['inner', 'outer']:
                    self.assertEqual(stage['options']['g2_' + side + '_radius_mode'], 'fixed_l2')
                self.assertEqual(stage['options']['g2_inner_radius'], .05)
                self.assertIn(stage['options']['g2_outer_radius'], [.05, .2])
                self.assertEqual(stage['options']['g2_quantization_diagnostics_interval'], 20)
                self.assertEqual(stage['options']['adam_epsilon'], 1e-6)
                self.assertEqual(stage['options']['weight_decay'], 0)
        self.assertEqual(cells, {(m, o, r) for m in ['1B', '3B'] for o in [1, 2, 3] for r in [.05, .2]})

    def test_invalid_pipeline_or_sampling_cadence_is_rejected(self):
        original = json.loads(PRESETS[0].read_text())
        for key, value in [('decoder_pipeline_version', 'legacy_bpe_concat'),
                           ('require_gradient_diagnostics', 'yes')]:
            cfg = copy.deepcopy(original)
            cfg[key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.json'
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    runner.load_config(path)
        for value in [True, -1, 1.5]:
            cfg = copy.deepcopy(original)
            cfg['training']['g2_quantization_diagnostics_interval'] = value
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                runner.validate_radii(cfg['training'])

    def test_legacy_profiles_cannot_silently_execute_on_corrected_data(self):
        legacy = runner.load_config(ROOT / 'configs/g2_llama/inclora_g2_llama1b_order1.json')
        with self.assertRaisesRegex(ValueError, 'inspection-only'):
            runner.validate_execution_profile(legacy)
        runner.validate_execution_profile(runner.load_config(PRESETS[0]))

    def test_legacy_baseline_csv_is_not_a_v2_comparator(self):
        with self.assertRaisesRegex(ValueError, 'matching corrected decoder pipeline'):
            collector.read_baseline(collector.BASELINE, decoder_pipeline_version='decoder_token_ids_v2')

    def test_v2_collection_leaves_historical_paired_columns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            pins = {}
            for task, relative in collector.TASK_RELATIVE.items():
                path = data / relative / 'test.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([{'label': 'label', 'sentence': 'x'}] * 7600))
                pins[str(path.resolve())] = {'bytes': path.stat().st_size, 'sha256': collector.sha256(path)}
            cfg = runner.load_config(PRESETS[0])
            cfg['data_dir'] = str(data)
            job = {'id': cfg['name'], 'model_size': cfg['model_size'], 'order': cfg['order'], 'seed': 0,
                   'config': cfg, 'plan': {'stages': [
                       {'smoke': False, 'stage': i, 'seed': 0} for i in range(1, 5)]}}
            manifest = {'schema': 'CurrentG2SixOrderQueue_v1', 'repository': str(ROOT),
                        'input_sha256': pins, 'jobs': [job]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'manifest.sha256').write_text(collector.sha256(root / 'manifest.json'))
            (root / 'runtime').mkdir()
            (root / 'runtime/queue_state.json').write_text(json.dumps(
                {'status': 'prepared', 'completed_stages': [], 'active_stage': None}))
            with patch.object(collector, 'read_baseline', side_effect=AssertionError('Read historical baseline for v2')):
                result = collector.collect(root)
            self.assertIsNone(result['baseline_csv'])
            self.assertIsNone(result['baseline_sha256'])
            self.assertEqual(result['order_results'][0]['paired_IncLoRA_FAA'], '')
            self.assertEqual(result['order_results'][0]['delta_FAA_vs_paired_IncLoRA'], '')

    def test_actual_training_arguments_accept_new_diagnostic_flag(self):
        from transformers import HfArgumentParser
        from run_uie_lora import UIETrainingArguments
        parser = HfArgumentParser(UIETrainingArguments)
        for radius in [.05, .2]:
            args, = parser.parse_args_into_dataclasses([
                '--output_dir', '/tmp/g2_tokv2_parser_only',
                '--g2_inner_radius_mode', 'fixed_l2', '--g2_inner_radius', '.05',
                '--g2_outer_radius_mode', 'fixed_l2', '--g2_outer_radius', str(radius),
                '--g2_quantization_diagnostics_interval', '20',
            ])
            self.assertEqual(args.g2_quantization_diagnostics_interval, 20)
            self.assertEqual(args.g2_outer_radius, radius)

    def test_native_bf16_g2_window_uses_complete_corrected_answer(self):
        import torch
        from transformers import AutoTokenizer, LlamaConfig, AdamW
        from peft import LoraConfig, TaskType, get_peft_model
        from model.llama import LlamaForCausalLM_with_lossmask
        from optimer_g2 import CurrentG2
        from uie_collator import DataCollatorForUIE
        from test_g2_trainer import trainer_stub, clean_pass
        torch.set_num_threads(1)
        torch.manual_seed(931)
        source = Path('/data/140-1/users/liying/Sharing/Models/meta-llama/Llama-3.2-1B')
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        config = LlamaConfig(vocab_size=128256, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2, use_cache=False)
        config._name_or_path = 'tiny/llama'
        model = get_peft_model(LlamaForCausalLM_with_lossmask(config), LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=.1,
            target_modules=['q_proj', 'v_proj']))
        for name, param in model.named_parameters():
            param.requires_grad_('loranew_' in name)
        collator = DataCollatorForUIE(tokenizer, model, max_source_length=32, max_target_length=8)
        def row(label):
            return {'Task': 'TC', 'Dataset': 'dbpedia', 'Samples': [], 'subset': 'train',
                    'Instance': {'label': label, 'sentence': 'example', 'instruction': '{0}\nAnswer:'}}
        batches = [collator([row(label)]) for label in ['Athlete', 'Educational Institution']]
        for batch, label in zip(batches, ['Athlete', 'Educational Institution']):
            actual = batch['labels'][:, 1:][batch['loss_mask'][:, :-1].bool()].tolist()
            self.assertEqual(actual, tokenizer(label, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id])
        with tempfile.TemporaryDirectory() as tmp:
            trainer = trainer_stub(model, tmp, ga=2)
            trainer.args.optimizer_type = 'g2-adamw_hf'
            trainer.args.max_grad_norm = 1.0
            trainer.args.bf16 = True
            trainer.data_collator = collator
            trainer.compute_loss_context_manager = lambda: torch.autocast('cpu', dtype=torch.bfloat16)
            params = [p for p in model.parameters() if p.requires_grad]
            base = AdamW(params, lr=.001, eps=1e-6, weight_decay=0, no_deprecation_warning=True)
            trainer.optimizer = CurrentG2(params, base, model, closure_gradients_reduced=True,
                inner_radius_mode='fixed_l2', outer_radius_mode='fixed_l2',
                inner_radius=.05, outer_radius=.2, quantization_diagnostics_interval=20)
            states = clean_pass(trainer, model, batches)
            trainer._run_g2_optimizer_window(model, batches, states)
            trainer.state.global_step += 1
            trainer.is_world_process_zero = lambda: True
            audit = trainer.finish_g2_training_audit()
            self.assertEqual(audit['decoder_pipeline_version'], 'decoder_token_ids_v2')
            self.assertEqual(audit['g2_quantization_diagnostics_interval'], 20)
            trace = json.loads((Path(tmp) / 'g2_step_trace.jsonl').read_text())
            self.assertEqual(trace['g2_ce_gradient_phases'], 3)
            self.assertEqual(trace['g2_base_optimizer_steps'], 1)
            self.assertEqual(trace['g2_final_direction'], 'g2')
            self.assertTrue(trace['g2_quantization_proxy']['sampled'])
            self.assertFalse(trace['g2_quantization_proxy']['proves_forward_visibility'])
            self.assertIsNotNone(trace['g2_cos_g0_g2'])
            self.assertGreaterEqual(trace['g2_g2_minus_g0_relative_norm'], 0)
            json.dumps(audit, allow_nan=False)
            json.dumps(trace, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
