"""Scientific protocol and admission/result gating for reflected-probe NLP."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts_llama' / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = module('two_point_runner_cpu', 'run_two_point_probe_llama.py')
queue = module('two_point_queue_cpu', 'run_two_point_probe_queue.py')
collector = module('two_point_collector_cpu', 'collect_two_point_probe_results.py')
PROFILES = sorted((ROOT / 'configs/two_point_probe_llama').glob('*.json'))


class TwoPointLaunchTests(unittest.TestCase):
    def test_all_catalog_profiles_expand_explicit_native_modes_without_outer_probes(self):
        self.assertEqual(len(PROFILES), 18)
        cells = set()
        for path in PROFILES:
            cfg = runner.load_config(path)
            mode = cfg['training']['probe_mode']
            cells.add((cfg['model_size'], cfg['order'], mode))
            plan = runner.build_plan(cfg, [0, 42, 1993], Path('/tmp/probe_catalog_plan_only'))
            self.assertEqual(plan['formal_stages'], 12)
            self.assertEqual(len(plan['stages']), 18)
            for stage in plan['stages']:
                self.assertEqual(stage['mode'], mode)
                self.assertEqual(stage['options']['optimizer_type'], 'probe-adamw_hf')
                self.assertEqual(stage['options']['probe_rho'], .05)
                self.assertEqual(stage['decoder_pipeline_version'], 'decoder_token_ids_v2')
                self.assertFalse(any(k.startswith('g2_') for k in stage['options']))
        self.assertEqual(len(cells), 18)

    def test_mismatched_protocol_flags_fail_before_execution(self):
        original = json.loads(PROFILES[0].read_text())
        for key, value in [('probe_mode', 'g2'), ('probe_rho', .2),
                           ('weight_decay', .01), ('adam_epsilon', 1e-8),
                           ('ewclora_lambda', 1), ('g2_inner_radius', .05)]:
            cfg = copy.deepcopy(original)
            cfg['training'][key] = value
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.json'
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    runner.load_config(path)
        cfg = copy.deepcopy(original)
        cfg['decoder_pipeline_version'] = 'legacy'
        with self.assertRaises(ValueError):
            runner.validate_execution_profile(cfg)

    def test_literal_new_wrappers_are_inspection_only_and_create_no_output(self):
        wrappers = sorted(ROOT.glob('scripts_order*_llama*/IncLoRA-*-TokV2_LLaMA3-*.sh'))
        wrappers = [p for p in wrappers if any(s in p.name for s in ['CleanGradient', 'SAMProbe', 'ReflectedProbe'])]
        self.assertEqual(len(wrappers), 18)
        with tempfile.TemporaryDirectory() as tmp:
            for index, path in enumerate(wrappers):
                target = Path(tmp) / f'not_created_{index}'
                subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True)
                result = subprocess.run(['bash', str(path), '--seeds', '0',
                    '--run-root', str(target), '--dry-run'], cwd=ROOT, text=True,
                    capture_output=True, check=True)
                self.assertEqual(json.loads(result.stdout)['formal_stages'], 4)
                self.assertFalse(target.exists())

    def test_actual_entrypoint_parser_accepts_the_registered_mode_flags(self):
        from transformers import HfArgumentParser
        from run_uie_lora import ModelArguments, DataTrainingArguments, UIETrainingArguments
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
        for mode in ['sgd', 'sam', 'reverse']:
            path = ROOT / f'configs/two_point_probe_llama/inclora_probe_{mode}_llama1b_order1_tokv2_rho005.json'
            stage = runner.build_plan(runner.load_config(path), [0], Path('/tmp/probe_parser_only'))['stages'][0]
            command = stage['command']
            args = command[command.index(str(ROOT / 'src/run_uie_lora.py')) + 1:]
            model_args, data_args, train_args = parser.parse_args_into_dataclasses(args)
            self.assertEqual(train_args.probe_mode, mode)
            self.assertEqual(train_args.probe_rho, .05)
            self.assertEqual(train_args.optimizer_type, 'probe-adamw_hf')
            self.assertEqual(train_args.gradient_accumulation_steps, 16)
            self.assertTrue(train_args.bf16)

    def test_same_point_pair_gate_rejects_different_clean_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for mode in ['sgd', 'sam', 'reverse']:
                out = root / mode
                out.mkdir()
                fields = {k: 'a' * 64 for k in ['initial_trainable_sha256',
                    'clean_gradient_sha256', 'first_window_batch_sha256']}
                (out / 'two_point_probe_training_audit.json').write_text(json.dumps(fields))
                jobs.append({'id': mode, 'model_size': '1B', 'mode': mode,
                             'plan': {'stages': [{'smoke': True, 'stage': 1, 'output_dir': str(out)}]}})
            manifest = {'jobs': jobs, 'queue_root': str(root)}
            queue.verify_initial_pairing(manifest, jobs[1], {})
            self.assertTrue((root / 'runtime/paired_sam.json').is_file())
            p = root / 'reverse/two_point_probe_training_audit.json'
            values = json.loads(p.read_text())
            values['clean_gradient_sha256'] = 'b' * 64
            p.write_text(json.dumps(values))
            with self.assertRaisesRegex(RuntimeError, 'clean_gradient_sha256'):
                queue.verify_initial_pairing(manifest, jobs[2], {})

    def test_stop_while_waiting_does_not_allocate_or_run_a_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'runtime').mkdir()
            (root / 'runtime/queue_state.json').write_text(json.dumps({'status': 'prepared', 'completed_stages': []}))
            cfg = runner.load_config(PROFILES[0])
            plan = runner.build_plan(cfg, [0], root / 'never_run')
            manifest = {'source': str(ROOT), 'repository': str(ROOT), 'queue_root': str(root),
                        'jobs': [{'id': cfg['name'], 'config': cfg, 'plan': plan, 'mode': cfg['training']['probe_mode']} ]}
            def stop(_):
                (root / 'runtime/STOP_AFTER_STAGE').touch()
            with patch.object(queue, 'verify_manifest', return_value=manifest), \
                 patch.object(queue, 'import_runner', return_value=runner), \
                 patch.object(runner, 'claim_gpus', side_effect=runner.GPUUnavailableError('busy')) as claim, \
                 patch.object(queue.time, 'sleep', side_effect=stop), \
                 patch.object(queue, 'run_child') as child:
                self.assertEqual(queue.run(root), 0)
                claim.assert_called_once_with('0,1,2,3')
                child.assert_not_called()
                self.assertFalse((root / 'never_run').exists())
            state = json.loads((root / 'runtime/queue_state.json').read_text())
            self.assertEqual(state['status'], 'stopped')

    def test_collector_pending_and_smoke_never_emit_final_faa(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'runtime').mkdir()
            cfg = runner.load_config(PROFILES[0])
            cfg['data_dir'] = str(root / 'data')
            pins = {}
            for task, relative in collector.metrics.TASK_RELATIVE.items():
                p = Path(cfg['data_dir']) / relative / 'test.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps([{'label': 'x'}] * 7600))
                pins[str(p)] = {'sha256': collector.metrics.sha256(p), 'bytes': p.stat().st_size}
            job = {'id': 'pending', 'model_size': '1B', 'order': 1, 'seed': 0,
                'mode': cfg['training']['probe_mode'], 'config': cfg,
                'plan': runner.build_plan(cfg, [0], root / 'none')}
            manifest = {'schema': collector.SCHEMA, 'decoder_pipeline_version': 'decoder_token_ids_v2',
                        'jobs': [job], 'input_sha256': pins, 'source': str(root / 'source'), 'source_sha256': {}}
            (root / 'source').mkdir()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'manifest.sha256').write_text(collector.metrics.sha256(root / 'manifest.json'))
            (root / 'runtime/queue_state.json').write_text(json.dumps({'status': 'waiting_gpus',
                'completed_stages': [{'key': 'pending:s0:smoke:1', 'receipt': {'status': 'pass'}}]}))
            result = collector.collect(root)
            row = result['order_results'][0]
            self.assertEqual(row['FAA'], '')
            self.assertEqual(row['AAA'], '')
            self.assertFalse(result['old_baseline_used'])

    def test_stop_arriving_during_cuda_witness_prevents_model_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'runtime').mkdir()
            (root / 'runtime/queue_state.json').write_text(json.dumps({'status': 'prepared', 'completed_stages': []}))
            cfg = runner.load_config(PROFILES[0])
            plan = runner.build_plan(cfg, [0], root / 'never_run')
            manifest = {'source': str(ROOT), 'repository': str(ROOT), 'queue_root': str(root),
                        'jobs': [{'id': cfg['name'], 'config': cfg, 'plan': plan, 'mode': cfg['training']['probe_mode']}]}
            def stop(*_):
                (root / 'runtime/STOP_AFTER_STAGE').touch()
            with patch.object(queue, 'verify_manifest', return_value=manifest), \
                 patch.object(queue, 'import_runner', return_value=runner), \
                 patch.object(runner, 'claim_gpus', return_value=([{'index': i} for i in range(4)], [])), \
                 patch.object(queue, 'run_witnesses', side_effect=stop) as witness, \
                 patch.object(queue, 'run_child') as child:
                self.assertEqual(queue.run(root), 0)
                witness.assert_called_once()
                child.assert_not_called()
            self.assertEqual(json.loads((root / 'runtime/queue_state.json').read_text())['status'], 'stopped')

    def test_pause_during_witness_releases_cards_and_repeats_admission_after_resume(self):
        class Lease:
            def __init__(self): self.closed = False
            def close(self): self.closed = True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'runtime').mkdir()
            (root / 'runtime/queue_state.json').write_text(json.dumps({'status': 'prepared', 'completed_stages': []}))
            cfg = runner.load_config(PROFILES[0])
            plan = runner.build_plan(cfg, [0], root / 'never_run')
            manifest = {'source': str(ROOT), 'repository': str(ROOT), 'queue_root': str(root),
                        'jobs': [{'id': cfg['name'], 'config': cfg, 'plan': plan, 'mode': cfg['training']['probe_mode']}]}
            leases = [Lease(), Lease()]
            calls = []
            def witness(*_):
                calls.append(True)
                (root / 'runtime' / ('PAUSE' if len(calls) == 1 else 'STOP_AFTER_STAGE')).touch()
            def resume(_):
                self.assertTrue(leases[0].closed)
                (root / 'runtime/PAUSE').unlink()
            reservations = [([{'index': i} for i in range(4)], [lease]) for lease in leases]
            with patch.object(queue, 'verify_manifest', return_value=manifest), \
                 patch.object(queue, 'import_runner', return_value=runner), \
                 patch.object(runner, 'claim_gpus', side_effect=reservations) as claim, \
                 patch.object(queue, 'run_witnesses', side_effect=witness), \
                 patch.object(queue.time, 'sleep', side_effect=resume), \
                 patch.object(queue, 'run_child') as child:
                self.assertEqual(queue.run(root), 0)
                self.assertEqual(claim.call_count, 2)
                child.assert_not_called()
            self.assertTrue(all(lease.closed for lease in leases))

    def test_collector_uses_same_official_normalized_metric(self):
        stages = {1: {'accuracy': {'a': 80}, 'seen_task_accuracy': 80},
            2: {'accuracy': {'a': 70, 'b': 90}, 'seen_task_accuracy': 80},
            3: {'accuracy': {'a': 60, 'b': 80, 'c': 90}, 'seen_task_accuracy': 230 / 3},
            4: {'accuracy': {'a': 50, 'b': 70, 'c': 80, 'd': 100},
                'seen_task_accuracy': 75, 'counts': {k: 7600 for k in ['a','b','c','d']}}}
        values = collector.metrics.continual_metrics(['a','b','c','d'], stages)
        self.assertEqual(values['FAA'], 75)
        self.assertAlmostEqual(values['BWT'], -20)
        self.assertAlmostEqual(values['AAA'], (80 + 80 + 230 / 3 + 75) / 4)
        self.assertEqual(collector.metrics.normalize(' Society & Culture! '), 'society culture')

    def test_complete_pair_raw_predictions_metrics_and_artifact_integrity(self):
        # Synthetic full-population fixture, not a trained model result.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source').mkdir()
            (root / 'runtime').mkdir()
            pins, jobs, done = {}, [], []
            data = root / 'data'
            for task, relative in collector.metrics.TASK_RELATIVE.items():
                p = data / relative / 'test.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps([{'label': 'x'}] * 7600))
                pins[str(p)] = {'sha256': collector.metrics.sha256(p), 'bytes': p.stat().st_size}
            for mode, bonus in [('sgd', 0), ('reverse', 5)]:
                cfg = runner.load_config(ROOT / f'configs/two_point_probe_llama/inclora_probe_{mode}_llama1b_order1_tokv2_rho005.json')
                cfg['data_dir'] = str(data)
                stages = []
                for stage, task in enumerate(cfg['tasks'], 1):
                    out = root / mode / str(stage)
                    (out / 'adapter').mkdir(parents=True)
                    spec = {'stage': stage, 'task': task, 'seed': 0, 'smoke': False,
                        'output_dir': str(out), 'expected_updates': 2, 'seen_tasks': cfg['tasks'][:stage]}
                    stages.append(spec)
                    predictions = out / 'predict_eval_predictions.jsonl'
                    threshold = (85 - 5 * stage + bonus) * 76
                    with predictions.open('w') as stream:
                        for dataset in cfg['tasks'][:stage]:
                            for index in range(7600):
                                stream.write(json.dumps({'Dataset': dataset, 'Instance': {'id': str(index), 'label': 'x'},
                                    'Prediction': 'x' if index < threshold else 'z'}) + '\n')
                    phases = 1 if mode == 'sgd' else 2
                    trace = out / 'probe_step_trace.jsonl'
                    trace.write_text('\n'.join(json.dumps({'probe_ce_gradient_phases': phases,
                        'probe_replay_closure_calls': phases - 1, 'probe_base_optimizer_steps': 1,
                        'probe_cos_g0_submitted': 1, 'probe_submitted_minus_g0_relative_norm': 0,
                        'probe_actual_update_over_probe_radius': None}) for _ in range(2)) + '\n')
                    audit = out / 'two_point_probe_training_audit.json'
                    audit.write_text('{}')
                    adapter = out / 'adapter/adapter_model.bin'
                    adapter.write_bytes(b'synthetic-unloaded-test-fixture')
                    adapter_cfg = out / 'adapter/adapter_config.json'
                    adapter_cfg.write_text('{}')
                    (out / 'train_results.json').write_text(json.dumps({'train_runtime': stage}))
                    (out / 'predict_results.json').write_text(json.dumps({'predict_runtime': stage / 2}))
                    receipt = {'status': 'pass', 'method': 'TwoPointProbe-Llama', 'mode': mode,
                        'model_size': '1B', 'order': 1, 'seed': 0, 'smoke': False,
                        'stage': stage, 'task': task, 'global_step': 2,
                        'prediction_sha256': collector.metrics.sha256(predictions),
                        'trace_sha256': collector.metrics.sha256(trace),
                        'audit_sha256': collector.metrics.sha256(audit),
                        'adapter_sha256': collector.metrics.sha256(adapter),
                        'adapter_config_sha256': collector.metrics.sha256(adapter_cfg)}
                    done.append({'key': f'{mode}:s0:formal:{stage}', 'job': mode,
                        'stage': stage, 'smoke': False, 'receipt': receipt})
                jobs.append({'id': mode, 'model_size': '1B', 'mode': mode, 'order': 1, 'seed': 0,
                             'config': cfg, 'plan': {'stages': stages}})
            manifest = {'schema': collector.SCHEMA, 'decoder_pipeline_version': 'decoder_token_ids_v2',
                'source': str(root / 'source'), 'source_sha256': {}, 'input_sha256': pins, 'jobs': jobs}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'manifest.sha256').write_text(collector.metrics.sha256(root / 'manifest.json'))
            state = {'status': 'completed', 'completed_stages': done}
            state_path = root / 'runtime/queue_state.json'
            state_path.write_text(json.dumps(state))
            result = collector.collect(root)
            reverse = next(r for r in result['order_results'] if r['mode'] == 'reverse')
            self.assertEqual(result['completed_orders'], 2)
            self.assertEqual(reverse['FAA'], 70)
            self.assertEqual(reverse['AAA'], 77.5)
            self.assertEqual(reverse['BWT'], -10)
            self.assertEqual(reverse['delta_FAA_vs_clean'], 5)
            self.assertEqual(reverse['delta_AAA_vs_clean'], 5)
            self.assertEqual(reverse['delta_BWT_vs_clean'], 0)
            self.assertEqual(reverse['delta_FAA_vs_SAM'], '')
            self.assertEqual(reverse['train_runtime_seconds'], 10)
            self.assertEqual(reverse['predict_runtime_seconds'], 5)
            done[0]['receipt']['stage'] = 999
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'identity or budget'):
                collector.collect(root)
            done[0]['receipt']['stage'] = 1
            state_path.write_text(json.dumps(state))
            (root / 'sgd/1/adapter/adapter_config.json').write_text('{"base_model_name_or_path":"changed"}')
            with self.assertRaisesRegex(ValueError, 'adapter configuration changed'):
                collector.collect(root)


if __name__ == '__main__':
    unittest.main()
