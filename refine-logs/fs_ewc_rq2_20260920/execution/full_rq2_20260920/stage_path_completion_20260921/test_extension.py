"""CPU-only guards for scientific scope and identity in the thin adapter."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import diagnostic_extension as extension


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.formal = json.loads((extension.BASE / 'configs/seed0.json').read_text())
        self.config = copy.deepcopy(self.formal)
        self.config['output_dir'] = str(extension.HERE / 'test_outputs/seed0')

    def test_path_allows_only_output_change(self):
        self.assertEqual(extension.validate_configuration(self.config, self.formal, 'path'), ['output_dir'])
        self.config['fisher_samples'] = 1259
        with self.assertRaisesRegex(ValueError, 'Unauthorized config'):
            extension.validate_configuration(self.config, self.formal, 'path')

    def test_no_write_into_original_formal_results(self):
        self.config['output_dir'] = str(extension.BASE / 'formal_results/extra')
        with self.assertRaisesRegex(ValueError, 'isolated'):
            extension.validate_configuration(self.config, self.formal, 'path')

    def test_radius_extension_is_exact_and_separate(self):
        self.config['sparse_train_radii'] = extension.EXTRA_RADII
        self.assertEqual(extension.validate_configuration(self.config, self.formal, 'sparse_train'),
                         ['output_dir', 'sparse_train_radii'])
        with self.assertRaisesRegex(ValueError, 'Unauthorized config'):
            extension.validate_configuration(self.config, self.formal, 'path')
        self.config['sparse_train_radii'] = [0, 12]
        with self.assertRaisesRegex(ValueError, 'restricted'):
            extension.validate_configuration(self.config, self.formal, 'sparse_train')

    def test_checkpoint_and_seed_switch_are_rejected(self):
        self.config['checkpoints']['sgd']['2'] = '/tmp/unrelated.pt'
        with self.assertRaises(ValueError):
            extension.validate_configuration(self.config, self.formal, 'path')
        self.config = copy.deepcopy(self.formal)
        self.config['output_dir'] = str(extension.HERE / 'test_outputs/seed0')
        self.config['training_seed'] = 1
        with self.assertRaises(ValueError):
            extension.validate_configuration(self.config, self.formal, 'path')

    def test_cli_accepts_missing_formal_stages(self):
        for task in (2, 3, 4, 6, 7, 8):
            args = extension.argument_parser().parse_args(
                ['--config', 'x.json', '--method', 'sgd', '--job', 'path', '--path-task', str(task)])
            self.assertEqual(args.path_task, task)

    def test_origin_guard_ignores_new_direction_but_detects_changed_fisher(self):
        directory = extension.BASE / 'formal_results/seed0/sgd/path_t01'
        rows = [json.loads(line) for line in (directory / 'states.jsonl').read_text().splitlines()]
        origin = copy.deepcopy(next(r for r in rows if r['state_id'] == 'r0'))
        origin['fisher_by_scope']['train']['q'] = {'sgd_t02': 99.0}
        self.assertTrue(extension.validate_origin(origin, {'rows': rows}, 'path')['passed'])
        origin['fisher_by_scope']['train']['raw_trace'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'raw Fisher trace'):
            extension.validate_origin(origin, {'rows': rows}, 'path')

    def test_real_reference_and_existing_endpoint_pass(self):
        coverage = extension.read(extension.BASE / 'final_report/coverage.json')
        reference = extension.load_reference(0, 'sgd', 'path_t01', coverage)
        trajectory = extension.load_reference(0, 'sgd', 'trajectory_fisher', coverage)
        args = extension.argument_parser().parse_args(
            ['--config', 'x.json', '--method', 'sgd', '--job', 'path', '--path-task', '1'])
        self.assertTrue(extension.validate_finished(reference['summary'], trajectory, args)['passed'])
        broken = copy.deepcopy(reference['summary'])
        broken['S_path'] *= 1.1
        with self.assertRaisesRegex(ValueError, 'Normalized path sensitivity'):
            extension.validate_finished(broken, trajectory, args)

    def test_manifest_hash_rejects_edit(self):
        data = {'seed': 0, 'method': 'sgd'}
        data['protocol_sha256'] = extension.canonical(data)
        extension.verify_protocol(data)
        data['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'protocol hash'):
            extension.verify_protocol(data)


if __name__ == '__main__':
    unittest.main()
