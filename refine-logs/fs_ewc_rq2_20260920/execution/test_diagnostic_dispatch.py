"""CPU regression for tensor directions and their persisted metadata."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diagnostic


class DirectionDispatchTests(unittest.TestCase):
    def exercise(self, module):
        runner = module.Runner.__new__(module.Runner)
        runner.config = {'directions': 2, 'radii': [0, 0.5, 1]}
        runner.device = 'cpu'
        runner.args = SimpleNamespace(method='sgd')
        runner.params, runner.anchor, runner.deltas = [], [], {}
        runner.model = runner.collector = None
        runner.x = runner.y = {'subset': None}
        runner.fisher_batch = 8
        runner.meta = {'coordinate_sha256': 'coords'}
        runner.data = {'manifest_sha256': 'data'}
        runner.checksums = {'sgd': {'0': 'checkpoint'}}
        origin = {'state_id': 'r0', 'point': {'kind': 'anchor', 'radius': 0},
                  'metrics': {'train': {'loss': 1.0}, 'test': {'loss': 1.0}},
                  'fisher': {'raw_trace': 10.0}}
        rows = {'r0': origin}
        def add(row):
            rows[row['state_id']] = row
            return row
        runner.journal = SimpleNamespace(rows=rows, add=add)
        runner.require_sanity = lambda: None
        runner.origin = lambda: origin
        runner.evaluate = lambda split: {'loss': 1.1}
        output = {}
        runner.finish = output.update
        vectors = {i: object() for i in range(2)}
        def direction(i, seed, device):
            return vectors[i], {'index': i, 'sha256': f'direction-{i}'}
        with patch.object(module, 'random_direction', side_effect=direction), \
             patch.object(module, 'set_point') as inject, \
             patch.object(module, 'tensor_hash', return_value='point'), \
             patch.object(module, 'fisher', return_value={'raw_trace': 9.0}):
            runner.neighborhood()
        self.assertEqual(len(rows), 5)
        self.assertEqual(inject.call_count, 4)
        for i in range(2):
            for radius in (0.5, 1):
                row = rows[f'r{radius:g}_u{i:02d}']
                self.assertEqual(row['point']['direction'], {'index': i, 'sha256': f'direction-{i}'})
                self.assertEqual(row['amount'], radius)
                inject.assert_any_call([], [], vectors[i], radius)
        self.assertTrue(output['paired_directions'])
        self.assertEqual([r['directions'] for r in output['radius_summary']], [1, 2, 2])

    def test_neighborhood_dispatch_preserves_direction_metadata(self):
        self.exercise(diagnostic)

    def test_archived_implementation_reproduces_original_failure(self):
        path = Path(__file__).parent / 'campaign_seed0_dualline_recovery1/before_fix/diagnostic.py'
        spec = importlib.util.spec_from_file_location('diagnostic_before_fix', path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'direction'"):
            self.exercise(old)


if __name__ == '__main__':
    unittest.main()
