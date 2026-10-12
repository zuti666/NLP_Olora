"""Ensure a ten-task continuation cannot leak final metrics into a two-task row."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
SPEC = importlib.util.spec_from_file_location('rq2_endpoint', Path(__file__).with_name('endpoint.py'))
endpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(endpoint)

class EndpointContract(unittest.TestCase):
    def test_sequence_uses_t01(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'metrics.json'
            p.write_text(json.dumps({'cnn': {'matrices': {'final': [[1]*10 for _ in range(10)],
                              't01': [[90, None], [85, 80]]}}}))
            result = endpoint.cil_accuracy(p)
            self.assertEqual([result[k] for k in ('a11','a21','a22')], [90,85,80])

    def test_sequence_without_two_task_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'metrics.json'
            p.write_text(json.dumps({'cnn': {'matrices': {'final': [[1]*10 for _ in range(10)]}}}))
            with self.assertRaises(ValueError):
                endpoint.cil_accuracy(p)

    def test_nonfinite_current_task_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'metrics.json'
            p.write_text(json.dumps({'cnn': {'matrices': {'final': [[90,None],[85,float('nan')]]}}}))
            with self.assertRaises(ValueError):
                endpoint.cil_accuracy(p)

if __name__ == '__main__':
    unittest.main()
