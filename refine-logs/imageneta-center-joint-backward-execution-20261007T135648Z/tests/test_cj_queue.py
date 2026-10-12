"""Queue gate regressions without GPU claims, launches, or live queue writes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import expected_budget,gpu_is_idle,sha256,upstream_ready
from run_queue import acquire_idle_gpu

class QueueGate(unittest.TestCase):
    def test_explicit_budget(self):
        for first in ('forward','backward_corrected'):
            for second,ce in [('forward',4),('backward_corrected',4),('centered',5)]:
                b=expected_budget(dict(first_location_estimator=first,second_residual_estimator=second))
                self.assertEqual(b['train_ce_per_update'],ce)
                self.assertEqual(b['diagnostic_ce_per_update'],0)
                self.assertEqual(b['local_plus_ce_per_update']+b['local_minus_ce_per_update'],ce-2)
        bb=expected_budget(dict(first_location_estimator='backward_corrected',second_residual_estimator='backward_corrected'))
        self.assertEqual(bb['local_plus_ce_per_update'],0)
        with self.assertRaises(ValueError):expected_budget(dict(first_location_estimator='typo',second_residual_estimator='forward'))

    def parent_fixture(self,tmp,status='running',completed=15):
        p=Path(tmp);jobs=[dict(id=f'formal_{i}',kind='training',smoke=False) for i in range(16)]
        parent=p/'suite.json';parent.write_text(json.dumps(dict(jobs=jobs)))
        records={j['id']:dict(status='completed' if i<completed else 'running') for i,j in enumerate(jobs)}
        state=p/'state.json';state.write_text(json.dumps(dict(status=status,jobs=records,updated_at='fixture')))
        evidence=p/'reference.txt';evidence.write_text('accepted baseline evidence')
        suite=dict(upstream_campaign=dict(suite_path=str(parent),suite_sha256=sha256(parent),state_path=str(state),required_formal_count=16),
                   reused_results=[dict(variant='FF',evidence_paths=dict(summary=str(evidence)),evidence_sha256=dict(summary=sha256(evidence)))])
        return suite,state,evidence

    def test_upstream_requires_success_not_only_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite,state,_=self.parent_fixture(tmp)
            self.assertFalse(upstream_ready(suite)['ready'])
            x=json.loads(state.read_text());x['status']='completed'
            state.write_text(json.dumps(x))
            self.assertFalse(upstream_ready(suite)['ready'])
            x['jobs']['formal_15']['status']='completed';state.write_text(json.dumps(x))
            self.assertTrue(upstream_ready(suite)['ready'])
            x['jobs']['formal_15']['status']='failed';x['status']='halted_failure';state.write_text(json.dumps(x))
            with self.assertRaises(RuntimeError):upstream_ready(suite)

    def test_reference_mutation_stops_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite,_,evidence=self.parent_fixture(tmp,'completed',16)
            evidence.write_text('mutated')
            with self.assertRaises(RuntimeError):upstream_ready(suite)

    def test_compute_pid_and_memory_and_util_all_required(self):
        idle=dict(memory_used_mb=2,utilization_percent=0,compute_pids=[])
        self.assertTrue(gpu_is_idle(idle))
        for key,value in [('memory_used_mb',500),('utilization_percent',5),('compute_pids',[99])]:
            self.assertFalse(gpu_is_idle(dict(idle,**{key:value})))
        self.assertFalse(gpu_is_idle(None))

    def test_after_lease_telemetry_recheck(self):
        idle=dict(index=2,uuid='GPU-fixture',memory_used_mb=2,utilization_percent=0,compute_pids=[])
        with tempfile.TemporaryDirectory() as tmp:
            suite=dict(gpu_lock_dir=tmp,allowed_gpus=[2])
            with patch('run_queue.query_gpus',return_value={2:dict(idle,compute_pids=[999])}):
                self.assertIsNone(acquire_idle_gpu(suite,{2:idle},set()))
            with patch('run_queue.query_gpus',return_value={2:idle}):
                claimed=acquire_idle_gpu(suite,{2:idle},set())
                self.assertEqual(claimed[0],2);self.assertEqual(len(claimed[2]),2)
                for lease in claimed[2]:lease.close()
            self.assertIsNone(acquire_idle_gpu(suite,{2:idle},{2}))

if __name__=='__main__':unittest.main(verbosity=2)
