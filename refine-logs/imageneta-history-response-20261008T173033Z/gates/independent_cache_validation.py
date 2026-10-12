"""CPU-only checkpoint corruption and reversed-smoke-order checks."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'trainer'))
import torch
from history_step import HistoryResponseStep
from validate_job import validate_cached_state
import smoke_pairing


def own_digest(values):
    h = hashlib.sha256()
    for v in values:
        v = v.detach().cpu().contiguous()
        h.update(str((tuple(v.shape), str(v.dtype))).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def fixture(direction='history_fused'):
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.4, -.7]]))
    w = .5 if direction == 'history_fused' else 0.
    step = HistoryResponseStep(model, list(model.parameters()), None,
                               dict(direction=direction, history_weight=w))
    opt = torch.optim.SGD(model.parameters(), lr=.04)
    def closure(x, y):
        return .5*model.weight.square().sum(), model.weight
    out = step.compute(None, None, closure, .04)
    step.apply_update(.04, opt)
    identity = {'indices': [3, 7], 'inputs_sha256': 'fixture'}
    last = dict(out['record'], identity=identity)
    state = dict(algorithm_state=step.state_dict(), algorithm=direction,
                 previous_batch_identity=identity)
    job = dict(direction=direction, history_weight=w)
    return state, last, job, out


class CacheValidation(unittest.TestCase):
    def test_valid_checkpoint_with_real_tensors_and_cpu_serialization(self):
        state, last, job, _ = fixture()
        stream = io.BytesIO()
        torch.save(state, stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location='cpu', weights_only=True)
        validate_cached_state(loaded, last, job, 1)

    def test_missing_cache_rejected(self):
        state, last, job, _ = fixture()
        del state['algorithm_state']
        with self.assertRaises(AssertionError):
            validate_cached_state(state, last, job, 1)

    def test_U_substituted_for_raw_gradient_even_with_updated_local_digest_rejected(self):
        state, last, job, out = fixture()
        h = state['algorithm_state']['history']
        h['raw_gradient'] = tuple(v.detach().cpu().clone() for v in out['U'])
        h['raw_gradient_sha256'] = own_digest(h['raw_gradient'])
        self.assertNotEqual(h['raw_gradient_sha256'], last['raw_gradient_sha256'])
        with self.assertRaises(AssertionError):
            validate_cached_state(state, last, job, 1)

    def test_nominal_step_instead_of_actual_FP32_step_rejected(self):
        state, last, job, out = fixture()
        h = state['algorithm_state']['history']
        h['displacement'] = tuple(-.04*v for v in out['U'])
        h['displacement_sha256'] = own_digest(h['displacement'])
        self.assertNotEqual(h['displacement_sha256'], last['actual_step_displacement_sha256'])
        with self.assertRaises(AssertionError):
            validate_cached_state(state, last, job, 1)

    def test_nonfinite_dtype_shape_distance_and_task_identity_tampering_rejected(self):
        state, last, job, _ = fixture()
        cases = []
        bad = copy.deepcopy(state); bad['algorithm_state']['history']['raw_gradient'][0][0, 0] = float('nan'); cases.append(bad)
        bad = copy.deepcopy(state); bad['algorithm_state']['history']['raw_gradient'] = tuple(v.double() for v in bad['algorithm_state']['history']['raw_gradient']); cases.append(bad)
        bad = copy.deepcopy(state); bad['algorithm_state']['history']['raw_gradient'] = (torch.ones(3),); cases.append(bad)
        bad = copy.deepcopy(state); bad['algorithm_state']['history']['length'] *= 2; cases.append(bad)
        bad = copy.deepcopy(state); bad['algorithm_state']['real_updates'] += 1; cases.append(bad)
        bad = copy.deepcopy(state); bad['previous_batch_identity']['indices'] = [99]; cases.append(bad)
        for bad in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises(AssertionError):
                    validate_cached_state(bad, last, job, 1)

    def test_future_only_serializes_no_history_and_rejects_injected_history(self):
        state, last, job, _ = fixture('future_only')
        self.assertIsNone(state['algorithm_state']['history'])
        validate_cached_state(state, last, job, 1)
        foreign, _, _, _ = fixture()
        state['algorithm_state']['history'] = foreign['algorithm_state']['history']
        with self.assertRaises(AssertionError):
            validate_cached_state(state, last, job, 1)

    def test_smoke_pairing_is_independent_of_job_order(self):
        with tempfile.TemporaryDirectory(prefix='history-smoke-order-') as td:
            root = Path(td)
            (root/'gates').mkdir()
            jobs = []
            for direction in ['future_only', 'history_fused']:
                out = root/direction
                out.mkdir()
                jobs.append(dict(id=direction, direction=direction, kind='training', smoke=True,
                                 output_dir=str(out)))
                summary = dict(initial_identity=dict(task_initial_parameter_sha256='same',
                                  frozen_origin_sha256='samefrozen', new_fc_rows_sha256={'weight':'samenewfc'}))
                (out/'final_summary.json').write_text(json.dumps(summary))
                rows = [dict(identity={'indices':[i]}, nominal_lr=.04, raw_gradient_sha256='g'+str(i),
                             future_gradient_sha256='gf'+str(i), U_sha256='U'+str(i),
                             actual_step_displacement_sha256='delta'+str(i),
                             history_used=(direction=='history_fused' and i==1), task_step_index=i)
                        for i in range(2)]
                (out/'step_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (root/'suite.json').write_text(json.dumps({'jobs':jobs}))
            with mock.patch.object(smoke_pairing, 'HERE', root), \
                 mock.patch.object(smoke_pairing, 'validate_job', return_value={'gate':'passed'}), \
                 contextlib.redirect_stdout(io.StringIO()):
                smoke_pairing.main()
            report = json.loads((root/'gates/smoke_pairing/smoke_pairing.json').read_text())
            self.assertEqual(report['status'], 'passed')
            self.assertTrue(all(report['checks'].values()))
            self.assertEqual(report['evidence'][0]['job'], 'history_fused')


if __name__ == '__main__':
    assert not torch.cuda.is_available()
    unittest.main(verbosity=2)
