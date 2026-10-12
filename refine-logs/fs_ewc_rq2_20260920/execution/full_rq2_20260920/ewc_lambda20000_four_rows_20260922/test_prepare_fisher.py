"""Failure tests for selected-card admission and original-input enforcement."""
import fcntl
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepare_under_test', HERE / 'prepare_fisher.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class AdmissionTests(unittest.TestCase):
    def test_gpu_zero_and_one_allowed_without_waiting_for_other_cards(self):
        self.assertEqual(set(prepare.GPU_UUIDS), set(map(str, range(7))))
        for gpu in ('0', '1'):
            uuid = prepare.GPU_UUIDS[gpu]
            replies = [uuid, '0, 0', prepare.GPU_UUIDS['3']]
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), \
                    patch.object(prepare.subprocess, 'check_output', side_effect=replies), \
                    patch.object(prepare, 'verify_inherited_gpu_lock', return_value={'verified': True}):
                receipt, actual = prepare.validate_gpu_admission(gpu)
            self.assertEqual(actual, uuid)
            self.assertEqual(receipt['scope'], 'selected_gpu_only')
            self.assertFalse(receipt['all_gpu_gate_required'])

    def test_gpu_seven_rejected(self):
        with self.assertRaisesRegex(ValueError, '0 through 6'):
            prepare.validate_gpu_admission('7')

    def test_wrong_uuid_rejected_before_lock_or_cuda(self):
        with patch.object(prepare.subprocess, 'check_output', return_value='GPU-wrong'), \
                patch.object(prepare, 'verify_inherited_gpu_lock') as lock:
            with self.assertRaisesRegex(ValueError, 'frozen UUID'):
                prepare.validate_gpu_admission('0')
            lock.assert_not_called()

    def test_selected_card_busy_rejected(self):
        uuid = prepare.GPU_UUIDS['0']
        for status, occupied in [('500, 0', ''), ('0, 5', ''), ('0, 0', uuid)]:
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=uuid), \
                    patch.object(prepare.subprocess, 'check_output', side_effect=[uuid, status, occupied]), \
                    patch.object(prepare, 'verify_inherited_gpu_lock', return_value={'verified': True}):
                with self.assertRaisesRegex(ValueError, 'no longer idle'):
                    prepare.validate_gpu_admission('0')

    def test_live_owned_lock_accepted_and_stays_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'GPU-test.lock'
            with path.open('a') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt = prepare.verify_inherited_gpu_lock(path)
                self.assertTrue(receipt['exclusive_lock_verified'])
                with path.open('a') as probe:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_open_but_unlocked_descriptor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'GPU-test.lock'
            with path.open('a'):
                with self.assertRaisesRegex(ValueError, 'not held exclusively'):
                    prepare.verify_inherited_gpu_lock(path)

    def test_no_inherited_descriptor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'GPU-test.lock'
            path.touch()
            with self.assertRaisesRegex(ValueError, 'queue-inherited'):
                prepare.verify_inherited_gpu_lock(path)


class InputPassTests(unittest.TestCase):
    def test_wrong_first_pixels_rejected_before_first_model_forward(self):
        import torch
        batch = (torch.tensor([0]), torch.zeros(1, 3, 2, 2), torch.tensor([0]))
        verified = prepare.VerifiedInputPass([batch], {'first_batch_pixels_targets_sha256': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'first-batch pixels'):
            next(iter(verified))
        self.assertIsNone(verified.result)

    def test_wrong_sample_order_rejected_before_prepared_receipt(self):
        import torch
        inputs, labels = torch.zeros(1, 3, 2, 2), torch.tensor([0])
        expected = dict(dataset_ids_sha256='dataset', generator_seed=1, num_workers=4,
                        sample_count=1, iteration=50, samples=1, batches=1,
                        first_batch_pixels_targets_sha256=prepare.tensor_hash([('input', inputs), ('target', labels)]),
                        indices_sha256='wrong')
        verified = prepare.VerifiedInputPass([(torch.tensor([0]), inputs, labels)], expected)
        with self.assertRaisesRegex(ValueError, 'indices_sha256'):
            list(verified)
        self.assertIsNone(verified.result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
