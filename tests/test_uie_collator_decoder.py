"""Decoder token-boundary and causal-loss regressions with offline Llama tokenizers."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uie_collator import DataCollatorForUIE
from uie_collator_OLoRA import DataCollatorForUIE as LegacyDataCollator
from model.llama import LlamaForCausalLM_with_lossmask

MODEL_ROOT = Path(os.environ.get('LLAMA_TOKENIZER_TEST_ROOT',
                  '/data/140-1/users/liying/Sharing/Models/meta-llama'))
AFFECTED = ['Athlete', 'Educational Institution', 'Natural Place', 'negative', 'Society & Culture']
NORMAL = ['Building', 'Company', 'positive', 'World', 'Science & Mathematics']


def example(label='Athlete', subset='train', sentence='example', instruction='Classify: {0}\nAnswer:'):
    return {'Task': 'CLS', 'Dataset': 'dbpedia', 'Samples': [], 'subset': subset,
            'class_label': 7, 'Instance': {'label': label, 'sentence': sentence,
                                         'instruction': instruction}}


def unpadded(batch, row=0):
    return batch['input_ids'][row][batch['attention_mask'][row].bool()].tolist()


def supervised(batch, row=0):
    # This is exactly the local Llama model's causal loss shift.
    return batch['labels'][row, 1:][batch['loss_mask'][row, :-1].bool()].tolist()


class DecoderTokenBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {}
        for name in ['Llama-3.2-1B', 'Llama-3.2-3B']:
            path = MODEL_ROOT / name
            if not (path / 'tokenizer.json').is_file():
                raise unittest.SkipTest('Required offline tokenizer missing: ' + str(path))
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
            tokenizer.pad_token = tokenizer.eos_token
            cls.tokenizers[name] = tokenizer
        cls.tokenizer = cls.tokenizers['Llama-3.2-1B']

    def collator(self, tokenizer=None, collator_cls=DataCollatorForUIE, **kwargs):
        settings = {'max_source_length': 512, 'max_target_length': 50}
        settings.update(kwargs)
        return collator_cls(tokenizer or self.tokenizer,
                            model=SimpleNamespace(config=SimpleNamespace(
                                _name_or_path='local/llama', model_type='llama')), **settings)

    def target(self, label, tokenizer=None):
        tokenizer = tokenizer or self.tokenizer
        return tokenizer(label, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]

    def test_all_affected_and_normal_labels_have_complete_supervision(self):
        for model, tokenizer in self.tokenizers.items():
            collator = self.collator(tokenizer=tokenizer)
            for label in AFFECTED + NORMAL:
                with self.subTest(model=model, label=label):
                    batch = collator([example(label)])
                    self.assertEqual(supervised(batch), self.target(label, tokenizer))
                    self.assertEqual(unpadded(batch).count(tokenizer.bos_token_id), 1)
                    self.assertEqual(unpadded(batch).count(tokenizer.eos_token_id), 1)
        # In the broken path the actual first Athlete token (86340) was never supervised.
        self.assertEqual(supervised(self.collator()([example('Athlete')]))[0], 86340)

    def test_training_prompt_is_exact_inference_prefix(self):
        collator = self.collator()
        for label in AFFECTED + NORMAL:
            train = collator([example(label)])
            infer = collator([example(label, subset='test')])
            prompt = unpadded(infer)
            self.assertEqual(unpadded(train)[:len(prompt)], prompt)
            self.assertEqual(prompt[-1], self.tokenizer(':', add_special_tokens=False)['input_ids'][0])
            self.assertEqual(unpadded(train)[len(prompt):], self.target(label))
            self.assertEqual(infer['loss_mask'].sum().item(), 0)
            self.assertTrue(torch.all(infer['labels'] == -100))

    def test_batched_left_padding_does_not_change_answer_mask(self):
        examples = [example('negative', sentence='short'),
                    example('Educational Institution', sentence='a moderately longer example')]
        collator = self.collator(pad_to_multiple_of=16)
        batch = collator(examples)
        self.assertEqual(batch['input_ids'].shape[1] % 16, 0)
        for i, row in enumerate(examples):
            self.assertEqual(supervised(batch, i), self.target(row['Instance']['label']))
            pad = ~batch['attention_mask'][i].bool()
            self.assertTrue(torch.all(batch['labels'][i][pad] == -100))
            self.assertTrue(torch.all(batch['loss_mask'][i][pad] == 0))
            single = self.collator()([row])
            self.assertEqual(unpadded(batch, i), unpadded(single))

    def test_long_source_keeps_answer_cue_and_full_target(self):
        collator = self.collator(max_source_length=12, max_target_length=8)
        row = example('Educational Institution', sentence='large document ' * 2000)
        train = collator([row])
        row['subset'] = 'test'
        infer = collator([row])
        prompt = unpadded(infer)
        self.assertEqual(len(prompt), 12)
        self.assertEqual(prompt[0], self.tokenizer.bos_token_id)
        self.assertTrue(self.tokenizer.decode(prompt, skip_special_tokens=True).endswith('Answer:'))
        self.assertEqual(unpadded(train)[:12], prompt)
        self.assertEqual(supervised(train), self.target('Educational Institution'))
        self.assertLessEqual(train['input_ids'].shape[1], 20)

    def test_long_target_reserves_eos_and_uses_independent_budget(self):
        collator = self.collator(max_source_length=8, max_target_length=5)
        label = 'This is a long class label with many words and tokens'
        row = example(label, sentence='long document ' * 500)
        batch = collator([row])
        expected = self.tokenizer(label, add_special_tokens=False)['input_ids'][:4]
        expected.append(self.tokenizer.eos_token_id)
        self.assertEqual(supervised(batch), expected)
        self.assertEqual(batch['input_ids'].shape[1], 13)

    def test_padding_max_length_includes_source_and_target_budget(self):
        collator = self.collator(max_source_length=16, max_target_length=6, padding='max_length')
        batch = collator([example()])
        self.assertEqual(batch['input_ids'].shape[1], 22)
        self.assertEqual(supervised(batch), self.target('Athlete'))
        infer = collator([example(subset='test')])
        self.assertEqual(infer['input_ids'].shape[1], 16)

    def test_eos_is_not_duplicated_when_present_in_label(self):
        batch = self.collator()([example('Building' + self.tokenizer.eos_token)])
        self.assertEqual(supervised(batch), self.target('Building'))
        self.assertEqual(unpadded(batch).count(self.tokenizer.eos_token_id), 1)

    def test_mixed_training_and_inference_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mix training and inference'):
            self.collator()([example(), example(subset='test')])

    def test_target_budget_cannot_drop_every_answer_token(self):
        with self.assertRaisesRegex(ValueError, 'retain an answer token'):
            self.collator(max_target_length=1)([example()])

    def test_bad_budgets_and_empty_batch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.collator()([])
        with self.assertRaisesRegex(ValueError, 'max_source_length must'):
            self.collator(max_source_length=0)([example()])
        with self.assertRaisesRegex(ValueError, 'max_target_length must'):
            self.collator(max_target_length=0)([example()])

    def test_probe_class_labels_are_kept_out_of_token_sequences(self):
        collator = self.collator(use_probe=True)
        batch = collator([example(), example('negative')])
        self.assertEqual(batch['class_labels'].tolist(), [7, 7])
        self.assertEqual(supervised(batch), self.target('Athlete'))
        self.assertEqual(DataCollatorForUIE.decoder_pipeline_version, 'decoder_token_ids_v2')
        self.assertNotIn('decoder_pipeline_version', batch)

    def test_text_only_keeps_raw_labels_and_rendered_sources(self):
        collator = self.collator(text_only=True)
        result = collator([example('Athlete')])
        self.assertEqual(result['labels'], ['Athlete'])
        self.assertIn('Answer:Athlete', result['inputs'][0])
        self.assertEqual(result['inputs'][0].count(self.tokenizer.bos_token), 1)

    def test_sample_log_records_complete_supervised_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.txt'
            self.collator(input_record_file=str(path))([example('Athlete'), example('negative')])
            logged = path.read_text()
            self.assertIn('Athlete' + self.tokenizer.eos_token, logged)
            self.assertIn('negative' + self.tokenizer.eos_token, logged)
            self.assertNotIn('!Athlete', logged)  # Old logging multiplied masked ids by zero.

    def test_legacy_olora_import_uses_identical_decoder_pipeline(self):
        primary = self.collator()
        legacy = self.collator(collator_cls=LegacyDataCollator)
        # Its legacy model dispatch requires a name containing llama, just as before.
        legacy.model.config._name_or_path = 'llama'
        for subset in ['train', 'dev', 'test']:
            a = primary([example(subset=subset)])
            b = legacy([example(subset=subset)])
            for key in ['input_ids', 'attention_mask', 'labels', 'loss_mask']:
                self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertIs(LegacyDataCollator.decoder_call, DataCollatorForUIE.decoder_call)

    def test_seq2seq_behavior_remains_identical_to_legacy_path(self):
        primary = self.collator()
        legacy = self.collator(collator_cls=LegacyDataCollator)
        for collator in (primary, legacy):
            collator.model.config._name_or_path = 't5'
            collator.model.config.model_type = 't5'
            collator.model.prepare_decoder_input_ids_from_labels = lambda labels: labels.clone()
        rows = [example('Athlete'), example('Building', sentence='a longer document')]
        a, b = primary(rows), legacy(rows)
        for key in ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertNotIn('loss_mask', a)
        primary.text_only = legacy.text_only = True
        self.assertEqual(primary(rows), legacy(rows))

    def test_causal_first_answer_loss_alignment_with_actual_llama_forward(self):
        torch.set_num_threads(2)
        batch = self.collator()([example('Athlete'), example('negative', sentence='a long enough input')])
        config = LlamaConfig(vocab_size=128256, hidden_size=8, intermediate_size=16,
                             num_hidden_layers=1, num_attention_heads=2,
                             pad_token_id=self.tokenizer.pad_token_id,
                             bos_token_id=self.tokenizer.bos_token_id,
                             eos_token_id=self.tokenizer.eos_token_id)
        model = LlamaForCausalLM_with_lossmask(config)
        model.train()
        captured = []
        def capture(_module, _inputs, output):
            output.retain_grad()
            captured.append(output)
        handle = model.lm_head.register_forward_hook(capture)
        output = model(**batch)
        logits = captured[0]
        shift_logits = logits[:, :-1]
        shift_labels = batch['labels'][:, 1:]
        predictor_mask = batch['loss_mask'][:, :-1].bool()
        expected = F.cross_entropy(shift_logits[predictor_mask], shift_labels[predictor_mask])
        self.assertTrue(torch.allclose(output.loss, expected, rtol=1e-6, atol=1e-6))
        output.loss.backward()
        grad_nonzero = logits.grad.abs().sum(-1) > 0
        expected_positions = batch['loss_mask'].bool()
        self.assertTrue(torch.equal(grad_nonzero, expected_positions))
        for row in range(2):
            first_answer = (batch['labels'][row] != -100).nonzero()[0].item()
            self.assertTrue(grad_nonzero[row, first_answer - 1].item())
            self.assertEqual(batch['labels'][row, first_answer].item(),
                             self.target(['Athlete', 'negative'][row])[0])
        handle.remove()


if __name__ == '__main__':
    unittest.main()
