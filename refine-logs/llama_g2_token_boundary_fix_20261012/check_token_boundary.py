"""Read-only CPU proof of the historical and fixed causal label boundary."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from transformers import AutoTokenizer
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from uie_collator import DataCollatorForUIE
OLD = ROOT / 'experiment_queue/llama_pure_g2_hf_allorders_gpu0123_20261011T192153Z_retry2/source/src/uie_collator.py'
spec = importlib.util.spec_from_file_location('historical_uie_collator', OLD)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
model = SimpleNamespace(config=SimpleNamespace(_name_or_path='llama', model_type='llama'))
records = []
for size in ['1B', '3B']:
    source = Path('/data/140-1/users/liying/Sharing/Models/meta-llama') / ('Llama-3.2-' + size)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    old = module.DataCollatorForUIE(tokenizer, model, max_source_length=512, max_target_length=50)
    new = DataCollatorForUIE(tokenizer, model, max_source_length=512, max_target_length=50)
    for label in ['Athlete', 'Educational Institution', 'Natural Place', 'negative', 'Society & Culture', 'Building']:
        row = {'Task': 'CLS', 'Dataset': 'proof', 'Samples': [], 'subset': 'train',
               'Instance': {'instruction': 'Classify: {0}\nAnswer:', 'sentence': 'example', 'label': label}}
        before, after = old([row]), new([row])
        infer_row = {**row, 'subset': 'test'}
        infer = new([infer_row])
        expected = tokenizer(label, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
        get_targets = lambda batch: batch['labels'][0, 1:][batch['loss_mask'][0, :-1].bool()].tolist()
        before_targets, after_targets = get_targets(before), get_targets(after)
        train_ids = after['input_ids'][0][after['attention_mask'][0].bool()].tolist()
        infer_ids = infer['input_ids'][0][infer['attention_mask'][0].bool()].tolist()
        assert after_targets == expected
        assert train_ids[:len(infer_ids)] == infer_ids
        assert train_ids.count(tokenizer.bos_token_id) == 1
        records.append({'model': size, 'label': label, 'expected_target_ids': expected,
                        'old_supervised_ids': before_targets, 'new_supervised_ids': after_targets,
                        'old_supervised_tokens': tokenizer.convert_ids_to_tokens(before_targets),
                        'new_supervised_tokens': tokenizer.convert_ids_to_tokens(after_targets),
                        'old_bos_count': before['input_ids'][0].tolist().count(tokenizer.bos_token_id),
                        'new_bos_count': 1, 'train_inference_prefix_equal': True,
                        'old_target_mismatch': before_targets != expected})
receipt = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
           'device': 'cpu', 'cuda_initialized': torch.cuda.is_initialized(),
           'historical_snapshot': str(OLD), 'historical_sha256': hashlib.sha256(OLD.read_bytes()).hexdigest(),
           'fixed_source_sha256': hashlib.sha256((ROOT / 'src/uie_collator.py').read_bytes()).hexdigest(),
           'decoder_pipeline_version': DataCollatorForUIE.decoder_pipeline_version, 'records': records}
assert not receipt['cuda_initialized']
out = Path(__file__).with_name('TOKEN_BOUNDARY_REGRESSION.json')
out.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'labels_checked': len(records), 'historical_mismatches': sum(r['old_target_mismatch'] for r in records), 'receipt': str(out)}))
