#!/usr/bin/env python3
"""Hash the live repair and CPU evidence, without changing historical files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source_paths = [
    'src/uie_collator.py', 'src/uie_collator_OLoRA.py', 'src/optimer_g2.py',
    'src/uie_trainer_lora.py', 'src/run_uie_lora.py',
    'scripts_llama/run_g2_llama.py', 'scripts_llama/run_g2_orders_queue.py',
    'scripts_llama/collect_g2_orders_results.py', 'scripts_llama/audit_g2_label_boundary.py',
    'scripts_llama/diagnose_g2_probe_precision.py', 'scripts_llama/test_g2_corrected_pipeline_cpu.py',
    'scripts_llama/test_run_g2_llama_cpu.py', 'scripts_llama/test_g2_orders_queue_cpu.py',
    'tests/test_uie_collator_decoder.py', 'tests/test_optimer_g2.py', 'tests/test_g2_trainer.py',
    'tests/test_g2_probe_precision.py', 'scripts_llama/G2_LLAMA_ORDER1.md',
    'scripts_llama/G2_LLAMA_CORRECTED_V2.md', '.gitignore',
]
files = [ROOT / name for name in source_paths]
files += sorted((ROOT / 'configs/g2_llama/corrected_v2').glob('*.json'))
files += sorted(ROOT.glob('scripts_order*_llama*/IncLoRA-G2-FixedQ005-*-TokV2_LLaMA3-*.sh'))
files += sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'CHANGE_MANIFEST.json')
files += [ROOT / 'refine-logs/llama_pure_g2_hf_results_20261012T084626Z/LABEL_BOUNDARY_ADDENDUM_20261012.md']
manifest = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'live code repair, configuration and CPU evidence only',
    'files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
    'corrected_training_results': None,
    'new_gpu_jobs': 0,
    'preserved_archived_report_sha256': hashlib.sha256((ROOT /
        'refine-logs/llama_pure_g2_hf_results_20261012T084626Z/RESULT_ANALYSIS.md').read_bytes()).hexdigest(),
}
(OUT / 'CHANGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'manifest_files': len(manifest['files']), 'new_gpu_jobs': 0}, indent=2))
