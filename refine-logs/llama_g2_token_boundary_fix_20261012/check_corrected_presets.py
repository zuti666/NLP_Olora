#!/usr/bin/env python3
"""Offline literal wrapper check; no CUDA/model imports or execution."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', G2_LLAMA_PYTHON='python3')
checks = []
for wrapper in sorted(ROOT.glob('scripts_order*_llama*/IncLoRA-G2-FixedQ005-*-TokV2_LLaMA3-*.sh')):
    subprocess.run(['bash', '-n', str(wrapper)], env=env, check=True, capture_output=True, text=True)
    output_root = Path('/tmp') / ('g2_v2_docs_' + wrapper.parent.name + '_' + wrapper.stem)
    if output_root.exists():
        raise RuntimeError(f'Dry-run destination must not exist: {output_root}')
    cmd = ['bash', str(wrapper), '--seeds', '0', '42', '1993',
           '--run-root', str(output_root), '--dry-run']
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert not output_root.exists(), 'dry-run created a training output tree'
    assert plan['seeds'] == [0, 42, 1993]
    assert plan['formal_orders'] == 3 and plan['formal_stages'] == 12
    assert len(plan['stages']) == 18
    assert sum(stage['smoke'] for stage in plan['stages']) == 6
    expected_R = .05 if '-R005-' in wrapper.name else .2
    for stage in plan['stages']:
        assert stage['decoder_pipeline_version'] == 'decoder_token_ids_v2'
        assert stage['require_gradient_diagnostics'] is True
        opts = stage['options']
        assert opts['g2_inner_radius_mode'] == opts['g2_outer_radius_mode'] == 'fixed_l2'
        assert opts['g2_inner_radius'] == .05 and opts['g2_outer_radius'] == expected_R
        assert opts['g2_quantization_diagnostics_interval'] == 20
        assert opts['optimizer_type'] == 'g2-adamw_hf'
        assert opts['adam_epsilon'] == 1e-6 and opts['weight_decay'] == 0
        assert opts['ewclora_lambda'] == opts['olora_lamda_1'] == opts['olora_lamda_2'] == 0
    checks.append({'wrapper': str(wrapper.relative_to(ROOT)), 'command': cmd,
                   'sha256': hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                   'syntax': 'PASS', 'literal_dry_run': 'PASS',
                   'seeds': plan['seeds'], 'planned_formal_stages': plan['formal_stages'],
                   'planned_smoke_stages': 6, 'inner_radius': .05, 'outer_radius': expected_R,
                   'output_tree_created': False})
assert len(checks) == 12, len(checks)
receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
           'wrapper_checks': checks, 'wrapper_count': len(checks),
           'planned_formal_stages_expanded': 144, 'planned_smoke_stages_expanded': 72,
           'actual_gpu_training_jobs_launched': 0, 'imports_cuda_or_models': False}
(OUT / 'PRESET_DRY_RUN_CHECK.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: value for key, value in receipt.items() if key != 'wrapper_checks'}, indent=2))
