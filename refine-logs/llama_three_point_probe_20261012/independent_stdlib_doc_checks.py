"""Fresh-doc guarded inspection and all-catalog wrapper checks, no GPU imports."""
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
TARGET = Path('/tmp/three_point_probe_docs_20261012')
COMMAND = ['scripts_llama/run_three_point_probe_llama.py', '--config',
           'configs/three_point_probe_llama/inclora_threepoint_central_llama1b_order1_tokv2_rho005.json',
           '--seeds', '0', '--run-root', str(TARGET), '--dry-run']
real_import, real_popen, old_argv = builtins.__import__, subprocess.Popen, sys.argv
imports = set()


def guarded_import(name, *args, **kwargs):
    imports.add(name)
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'datasets', 'accelerate', 'deepspeed', 'peft'}:
        raise AssertionError('Dry-run attempted nonstdlib/GPU-stack import: ' + name)
    return real_import(name, *args, **kwargs)


def forbidden_child(*args, **kwargs):
    raise AssertionError('Dry-run attempted subprocess/GPU query')


assert not TARGET.exists()
capture = io.StringIO()
try:
    builtins.__import__, subprocess.Popen, sys.argv = guarded_import, forbidden_child, COMMAND
    with contextlib.redirect_stdout(capture):
        runpy.run_path(str(ROOT / COMMAND[0]), run_name='__main__')
finally:
    builtins.__import__, subprocess.Popen, sys.argv = real_import, real_popen, old_argv
guarded_plan = json.loads(capture.getvalue())
assert not TARGET.exists()
assert guarded_plan == json.loads((OUT / 'DOC_DRY_RUN_STDOUT.txt').read_text())
configs = sorted((ROOT / 'configs/three_point_probe_llama').glob('*.json'))
wrappers = sorted(ROOT.glob('scripts_order*_llama*/IncLoRA-*ThreePoint*-TokV2_LLaMA3-*.sh'))
assert len(configs) == len(wrappers) == 24
cells, checked = set(), []
for index, wrapper in enumerate(wrappers):
    never_output = Path('/tmp') / ('reviewer_threepoint_catalog_' + str(index))
    assert not never_output.exists()
    subprocess.run(['bash', '-n', str(wrapper)], check=True, capture_output=True)
    result = subprocess.run(['bash', str(wrapper), '--seeds', '0', '--run-root',
                             str(never_output), '--dry-run'], cwd=ROOT,
                            check=True, text=True, capture_output=True)
    plan = json.loads(result.stdout)
    assert not never_output.exists()
    assert plan['formal_updates_per_complete_order'] == 514
    assert plan['formal_ce_phases_per_complete_order'] == 1542
    assert plan['formal_replay_calls_per_complete_order'] == 1028
    assert len(plan['stages']) == 6
    cells.add((plan['model_size'], plan['order'], plan['mode']))
    checked.append({'wrapper': str(wrapper.relative_to(ROOT)), 'status': 'passed',
                    'plan_stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
                    'created_output': False})
assert cells == {(size, order, mode) for size in ('1B', '3B') for order in (1, 2, 3)
                 for mode in ('g0', 'g1_plus', 'backward_reflected', 'central')}
spec = json.loads((ROOT / '.aris/compute/nlora_req_local_spec.json').read_text())
env_sha = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert env_sha == '0715917e1ad74c80ae4defaf61a143c33d13857323b67d09ead666a792fa61b2'
result = {'status': 'passed', 'guarded_doc_command': 'python3 ' + ' '.join(COMMAND),
          'guarded_stdout_matches_literal': True, 'gpu_stack_imported': False,
          'subprocess_or_gpu_query_attempted': False, 'documented_run_root_created': False,
          'canonical_environment_sha256': env_sha, 'config_count': len(configs),
          'wrapper_count': len(wrappers), 'wrappers': checked,
          'actual_gpu_actions': False, 'prepared_campaign': False}
(OUT / 'INDEPENDENT_STDLIB_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'wrappers'}))
