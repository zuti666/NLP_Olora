"""Aggregate a completed seed prefix and keep its gate receipt immutable."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--seeds', required=True)
    p.add_argument('--allow-smoke', action='store_true')
    args = p.parse_args()
    cmd = [sys.executable, str(Path(__file__).with_name('summarize.py')), '--output-root',
           str(args.output_root), '--seeds', args.seeds]
    if args.allow_smoke:
        cmd += ['--allow-smoke']
    subprocess.run(cmd, check=True)
    output = args.output_root / 'comparison'
    seed = args.seeds.split(',')[-1]
    archive = output / f'through_seed{seed}'
    archive.mkdir(exist_ok=False)
    for file in output.iterdir():
        if file.is_file() and not file.name.startswith('COMPLETE_seed'):
            shutil.copyfile(file, archive / file.name)
    receipt = json.loads((output / 'COMPLETE.json').read_text())
    receipt['summary'] = str((archive / 'summary.json').resolve())
    text = json.dumps(receipt, indent=2) + '\n'
    (archive / 'COMPLETE.json').write_text(text)
    (output / f'COMPLETE_seed{seed}.json').write_text(text)
