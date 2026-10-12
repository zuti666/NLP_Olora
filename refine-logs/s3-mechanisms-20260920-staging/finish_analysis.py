"""Collect the final native-protocol table and CPU boundary algebra automatically."""
from pathlib import Path
import json
import os
import subprocess
import sys

from common import atomic_json, now
from summarize_mechanisms import main as summarize

HERE = Path(__file__).resolve().parent


def main():
    summarize()
    jobs = json.loads((HERE / 'suite.json').read_text())['jobs']
    results = json.loads((HERE / 'results.json').read_text())['jobs']
    statuses = {row['id']: row['status'] for row in results}
    out = HERE / 'boundary_analysis'
    out.mkdir(exist_ok=True)
    rows = []
    for job in jobs:
        if statuses[job['id']] != 'completed':
            continue
        prefix = out / job['id']
        if prefix.with_suffix('.hashes.json').exists():
            rows.append({'id': job['id'], 'status': 'existing', 'prefix': str(prefix)})
            continue
        config = json.loads(Path(job['config']).read_text())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (out / (job['id'] + '.log')).open('xb') as handle:
            run = subprocess.run([sys.executable, str(HERE / 'preflight/analyze_boundaries.py'),
                                  '--input', str(Path(config['outputs_root']) / 'mechanism_states'),
                                  '--output-prefix', str(prefix)], env=env,
                                 stdout=handle, stderr=subprocess.STDOUT, check=False)
        rows.append({'id': job['id'], 'status': 'passed' if run.returncode == 0 else 'failed',
                     'returncode': run.returncode, 'prefix': str(prefix)})
    atomic_json(out / 'status.json', {'updated_at': now(), 'jobs': rows})
    return int(any(row['status'] == 'failed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
