#!/usr/bin/env python3
"""Run one approved seed-aware P-family job without changing its numerical helpers."""
import argparse
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from contract import HERE, check_job, read_json, select_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    suite = read_json(args.suite)
    _, job = check_job(suite, select_job(suite, args.job))
    trainer = HERE / 'trainer/four_probe_train.py'
    sys.path.insert(0, str(trainer.parent))
    sys.argv = [str(trainer), '--source', str(Path(suite['source_root']).resolve()),
                '--config', str(Path(job['config']).resolve()),
                '--output-dir', str(Path(job['output_dir']).resolve())]
    if job['smoke']:
        sys.argv += ['--max-tasks', '2', '--max-epochs', '1', '--max-batches', '1']
    if args.dry_run:
        sys.argv += ['--dry-run']
    runpy.run_path(str(trainer), run_name='__main__')


if __name__ == '__main__':
    main()
