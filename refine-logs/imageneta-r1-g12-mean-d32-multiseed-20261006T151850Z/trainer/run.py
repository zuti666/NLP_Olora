#!/usr/bin/env python3
"""Optional suite/job adapter; uses the same direct training CLI as job workers."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    suite=json.loads(Path(args.suite).read_text())
    matches=[job for job in suite['jobs'] if job['id']==args.job]
    if len(matches)!=1:raise ValueError('Exactly one matching training job required')
    job=matches[0]
    if job['kind']!='training':raise ValueError('Adapter only supports training jobs')
    command=[sys.executable,str(Path(__file__).with_name('four_probe_train.py')),
             '--source',suite['source_root'],'--config',job['config'],
             '--output-dir',job['output_dir']]
    if job.get('smoke',False) or job.get('phase')=='smoke':
        command += ['--max-tasks',str(job.get('expected_tasks',2)),
                    '--max-epochs',str(job.get('expected_epochs_per_task',1)),
                    '--max-batches',str(job.get('max_batches',1))]
    if args.dry_run:command.append('--dry-run')
    os.execv(sys.executable,command)


if __name__=='__main__':main()
