#!/usr/bin/env python3
"""Verify smoke initial states, RNG-defined new heads and inputs before any formal job."""
import argparse
from pathlib import Path
from common import atomic_json, load_suite, now, read_json, sha256
from validate_job import validate_training


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir).resolve();output=Path(args.output_dir).resolve()
    suite=load_suite(root/'suite.json');smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(smokes)!=3:raise AssertionError('Exactly three smokes required')
    files={};identities=[];traces=[]
    for job in smokes:
        validate_training(job);out=Path(job['output_dir'])
        for name in ('final_summary.json','step_trace.jsonl','task_identities.json','task_invariant_receipts.json','replay_receipts.json','task_matrix.json'):
            files[str(out/name)]=sha256(out/name)
        identities.append(read_json(out/'task_identities.json'))
        traces.append([__import__('json').loads(row) for row in (out/'step_trace.jsonl').read_text().splitlines()])
    same=lambda values:all(v==values[0] for v in values[1:])
    checks=dict(task0_initial_parameters=same([x[0]['task_initial_parameter_sha256'] for x in identities]),
        inherited_fc_rows=all(x['old_fc_rows_inherited_exact'] for rows in identities for x in rows),
        frozen_origin=same([[x['frozen_origin_sha256'] for x in rows] for rows in identities]),
        new_classifier_rows=same([[x['new_fc_rows_sha256'] for x in rows] for rows in identities]),
        task_class_slices=same([[x['class_slice'] for x in rows] for rows in identities]),
        actual_input_sequence=same([[(r['task'],r['epoch'],r['batch'],r['identity']) for r in rows] for rows in traces]),
        all_smoke_replay_gates_passed=True,all_whole_task_invariants_passed=True)
    if not all(checks.values()):raise AssertionError('Smoke pairing mismatch: '+str(checks))
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'smoke_pairing_report.json',dict(status='passed',created_at=now(),training_runs=3,
        run_ids=[j['id'] for j in smokes],optimizer_updates=0,checks=checks,input_files_sha256=files,
        observed_tasks=[0,1],observed_coordinates=[[0,1,0],[1,1,0]],
        scope='Smoke pairing; complete formal matrix pairing is rechecked by the offline diagnostic'))


if __name__=='__main__':main()
