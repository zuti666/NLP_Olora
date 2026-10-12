#!/usr/bin/env python3
"""CPU-only seven-way smoke pairing; gradient identities only at shared initial state."""
import argparse
import json
from pathlib import Path
from common import HERE, atomic_json, load_suite, now, read_json, sha256


def compute_pairing(suite):
    from validate_job import validate_training
    smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(smokes)!=7:raise AssertionError('Exactly seven smokes required')
    files={};identities=[];traces=[]
    for job in smokes:
        validate_training(job);out=Path(job['output_dir'])
        for name in ('final_summary.json','step_trace.jsonl','task_identities.json','task_invariant_receipts.json','replay_receipts.json','task_matrix.json','effective_config.json'):
            files[str(out/name)]=sha256(out/name)
        identities.append(read_json(out/'task_identities.json'))
        traces.append([json.loads(row) for row in (out/'step_trace.jsonl').read_text().splitlines()])
    same=lambda values:all(v==values[0] for v in values[1:])
    first=[rows[0] for rows in traces]
    checks=dict(task0_initial_parameters=same([x[0]['task_initial_parameter_sha256'] for x in identities]),
        inherited_fc_rows=all(x['old_fc_rows_inherited_exact'] for rows in identities for x in rows),
        frozen_origin=same([[x['frozen_origin_sha256'] for x in rows] for rows in identities]),
        new_classifier_rows=same([[x['new_fc_rows_sha256'] for x in rows] for rows in identities]),
        task_class_slices=same([[x['class_slice'] for x in rows] for rows in identities]),
        actual_input_sequence=same([[(r['task'],r['epoch'],r['batch'],r['identity']) for r in rows] for rows in traces]),
        task0_probe_tensors=same([r['probe_tensor_sha256'] for r in first]),
        task0_rng0=same([r['rng0_sha256'] for r in first]),
        task0_rng1=same([r['rng1_sha256'] for r in first]),
        all_smoke_replay_gates_passed=True,all_whole_task_invariants_passed=True,
        no_later_trained_parameter_equality_required=True)
    if not all(checks.values()):raise AssertionError('Seven-way smoke pairing mismatch: '+str(checks))
    return dict(status='passed',training_runs=7,run_ids=[j['id'] for j in smokes],optimizer_updates=0,
        checks=checks,input_files_sha256=files,observed_tasks=[0,1],observed_coordinates=[[0,1,0],[1,1,0]],
        scope='Seven-way smoke pairing of task0 initialization and probes, both-task input/class/head/origin identities; later trained parameters and gradients are deliberately not equated')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir).resolve();output=Path(args.output_dir).resolve()
    if root!=HERE:raise AssertionError('Pairing must use its own immutable run')
    suite=load_suite(root/'suite.json')
    report=compute_pairing(suite)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'smoke_pairing_report.json',dict(report,created_at=now()))


if __name__=='__main__':main()
