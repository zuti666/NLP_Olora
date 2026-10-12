#!/usr/bin/env python3
"""CPU-only four-way smoke pairing within each of two seeds; shared-initial gradients only."""
import argparse
import json
from pathlib import Path
from common import HERE, atomic_json, load_suite, now, read_json, sha256


def compute_pairing(suite):
    from validate_job import validate_training
    smokes=[j for j in suite['jobs'] if j['kind']=='training' and j['smoke']]
    if len(smokes)!=8 or {j['seed'] for j in smokes}!={0,1}:
        raise AssertionError('Exactly eight smokes across seeds 0/1 required')
    files={};groups={}
    for seed in (0,1):
        jobs=[j for j in smokes if j['seed']==seed]
        if len(jobs)!=4 or {j['direction'] for j in jobs}!={'g1_plus','backward_reflected','central','g0'}:
            raise AssertionError('Exactly four registered directions per seed required')
        identities=[];traces=[]
        for job in jobs:
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
            correct_seed=all(r['seed']==seed for rows in traces for r in rows),
            all_smoke_replay_gates_passed=True,all_whole_task_invariants_passed=True,
            no_later_trained_parameter_equality_required=True)
        if not all(checks.values()):raise AssertionError('Within-seed four-way smoke pairing mismatch: '+str(seed)+' '+str(checks))
        groups[str(seed)]=dict(seed=seed,run_ids=[j['id'] for j in jobs],checks=checks)
    return dict(status='passed',training_runs=8,run_ids=[j['id'] for j in smokes],optimizer_updates=0,
        seeds=[0,1],seed_groups=groups,input_files_sha256=files,observed_tasks=[0,1],
        observed_coordinates=[[0,1,0],[1,1,0]],cross_seed_parameter_equality_required=False,
        scope='Four-way pairing separately within each seed: task0 initialization/probes/RNG and both-task input/class/head/origin identities; later trained states and different seeds are not equated')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();root=Path(args.run_dir).resolve();output=Path(args.output_dir).resolve()
    if root!=HERE:raise AssertionError('Pairing must use its own immutable run')
    suite=load_suite(root/'suite.json')
    report=compute_pairing(suite)
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'smoke_pairing_report.json',dict(report,created_at=now()))


if __name__=='__main__':main()
