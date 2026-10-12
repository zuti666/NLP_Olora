"""Read-only ten-smoke pairing gate. No CUDA work and no training updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    suite = read(args.suite)
    root = Path(args.suite).parent
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Pairing gate requires fresh output')
    checks, evidence, cells = {}, {}, {}
    for job in suite['jobs']:
        if job['kind'] != 'training' or not job['smoke']:
            continue
        jid = job['id']; p = Path(job['output_dir'])
        summary = read(p/'final_summary.json')
        completed = read(root/'runtime/jobs'/jid/'completed.json')
        exit_record = read(root/'runtime/jobs'/jid/'exit.json')
        rows = [json.loads(v) for v in (p/'step_trace.jsonl').read_text().splitlines()]
        scope = read(p/'model_scope_receipt.json')
        checks[jid+'_accepted'] = (exit_record['returncode'] == 0 and
            completed['summary_sha256'] == digest(p/'final_summary.json') and
            summary['status'] == 'completed' and summary['smoke'] is True and
            summary['completed_steps'] == 2 and summary['completed_epochs'] == 1 and
            len(rows) == 2 and all(summary['checks'].values()))
        checks[jid+'_scope'] = (
            (job['train_scope'] == 'lora' and scope['lora_factor_tensor_count'] > 0 and
             scope['frozen_parameter_tensors'] > 0) or
            (job['train_scope'] == 'full_ft' and scope['no_lora_modules'] and
             scope['all_backbone_and_head_trainable'] and
             scope['lora_factor_tensor_count'] == 0 and scope['frozen_parameter_tensors'] == 0))
        cells[job['train_scope'], job['method']] = (summary, rows, scope)
        for name in ('final_summary.json','step_trace.jsonl','model_scope_receipt.json',
                     'initial_identity.json','cuda_witness.json'):
            evidence[str(p/name)] = digest(p/name)
    checks['all_ten_cells'] = len(cells) == 10
    anchor_summary, anchor_rows, anchor_scope = cells['lora','sgd']
    identity_keys = ('input_sha256','target_sha256','indices_sha256','data_seed')
    for (scope_name, method), (summary, rows, scope) in cells.items():
        key = scope_name+'_'+method
        checks[key+'_same_inputs_two_steps'] = all(
            all(row['identity'][k] == anchor_rows[i]['identity'][k] for k in identity_keys)
            for i,row in enumerate(rows))
        checks[key+'_same_head_and_origin'] = (
            scope['initial_fc_sha256'] == anchor_scope['initial_fc_sha256'] and
            scope['pretrained_backbone_sha256'] == anchor_scope['pretrained_backbone_sha256'] and
            summary['initial_identity']['task_initial_parameter_sha256'] ==
            anchor_summary['initial_identity']['task_initial_parameter_sha256'])
        checks[key+'_first_base_loss_same'] = rows[0]['loss0'] == anchor_rows[0]['loss0']
        if scope_name == 'lora':
            checks[key+'_first_g0_same'] = rows[0]['g0_sha256'] == anchor_rows[0]['g0_sha256']
        else:
            motion = rows[0]['actual_first_update_scope_displacement']
            checks[key+'_real_backbone_and_fc_update'] = (
                motion['backbone_and_fc_both_updated'] and
                motion['backbone_parameter_displacement_norm'] > 0 and motion['fc_parameter_displacement_norm'] > 0 and
                motion['backbone_nonzero_updated_tensor_count'] > 0 and motion['fc_nonzero_updated_tensor_count'] > 0)
    for reference in suite['reused_results']:
        method = reference['method']; summary, rows, scope = cells['lora',method]
        p = Path(reference['output_dir'])
        old_summary = read(reference['summary_path'])
        old_rows = []
        with (p/'step_trace.jsonl').open() as handle:
            old_rows = [json.loads(next(handle)) for _ in range(2)]
        checks['reuse_'+method+'_summary_pin'] = digest(reference['summary_path']) == reference['summary_sha256']
        checks['reuse_'+method+'_initial_identity'] = (
            summary['initial_identity']['task_initial_parameter_sha256'] ==
            old_summary['initial_identity']['task_initial_parameter_sha256'])
        shared = ('base_parameter_sha256','g0_sha256','U_sha256','loss0','current_lr')
        checks['reuse_'+method+'_first_two_update_vectors'] = all(
            all(new[k] == old[k] for k in shared) and
            all(new['identity'][k] == old['identity'][k] for k in identity_keys)
            for new,old in zip(rows,old_rows))
        # Common positions/readouts are also paired at the first step.
        for gradient in ('g1_sha256','g2_sha256'):
            if old_rows[0].get(gradient) is not None:
                checks['reuse_'+method+'_'+gradient] = rows[0].get(gradient) == old_rows[0][gradient]
        evidence[str(p/'final_summary.json')] = digest(p/'final_summary.json')
        evidence[str(p/'step_trace.jsonl')] = digest(p/'step_trace.jsonl')
    for path, sha in suite['reused_result_pins'].items():
        checks['reuse_pin_'+Path(path).parent.name+'_'+Path(path).name] = digest(path) == sha
    report = dict(status='completed' if all(checks.values()) else 'failed',checks=checks,
        evidence_hashes=evidence,training_updates=0,gradient_calls=0,gpu_used=False,
        cells_checked=len(cells),scope_pairing='same origin and FC; fullFT actual backbone and head displacement',
        historical_pairing='LoRA SGD/SAM/g2 initial parameters and two realized input/update vectors')
    out.mkdir(parents=True)
    (out/'final_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),
        failed=[name for name,passed in checks.items() if not passed]),allow_nan=False),flush=True)
    if report['status'] != 'completed':
        raise AssertionError('Pairing gate failed')


if __name__ == '__main__':
    main()
