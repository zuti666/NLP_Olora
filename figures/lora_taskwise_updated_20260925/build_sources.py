#!/usr/bin/env python3
"""Freeze the requested ImageNet-R/A rho=1.0 sources; retain other plot runs."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / 'lora_taskwise_20260918/aa_times'
COPYFLAT = Path('/data/140-1/users/liying/CopyFlatCV')
INPUTS = {
    'imagenetr': (COPYFLAT/'refine-logs/imagenetr-r1-seeds012-20260924/results.json', 'jobs'),
    'imageneta': (COPYFLAT/'refine-logs/s3-r1-seeds01-20260924/MULTISEED_RESULTS.json', 'rows'),
}
METHOD = 'GAM-EWC+Merge'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_sources():
    old_path = REFERENCE/'sources.json'
    old = json.loads(old_path.read_text())
    retained, replaced = [], []
    for run in old['runs']:
        if not run.get('include', True) or run['method'] == 'InfLoRA':
            continue
        (replaced if run['method'] == METHOD and run['dataset'] in INPUTS else retained).append(run)
    updates, summaries = [], []
    for dataset, (source, rows_key) in INPUTS.items():
        data = json.loads(source.read_text())
        rows = data[rows_key]
        assert sorted(row['seed'] for row in rows) == [0, 1, 2]
        summaries.append({'dataset': dataset, 'path': str(source), 'sha256': digest(source)})
        for row in rows:
            assert row['status'] == 'completed'
            metrics = Path(row['metrics_path'])
            log = metrics.with_name(metrics.name.replace('_cl_metrics.json', '.log'))
            metrics_hash = digest(metrics)
            if 'metrics_sha256' in row:
                assert row['metrics_sha256'] == metrics_hash
            config = dict(re.findall(r'\[trainer\.py\] => (\w+): ([^\n]+)', log.read_text()))
            fields = ['dataset', 'seed', 'init_cls', 'increment', 'nb_tasks', 'class_shuffle',
                      'backbone_type', 'model_name', 'lora_rank', 'batch_size', 'optimizer',
                      'optimizer_type', 'init_epoch', 'epochs', 'init_lr', 'lrate', 'ewc_lambda',
                      'gam_grad_rho', 'gam_grad_norm_rho', 'ewc_fisher_damping_alpha',
                      'in_task_fisher_interval', 'in_task_fisher_omega', 'in_task_weight_merge',
                      'in_task_path_weight_beta']
            assert config['dataset'] == dataset and int(config['seed']) == row['seed']
            assert float(config['gam_grad_norm_rho']) == 1.0
            updates.append({
                'dataset': dataset, 'method': METHOD, 'paper_method': 'FS-EWC',
                'experiment_name': 'GAM-EWC+A, outer radius 1.0',
                'seed': row['seed'], 'include': True, 'status': 'complete',
                'log': str(log), 'sha256': digest(log),
                'metrics_path': str(metrics), 'metrics_sha256': metrics_hash,
                'source_summary': str(source), 'source_run_id': row['id'],
                'expected_sample_faa': row.get('FAA', row.get('faa_sample_weighted')),
                'expected_sample_aaa': row.get('AAA', row.get('aaa_sample_weighted')),
                'logged_config': {key: config[key] for key in fields if key in config},
                'binding': 'Latest requested complete rho=1.0 seeds 0/1/2; replaces historical group.',
            })
    result = {
        'created_date': '2026-09-25',
        'selection_rule': 'Replace only ImageNet-R/A GAM-EWC+Merge with all three supplied rho=1.0 runs.',
        'source_summaries': summaries,
        'reference_manifest': {'path': str(old_path), 'sha256': digest(old_path)},
        'style_reference': str(REFERENCE/'output/lora_taskwise_aa_compact_tighty_no_inflora_display.json'),
        'dataset_order': ['imagenetr', 'imageneta', 'domainnet', 'cifar100'],
        'plotted_methods': ['Vanilla LoRA', 'SD-LoRA', 'EWCLoRA', METHOD],
        'display_alias': {'FS-EWC': METHOD},
        'excluded_methods': ['InfLoRA'],
        'updated_groups': ['imagenetr/GAM-EWC+Merge', 'imageneta/GAM-EWC+Merge'],
        'replaced_historical_runs': replaced,
        'runs': retained + updates,
    }
    (HERE/'sources.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Frozen {len(retained)} retained + {len(updates)} new runs; {len(replaced)} historical runs replaced.')
    return result


if __name__ == '__main__':
    build_sources()
