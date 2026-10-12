#!/usr/bin/env python3
"""Pin existing original logs; do not choose seed subsets by reported accuracy."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/data/140-1/users/liying/low-rank-cl/Previous_Version/low-rank-cl')
COPY = Path('/data/140-1/users/liying/CopyFlatCV')
AUDIT = COPY / 'GAM-EWC/evidence/iclr2027_paper_20260917/iclr_main_results_20260917.json'
ALIASES = {'cifar224':'cifar100','domainnet':'domainnet','imagenetr':'imagenetr','imageneta':'imageneta'}


def pin(run):
    path = Path(run['log'])
    if path.exists():
        raw = path.read_bytes()
        run['sha256'] = hashlib.sha256(raw).hexdigest()
        text = raw.decode(errors='replace')
        config = {}
        for line in text.splitlines():
            match = re.search(r'=> ([\w]+): (.*)$', line)
            if match and match[1] in {'seed','dataset','method','model_name','load','backbone_type',
                'init_cls','increment','sessions','nb_tasks','shuffle','class_shuffle','epochs',
                'init_epoch','lrate','init_lr','fc_lrate','rank','lora_rank','optimizer',
                'optimizer_type','batch_size','weight_decay','lambda','ewc_lambda','gamma','ewc_gamma',
                'gam_grad_rho','gam_grad_norm_rho','gam_grad_gamma','in_task_weight_merge',
                'in_task_weight_merge_omega','in_task_fisher_interval','ewc_fisher_damping_alpha'}:
                config[match[1]] = match[2]
        run['logged_config'] = config
    elif run.get('include',True):
        raise FileNotFoundError(path)
    return run


def main():
    audit = json.loads(AUDIT.read_text())
    runs = []
    for row in audit['raw_runs']:
        if not row['group'].startswith('LoRA_merge_'):
            continue
        run = {'dataset': ALIASES[row['dataset']], 'method':'GAM-EWC+Merge',
               'paper_method':'FS-EWCLoRA','seed':int(row['seed']), 'log':row['log'],
               'include':not row['nonfinite'],
               'status':'complete' if not row['nonfinite'] else 'excluded_training_nan',
               'source_group':row['group'],
               'binding':'historical implementation lineage; full available seed cohort differs from paper'}
        pin(run)
        if run['sha256'] != row['sha256_log']:
            raise ValueError(f"Historical source changed: {row['log']}")
        runs.append(run)
    for dataset, dirname, tasks, complete in [('cifar100','cifar100',10,5),('domainnet','domainnet',5,3),
                                               ('imagenetr','imagenet-r',10,4),('imageneta','imagenet-a',10,5)]:
        directory = COPY / 'logs_ewc/imagenet-a/epoch10' if dataset=='imageneta' else ARCHIVE / f'logs/ewc_lora_redo/{dirname}/t{tasks}'
        for seed in range(5):
            path = directory / f'seed{seed}.log'
            status = 'complete' if seed<complete else ('excluded_incomplete_data_file_error' if path.exists() else 'not_started')
            runs.append(pin({'dataset':dataset,'method':'EWCLoRA','seed':seed,'log':str(path),
                             'include':seed<complete,'status':status,
                             'binding':'table matched' if dataset!='domainnet' else 'table means differ by <=0.01 pp'}))
    reported = []
    for row in audit['paper_rows']:
        method = row['method']
        if method not in ['Vanilla LoRA','InfLoRA','SD-LoRA','CL-LoRA','BiLoRA','EWCLoRA','FS-EWCLoRA']:
            continue
        for dataset, v in row['datasets'].items():
            reported.append({'dataset':ALIASES[dataset],
                             'method': 'GAM-EWC+Merge' if method=='FS-EWCLoRA' else method,
                             'paper_method':method,'accuracy_mean':v['final']['mean'],
                             'accuracy_std':v['final']['sd'],'aaa_mean':v['aaa']['mean'],
                             'aaa_std':v['aaa']['sd']})
    manifest = {'schema_version':1, 'source_audit':str(AUDIT),
                'selection_rule':'All complete finite runs in explicitly identified historical configuration families; no endpoint-based subset selection.',
                'metric':'Per-stage sample-weighted classifier top1. Running AAA computed per seed as cumulative stage mean.',
                'std_ddof':1,'runs':runs,'reported_table':reported,
                'display_labels':{},
                'figure_footnote':'Mean ± sample SD. EWCLoRA n=5/3/4/5; GAM-EWC+Merge n=4/6/6/6 (dataset order).\nComplete finite runs; method-specific training recipes. Missing table baselines are omitted.',
                'caption_note':'Historical method-specific configurations are retained; this is a descriptive comparison, not a matched-budget causal ablation. GAM-EWC+Merge is the implementation traced from the old FS-EWCLoRA row, using all complete finite runs instead of paper-matching seed subsets. CIFAR GAM seeds 3/5 failed with NaN. DomainNet EWC seed 3 and ImageNet-R EWC seed 4 are incomplete; DomainNet EWC seed 4 did not start. Other requested LoRA methods lack table-bound multi-seed curves in the searched archives and are omitted.',
                'unresolved_methods':['Vanilla LoRA','InfLoRA','SD-LoRA','CL-LoRA','BiLoRA']}
    (HERE/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    refs = HERE/'baseline_reference_sources.json'
    if refs.exists():
        manifest = dict(manifest)
        manifest['runs'] = list(runs)
        for row in json.loads(refs.read_text())['rows']:
            if row['method'] not in ('Vanilla LoRA','InfLoRA','SD-LoRA'):
                continue
            manifest['runs'].append(pin({'dataset':ALIASES.get(row['dataset'],row['dataset']),
                                        'method':row['method'],'seed':row['seed'],'log':row['log'],
                                        'include':True,'status':'reference_only_unbound_to_table',
                                        'binding':'different available seed0 run; not a reconstruction of reported table'}))
        manifest['display_labels'] = {m:m+' †' for m in ['Vanilla LoRA','InfLoRA','SD-LoRA']}
        manifest['figure_footnote'] = '† Unbound single-seed references (n=1; no band). EWC n=5/3/4/5; GAM n=4/6/6/6. Bands: mean ± sample SD.\nMethod-specific training recipes; reference endpoints differ from the table. CL-LoRA / BiLoRA raw curves unavailable.'
        manifest['caption_note'] += ' Daggered curves are separately identified available seed-0 reference runs with differing logged configurations and table endpoints; no variance band is estimated from one run. They must not be read as the table baseline replications.'
        (HERE/'sources_with_references.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'Pinned {sum(r["include"] for r in runs)} complete logs; retained {sum(not r["include"] for r in runs)} excluded/unstarted run entries.')


if __name__=='__main__':
    main()
