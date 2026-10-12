"""Generate planning artifacts only; no experiment is launched."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
DS = ['aircraft', 'cars196', 'cub200', 'flowers', 'het5']
METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']
SLUG = dict(zip(METHODS, ['vanilla', 'inflora', 'olora', 'sdlora', 'cllora', 'ewclora', 'gam_ewclora']))
FROZEN = '/data/140-1/users/liying/CopyFlatCV/refine-logs/gam-ewc-fine-gapfill-20260918-attempt2/code'


def read(name):
    return json.loads((ROOT / name).read_text())


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


current = json.loads((ROOT.parent / 'finegrained_std_20260922/statistics.json').read_text())
base = {(g['method'], g['dataset']): g for g in read('baseline_plan.json')['groups']}
ewc = {g['dataset']: g for g in read('ewc_plan.json')['rows']}
cg = {(g['method'], g['dataset']): g for g in read('cl_gam_plan.json')['rows']}
previous = {(g['method'], g['dataset']): g for g in current['cohorts']}
lr = dict(zip(DS, [.05, .05, .01, .0025, .02]))
tasks = dict(zip(DS, [[10]*10, [16]+[20]*9, [20]*10, [12]+[10]*9, [200,100,196,102,37]]))
lam = dict(zip(DS, [200,80000,200000,10,50000]))
gam = dict(zip(DS, [[.01,.1,.02], [.05,.2,.03], [.2,.2,.1], [.2,.2,.1], [.2,.2,.1]]))
cohorts, runs, budget = [], [], []
for method in METHODS:
    for ds in DS:
        old = previous[method, ds]
        conditional = method == 'CL-LoRA' or (method == 'GAM-EWCLoRA' and ds == 'het5')
        proposed_new = [1,2] if conditional else [0,1,2]
        historical_missing = sorted({0,1,2} - set(old['seeds']))
        if (method, ds) in base:
            r = base[method, ds]
            config_source = r['source_config']
            code_source = '/data/140-1/users/liying/Flatness_CV'
            observed_hours = r['historical_log_elapsed_minutes'] / 60
            timing_basis = 'historical_log_last_run_span'
        elif method == 'EWCLoRA':
            r = ewc[ds]
            config_source = r['source_config']
            code_source = FROZEN
            observed_hours = r['reference_elapsed_mean_minutes'] / 60
            timing_basis = 'mean_historical_log_span'
        else:
            r = cg[method, ds]
            config_source = r.get('config_path', r.get('historical_config'))
            code_source = r.get('source_root', FROZEN)
            observed_hours = r.get('wall_hours', r.get('historical_log_span_hours'))
            timing_basis = 'receipt_process_wall_hours' if 'wall_hours' in r else 'historical_log_span'
        lr_this = .02 if method == 'SDLoRA' and ds == 'aircraft' else lr[ds]
        epochs = 40 if ds != 'het5' or method in ['InfLoRA', 'OLoRA', 'SDLoRA'] else 30
        protocol = dict(lrate=lr_this, init_lr=lr_this, epochs=epochs, init_epoch=epochs,
                        batch_size=128, lora_rank=16, optimizer='sgd', momentum=0,
                        weight_decay=0, scheduler='cosine', class_shuffle=False,
                        task_splits=tasks[ds], min_lr=0 if method == 'CL-LoRA' else .1*lr_this,
                        precision='fp32', seed_values=[0,1,2])
        if method == 'InfLoRA':
            protocol.update(lamb=.7, lame=.99)
        if method == 'OLoRA':
            protocol.update(olora_lamda_1=1, olora_lamda_2=0, learn_alpha=False)
        if method == 'SDLoRA':
            protocol.update(learn_alpha=True, max_prev_tasks=200)
        if method == 'CL-LoRA':
            protocol.update(shared_pos=list(range(6)), specific_pos=list(range(6,12)),
                            msa=[1,0,1], kd_ratio=5, temperature=2, orth_ratio=1e-4)
        if method in ['EWCLoRA', 'GAM-EWCLoRA']:
            protocol.update(ewc_lambda=lam[ds] if method == 'EWCLoRA' else 2000,
                            ewc_gamma=1, ewc_eta=0, ewc_max_batches=100,
                            ewc_normalize_fisher=True, ewc_normalize_accumulated_fisher=False,
                            fisher_mode='batch_empirical', ewc_fisher_damping_beta=0,
                            ewc_dual_ascent=False)
        if method == 'EWCLoRA':
            protocol.update(implementation='native' if ds == 'het5' else 'alpha0_wrapper',
                            fs_alpha=0,
                            fs_direction_mode='native_sgd' if ds == 'het5' else ('alpha0' if ds in ['cars196','flowers'] else 'ratio'),
                            fs_preserve_probe_rng=ds in ['cars196','flowers','het5'],
                            ewc_preserve_fisher_rng=ds in ['cars196','flowers','het5'])
        if method == 'GAM-EWCLoRA':
            rho,nrho,gamma = gam[ds]
            protocol.update(optimizer_type='gam', gam_grad_rho=rho, gam_grad_norm_rho=nrho,
                            gam_grad_gamma=gamma, gam_grad_beta_1=1, gam_grad_beta_2=1,
                            gam_grad_beta_3=1, gam_adaptive=False,
                            ewc_preserve_fisher_rng=ds == 'het5',
                            effective_clean_rng_isolation=False,
                            effective_deterministic_loader_rng=False)
        group = dict(cohort_id=f'{SLUG[method]}_{ds}', method=method, dataset=ds,
                     target_seeds=[0,1,2], existing_historical_seeds=old['seeds'],
                     nominal_missing_seeds=historical_missing, proposed_new_seeds=proposed_new,
                     conditional_reuse_seeds=[0] if conditional else [],
                     historical_results_policy='conditional_seed0_reuse' if conditional else 'reference_only',
                     selected_config_source=config_source,
                     proposed_source_tree_to_freeze=code_source,
                     proposed_core_protocol=protocol, historical_runs=old['runs'],
                     runtime_preflight_complete=False, training_started=False)
        cohorts.append(group)
        budget.append(dict(method=method, dataset=ds, historical_single_run_hours=observed_hours,
                           timing_basis=timing_basis,
                           nominal_missing_count=len(historical_missing),
                           proposed_new_count=len(proposed_new),
                           nominal_missing_linear_hours=len(historical_missing)*observed_hours,
                           proposed_new_linear_hours=len(proposed_new)*observed_hours))
        for seed in [0,1,2]:
            reuse = conditional and seed == 0
            runs.append(dict(run_id=f'{SLUG[method]}_{ds}_s{seed}', method=method, dataset=ds,
                             seed=seed, stage='REUSE_CHECK' if reuse else 'FORMAL',
                             status='PENDING_PROVENANCE_CHECK' if reuse else 'PLANNED_NOT_STARTED',
                             nominal_historical_missing=seed in historical_missing,
                             proposed_new_run=not reuse,
                             expected_tasks=len(tasks[ds]), epochs_per_task=epochs,
                             lr=lr_this, selected_config_source=config_source,
                             source_tree_to_freeze=code_source,
                             output_key=f'seed012_current_style/{SLUG[method]}/{ds}/s{seed}/attempt1'))

assert len(cohorts) == 35 and len(runs) == 105
assert sum(r['nominal_historical_missing'] for r in runs) == 68
assert sum(r['proposed_new_run'] for r in runs) == 99
assert len({r['run_id'] for r in runs}) == 105
assert all(len(set(g['target_seeds'])) == 3 for g in cohorts)
assert sum(len(g['conditional_reuse_seeds']) for g in cohorts) == 6
save('COHORTS.json', cohorts)
save('TARGET_PROTOCOL_DRAFT.json', dict(status='PLAN_ONLY_NOT_EXECUTABLE',
     scope='current-style, within-cohort matched seeds; not globally harmonized',
     baseline_source='complete Flatness_CV snapshot to be created',
     other_source=FROZEN, target_seeds=[0,1,2],
     requirements=['resolve effective defaults before launch', 'immutable source/env/data/checkpoint',
                   'independent model and optimizer initialization', 'no tuning based on seed1/2 test scores',
                   'no mixing failed, partial or different-protocol runs', 'preserve actual legacy RNG behavior for reused seed0 cohorts'],
     cohorts=[{k:g[k] for k in ['cohort_id','selected_config_source','proposed_source_tree_to_freeze','proposed_core_protocol']} for g in cohorts]))
with (ROOT/'RUN_MATRIX.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(runs[0]));writer.writeheader();writer.writerows(runs)
counts=dict(cohorts=35, target_total_runs=105, existing_historical_runs=37,
            nominal_additional_runs_if_all_old_runs_compatible=68,
            recommended_new_formal_runs=99, conditional_reuse_runs=6,
            upper_bound_without_reuse=105, optional_gam_protocol_switch_new_runs=95,
            planned_smoke_runs=9, smoke_runs_included_in_formal_count=False,
            training_started=False,
            recommended_new_by_method=dict(Counter(r['method'] for r in runs if r['proposed_new_run'])))
save('PLAN_COUNTS.json',counts)
total=sum(b['proposed_new_linear_hours'] for b in budget)
save('BUDGET.json',dict(interpretation='Mixed historical log intervals and receipt process wall times, mechanically extrapolated; log intervals omit pre-log startup/queueing and may include unknown waits. Not measured process/GPU-hours or promised completion time.',
                       recommended_formal_mixed_historical_hours_proxy=total,
                       nominal_missing_mixed_historical_hours_proxy=sum(b['nominal_missing_linear_hours'] for b in budget),
                       planning_proxy_with_20pct_reserve=total*1.2,
                       planning_proxy_with_30pct_reserve=total*1.3,
                       measured_gpu_hours=None, current_gpu_availability='unavailable: local nvidia-smi driver communication failed',
                       rows=budget))
tracker=['# seed 0/1/2 实验跟踪表','',
         '计划状态：未启动。99条待新跑，6条旧seed0待来源校验；后者若失败则改为新跑。Smoke不计入本表。','',
         '| Run ID | 阶段 | 方法 | 数据集 | Seed | 状态 | 任务×epochs |',
         '|---|---|---|---|---:|---|---|']
tracker += [f"| {r['run_id']} | {r['stage']} | {r['method']} | {r['dataset']} | {r['seed']} | {r['status']} | {r['expected_tasks']}×{r['epochs_per_task']} |" for r in runs]
(ROOT/'EXPERIMENT_TRACKER.md').write_text('\n'.join(tracker)+'\n')
save('VALIDATION.json', dict(counts_verified=True, unique_run_ids=105, seeds_per_cohort=[0,1,2],
     nominal_missing_by_dataset=dict(Counter(r['dataset'] for r in runs if r['nominal_historical_missing'])),
     no_standalone_pets_cohort=True, pets_retained_in_het5=True,
     no_training_launched=True, executable_configuration_validated=False,
     structural_validation_only=True))
stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
for name in ['EXPERIMENT_PLAN','EXPERIMENT_TRACKER']:
    content=(ROOT/f'{name}.md').read_bytes()
    (ROOT/f'{name}_{stamp}.md').write_bytes(content)
    (ROOT/f'{name}.md').write_bytes(content)
print(json.dumps(counts,ensure_ascii=False))
print('mixed historical interval-hours proxy',total,'reserve range',total*1.2,total*1.3)
