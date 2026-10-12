#!/usr/bin/env python
"""CPU-only descriptive analysis; keeps three seeds as independent units."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, statistics
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
BRANCHES = ['S', 'R']
sources = {}

def read_json(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

def read_csv(name):
    path = ROOT / name
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))

def summary(values):
    return {'n_seeds': len(values), 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'values': values}

def fmt(d, precision=3):
    return f"{d['mean']:.{precision}f} ± {d['sample_sd']:.{precision}f}"

def numeric_finite(value):
    if isinstance(value, dict): return all(numeric_finite(v) for v in value.values())
    if isinstance(value, list): return all(numeric_finite(v) for v in value)
    if isinstance(value, float): return math.isfinite(value)
    return True

curves, factors, decompositions, valid = {}, {}, {}, []
for seed in SEEDS:
    for branch in BRANCHES:
        job = f'formal_{branch}_s{seed}'
        receipt = read_json(ROOT / 'runtime/jobs' / job / 'completed.json')
        final_path = ROOT / 'outputs' / job / 'final_summary.json'
        final = read_json(final_path)
        assert receipt['exit']['returncode'] == 0
        assert receipt['summary_sha256'] == hashlib.sha256(final_path.read_bytes()).hexdigest()
        assert final['seed'] == seed and final['branch'] == branch and not final['smoke']
        assert final['completed_steps'] == 400 and final['completed_epochs'] == 50
        assert all(final['checks'].values())
        retention = read_json(ROOT / 'outputs' / job / 'retention_curve.json')['epochs']
        clean = read_json(ROOT / 'outputs' / job / 'clean_training_curve.json')['epochs']
        ff = read_json(ROOT / 'outputs' / job / 'factor_state_curve.json')['epochs']
        dd = read_json(ROOT / 'outputs' / job / 'factor_update_decomposition.json')['steps']
        assert len(retention) == 51 and len(clean) == 50 and len(ff) == 50 and len(dd) == 100
        assert numeric_finite(retention) and numeric_finite(clean) and numeric_finite(ff) and numeric_finite(dd)
        assert all(len(e['layers']) == 24 for e in ff + dd)
        curves[(seed, branch)] = {e['epoch']: e for e in retention}
        clean_map = {e['epoch']: e for e in clean}
        curves[(seed, branch)][0]['clean_training_ce'] = final['clean_training_start']['ce']
        for e in range(1, 51):
            curves[(seed, branch)][e]['clean_training_ce'] = clean_map[e]['ce']
        factors[(seed, branch)] = {e['epoch']: e for e in ff}
        decompositions[(seed, branch)] = dd
        valid.append({'job': job, 'seed': seed, 'branch': branch,
                      'accepted': True, 'validator_checks': receipt['validated_checks'],
                      'completion_at_utc': receipt['completed_at']})

fit = read_csv('FIT_MATCHED_COMPARISONS.csv')
fit_seed = []
for seed in SEEDS:
    rr = [r for r in fit if int(r['seed']) == seed]
    item = {'seed': seed, 'common_bins': len(rr),
            'ce_range': [min(float(r['ce_bin_low']) for r in rr), max(float(r['ce_bin_high']) for r in rr)],
            'mean_absolute_clean_ce_gap': statistics.mean(abs(float(r['R_clean_training_ce'])-float(r['S_clean_training_ce'])) for r in rr),
            'max_absolute_clean_ce_gap': max(abs(float(r['R_clean_training_ce'])-float(r['S_clean_training_ce'])) for r in rr)}
    for name in ['old_accuracy_R_minus_S', 'new_accuracy_R_minus_S', 'new_conditional20_ce_R_minus_S']:
        vv = [float(r[name]) for r in rr]
        item[name] = {'descriptive_mean_across_bins': statistics.mean(vv),
                      'positive_bins': sum(v>0 for v in vv), 'zero_bins': sum(v==0 for v in vv),
                      'negative_bins': sum(v<0 for v in vv), 'min': min(vv), 'max': max(vv)}
    item['median_factor_R_minus_S'] = {}
    for name, key in [('qhat', 'cross_layer_qhat'), ('qhat_change', 'cross_layer_qhat_change'), ('q_change_fro', 'cross_layer_q_change_fro')]:
        vv = [factors[(seed,'R')][int(r['R_first_epoch'])][key]['median'] - factors[(seed,'S')][int(r['S_first_epoch'])][key]['median'] for r in rr]
        item['median_factor_R_minus_S'][name] = {'descriptive_mean_across_bins': statistics.mean(vv), 'values': vv}
    fit_seed.append(item)

selected = {}
for epoch in [1, 10, 25, 50]:
    selected[epoch] = {}
    for branch in BRANCHES:
        selected[epoch][branch] = {}
        for label, key in [('qhat','cross_layer_qhat'), ('qhat_change','cross_layer_qhat_change'), ('q_change_fro','cross_layer_q_change_fro')]:
            selected[epoch][branch][label] = summary([factors[(s,branch)][epoch][key]['median'] for s in SEEDS])
            selected[epoch][branch][label]['within_seed_q25'] = [factors[(s,branch)][epoch][key]['q25'] for s in SEEDS]
            selected[epoch][branch][label]['within_seed_q75'] = [factors[(s,branch)][epoch][key]['q75'] for s in SEEDS]
        for label in ['lora_norm', 'fc_norm', 'joint_norm']:
            selected[epoch][branch][label] = summary([factors[(s,branch)][epoch]['effective_displacement'][label] for s in SEEDS])
        selected[epoch][branch]['fc_squared_norm_share_of_joint'] = summary([factors[(s,branch)][epoch]['effective_displacement']['fc_norm']**2 / factors[(s,branch)][epoch]['effective_displacement']['joint_norm']**2 for s in SEEDS])

update_phase = []
for seed in SEEDS:
    for branch in BRANCHES:
        for first, last in [(1,25),(26,50)]:
            steps = [e for e in decompositions[(seed,branch)] if first <= e['epoch'] <= last]
            layers = [r for e in steps for r in e['layers']]
            phase = {'seed':seed,'branch':branch,'epoch_range':[first,last], 'sampled_steps':len(steps), 'layer_records':len(layers)}
            for name in ['pi_A','T_A_T_B_cosine','delta_q_first_fro','delta_q_second_fro','delta_q_fro','T_A_norm','T_B_norm','T_cross_norm','actual_delta_W_norm']:
                vv = [r[name] for r in layers if r[name] is not None]
                phase[name] = {'descriptive_median': statistics.median(vv), 'q25':float(np.quantile(vv,.25)), 'q75':float(np.quantile(vv,.75))}
            phase['max_q_identity_relative_error'] = max(r['delta_q_identity_relative_error'] for r in layers)
            phase['max_q_identity_state_scale_error'] = max(r['delta_q_identity_state_scale_error'] for r in layers)
            phase['median_cross_over_linear_effective_norm'] = statistics.median(r['T_cross_norm']/(r['T_A_norm']+r['T_B_norm']+1e-30) for r in layers)
            update_phase.append(phase)

# Effective endpoint differences directly compare BA and all FC tensors. This is
# a CPU-only derived metric, not a candidate update or a training modification.
direct = []
for seed in SEEDS:
    for epoch in [10,25,50]:
        states = {}
        for branch in BRANCHES:
            path = ROOT / 'outputs' / f'formal_{branch}_s{seed}' / 'checkpoints' / f'epoch_{epoch:03d}.pt'
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            cp = torch.load(path, map_location='cpu', weights_only=False)
            assert cp['epoch'] == epoch and cp['seed'] == seed and cp['branch'] == branch
            states[branch] = cp['trainable_state']
        akeys = [k for k in states['S'] if '.linear_a_' in k]
        assert len(akeys) == 24
        lora_s2, fc_s2, perlayer = 0., 0., []
        for ak in akeys:
            bk = ak.replace('.linear_a_', '.linear_b_')
            aS,bS = states['S'][ak].double(), states['S'][bk].double()
            aR,bR = states['R'][ak].double(), states['R'][bk].double()
            dw = bR@aR - bS@aS
            n = float(dw.norm()); lora_s2 += n*n
            perlayer.append({'layer_name':ak.rsplit('.linear_a_',1)[0] + ':' + ak.rsplit('.linear_a_',1)[1], 'BA_R_minus_S_norm':n})
        fckeys = [k for k in states['S'] if k not in akeys and '.linear_b_' not in k]
        for k in fckeys:
            assert k.startswith('fc.'), k
            fc_s2 += float((states['R'][k].double()-states['S'][k].double()).square().sum())
        direct.append({'seed':seed,'epoch':epoch,'lora_BA_difference_norm':math.sqrt(lora_s2),'FC_difference_norm':math.sqrt(fc_s2),'joint_difference_norm':math.sqrt(lora_s2+fc_s2),'FC_squared_norm_share':fc_s2/(lora_s2+fc_s2),'all_24_layers':perlayer})

tail = []
paired = []
for seed in SEEDS:
    for branch in BRANCHES:
        c25,c50 = (curves[(seed,branch)][e] for e in [25,50])
        tail.append({'seed':seed,'branch':branch,
                     'old_50_minus_25':c50['old_test_allseen40']['accuracy']-c25['old_test_allseen40']['accuracy'],
                     'new_50_minus_25':c50['current_test_allseen40']['accuracy']-c25['current_test_allseen40']['accuracy'],
                     'FAA_50_minus_25':c50['allseen40_task_macro_accuracy']-c25['allseen40_task_macro_accuracy']})
    for epoch in [25,50]:
        s,r = (curves[(seed,b)][epoch] for b in BRANCHES)
        paired.append({'seed':seed,'epoch':epoch,
            'old_R_minus_S':r['old_test_allseen40']['accuracy']-s['old_test_allseen40']['accuracy'],
            'new_R_minus_S':r['current_test_allseen40']['accuracy']-s['current_test_allseen40']['accuracy'],
            'FAA_R_minus_S':r['allseen40_task_macro_accuracy']-s['allseen40_task_macro_accuracy'],
            'clean_training_CE_R_minus_S':r['clean_training_ce']-s['clean_training_ce'],
            'new_conditional20_CE_R_minus_S':r['current_test_conditional20']['ce']-s['current_test_conditional20']['ce']})
tail_summary = {b:{m:summary([r[m] for r in tail if r['branch']==b]) for m in ['old_50_minus_25','new_50_minus_25','FAA_50_minus_25']} for b in BRANCHES}
paired_summary = {e:{m:summary([r[m] for r in paired if r['epoch']==e]) for m in ['old_R_minus_S','new_R_minus_S','FAA_R_minus_S','clean_training_CE_R_minus_S','new_conditional20_CE_R_minus_S']} for e in [25,50]}

report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'campaign':str(ROOT), 'analysis_type':'CPU-only descriptive, no training and no GPU',
    'formal_validity_ledger':{'expected':6,'accepted':len(valid),'missing':0,'failed_formal':0,'runs':valid},
    'independent_units':{'seeds':SEEDS,'n':3,'not_independent':'31 fit bins, 24 layers, 50 epochs and 100 sampled updates per branch'},
    'factor_state_record_count':6*50*24,'factor_update_record_count':6*100*24,
    'paired_performance_raw':paired,'paired_performance_summary':paired_summary,'common_SGD_tail_raw':tail,'common_SGD_tail_summary':tail_summary,
    'fit_matching':{'preregistered_bin_width':.05,'range':[0,5],'selection':'first entry per branch; shared bins only; no test selection',
                    'bin_count':len(fit),'per_seed_descriptive':fit_seed,
                    'limitations':['Bins share the same trajectory and are not independent.', 'A width .05 bin matches fit approximately, not exactly; matched epochs and cosine learning rates may differ.', 'Coverage differs across seeds; do not turn pooled 31-bin win counts into significance.']},
    'factor_selected_epochs':selected,'actual_update_phase_descriptive':update_phase,
    'direct_effective_state_R_minus_S':direct,
    'supported_claims':[
        'S_F25 followed by common SGD25 has a positive paired FAA difference in all three seeds at e25 and e50 under this two-task protocol.',
        'The advantage persists through the common SGD tail; its mean decreases slightly rather than increasing.',
        'The final advantage is mainly new-task accuracy; final old-task accuracy ties in seeds 1/2 and improves only in seed 0.',
        'New conditional test CE is lower in every preregistered shared fit bin, a descriptive signal that is not explained solely by less new-task fitting.',
        'All-layer factor statistics and directly measured BA and FC endpoint differences show distinct effective states rather than only an equivalent reparameterization.'],
    'unsupported_claims':[
        'A consistent reduction of forgetting across all seeds.',
        'Qhat reduction or a changed A/B update allocation uniquely causes better performance.',
        'The benefit is independent of update scale or classifier-head changes; same-state matched diagnostics remain a separate prerequisite.',
        'The exact Hessian response explanation is established; numerical gate status must be reported separately.',
        'The result transfers to ten tasks, a pure SGD baseline, or full fine-tuning; those cells are outside this design.'],
    'sources_sha256':sources}
(OUT/'FACTOR_FIT_ANALYSIS_20261012.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

text = ['# Factor, fitting and common-SGD-tail analysis (2026-10-12)', '',
        'Six of six formal branches are accepted. This analysis uses three paired seeds; layers, epochs, sampled updates and fit bins are repeated observations.', '',
        'S: SAM for Task1 epochs 1–25, then SGD for 26–50. R: native S_F g2 for 1–25, then the same SGD schedule for 26–50. Both start from one shared Task0-SGD50 state per seed. This is a two-task protocol, distinct from prior ten-task results.', '',
        '## Paired outcomes', '', '| Epoch | Old R−S (pp) | New R−S (pp) | FAA R−S (pp) | Clean train CE R−S | New conditional test CE R−S |', '|---|---:|---:|---:|---:|---:|']
for e in [25,50]:
    p=paired_summary[e]
    text.append(f"| {e} | {fmt(p['old_R_minus_S'])} | {fmt(p['new_R_minus_S'])} | {fmt(p['FAA_R_minus_S'])} | {fmt(p['clean_training_CE_R_minus_S'],5)} | {fmt(p['new_conditional20_CE_R_minus_S'],5)} |")
text += ['', 'FAA improves in all three seeds at both endpoints. At e50, old accuracy improves in seed 0 and is exactly equal in seeds 1 and 2. New accuracy is equal in seed 0 and improves in seeds 1 and 2. The gain therefore cannot be described as consistent forgetting reduction.', '',
        f"During common SGD, FAA increases by {fmt(tail_summary['S']['FAA_50_minus_25'])} pp for S and {fmt(tail_summary['R']['FAA_50_minus_25'])} pp for R. The paired advantage changes from {paired_summary[25]['FAA_R_minus_S']['mean']:.3f} to {paired_summary[50]['FAA_R_minus_S']['mean']:.3f} pp: persistence, not amplification.", '',
        '## Fitting match (descriptive)', '', '| Seed | Shared first-entry bins | Mean old R−S (pp) | Mean new R−S (pp) | Mean conditional test CE R−S | CE-bin range |', '|---|---:|---:|---:|---:|---|']
for d in fit_seed:
    text.append(f"| {d['seed']} | {d['common_bins']} | {d['old_accuracy_R_minus_S']['descriptive_mean_across_bins']:.3f} | {d['new_accuracy_R_minus_S']['descriptive_mean_across_bins']:.3f} | {d['new_conditional20_ce_R_minus_S']['descriptive_mean_across_bins']:.5f} | [{d['ce_range'][0]:.2f}, {d['ce_range'][1]:.2f}) |")
text += ['', 'Conditional new-task test CE is lower for R in all 31 shared bins. New joint40 accuracy never decreases in these bins; old accuracy is mixed across seeds. This supports a descriptive improvement in new-task testing at approximately matched fitting, while old-task retention remains unresolved.', '',
        'Each bin has width .05 and selects the first entering epoch. Fitting is approximate; epoch and cosine LR can differ between branches. The 31 bins are not 31 independent trials, and seed coverage differs. No significance claim follows from a pooled bin win rate.', '',
        '## All-layer factor and effective-state evidence', '', '| Epoch | Branch | Median Qhat (3-seed mean±SD) | Median Qhat change | Median Q drift (Frobenius norm) | LoRA displacement from start | FC displacement from start |', '|---|---|---:|---:|---:|---:|---:|']
for e in [25,50]:
    for b in BRANCHES:
        f=selected[e][b]
        text.append(f"| {e} | {b} | {fmt(f['qhat'],6)} | {fmt(f['qhat_change'],6)} | {fmt(f['q_change_fro'],6)} | {fmt(f['lora_norm'],5)} | {fmt(f['fc_norm'],5)} |")
text += ['', 'All 24 Q/V modules are included, with within-seed medians and IQR preserved in JSON. Across seeds R has larger effective LoRA displacement and larger Q drift from the common start. Qhat decreases more in R, but its absolute change is small and is not evidence of a unique beneficial balancing mechanism.', '',
        'Direct CPU checkpoint comparison measures ||BA_R−BA_S|| and FC differences, so the two states differ in effective weights rather than merely in factor coordinates. The FC contributes more than 95% of the squared joint displacement from the shared start in both arms; its role is not experimentally isolated.', '',
        '| Epoch | BA R−S difference norm (mean±SD) | FC R−S difference norm (mean±SD) | Joint norm (mean±SD) |', '|---|---:|---:|---:|']
for e in [10,25,50]:
    d=[r for r in direct if r['epoch']==e]
    text.append(f"| {e} | {fmt(summary([r['lora_BA_difference_norm'] for r in d]),5)} | {fmt(summary([r['FC_difference_norm'] for r in d]),5)} | {fmt(summary([r['joint_difference_norm'] for r in d]),5)} |")
maxerr=max(r['max_q_identity_relative_error'] for r in update_phase)
stateerr=max(r['max_q_identity_state_scale_error'] for r in update_phase)
text += ['', f'The actual ΔQ identity is accurate: maximum relative reconstruction error {maxerr:.3g}; maximum state-scale error {stateerr:.3g}, over 14,400 layer-update records. Reconstructing an algebraic identity validates measurement, not a causal mechanism.', '',
        'During the common SGD tail, median first-order ΔQ is approximately 5e−9, near the residual from committing FP32 factor tensors; finite-step second-order drift remains. Earlier SAM/S_F probes produce larger first-order drift. Under a loss depending on BA, equal-LR base-point SGD cancels the ideal first-order imbalance change algebraically; a gradient evaluated at displaced factors need not cancel when committed at the base. The observations are consistent with that structure, but do not show that the imbalance drives generalization.', '',
        '## Claim boundaries', '', 'Supported: a response-trained Task1 state gives higher two-task FAA, mainly through new-task testing, and the difference remains after 25 common SGD epochs. Effective LoRA and FC state differences are directly observed.', '',
        'Not established: consistent old-task retention, unique mediation by Qhat or A/B allocation, scale independence, head independence, exact Hessian correspondence, or transfer to ten tasks/full fine-tuning. Offline numerical and matched-step diagnostics must be reported separately.', '',
        'No new training, GPU use, threshold changes or frozen-file edits were performed. The complete JSON includes the source hashes, all seed records, all-layer summaries and claim boundaries.']
(OUT/'FACTOR_FIT_ANALYSIS_20261012.md').write_text('\n'.join(text)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'S':'#31688e','R':'#d95f02'}
labels={'S':'SAM25 → SGD25','R':'S_F g2 (25) → SGD25'}
def draw_curve(ax, getter, epochs=range(51), log=False):
    x=np.array(list(epochs))
    for b in BRANCHES:
        vals=np.array([[getter(curves[(s,b)][int(e)]) for e in x] for s in SEEDS])
        mean,sd=vals.mean(0),vals.std(0,ddof=1)
        ax.plot(x,mean,color=colors[b],label=labels[b],lw=1.8)
        ax.fill_between(x,mean-sd,mean+sd,color=colors[b],alpha=.16)
    ax.axvline(25,color='#555555',ls='--',lw=1)
    ax.set_xlabel('Task1 epoch');ax.grid(alpha=.15)
    if log:ax.set_yscale('log')

fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
draw_curve(axes[0,0],lambda d:d['old_test_allseen40']['accuracy']);axes[0,0].set_ylabel('Old-task accuracy, joint40 (%)')
draw_curve(axes[0,1],lambda d:d['current_test_allseen40']['accuracy']);axes[0,1].set_ylabel('New-task accuracy, joint40 (%)')
draw_curve(axes[1,0],lambda d:d['allseen40_task_macro_accuracy']);axes[1,0].set_ylabel('Task-macro FAA (%)')
draw_curve(axes[1,1],lambda d:d['clean_training_ce'],log=True);axes[1,1].set_ylabel('Clean new-task train CE (20 classes)')
axes[0,0].legend(loc='lower left',frameon=False)
fig.suptitle('ImageNet-A: paired state formation followed by common SGD\nMean ± sample SD, seeds 0/1/2; dashed line = switch after epoch 25',fontsize=12)
fig.savefig(OUT/'PERFORMANCE_AND_FIT_20261012.png',dpi=200)
fig.savefig(OUT/'PERFORMANCE_AND_FIT_20261012.pdf')

with PdfPages(OUT/'FACTOR_FIT_EVIDENCE_20261012.pdf') as pdf:
    pdf.savefig(fig)
    fig2,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,key,label in zip(axs.flat,['cross_layer_qhat_change','cross_layer_q_change_fro','lora_norm','fc_norm'],['Cross-layer median Qhat − Qhat start','Cross-layer median ||Q − Qstart||F','Effective LoRA displacement from start','Classifier displacement from start']):
        for b in BRANCHES:
            vv=[]
            for s in SEEDS:
                vv.append([factors[(s,b)][e][key]['median'] if key.startswith('cross_layer') else factors[(s,b)][e]['effective_displacement'][key] for e in range(1,51)])
            vv=np.array(vv);m,sd=vv.mean(0),vv.std(0,ddof=1)
            ax.plot(range(1,51),m,color=colors[b],label=labels[b],lw=1.8)
            ax.fill_between(range(1,51),m-sd,m+sd,color=colors[b],alpha=.16)
        ax.axvline(25,color='#555555',ls='--',lw=1);ax.set_xlabel('Task1 epoch');ax.set_ylabel(label);ax.grid(alpha=.15)
    axs[0,0].legend(frameon=False)
    fig2.suptitle('All 24 Q/V layers: each trajectory reduced to its predeclared layer median\nBands are seed SD, not layer uncertainty',fontsize=12)
    pdf.savefig(fig2)
    fig2.savefig(OUT/'FACTOR_STATE_20261012.png',dpi=200)
    fig3,axs=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,seed in zip(axs,SEEDS):
        rr=[r for r in fit if int(r['seed'])==seed]
        x=np.array([float(r['ce_bin_low'])+.025 for r in rr])
        ax.plot(x,[float(r['new_accuracy_R_minus_S']) for r in rr],'o-',color=colors['R'],label='New joint40 accuracy')
        ax.plot(x,[float(r['old_accuracy_R_minus_S']) for r in rr],'s-',color=colors['S'],label='Old joint40 accuracy')
        ax.axhline(0,color='#777777',lw=1);ax.set_title(f'Seed {seed} ({len(rr)} common bins)');ax.set_xlabel('Clean train CE bin midpoint');ax.grid(alpha=.15)
    axs[0].set_ylabel('First-entry matched-bin R − S (pp)');axs[0].legend(frameon=False)
    fig3.suptitle('Descriptive approximate fit matching: bins are repeated observations',fontsize=12)
    pdf.savefig(fig3)
plt.close('all')
print(json.dumps({'output':str(OUT),'formal_accepted':len(valid),'fit_bins':len(fit),'seed_mean_new_fit_accuracy_gains':[r['new_accuracy_R_minus_S']['descriptive_mean_across_bins'] for r in fit_seed], 'direct_effective_differences':[{k:v for k,v in r.items() if k!='all_24_layers'} for r in direct],'paired_summary':paired_summary},indent=2))
