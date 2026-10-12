"""CPU-only two-method redraw with measured-grid, per-seed tolerance radii."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT.parent
REFERENCE = BASE / 'radius50_integer_seed0_halfdirs_20260921/figures/replot_20260922_v2'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/rq2-two-method-tolerance-mpl')
GROUPS = ('ewc', 'probe005_norm1')
LABELS = {'ewc': 'EWCLoRA', 'probe005_norm1': 'FS-EWCLoRA'}
COLORS = {'ewc': '#0072B2', 'probe005_norm1': '#009E73'}
MARKERS = {'ewc': 'o', 'probe005_norm1': 's'}
LINES = {'ewc': '-', 'probe005_norm1': '-.'}
RADII = [i / 2 for i in range(11)]
EPSILONS = [.25, .5, 1., 1.5, 2., 2.5, 3.]
REFERENCE_EPSILONS = [.005, .01, .02, .03, .04, .05]
JOBS = {0: 'task0_positive', 1: 'task1_negative'}
METRICS = ('D_train', 'train_accuracy_percent', 'raw_fisher_trace', 'absolute_train_ce')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def main(task=None):
    pc = load('two_methods_plot_style', BASE / 'figures_formal_20260921/plot_common.py')
    reference = load('reference_measured_tolerance', REFERENCE / 'plot_b1.py')
    published = read(ROOT / 'report/radius_intervention_complete.json')
    assert published['status'] == 'complete' and published['validated_jobs'] == 12
    trusted = {x['path']: x['sha256'] for x in published['source_files'] + published['output_files']}
    sources = {}

    def verified(path):
        path = Path(path).resolve()
        digest = sha(path)
        assert digest == trusted[str(path)], f'Source changed: {path}'
        sources[str(path)] = digest
        return path

    frame = pc.pd.read_csv(verified(ROOT / 'figures/all_per_seed.csv'))
    frame = frame[frame.group.isin(GROUPS)].copy()
    assert len(frame) == 132 and set(frame.group) == set(GROUPS)
    assert not frame.duplicated(['gradient_task', 'group', 'seed', 'radius']).any()
    assert set(frame.gradient_task) == {0, 1} and set(frame.seed) == {0, 1, 2}
    assert (frame.directions == 1).all()
    assert (frame.train_samples == 1259).all() and (frame.fisher_samples == 256).all()
    assert (frame.D_train >= 0).all()
    pc.np.testing.assert_allclose(frame.D_train, frame.G_positive_train, rtol=0, atol=0)
    # Rebind every selected CSV row to its saved raw measurement, not other arms.
    for (direction, group, seed), curve in frame.groupby(['gradient_task', 'group', 'seed']):
        assert sorted(curve.radius.tolist()) == RADII
        if group == 'ewc':
            folder, job = (('fixed_gradient_extension_20260921', 'fixed_gradient') if direction == 0
                           else ('task1_gradient_extension_20260921', 'task1_gradient'))
            directory = BASE / folder / f'results/seed{seed}/sgd/{job}'
        else:
            directory = ROOT / f'results/{group}/seed{seed}/gam/{JOBS[direction]}'
        receipt = read(verified(directory / 'extension_receipt.json'))
        assert receipt['status'] == 'complete'
        raw_path = verified(directory / 'states.jsonl')
        assert sha(raw_path) == receipt['states_sha256']
        rows = [json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
        points = {float(r['point']['radius']): r for r in rows}
        origin = points[0]['metrics']['train']['loss']
        for row in curve.itertuples():
            raw = points[row.radius]
            fisher = raw['fisher_by_scope']['subset']
            assert fisher['normalized'] is False and fisher['regularizer'] is None
            assert fisher['samples'] == 256 and raw['metrics']['train']['samples'] == 1259
            expected = dict(D_train=raw['metrics']['train']['loss']-origin,
                            train_accuracy_percent=raw['metrics']['train']['accuracy'],
                            raw_fisher_trace=fisher['raw_trace'],
                            absolute_train_ce=raw['metrics']['train']['loss'])
            for metric, value in expected.items():
                assert math.isclose(getattr(row, metric), value, rel_tol=1e-10, abs_tol=1e-12)
    aggregate_rows = []
    for key, curve in frame.groupby(['gradient_task', 'group', 'radius']):
        assert sorted(curve.seed.tolist()) == [0, 1, 2]
        for metric in METRICS:
            values = curve.sort_values('seed')[metric].to_numpy(float)
            aggregate_rows.append(dict(gradient_task=int(key[0]), group=key[1], radius=key[2],
                metric=metric, mean=float(values.mean()), sample_sd=float(values.std(ddof=1)), n=3))
    aggregate = pc.pd.DataFrame(aggregate_rows)
    prior = pc.pd.read_csv(verified(ROOT / 'figures/all_mean_sample_sd.csv'))
    for row in aggregate.itertuples():
        saved = prior[(prior.gradient_task == row.gradient_task) & (prior.group == row.group)
                      & (prior.radius == row.radius) & (prior.metric == row.metric)].iloc[0]
        assert math.isclose(row.mean, saved['mean'], rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(row.sample_sd, saved.sample_sd, rel_tol=1e-10, abs_tol=1e-12)
    # Directly reuse the requested reference's prefix-safe inversion function.
    # Override its threshold list in memory only; the source file is unchanged.
    assert reference.EPSILONS == REFERENCE_EPSILONS
    reference.EPSILONS = REFERENCE_EPSILONS + EPSILONS
    tolerance_frames = []
    for direction in (0, 1):
        dense = frame[frame.gradient_task == direction].rename(columns={'group': 'method', 'G_positive_train': 'G_train'})
        table = reference.measured_tolerances(pc, dense)
        table['gradient_task'] = direction
        table['threshold_set'] = ['display' if e in EPSILONS else 'reference' for e in table.epsilon]
        tolerance_frames.append(table)
    tolerance = pc.pd.concat(tolerance_frames, ignore_index=True)
    tolerance_means = []
    for (direction, group, epsilon), values in tolerance.groupby(['gradient_task', 'method', 'epsilon']):
        assert sorted(values.seed.tolist()) == [0, 1, 2]
        censored = int(values.right_censored.sum())
        radii = values.largest_safe_measured_radius.to_numpy(float)
        tolerance_means.append(dict(gradient_task=int(direction), group=group, epsilon=epsilon,
            threshold_set='display' if epsilon in EPSILONS else 'reference',
            mean=None if censored else float(radii.mean()),
            sample_sd=None if censored else float(radii.std(ddof=1)),
            observed_lower_bound_mean=float(radii.mean()), censored_seeds=censored,
            plotted=censored == 0, n=3, interpolation_used=False))
    tolerance_mean = pc.pd.DataFrame(tolerance_means)
    assert (tolerance_mean.censored_seeds == 0).all()
    assert (tolerance_mean.loc[tolerance_mean.threshold_set == 'reference', 'mean'] == 0).all()
    frame.to_csv(HERE / 'selected_per_seed.csv', index=False)
    aggregate.to_csv(HERE / 'selected_mean_sample_sd.csv', index=False)
    tolerance.to_csv(HERE / 'tolerance_per_seed.csv', index=False)
    tolerance_mean.to_csv(HERE / 'tolerance_mean_sample_sd.csv', index=False)
    for path in [Path(__file__), HERE / 'plot_task0.py', HERE / 'plot_task1.py',
                 ROOT / 'report/radius_intervention_complete.json',
                 REFERENCE / 'plot_b1.py', REFERENCE / 'CAPTION.md',
                 BASE / 'figures_formal_20260921/plot_common.py']:
        sources[str(path)] = sha(path)
    font = pc.setup_style()
    pc.plt.rcParams.update({'axes.labelsize': 8.2, 'xtick.labelsize': 7.2,
                           'ytick.labelsize': 7.2, 'axes.labelpad': 1.5})
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    for direction in ((0, 1) if task is None else (task,)):
        out = HERE / JOBS[direction]
        out.mkdir(exist_ok=True)
        pc.HERE = out
        fig, axes = pc.plt.subplots(1, 4, figsize=(7.6, 1.85))
        fig.subplots_adjust(left=.075, right=.991, bottom=.30, top=.966, wspace=.56)
        labels = (r'Loss change $\overline{D_0(r)}$', 'Accuracy (%)', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$')
        for panel, (ax, metric, label) in enumerate(zip(axes, METRICS[:3], labels)):
            shown = aggregate[(aggregate.gradient_task == direction) & (aggregate.metric == metric)]
            for group in GROUPS:
                curve = shown[shown.group == group].sort_values('radius')
                xx, yy, sd = curve.radius.to_numpy(), curve['mean'].to_numpy(), curve.sample_sd.to_numpy()
                if panel < 2:
                    ax.fill_between(xx, yy-sd, yy+sd, color=COLORS[group], alpha=.13, linewidth=0)
                else:
                    for seed in (0, 1, 2):
                        each = frame[(frame.gradient_task == direction) & (frame.group == group)
                                     & (frame.seed == seed)].sort_values('radius')
                        ax.plot(each.radius, each.raw_fisher_trace, color=COLORS[group],
                                alpha=.25, linewidth=.55, marker='.', markersize=2, zorder=1)
                ax.plot(xx, yy, color=COLORS[group], linestyle=LINES[group], linewidth=1.25,
                        marker=MARKERS[group], markersize=3, label=LABELS[group], zorder=3)
            ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.06,5.06), xticks=[0,1,2,3,4,5])
            if panel < 2:
                lo = float((shown['mean']-shown.sample_sd).min())
                hi = float((shown['mean']+shown.sample_sd).max())
                if panel == 0:
                    lo = min(lo, 0.)
                    ax.axhline(0, color='#999999', linewidth=.45, zorder=0)
                pad = max((hi-lo)*.045, 1e-8)
                ax.set_ylim(lo-pad, hi+pad)
            else:
                # Selected raw Fisher values have no huge excluded-arm spike;
                # restore linear scale and include every selected individual seed.
                values = frame.loc[frame.gradient_task == direction, 'raw_fisher_trace']
                ax.set_ylim(0, float(values.max())*(1.40 if direction == 0 else 1.07))
            lo, hi = ax.get_ylim()
            ticks = MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]).tick_values(lo,hi)
            ax.set_yticks([t for t in ticks if lo <= t <= hi])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_: f'{value:g}'))
            legend_loc = 'lower right' if panel == 0 else ('upper left' if panel == 2 and direction == 1 else 'upper right')
            ax.legend(loc=legend_loc, frameon=False,
                      fontsize=7.2, handlelength=1.2, handletextpad=.3, labelspacing=.25, borderaxespad=.35)
        ax = axes[3]
        shown = tolerance_mean[(tolerance_mean.gradient_task == direction) & (tolerance_mean.threshold_set == 'display')]
        for group in GROUPS:
            curve = shown[shown.group == group].sort_values('epsilon')
            ax.plot(curve.epsilon, curve['mean'], color=COLORS[group], linestyle=LINES[group],
                    marker=MARKERS[group], markersize=3, linewidth=1.25, label=LABELS[group])
        # Reference-style mean-only tolerance panel. Per-seed values/SD are in CSV.
        # Common bounds make Task0 and Task1 panels directly comparable.
        upper = float(tolerance_mean.loc[tolerance_mean.threshold_set == 'display', 'mean'].max())*1.10
        ax.set(xlabel=r'Allowed damage $\epsilon$', ylabel=r'Tolerance radius $\hat r_\epsilon$',
               xlim=(.16,3.09), ylim=(-.10,upper), xticks=EPSILONS,
               xticklabels=[f'{e:g}' for e in EPSILONS], yticks=[0,.5,1,1.5,2,2.5])
        ax.tick_params(axis='x', labelrotation=90)
        ax.legend(loc='upper left', frameon=False, fontsize=7.2, handlelength=1.2,
                  handletextpad=.3, labelspacing=.25, borderaxespad=.35)
        name = f'Figure_{JOBS[direction]}_blue_green_tolerance'
        metadata = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            font=font, source_files=[dict(path=k,sha256=v) for k,v in sources.items()],
            selected_groups=list(GROUPS), displayed_labels=LABELS, displayed_parameter_values=False,
            fs_training_config=dict(gam_grad_rho=.05,gam_grad_norm_rho=1.),
            gradient_task=direction, direction_sign=1 if direction == 0 else -1,
            measured_radii=RADII, training_seeds=[0,1,2], directions_per_seed=1,
            allowed_damage_thresholds=EPSILONS, original_reference_thresholds=REFERENCE_EPSILONS,
            tolerance_function=str(REFERENCE / 'plot_b1.py')+'::measured_tolerances',
            tolerance_order='invert each seed first, then average three radii',
            right_censored_means='leave empty if any seed is censored',
            interpolation_used=False, new_model_evaluations=False,
            fisher_scale='linear, no numerical scaling/normalization/lambda',
            fisher_individual_seed_traces=True, axis_ranges=[list(ax.get_ylim()) for ax in axes])
        result = pc.save_figure(fig,name,metadata)
        result['statistics'] = ('All central curves: arithmetic mean of 3 training seeds. '
            'Panels 1-2: mean +/- sample SD; panel 3: mean plus individual-seed traces; '
            'panel 4: mean of per-seed grid tolerances, no uncertainty band (all values/SD saved).')
        (out / f'{name}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        objective = ('positive gradient of old Task0 training CE' if direction == 0 else
                     'negative gradient of new Task1 training CE with the paired untrained new20 head')
        caption = (f'Two-method fixed-direction diagnostic using the {objective}. '
            'Blue is EWCLoRA; green is the newly trained FS-EWCLoRA with GAM probe radius 0.05 and outer radius 1.0. '
            'Formal ImageNet-A, ViT-B/16 with rank-10 Q/V LoRA, training seeds 0/1/2. '
            'Each anchor has its own direction, computed once and held fixed for r=0,0.5,...,5 in effective Q/V weights. '
            'All panels evaluate the old Task0 model with its own fixed old20 head. '
            'Panels 1 and 2 show mean signed Task0 training CE change and accuracy (%) on all 1259 training images, '
            'with sample-SD shading across three seeds. Symmetric SD ribbons are descriptive and can extend below physical bounds. '
            'Panel 3 uses a linear vertical axis for raw empirical Fisher trace on the same 256 old training samples, '
            'with arithmetic mean curves and faint individual-seed curves; no Fisher SD band, normalization, floor or EWC lambda. '
            'Panel 4 uses the reference measured_tolerances function: for each seed, take the prefix maximum of '
            'positive loss change and return the last measured radius before its first value above epsilon. '
            'Average these three grid radii after inversion; no SD band is drawn in panel 4. '
            'Thresholds are 0.25,0.5,1,1.5,2,2.5,3 mean CE units per image. All displayed results are bracketed '
            'by adjacent measured radii separated by 0.5. The reference thresholds 0.005--0.05 all give zero here '
            'and are preserved in the CSV. No interpolation or new model evaluation is used. '
            'Connecting lines are visual guides; tolerance applies to the measured fixed ray, not all directions in a ball. '
            'Any future right-censored mean would be left empty rather than averaging only uncensored seeds.')
        (out / 'CAPTION.md').write_text(caption+'\n')
        (out / 'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+name+'.pdf}\n\\caption{'+caption.replace('%','\\%').replace('_','\\_')+'}\n\\end{figure*}\n')
        print(json.dumps(dict(direction=direction,outputs=result['outputs'],axis_ranges=result['axis_ranges'])))
    for item in published['output_files']:
        assert sha(item['path']) == item['sha256'], f'Prior output changed: {item["path"]}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direction', choices=['0','1','both'], default='both')
    args = parser.parse_args()
    main(None if args.direction == 'both' else int(args.direction))
