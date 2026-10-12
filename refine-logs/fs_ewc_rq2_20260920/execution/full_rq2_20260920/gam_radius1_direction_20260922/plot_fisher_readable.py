"""Save an additional positive-log-Fisher view; preserve original linear figures."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('radius_publisher_readonly_style', ROOT / 'publish.py')
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)
    receipt_path = ROOT / 'report/radius_intervention_complete.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['validated_jobs'] == 12
    saved = {v['path']: v['sha256'] for v in receipt['output_files']}
    inputs = [ROOT / 'figures/all_per_seed.csv', ROOT / 'figures/all_mean_sample_sd.csv']
    for path in inputs:
        assert sha(path) == saved[str(path)], f'Published data changed: {path}'
    rows = list(csv.DictReader(inputs[0].open()))
    aggregates = list(csv.DictReader(inputs[1].open()))
    assert len(rows) == 264
    for row in rows:
        assert math.isfinite(float(row['raw_fisher_trace'])) and float(row['raw_fisher_trace']) > 0
    pc, font = pub.plotting()
    from matplotlib.ticker import FuncFormatter, MaxNLocator, LogLocator, LogFormatterMathtext
    sources = [dict(path=str(path), sha256=sha(path)) for path in
               [Path(__file__), ROOT / 'publish.py', receipt_path] + inputs]
    outputs = []
    for task in (0, 1):
        out = ROOT / 'figures_readable' / pub.JOBS[task]
        out.mkdir(parents=True, exist_ok=True)
        pc.HERE = out
        fig, axes = pc.plt.subplots(1, 3, figsize=(9.1, 2.35))
        fig.subplots_adjust(left=.067, right=.992, bottom=.255, top=.965, wspace=.43)
        labels = (r'Loss change $\overline{D_0(r)}$', 'Accuracy (%)', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$')
        for i, (ax, metric, label) in enumerate(zip(axes, pub.PLOT_METRICS, labels)):
            shown = [x for x in aggregates if int(x['gradient_task']) == task and x['metric'] == metric]
            assert len(shown) == 44
            for group in pub.GROUPS:
                series = sorted((x for x in shown if x['group'] == group), key=lambda x: float(x['radius']))
                xx = [float(x['radius']) for x in series]
                yy = [float(x['mean']) for x in series]
                sd = [float(x['sample_sd']) for x in series]
                assert xx == pub.RADII
                if i < 2:
                    ax.fill_between(xx, [a-b for a,b in zip(yy,sd)], [a+b for a,b in zip(yy,sd)],
                                    color=pub.COLORS[group], alpha=.12, linewidth=0)
                else:
                    for seed in range(3):
                        each = sorted((x for x in rows if int(x['gradient_task']) == task
                                       and x['group'] == group and int(x['seed']) == seed),
                                      key=lambda x: float(x['radius']))
                        assert [float(x['radius']) for x in each] == pub.RADII
                        ax.plot(xx, [float(x['raw_fisher_trace']) for x in each],
                                color=pub.COLORS[group], alpha=.25, linewidth=.55,
                                marker='.', markersize=2, zorder=1)
                ax.plot(xx, yy, color=pub.COLORS[group], linestyle=pub.LINES[group],
                        marker=pub.MARKERS[group], markersize=3, linewidth=1.2,
                        label=pub.LABELS[group], zorder=3)
            ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.06,5.06), xticks=list(range(6)))
            if i < 2:
                lo = min(float(x['mean'])-float(x['sample_sd']) for x in shown)
                hi = max(float(x['mean'])+float(x['sample_sd']) for x in shown)
                if metric == 'D_train':
                    lo = min(lo, 0.)
                    ax.axhline(0, color='#999999', linewidth=.45, zorder=0)
                pad = max((hi-lo)*.05, 1e-8)
                ax.set_ylim(lo-pad, hi+pad)
                ticks = MaxNLocator(nbins=4, steps=[1,2,2.5,5,10]).tick_values(lo-pad,hi+pad)
                ax.set_yticks([v for v in ticks if lo-pad <= v <= hi+pad])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:.2g}' if abs(v)>=1e4 or 0<abs(v)<1e-3 else f'{v:g}'))
            else:
                values = [float(x['raw_fisher_trace']) for x in rows if int(x['gradient_task']) == task]
                ax.set_yscale('log')
                ax.set_ylim(min(values)*.4, max(values)*2.5)
                lo, hi = ax.get_ylim()
                ticks = LogLocator(base=10, numticks=6).tick_values(lo, hi)
                ax.set_yticks([v for v in ticks if lo <= v <= hi])
                ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.minorticks_off()
            ax.legend(loc='best', frameon=False, fontsize=7.3, handlelength=1.35,
                      handletextpad=.3, labelspacing=.22, borderaxespad=.35, borderpad=0)
        name = 'Figure_' + pub.JOBS[task] + '_training_radius_intervention_log_fisher'
        outputs.append(pc.save_figure(fig, name, dict(
            font=font, source_files=sources, gradient_task=task, measured_radii=pub.RADII,
            groups=list(pub.GROUPS), arithmetic_mean=True, log_transform_values=False,
            first_two_panels='Same mean and sample SD as the original linear figures',
            fisher_panel='Positive logarithmic y axis; thick arithmetic means and thin individual-seed traces; no SD band',
            no_floor_or_clipping=True, extra_fisher_normalization_or_lambda=False,
            axis_ranges=[list(ax.get_ylim()) for ax in axes])))
        caption = pub.caption(task).replace(
            'Lines and shading show arithmetic mean and one sample standard deviation (ddof=1) across three training seeds, not SEM or confidence intervals.',
            'The first two panels show arithmetic mean and one sample standard deviation (ddof=1) across three training seeds, not SEM or confidence intervals. '
            'In the Fisher panel, the vertical axis is logarithmic, the thick curves are arithmetic means of the original positive values, '
            'and thin curves show the three individual training seeds. No standard-deviation band is drawn in the Fisher panel; '
            'all individual values and peaks are retained, without clipping, flooring, or extra normalization.')
        (out / 'CAPTION.md').write_text(caption + '\n')
    for path in inputs:
        assert sha(path) == saved[str(path)], 'Read-only source unexpectedly changed'
    report = dict(status='complete', figures=outputs, source_files=sources, new_gpu_measurements=False,
                  original_linear_figures_preserved=True, rendered_visual_review='pending actual PNG inspection')
    (ROOT / 'report/readable_figures_complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete', figures=len(outputs), new_gpu_measurements=False)))


if __name__ == '__main__':
    main()
