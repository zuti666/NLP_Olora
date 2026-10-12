"""Style-only B1 revision, saved separately from both previous result versions."""
from pathlib import Path
import os
import sys
import json
import shutil

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
SOURCE = PARENT / 'B1_with_tolerance_20260921'
NAME = 'Figure_B1_neighborhood_styled'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))
sys.path.insert(0, str(PARENT))
import plot_common as pc

pd, np, plt = pc.pd, pc.np, pc.plt


def main():
    preserved = {str(p): pc.sha(p) for directory in (PARENT, SOURCE)
                 for p in directory.iterdir() if p.is_file()}
    receipt = pc.read(SOURCE / 'ARTIFACT_MANIFEST.json')
    recorded = {r['path']: r['sha256'] for r in receipt['files']}
    filenames = ['dense_training_per_seed.csv', 'fisher_training_per_seed.csv',
                 'first_three_mean_sd.csv', 'tolerance_step_per_seed.csv',
                 'tolerance_step_mean_sd.csv', 'saved_G_tolerance_train_and_test.csv']
    sources = []
    for name in filenames:
        path = SOURCE / name
        assert pc.sha(path) == recorded[str(path)]
        sources.append(dict(path=str(path), sha256=pc.sha(path)))
        shutil.copyfile(path, OUT / name)
    dense = pd.read_csv(OUT / filenames[0])
    sparse = pd.read_csv(OUT / filenames[1])
    inverse = pd.read_csv(OUT / 'tolerance_step_per_seed.csv')
    tolerance = pd.read_csv(OUT / 'tolerance_step_mean_sd.csv')
    recomputed = pc.mean_sd(inverse, 'epsilon', 'tolerance_radius')
    key = ['method','epsilon']
    recomputed = recomputed.sort_values(key).reset_index(drop=True)
    target = tolerance.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(recomputed[key], target[key])
    np.testing.assert_allclose(recomputed[['mean','sd']], target[['mean','sd']], rtol=1e-12, atol=1e-12)

    font = pc.setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.6,1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.26, top=.966, wspace=.56)
    settings = [
        (dense, 'G_train', r'Loss $D_t(\boldsymbol{\theta},\boldsymbol{\Delta})$', 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy', 'lower left'),
        (sparse, 'raw_fisher_trace', r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$', 'upper left'),
    ]
    aggregates = []
    for ax, (frame, value, label, location) in zip(axes[:3], settings):
        stats = pc.draw_mean_sd(ax, frame, 'radius', value,
                               marker_every=4 if value != 'raw_fisher_trace' else 1)
        pc.limits_from_sd(ax, stats, include_zero=value=='G_train')
        ax.set(xlabel=r'Weight drift $r$', ylabel=label, xlim=(-.5,20.5), xticks=[0,10,20])
        pc.internal_legend(ax, location)
        stats['metric'] = value; aggregates.append(stats)
    old = pd.read_csv(SOURCE/'first_three_mean_sd.csv')
    actual = pd.concat(aggregates, ignore_index=True)
    keys = ['method','radius','metric']
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  old.sort_values(keys).reset_index(drop=True), check_exact=False, atol=1e-12, rtol=1e-12)

    ax = axes[3]
    ax.axvline(.01, color='.65', linestyle=':', linewidth=.8, zorder=0)
    for method in pc.METHODS:
        rows = tolerance.query('method == @method').sort_values('epsilon')
        x,y,sd = (rows[c].to_numpy() for c in ['epsilon','mean','sd'])
        mark_indices = np.flatnonzero(np.isin(x,[.001,.005,.01,.02])).tolist()
        assert len(mark_indices) == 4
        ax.fill_between(x,y-sd,y+sd,step='post',color=pc.COLORS[method],alpha=.16,linewidth=0)
        # The same Line2D carries both line and markers, making its legend
        # identical in style to the first three panels.
        ax.step(x,y,where='post',color=pc.COLORS[method],linestyle=pc.LINES[method],
                linewidth=1.25,marker=pc.MARKERS[method],markersize=3.1,markevery=mark_indices,
                label=pc.LABELS[method],zorder=3)
        focus=rows[rows.epsilon==.01].iloc[0]
        ax.errorbar([.01],[focus['mean']],yerr=[focus['sd']],linestyle='none',
                    color=pc.COLORS[method],capsize=2,linewidth=.8,capthick=.8,zorder=5)
    ax.set(xlabel=r'Allowed damage $\epsilon$',ylabel=r'Tolerance radius $\hat r_\epsilon$',
           xlim=(-.001,.0515),ylim=(-.55,20.85),yticks=[0,5,10,15,20],
           xticks=[0,.01,.03,.05],xticklabels=['0','0.01','0.03','0.05'])
    pc.internal_legend(ax,'lower right')

    encodings=[]
    for panel,ax in enumerate(axes,1):
        encoding={}
        for method in pc.METHODS:
            lines=[line for line in ax.lines if line.get_label()==pc.LABELS[method]]
            assert len(lines)==1
            line=lines[0]
            assert line.get_color()==pc.COLORS[method] and line.get_marker()==pc.MARKERS[method]
            encoding[method]=dict(label=line.get_label(),color=line.get_color(),marker=line.get_marker(),
                linestyle=line.get_linestyle(),linewidth=line.get_linewidth(),markersize=line.get_markersize(),
                dash_pattern=line._unscaled_dash_pattern)
        encodings.append(encoding)
    assert all(encoding==encodings[0] for encoding in encodings)
    assert all('scan limit' not in t.get_text() for ax in axes for t in ax.texts)
    metadata=dict(font=font,source_files=sources,source_cohort='formal ImageNet-A training seeds 0/1/2',
        old_versions_preserved=preserved,styling_only=True,data_values_unchanged=True,
        csvs_identical_to_previous_version=True,method_encodings=encodings,
        colors_lines_markers_and_labels_identical_across_panels=True,
        removed_visual_annotations=['scan limit text','gray censor arrow','horizontal scan-limit guide'],
        first_panel_display_notation='Loss D_t(theta, Delta); aggregation remains G=mean(max(D_t,0)) as defined in caption',
        fisher_definition='mean true-label per-example squared-gradient norm, unnormalized, no EWC lambda',
        censoring='Tolerance capped at measured r=20; described in caption, no on-plot annotation',
        script=dict(path=str(Path(__file__).resolve()),sha256=pc.sha(__file__)))
    result=pc.save_figure(fig,f'{OUT.name}/{NAME}',metadata)
    assert all(pc.sha(Path(p))==digest for p,digest in preserved.items())
    result['all_prior_version_file_hashes_unchanged']=True
    (OUT/f'{NAME}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    caption=(SOURCE/'CAPTION.md').read_text()
    caption=caption.replace('mean positive training CE damage, training accuracy (%)',
        r'mean positive training CE damage (the first axis abbreviates the underlying damage quantity as $D_t(\boldsymbol{\theta},\boldsymbol{\Delta})$; the plotted aggregation remains $G=\mathbb{E}_u[D_t]_+$ with the positive part taken before averaging), training accuracy (%)')
    # Use an unambiguous expression for expectation of the positive part.
    caption=caption.replace(r'$G=\mathbb{E}_u[D_t]_+$',r'$G=\mathbb{E}_u[\max(D_t,0)]$')
    caption=caption.replace('the shared upward arrow denotes right-censoring rather than equality of their unknown boundaries beyond the scan.',
        'the plateaus at 20 are right-censored and do not establish equality of the unknown boundaries beyond the scan.')
    caption += '\nAcross all four panels, EWCLoRA uses a blue solid line with circles, and FS-EWCLoRA an orange dashed line with diamonds; legends use the same encodings. The concise Fisher label still denotes the unnormalized empirical trace.\n'
    (OUT/'CAPTION.md').write_text(caption)
    (OUT/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+NAME+'.pdf}\n\\caption{'+caption.strip()+'}\n\\label{fig:rq2-training-neighborhood-styled}\n\\end{figure*}\n')
    print(json.dumps(dict(outputs=result['outputs'],method_encodings_consistent=True,data_unchanged=True,
                          prior_versions_preserved=True),indent=2))


if __name__=='__main__':
    main()
