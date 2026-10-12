"""Style-only B1 revision, saved separately from both previous result versions."""
from pathlib import Path
import os
import sys
import json
import shutil

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
SOURCE = PARENT / 'B1_style_revision_v2_20260921'
NAME = 'Figure_B1_neighborhood_thresholds'
ORIGINAL_EPS = [.001,.005,.01,.02,.05]
ADDED_EPS = [.0025,.0075,.015,.03,.04]
ALL_EPS = sorted(ORIGINAL_EPS + ADDED_EPS)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))
sys.path.insert(0, str(PARENT))
import plot_common as pc

pd, np, plt = pc.pd, pc.np, pc.plt


def main():
    preserved = {str(p): pc.sha(p) for directory in (PARENT, SOURCE, PARENT / 'B1_with_tolerance_20260921')
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
    # Derive additional thresholds from saved per-radius measurements,
    # not from interpolating a plotted curve or evaluating a new model.
    curves = {}; selected = []; breakpoints = {0., *ALL_EPS}
    for seed in range(3):
        for method in pc.METHODS:
            summary,manifest,refs=pc.audited_summary(seed,method,'dense')
            sources.extend(refs)
            radii=np.array([row['radius'] for row in summary['radius_summary']])
            assert len(radii)==81 and np.all(np.diff(radii)==.25)
            for split in ('train','test'):
                damage=np.array([row[split]['G'] for row in summary['radius_summary']])
                prefix=np.maximum.accumulate(damage)
                curves[seed,method,split]=(radii,prefix)
                if split=='train':
                    old=dense.query('seed == @seed and method == @method').sort_values('radius')
                    np.testing.assert_allclose(damage,old.G_train,rtol=1e-12,atol=1e-12)
                    breakpoints.update(float(x) for x in prefix if 0<=x<=.05)
                for epsilon in ALL_EPS:
                    index=int(np.searchsorted(prefix,epsilon,side='right'))-1
                    radius=float(radii[index]);censored=index==80
                    if epsilon in ORIGINAL_EPS:
                        stored=next(row for row in summary['tolerance'] if row['split']==split and row['metric']=='G' and row['epsilon']==epsilon)
                        assert stored['prefix_radius']==radius and stored['right_censored']==censored
                    selected.append(dict(seed=seed,method=method,split=split,epsilon=epsilon,
                        tolerance_radius=radius,right_censored=censored,
                        threshold_kind='original' if epsilon in ORIGINAL_EPS else 'added_postprocessing'))
    selected=pd.DataFrame(selected)
    selected.to_csv(OUT/'selected_thresholds_per_seed.csv',index=False)
    selected_agg=[]
    for split in ('train','test'):
        frame=selected.query('split == @split')
        stats=pc.mean_sd(frame,'epsilon','tolerance_radius')
        c=frame.groupby(['method','epsilon']).right_censored.sum().rename('censored_seeds').reset_index()
        stats=stats.merge(c,on=['method','epsilon'],validate='one_to_one')
        stats['split']=split
        stats['threshold_kind']=stats.epsilon.map(lambda e:'original' if e in ORIGINAL_EPS else 'added_postprocessing')
        selected_agg.append(stats)
    selected_agg=pd.concat(selected_agg,ignore_index=True)
    selected_agg.to_csv(OUT/'selected_thresholds_mean_sd.csv',index=False)
    inverse=[]
    for (seed,method,split),(radii,prefix) in curves.items():
        if split!='train':continue
        for epsilon in sorted(breakpoints):
            index=int(np.searchsorted(prefix,epsilon,side='right'))-1
            inverse.append(dict(seed=seed,method=method,epsilon=epsilon,
                tolerance_radius=float(radii[index]),right_censored=index==80))
    inverse=pd.DataFrame(inverse)
    tolerance=pc.mean_sd(inverse,'epsilon','tolerance_radius')
    counts=inverse.groupby(['method','epsilon']).right_censored.sum().rename('censored_seeds').reset_index()
    tolerance=tolerance.merge(counts,on=['method','epsilon'],validate='one_to_one')
    inverse.to_csv(OUT/'tolerance_step_per_seed.csv',index=False)
    tolerance.to_csv(OUT/'tolerance_step_mean_sd.csv',index=False)

    font = pc.setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.6,1.85))
    fig.subplots_adjust(left=.075, right=.991, bottom=.32, top=.966, wspace=.56)
    settings = [
        (dense, 'G_train', r'Loss change $\overline{D_t(\boldsymbol{\theta},\boldsymbol{\Delta})}$', 'upper left'),
        (dense, 'train_accuracy_percent', 'Accuracy (%)', 'lower left'),
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
    displayed, omitted, tail_cutoffs = [], [], {}
    for method in pc.METHODS:
        complete_rows = tolerance.query('method == @method').sort_values('epsilon').reset_index(drop=True)
        hit_indices = np.flatnonzero(complete_rows.censored_seeds.to_numpy() == 3)
        assert len(hit_indices) > 0
        end = int(hit_indices[0])
        rows = complete_rows.iloc[:end+1].copy()
        hidden = complete_rows.iloc[end+1:].copy()
        assert rows.iloc[-1]['mean'] == 20 and (hidden['mean'] == 20).all()
        assert (hidden.censored_seeds == 3).all()
        displayed.append(rows); omitted.append(hidden)
        tail_cutoffs[method] = dict(first_all_seeds_capped_epsilon=float(rows.iloc[-1].epsilon),
            displayed_rows=len(rows), omitted_repeated_cap_rows=len(hidden))
        x,y,sd = (rows[c].to_numpy() for c in ['epsilon','mean','sd'])
        mark_indices = np.flatnonzero(np.isin(x,ALL_EPS)).tolist()
        assert len(mark_indices) >= 7
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
           xticks=ORIGINAL_EPS,xticklabels=['0.001','0.005','0.01','0.02','0.05'])
    # Keep a linear epsilon axis. Close small thresholds require vertical labels.
    ax.tick_params(axis='x',labelrotation=90,labelsize=7)
    fig.align_xlabels(axes)
    pc.internal_legend(ax,'lower right')
    pd.concat(displayed, ignore_index=True).to_csv(OUT/'tolerance_displayed_mean_sd.csv',index=False)
    pd.concat(omitted, ignore_index=True).to_csv(OUT/'tolerance_omitted_cap_tail.csv',index=False)

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
        old_versions_preserved=preserved,display_only_revision=True,data_values_unchanged=True,
        original_measurements_unchanged=True,method_encodings=encodings,
        original_threshold_ticks=ORIGINAL_EPS,added_thresholds=ADDED_EPS,
        threshold_axis_scale="linear",exact_threshold_breakpoints=len(breakpoints),
        epsilon_definition="Mean positive change in sample-averaged old-task CE, then averaged over 100 directions; not summed loss or per-example positive damage",
        colors_lines_markers_and_labels_identical_across_panels=True,
        removed_visual_annotations=['scan limit text','gray censor arrow','horizontal scan-limit guide'],
        first_panel_display_notation='Loss change overline(D_t(theta, Delta)); aggregation remains G=mean(max(D_t,0)) as defined in caption',
        tail_cutoffs=tail_cutoffs,
        omitted_rows_rule='Only subsequent rows after first all-three-seeds reach r=20 are omitted from line/band; full source CSVs retained.',
        fisher_definition='mean true-label per-example squared-gradient norm, unnormalized, no EWC lambda',
        censoring='Each line stops at its first fully capped mean of 20; remaining r=20 plateau omitted, not extrapolated; cap explained in caption',
        script=dict(path=str(Path(__file__).resolve()),sha256=pc.sha(__file__)))
    result=pc.save_figure(fig,f'{OUT.name}/{NAME}',metadata)
    assert all(pc.sha(Path(p))==digest for p,digest in preserved.items())
    result['all_prior_version_file_hashes_unchanged']=True
    (OUT/f'{NAME}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    caption = "Training-neighborhood diagnostics on the formal three-seed ImageNet-A cohort (ViT-B/16 with rank-10 Q/V LoRA). Panels show average positive training loss change, training accuracy (%), empirical Fisher trace, and training tolerance radius. The first axis uses simplified average-damage notation; its statistic is $G(r)=K^{-1}\\sum_{k=1}^K\\max\\{N^{-1}\\sum_{i=1}^N[\\ell_i(\\theta+ru_k)-\\ell_i(\\theta)],0\\}$, with N=1,259 training examples and K=100 common directions. CE is averaged over samples before taking each direction's positive part; this is neither summed training loss nor the mean of individual positive sample changes. Loss/accuracy use 81 radii from 0 to 20 in steps of 0.25. Fisher uses the same Task-0 anchors and effective Q/V coordinates, a fixed 256-example subset, the first 16 paired directions, and seven measured radii (0, 0.5, 1, 2, 5, 10, 20); it is the unnormalized true-label empirical trace, without an EWC coefficient. Fisher connecting segments are guides. The origin is measured once. The fourth panel uses the exact prefix inverse $\\hat r_\\epsilon=\\max\\{r_j:\\max_{i\\le j}G(r_i)\\le\\epsilon\\}$, computed per seed before averaging. Its linear axis labels the originally stored thresholds 0.001, 0.005, 0.01, 0.02, 0.05. Additional thresholds 0.0025, 0.0075, 0.015, 0.03, 0.04 are derived from the same measured loss grid by postprocessing, not by new model evaluations. Markers show selected thresholds that lie within each displayed line; the 0.01 guide is retained. All threshold values, including omitted capped markers, are exported. Each method's line stops at the first threshold where all three seeds reach the scan cap r=20; the later fully capped horizontal tail is omitted. Some seeds are censored earlier. Endpoints at 20 are scan limits, not identified true boundaries. Curves and bands are mean and one sample SD over training seeds 0/1/2. EWCLoRA uses blue solid circles and FS-EWCLoRA orange dashed diamonds. These finite-direction training diagnostics do not guarantee every sample/direction or improved test tolerance."+'\n'
    (OUT/'CAPTION.md').write_text(caption)
    (OUT/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+NAME+'.pdf}\n\\caption{'+caption.strip()+'}\n\\label{fig:rq2-training-neighborhood-thresholds}\n\\end{figure*}\n')
    print(json.dumps(dict(outputs=result['outputs'],method_encodings_consistent=True,data_unchanged=True,
                          prior_versions_preserved=True),indent=2))


if __name__=='__main__':
    main()
