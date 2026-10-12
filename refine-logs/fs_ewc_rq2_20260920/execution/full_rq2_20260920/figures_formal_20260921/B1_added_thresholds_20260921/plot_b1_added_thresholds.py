"""New figure version with the five user-selected additional tolerance levels."""
from pathlib import Path
import os
import sys
import json
import shutil

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
SOURCE=PARENT/'B1_threshold_revision_20260921'
NAME='Figure_B1_added_thresholds'
LEVELS=[.0025,.0075,.015,.03,.04]
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'.mplcache'))
sys.path.insert(0,str(PARENT))
import plot_common as pc
pd,np,plt=pc.pd,pc.np,pc.plt


def main():
    before={str(p):pc.sha(p) for directory in (PARENT,SOURCE) for p in directory.iterdir() if p.is_file()}
    manifest=pc.read(SOURCE/'ARTIFACT_MANIFEST.json')
    hashes={r['path']:r['sha256'] for r in manifest['files']}
    names=['dense_training_per_seed.csv','fisher_training_per_seed.csv','first_three_mean_sd.csv',
           'selected_thresholds_per_seed.csv','selected_thresholds_mean_sd.csv']
    inputs=[]
    for name in names:
        p=SOURCE/name
        assert pc.sha(p)==hashes[str(p)]
        shutil.copyfile(p,OUT/name)
        inputs.append(dict(path=str(p),sha256=pc.sha(p)))
    dense=pd.read_csv(OUT/names[0]);fisher=pd.read_csv(OUT/names[1])
    all_seed=pd.read_csv(OUT/'selected_thresholds_per_seed.csv')
    selected=all_seed[(all_seed['split']=='train') & all_seed.epsilon.isin(LEVELS)].copy()
    assert len(selected)==30
    stats=pc.mean_sd(selected,'epsilon','tolerance_radius')
    caps=selected.groupby(['method','epsilon']).right_censored.sum().rename('censored_seeds').reset_index()
    stats=stats.merge(caps,on=['method','epsilon'],validate='one_to_one').sort_values(['method','epsilon'])
    expected=pd.read_csv(OUT/'selected_thresholds_mean_sd.csv')
    expected=expected[(expected['split']=='train') & expected.epsilon.isin(LEVELS)].sort_values(['method','epsilon'])
    np.testing.assert_allclose(stats[['mean','sd','n','censored_seeds']],
                               expected[['mean','sd','n','censored_seeds']],rtol=1e-12,atol=1e-12)
    selected.to_csv(OUT/'fourth_panel_per_seed.csv',index=False)
    stats.to_csv(OUT/'fourth_panel_mean_sd.csv',index=False)
    expected_rounded={'sgd':[5.750,9.333,12.500,16.667,18.667],
                      'gam':[6.583,10.833,14.417,18.917,20.]}
    for method in pc.METHODS:
        rows=stats.query('method == @method').sort_values('epsilon')
        assert rows.epsilon.tolist()==LEVELS
        np.testing.assert_allclose(rows['mean'],expected_rounded[method],rtol=0,atol=.000501)

    font=pc.setup_style()
    fig,axes=plt.subplots(1,4,figsize=(7.6,1.85))
    fig.subplots_adjust(left=.075,right=.991,bottom=.32,top=.966,wspace=.56)
    specs=[(dense,'G_train',r'Loss change $\overline{D_t(\boldsymbol{\theta},\boldsymbol{\Delta})}$','upper left'),
           (dense,'train_accuracy_percent','Accuracy (%)','lower left'),
           (fisher,'raw_fisher_trace',r'Fisher $\operatorname{tr}F_0^{\mathrm{emp}}$','upper left')]
    summaries=[]
    for ax,(frame,value,label,location) in zip(axes[:3],specs):
        agg=pc.draw_mean_sd(ax,frame,'radius',value,marker_every=4 if value!='raw_fisher_trace' else 1)
        pc.limits_from_sd(ax,agg,include_zero=value=='G_train')
        ax.set(xlabel=r'Weight drift $r$',ylabel=label,xlim=(-.5,20.5),xticks=[0,10,20])
        pc.internal_legend(ax,location)
        agg['metric']=value;summaries.append(agg)
    key=['method','radius','metric']
    actual=pd.concat(summaries,ignore_index=True).sort_values(key).reset_index(drop=True)
    old=pd.read_csv(SOURCE/'first_three_mean_sd.csv').sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,old,check_exact=False,rtol=1e-12,atol=1e-12)

    ax=axes[3]
    for method in pc.METHODS:
        rows=stats.query('method == @method').sort_values('epsilon')
        x,y=rows.epsilon.to_numpy(),rows['mean'].to_numpy()
        ax.plot(x,y,color=pc.COLORS[method],linestyle=pc.LINES[method],linewidth=1.25,
                marker=pc.MARKERS[method],markersize=3.1,label=pc.LABELS[method],zorder=3)
        uncensored=rows[rows.censored_seeds==0]
        ax.errorbar(uncensored.epsilon,uncensored['mean'],yerr=uncensored.sd,
                    color=pc.COLORS[method],linestyle='none',capsize=2,linewidth=.8,capthick=.8,zorder=2)
    capped=stats[(stats.method=='gam') & (stats.epsilon==.04)].iloc[0]
    assert capped['mean']==20 and capped.censored_seeds==3
    # A simple bound label preserves censor information without a scan-limit line/arrow.
    ax.annotate(r'$\geq 20$',xy=(.04,20),xytext=(-3,4),textcoords='offset points',
                ha='right',va='bottom',fontsize=7,color=pc.COLORS['gam'])
    ax.set(xlabel=r'Allowed damage $\epsilon$',ylabel=r'Tolerance radius $\hat r_\epsilon$',
           xlim=(0,.0415),ylim=(-.55,22.1),yticks=[0,5,10,15,20],
           xticks=LEVELS,xticklabels=['0.0025','0.0075','0.015','0.03','0.04'])
    ax.tick_params(axis='x',labelrotation=90,labelsize=7)
    pc.internal_legend(ax,'lower right');fig.align_xlabels(axes)
    # Check the actual method encodings in every panel, not just rc defaults.
    encodings=[]
    for ax in axes:
        mapping={}
        for method in pc.METHODS:
            line=next(l for l in ax.lines if l.get_label()==pc.LABELS[method])
            mapping[method]=dict(color=line.get_color(),marker=line.get_marker(),
                dash=line._unscaled_dash_pattern,linewidth=line.get_linewidth(),markersize=line.get_markersize())
        encodings.append(mapping)
    assert all(x==encodings[0] for x in encodings)
    meta=dict(font=font,source_files=inputs,source_cohort='formal ImageNet-A training seeds 0/1/2',
        displayed_thresholds=LEVELS,displayed_fourth_panel_values=stats.to_dict('records'),
        first_three_aggregates_unchanged=True,no_new_gpu_evaluations=True,
        fourth_panel='Exactly five selected train threshold means; sample SD error bars at uncensored points; straight segments are guides.',
        cap_annotation='FS-EWCLoRA epsilon=.04 is displayed at scan cap20 and labeled >=20; all3 seeds censored, no zero-SD certainty claim.',
        previous_version_file_sha256=before,all_panel_method_encodings=encodings,
        script=dict(path=str(Path(__file__).resolve()),sha256=pc.sha(__file__)))
    result=pc.save_figure(fig,f'{OUT.name}/{NAME}',meta)
    assert all(pc.sha(Path(p))==digest for p,digest in before.items())
    result['prior_versions_preserved']=True
    (OUT/f'{NAME}_validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    caption=r'''Training-neighborhood diagnostics on the formal three-seed ImageNet-A cohort (ViT-B/16, rank-10 Q/V LoRA). The first three panels retain the previously reported mean positive CE damage, training accuracy (%), and empirical Fisher trace. Loss and accuracy use 1,259 Task-0 training examples, 100 paired directions and 81 measured radii (0–20, step 0.25). Fisher uses the same anchors and effective Q/V coordinates, a fixed 256-example training subset, 16 paired directions, and seven measured radii (0, 0.5, 1, 2, 5, 10, 20), with no EWC scaling or normalization. Fisher segments are visual guides. The first-axis average-damage shorthand refers to $G(r)=K^{-1}\sum_k\max\{N^{-1}\sum_i[\ell_i(\theta+r u_k)-\ell_i(\theta)],0\}$: sample-average CE differences are clipped per direction before averaging directions. The fourth panel shows only the five selected training tolerance levels $\epsilon\in\{0.0025,0.0075,0.015,0.03,0.04\}$ on a linear axis. Each radius is obtained from the measured-grid prefix rule $\hat r_\epsilon=\max\{r_j:\max_{i\le j}G(r_i)\le\epsilon\}$ separately for each training seed, before mean/sample-SD aggregation. Fourth-panel connecting segments are visual guides between selected threshold summaries, not evaluations of intermediate thresholds or interpolated radius estimates. At $\epsilon=0.04$, all three FS-EWCLoRA seeds reach the measurement limit 20; its displayed marker is annotated $\geq20$ and does not identify the true boundary. The capped zero SD is not shown as evidence of zero uncertainty. Shading in the first three panels and error bars at uncensored fourth-panel points are one sample SD over the three training seeds. All ten threshold levels and both train/test summaries remain available in the accompanying CSVs. This revision adds no model evaluations. Method colors, lines and markers are consistent across panels.'''
    (OUT/'CAPTION.md').write_text(caption+'\n')
    (OUT/'include.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{'+NAME+'.pdf}\n\\caption{'+caption+'}\n\\label{fig:rq2-five-added-thresholds}\n\\end{figure*}\n')
    print(json.dumps(dict(outputs=result['outputs'],fourth_panel=stats.to_dict('records'),prior_versions_preserved=True),indent=2))


if __name__=='__main__':main()
