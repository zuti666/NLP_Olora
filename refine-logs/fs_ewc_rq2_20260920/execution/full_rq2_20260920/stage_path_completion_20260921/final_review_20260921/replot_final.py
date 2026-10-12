"""Re-read both completed diagnostics and render reviewed copies, preserving queue receipts."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
FIGURES=HERE/'figures'
REPORT=HERE/'replot_validation'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clear_path_legend(fig,style):
    """Keep the shared method legends in the first three panels, clear of all data."""
    fig.axes[0].set_ylabel(r'Weight drift $\delta$',labelpad=3)
    ax=fig.axes[3]
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return dict(first_panel_ylabel=r'Weight drift $\delta$',
                shared_method_legend_panels=[1,2,3],
                last_panel_redundant_legend_removed=True,
                last_panel_legend_cannot_obscure_mean_or_sd=True)


def main():
    FIGURES.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    originals=[EXT/'report/complete.json',EXT/'report/radius_complete.json',
               EXT/'campaign_idle_gpu146/pipeline_state.json']
    before={str(p):sha(p) for p in originals}
    assert json.loads(originals[-1].read_text())['status']=='complete'
    peak=json.loads((HERE/'FISHER_PEAK_VERIFICATION.json').read_text())
    assert peak['status']=='passed'
    b1=load('reviewed_radius_publisher',EXT/'publish_radius.py')
    b1.FIGURES=FIGURES;b1.REPORT=REPORT
    common,dense,sparse,new,evidence=b1.validate_all()
    evidence.update(final_replot_script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
                    targeted_peak_verification=dict(path=str(HERE/'FISHER_PEAK_VERIFICATION.json'),sha256=sha(HERE/'FISHER_PEAK_VERIFICATION.json')))
    b1.publish(common,dense,sparse,new,evidence)
    note='''\nThe EWCLoRA peak at r=17 is retained. It originates from seed0, public direction14 (trace 68,312.656); same-protocol repeats at r=16/17/18 reproduced the recorded values exactly. One training example contributes 96.46% of that direction's trace sum. A separate single-example autograd calculation agrees with the single-example collector; this verification does not add another training seed or replace the original observation.\n'''
    caption=FIGURES/'B1_radius_extended_caption.md'
    caption.write_text(caption.read_text()+note)
    receipt_path=REPORT/'radius_complete.json'
    receipt=json.loads(receipt_path.read_text())
    for record in receipt['output_files']:
        if Path(record['path'])==caption:
            record['sha256']=sha(caption)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    nodes=replot_b2()
    for p in originals:assert sha(p)==before[str(p)], 'Original queue receipt changed'
    result=dict(status='replotted',recomputed_paths=54,path_nodes=len(nodes),B1_extension_jobs=6,
        original_queue_receipts_unchanged=True,source_queue_sha256=before,
        outputs={p.name:sha(p) for p in FIGURES.iterdir() if p.suffix in ('.png','.pdf')},
        rendered_review='pending final root and secondary inspection')
    (HERE/'REPLOT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def replot_b2():
    """Re-render only B2 after a label/layout edit, preserving the B1 artifacts."""
    b2=load('reviewed_path_publisher',EXT/'publish.py')
    b2.OUT=FIGURES;b2.REPORT=REPORT;b2.style.HERE=FIGURES
    original_save=b2.style.save_figure
    def save_final(fig,name,metadata):
        metadata['path_legend_placement']=clear_path_legend(fig,b2.style)
        metadata['final_replot_script']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
        return original_save(fig,name,metadata)
    b2.style.save_figure=save_final
    trajectory,paths,nodes,evidence=b2.collect()
    assert evidence['validated_paths']==54 and len(paths)==54
    b2.write_json(REPORT/'coverage.json',evidence)
    b2.publish(trajectory,paths,nodes,evidence)
    return nodes


if __name__=='__main__':main()
