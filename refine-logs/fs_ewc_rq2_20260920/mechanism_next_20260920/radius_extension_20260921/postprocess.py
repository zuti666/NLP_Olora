"""Publish the extended three-panel B only after complete measurement validation."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / 'figures_redesign_20260920'
ALIASES = HERE.parent / 'figures'
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    subprocess.run([sys.executable, str(FIG / 'gen_fig_B_neighborhood_merged.py'),
                    '--radius-extension', str(HERE)], check=True)
    checks = json.loads((FIG / 'B_provenance.json').read_text())
    assert checks['status'] == 'passed' and checks['radius_extension_applied']
    assert checks['source_states_used'] == 482 and len(checks['plotted_data']) == 6
    assert all(len(item['x']) == 16 for item in checks['plotted_data'])
    fontnames = sorted(set(x.decode() for x in re.findall(
        rb'/(?:BaseFont|FontName)\s*/([^\s/>]+)', (FIG / 'Figure_B_neighborhood_merged.pdf').read_bytes())))
    assert fontnames and all('TimesNewRoman' in x for x in fontnames)
    checks['pdf_embedded_font_names'] = fontnames
    checks['source_audit_file'] = str(HERE / 'CURRENT_PROTOCOL.md')
    # Numerical/layout gates do not substitute for a human/agent image review.
    checks['rendered_visual_review'] = 'pending inspection of newly extended curves'
    for name in ('B_provenance.json', 'Figure_B_neighborhood_merged_validation.json'):
        write(FIG / name, checks)
    shutil.copyfile(FIG / 'B_provenance.json', ALIASES / 'Figure_B_provenance.json')
    shutil.copyfile(FIG / 'Figure_B_neighborhood_merged_validation.json', ALIASES / 'Figure_B_neighborhood_validation.json')
    for suffix in ('pdf', 'png'):
        source = FIG / f'Figure_B_neighborhood_merged.{suffix}'
        assert sha(source) == sha(ALIASES / f'Figure_B_neighborhood.{suffix}')
        shutil.copyfile(source, HERE / f'Figure_B_neighborhood_extended.{suffix}')
    shutil.copyfile(FIG / 'source_data/B_train_plot_data.csv', HERE / 'B_train_plot_data_extended.csv')
    cap = (FIG / 'B_caption.md').read_text().strip()
    tex = cap.replace('%', r'\%')
    (HERE / 'Figure_B_neighborhood_extended_caption.md').write_text(cap + '\n')
    p = FIG / 'captions.md'
    old = p.read_text(); marker = '**Figure C.**'
    assert marker in old
    p.write_text(cap + '\n\n' + marker + old.split(marker, 1)[1])
    for p in (FIG / 'latex_includes.tex', ALIASES / 'latex_includes.tex'):
        old = p.read_text(); start = old.index(r'\caption{'); stop = old.index('\n\\label', start)
        p.write_text(old[:start] + '\\caption{' + tex + '}' + old[stop:])
    p = ALIASES / 'CAPTIONS.md'; old = p.read_text()
    start, stop = old.index('**Figure B**'), old.index('**Figure C**')
    p.write_text(old[:start] + '**Figure B**\n\n' + cap + '\n\n' + old[stop:])
    p = ALIASES / 'captions.json'; captions = json.loads(p.read_text())
    captions['Figure B'] = cap; write(p, captions)
    receipt = dict(status='complete', figure_generated_this_run=True, original_states=194,
                   new_states=288, combined_states=482, radii=checks['radii'],
                   training_rows=32, figure=str(FIG / 'Figure_B_neighborhood_merged.pdf'),
                   completed_utc=datetime.now(timezone.utc).isoformat(),
                   original_raw_data_modified=False, rendered_visual_review='pending')
    for filename in ('pdf', 'png'):
        receipt[filename + '_sha256'] = sha(FIG / f'Figure_B_neighborhood_merged.{filename}')
    # Future runs of either existing plot entrypoint reuse the completed extension.
    write(FIG / 'active_radius_extension.json', dict(directory=str(HERE), status='complete',
          merge_script_sha256=sha(HERE / 'merge_radius_extension.py')))
    write(HERE / 'POSTPROCESS.json', receipt)
    (HERE / 'RESULTS.md').write_text(
        '# Additional-radius measurement completed\n\n'
        'Both methods completed all9 additional radii ×16 directions =288 new perturbation states. '
        'All identities and shared directions match the original historical ImageNet-A B cohort. '
        'The16-radius three-panel figure has been regenerated from482 unique states. '
        'No training was run. Rendered-figure review is tracked separately.\n\n'
        '[PDF](Figure_B_neighborhood_extended.pdf) · [PNG](Figure_B_neighborhood_extended.png)\n')
    manifest_path = FIG / 'ARTIFACT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(updated_utc=receipt['completed_utc'], B_revision='historical_radius_extension_11_to_19_1x3',
                    B_review_status='pending extended-curve image inspection',
                    B_source_audit_file=str(HERE / 'CURRENT_PROTOCOL.md'))
    for item in manifest['artifacts']:
        path = ROOT / item['path']
        item.update(bytes=path.stat().st_size, sha256=sha(path))
    write(manifest_path, manifest)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
