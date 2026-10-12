#!/usr/bin/env python3
"""Check final SVG/PDF artifacts, cohort replacement, and retained task curves."""
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import fitz
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE/'output'
REFERENCE = HERE.parent/'lora_taskwise_20260918/aa_times'
NS = {'s': 'http://www.w3.org/2000/svg'}
NUMBER = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
COLORS = {'Vanilla LoRA': '#777777', 'SD-LoRA': '#d55e00',
          'EWCLoRA': '#009e73', 'GAM-EWC+Merge': '#cc3377'}


def axes_in(path):
    return [g for g in ET.parse(path).findall('.//s:g', NS)
            if g.get('id', '').startswith('axes_')]


def paths_by_color(ax):
    return {color: path.get('d') for group in ax if group.get('id', '').startswith('line2d_')
            for path in group if path.tag.endswith('path') and path.get('clip-path')
            for color in COLORS.values() if color in path.get('style', '')}


def main():
    manifest = json.loads((HERE/'sources.json').read_text())
    old = json.loads((REFERENCE/'sources.json').read_text())
    original = {(r['dataset'], r['method'], r['seed']): r for r in old['runs']}
    retained = 0
    for run in manifest['runs']:
        if 'metrics_path' not in run:
            assert run == original[(run['dataset'], run['method'], run['seed'])]
            retained += 1
    assert retained == 35 and len(manifest['runs']) == 41
    assert not any(r['method']=='InfLoRA' for r in manifest['runs'])
    old_paths = {r['log'] for r in manifest['replaced_historical_runs']}
    assert not old_paths.intersection(r['log'] for r in manifest['runs'])
    with (OUTPUT/'stage_statistics.csv').open() as f:
        stats = list(csv.DictReader(f))
    checks = []
    reference_stem = 'lora_taskwise_aa_compact_tighty_no_inflora'
    reference_axes = axes_in(REFERENCE/'output'/f'{reference_stem}.svg')
    retained_task_curves = 0
    for metric in ['sample', 'task']:
        stem = f'lora_taskwise_{metric}_aa_updated'
        meta = json.loads((OUTPUT/f'{stem}_display.json').read_text())
        new_axes = axes_in(OUTPUT/f'{stem}.svg')
        point_count, max_error = 0, 0.0
        for index, (ax, dataset) in enumerate(zip(new_axes, manifest['dataset_order'])):
            patch = next(g for g in ax if g.get('id', '').startswith('patch_'))
            corners = np.array([float(v) for v in NUMBER.findall(patch[0].get('d'))]).reshape(-1, 2)
            xleft, ytop = corners.min(axis=0)
            xright, ybottom = corners.max(axis=0)
            lo, hi = meta['y_ranges'][dataset]['limits']
            curves = paths_by_color(ax)
            assert set(curves) == set(COLORS.values())
            steps = 5 if dataset=='domainnet' else 10
            for method, color in COLORS.items():
                coords = np.array([float(v) for v in NUMBER.findall(curves[color])]).reshape(-1, 2)
                observed_x = .8+(coords[:,0]-xleft)/(xright-xleft)*(steps-.6)
                observed_y = hi-(coords[:,1]-ytop)/(ybottom-ytop)*(hi-lo)
                rows = sorted([r for r in stats if (r['dataset'], r['method'], r['metric'])==
                               (dataset, method, metric)], key=lambda r:int(r['stage']))
                expected = np.array([float(r['mean']) for r in rows])
                np.testing.assert_allclose(observed_x, np.arange(1,steps+1), atol=1e-6, rtol=0)
                np.testing.assert_allclose(observed_y, expected, atol=1e-6, rtol=0)
                assert all(lo<=float(r['lower'])<=float(r['upper'])<=hi for r in rows)
                point_count += len(expected)
                max_error = max(max_error, float(np.max(np.abs(observed_y-expected))))
                updated = dataset in ['imagenetr','imageneta'] and method=='GAM-EWC+Merge'
                if metric=='task' and not updated:
                    assert curves[color] == paths_by_color(reference_axes[index])[color]
                    retained_task_curves += 1
        with fitz.open(OUTPUT/f'{stem}.pdf') as doc, fitz.open(REFERENCE/'output'/f'{reference_stem}.pdf') as previous:
            assert len(doc)==1 and not doc[0].get_images()
            assert tuple(doc[0].rect)==tuple(previous[0].rect)
            text = doc[0].get_text()
            assert 'InfLoRA' not in text and 'FS-EWC n=3/3/6/4' in text
            spans = [s for b in doc[0].get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
            assert all(doc[0].rect.contains(fitz.Rect(s['bbox'])) for s in spans)
            titles = [s['text'] for s in sorted(spans,key=lambda s:s['bbox'][0])
                      if s['text'] in ['ImageNet-R','ImageNet-A','DomainNet','CIFAR-100']]
            assert titles == ['ImageNet-R','ImageNet-A','DomainNet','CIFAR-100']
        checks.append({'metric': metric, 'verified_svg_points': point_count,
                       'max_svg_rounding_error_pp': max_error,
                       'all_curve_bands_inside_axes': True, 'all_pdf_text_inside_page': True,
                       'page_size_identical_to_reference': True, 'vector_pdf': True})
    assert retained_task_curves == 14
    result = {'retained_raw_log_records_identical': retained,
              'new_rho1_runs': 6, 'replaced_historical_runs_not_loaded': 12,
              'retained_task_metric_curves_with_identical_svg_coordinates': retained_task_curves,
              'plots': checks}
    (OUTPUT/'render_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
              and p.name!='ARTIFACT_MANIFEST.json'}
    (HERE/'ARTIFACT_MANIFEST.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
