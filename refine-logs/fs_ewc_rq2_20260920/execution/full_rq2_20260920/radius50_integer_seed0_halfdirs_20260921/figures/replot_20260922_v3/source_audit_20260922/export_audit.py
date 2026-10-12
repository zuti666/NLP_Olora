#!/usr/bin/env python3
"""Export every seed0 measurement backing current B1, with selection and lineage.

Read-only for experiment/plot sources. Writes audit outputs only alongside this
script. No model loading, GPU work, interpolation, or figure changes.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
CAMPAIGN = FIGURE.parents[1]
BASE = CAMPAIGN.parent
METHODS = ('sgd', 'gam')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def main():
    receipt_path = CAMPAIGN / 'report/radius50_complete.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'complete' and receipt['validated_jobs'] == 4
    for item in receipt['source_files'] + receipt['output_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    snapshot = {str(p): sha(p) for p in FIGURE.iterdir() if p.is_file()}
    partitions = [
        ('formal', BASE / 'formal_results/seed0', ('dense', 'sparse_train')),
        ('fisher_11_19', BASE / 'stage_path_completion_20260921/results/seed0', ('sparse_train',)),
        ('coarse_25_50', BASE / 'radius50_extension_20260921/results/seed0', ('dense', 'sparse_train')),
        ('integer_25_50', CAMPAIGN / 'results/seed0', ('dense', 'sparse_train')),
    ]
    all_rows, sources, direction_library = [], [], {}
    for partition, directory, jobs in partitions:
        for method in METHODS:
            for job in jobs:
                folder = directory / method / job
                path = folder / 'states.jsonl'
                states = [json.loads(line) for line in path.read_text().splitlines()]
                meta = read(folder / 'manifest.json')
                summary = read(folder / 'summary.json')
                assert summary['status'] == 'complete'
                assert meta['seed'] == 0 and meta['method'] == method and meta['job'] == job
                origins = [s for s in states if s['point']['radius'] == 0]
                assert len(origins) == 1
                origin = origins[0]
                chosen = 0
                for line_number, state in enumerate(states, 1):
                    finite(state)
                    assert state['anchor_restored_exactly'] is True
                    assert state['protocol_sha256'] == meta['protocol_sha256']
                    assert state['data_manifest_sha256'] == meta['data_manifest_sha256']
                    assert state['coordinate_sha256'] == meta['coordinate_sha256']
                    radius = state['point']['radius']
                    direction = state['point'].get('direction', {})
                    index = direction.get('index')
                    selected, reason = True, 'used'
                    if radius == 0 and partition != 'formal':
                        selected, reason = False, 'repeat_origin_validation_only'
                    elif job == 'dense' and radius > 0 and index >= 50:
                        selected, reason = False, 'old_dense_direction_50_to_99_not_in_current_cohort'
                    if radius > 0:
                        assert abs(state['effective_weight_drift'] - radius) <= max(1e-6, radius * 1e-6)
                        if index in direction_library:
                            assert direction == direction_library[index]
                        else:
                            direction_library[index] = direction
                    chosen += int(selected)
                    train, test = state['metrics']['train'], state['metrics']['test']
                    fisher = state['fisher_by_scope'].get('subset', {})
                    if job == 'sparse_train':
                        assert fisher['samples'] == 256 and fisher['normalized'] is False
                        assert fisher['floor'] is None and fisher['regularizer'] is None
                    delta = train['loss'] - origin['metrics']['train']['loss']
                    all_rows.append(dict(
                        source_partition=partition, method=method, seed=0, job=job,
                        source_file=str(path), source_line=line_number, state_id=state['state_id'],
                        selected_for_current_plot=selected, selection_reason=reason,
                        radius=radius, direction_index=index, direction_seed=direction.get('seed'),
                        direction_sha256=direction.get('sha256'), direction_norm=direction.get('norm'),
                        actual_weight_drift=state['effective_weight_drift'],
                        train_examples=train['samples'], train_absolute_CE=train['loss'],
                        origin_train_CE=origin['metrics']['train']['loss'],
                        train_signed_CE_change=delta, train_positive_CE_change=max(delta, 0),
                        train_accuracy_percent=train['accuracy'], test_examples=test['samples'],
                        test_absolute_CE=test['loss'], test_accuracy_percent=test['accuracy'],
                        raw_fisher_trace=fisher.get('raw_trace'), fisher_examples=fisher.get('samples'),
                        fisher_subset_CE=fisher.get('subset_loss'), protocol_sha256=state['protocol_sha256'],
                        checkpoint_sha256=state['checkpoint_sha256']['0'],
                        anchor_restored_exactly=state['anchor_restored_exactly']))
                sources.append(dict(partition=partition, method=method, seed=0, job=job,
                    directory=str(folder), raw_rows=len(states), selected_rows=chosen,
                    excluded_rows=len(states)-chosen,
                    measured_radii=sorted({s['point']['radius'] for s in states}),
                    states=dict(path=str(path), sha256=sha(path)),
                    manifest=dict(path=str(folder/'manifest.json'),sha256=sha(folder/'manifest.json')),
                    summary=dict(path=str(folder/'summary.json'),sha256=sha(folder/'summary.json')),
                    protocol_sha256=meta['protocol_sha256'],
                    config_sha256=meta['config_sha256'],
                    config=meta['config']))
    selected_rows = [r for r in all_rows if r['selected_for_current_plot']]
    assert len(all_rows) == 20526 and len(selected_rows) == 11916
    write_csv('ALL_SOURCE_MEASUREMENTS.csv', all_rows)
    write_csv('SELECTED_PLOT_MEASUREMENTS.csv', selected_rows)
    groups = defaultdict(list)
    for row in selected_rows:
        groups[(row['job'], row['method'], row['radius'])].append(row)
    recomputed = []
    for (job, method, radius), rows in sorted(groups.items()):
        count = 1 if radius == 0 else (50 if job == 'dense' else 16)
        assert len(rows) == count
        item = dict(job=job, method=method, seed=0, radius=radius, directions=count,
            G_train=math.fsum(r['train_positive_CE_change'] for r in rows)/count,
            train_accuracy_percent=math.fsum(r['train_accuracy_percent'] for r in rows)/count,
            train_absolute_CE_mean=math.fsum(r['train_absolute_CE'] for r in rows)/count,
            source_file=rows[0]['source_file'], source_partition=rows[0]['source_partition'])
        if job == 'sparse_train':
            item['raw_fisher_trace'] = math.fsum(r['raw_fisher_trace'] for r in rows)/count
        recomputed.append(item)
    write_csv('RECOMPUTED_BY_RADIUS.csv', recomputed)
    lookup = {(r['job'],r['method'],r['radius']):r for r in recomputed}
    max_error = 0.
    for file, job, values in (
        ('dense_training.csv', 'dense', ['G_train','train_accuracy_percent']),
        ('fisher_training.csv', 'sparse_train', ['G_train','train_accuracy_percent','raw_fisher_trace'])):
        for row in csv.DictReader((FIGURE/file).open()):
            calc = lookup[(job,row['method'],float(row['radius']))]
            for value in values:
                error = abs(calc[value] - float(row[value]))
                max_error = max(max_error, error)
                assert math.isclose(calc[value],float(row[value]),rel_tol=1e-12,abs_tol=1e-12)
    plot_rows = list(csv.DictReader((FIGURE/'plotted_values.csv').open()))
    for row in plot_rows:
        job='sparse_train' if row['metric']=='raw_fisher_trace' else 'dense'
        assert math.isclose(lookup[(job,row['method'],float(row['radius']))][row['metric']],
                            float(row['mean']),rel_tol=1e-12,abs_tol=1e-12)
    tolerances=list(csv.DictReader((FIGURE/'tolerance_bounds.csv').open()))
    for row in tolerances:
        maximum, passed, failed = 0., None, None
        for item in sorted([r for r in recomputed if r['job']=='dense' and r['method']==row['method']],
                           key=lambda r:r['radius']):
            maximum=max(maximum,item['G_train'])
            if maximum>float(row['epsilon']):
                failed=item['radius']; break
            passed=item['radius']
        assert passed==float(row['largest_safe_measured_radius'])
        assert failed==float(row['first_failed_measured_radius'])
    data=read(BASE/'formal_results/seed0/sgd/data_manifest.json')
    images=[]
    for split,records in data['records'].items():
        for record in records:
            images.append(dict(split=split, image_id=record['id'], label=record['label'],
                image_sha256=record['sha256'], in_fisher_subset=record['id'] in data['subset_ids']))
    write_csv('TASK0_IMAGES.csv', images)
    training={method:read(BASE/f'formal_training/anchor_{method}_seed0/run_manifest.json') for method in METHODS}
    (HERE/'TRAINING_AND_DIAGNOSTIC_CONFIGS.json').write_text(json.dumps(dict(
        training=training,data_definition={k:v for k,v in data.items() if k!='records'},
        diagnostic_configs={f"{s['partition']}_{s['method']}_{s['job']}":s['config'] for s in sources}),indent=2)+'\n')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(sources,indent=2)+'\n')
    report=dict(status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_source_files=len(sources), raw_source_rows=len(all_rows),selected_rows=len(selected_rows),
        selection_counts=dict(Counter(r['selection_reason'] for r in all_rows)),
        selected_counts_by_job=dict(Counter(r['job'] for r in selected_rows)),
        recomputed_radius_rows=len(recomputed),checked_plotted_rows=len(plot_rows),checked_tolerance_rows=len(tolerances),
        max_absolute_summary_difference=max_error, present_radii_21_to_24=sorted({r['radius'] for r in all_rows if r['radius'] in [21,22,23,24]}),
        publication_receipt=dict(path=str(receipt_path),sha256=sha(receipt_path)),
        original_publication_source_files_verified=len(receipt['source_files']),
        original_publication_outputs_verified=len(receipt['output_files']),
        same_direction_hashes_all_sources_and_methods=True,no_gpu_work=True,
        previous_artifacts_preserved=True)
    for path,digest in snapshot.items():
        assert sha(path)==digest, path
    (HERE/'AUDIT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
