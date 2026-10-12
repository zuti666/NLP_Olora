#!/usr/bin/env python3
"""Idempotent CPU-only reader for the three-gaps fixed offline protocol.

No torch/model/source/queue imports, checkpoint loading, GPU discovery, training,
or writes outside results/offline_analysis. Partial restored cells are useful
even before their aggregate receives a completion receipt. Execution validity,
job completion and scientific acceptance are deliberately separate quantities.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/olora_offline_collector_mpl')
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'imageneta-response-state-chain-gpu456-20261011T195426Z'
SEEDS = (0, 1, 2)
BRANCHES = ('S', 'R')
DATASETS = dict(train_old=1141, train_new=906, dev_old=118, dev_new=91,
                test_old=307, test_new=254)
EPOCHS = (0, 1, 10, 25, 50)
GRIDS = dict(original=(.001, .003, .01),
             exploratory=(1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01))
KINDS = ('curvature', 'matching', 'evaluation')
PLANNED = dict(curvature=36, matching=24, evaluation=30)
PROTOCOL = 'three_gaps_live_cpu_collector_v1_20261012'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON observation')
    if isinstance(value, dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value, list):
        for v in value: finite_tree(v)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Identical inputs produce identical bytes and do not touch artifact mtimes.
    if path.exists() and path.read_text() == value: return
    fd, name = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False,
                                 allow_nan=False, sort_keys=True)+'\n')


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=keys or ['no_available_rows'], lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                     if isinstance(v, (dict, list, tuple)) else v
                     for k, v in row.items()} for row in rows)
    atomic_text(path, handle.getvalue())


class Reader:
    """Read each JSON once; record the bytes used, including missing inputs."""
    def __init__(self):
        self.ledger = {}
        self.cache = {}
        self.pinned = {}

    def json(self, path):
        path = Path(path)
        key = str(path.absolute())
        if key in self.cache: return self.cache[key]
        entry = dict(path=key, available=path.exists(), used=False, sha256=None)
        value = None
        if path.exists():
            try:
                raw = path.read_bytes()
                entry['sha256'] = hashlib.sha256(raw).hexdigest()
                value = json.loads(raw)
                finite_tree(value)
                entry.update(used=True, bytes=len(raw))
            except (OSError, ValueError, TypeError) as error:
                entry['error'] = str(error)
        self.ledger[key] = entry
        self.cache[key] = value
        return value

    def binary(self, path):
        path = Path(path)
        key = str(path.absolute())
        if key in self.ledger: return self.ledger[key]
        entry = dict(path=key, available=path.exists(), used=False, sha256=None)
        try:
            if path.exists():
                entry.update(sha256=digest(path), bytes=path.stat().st_size, used=True)
        except OSError as error:
            entry['error'] = str(error)
        self.ledger[key] = entry
        return entry


def require(condition, message):
    if not condition: raise ValueError(message)


def restored(record):
    require(isinstance(record, dict) and bool(record) and
            all(v is True for v in record.values()), 'State/RNG restoration not exact')


def identity(row, kind):
    return (int(row['seed']), str(row['branch']), int(row['epoch'])) + (
        (int(row['batch_index']),) if kind != 'evaluation' else ())


def expected(job):
    seed = int(job['seed'])
    if job['pilot']:
        if job['kind'] == 'curvature': return {(seed, 'S', 1, 0)}
        if job['kind'] == 'matching': return {(seed, 'S', 10, 0)}
        return {(seed, b, 1) for b in BRANCHES}
    epochs = EPOCHS if job['kind'] == 'evaluation' else (
        (1, 10, 25) if job['kind'] == 'curvature' else (10, 25))
    if job['kind'] == 'evaluation':
        return {(seed, b, e) for b in BRANCHES for e in epochs}
    return {(seed, b, e, batch) for b in BRANCHES for e in epochs for batch in (0, 1)}


def adjacent_gate(rows, limit=.08):
    ok = [r.get('degenerate') is False and isinstance(r.get('relative_error'), (int, float))
          and math.isfinite(r['relative_error']) and r['relative_error'] <= limit for r in rows]
    return any(a and b for a, b in zip(ok, ok[1:]))


def validate_gate(gate):
    """Recompute both gates; first-order agreement never substitutes for HVPs."""
    require(gate.get('fixed_base') is True and gate.get('second_hvp_vector_detached') is True,
            'Hessian BASE/detachment contract differs')
    first = gate['first_order']
    comparison = first['comparison']
    first_ok = (first.get('repeated_gradient_exact') is True and
                comparison.get('degenerate') is False and
                isinstance(comparison.get('relative_error'), (int, float)) and
                comparison['relative_error'] <= 2e-4 and
                isinstance(comparison.get('cosine'), (int, float)) and
                comparison['cosine'] >= .99999)
    require(first.get('accepted') == first_ok, 'First-order gate stored flag inconsistent')
    original, extended = [], []
    for product in ('H_g0', 'H_detached_H_g0'):
        block = gate.get('finite_difference', {}).get(product)
        if block is None:
            original.append(False); extended.append(False)
            continue
        for name, distances in GRIDS.items():
            rows = block[name]
            require(tuple(r['distance'] for r in rows) == distances, 'FD grid scope differs')
            passed = adjacent_gate(rows)
            require(block[name+'_passed'] == passed, 'Stored adjacent-point gate differs')
            (original if name == 'original' else extended).append(passed)
    op, ep = bool(first_ok and all(original)), bool(first_ok and all(extended))
    require(gate.get('original_gate_accepted') == op, 'Original HVP acceptance inconsistent')
    require(gate.get('extended_gate_accepted') == ep, 'Exploratory HVP acceptance inconsistent')
    require(gate.get('accepted') == ep, 'Overall exploratory HVP acceptance inconsistent')
    return op, ep


def validate_arrays(row, reader):
    path = Path(row['logits_file'])
    entry = reader.binary(path)
    require(entry['available'] and entry['sha256'] == row['logits_sha256'], 'Saved logits missing/hash differs')
    with np.load(path, allow_pickle=False) as arrays:
        for name, count in DATASETS.items():
            prefix = name+'__'
            labels, logits = arrays[prefix+'labels'], arrays[prefix+'logits']
            require(logits.shape == (count, 40) and labels.shape == (count,), 'Saved logits shape differs')
            require(logits.dtype == np.float32 and np.isfinite(logits).all(), 'Logits not finite native FP32')
            require(np.issubdtype(labels.dtype, np.integer), 'Noninteger labels')
            lo = 0 if name.endswith('_old') else 20
            require(((labels >= lo) & (labels < lo+20)).all(), 'Saved labels outside group')
            own = logits[:, lo:lo+20]
            other = logits[:, 20:40] if lo == 0 else logits[:, :20]
            pred = logits.argmax(1)
            conditional_pred = own.argmax(1)+lo
            cross = (pred < lo) | (pred >= lo+20)
            within = (pred != labels) & ~cross
            metrics = row['datasets'][name]
            for key, value in (('prediction_joint40', pred), ('prediction_conditional20', conditional_pred),
                               ('cross_group_error', cross), ('within_group_error', within)):
                require(np.array_equal(arrays[prefix+key], value), 'Saved prediction/error flags inconsistent')
            require(int((pred == labels).sum()) == metrics['joint40']['correct'], 'Logits/joint count inconsistent')
            require(int((conditional_pred == labels).sum()) == metrics['conditional20']['correct'],
                    'Logits/conditional count inconsistent')
            require(int(cross.sum()) == metrics['cross_group_error_count'] and
                    int(within.sum()) == metrics['within_group_error_count'], 'Logits error partition differs')
            require(int(((conditional_pred == labels) & cross).sum()) ==
                    metrics['conditional_correct_lost_to_other_group_count'], 'Competition-only count differs')
            require(int(((conditional_pred != labels) & cross).sum()) ==
                    metrics['conditional_error_and_cross_group_count'], 'Overlapping conditional/cross errors differ')
            # Compare saved per-example CE identity, without recreating GPU reduction.
            cce, jce = arrays[prefix+'conditional_ce'], arrays[prefix+'joint_ce']
            penalty = arrays[prefix+'ce_competition_penalty']
            require(cce.shape == jce.shape == penalty.shape == (count,) and
                    np.isfinite(cce).all() and np.isfinite(jce).all() and np.isfinite(penalty).all(),
                    'Saved CE arrays missing/nonfinite')
            error = float(np.max(np.abs(jce.astype(float)-cce-penalty)))
            require(error <= 3e-6, 'Saved CE competition partition failed')
            require(abs(float(np.mean(penalty, dtype=np.float64))-
                        metrics['ce_competition_penalty']) <= 3e-6, 'Saved CE penalty differs')
            # Float64 logsumexp independent check of CE and competition term.
            z = logits.astype(np.float64)
            def lse(v):
                high = v.max(1)
                return high+np.log(np.exp(v-high[:, None]).sum(1))
            joint = lse(z)-z[np.arange(count), labels]
            conditional = lse(z[:, lo:lo+20])-z[np.arange(count), labels]
            competition = lse(z)-lse(z[:, lo:lo+20])
            require(np.max(np.abs(joint-jce)) <= 3e-6 and
                    np.max(np.abs(conditional-cce)) <= 3e-6 and
                    np.max(np.abs(competition-penalty)) <= 3e-6,
                    'CE arrays disagree with saved logits')
            require(abs(float(joint.mean())-metrics['joint40']['ce']) <= 1e-5 and
                    abs(float(conditional.mean())-metrics['conditional20']['ce']) <= 1e-5,
                    'Dataset CE means disagree with logits')
            for key in arrays.files:
                value = arrays[key]
                require(not np.issubdtype(value.dtype, np.floating) or np.isfinite(value).all(),
                        'Nonfinite saved array '+key)
    return dict(logits_sha256=entry['sha256'], partition_recomputed_from_saved_logits=True)


def validate_row(row, job, reader):
    finite_tree(row)
    key = identity(row, job['kind'])
    require(key in expected(job), 'Unexpected fixed cell identity')
    require(row.get('checkpoint_sha256') and row.get('checkpoint'), 'Checkpoint provenance missing')
    checkpoint = Path(row['checkpoint'])
    require(checkpoint.name == 'epoch_%03d.pt'%row['epoch'] and
            checkpoint.parent.parent.name == 'formal_%s_s%d'%(row['branch'], row['seed']),
            'Checkpoint path identity inconsistent')
    if reader.pinned:
        require(reader.pinned.get(str(checkpoint.resolve())) == row['checkpoint_sha256'],
                'Checkpoint recorded hash differs from frozen implementation manifest')
    require(row.get('submitted_training_updates', 0) == 0, 'Offline cell submitted training update')
    result = dict(validated=True)
    if job['kind'] == 'curvature':
        d = row['diagnostic']; restored(d['restoration'])
        require(d['submitted_training_updates'] == 0, 'Curvature training update')
        require(bool(d.get('fp64_fallback_declared')) == bool(job['pilot']), 'Formal FP64 fallback forbidden')
        if 'fp64_restoration' in d: restored(d['fp64_restoration'])
        if not job['pilot']: require('fp64' not in d, 'FP64 in formal curvature cell')
        op, ep = validate_gate(d['fp32']) if 'fp32' in d else (False, False)
        fp64_original, fp64_extended = validate_gate(d['fp64']) if 'fp64' in d else (False, False)
        permitted = ep or fp64_extended
        require(d['candidate_comparisons_permitted'] == permitted, 'Unverified Hessian candidate comparison')
        require(len(d['scales']) == (2 if permitted else 0), 'Missing/unverified scale comparisons')
        if permitted:
            require([r['scale'] for r in d['scales']] == [1., .5], 'Declared joint half-scales differ')
            for scale in d['scales']:
                require(scale['candidate_precision'] == 'torch.float32', 'Native candidate precision changed')
                require(scale['curvature_precision'] == ('torch.float64_exploratory_fallback'
                        if fp64_extended and not ep else 'torch.float32'), 'Curvature reference precision differs')
        result.update(scientific_status=d['status'], original_fp32_gate=op,
                      exploratory_fp32_gate=ep, exploratory_fp64_gate=fp64_extended,
                      fp64_original_grid_gate=fp64_original, candidate_comparison_permitted=permitted)
    elif job['kind'] == 'matching':
        restored(row['restoration']); d = row['match']; restored(d['restoration'])
        require(row['HVP_calls'] == 0 and d['submitted_training_updates'] == 0, 'Matching has HVP/training')
        accepted = d['status'] == 'accepted'
        if accepted:
            require(d['native_repeat_exact_effective_norms'] is True, 'Native repeat differs')
            require(set(d['direction_parts']) == {'all', 'lora_only', 'fc_only'}, 'Missing vector partitions')
            require(d['scalar_solver']['scale'] > 0 and d['norm_match_relative_error'] <= 1e-8,
                    'Finite effective norm scalar match failed')
            require(abs(d['g2_matched']['effective_norm']-d['sam']['effective_norm']) <=
                    max(1e-12, 1e-8*d['sam']['effective_norm']), 'Reported matched norms differ')
            for component in ('sam', 'g2_native', 'g2_matched'):
                require(len(d[component]['layers']) == 24, 'Missing all24 LoRA layers')
                require({r['layer_index'] for r in d[component]['layers']} == set(range(24)),
                        'Duplicate/missing LoRA layer identity')
        result.update(scientific_status=d['status'], accepted_match=accepted)
    else:
        restored(row['restoration'])
        require(row['status'] == 'accepted' and row['test_used_for_selection'] is False, 'Invalid checkpoint evaluation')
        require(row.get('full_model_parameter_sha256') and row.get('dev_split_sha256'), 'Evaluation state provenance missing')
        require(set(row['datasets']) == set(DATASETS), 'Six fixed datasets required')
        require(bool(row['agreement']) and all(x['accepted'] is True for x in row['agreement'].values()),
                'Historical metrics disagreement')
        for name, count in DATASETS.items():
            d = row['datasets'][name]
            require(d['samples'] == count and d['group'] == name.split('_')[1], 'Fixed dataset/group count differs')
            for mode in ('conditional20', 'joint40'):
                block = d[mode]
                require(block['correct']+block['errors'] == count and
                        abs(block['accuracy']-100*block['correct']/count) <= 1e-8,
                        'Accuracy/count identity differs')
            require(d['cross_group_error_count']+d['within_group_error_count'] == d['joint40']['errors'],
                    'Cross/within error partition failed')
            require(d['partition_identity_passed'] is True and d['ce_partition_max_abs_error'] <= 3e-6,
                    'CE partition gate failed')
            require(abs(d['joint40']['ce']-d['conditional20']['ce']-d['ce_competition_penalty']) <= 1e-5,
                    'CE mean partition differs')
        require(abs(row['FAA']-(row['datasets']['test_old']['joint40']['accuracy']+
                               row['datasets']['test_new']['joint40']['accuracy'])/2) <= 1e-8,
                'FAA alias inconsistent')
        result.update(validate_arrays(row, reader), scientific_status='fixed_checkpoint_evaluation_accepted')
    return result


def load_job(job, reader, queue):
    folder = Path(job['output_dir'])
    report = reader.json(folder/job['report_file'])
    receipt = reader.json(folder/'completion_receipt.json')
    job_state = queue.get('jobs', {}).get(job['id'], {})
    exit_record = reader.json(HERE/'runtime/jobs'/job['id']/'exit.json')
    rows, issues = {}, []
    snapshots = []
    if report is not None:
        try:
            require(report['seed'] == job['seed'] and report['pilot'] == job['pilot'], 'Aggregate identity differs')
            require(report['submitted_training_updates'] == 0, 'Aggregate training updates')
            snapshots.extend(report['results'])
        except (KeyError, TypeError, ValueError) as error:
            issues.append('aggregate: '+str(error))
    pattern = 'checkpoint*_batch*.json' if job['kind'] == 'curvature' else (
              'branch*_epoch*_batch*.json' if job['kind'] == 'matching' else 'branch*_epoch*.json')
    for path in sorted(folder.glob(pattern)):
        value = reader.json(path)
        if value is not None: snapshots.append(value)
    invalid = []
    conflicts = set()
    for row in snapshots:
        try:
            key = identity(row, job['kind'])
            if key in rows:
                if rows[key] != row: conflicts.add(key)
                continue
            rows[key] = row
        except (KeyError, TypeError, ValueError) as error:
            invalid.append(dict(row=row, reason=str(error)))
    accepted, all_rows, cell_ledger = [], [], []
    for key in sorted(expected(job) | set(rows)):
        row = rows.get(key)
        base = dict(job_id=job['id'], kind=job['kind'], stratum='pilot' if job['pilot'] else 'formal',
                    seed=key[0], branch=key[1], epoch=key[2], batch_index=key[3] if len(key)>3 else None,
                    expected=key in expected(job), available=row is not None, validated=False,
                    queue_job_status=job_state.get('status', 'not_launched'))
        if row is not None:
            all_rows.append(row)
            try:
                require(key not in conflicts, 'Conflicting aggregate/per-cell snapshots')
                audit = validate_row(row, job, reader)
                base.update(audit)
                accepted.append(dict(job_id=job['id'], kind=job['kind'], stratum=base['stratum'],
                                     row=row, audit=audit))
            except (KeyError, TypeError, ValueError, OSError, IndexError) as error:
                base['invalid_reason'] = str(error)
        cell_ledger.append(base)
    receipt_valid = False
    if receipt is not None and report is not None:
        try:
            require(receipt['submitted_training_updates'] == 0, 'Receipt training update')
            require(receipt['results_sha256'] == reader.ledger[str((folder/job['report_file']).absolute())]['sha256'],
                    'Receipt/report hash differs (may be a live snapshot)')
            require(len(report['results']) == job['expected_items'] == len(expected(job)), 'Receipt scope incomplete')
            require(len(accepted) == job['expected_items'] and not invalid, 'Receipt has invalid/unexpected cells')
            require(receipt['status'] == ('accepted_execution' if job['kind']=='curvature' else 'accepted'),
                    'Receipt execution status differs')
            require(report['status'] == ('completed_exploratory' if job['kind']=='curvature' else 'completed'),
                    'Aggregate not execution complete')
            receipt_valid = True
        except (KeyError, TypeError, ValueError) as error:
            issues.append('receipt: '+str(error))
    job_ledger = dict(job_id=job['id'], kind=job['kind'], pilot=job['pilot'], seed=job['seed'],
        expected_items=job['expected_items'], available_cells=len(rows), validated_executed_cells=len(accepted),
        report_status=report.get('status') if report else None, queue_status=job_state.get('status', 'not_launched'),
        exit_returncode=exit_record.get('returncode') if exit_record else None,
        receipt_present=receipt is not None, receipt_valid=receipt_valid,
        execution_complete=receipt_valid and job_state.get('status') == 'completed' and
                           exit_record is not None and exit_record.get('returncode') == 0,
        issues=issues, malformed_rows=invalid)
    return accepted, all_rows, cell_ledger, job_ledger


def stat(values, seeds, required=SEEDS):
    return dict(n=len(values), seeds=list(seeds), values=list(values),
                mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                required_seed_set=list(required), complete_seed_cohort=tuple(sorted(seeds)) == tuple(required))


def summarize_cells(rows, group_keys, metric_keys):
    """Cells/layers never enlarge independent n: first cells -> seed, then seeds."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k) for k in group_keys)].append(row)
    seed_rows, summaries = [], []
    for group, members in sorted(grouped.items(), key=lambda x:repr(x[0])):
        labels = dict(zip(group_keys, group))
        by_seed = collections.defaultdict(list)
        for row in members: by_seed[row['seed']].append(row)
        for metric in metric_keys:
            values, seeds, counts = [], [], []
            for seed, cells in sorted(by_seed.items()):
                actual = [r[metric] for r in cells if isinstance(r.get(metric), (float, int)) and
                          not isinstance(r.get(metric), bool) and math.isfinite(r[metric])]
                if not actual: continue
                value = statistics.mean(actual)
                seed_rows.append(dict(labels, metric=metric, seed=seed, available_cells=len(actual),
                                      cell_values=actual, within_seed_cell_mean=value))
                values.append(value); seeds.append(seed); counts.append(len(actual))
            summaries.append(dict(labels, metric=metric, **stat(values, seeds), cells_per_seed=counts,
                                  estimand='unweighted cell mean within seed, then seed mean/sample SD'))
    return seed_rows, summaries


def flatten_numeric(value, prefix=''):
    out = {}
    for key, item in value.items():
        name = prefix+key
        if isinstance(item, dict): out.update(flatten_numeric(item, name+'.'))
        elif isinstance(item, (float, int)) and not isinstance(item, bool): out[name] = item
    return out


def base(item):
    r = item['row']
    return dict(stratum=item['stratum'], job_id=item['job_id'], seed=r['seed'], branch=r['branch'],
                epoch=r['epoch'], batch_index=r.get('batch_index'), checkpoint_sha256=r['checkpoint_sha256'])


def curvature_tables(items):
    cells, grid, corrections = [], [], []
    for item in items:
        r, d, b = item['row'], item['row']['diagnostic'], base(item)
        cells.append(dict(b, **item['audit'], failure_layer=d.get('failure_layer'),
                          probe_lr=r['probe_lr'], epoch_lr_used=r['epoch_lr_used']))
        for precision in ('fp32', 'fp64'):
            gate = d.get(precision)
            if not gate: continue
            for product, block in gate.get('finite_difference', {}).items():
                for name in GRIDS:
                    for point in block[name]:
                        grid.append(dict(b, precision=precision, product=product, grid=name,
                                         grid_passed=block[name+'_passed'], **point))
        for scale in d['scales']:
            corrections.append(dict(b, **scale, original_fp32_gate=item['audit']['original_fp32_gate'],
                                    exploratory_fp32_gate=item['audit']['exploratory_fp32_gate'],
                                    exploratory_fp64_gate=item['audit']['exploratory_fp64_gate']))
    return cells, grid, corrections


def half_scale_changes(corrections):
    grouped=collections.defaultdict(dict)
    for row in corrections:
        key=tuple(row[k] for k in ('stratum','seed','branch','epoch','batch_index','curvature_precision','candidate_precision'))
        grouped[key][row['scale']]=row
    changes=[]
    for key,rows in sorted(grouped.items(),key=lambda x:repr(x[0])):
        if set(rows)!={1.,.5}:continue
        full,half=rows[1.],rows[.5]
        out=dict(zip(('stratum','seed','branch','epoch','batch_index','curvature_precision','candidate_precision'),key))
        for metric in ('relative_error','cosine','delta_native_norm','delta_quadratic_norm'):
            a,b=full.get(metric),half.get(metric)
            out[metric+'_half_minus_full']=b-a if a is not None and b is not None else None
            out[metric+'_half_over_full']=b/a if a is not None and b is not None and a!=0 else None
        changes.append(out)
    return changes


def matching_tables(items):
    cells, layers = [], []
    for item in items:
        d, b = item['row']['match'], base(item)
        row = dict(b, status=d['status'], accepted_match=item['audit']['accepted_match'],
                   mapping_precision='float64 finite BA/FC geometry from represented native FP32 parameters/gradients',
                   probe_lr=item['row']['probe_lr'], **flatten_numeric(d))
        for candidate in ('sam', 'g2_native', 'g2_matched'):
            parts = d.get(candidate, {}).get('layers', [])
            for layer in parts: layers.append(dict(b, candidate=candidate, **layer))
            for metric in ('delta_Q_norm', 'delta_Q_first_order_norm', 'delta_Q_second_order_norm',
                           'pi_A', 'T_A_norm', 'T_B_norm', 'T_cross_norm', 'T_A_T_B_cosine'):
                values = [layer[metric] for layer in parts if layer.get(metric) is not None]
                row[candidate+'.'+metric+'.mean_layers'] = statistics.mean(values) if values else None
                row[candidate+'.'+metric+'.available_layers'] = len(values)
        cells.append(row)
    return cells, layers


def evaluation_tables(items):
    metrics, gaps = [], []
    for item in items:
        r, b = item['row'], base(item)
        for name, d in r['datasets'].items():
            metrics.append(dict(b, dataset=name, split=name.split('_')[0], group=d['group'],
                                FAA=r['FAA'], **flatten_numeric(d)))
        for group in ('old', 'new'):
            train = r['datasets']['train_'+group]
            for split in ('dev', 'test'):
                target = r['datasets'][split+'_'+group]
                for mode in ('conditional20', 'joint40'):
                    gaps.append(dict(b, group=group, heldout_split=split, mode=mode,
                        train_accuracy=train[mode]['accuracy'], heldout_accuracy=target[mode]['accuracy'],
                        train_minus_heldout_accuracy_pp=train[mode]['accuracy']-target[mode]['accuracy'],
                        heldout_minus_train_ce=target[mode]['ce']-train[mode]['ce']))
    return metrics, gaps


def paired_evaluation(metrics, gaps, e0):
    pairs, summaries = [], []
    grouped = collections.defaultdict(dict)
    for row in metrics:
        if row['stratum']=='formal': grouped[row['seed'], row['epoch'], row['dataset']][row['branch']] = row
    for (seed, epoch, dataset), arms in sorted(grouped.items()):
        if set(arms) != set(BRANCHES): continue
        s, r = arms['S'], arms['R']
        shared_verified = e0.get(seed, {}).get('verified', False)
        for metric in sorted(set(s) & set(r)):
            if metric in ('seed', 'epoch', 'batch_index', 'samples') or not isinstance(s[metric], (int, float)): continue
            if isinstance(s[metric], bool): continue
            pairs.append(dict(seed=seed, epoch=epoch, dataset=dataset, metric=metric,
                S=s[metric], R=r[metric], R_minus_S=r[metric]-s[metric], shared_e0_verified=shared_verified))
        # Native evaluator used different FP32 mean reduction paths; retain the
        # tiny residual rather than pretending the observed means are exact.
        residual = ((r['joint40.ce']-s['joint40.ce'])-
                    (r['conditional20.ce']-s['conditional20.ce'])-
                    (r['ce_competition_penalty']-s['ce_competition_penalty']))
        pairs.append(dict(seed=seed,epoch=epoch,dataset=dataset,metric='paired_CE_partition_residual',
                          S=s['joint40.ce']-s['conditional20.ce']-s['ce_competition_penalty'],
                          R=r['joint40.ce']-r['conditional20.ce']-r['ce_competition_penalty'],
                          R_minus_S=residual,shared_e0_verified=shared_verified,
                          paired_CE_partition_identity_passed=abs(residual)<=2e-5))
    grouped_pairs = collections.defaultdict(list)
    for row in pairs: grouped_pairs[row['epoch'], row['dataset'], row['metric']].append(row)
    for (epoch, dataset, metric), rows in sorted(grouped_pairs.items()):
        values, seeds = [r['R_minus_S'] for r in rows], [r['seed'] for r in rows]
        ss = stat(values, seeds)
        summaries.append(dict(epoch=epoch, dataset=dataset, metric=metric, contrast='R minus S', **ss,
            positive_count=sum(v>0 for v in values), negative_count=sum(v<0 for v in values), tie_count=sum(v==0 for v in values),
            primary_endpoint=epoch in (25,50), all_shared_e0_verified=all(r['shared_e0_verified'] for r in rows),
            formal_paired_stat_available=ss['complete_seed_cohort'] and all(r['shared_e0_verified'] for r in rows),
            interpretation='paired descriptive n=3 only when all required seeds/e0 identities are available; no significance claim'))
    gap_pairs = []
    grouped_gaps = collections.defaultdict(dict)
    for row in gaps:
        if row['stratum']=='formal':
            grouped_gaps[row['seed'],row['epoch'],row['group'],row['heldout_split'],row['mode']][row['branch']] = row
    for (seed, epoch, group, split, mode), arms in sorted(grouped_gaps.items()):
        if set(arms) != set(BRANCHES): continue
        gap_pairs.append(dict(seed=seed,epoch=epoch,group=group,heldout_split=split,mode=mode,
            R_minus_S_train_minus_heldout_accuracy_pp=arms['R']['train_minus_heldout_accuracy_pp']-arms['S']['train_minus_heldout_accuracy_pp'],
            R_minus_S_heldout_minus_train_ce=arms['R']['heldout_minus_train_ce']-arms['S']['heldout_minus_train_ce']))
    return pairs, summaries, gap_pairs


def verify_shared_start(items, reader):
    indexed = {(r['row']['seed'],r['row']['branch']):r['row'] for r in items
               if r['stratum']=='formal' and r['row']['epoch']==0}
    result = {}
    for seed in SEEDS:
        out = dict(seed=seed, available_branches=[b for b in BRANCHES if (seed,b) in indexed], verified=False)
        if len(out['available_branches']) == 2:
            s, r = indexed[seed,'S'], indexed[seed,'R']
            out['full_parameter_sha256_equal'] = s['full_model_parameter_sha256'] == r['full_model_parameter_sha256']
            out['dev_split_sha256_equal'] = s['dev_split_sha256'] == r['dev_split_sha256']
            out['all_saved_npz_arrays_equal'] = False
            with np.load(s['logits_file'], allow_pickle=False) as a, np.load(r['logits_file'], allow_pickle=False) as b:
                keys_equal = set(a.files)==set(b.files)
                differences = [k for k in sorted(set(a.files)&set(b.files))
                               if a[k].dtype!=b[k].dtype or not np.array_equal(a[k],b[k])]
                out.update(array_keys_equal=keys_equal, differing_array_keys=differences,
                           compared_array_count=len(set(a.files)&set(b.files)),
                           all_saved_npz_arrays_equal=keys_equal and not differences)
            out['verified'] = out['full_parameter_sha256_equal'] and out['dev_split_sha256_equal'] and out['all_saved_npz_arrays_equal']
        result[seed] = out
    return result


def original_curvature(reader, items):
    original, grids, comparison = [], [], []
    new = {(i['row']['seed'],i['row']['branch'],i['row']['epoch'],i['row']['batch_index']):i
           for i in items if i['stratum']=='formal'}
    for path in sorted((OLD/'outputs').glob('offline*/diagnostic_results.json')):
        report = reader.json(path)
        if not report: continue
        for row in report.get('results', []):
            gate = row.get('correction', {}).get('numerical_gate', {})
            record = dict(source=str(path), source_sha256=reader.ledger[str(path.absolute())]['sha256'],
                stratum='historical_original', seed=row['seed'], branch=row['branch'], epoch=row['epoch'],
                batch_index=row['batch_index'], checkpoint_sha256=row['checkpoint_sha256'],
                original_gate_accepted=gate.get('accepted', False), original_gate_status=gate.get('status'),
                report_status=report.get('status'), raw_row=row)
            original.append(record)
            for product, block in gate.get('finite_difference', {}).items():
                for point in block.get('grid', []):
                    grids.append(dict(source=str(path),stratum='historical_original',seed=row['seed'],branch=row['branch'],
                                      epoch=row['epoch'],batch_index=row['batch_index'],product=product,grid='original',precision='fp32',**point))
            current = new.get((row['seed'],row['branch'],row['epoch'],row['batch_index']))
            cmp = dict(seed=row['seed'],branch=row['branch'],epoch=row['epoch'],batch_index=row['batch_index'],
                       historical_original_gate=gate.get('accepted',False), new_formal_available=current is not None)
            if current:
                nr = current['row']
                cmp.update(checkpoint_hash_same=row['checkpoint_sha256']==nr['checkpoint_sha256'],
                           batch_identity_same=row['batch_identity']==nr['batch_identity'],
                           new_original_fp32_gate=current['audit']['original_fp32_gate'],
                           new_exploratory_fp32_gate=current['audit']['exploratory_fp32_gate'])
                cmp['same_state_same_batch_comparison'] = cmp['checkpoint_hash_same'] and cmp['batch_identity_same']
            comparison.append(cmp)
    return original, grids, comparison


def plot_evaluation(out, metrics, gaps, content_digest):
    """Standalone figure after all formal fixed evals and e0 checks are valid."""
    receipt_path = out/'EVALUATION_FIGURE.json'
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    paths = [out/'CONDITIONAL_JOINT_GAPS.png',out/'CONDITIONAL_JOINT_GAPS.pdf']
    if prior.get('input_sha256') == content_digest and all(p.exists() for p in paths): return prior
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 3, figsize=(13.6,7.5), sharex=True)
    color = {'S':'#31688e','R':'#b63679'}
    positions=list(range(len(EPOCHS)))
    for column, group in enumerate(('old','new')):
        for branch in BRANCHES:
            for row_index, mode in enumerate(('conditional20','joint40')):
                ax = axes[row_index,column]
                means, errors = [], []
                for epoch in EPOCHS:
                    cells = [r['train_minus_heldout_accuracy_pp'] for r in gaps if r['stratum']=='formal' and
                             r['group']==group and r['heldout_split']=='test' and r['mode']==mode and
                             r['branch']==branch and r['epoch']==epoch]
                    means.append(statistics.mean(cells)); errors.append(statistics.stdev(cells))
                ax.errorbar(positions,means,yerr=errors,color=color[branch],marker='o',capsize=3,label=branch)
                ax.set_title(group.title()+' '+mode+' generalization gap')
                ax.set_ylabel('Train - test accuracy (pp)'); ax.axhline(0,color='gray',lw=.6)
        for row_index, error_kind in enumerate(('cross_group_error_rate','within_group_error_rate')):
            ax = axes[row_index,2]
            for branch in BRANCHES:
                means, errors = [], []
                for epoch in EPOCHS:
                    cells = [r[error_kind] for r in metrics if r['stratum']=='formal' and
                             r['dataset']=='test_'+group and r['branch']==branch and r['epoch']==epoch]
                    means.append(statistics.mean(cells)); errors.append(statistics.stdev(cells))
                ax.errorbar(positions,means,yerr=errors,color=color[branch],linestyle='-' if group=='old' else '--',
                            marker='o' if group=='old' else '^',capsize=3,label=branch+' '+group)
            ax.set_title(('Cross-group' if row_index==0 else 'Within-group')+' joint40 test errors')
            ax.set_ylabel('Error rate (%)')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.set_xticks(positions,labels=[str(e) for e in EPOCHS]);ax.legend(fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Fixed checkpoints (epoch; equally spaced categories)')
    fig.suptitle('Fixed checkpoints: conditional/joint generalization gaps and error partitions\n'
                 'Mean ± sample SD over 3 paired training seeds; epoch 0 is shared within each seed',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92))
    for path in paths:
        suffix = path.suffix
        fd,name=tempfile.mkstemp(prefix=path.stem+'.',suffix=suffix,dir=out);os.close(fd)
        try:
            fig.savefig(name,dpi=180,metadata={'CreationDate':None,'ModDate':None} if suffix=='.pdf' else None)
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)
    plt.close(fig)
    return dict(status='generated',input_sha256=content_digest,n=3,
                files={str(p):digest(p) for p in paths},
                units='accuracy gaps percentage points; error rates percent; error bars sample SD')


def fmt(s):
    if not s or s.get('n',0)==0:return '未获得'
    sd = 'N/A' if s['sample_sd'] is None else '%.4f'%s['sample_sd']
    return '%.4f ± %s（实际配对 n=%d）'%(s['mean'],sd,s['n'])


def analysis_markdown(summary, paired, originals, shared):
    counts = summary['completeness']['formal']
    text = ['# 三处缺口离线结果分析', '',
            '本文件由 CPU 汇总器重复生成；只读取已保存结果。输入台账和 SHA256 在 `SOURCE_AVAILABLE_LEDGER.json`；'
            '所有原始行、逐种子值和配对差值均另存 CSV/JSON。缺失与结构性 null 不填成零。', '',
            '|正式模块|已出现单元|有效执行单元|固定计划|科学可用单元|',
            '|---|---:|---:|---:|---:|']
    for kind in KINDS:
        c=counts[kind]
        text.append('|%s|%d|%d|%d|%d|'%(kind,c['available'],c['validated_executed'],c['expected'],c['scientifically_available']))
    text.extend(['', '队列状态：`%s`；任务状态计数：`%s`。有效执行不等于完整任务验收，exit=0也不等于机制门槛通过。'%(
        summary['queue_status'],json.dumps(summary['queue_counts'],ensure_ascii=False,sort_keys=True)),
        '三项 pilot 单独进入 `pilot` 原始表与摘要；不会补充正式 seed 数，也不与正式值混合。正式曲率36单元统一FP32、禁止FP64 fallback；'
        'pilot 的FP64是独立探索精度，不能追认原FP32 HVP或原三点网格。', '',
        '共享e0：'+ '；'.join('seed%d=%s'%(s,'已逐参数hash/NPZ全部数组核验一致' if shared[s]['verified'] else '尚未核验通过') for s in SEEDS)+'。'
        '正式评估计划30检查点对应27个不同状态，共同e0不构成额外独立样本。', '',
        '独立重复单位为训练seed。曲率/匹配的同一分支、epoch、量及精度下先把批次汇成每seed均值，再报告跨seed μ±样本SD；'
        '每层匹配分解先在cell内作描述均值，再汇seed，保留24层原值。n<2时样本SD为N/A；部分结果只作诊断，'
        '不形成正式性能排序。主要e25/e50的正式配对描述须恰有seed0/1/2且共同e0一致，n=3仍不作显著性包装。', '',
        '## 固定检查点的条件测试、竞争与保持', '',
        '每个e0/1/10/25/50分别评估old/new clean train、dev和official test。条件20类排除另一任务类别竞争，'
        '共同40类允许竞争；泛化差距定义为train accuracy−heldout accuracy（百分点），CE差距定义为heldout CE−train CE。'
        '这些是固定检查点证据，没有用test挑checkpoint，也不是严格相同拟合对比。', '',
        'R−S 的条件20测试提升若三seed一致，会增强“本任务辨别泛化改善”的描述；只在共同40提升而条件20不提升，'
        '更符合跨组竞争变化。两者可以同时发生；旧条件能力与旧共同准确率的分离帮助区分旧辨别能力和新类别竞争。'
        '须同时查看train拟合程度、dev、残余trainCE与阶段；仍不能据此确定因果机制。', '',
        '|主要epoch|数据|指标|配对R−S μ±样本SD|正式三seed/e0条件|',
        '|---|---|---|---|---|'])
    wanted = {'conditional20.accuracy','joint40.accuracy','conditional20.ce','joint40.ce',
              'ce_competition_penalty','cross_group_error_rate','within_group_error_rate'}
    selected=[r for r in paired if r['primary_endpoint'] and r['dataset'] in ('test_old','test_new') and r['metric'] in wanted]
    if not selected:text.append('|25/50|old/new test|全部|未获得|未完成|')
    for row in selected:
        text.append('|%d|%s|%s|%s|%s|'%(row['epoch'],row['dataset'],row['metric'],fmt(row),
                    '满足' if row['formal_paired_stat_available'] else '尚不满足'))
    endpoint = [r for r in paired if r['epoch']==50 and r['dataset']=='test_new']
    lookup = {r['metric']:r for r in endpoint}
    if lookup:
        text.extend(['', 'e50新类测试必须分量解释：conditional20 accuracy的R−S为%s，conditional20 CE为%s，'
                     'joint40 accuracy为%s，competition penalty为%s。'%(
                         fmt(lookup.get('conditional20.accuracy')),fmt(lookup.get('conditional20.ce')),
                         fmt(lookup.get('joint40.accuracy')),fmt(lookup.get('ce_competition_penalty'))),
                     '逐seed conditional20 accuracy差值为`%s`；不能把CE减少或共同40提升写成每seed条件正确率都增加。'%
                     json.dumps(dict(zip(lookup.get('conditional20.accuracy',{}).get('seeds',[]),
                                         lookup.get('conditional20.accuracy',{}).get('values',[]))),ensure_ascii=False)])
    faa=[r for r in paired if r['primary_endpoint'] and r['dataset']=='test_new' and r['metric']=='FAA']
    if faa:text.extend(['', '固定检查点FAA配对R−S：'+ '；'.join('e%d %s'%(r['epoch'],fmt(r)) for r in faa)+'。'])
    text.extend(['', '联合错误严格按跨组错误与组内错误分解，二者相加等于joint40错误。conditional20错误与跨组错误可重叠，'
        '不能直接相加。逐样本CE恒等式为joint40 CE=conditional20 CE+log(1+Z_other/Z_own)。'
        '配对R−S的joint CE差异也应等于conditional CE差异与competition penalty差异之和；'
        '汇总保留这些贡献、重叠错误、真实类概率、margin及逐样本数组。', '',
        '## Hessian数值与有限位移近似', '',
        '历史原网格失败结果另存 `HISTORICAL_CURVATURE_RAW.json` 和网格CSV；不改写、不丢弃。已读取历史执行单元%d。'%len(originals),
        '原网格(.001/.003/.01)与扩展探索网格(1e-5/3e-5/1e-4/3e-4/.001/.003/.01)分别判断：'
        'H g0与H detach(H g0)各需要两个相邻点relative error≤.08。所有relative error/cosine点保留；'
        '一阶backend一致不替代高阶门槛。正式原网格通过%d/%d、探索FP32通过%d/%d、允许候选修正比较%d/%d。'%(
            counts['curvature']['original_fp32_pass'],counts['curvature']['validated_executed'],
            counts['curvature']['exploratory_fp32_pass'],counts['curvature']['validated_executed'],
            counts['curvature']['scientifically_available'],counts['curvature']['validated_executed']),
        '只有独立HVP收敛后，才解释原生FP32的g2−g0与冻结Hessian二次修正比较。共同尺度1与.5同时缩小下降距离和外半径；'
        '报告cosine、relative error与两修正范数，半尺度误差缩小可以支持有限位移偏差的解释，不设置事后通过阈值。'
        '曲率数值门槛仍可能失败：此时局部对应关系不可判定；这不会否定已验收的实际FAA结果。', '',
        'relative error口径：FD误差分母为exact HVP范数，候选修正误差分母为Delta_quad范数，'
        '有效方向native/matched误差分母为SAM有限有效位移范数；实现均加1e-12。'
        '传入的reference_norm仅决定退化阈值，不是relative error分母。因此修正relative error不能解读为相对g0或Delta_native误差。', '',
        '## 有效步幅与分解边界', '',
        '匹配独立于HVP门槛，以同一状态SAM有限(全部BA,全部FC)位移范数为目标，求g2最小正四次根。'
        '保存raw与matched的全空间/LoRA/FC范数、cosine、relative error、匹配LR倍率；退化与无正根是不可用，不是零误差。'
        '有限几何用FP64从原生FP32参数/梯度值计算，既不更换native梯度也不提交更新。瞬时范数匹配不能排除长期学习率、'
        '训练步幅与沿轨迹适应效应。',
        'ΔQ一阶/二阶、πA、TA/TB/Tcross和分组件范数是有效状态分配的描述量，不是因果贡献；'
        '范数不可直接作为相加贡献份额，FC/LoRA方向差异也不能替代组件交换等因果检验。', '',
        '图状态：`%s`。仅全部30正式评估有效且三seed共同e0一致后，生成独立PDF/PNG（old/new条件/共同泛化差距与跨/组内错误）；'
        '当前缺失不会画成零。'%summary['figure']['status'], '',
        '后续问题仅在结果提供依据时提出：如需排除长期步幅效应或纯分类头解释，应另立固定协议；本汇总器不新增训练、'
        '不扫描学习率/半径、不按机制近似表现选择backend。', ''])
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=HERE/'suite.json')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results/offline_analysis')
    args = parser.parse_args()
    out=args.output_dir.resolve()
    require(out == (HERE/'results/offline_analysis').resolve() or
            (HERE/'results/offline_analysis').resolve() in out.parents,
            'Collector output must stay under results/offline_analysis')
    out.mkdir(parents=True,exist_ok=True)
    reader=Reader();suite=reader.json(args.suite)
    require(suite is not None, 'Suite unavailable')
    queue=reader.json(HERE/'runtime/queue_state.json') or {}
    pins=reader.json(HERE/'implementation_manifest.json') or {}
    reader.pinned={str(Path(p).resolve()):h for p,h in pins.get('files',{}).items()}
    require(suite['submitted_training_updates']==0, 'Suite training update')
    require({j['kind'] for j in suite['jobs']}==set(KINDS), 'Suite kind scope differs')
    for kind in KINDS:
        jobs=[j for j in suite['jobs'] if j['kind']==kind and not j['pilot']]
        require(sorted(j['seed'] for j in jobs)==list(SEEDS), 'Formal seed cohort differs')
        require(sum(j['expected_items'] for j in jobs)==PLANNED[kind], 'Fixed formal count differs')
    require(len([j for j in suite['jobs'] if j['pilot']])==3, 'Three separate pilots required')
    accepted=[];raw={};cells=[];jobs=[]
    for job in suite['jobs']:
        a,r,c,j=load_job(job,reader,queue)
        accepted.extend(a);raw[job['id']]=r;cells.extend(c);jobs.append(j)
    by_kind={kind:[i for i in accepted if i['kind']==kind] for kind in KINDS}
    curvature,grid,corrections=curvature_tables(by_kind['curvature'])
    half_changes=half_scale_changes(corrections)
    matching,layers=matching_tables(by_kind['matching'])
    evaluation,gaps=evaluation_tables(by_kind['evaluation'])
    shared=verify_shared_start(by_kind['evaluation'],reader)
    paired,paired_summary,gap_pairs=paired_evaluation(evaluation,gaps,shared)
    original,old_grid,old_vs_new=original_curvature(reader,by_kind['curvature'])
    cs,csummary=summarize_cells(grid,('stratum','branch','epoch','precision','product','grid','distance'),
                               ('relative_error','cosine','a_norm','b_norm'))
    ccs,ccsummary=summarize_cells(corrections,('stratum','branch','epoch','scale','curvature_precision','candidate_precision'),
                                ('relative_error','cosine','delta_native_norm','delta_quadratic_norm'))
    hkeys=sorted({k for r in half_changes for k in r if k.endswith('_half_minus_full') or k.endswith('_half_over_full')})
    hs,hsummary=summarize_cells(half_changes,('stratum','branch','epoch','curvature_precision','candidate_precision'),hkeys)
    mm=sorted({k for r in matching for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}-
              {'seed','epoch','batch_index'})
    ms,msummary=summarize_cells(matching,('stratum','branch','epoch'),mm)
    es,esummary=summarize_cells(evaluation,('stratum','branch','epoch','dataset'),
                             ('conditional20.accuracy','conditional20.ce','joint40.accuracy','joint40.ce',
                              'ce_competition_penalty','cross_group_error_rate','within_group_error_rate'))
    gs,gsummary=summarize_cells(gaps,('stratum','branch','epoch','group','heldout_split','mode'),
                              ('train_minus_heldout_accuracy_pp','heldout_minus_train_ce'))
    completeness={}
    for stratum in ('formal','pilot'):
        completeness[stratum]={}
        for kind in KINDS:
            cc=[c for c in cells if c['stratum']==stratum and c['kind']==kind]
            vv=[c for c in cc if c['validated']]
            scientific=sum(c.get('candidate_comparison_permitted',False) if kind=='curvature' else
                           c.get('accepted_match',False) if kind=='matching' else True for c in vv)
            completeness[stratum][kind]=dict(expected=sum(c['expected'] for c in cc),available=sum(c['available'] for c in cc),
                validated_executed=len(vv),missing=sum(not c['available'] and c['expected'] for c in cc),
                invalid=sum(c['available'] and not c['validated'] for c in cc),scientifically_available=scientific,
                original_fp32_pass=sum(c.get('original_fp32_gate',False) for c in vv),
                exploratory_fp32_pass=sum(c.get('exploratory_fp32_gate',False) for c in vv),
                exploratory_fp64_pass=sum(c.get('exploratory_fp64_gate',False) for c in vv))
    eval_complete=(completeness['formal']['evaluation']['validated_executed']==30 and
                   completeness['formal']['evaluation']['invalid']==0 and all(x['verified'] for x in shared.values()))
    content_digest=hashlib.sha256(json.dumps([evaluation,gaps,shared,digest(__file__)],sort_keys=True).encode()).hexdigest()
    figure=plot_evaluation(out,evaluation,gaps,content_digest) if eval_complete else dict(
        status='pending_all_30_valid_formal_evaluations_and_shared_e0',available_valid_checkpoints=completeness['formal']['evaluation']['validated_executed'])
    write_json(out/'EVALUATION_FIGURE.json',figure)
    # Hash source scripts, accepted kernels, plan and upstream descriptive analysis.
    source_pin_checks=[]
    for path in [Path(__file__),HERE/'curvature_diagnostics.py',HERE/'effective_match.py',HERE/'checkpoint_eval.py',
                 HERE/'EXPERIMENT_PLAN.md',HERE/'implementation_manifest.json',OLD/'source_manifest.json',
                 OLD/'diagnostics/offline_core.py',OLD/'trainer/effective_step.py',
                 HERE/'results/state_analysis/STATE_FORMATION_ANALYSIS.json',HERE/'results/state_analysis/RESULT_ANALYSIS.md']:
        entry=reader.binary(path)
        expected_sha=reader.pinned.get(str(path.resolve()))
        if expected_sha is not None:
            source_pin_checks.append(dict(path=str(path),expected_sha256=expected_sha,
                                          observed_sha256=entry['sha256'],unchanged=expected_sha==entry['sha256']))
    source_entries=sorted(reader.ledger.values(),key=lambda r:r['path'])
    input_sha=hashlib.sha256(json.dumps(source_entries,sort_keys=True).encode()).hexdigest()
    queue_counts=dict(collections.Counter(r.get('status','unknown') for r in queue.get('jobs',{}).values()))
    summary=dict(protocol=PROTOCOL,campaign=str(OLD),source_run=str(HERE),cpu_only=True,submitted_training_updates=0,
        input_snapshot_sha256=input_sha,collector_sha256=digest(__file__),queue_status=queue.get('status','not_launched'),
        queue_counts=queue_counts,queue_reported_counts=queue.get('counts'),completeness=completeness,
        planned_distinct_eval_states=27,verified_distinct_eval_states=len({(i['row']['seed'],i['row']['full_model_parameter_sha256'])
            for i in by_kind['evaluation'] if i['stratum']=='formal'}),shared_e0=shared,
        frozen_source_pin_checks=source_pin_checks,
        all_checked_frozen_sources_unchanged=all(r['unchanged'] for r in source_pin_checks),
        formal_jobs_execution_complete=sum(j['execution_complete'] for j in jobs if not j['pilot']),
        formal_jobs_planned=9,pilot_jobs_execution_complete=sum(j['execution_complete'] for j in jobs if j['pilot']),
        figure=figure,statistics=dict(unit='training seed',sample_sd_ddof=1,paired_contrast='R minus S',
            primary_epochs=[25,50],missing_values='excluded, never imputed as zero',pilot_pooling=False),
        scientific_claim_status='complete_descriptive_evaluation_available' if eval_complete else 'partial_diagnostic_only',
        curvature_grid_summary=csummary,curvature_correction_summary=ccsummary,curvature_half_scale_summary=hsummary,
        matching_summary=msummary,
        evaluation_summary=esummary,gap_summary=gsummary,paired_evaluation_summary=paired_summary)
    write_json(out/'OFFLINE_ANALYSIS.json',summary)
    write_json(out/'RAW_RESULTS.json',raw)
    write_json(out/'PRIMARY_PAIRED_STATS.json',dict(required_seeds=list(SEEDS),epochs=[25,50],
               rows=[r for r in paired_summary if r['primary_endpoint']],
               conditional_accuracy_is_distinct_from_conditional_CE_and_joint_accuracy=True,
               fixed_checkpoints_not_fit_matched=True))
    write_json(out/'HISTORICAL_CURVATURE_RAW.json',original)
    write_json(out/'SOURCE_AVAILABLE_LEDGER.json',dict(input_snapshot_sha256=input_sha,inputs=source_entries))
    for name,rows in dict(CELL_LEDGER=cells,JOB_LEDGER=jobs,CURVATURE_CELLS=curvature,CURVATURE_GRID=grid,
        CURVATURE_CORRECTIONS=corrections,CURVATURE_GRID_SEED=cs,CURVATURE_GRID_SUMMARY=csummary,
        CURVATURE_CORRECTION_SEED=ccs,CURVATURE_CORRECTION_SUMMARY=ccsummary,
        CURVATURE_HALF_SCALE_CHANGE=half_changes,CURVATURE_HALF_SCALE_SEED=hs,CURVATURE_HALF_SCALE_SUMMARY=hsummary,
        HISTORICAL_CURVATURE_GRID=old_grid,HISTORICAL_VS_NEW_CURVATURE=old_vs_new,
        MATCHING_CELLS=matching,MATCHING_LAYERS=layers,MATCHING_SEED=ms,MATCHING_SUMMARY=msummary,
        EVALUATION_DATASETS=evaluation,EVALUATION_SEED=es,EVALUATION_SUMMARY=esummary,
        GENERALIZATION_GAPS=gaps,GENERALIZATION_GAP_SEED=gs,GENERALIZATION_GAP_SUMMARY=gsummary,
        PAIRED_EVALUATION=paired,PAIRED_EVALUATION_SUMMARY=paired_summary,PAIRED_GENERALIZATION_GAPS=gap_pairs).items():
        write_csv(out/(name+'.csv'),rows)
    atomic_text(out/'RESULT_ANALYSIS.md',analysis_markdown(summary,paired_summary,original,shared))
    artifacts={str(p):digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACTS.json'}
    write_json(out/'ARTIFACTS.json',dict(collector_sha256=digest(__file__),input_snapshot_sha256=input_sha,files=artifacts))
    print(json.dumps(dict(protocol=PROTOCOL,output_dir=str(out),completeness=completeness,
                         queue_counts=queue_counts,figure_status=figure['status'],
                         cpu_only=True,submitted_training_updates=0),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
