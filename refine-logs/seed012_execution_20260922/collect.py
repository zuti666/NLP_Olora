"""Collect verified completed formal runs; never infer uncertainty from one run.

No experiment-code imports or training. Each output is replaced atomically.
"""
from __future__ import annotations
import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
METHODS = ['Vanilla LoRA', 'InfLoRA', 'OLoRA', 'SDLoRA', 'CL-LoRA', 'EWCLoRA', 'GAM-EWCLoRA']
DATASETS = ['aircraft', 'cars196', 'cub200', 'flowers', 'het5']
LABELS = dict(zip(DATASETS, ['Aircraft', 'Cars', 'CUB-200', 'Flowers', '5Datasets']))
TASKS = {'aircraft': 10, 'cars196': 10, 'cub200': 10, 'flowers': 10, 'het5': 5}
SEEDS = [0, 1, 2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, expected_sha=None):
    path = Path(path)
    data = path.read_bytes()
    actual = digest(data)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f'SHA256 mismatch: {path}; expected={expected_sha}, actual={actual}')
    return json.loads(data), actual


def canonical_seed(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f'Invalid scalar seed: {value!r}')
    result = int(value)
    if result not in SEEDS:
        raise ValueError(f'Seed outside fixed target 0/1/2: {value!r}')
    return result


def identity(record):
    method, dataset, seed = record['method'], record['dataset'], canonical_seed(record['seed'])
    if method not in METHODS or dataset not in DATASETS:
        raise ValueError(f'Unknown target method/dataset: {method!r}, {dataset!r}')
    return method, dataset, seed


def matrix_metrics(raw, expected_tasks):
    """Use only explicitly CNN task matrices, never pooled top1 or legacy AAA."""
    candidates = []
    paths = [('cnn', 'matrices', 'final'), ('cnn_task_accuracies',),
             ('cnn_task_accuracies', 'final'), ('cnn', 'task_accuracies')]
    for path in paths:
        node = raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, list):
            candidates.append(('.'.join(path), node))
    if not candidates:
        raise ValueError('No final CNN task-accuracy matrix in a supported schema')
    normalized = []
    for schema, matrix in candidates:
        n = int(expected_tasks)
        if len(matrix) != n:
            raise ValueError(f'{schema}: expected {n} tasks, received {len(matrix)}')
        lower = []
        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) not in (i + 1, n):
                raise ValueError(f'{schema}: invalid row {i} shape; require triangular or square matrix')
            observed = row[:i + 1]
            if any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 <= x <= 100 for x in observed):
                raise ValueError(f'{schema}: nonfinite/nonpercentage observed value in row {i}')
            lower.append([float(x) for x in observed])
        normalized.append((schema, lower))
    if any(rows != normalized[0][1] for _, rows in normalized[1:]):
        raise ValueError('Conflicting supported CNN task matrices in one metrics JSON')
    schema, lower = normalized[0]
    stages = [statistics.mean(row) for row in lower]
    return {'tasks': len(lower), 'FAA': stages[-1], 'AAA_stage': statistics.mean(stages),
            'stage_means': stages, 'lower_triangle': lower, 'schema': schema}


def validate_candidate(candidate, root):
    method, dataset, seed = identity(candidate)
    expected = candidate.get('expected_tasks', TASKS[dataset])
    if isinstance(expected, bool) or int(expected) != TASKS[dataset]:
        raise ValueError(f'Unexpected task count for {dataset}: {expected}')
    path = candidate['metric_path']
    if not candidate.get('metric_sha256'):
        raise ValueError('Missing required metric_sha256')
    metric_path = Path(path)
    if not metric_path.is_absolute():
        metric_path = root / metric_path
    raw, metric_sha = read_json(metric_path, candidate['metric_sha256'])
    metrics = matrix_metrics(raw, expected)
    cfg_sha = None
    if candidate.get('config'):
        if not candidate.get('config_sha256'):
            raise ValueError('Configuration path provided without config_sha256')
        config_path = Path(candidate['config'])
        if not config_path.is_absolute():
            config_path = root / config_path
        cfg, cfg_sha = read_json(config_path, candidate['config_sha256'])
        cfg_seed = cfg.get('seed')
        if cfg_seed is not None and cfg_seed not in (seed, [seed]):
            raise ValueError(f'Config seed differs from run identity: {cfg_seed!r} versus {seed}')
        if cfg.get('dataset', dataset) != dataset:
            raise ValueError('Config dataset differs from run identity')
    declared = candidate.get('metrics', {})
    for label, aliases in [('FAA', ['FAA', 'faa']), ('AAA_stage', ['AAA_stage', 'aaa_stage'])]:
        for alias in aliases:
            if alias in declared:
                value = declared[alias]
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or abs(value - metrics[label]) > 1e-8):
                    raise ValueError(f'Receipt {alias} differs from recomputed {label}')
    return {'method': method, 'dataset': dataset, 'seed': seed,
            'origin': candidate['_origin'], 'id': candidate.get('id'),
            'receipt': candidate.get('_receipt'), 'receipt_sha256': candidate.get('_receipt_sha256'),
            'metric_path': str(metric_path.resolve()), 'metric_sha256': metric_sha,
            'config': candidate.get('config'), 'config_sha256': cfg_sha,
            'protocol_sha256': candidate.get('protocol_sha256'),
            'source_root': candidate.get('source_root'), **metrics}


def reject(excluded, record, reason):
    excluded.append({key: record.get(key) for key in
                     ['id', 'method', 'dataset', 'seed', 'metric_path']}
                    | {'origin': record.get('origin', record.get('_origin')),
                       'receipt': record.get('receipt', record.get('_receipt')),
                       'reason': str(reason)})


def summarize(runs):
    groups = []
    for method in METHODS:
        for dataset in DATASETS:
            selected = sorted((r for r in runs if r['method'] == method and r['dataset'] == dataset),
                              key=lambda r: r['seed'])
            seeds = [r['seed'] for r in selected]
            if len(seeds) != len(set(seeds)):
                raise ValueError(f'Internal duplicate seeds for {method}/{dataset}')
            n = len(selected)
            row = {'method': method, 'dataset': dataset, 'n': n, 'seeds': seeds,
                   'missing_seeds': [s for s in SEEDS if s not in seeds],
                   'status': 'complete_n3' if n == 3 else ('partial' if n else 'unavailable')}
            for metric in ['FAA', 'AAA_stage']:
                values = [r[metric] for r in selected]
                row[metric + '_mean'] = statistics.mean(values) if n else None
                row[metric + '_sample_sd'] = statistics.stdev(values) if n >= 2 else None
                row[metric + '_sample_variance'] = statistics.variance(values) if n >= 2 else None
            groups.append(row)
    return groups


def atomic_write(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def collect(root):
    suite_path = root / 'suite.json'
    if not suite_path.is_file():
        print(f'Collection postponed: {suite_path} is not available; no outputs changed.')
        return 0
    suite, suite_sha = read_json(suite_path)
    jobs, reused = suite.get('jobs', []), suite.get('reused', [])
    if not isinstance(jobs, list) or not isinstance(reused, list):
        raise ValueError('suite.jobs and suite.reused must be lists')
    excluded, pending, candidates = [], [], []
    counts = collections.Counter(j.get('id') for j in jobs)
    for job in jobs:
        if job.get('stage') != 'formal':
            continue
        job = dict(job, _origin='new_formal')
        try:
            identity(job)
            job_id = job['id']
            if not isinstance(job_id, str) or Path(job_id).name != job_id or job_id in ('.', '..'):
                raise ValueError('Invalid job ID')
            if counts[job_id] != 1:
                raise ValueError('Duplicate job ID in suite')
            path = root / 'receipts' / (job_id + '.json')
            if not path.is_file():
                pending.append({'id': job_id, 'method': job['method'], 'dataset': job['dataset'],
                                'seed': job['seed'], 'status': 'pending_no_receipt'})
                continue
            receipt, receipt_sha = read_json(path)
            if receipt.get('status') != 'completed':
                pending.append({'id': job_id, 'method': job['method'], 'dataset': job['dataset'],
                                'seed': job['seed'], 'status': receipt.get('status', 'unknown'),
                                'error': receipt.get('error')})
                continue
            for key in ['id', 'method', 'dataset', 'seed', 'stage', 'expected_tasks', 'config', 'config_sha256']:
                if key not in job or receipt.get(key) != job[key]:
                    raise ValueError(f'Completed receipt differs from suite field {key}')
            if receipt.get('exit_code', 0) != 0:
                raise ValueError('Completed receipt has nonzero exit code')
            candidates.append(dict(job, **{k: v for k, v in receipt.items() if k not in job},
                                   _receipt=str(path), _receipt_sha256=receipt_sha))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reject(excluded, job, exc)
    for i, record in enumerate(reused):
        candidate = dict(record, _origin='reused', id=record.get('id', f'reused_{i}'))
        if record.get('stage', 'formal') != 'formal':
            reject(excluded, candidate, 'A reused entry is not formal')
            continue
        if record.get('status', 'completed') != 'completed':
            reject(excluded, candidate, 'A reused entry is not completed')
            continue
        if record.get('receipt'):
            try:
                archived, receipt_sha = read_json(record['receipt'], record.get('receipt_sha256'))
                if archived.get('status') != 'completed' or archived.get('exit_code', 0) != 0:
                    raise ValueError('Reused archived receipt is not successful')
                for key in ['id', 'method', 'dataset', 'seed', 'metric_path', 'metric_sha256', 'config', 'config_sha256']:
                    value = archived.get(key)
                    if key == 'method' and value == 'GAM-EWC':
                        value = 'GAM-EWCLoRA'  # Same learner, archived display label.
                    if value != candidate.get(key):
                        raise ValueError(f'Reused receipt differs from suite field {key}')
                candidate['_receipt'] = record['receipt']
                candidate['_receipt_sha256'] = receipt_sha
            except (OSError, ValueError, TypeError, KeyError) as exc:
                reject(excluded, candidate, exc)
                continue
        candidates.append(candidate)
    valid = []
    for candidate in candidates:
        try:
            valid.append(validate_candidate(candidate, root))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            reject(excluded, candidate, exc)
    # Ambiguities are excluded wholesale: no first/last/best-seed selection rule.
    by_identity = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for index, row in enumerate(valid):
        by_identity[(row['method'], row['dataset'], row['seed'])].append(index)
        by_source[row['metric_path']].append(index)
    blocked = collections.defaultdict(list)
    for ids in by_identity.values():
        if len(ids) > 1:
            for i in ids:
                blocked[i].append('Duplicate completed candidates for method/dataset/seed')
    for ids in by_source.values():
        if len(ids) > 1:
            for i in ids:
                blocked[i].append('Metrics file reused for multiple candidates')
    by_group = collections.defaultdict(list)
    for i, row in enumerate(valid):
        by_group[(row['method'], row['dataset'])].append(i)
    for ids in by_group.values():
        fingerprints = {valid[i]['protocol_sha256'] for i in ids if valid[i]['protocol_sha256']}
        if len(fingerprints) > 1:
            for i in ids:
                blocked[i].append('Conflicting explicit nonseed protocol fingerprints')
    runs = []
    for i, row in enumerate(valid):
        if i in blocked:
            reject(excluded, row, '; '.join(blocked[i]))
        else:
            runs.append(row)
    runs.sort(key=lambda r: (METHODS.index(r['method']), DATASETS.index(r['dataset']), r['seed']))
    groups = summarize(runs)
    now = datetime.now(timezone.utc).isoformat()
    result = {'schema_version': 1, 'generated_at_utc': now, 'suite_path': str(suite_path),
              'suite_sha256': suite_sha, 'target_seeds': SEEDS,
              'statistics': 'FAA=final task-macro mean; AAA_stage=mean of stage task-macro means; sample SD/variance use ddof=1. n<2 has null SD/variance.',
              'provenance_scope': 'Suite-approved cohorts only. Config and metrics bytes verified here; full code/data/environment equivalence is established by suite preparation, not inferred from accuracy.',
              'counts': {'accepted_runs': len(runs), 'new_formal': sum(r['origin'] == 'new_formal' for r in runs),
                         'reused': sum(r['origin'] == 'reused' for r in runs), 'complete_groups_n3': sum(r['n'] == 3 for r in groups),
                         'groups_total': len(groups), 'excluded': len(excluded), 'not_completed': len(pending)},
              'runs': runs, 'groups': groups, 'excluded': excluded, 'not_completed': pending}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(groups[0]))
    writer.writeheader()
    for row in groups:
        writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    lines = ['# Seed 0/1/2 实验结果', '', f'更新时间（UTC）：{now}', '',
             f'通过验证 {len(runs)}/105 次；完整三seed组 {result["counts"]["complete_groups_n3"]}/35；排除 {len(excluded)} 条；未完成正式任务 {len(pending)} 条。', '',
             'FAA / AAA_stage 均由完整CNN任务矩阵重算。每项显示 mean (sample SD)，单位为百分点；n=1时SD不可估计，n<3为部分结果。Smoke不计入。', '',
             '| Method | Dataset | n | Seeds | FAA mean (SD) | AAA mean (SD) | Missing |',
             '|---|---|---:|---|---:|---:|---|']
    def display(row, metric):
        value, sd = row[metric + '_mean'], row[metric + '_sample_sd']
        return '--' if value is None else f'{value:.2f} ({sd:.2f})' if sd is not None else f'{value:.2f} (SD unavailable)'
    for row in groups:
        lines.append(f'| {row["method"]} | {LABELS[row["dataset"]]} | {row["n"]} | {row["seeds"]} | {display(row, "FAA")} | {display(row, "AAA_stage")} | {row["missing_seeds"]} |')
    if excluded:
        lines += ['', '## 排除项', '']
        for row in excluded:
            lines.append(f'- {row.get("id")}: {row["reason"]}')
    lines += ['', '完整逐run来源、哈希、矩阵、未完成状态与排除原因见 `results.json`。`results.csv`为35组统计。不会把旧最高值与新SD拼接，也不从任务/epoch间波动估计跨seed标准差。', '']
    atomic_write(root / 'results.json', json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    atomic_write(root / 'results.csv', buffer.getvalue())
    atomic_write(root / 'SUMMARY.md', '\n'.join(lines))
    print(json.dumps(result['counts'], ensure_ascii=False))
    if excluded:
        print(f'Validation exclusions recorded in {root / "results.json"}', file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Directory containing suite.json and receipts/')
    args = parser.parse_args(argv)
    try:
        return collect(args.root.resolve())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'COLLECTOR ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
