#!/usr/bin/env python3
"""Validate and report the formal three-seed RQ2 cohort; CPU-only, no training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
BRANCHES = {'A': ('sgd', 0), 'B': ('gam', 0), 'C': ('sgd', 2000), 'D': ('gam', 2000)}
DIAGNOSTICS = ('dense', 'sparse_train', 'sparse_test', 'sparse_train_full',
               'trajectory_fisher', 'path_t01', 'path_t05', 'path_t09', 'bridge_full')
SCIENTIFIC_FIELDS = ('dataset', 'class_shuffle', 'init_cls', 'increment', 'batch_size', 'epochs',
                     'lrate', 'momentum', 'weight_decay', 'scheduler', 'ewc_gamma', 'ewc_eta',
                     'ewc_max_batches', 'fisher_mode', 'ewc_fisher_source', 'fisher_split',
                     'ewc_normalize_fisher', 'ewc_normalize_accumulated_fisher',
                     'ewc_dual_ascent', 'ewc_project_grad', 'approximate_fisher_in_task',
                     'weight_merge', 'in_task_weight_merge', 'endpoint_weight_merge',
                     'delta_w_penalty', 'train_num_workers', 'persistent_workers')
METRICS = ('a11', 'a21', 'a22', 'BWT', 'average_accuracy', 'raw_trace_anchor_train256',
           'raw_trace_endpoint_train256', 'drift_l2', 'drift_squared', 'D0_train', 'D0_test')


def fail(message):
    raise ValueError(message)


def require(condition, message):
    if not condition:
        fail(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=None)
def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: fail(f'Nonfinite JSON: {path}: {value}'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def json_lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line, parse_constant=lambda value: fail(f'Nonfinite JSONL: {path}: {value}'))


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value, label):
    value = float(value)
    require(math.isfinite(value), f'Nonfinite {label}')
    return value


def within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def mean_sd(values):
    require(len(values) == 3, 'Formal aggregate requires exactly three training seeds')
    return {'mean': statistics.mean(values), 'sample_sd_ddof1': statistics.stdev(values), 'n': 3}


def validate_endpoint(path, root, seed, branch):
    result = read_json(path)
    method, lam = BRANCHES[branch]
    require(result.get('status') == 'complete', f'Endpoint incomplete: {path}')
    require(result.get('seed') == seed and result.get('method') == method, f'Endpoint seed/method mismatch: {path}')
    require(float(result.get('ewc_lambda', -1)) == lam, f'Endpoint protection mismatch: {path}')
    require(result.get('task0_shaping') is (method == 'gam'), f'Anchor shaping mismatch: {path}')
    require(result.get('future_optimizer') == 'sgd' and result.get('shaping_in_task1') is False,
            f'Future optimizer must be SGD with shaping off: {path}')
    require(result.get('old_task_head_exactly_unchanged') is True and result.get('nonlora_backbone_exactly_unchanged') is True,
            f'Endpoint restoration witness missing: {path}')
    manifest_path = Path(result['training_manifest'])
    require(within(manifest_path, root / 'formal_training'), f'Historical/pilot training is excluded: {manifest_path}')
    manifest = read_json(manifest_path)
    config = manifest.get('requested_config', manifest['config'])
    require(manifest['seed'] == seed and manifest['anchor_method'] == method, f'Training provenance mismatch: {path}')
    require(manifest['config']['optimizer_type'] == 'sgd' and float(manifest['config']['lambda_flat']) == 0,
            f'Training Task1 is not plain SGD: {path}')
    require(float(config['ewc_lambda']) == lam and config['epochs'] == 50 and config['batch_size'] == 128,
            f'Training protocol mismatch: {path}')
    require(manifest.get('matmul_allow_tf32') is False and manifest.get('cudnn_allow_tf32') is False,
            f'Full-FP32 training receipt missing: {path}')
    signature = result['signature']
    require(canonical_hash(signature) == result['signature_sha256'], f'Endpoint signature corrupt: {path}')
    require(sha_file(manifest_path) == signature['training_manifest_sha256'], f'Training manifest changed: {path}')
    require(sha_file(Path(result['metrics_json'])) == signature['metrics_sha256'], f'Training metrics changed: {path}')
    complete_path = manifest_path.parent / 'COMPLETE.json'
    require(read_json(complete_path).get('status') == 'complete', f'Training incomplete: {complete_path}')
    require(sha_file(complete_path) == signature['training_complete_sha256'], f'Training completion changed: {path}')
    for task, checkpoint in result['checkpoints'].items():
        require(within(checkpoint, root / 'formal_training'), f'Historical checkpoint excluded: {checkpoint}')
        require(sha_file(Path(checkpoint)) == signature['checkpoint_sha256'][task], f'Checkpoint changed: {checkpoint}')
    for point in ('anchor', 'endpoint'):
        fish = result[point]['fisher']
        require(fish['samples'] == 256 and fish['normalized'] is False and fish['floor'] is None,
                f'Fisher scope changed: {path}/{point}')
        require(numeric(fish['raw_trace'], f'{point} raw trace') > 0, f'Invalid trace: {path}')
    acc = result['accuracy_CIL']
    row = {'seed': seed, 'branch': branch, 'anchor_method': method, 'task1_optimizer': 'sgd', 'ewc_lambda': lam}
    row.update({key: numeric(acc[key], key) for key in ('a11', 'a21', 'a22')})
    require(all(0 <= row[k] <= 100 for k in ('a11', 'a21', 'a22')), f'Accuracy range invalid: {path}')
    row.update(BWT=row['a21'] - row['a11'], average_accuracy=(row['a21'] + row['a22']) / 2,
               raw_trace_anchor_train256=numeric(result['anchor']['fisher']['raw_trace'], 'anchor trace'),
               raw_trace_endpoint_train256=numeric(result['endpoint']['fisher']['raw_trace'], 'endpoint trace'),
               drift_l2=numeric(result['effective_weight_drift'], 'drift'))
    require(row['drift_l2'] >= 0, f'Negative drift: {path}')
    row['drift_squared'] = row['drift_l2'] ** 2
    for split in ('train', 'test'):
        base = numeric(result['anchor']['metrics'][split]['loss'], f'base {split} loss')
        final = numeric(result['endpoint']['metrics'][split]['loss'], f'final {split} loss')
        damage = numeric(result['D0'][split], f'{split} damage')
        require(abs(final - base - damage) < 1e-10, f'Damage mismatch: {path}/{split}')
        row.update({f'base_loss_{split}': base, f'final_loss_{split}': final, f'D0_{split}': damage})
    row.update(summary=str(path), summary_sha256=sha_file(path), training_manifest=str(manifest_path),
               rng_receipts=str(manifest_path.parent / 'rng_receipts.jsonl'),
               anchor_checkpoint_sha256=signature['checkpoint_sha256']['0'])
    identity = {'rng_protocol': manifest['protocol'], 'source_manifest_sha256': manifest['source_manifest_sha256'],
                'runner_sha256': manifest['runner_sha256'], 'torch': manifest['torch'],
                'torchvision': manifest['torchvision'], 'timm': manifest['timm'],
                'scientific_task1_config': {key: config.get(key) for key in SCIENTIFIC_FIELDS},
                'diagnostic_code_sha256': signature['code_sha256'], 'common_code_sha256': signature['common_code_sha256']}
    return row, identity


def check_rng(seed_rows):
    schemas = {
        'task_rng_reset': (('task',), ('task_seed', 'rng_protocol', 'optimizer_type')),
        'head_initialized': (('task',), ('old_classes', 'total_classes', 'task_seed', 'new_head_sha256')),
        'loader_created': (('task', 'loader_serial'), ('shuffle', 'generator_seed', 'num_workers', 'sample_count', 'dataset_ids_sha256')),
        'loader_iteration': (('task', 'loader_serial', 'iteration'), ('shuffle', 'generator_seed', 'num_workers', 'sample_count',
                             'dataset_ids_sha256', 'batches', 'samples', 'indices_sha256', 'first_batch_pixels_targets_sha256')),
    }
    require(set(seed_rows) == set(BRANCHES), 'Four branches are required for RNG pairing')
    index, evidence = {}, {}
    for branch, row in seed_rows.items():
        path = Path(row['rng_receipts'])
        events = list(json_lines(path))
        index[branch] = {}
        for event, (key_fields, value_fields) in schemas.items():
            selected = [r for r in events if r.get('task') == 1 and r.get('event') == event and
                        (event not in ('loader_created', 'loader_iteration') or r.get('shuffle') is True)]
            require(selected, f'Missing {branch} Task1 {event}')
            keyed = {}
            for entry in selected:
                key = tuple(entry.get(k) for k in key_fields)
                require(key not in keyed, f'Duplicate Task1 RNG receipt {branch}/{event}/{key}; inspect resume history')
                require(all(k in entry and entry[k] is not None for k in value_fields), f'Incomplete RNG receipt {branch}/{event}/{key}')
                keyed[key] = {k: entry[k] for k in value_fields}
                if event == 'loader_iteration':
                    require(entry['samples'] == entry['sample_count'], f'Incomplete loader iteration {branch}/{key}')
            index[branch][event] = keyed
        iterations = index[branch]['loader_iteration']
        serials = {key[1] for key in iterations}
        require(any(set(range(50)).issubset({k[2] for k in iterations if k[1] == serial}) for serial in serials),
                f'{branch} lacks 50 complete Task1 epochs')
        evidence[branch] = {'path': str(path), 'sha256': sha_file(path),
                            'event_counts': {event: len(entries) for event, entries in index[branch].items()}}
    for branch in ('B', 'C', 'D'):
        for event in schemas:
            require(index[branch][event] == index['A'][event], f'Task1 RNG mismatch: {branch} vs A / {event}')
    for plain, protected in (('A', 'C'), ('B', 'D')):
        require(seed_rows[plain]['anchor_checkpoint_sha256'] == seed_rows[protected]['anchor_checkpoint_sha256'],
                f'Protection pair starts from different anchor: {plain}/{protected}')
    return {'status': 'complete', 'passed': True, 'artifacts': evidence,
            'scope': 'Task1 new head, RNG reset, all shuffled sample orders, and each iteration first-batch pixels/labels; evaluation loaders excluded. This does not claim every augmented batch was individually hashed.'}


def validate_diagnostic(path, seed, method, job):
    result = read_json(path)
    require(result.get('status') == 'complete', f'Diagnostic incomplete: {path}')
    require(result.get('seed') == seed and result.get('method') == method, f'Diagnostic identity mismatch: {path}')
    manifest_path = path.with_name('manifest.json')
    manifest = read_json(manifest_path)
    protocol = manifest.pop('protocol_sha256')
    require(canonical_hash(manifest) == protocol and result['protocol_sha256'] == protocol,
            f'Diagnostic signature mismatch: {path}')
    require(result.get('end_restoration', {}).get('predictions_match') is True, f'Restoration witness missing: {path}')
    ids = set()
    journal = path.with_name('states.jsonl')
    for row in json_lines(journal):
        require(row['protocol_sha256'] == protocol and row['state_id'] not in ids, f'Diagnostic journal mismatch: {path}')
        require(row.get('anchor_restored_exactly') is True, f'Point restoration missing: {path}/{row["state_id"]}')
        ids.add(row['state_id'])
    require(len(ids) == result['states_completed'], f'Diagnostic state count mismatch: {path}')
    if job in ('dense', 'sparse_train', 'sparse_test', 'sparse_train_full'):
        expected_states = {'dense': 8001, 'sparse_train': 97, 'sparse_test': 49, 'sparse_train_full': 13}[job]
        require(result['states_completed'] == expected_states, f'Formal grid cardinality mismatch: {path}')
        radius_expected = {'dense': [i / 4 for i in range(81)], 'sparse_train': [0, .5, 1, 2, 5, 10, 20],
                           'sparse_test': [0, 1, 10, 20], 'sparse_train_full': [0, 1, 10, 20]}[job]
        require([r['radius'] for r in result['radius_summary']] == radius_expected, f'Formal radii mismatch: {path}')
        expected_k = {'dense': 100, 'sparse_train': 16, 'sparse_test': 16, 'sparse_train_full': 4}[job]
        require(result['direction_indices'] == list(range(expected_k)), f'Formal direction library mismatch: {path}')
        if job == 'dense':
            require(result['forward_only'] is True and result['gradient_scope'] == [], f'Dense job did not remain forward-only: {path}')
        else:
            scope, count = {'sparse_train': ('subset', 256), 'sparse_test': ('test', 307), 'sparse_train_full': ('train', 1259)}[job]
            require(result['gradient_scope'] == [scope], f'Gradient scope mismatch: {path}')
            for radius in result['radius_summary']:
                trace = radius['raw_trace_by_scope'][scope]
                require(trace['samples'] == count and numeric(trace['mean'], 'trace') > 0, f'Fisher sample count invalid: {path}')
    elif job == 'trajectory_fisher':
        require(result['task_count'] == 10 and [r['task'] for r in result['tasks']] == list(range(10)), f'Trajectory incomplete: {path}')
        for row in result['tasks']:
            require(row['fisher_by_scope']['train']['samples'] == 1259 and row['endpoint_recovery']['passed'] is True,
                    f'Trajectory train Fisher/recovery incomplete: {path}')
            if row['task'] in (0, 1, 5, 9):
                require(row['fisher_by_scope']['test']['samples'] == 307, f'Trajectory test Fisher missing: {path}')
    elif job.startswith('path_t'):
        require(result['target_task'] == int(job[-2:]) and result['samples'] == 1259, f'Path target/scope mismatch: {path}')
        require(result['selected_nodes'] in (9, 17, 33) and result['numerical_gate_passed'] is True and
                result['quadrature_converged'] is True and result['bound_passed_with_numeric_tolerance'] is True,
                f'Path numerical gate failed: {path}')
        require(result['endpoint_recovery']['passed'] is True, f'Path endpoint mismatch: {path}')
    elif job == 'bridge_full':
        require(result['random_directions'] == 100, f'Bridge random controls incomplete: {path}')
        require(set(result['actual_drift_radii_included']) == {'sgd_t09', 'gam_t09'}, f'Bridge lacks actual directions: {path}')
        require(all(any(abs(v - r) < 1e-12 for r in result['radii']) for v in result['actual_drift_radii_included'].values()),
                f'Bridge lacks actual radii: {path}')
        require(result['states_completed'] == 1 + (len(result['radii']) - 1) * 102, f'Bridge state count invalid: {path}')
    return result, {'summary': str(path), 'sha256': sha_file(path), 'states': len(ids), 'protocol_sha256': protocol}


def resolve_b3_path(root, plan, seed, branch):
    mapped = plan.get('report_inputs', {}).get('b3', {}).get(str(seed), {}).get(branch)
    return Path(mapped) if mapped else root / 'formal_tables' / f'seed{seed}' / branch / 'summary.json'


def aggregate(rows):
    result = []
    for branch in BRANCHES:
        selected = sorted((r for r in rows if r['branch'] == branch), key=lambda r: r['seed'])
        require([r['seed'] for r in selected] == [0, 1, 2], f'Branch {branch} lacks three distinct seeds')
        result.append({'branch': branch, 'anchor_method': BRANCHES[branch][0], 'ewc_lambda': BRANCHES[branch][1],
                       'statistics': {key: mean_sd([r[key] for r in selected]) for key in METRICS}})
    return result


def contrasts(rows):
    index = {(r['seed'], r['branch']): r for r in rows}
    out = []
    for seed in range(3):
        for name, left, right in [('B-A', 'B', 'A'), ('D-C', 'D', 'C'), ('C-A', 'C', 'A'), ('D-B', 'D', 'B')]:
            out.append({'seed': seed, 'contrast': name, **{k: index[seed, left][k] - index[seed, right][k] for k in METRICS}})
        out.append({'seed': seed, 'contrast': '(D-C)-(B-A)',
                    **{k: index[seed, 'D'][k] - index[seed, 'C'][k] - index[seed, 'B'][k] + index[seed, 'A'][k] for k in METRICS}})
    return out


def tex_cell(stat, digits=2):
    return f"${stat['mean']:.{digits}f} \\pm {stat['sample_sd_ddof1']:.{digits}f}$"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--formal-plan', '--plan', dest='formal_plan', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = (args.formal_plan or root / 'formal_plan.json').resolve()
    plan = read_json(plan_path)
    output = root / 'final_report'
    output.mkdir(parents=True, exist_ok=True)
    errors, rows, identities, paired, diagnostics, receipts = [], [], [], {}, {}, []
    # Plan receipts include training and endpoint jobs.  Exclude only this report
    # itself to avoid a circular prerequisite.
    for stage in plan.get('stages', []):
        for job in stage['jobs']:
            path = Path(job['receipt'])
            if path.resolve() == (output / 'COMPLETE.json').resolve() or any(Path(str(arg)).name == 'report.py' for arg in job.get('command', [])):
                continue
            try:
                value = read_json(path)
                require(all(value.get(k) == v for k, v in job['receipt_require'].items()), f'Plan gate failed: {job["id"]}')
                receipts.append({'job': job['id'], 'receipt': str(path), 'sha256': sha_file(path), 'passed': True})
            except Exception as exc:
                errors.append(f'plan/{job["id"]}: {exc}')
    for seed in range(3):
        seed_rows = {}
        for branch in BRANCHES:
            path = resolve_b3_path(root, plan, seed, branch)
            try:
                row, identity = validate_endpoint(path, root, seed, branch)
                rows.append(row)
                identities.append(identity)
                seed_rows[branch] = row
            except Exception as exc:
                errors.append(f'B3/seed{seed}/{branch}: {exc}')
        try:
            paired[str(seed)] = check_rng(seed_rows)
        except Exception as exc:
            paired[str(seed)] = {'passed': False, 'error': str(exc)}
            errors.append(f'RNG/seed{seed}: {exc}')
    if identities and any(value != identities[0] for value in identities[1:]):
        errors.append('Formal B3 runs differ in runner/source/library identity or shared future-training settings; no cross-protocol pooling.')
    for seed in range(3):
        for method in ('sgd', 'gam'):
            for job in DIAGNOSTICS:
                key = (seed, method, job)
                path = root / 'formal_results' / f'seed{seed}' / method / job / 'summary.json'
                try:
                    diagnostics[key] = validate_diagnostic(path, seed, method, job)
                except Exception as exc:
                    errors.append(f'diagnostic/seed{seed}/{method}/{job}: {exc}')
    complete = not errors and len(rows) == 12 and len(diagnostics) == 54
    statistics_rows = aggregate(rows) if len(rows) == 12 and all(p['passed'] for p in paired.values()) and len({canonical_hash(v) for v in identities}) == 1 else []
    contrast_rows = contrasts(rows) if statistics_rows else []
    write_csv(output / 'individual.csv', rows, fields=list(rows[0]) if rows else ['seed', 'branch'])
    if contrast_rows:
        write_csv(output / 'paired_contrasts.csv', contrast_rows)
    write_json(output / 'B3_PAIRED_RNG.json', paired)
    write_json(output / 'aggregate.json', statistics_rows)
    write_json(output / 'coverage.json', {'validated_b3': len(rows), 'expected_b3': 12,
        'validated_diagnostics': len(diagnostics), 'expected_diagnostics': 54,
        'diagnostics': [{'seed': s, 'method': m, 'job': j, **evidence} for (s, m, j), (_, evidence) in diagnostics.items()],
        'plan_receipts': receipts, 'errors': errors})
    lines = ['# Formal RQ2 three-seed results', '', f"状态：{'complete' if complete else 'partial'}。B3 {len(rows)}/12；B1/B2 {len(diagnostics)}/54 作业通过完整范围核验。", '',
             '本报告仅汇总 formal_training 下同一新训练协议的 ImageNet-A 结果；未合并历史 seed0 pilot、旧 ImageNet-R 表或历史十任务链。所有 B3 后续任务统一 SGD、50 epochs、lr 0.02、batch 128；C/D 的 EWC λ=2000，A/B 为0。', '',
             '数值为 seeds 0/1/2 的均值 ± 样本标准差（ddof=1）；三seed不用于强显著性声明。a11使用当前FP32评估；a21/a22继承训练记录的精度，BWT在每seed先求差再聚合。', '']
    if statistics_rows:
        lines += ['| 分支 | 起点 | λ | a11 (%) | a21 (%) | a22 (%) | BWT (pp) | 平均 (%) |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for entry in statistics_rows:
            st = entry['statistics']
            cell = lambda k, d=2: f"{st[k]['mean']:.{d}f} ± {st[k]['sample_sd_ddof1']:.{d}f}"
            lines.append(f"| {entry['branch']} | {entry['anchor_method']} | {entry['ewc_lambda']} | " + ' | '.join(cell(k) for k in ('a11', 'a21', 'a22', 'BWT', 'average_accuracy')) + ' |')
        lines += ['', '| 分支 | 旧任务 raw trace：anchor | 旧任务 raw trace：endpoint | 有效权重位移 L2 | 位移平方 | train D0 | test D0 |', '|---|---:|---:|---:|---:|---:|---:|']
        for entry in statistics_rows:
            st = entry['statistics']
            cell = lambda k, d: f"{st[k]['mean']:.{d}f} ± {st[k]['sample_sd_ddof1']:.{d}f}"
            lines.append(f"| {entry['branch']} | " + ' | '.join(cell(k, d) for k, d in [('raw_trace_anchor_train256', 3), ('raw_trace_endpoint_train256', 3), ('drift_l2', 6), ('drift_squared', 6), ('D0_train', 6), ('D0_test', 6)]) + ' |')
        lines += ['', 'Fisher两列均在同一旧任务256个train样本上，用固定旧20类head、真实标签逐样本梯度及有效Q/V权重坐标测量；不归一化、无floor。endpoint表示后续任务训练后的旧任务Fisher，不能改称新任务Fisher。参数移动为 ||W_eff(t1)−W_eff(t0)||_F，排除分类头；平方列为每seed范数平方后聚合。', '',
                  '| 配对对照 | Δa21 (pp) | Δa22 (pp) | ΔBWT (pp) | Δ平均 (pp) | Δtest D0 |', '|---|---:|---:|---:|---:|---:|']
        for name in ('B-A', 'D-C', 'C-A', 'D-B', '(D-C)-(B-A)'):
            values = [r for r in contrast_rows if r['contrast'] == name]
            formatted = []
            for key in ('a21', 'a22', 'BWT', 'average_accuracy', 'D0_test'):
                st = mean_sd([r[key] for r in values])
                digits = 6 if key == 'D0_test' else 3
                formatted.append(f"{st['mean']:.{digits}f} ± {st['sample_sd_ddof1']:.{digits}f}")
            lines.append('| ' + name + ' | ' + ' | '.join(formatted) + ' |')
        lines += ['', 'B−A衡量无保护时的起点作用；D−C衡量有保护时的起点作用，包含各自Fisher矩阵差异。C−A和D−B衡量保护作用。损伤差值为负才表示更低损伤；accuracy/BWT差值为正才表示改善。不能由更低trace单独推出因果或普遍保持收益。']
    else:
        lines += ['B3正式均值表尚未通过12条记录、同协议和四分支RNG配对核验；不以缺失值或历史结果补齐。']
    lines += ['', '| seed | 方法 | B1/B2诊断通过 | 必须作业数 |', '|---:|---|---:|---:|']
    for seed in range(3):
        for method in ('sgd', 'gam'):
            lines.append(f'| {seed} | {method} | {sum((seed, method, j) in diagnostics for j in DIAGNOSTICS)} | 9 |')
    lines += ['', 'B1覆盖81半径×100公共方向dense loss、256样本train sparse、全307 test sparse、全1259 train固定首4方向一致性检查，以及两真实范数的交叉方向/同半径随机对照。B2覆盖t0…9全train Fisher、t0/1/5/9 test Fisher及t1/5/9全train路径。详细receipt、protocol hash和缺失项见coverage.json。', '',
              'B2 GAM十任务链持续使用GAM；B3 GAM起点的未来任务使用SGD。二者代表不同实验，不将B2完整方法的t1冒充B3起点消融。M4 full-model复验为附件条件项，本报告不声称已完成。']
    if errors:
        lines += ['', '未通过项：', ''] + [f'- {error}' for error in errors]
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    tex = [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
           r'\caption{Formal two-task ImageNet-A LoRA diagnostics (seeds 0, 1, 2; mean $\pm$ sample standard deviation). Task 2 always uses SGD. Both raw Fisher traces use the same 256 old-task training examples and fixed old-task head. The endpoint trace is an old-task statistic after Task 2. Effective Q/V drift excludes the classifier.}',
           r'\label{tab:formal_rq2_b3}', r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{llrrrrrrrrr}', r'\toprule',
           r'Branch & Anchor & $\lambda$ & $a_{11}$ & $a_{21}$ & $a_{22}$ & BWT & Avg. & $T_0(\theta_0)$ & $T_0(\theta_1)$ & $\|\Delta W_{QV}\|_F$ \\', r'\midrule']
    for entry in statistics_rows:
        st = entry['statistics']
        cells = [entry['branch'], entry['anchor_method'].upper(), str(entry['ewc_lambda'])]
        cells += [tex_cell(st[k], 2) for k in ('a11', 'a21', 'a22', 'BWT', 'average_accuracy')]
        cells += [tex_cell(st[k], 3) for k in ('raw_trace_anchor_train256', 'raw_trace_endpoint_train256', 'drift_l2')]
        tex.append(' & '.join(cells) + r' \\')
    if not statistics_rows:
        tex.append(r'\multicolumn{11}{c}{Pending: formal provenance and paired-seed validation incomplete.} \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'RESULTS.tex').write_text('\n'.join(tex) + '\n')
    payload = {'status': 'complete' if complete else 'partial', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'plan': str(plan_path), 'plan_sha256': sha_file(plan_path), 'report_code_sha256': sha_file(Path(__file__)),
               'validated_b3': len(rows), 'expected_b3': 12, 'validated_diagnostics': len(diagnostics), 'expected_diagnostics': 54,
               'all_four_branch_rng_checks_passed': all(p['passed'] for p in paired.values()),
               'scientific_scope': 'Formal three-seed ImageNet-A B1/B2/B3 only; no historical/pilot pooling; M4 conditional excluded',
               'errors': errors, 'artifacts': {name: sha_file(output / name) for name in
                                             ('RESULTS.md', 'RESULTS.tex', 'individual.csv', 'aggregate.json', 'coverage.json', 'B3_PAIRED_RNG.json')}}
    write_json(output / 'STATUS.json', payload)
    if complete:
        write_json(output / 'COMPLETE.json', payload)
    else:
        existing = output / 'COMPLETE.json'
        if existing.exists():
            existing.replace(output / f'COMPLETE.invalidated.{int(datetime.now().timestamp())}.json')
    print(json.dumps({'status': payload['status'], 'report': str(output / 'RESULTS.md'), 'validated_b3': len(rows),
                      'validated_diagnostics': len(diagnostics), 'error_count': len(errors)}, ensure_ascii=False))
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main())
