"""Read completed formal results and build the requested seven-row table on CPU.

Does not train, evaluate a model, alter frozen inputs, or impute missing arms.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extension-root', type=Path, help='Root containing training/{shaping_only,task2_shaping}_seed0..2')
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--require-extension-complete', action='store_true', help='Fail before writing unless all six extension runs validate')
args = parser.parse_args()
EXTENSION = args.extension_root.resolve() if args.extension_root else None
OUT = args.output_dir.resolve() if args.output_dir else ((EXTENSION / 'final_report') if EXTENSION else SCRIPT_DIR)
if args.require_extension_complete and EXTENSION is None:
    parser.error('--require-extension-complete requires --extension-root')
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
EXTENSION_VALIDATION = []


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def stats(values):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": 3}


def fmt(stat, digits=2, latex=False):
    if stat is None:
        return r"\textemdash" if latex else "待补"
    a, b = stat["mean"], stat["sample_sd"]
    return (rf"\ExpRes{{{a:.{digits}f}}}{{{b:.{digits}f}}}" if latex
            else f"{a:.{digits}f}±{b:.{digits}f}")



def sha_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    value = digest.hexdigest()
    SOURCES[str(path)] = value
    return value


def read_jsonl(path):
    path = Path(path)
    sha_file(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_extension_cell(arm, seed, parent, lam, anchor):
    """Validate completed new training and derive effective-weight drift on CPU."""
    run = EXTENSION / 'training' / f'{arm}_seed{seed}'
    receipt_path = run / 'COMPLETE.json'
    if not receipt_path.exists():
        if args.require_extension_complete:
            raise ValueError(f'Required extension is incomplete: {receipt_path}')
        return None
    receipt = read_json(receipt_path)
    manifest_path = run / 'run_manifest.json'
    manifest = read_json(manifest_path)
    assert receipt['status'] == 'complete' and receipt['completed_tasks'] == 2
    assert Path(receipt['manifest']).resolve() == manifest_path.resolve()
    assert receipt['identity_sha256'] == manifest['identity_sha256']
    assert manifest['seed'] == seed and manifest['anchor_method'] == parent
    assert manifest.get('run_name', f'{arm}_seed{seed}') == f'{arm}_seed{seed}'
    assert manifest['future_optimizer'] == 'gam' and manifest['arm'] == arm
    assert manifest['mode'] == 'continuation' and manifest['smoke_only'] is False
    assert manifest['task0_shaping'] == (parent == 'gam') and manifest['task1_shaping'] is True
    identity_keys = ('protocol', 'mode', 'anchor_method', 'future_optimizer', 'arm', 'smoke_only',
                     'task0_shaping', 'task1_shaping', 'parent_validation', 'seed', 'requested_config',
                     'parent', 'source_manifest_sha256', 'runner_sha256')
    original_identity = {k: manifest[k] for k in identity_keys}
    assert hashlib.sha256(json.dumps(original_identity, sort_keys=True).encode()).hexdigest() == manifest['identity_sha256']
    assert manifest['runner_sha256'] == sha_file(EXTENSION / 'train.py')
    required = {'dataset': 'imageneta', 'seed': [seed], 'optimizer_type': 'gam',
                'lambda_flat': 1.0, 'ewc_lambda': lam, 'epochs': 50, 'lrate': .02,
                'batch_size': 128, 'lora_rank': 10, 'scheduler': 'cosine',
                'momentum': 0, 'weight_decay': 0.0, 'init_cls': 20, 'increment': 20,
                'class_shuffle': False, 'max_train_tasks': 2, 'resume_from_task': 0,
                'gam_base_optimizer': 'sgd', 'gam_grad_rho': .05, 'gam_grad_norm_rho': .2,
                'memory_size': 0, 'memory_per_class': 0, 'ewc_gamma': .9, 'ewc_eta': 0.0,
                'ewc_normalize_fisher': True, 'ewc_normalize_accumulated_fisher': True,
                'rq2_rng_protocol': 'rq2_shared_task_rng_v1', 'rq2_smoke_only': False,
                'rq2_extension_arm': arm, 'train_num_workers': 4, 'eval_num_workers': 4}
    for config_name in ('config', 'requested_config'):
        cfg = manifest[config_name]
        for key, expected in required.items():
            assert cfg[key] == expected, (run, config_name, key, cfg[key], expected)
        for flag in ('weight_merge', 'in_task_weight_merge', 'approximate_fisher_in_task',
                     'ewc_project_grad', 'delta_w_penalty'):
            assert cfg[flag] is False, (run, flag)
        assert Path(cfg['resume_from_checkpoint_dir']).resolve() == Path(anchor['checkpoint']).parent.resolve()
        assert Path(cfg['save_task_checkpoint_dir']).resolve() == (run / 'checkpoints').resolve()
    assert manifest['matmul_allow_tf32'] is False and manifest['cudnn_allow_tf32'] is False
    frozen = BASE.parent / 'source_frozen.json'
    assert manifest['source_manifest_sha256'] == sha_file(frozen)
    assert Path(manifest['source']).resolve() == (BASE.parent / 'source').resolve()
    for entry in read_json(frozen)['files']:
        assert sha_file(BASE.parent / 'source' / entry['path']) == entry['sha256']
    # The parent checkpoint, its Fisher, and its metrics must be the intended seed's formal anchor.
    anchor_run = BASE / f'formal_training/anchor_{parent}_seed{seed}'
    expected_parent = {'task_00.pt': Path(anchor['checkpoint']),
                       'fisher_state_0.pt': anchor_run / 'checkpoints/fisher_state_0.pt',
                       'metrics': Path(anchor['metrics']),
                       'run_manifest.json': anchor_run / 'run_manifest.json',
                       'COMPLETE.json': anchor_run / 'COMPLETE.json'}
    assert manifest['parent_validation']['passed'] is True
    assert manifest['parent_validation']['seed'] == seed and manifest['parent_validation']['anchor_method'] == parent
    assert Path(manifest['parent_validation']['formal_parent']).resolve() == anchor_run.resolve()
    for name, expected in expected_parent.items():
        saved = manifest['parent'][name]
        assert Path(saved['path']).resolve() == expected.resolve(), (name, saved['path'], expected)
        assert sha_file(expected) == saved['sha256']
    assert manifest['parent']['task_00.pt']['sha256'] == anchor['checkpoint_sha256']
    valid = receipt['validation']
    assert valid['epochs_per_new_task'] == 50 and valid['validated_tasks'] == [1]
    metrics_path = Path(valid['metrics'])
    assert metrics_path.resolve().is_relative_to(run.resolve())
    metrics = read_json(metrics_path)
    assert SOURCES[str(metrics_path)] == valid['metrics_sha256']
    for name, asset in valid['checkpoint_assets'].items():
        path = run / name
        assert path.resolve().is_relative_to((run / 'checkpoints').resolve())
        assert path.stat().st_size == asset['bytes'] and sha_file(path) == asset['sha256']
    endpoint_path = run / 'checkpoints/task_01.pt'
    assert 'checkpoints/task_01.pt' in valid['checkpoint_assets']
    runtime = read_jsonl(run / 'runtime_receipts.jsonl')
    rng = read_jsonl(run / 'rng_receipts.jsonl')
    assert SOURCES[str(run / 'runtime_receipts.jsonl')] == valid['runtime_receipts_sha256']
    assert SOURCES[str(run / 'rng_receipts.jsonl')] == valid['rng_receipts_sha256']
    resume = [r for r in runtime if r.get('event') == 'complete_resume']
    assert resume and all(r['verified_exact'] and r['task'] == 0 and
        r['checkpoint_sha256'] == anchor['checkpoint_sha256'] and
        r['fisher_sha256'] == manifest['parent']['fisher_state_0.pt']['sha256'] for r in resume)
    assert any(r.get('event') == 'task_complete' and r.get('task') == 1 for r in runtime)
    resets = [r for r in rng if r.get('event') == 'task_rng_reset' and r.get('task') == 1]
    assert resets and all(r['optimizer_type'] == 'gam' for r in resets)
    iterations = [r for r in rng if r.get('event') == 'loader_iteration' and r.get('task') == 1 and r.get('shuffle')]
    by_iter = {(r['loader_serial'], r['iteration']): r for r in iterations}
    assert len(by_iter) == len(iterations), 'Duplicated training passes; investigate resume receipts'
    assert all((0, i) in by_iter for i in range(50))
    assert all(by_iter[0, i]['samples'] == by_iter[0, i]['sample_count'] == 997 and
               by_iter[0, i]['batches'] == 8 for i in range(50))
    reference = BASE / 'formal_training' / (f'B3_D_gam_lam2000_seed{seed}' if parent == 'gam'
                                           else f'sequence_sgd_seed{seed}')
    reference_rng = read_jsonl(reference / 'rng_receipts.jsonl')
    for event in ('task_rng_reset', 'head_initialized', 'loader_created', 'loader_iteration'):
        def select(rows):
            return [{k: v for k, v in r.items() if k != 'optimizer_type'} for r in rows
                    if r.get('event') == event and r.get('task') == 1 and
                    (event not in ('loader_created', 'loader_iteration') or r.get('shuffle') is True)]
        assert select(rng) == select(reference_rng), (run, 'Paired Task1 RNG mismatch', event)
    logs = '\n'.join(p.read_text() for p in (run / 'trainer_outputs').rglob('*.log'))
    assert re.search(r'Task 1, Epoch 50/50\b', logs), 'Missing actual 50/50 training log'
    mechanism = metrics['mechanism']['steps']['1']
    assert mechanism['lambda_flat'] == 1.0 and mechanism['lambda_ewc'] == lam
    matrix = metrics['cnn']['matrices']['t01']
    assert len(matrix) == 2 and all(len(r) == 2 for r in matrix)
    assert metrics['cnn']['matrices']['final'][1][:2] == matrix[1]
    assert abs(matrix[0][0] - anchor['a11']) < .0051
    assert all(math.isfinite(float(v)) for v in matrix[1])
    local_rows = read_jsonl(run / 'task0_local_metrics.jsonl')
    local = {r['task']: r for r in local_rows}
    assert math.isclose(local[0]['test']['seen_accuracy'], anchor['a11'], abs_tol=1e-10)
    assert round(local[1]['test']['seen_accuracy'], 2) == matrix[1][0]
    # No model forward pass: only the 24 stored BA products and CPU scalar reductions.
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(BASE.parent))
    from rq2_common import read_state, effective_weights, norm_sq
    old, new = read_state(anchor['checkpoint']), read_state(endpoint_path)
    assert old['fc.weight'].shape == (20, 768) and new['fc.weight'].shape == (40, 768)
    assert torch.equal(old['fc.weight'], new['fc.weight'][:20])
    assert torch.equal(old['fc.bias'], new['fc.bias'][:20])
    for key, value in old.items():
        if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
            assert torch.equal(value, new[key]), f'Frozen backbone changed: {key}'
    delta = [b - a for a, b in zip(effective_weights(old), effective_weights(new))]
    assert len(delta) == 24 and all(torch.isfinite(x).all() for x in delta)
    drift_squared = norm_sq(delta)
    drift = math.sqrt(drift_squared)
    assert math.isclose(drift, local[1]['effective_weight_displacement'], rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(drift_squared, mechanism['delta_norm'], rel_tol=1e-5, abs_tol=1e-8)
    proof = {'arm': arm, 'seed': seed, 'status': 'complete',
             'manifest_sha256': SOURCES[str(manifest_path)], 'identity_sha256': manifest['identity_sha256'],
             'parent_checkpoint_sha256': anchor['checkpoint_sha256'],
             'endpoint_checkpoint_sha256': valid['checkpoint_assets']['checkpoints/task_01.pt']['sha256'],
             'epochs': 50, 'training_batches_per_epoch': 8, 'actual_future_optimizer': 'gam',
             'lambda_flat': 1.0, 'ewc_lambda': lam, 'paired_task1_rng_passed': True,
             'old_head_unchanged': True, 'frozen_backbone_unchanged': True,
             'drift_cpu_l2': drift, 'drift_cpu_l2_squared': drift_squared,
             'drift_calculation': '24 CPU effective Q/V BA differences; FP64 squared-norm reduction; no model evaluation'}
    EXTENSION_VALIDATION.append(proof)
    return {'a21': float(matrix[1][0]), 'a22': float(matrix[1][1]), 'd_QV_cont': drift,
            'result_source': str(metrics_path), 'drift_source': str(endpoint_path)}


raw_csv = BASE / 'final_report/individual.csv'
SOURCES[str(raw_csv)] = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
existing = list(csv.DictReader(raw_csv.open()))
by_cell = {(int(x['seed']), x['branch']): x for x in existing}
anchors = {}
checks = []
for method, branch in [('sgd', 'A'), ('gam', 'B')]:
    for seed in range(3):
        run = BASE / f'formal_training/anchor_{method}_seed{seed}'
        receipt = read_json(run / 'COMPLETE.json')
        manifest = read_json(run / 'run_manifest.json')
        metrics_path = Path(receipt['validation']['metrics'])
        metrics = read_json(metrics_path)
        assert SOURCES[str(metrics_path)] == receipt['validation']['metrics_sha256']
        assert receipt['status'] == 'complete' and manifest['seed'] == seed
        assert manifest['anchor_method'] == method
        m = metrics['mechanism']['steps']['0']
        assert m['fisher_estimator_family'] == 'batch_gradient_square_proxy'
        assert not m['fisher_is_per_sample_empirical']
        assert m['fisher_num_samples'] == 1259 and not m['in_task_fisher_used_path']
        assert m['in_task_weight_merge_applied'] == 0
        assert m['train_ewc_mean'] == 0  # Task 1 has no historical task to protect.
        row = by_cell[(seed, branch)]
        anchors[(method, seed)] = {
            'a11': float(row['a11']),
            'B1_batch_proxy': m['in_task_fisher_final_raw_trace'],
            'raw_trace_anchor_train256': float(row['raw_trace_anchor_train256']),
            'checkpoint_sha256': row['anchor_checkpoint_sha256'],
            'checkpoint': str(run / 'checkpoints/task_00.pt'),
            'metrics': str(metrics_path),
        }
        checks.append({'kind': 'anchor', 'method': method, 'seed': seed, 'passed': True})

arms = [
    ('vanilla', 'Vanilla LoRA', 'sgd', 'sgd', 0, 'C', 'C', 'A'),
    ('shaping_only', 'Shaping only', 'gam', 'gam', 0, 'C+F', 'C+F', None),
    ('ewclora', 'EWCLoRA', 'sgd', 'sgd', 2000, 'C', 'C+P', 'C'),
    ('fs_ewclora', 'FS-EWCLoRA', 'gam', 'gam', 2000, 'C+F', 'C+F+P', 'sequence_gam'),
    ('task1_shaping_no_ewc', 'Task-1 shaping without EWC', 'gam', 'sgd', 0, 'C+F', 'C', 'B'),
    ('task1_shaping', 'Task-1 shaping', 'gam', 'sgd', 2000, 'C+F', 'C+P', 'D'),
    ('task2_shaping', 'Task-2 shaping', 'sgd', 'gam', 2000, 'C', 'C+F+P', None),
]
numeric = ['a11', 'B1_batch_proxy', 'raw_trace_anchor_train256', 'a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']
records, grouped, ledger = [], [], []
for arm, label, parent, future, lam, obj1, obj2, source in arms:
    rows = []
    for seed in range(3):
        anchor = anchors[(parent, seed)]
        row = {'arm': arm, 'variant': label, 'seed': seed, 'parent_method': parent,
               'future_optimizer': future, 'ewc_lambda': lam, 'task1_objective': obj1,
               'task2_objective': obj2, 'status': 'complete' if source else 'not_run',
               'a11': anchor['a11'], 'B1_batch_proxy': anchor['B1_batch_proxy'],
               'raw_trace_anchor_train256': anchor['raw_trace_anchor_train256'],
               'parent_checkpoint': anchor['checkpoint'], 'anchor_metrics': anchor['metrics']}
        row.update({k: None for k in ['a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']})
        if source == 'sequence_gam':
            run = BASE / f'formal_training/sequence_gam_seed{seed}'
            receipt = read_json(run / 'COMPLETE.json')
            manifest = read_json(run / 'run_manifest.json')
            path = Path(receipt['validation']['metrics'])
            metrics = read_json(path)
            assert SOURCES[str(path)] == receipt['validation']['metrics_sha256']
            assert receipt['status'] == 'complete'
            config = manifest['config']
            assert config['optimizer_type'] == 'gam' and config['lambda_flat'] == 1
            assert config['ewc_lambda'] == 2000 and config['epochs'] == 50
            for flag in ['weight_merge', 'in_task_weight_merge', 'approximate_fisher_in_task']:
                assert config[flag] is False
            matrix = metrics['cnn']['matrices']['t01']
            assert abs(matrix[0][0] - anchor['a11']) < .0051
            trajectory_path = BASE / f'formal_results/seed{seed}/gam/trajectory_fisher/summary.json'
            trajectory = read_json(trajectory_path)
            assert trajectory['status'] == 'complete'
            t0 = next(x for x in trajectory['tasks'] if x['task'] == 0)
            t1 = next(x for x in trajectory['tasks'] if x['task'] == 1)
            assert t0['endpoint_recovery']['checkpoint_sha256'] == anchor['checkpoint_sha256']
            assert t1['endpoint_recovery']['passed']
            row.update(a21=matrix[1][0], a22=matrix[1][1],
                       d_QV_cont=t1['effective_weight_drift'],
                       result_source=str(path), drift_source=str(trajectory_path))
            assert math.isclose(row['d_QV_cont'] ** 2, t1['effective_weight_drift_squared'], rel_tol=1e-10)
        elif source:
            old = by_cell[(seed, source)]
            summary_path = Path(old['summary'])
            read_json(summary_path)
            assert SOURCES[str(summary_path)] == old['summary_sha256']
            assert old['anchor_method'] == parent and old['task1_optimizer'] == future
            assert float(old['ewc_lambda']) == lam
            row.update(a21=float(old['a21']), a22=float(old['a22']), d_QV_cont=float(old['drift_l2']),
                       result_source=str(summary_path), drift_source=str(summary_path))
        if source is None and EXTENSION is not None:
            new = validate_extension_cell(arm, seed, parent, lam, anchor)
            if new is not None:
                row.update(new)
                row['status'] = 'complete'
        complete = row['status'] == 'complete'
        if complete:
            row['BWT'] = row['a21'] - row['a11']
            row['average_accuracy'] = (row['a21'] + row['a22']) / 2
            assert all(math.isfinite(row[k]) for k in numeric)
        ledger.append({'arm': arm, 'seed': seed, 'requested_main_row': True,
                       'status': row['status'], 'anchor_available': True,
                       'observed_metrics_valid': complete, 'missing_metrics': [k for k in numeric if row[k] is None]})
        records.append(row); rows.append(row)
    grouped.append({'arm': arm, 'variant': label, 'task1_objective': obj1, 'task2_objective': obj2,
                    'status': 'complete' if all(x['status'] == 'complete' for x in rows) else 'pending',
                    'statistics': {k: stats([x[k] for x in rows]) if all(x[k] is not None for x in rows) else None for k in numeric}})

paired = []
for left, right in [('fs_ewclora', 'ewclora'), ('fs_ewclora', 'task1_shaping'), ('task1_shaping', 'ewclora'),
                    ('task1_shaping_no_ewc', 'vanilla'), ('task1_shaping', 'task1_shaping_no_ewc'),
                    ('shaping_only', 'vanilla'), ('fs_ewclora', 'shaping_only'),
                    ('task2_shaping', 'ewclora'), ('fs_ewclora', 'task2_shaping')]:
    aa = {x['seed']: x for x in records if x['arm'] == left}
    bb = {x['seed']: x for x in records if x['arm'] == right}
    if not all(aa[s]['status'] == bb[s]['status'] == 'complete' for s in range(3)):
        continue
    for key in ['a21', 'a22', 'BWT', 'average_accuracy', 'd_QV_cont']:
        deltas = [aa[s][key] - bb[s][key] for s in range(3)]
        paired.append({'contrast': f'{left}-{right}', 'metric': key, **stats(deltas),
                       'seed0_delta': deltas[0], 'seed1_delta': deltas[1], 'seed2_delta': deltas[2],
                       'positive_count': sum(v > 0 for v in deltas)})

validated = sum(x['status'] == 'complete' for x in records)
expected = len(arms) * 3
missing = expected - validated
if args.require_extension_complete:
    assert len(EXTENSION_VALIDATION) == 6 and validated == expected == 21
data = {'scope': 'Formal ImageNet-A; seven requested two-task arms',
        'seeds': [0, 1, 2], 'expected_main_cells': expected, 'validated_main_cells': validated,
        'missing_main_cells': missing, 'this_script_launches_training': False,
        'extension_root': str(EXTENSION) if EXTENSION else None, 'extension_validation': EXTENSION_VALIDATION,
        'definitions': {'B1_batch_proxy': 'Task-1 endpoint training batch-gradient delta-W Fisher proxy, before normalization; 1259 samples',
                        'd_QV_cont': 'Frobenius norm of total effective Q/V weight change from Task 1 endpoint to Task 2 endpoint; excludes heads',
                        'BWT': 'a21 minus common unrounded FP32 a11, computed per seed',
                        'missing': 'not_run; never zero or an estimated result'},
        'aggregates': grouped, 'individual': records, 'paired_contrasts': paired,
        'source_sha256': SOURCES}
(OUT / 'table_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(OUT / 'VALIDITY_LEDGER.json').write_text(json.dumps({'expected_main_cells': expected, 'validated_main_cells': validated,
    'missing_main_cells': missing, 'invalid_observed_cells': 0, 'anchor_checks': checks,
    'extension_validation': EXTENSION_VALIDATION, 'cells': ledger}, ensure_ascii=False, indent=2) + '\n')
for filename, rows in [('individual.csv', records), ('paired_contrasts.csv', paired)]:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / filename).open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=keys); w.writeheader(); w.writerows(rows)

md = ['# ImageNet-A：七行两任务诊断表', '',
      f'状态：{validated}/{expected}个seed×分支结果通过核验，{missing}个未有完成结果。GAM→SGD无EWC已纳入主表，位于FS-EWCLoRA之后。' + (' 六个补充分支已全部完成。' if missing == 0 else ' Shaping only与Task-2 shaping各三个seed待完成；已有父起点可复用。'), '',
      '第一/第二任务对应代码task0/task1。数值为seed0/1/2均值±样本标准差；a21/a22为40类已见类别下的任务分组准确率，BWT逐seed用a21−a11计算。第一任务无历史保护损失；第二任务含P的行λ=2000，其余λ=0。', '',
      '| Variant | Task 1 | Task 2 | a11 | B1 batch proxy | a21 | a22 | BWT | d_QV_cont | 两任务平均 |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
def mdrow(g):
    s = g['statistics']
    return '| ' + ' | '.join([g['variant'], g['task1_objective'], g['task2_objective']] +
        [fmt(s[k], d) for k, d in [('a11',2),('B1_batch_proxy',3),('a21',2),('a22',2),('BWT',2),('d_QV_cont',4),('average_accuracy',2)]]) + ' |'
md += [mdrow(g) for g in grouped]
md += ['', ('缺失行中的a11/B1是其已完成、待复用父起点的测量，不代表第二任务已执行。' if missing else '全部21个seed×分支结果已完成；新补六条使用同seed父起点，并通过50轮、实际GAM、配对RNG和CPU权重位移核验。'), '',
       '**指标对应**', '',
       '- B1取正式anchor训练日志的in_task_fisher_final_raw_trace：原生batch-gradient平方proxy，未归一化、无路径平均。SGD/GAM的原始逐seed值与路径见individual.csv。',
       '- 之前报告的1451.954/634.510是固定旧train256上的逐样本raw Fisher；它与本表B1不同，不能混填。两种指标均已记录。',
       '- 实际LoRA作用在Q/V，且第二任务延续同一LoRA分支，故将旧表fresh-adapter r_KV改为d_QV_cont（持续分支的端点间位移）；该量排除分类头，不是A/B因子的参数距离，也不可与旧fresh-KV范数直接横比。',
       '- F表示GAM实现的shaping更新，不声称代码直接优化一个显式Fisher-trace惩罚。',
       '- FS-EWCLoRA复用正式sequence_gam的t01矩阵及B2 t1有效位移；父anchor与其它同seed分支一致，无需重训这3条。',
       '- FS分支尚无B3同口径的endpoint-train256逐样本raw Fisher，但用户本表只需第一任务B1，因此不阻塞这行。',
       '- 当前表不包含full-model对应实验；caption不声称尚未完成的full-model附录结果存在。', '',
       '**已完成分支的配对差值**', '', '| 对照 | 指标 | 均值±SD | seed0/1/2差值 |', '|---|---|---:|---|']
for p in paired:
    if p['metric'] in ['a21','a22','average_accuracy']:
        md.append(f"| {p['contrast']} | {p['metric']} | {p['mean']:+.3f}±{p['sample_sd']:.3f} | {p['seed0_delta']:+.3f}/{p['seed1_delta']:+.3f}/{p['seed2_delta']:+.3f} |")
md += ['', '上述差值仅用于三seed均完成的匹配分支。缺失分支不参与差值，也不以估计值填充。GAM起点→未来SGD与未来持续GAM是不同对照，不能混用。', '',
       '**审查与复现**', '',
       f'- [配置实现与最少补训（历史审计）]({SCRIPT_DIR / "CONFIG_IMPLEMENTATION_AUDIT.md"})',
       f'- [连续GAM结果复用]({SCRIPT_DIR / "CONTINUOUS_GAM_REUSE.md"})',
       f'- [指标定义审查]({SCRIPT_DIR / "METRIC_MAPPING_AUDIT.md"})',
       f'- [原始缺失实验清单（历史审计）]({SCRIPT_DIR / "MISSING_EXPERIMENTS.md"})',
       '- [LaTeX表](table_imageneta.tex) / [逐seed CSV](individual.csv) / [完整数据与来源SHA](table_data.json) / [完整性台账](VALIDITY_LEDGER.json)',
       '- CPU复现：原始15个已完成单元格用build_table.py；补充全表用seven_row_extension_20260921/report.py。脚本不启动模型或训练，只读取结果并在CPU计算权重位移。', '']
(OUT / 'TABLE_IMAGENETA.md').write_text('\n'.join(md))

def obj(x):
    return '$' + '+'.join(r'\mathsf '+v for v in x.split('+')) + '$'
tex = [r'% Requires booktabs; values are mean with sample-SD subscripts.',
       r'\providecommand{\ExpRes}[2]{\ensuremath{#1_{\!\pm #2}}}',
       r'\begin{table*}[t]', r'\centering',
       r'\caption{Two-task diagnostics of solution shaping and historical protection with rank-10 Q/V LoRA on ImageNet-A. $\mathsf C$, $\mathsf F$, and $\mathsf P$ denote current-task fitting, the implemented GAM shaping update, and historical EWC protection. Values are means over seeds 0, 1, and 2, with sample standard deviations as subscripts. $\mathcal B_1$ is the unnormalized Task-1 endpoint batch-gradient Fisher proxy in effective Q/V coordinates (fit loader, batch size 128); $d_{QV}^{\mathrm{cont}}$ is the endpoint-to-endpoint effective Q/V weight change of the same continued LoRA branch during Task 2, excluding classifier heads; this is not a fresh-adapter norm. References are endpoints without trajectory averaging. Dashes indicate unrun Task-2 branches; their Task-1 entries reuse the measured parent anchors.}',
       r'\label{tab:two_task_lora_diagnostics_imageneta}', r'\scriptsize',
       r'\setlength{\tabcolsep}{3.0pt}', r'\renewcommand{\arraystretch}{1.08}',
       r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccccc@{}}',
       r'\toprule',
       r'& \multicolumn{2}{c}{Training objective} & \multicolumn{2}{c}{Task 1 endpoint} & \multicolumn{3}{c}{Task 2 endpoint} & Task 2 update \\',
       r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}',
       r'Variant & Task 1 & Task 2 & $a_{11}\uparrow$ & $\mathcal B_1$ & $a_{21}\uparrow$ & $a_{22}\uparrow$ & BWT$\uparrow$ & $d_{QV}^{\mathrm{cont}}$ \\',
       r'\midrule']
for i, g in enumerate(grouped):
    if i == 4: tex.append(r'\midrule')
    s = g['statistics']
    tex.append(' & '.join([g['variant'], obj(g['task1_objective']), obj(g['task2_objective'])] +
        [fmt(s[k], d, True) for k,d in [('a11',2),('B1_batch_proxy',3),('a21',2),('a22',2),('BWT',2),('d_QV_cont',4)]]) + r' \\')
tex += [r'\bottomrule', r'\end{tabular*}', r'\end{table*}', '']
(OUT / 'table_imageneta.tex').write_text('\n'.join(tex))
print(json.dumps({'main_validated':validated,'main_expected':expected,'missing_training_runs':missing,
                  'batch_proxy_sgd':grouped[0]['statistics']['B1_batch_proxy'],
                  'batch_proxy_gam':grouped[3]['statistics']['B1_batch_proxy'],
                  'fs_ewclora':grouped[3]['statistics']}, ensure_ascii=False))
