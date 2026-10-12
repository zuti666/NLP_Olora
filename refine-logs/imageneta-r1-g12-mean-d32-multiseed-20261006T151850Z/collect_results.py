"""Collect six new endpoints plus three external seed2 cells without duplicated training."""
import csv
import io
import json
import statistics
import subprocess
from pathlib import Path
from common import atomic_json, atomic_text, now, sha256
from validate_job import finite_tree, metrics, validate_job

H = Path(__file__).resolve().parent
LABELS = {'g1_plus_g2': 'g1+g2', 'half_g1_plus_g2': '(g1+g2)/2',
          'g1_plus_d32': 'g1+(g3-g2)'}
METRICS = {'AAA': 'AAA_task_macro', 'FAA': 'FAA_task_macro', 'LA': 'LA',
           'BWT': 'BWT', 'forgetting': 'forgetting'}


def normalized_direction(name):
    return 'g1_plus_d32' if name == 'g1_plus_second_difference' else name


def check_external(item):
    """Trust successful old controller acceptance only after rechecking its evidence and hashes."""
    root = Path(item['run_root'])
    suite = json.loads((root / 'suite.json').read_text())
    job = next(j for j in suite['jobs'] if j['id'] == item['job'])
    state = json.loads((root / 'runtime/queue_state.json').read_text())
    record = state['jobs'][job['id']]
    direction = normalized_direction(item['direction'])
    if record['status'] == 'failed':
        raise RuntimeError('External seed2 failed: ' + job['id'])
    row = dict(job=job['id'], direction=direction, label=LABELS[direction], seed=2,
               status=record['status'], accepted=False, external=True, output_dir=job['output_dir'])
    if sha256(item['config_path']) != item['config_sha256']:
        raise RuntimeError('External configuration changed')
    if record['status'] != 'completed':
        return row
    # The external campaign owns its condition-specific validator and immutable
    # execution pins. Run it in a separate interpreter so its common.HERE cannot
    # accidentally resolve to this supplement's module cache.
    pins = json.loads((root / 'RUN_PINS.json').read_text())
    for name, expected in pins['files'].items():
        if sha256(root / name) != expected:
            raise RuntimeError('External frozen execution artifact changed: ' + name)
    if sha256(suite['source_manifest']) != suite['source_manifest_sha256']:
        raise RuntimeError('External source manifest changed')
    original_check = "import sys,json;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root));from common import verify_source;from validate_job import validate_job;s=json.loads((root/'suite.json').read_text());verify_source(s);j=next(x for x in s['jobs'] if x['id']==sys.argv[2]);print(json.dumps(validate_job(j)))"
    validated = json.loads(subprocess.check_output(
        [suite['python'], '-c', original_check, str(root), job['id']],
        cwd=root, text=True, timeout=120))
    out = Path(job['output_dir'])
    path = out / 'final_summary.json'
    a = json.loads(path.read_text())
    receipt_path = root / 'runtime/jobs' / job['id'] / 'completed.json'
    receipt = json.loads(receipt_path.read_text())
    digest = sha256(path)
    if receipt['exit']['returncode'] != 0 or digest != receipt['summary_sha256'] or digest != record['summary_sha256'] or digest != validated['summary_sha256']:
        raise RuntimeError('External completion receipt mismatch')
    if item.get('source_sha256'):
        for key, expected in item['source_sha256'].items():
            if sha256(item['sources'][key]) != expected:
                raise RuntimeError('Frozen external source changed: ' + key)
    finite_tree(a)
    cfg = json.loads((out / 'effective_config.json').read_text())
    seed = cfg['seed'][0] if isinstance(cfg['seed'], list) else cfg['seed']
    if (cfg['dataset'], cfg['lora_rank'], seed, cfg['gam_grad_rho'], cfg['gam_grad_norm_rho'],
        cfg['attention_backend'], cfg['init_epoch'], cfg['epochs'], cfg['init_lr'], cfg['lrate']) != (
        'imageneta', 10, 2, .05, 1., 'native', 50, 50, .04, .02):
        raise RuntimeError('External effective configuration mismatch')
    if cfg['ewc_lambda'] != 0 or not cfg['ewc_skip_task_fisher'] or any(
        cfg[k] for k in ['approximate_fisher_in_task', 'in_task_weight_merge', 'task_endpoint_weight_merge']):
        raise RuntimeError('External prohibited module enabled')
    if (a['status'], a['completed_tasks'], a['completed_epochs'], a['completed_steps'],
        a['seed'], a['r'], a['rho']) != ('completed', 10, 500, 2550, 2, 1., .05):
        raise RuntimeError('External scientific cell/budget mismatch')
    if normalized_direction(a['update_direction']) != direction or not all(a['checks'].values()):
        raise RuntimeError('External direction/acceptance mismatch')
    c = a['counters']
    if c['training_ce_gradient_passes'] != 10200 or c['training_sgd_updates'] != 2550:
        raise RuntimeError('External probe budget mismatch')
    if any(c[k] for k in ['fisher_estimation_passes', 'weight_merge_updates', 'forbidden_work_attempts',
                         'training_hvp_passes', 'training_create_graph_passes']):
        raise RuntimeError('External forbidden work')
    for name, key in [('task_matrix.json', 'task_matrix_sha256'), ('step_trace.jsonl', 'step_trace_sha256')]:
        if sha256(out / name) != a[key]:
            raise RuntimeError('External matrix/trace changed')
    m = json.loads((out / 'task_matrix.json').read_text())
    matrix = m['matrix']
    if len(matrix) != 10 or any(len(r) != i+1 for i, r in enumerate(matrix)):
        raise RuntimeError('External incomplete triangle')
    finite_tree(matrix)
    recomputed = metrics(matrix)
    for key, value in recomputed.items():
        if abs(m[key]-value) > 1e-9 or abs(a['final_metrics'][key]-value) > 1e-9:
            raise RuntimeError('External metric mismatch')
    row.update(accepted=True, source_receipt=str(receipt_path), source_matrix=str(out/'task_matrix.json'),
               source_summary=str(path), summary_sha256=digest,
               **{key: recomputed[source] for key, source in METRICS.items()})
    return row


def external_rows():
    inventory = json.loads((H / 'evidence/inventory.json').read_text())
    return [check_external(i) for i in inventory['reusable_completed'] + inventory['pending_existing']]


def external_seed2_ready():
    return all(row['accepted'] for row in external_rows())


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def main():
    suite = json.loads((H / 'suite.json').read_text())
    state_file = H / 'runtime/queue_state.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'jobs': {}}
    rows = external_rows()
    for job in suite['jobs']:
        if job['kind'] != 'training' or job['smoke']:
            continue
        record = state['jobs'].get(job['id'], {'status': 'pending'})
        row = dict(job=job['id'], direction=job['direction'], label=LABELS[job['direction']],
                   seed=job['seed'], status=record['status'], accepted=False,
                   external=False, output_dir=job['output_dir'])
        if record['status'] == 'completed':
            validated = validate_job(job)
            if validated['summary_sha256'] != record['summary_sha256']:
                raise RuntimeError('Current accepted endpoint changed')
            out = Path(job['output_dir'])
            m = json.loads((out / 'task_matrix.json').read_text())
            row.update(accepted=True, source_receipt=str(H/'runtime/jobs'/job['id']/'completed.json'),
                       source_matrix=str(out/'task_matrix.json'), source_summary=str(out/'final_summary.json'),
                       **{key: m[source] for key, source in METRICS.items()})
        rows.append(row)
    assert len(rows) == 9 and len({(r['direction'], r['seed']) for r in rows}) == 9
    aggregates = []
    for direction, label in LABELS.items():
        group = sorted((r for r in rows if r['direction'] == direction and r['accepted']), key=lambda r:r['seed'])
        complete = len(group) == 3 and [r['seed'] for r in group] == [0, 1, 2]
        a = dict(direction=direction, label=label, expected_seeds='0;1;2',
                 completed_seeds=';'.join(str(r['seed']) for r in group), n=len(group),
                 status='complete_3_seeds' if complete else 'partial_seeds')
        for key in METRICS:
            a[key+'_mean'] = statistics.mean(r[key] for r in group) if complete else None
            a[key+'_std'] = statistics.stdev(r[key] for r in group) if complete else None
        aggregates.append(a)
    contrasts = []
    for left, right in [('g1_plus_g2', 'half_g1_plus_g2'), ('g1_plus_d32', 'g1_plus_g2')]:
        aa = {r['seed']:r for r in rows if r['accepted'] and r['direction'] == left}
        bb = {r['seed']:r for r in rows if r['accepted'] and r['direction'] == right}
        seeds = sorted(set(aa) & set(bb))
        c = dict(lhs=left, rhs=right, paired_seeds=seeds, n=len(seeds), accepted=seeds==[0,1,2])
        if c['accepted']:
            for key in METRICS:
                differences = [aa[s][key]-bb[s][key] for s in seeds]
                c[key] = dict(per_seed=differences, mean=statistics.mean(differences),
                              sample_std=statistics.stdev(differences), wins=sum(x>0 for x in differences))
        contrasts.append(c)
    csv_write(H/'summary.csv', rows)
    csv_write(H/'THREE_SEED_SUMMARY.csv', aggregates)
    atomic_json(H/'results.json', dict(updated_at=now(), seed_rows=rows, aggregates=aggregates, contrasts=contrasts))
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    lines = ['# ImageNet-A r=1：Sum12 / Mean12 / g1+d32 三seed', '',
             '固定seed0/1/2、rank10、rho=.05、10×20类、50epochs/任务；4次正向探针、精确恢复/R0重放、自然LR=.04/.02。无Fisher/EWC/权重融合/投影/HVP。', '',
             '本队列仅补6项seed0/1；2项seed2已完成并复用，Mean12 seed2由原有等待队列执行，不重复启动。', '',
             '| 更新方向 | 完整seed | AAA均值±样本SD | FAA均值±样本SD | 状态 |',
             '|---|---|---:|---:|---|']
    for a in aggregates:
        lines.append(f"| {a['label']} | {a['completed_seeds'] or '—'} | {fmt(a['AAA_mean'])}±{fmt(a['AAA_std'])} | {fmt(a['FAA_mean'])}±{fmt(a['FAA_std'])} | {a['status']} |")
    lines += ['', '| 作业 | seed | 来源 | 状态 | AAA | FAA |', '|---|---:|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['job']} | {r['seed']} | {'复用/旧队列' if r['external'] else '本轮'} | {r['status']} | {fmt(r.get('AAA'))} | {fmt(r.get('FAA'))} |")
    lines += ['', '只对完整seed0/1/2计算正式均值、样本SD和配对比较；缺失或失败不补零。Mean为最终梯度乘0.5，名义LR不再减半，不等同动态有效BA+FC步长匹配。',
              '此前原生g1+g3多seed采用另一RNG/CE协议，独立保留；本轮三方向不跨协议合并排名。']
    atomic_text(H/'RESULTS.md', '\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
