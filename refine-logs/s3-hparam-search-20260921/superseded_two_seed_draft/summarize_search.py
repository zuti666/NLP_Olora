"""Combine immutable reused receipts and current search results without ranking incomplete points."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(p)


def main():
    completed = {}
    for item in read(HERE / 'reused_cells.json'):
        assert sha(item['receipt_path']) == item['receipt_sha256']
        receipt = read(item['receipt_path'])
        assert receipt['status'] == 'completed'
        assert sha(item['metrics_path']) == item['metrics_sha256']
        completed[item['id']] = dict(receipt['result'], status='completed')
    current = read(HERE / 'results.json')['jobs'] if (HERE / 'results.json').exists() else []
    completed.update({r['id']: r for r in current})
    metric_keys = ['faa_task_macro', 'faa_sample_weighted', 'aaa_task_macro', 'forgetting']
    rows = []
    for cell in read(HERE / 'grid_cells.json'):
        result = completed.get(cell['id'], {})
        row = dict(cell, status=result.get('status', 'pending'))
        row.update({key: result.get(key) for key in metric_keys})
        rows.append(row)
    with (HERE / 'HYPERPARAM_RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = []
    for axis in dict.fromkeys(r['axis'] for r in rows):
        for value in dict.fromkeys(r['value'] for r in rows if r['axis'] == axis):
            match = sorted([r for r in rows if r['axis'] == axis and r['value'] == value], key=lambda r: r['seed'])
            assert [r['seed'] for r in match] == [2, 3]
            done = [r for r in match if r['status'] == 'completed']
            pair = {'axis': axis, 'value': value, 'completed_seeds': len(done), 'expected_seeds': 2,
                    'seed2_faa': match[0]['faa_task_macro'], 'seed3_faa': match[1]['faa_task_macro']}
            for key in metric_keys:
                vals = [r[key] for r in done]
                pair[key + '_mean'] = statistics.mean(vals) if len(vals) == 2 else None
                pair[key + '_sample_sd'] = statistics.stdev(vals) if len(vals) == 2 else None
            # Paired delta uses one fixed, same-seed contemporary center for each seed.
            centers = {2: completed['s3_center_s2'], 3: completed['s3_mech_center_s3']}
            differences = [r['faa_task_macro'] - centers[r['seed']]['faa_task_macro'] for r in done]
            pair['paired_faa_deltas'] = differences
            pair['paired_faa_delta_mean'] = statistics.mean(differences) if len(differences) == 2 else None
            pair['paired_faa_delta_sample_sd'] = statistics.stdev(differences) if len(differences) == 2 else None
            pairs.append(pair)
    new_ids = {r['id'] for r in rows if r['source'] == 'new_search'}
    n_completed = sum(completed.get(ident, {}).get('status') == 'completed' for ident in new_ids)
    report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'new_completed': n_completed,
              'new_total': len(new_ids), 'reused_unique_completed': 9, 'unique_cells': 30,
              'grid_complete': n_completed == len(new_ids), 'rows': rows, 'paired_points': pairs,
              'scope': 'Fixed exploratory one-axis grid. Pair means/SD require both seeds. Original June29 reference scores are kept separately.'}
    write(HERE / 'HYPERPARAM_RESULTS.json', report)
    lines = ['# S3 两seed单轴超参搜索', '',
             f"更新：{report['updated_at']}。已复用9个完整配置；新增完成 {n_completed}/{len(new_ids)}。", '',
             '以下原始值仅来自完整10任务结果；两seed均完成后才计算配对均值/样本标准差。', '',
             '| 轴 | 值 | seed2 FAA | seed3 FAA | 完成seed |', '|---|---:|---:|---:|---:|']
    for pair in pairs:
        vals = [('—' if pair[k] is None else f'{pair[k]:.3f}') for k in ['seed2_faa', 'seed3_faa']]
        lines.append(f"| {pair['axis']} | {pair['value']} | {vals[0]} | {vals[1]} | {pair['completed_seeds']}/2 |")
    lines += ['', '原日志中心FAA：seed2=53.311、seed3=51.968；本表采用同冻结源码的新执行中心53.419/52.006，不将两次执行的数值混用。',
              '', 'weightmerge系数为正值扫描；已有保留快照CE的matched-no-merge单独作为零融合控制，不伪装成原生omega=0。',
              '', '详细完成状态与任务进度见 EXPERIMENT_TRACKER.md；所有GPU0–6均可分配，仍被占用的GPU等待释放。']
    (HERE / 'HYPERPARAM_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: report[k] for k in ['new_completed', 'new_total', 'reused_unique_completed', 'grid_complete']}))


if __name__ == '__main__':
    main()
