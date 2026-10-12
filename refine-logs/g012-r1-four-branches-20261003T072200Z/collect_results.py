"""Validated full10 scores only; historical baselines retain their provenance."""
import csv
import io
import json
from pathlib import Path
from common import HERE, atomic_json, atomic_text, load_suite, now, read_json
from experiment_contract import verify_reuse
from validate_job import validate_job


def matrix_metrics(matrix):
    n = len(matrix)
    if n != 10 or any(len(row) != i + 1 for i, row in enumerate(matrix)):
        raise RuntimeError('A complete ten-task matrix is required')
    mean = lambda values: sum(values) / len(values)
    return {'AAA': mean([mean(row) for row in matrix]), 'FAA': mean(matrix[-1]),
            'LA': mean([matrix[i][i] for i in range(n)]),
            'BWT': mean([matrix[-1][i] - matrix[i][i] for i in range(n - 1)]),
            'forgetting': mean([max(matrix[t][i] for t in range(i, n)) - matrix[-1][i]
                                for i in range(n - 1)])}


def main():
    suite = load_suite()
    state_path = HERE / 'runtime/queue_state.json'
    state = read_json(state_path) if state_path.exists() else {'jobs': {}}
    rows = []
    for job in suite['jobs']:
        if job['smoke']:
            continue
        lifecycle = state['jobs'].get(job['id'], {'status': 'pending'})
        status = lifecycle['status']
        row = dict(job=job['id'], branch=job['branch'], direction=job['direction'],
                   mode=job['mode'], seed=2, radius=1., status=status,
                   source='new_r1_full10', tasks=0, AAA=None, FAA=None, LA=None,
                   BWT=None, forgetting=None, gpu=lifecycle.get('gpu'),
                   output_dir=job['output_dir'])
        output = Path(job['output_dir'])
        if (output / 'task_matrix.json').exists():
            matrix = read_json(output / 'task_matrix.json')['matrix']
            row['tasks'] = len(matrix)
        if status == 'completed':
            validate_job(job)
            row.update(matrix_metrics(matrix))
        rows.append(row)
    report = verify_reuse(suite)
    for old in report['jobs']:
        matrix = read_json(old['matrix'])['matrix']
        rows.append(dict(job=old['id'], branch=old['branch'], direction=old['direction'],
                         mode=old['mode'], seed=2, radius='independent_of_r',
                         status='validated_reuse', source='old_four_probe_baseline_r025_trace',
                         tasks=10, **matrix_metrics(matrix), gpu=None,
                         output_dir=old['output_dir']))
    rows.sort(key=lambda r: r['job'])
    updated = now()
    atomic_json(HERE / 'results.json', {'updated_at': updated, 'rows': rows})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(HERE / 'summary.csv', stream.getvalue())
    fmt = lambda v: '—' if v is None else f'{v:.3f}'
    lines = ['# ImageNet-A：r=1 正反探针与差分符号', '', '更新：' + updated, '',
             'seed2、rank10、rho=.05，10任务×50 epochs；新完整训练12项，半径无关基准复用4项。',
             '匹配参考为候选自身当前状态的raw g1+实际BA＋全部已见FC位移。',
             '未验收完整训练的成绩留空；历史基准的r=.25外点诊断不能用于r=1机制统计。', '',
             '| 条件 | 步长 | 状态 | 已完成任务 | AAA (%) | FAA (%) | GPU |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| ' + ' | '.join([row['job'], row['mode'], row['status'],
                     str(row['tasks']), fmt(row['AAA']), fmt(row['FAA']),
                     '—' if row['gpu'] is None else str(row['gpu'])]) + ' |')
    atomic_text(HERE / 'RESULTS.md', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
