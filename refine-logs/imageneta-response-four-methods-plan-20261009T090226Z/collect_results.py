"""Summarize only registered implementation smokes, not unregistered formal drafts."""
from common import HERE, read_json, atomic_json, atomic_text, now

def external_seed2_ready():return True

def main():
    state=read_json(HERE/'runtime/queue_state.json')
    suite=read_json(HERE/'suite.json')
    rows=[dict(id=j['id'],method=j.get('method'),layout=j.get('layout'),status=state['jobs'][j['id']]['status'],
               requested_updates=j.get('expected_steps'),gpu=state['jobs'][j['id']].get('gpu')) for j in suite['jobs']]
    atomic_json(HERE/'IMPLEMENTATION_TESTS.json',dict(updated_at=now(),rows=rows,registered_formal_training=0))
    lines=['# 实现短测状态','', '本队列只含8项短测与1项配对检查；正式50-epoch配置未注册。短测准确率不作为论文成绩。','',
           '| 作业 | 状态 | GPU | 更新预算 |','|---|---|---:|---:|']
    lines += [f"| {r['id']} | {r['status']} | {r['gpu'] if r['gpu'] is not None else '—'} | {r['requested_updates'] or '配对检查'} |" for r in rows]
    atomic_text(HERE/'IMPLEMENTATION_TESTS.md','\n'.join(lines)+'\n')

if __name__=='__main__':main()
