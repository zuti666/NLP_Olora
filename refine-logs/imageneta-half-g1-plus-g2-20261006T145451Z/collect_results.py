"""Compare accepted halfsum12 against frozen completed sum12 and sum12-minus-g0 controls."""
import csv,io,json,statistics
from pathlib import Path
from common import atomic_json,atomic_text,now,sha256
R=Path(__file__).resolve().parent

def controls():
 old=json.loads((R/'historical_inventory.json').read_text())['reused_controls']
 for x in old:
  if sha256(x['source_summary'])!=x['source_summary_sha256'] or sha256(x['source_matrix'])!=x['source_matrix_sha256']:raise RuntimeError('Historical control changed')
 return old

def main():
 s=json.loads((R/'suite.json').read_text());old=controls();p=R/'runtime/queue_state.json'
 state=json.loads(p.read_text()) if p.exists() else {'jobs':{}};rows=[]
 for j in s['jobs']:
  if j['kind']!='training' or j['smoke']:continue
  out=Path(j['output_dir']);p=out/'task_matrix.json';m=json.loads(p.read_text()) if p.exists() else {}
  status=state['jobs'].get(j['id'],{}).get('status','pending');accepted=status=='completed' and len(m.get('matrix',[]))==10
  ref=next(x for x in old if x['radius']==j['radius'] and x['direction']=='g1_plus_g2')
  sub=next(x for x in old if x['radius']==j['radius'] and x['direction']=='g1_plus_g2_minus_g0')
  g2=next(x for x in old if x['radius']==j['radius'] and x['direction']=='g2')
  trace=[json.loads(x) for x in (out/'step_trace.jsonl').read_text().splitlines()] if accepted else []
  ratios=[x['norms']['U']/x['norms']['g2'] for x in trace if x['norms']['g2']>0]
  cos=[x['cosines']['U_g0'] for x in trace if x['cosines']['U_g0'] is not None]
  row=dict(job=j['id'],direction=j['direction'],branch='P',radius=j['radius'],seed=2,mode='natural',status=status,tasks=len(m.get('matrix',[])),final_metrics_accepted=accepted,
   AAA=m.get('AAA_task_macro') if accepted else None,FAA=m.get('FAA_task_macro') if accepted else None,LA=m.get('LA') if accepted else None,forgetting=m.get('forgetting') if accepted else None,BWT=m.get('BWT') if accepted else None,
   progress_AAA=m.get('AAA_task_macro'),progress_FAA=m.get('FAA_task_macro'),median_update_over_g2_norm=statistics.median(ratios) if ratios else None,median_update_g0_cosine=statistics.median(cos) if cos else None,output_dir=str(out))
  for label,refrow in [('sum12',ref),('sum12minus0',sub),('g2',g2)]:
   row[label+'_reference_AAA']=refrow['AAA_task_macro'];row[label+'_reference_FAA']=refrow['FAA_task_macro']
   row['delta_vs_'+label+'_AAA']=m['AAA_task_macro']-refrow['AAA_task_macro'] if accepted else None
   row['delta_vs_'+label+'_FAA']=m['FAA_task_macro']-refrow['FAA_task_macro'] if accepted else None
  rows.append(row)
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic_text(R/'summary.csv',buf.getvalue());atomic_json(R/'results.json',dict(updated_at=now(),new_rows=rows,reused_controls=old))
 f=lambda v:'—' if v is None else f'{v:.3f}'
 lines=['# ImageNet-A (g1+g2)/2','',
  '固定seed2/rank10/rho=.05，r=.25/1，正向P探针。g3仍计算但不参与U；4CE/1自然SGD，仅最终梯度乘.5；无投影/额外最终归一化或匹配，无Fisher/EWC/融合/HVP。','',
  '| r | 状态 | 完成任务 | 最终AAA | 最终FAA | 相对g1+g2 AAA/FAA差值 |','|---:|---|---:|---:|---:|---:|']
 for x in rows:lines.append(f"| {x['radius']} | {x['status']} | {x['tasks']} | {f(x['AAA'])} | {f(x['FAA'])} | {f(x['delta_vs_sum12_AAA'])}/{f(x['delta_vs_sum12_FAA'])} |")
 lines+=['','已验证完成的历史参照，任务等权%，直接复用：','','| 方向 | r | AAA | FAA |','|---|---:|---:|---:|']
 for x in old:lines.append(f"| {x['direction']} | {x['radius']} | {f(x['AAA_task_macro'])} | {f(x['FAA_task_macro'])} |")
 lines+=['','在同一参数/批次的梯度状态，新U为旧g1+g2的0.5倍，名义学习率不变。共同状态下仅改变更新幅度；独立训练轨迹会分开，后续探针梯度可不同；单seed不能证明纯方向或Hessian机制。完整10任务/500epochs/2550updates且exit0/收据/config/trace/matrix验收后才填最终成绩；部分任务指标仅是进度。']
 atomic_text(R/'RESULTS.md','\n'.join(lines)+'\n')
if __name__=='__main__':main()
