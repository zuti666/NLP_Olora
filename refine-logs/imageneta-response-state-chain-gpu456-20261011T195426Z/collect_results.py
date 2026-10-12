"""Read accepted branch endpoints only; preserve partial results as explicitly partial."""
from pathlib import Path
import csv,io,json,statistics
from common import atomic_json,atomic_text,load_suite
from aggregate_helpers import fit_match,mean_sd
H=Path(__file__).resolve().parent

def main():
 suite=load_suite();state=json.loads((H/'runtime/queue_state.json').read_text()) if (H/'runtime/queue_state.json').exists() else {'jobs':{}}
 rows=[];curves=[];fitrows=[];tails=[];factorrows=[]
 for j in suite['jobs']:
  if not j.get('formal') or state['jobs'].get(j['id'],{}).get('status')!='completed':continue
  s=json.loads((Path(j['output_dir'])/'final_summary.json').read_text())
  for epoch in [25,50]:
   r=next(x for x in s['retention_curve'] if x['epoch']==epoch)
   old=r['old_test_allseen40'];new=r['current_test_allseen40'];start=s['retention_curve'][0]['old_test_allseen40']
   row=dict(run=j['id'],seed=j['seed'],branch=j['chain_branch'],epoch=epoch,old_accuracy=old['accuracy'],new_accuracy=new['accuracy'],FAA=(old['accuracy']+new['accuracy'])/2,old_decline=start['accuracy']-old['accuracy'],old_ce_increase=old['ce']-start['ce'],old_ce=old['ce'],new_ce=new['ce'])
   rows.append(row)
  for r in s['retention_curve']:
   curves.append(dict(run=j['id'],seed=j['seed'],branch=j['chain_branch'],**r))
   if r['epoch']:
    fitrows.append(dict(run=j['id'],seed=j['seed'],branch=j['chain_branch'],epoch=r['epoch'],clean_training_ce=r['clean_training']['ce'],clean_training_accuracy=r['clean_training']['accuracy'],old_accuracy=r['old_test_allseen40']['accuracy'],new_accuracy=r['current_test_allseen40']['accuracy'],new_conditional20_ce=r['current_test_conditional20']['ce']))
  e25,e50=[r for r in rows if r['run']==j['id']]
  tails.append(dict(run=j['id'],seed=j['seed'],branch=j['chain_branch'],**{k+'_50_minus_25':e50[k]-e25[k] for k in ['old_accuracy','new_accuracy','FAA','old_decline']}))
  factors=json.loads((Path(j['output_dir'])/'factor_state_curve.json').read_text())['epochs']
  for f in factors:
   for layer in f['layers']:factorrows.append(dict(run=j['id'],seed=j['seed'],branch=j['chain_branch'],epoch=f['epoch'],clean_training_ce=f['clean_training_ce'],**layer))
 def csvwrite(name,values):
  f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(values[0]) if values else ['run','seed','branch','epoch','old_accuracy','new_accuracy','FAA','old_decline','old_ce_increase','old_ce','new_ce']);w.writeheader();w.writerows(values);atomic_text(H/name,f.getvalue())
 csvwrite('FORMAL_RESULTS.csv',rows)
 atomic_json(H/'RETENTION_CURVES.json',curves)
 for name,values in [('FIT_CURVES.csv',fitrows),('FIT_MATCHED_COMPARISONS.csv',fit_match(fitrows)),('TAIL_CHANGES.csv',tails),('LORA_FACTOR_ALL_LAYERS.csv',factorrows)]:csvwrite(name,values)
 accepted=sum(state['jobs'].get(j['id'],{}).get('status')=='completed' for j in suite['jobs'] if j.get('formal'))
 txt=['# 三层证据链实时结果','',f'正式Task1分支已验收：{accepted}/6；仅完成分支进入下表。seed0/1/2配对，主时间点e25/e50。','', '| Branch | Seed | Epoch | Old (%) | New (%) | FAA (%) | Old decline (pp) |','|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:txt.append(f"| {r['branch']} | {r['seed']} | {r['epoch']} | {r['old_accuracy']:.3f} | {r['new_accuracy']:.3f} | {r['FAA']:.3f} | {r['old_decline']:.3f} |")
 contrasts=[]
 for ep in [25,50]:
  paired={}
  for r in rows:
   if r['epoch']==ep:paired.setdefault(r['seed'],{})[r['branch']]=r
  for seed,rr in paired.items():
   if set(rr)=={'S','R'}:contrasts.append(dict(seed=seed,epoch=ep,**{k:rr['R'][k]-rr['S'][k] for k in ['old_accuracy','new_accuracy','FAA','old_decline','old_ce_increase','new_ce']}))
 atomic_json(H/'PAIRED_DIFFERENCES.json',contrasts)
 paired_summary=[]
 for ep in [25,50]:
  rr=[r for r in contrasts if r['epoch']==ep]
  if rr:paired_summary.append(dict(epoch=ep,paired_seeds=[r['seed'] for r in rr],**{k:mean_sd([r[k] for r in rr]) for k in ['old_accuracy','new_accuracy','FAA','old_decline','old_ce_increase','new_ce']}))
 atomic_json(H/'PAIRED_SUMMARY.json',paired_summary)
 if accepted==6:
  txt+=['','三seed均值±样本标准差：','', '| Branch | Epoch | Old (%) | New (%) | FAA (%) | Old decline (pp) |','|---|---:|---:|---:|---:|---:|']
  for branch in ['S','R']:
   for ep in [25,50]:
    rr=[r for r in rows if r['branch']==branch and r['epoch']==ep]
    fields=[f"{statistics.mean(r[k] for r in rr):.3f}±{statistics.stdev(r[k] for r in rr):.3f}" for k in ['old_accuracy','new_accuracy','FAA','old_decline']]
    txt.append('| '+ ' | '.join([branch,str(ep)]+fields)+' |')
 if accepted==6:
  txt+=['','R−S配对差（均值±样本SD，3独立seed）：','', '| Epoch | Old (pp) | New (pp) | FAA (pp) | Old decline (pp) |','|---|---:|---:|---:|---:|']
  for group in paired_summary:
   txt.append('| '+ ' | '.join([str(group['epoch'])]+[f"{group[k]['mean']:.3f}±{group[k]['sample_sd']:.3f}" for k in ['old_accuracy','new_accuracy','FAA','old_decline']])+' |')
 txt+=['','TAIL_CHANGES.csv记录e50−e25；FIT_MATCHED_COMPARISONS.csv只比较首次进入同一.05宽cleanCE箱的记录；所有24层均在LORA_FACTOR_ALL_LAYERS.csv。缺少共同拟合箱时不外推。']
 atomic_text(H/'RESULTS.md','\n'.join(txt)+'\n')
 print(json.dumps({'accepted_formal':accepted,'endpoint_rows':len(rows)}))
if __name__=='__main__':main()
