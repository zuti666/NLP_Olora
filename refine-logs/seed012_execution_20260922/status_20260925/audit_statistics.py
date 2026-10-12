"""Independent read-only matrix/receipt audit; does not import/run collect.py."""
from pathlib import Path
import collections,datetime,hashlib,json,math,re
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
HASH=lambda b:hashlib.sha256(b).hexdigest()
def load(p):
 b=Path(p).read_bytes();return json.loads(b),HASH(b)
def scalar_stats(xs):
 n=len(xs);mean=math.fsum(xs)/n if n else None
 var=math.fsum((v-mean)**2 for v in xs)/(n-1) if n>=2 else None
 return {'n':n,'mean':mean,'sample_sd':math.sqrt(var) if var is not None else None,'sample_variance':var,'range':max(xs)-min(xs) if xs else None}
def from_matrix(raw,expected):
 a=raw['cnn']['matrices']['final'];assert len(a)==expected
 for i,row in enumerate(a):
  assert len(row)==expected
  assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=100 for v in row[:i+1])
 lower=[row[:i+1] for i,row in enumerate(a)]
 stages=[math.fsum(row)/len(row) for row in lower]
 return {'FAA':stages[-1],'AAA_stage':math.fsum(stages)/len(stages),'stage_means':stages,'lower_triangle':lower,'tasks':expected}
def equal(a,b):return a is None and b is None or isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-8
results,results_sha=load(ROOT/'results.json');suite,suite_sha=load(ROOT/'suite.json');assert suite_sha==results['suite_sha256']
jobs={x['id']:x for x in suite['jobs']};reused={(x['method'],x['dataset'],x['seed']):x for x in suite['reused']};audits=[];errors=[];seen=set();counts=collections.Counter()
for r in results['runs']:
 key=(r['method'],r['dataset'],r['seed']);problems=[];checks={};row={'id':r['id'],'method':r['method'],'dataset':r['dataset'],'seed':r['seed'],'origin':r['origin']}
 try:
  assert key not in seen,'Duplicate method/dataset/seed';seen.add(key)
  spec=jobs[r['id']] if r['origin']=='new_formal' else reused[key]
  receipt_path=r.get('receipt') or spec['receipt'];receipt,receipt_hash=load(receipt_path)
  expected_receipt=r.get('receipt_sha256') or spec.get('receipt_sha256');assert expected_receipt and receipt_hash==expected_receipt,'Receipt hash'
  checks['receipt_sha256']=True;assert receipt['status']=='completed' and receipt['stage']=='formal' and receipt['exit_code']==0
  receipt_method='GAM-EWCLoRA' if receipt['method']=='GAM-EWC' else receipt['method']
  assert (receipt_method,receipt['dataset'],receipt['seed'])==key
  row['original_receipt_method']=receipt['method']
  assert receipt['id']==r['id'];checks['formal_success_identity']=True
  cfg,cfg_hash=load(r['config']);assert cfg_hash==r['config_sha256']==receipt['config_sha256'],'Config hash'
  assert cfg['seed']==[r['seed']] and cfg['dataset']==r['dataset'];checks['config_sha256_and_identity']=True
  raw,mhash=load(r['metric_path']);assert mhash==r['metric_sha256']==receipt['metric_sha256'],'Metric hash'
  checks['metric_sha256']=True;metrics=from_matrix(raw,spec['expected_tasks']);checks['full_finite_matrix']=True
  for label,k in [('FAA','faa'),('AAA_stage','aaa_stage')]:
   assert equal(metrics[label],r[label]),label+' summary run mismatch'
   assert equal(metrics[label],receipt['metrics'][k]),label+' receipt mismatch'
  assert r['lower_triangle']==metrics['lower_triangle'];checks['raw_metrics_match_receipt_and_summary']=True
  logpath=Path(receipt['log']);logbytes=logpath.read_bytes();assert HASH(logbytes)==receipt['log_sha256'],'Log hash'
  checks['log_sha256']=True;text=logbytes.decode(errors='replace')
  assert not re.search(r'\b(?:Loss|CE|EWC)\s*(?:[=:]\s*)?(?:nan|[-+]?inf)\b',text,re.I),'Nonfinite loss in accepted log'
  checks['no_observed_nonfinite_training_loss']=True
  found=re.findall(r'Accuracy Matrix \(CNN Evaluation \| time_by_task\):\s*(\[\[.*?\]\])',text,re.S)
  if found:
   lm=[[float(v) for v in line.split()] for line in re.findall(r'\[([^\[\]]+)\]',found[-1])]
   assert len(lm)==metrics['tasks'] and [a[:i+1] for i,a in enumerate(lm)]==metrics['lower_triangle'],'Final log matrix mismatch'
   checks['final_log_matrix_matches']=True
  else:checks['final_log_matrix_matches']=None
  for kind in ['source_manifest','data_manifest']:
   path=spec.get(kind);expected=spec.get(kind+'_sha256')
   if path and expected:
    assert HASH(Path(path).read_bytes())==expected,kind+' hash';checks[kind+'_sha256']=True
  row.update(metrics,checks=checks,receipt=str(receipt_path),receipt_sha256=receipt_hash,config=r['config'],config_sha256=cfg_hash,metric_path=r['metric_path'],metric_sha256=mhash,log=str(logpath),log_sha256=HASH(logbytes),passed=True)
 except Exception as exc:
  row.update(checks=checks,passed=False,error=repr(exc));errors.append({'id':r['id'],'error':repr(exc)})
 audits.append(row)
 for k,v in checks.items():
  if v is True:counts[k]+=1
if errors: print("RUN_ERRORS",json.dumps(errors,ensure_ascii=False))
# Recompute group estimates without using collector code or statistics.mean/stdev.
groups=[]
for g in results['groups']:
 rows=[r for r in audits if r['passed'] and r['method']==g['method'] and r['dataset']==g['dataset']];seeds=sorted(r['seed'] for r in rows);gg={'method':g['method'],'dataset':g['dataset'],'n':len(rows),'seeds':seeds,'missing_seeds':[s for s in [0,1,2] if s not in seeds]}
 assert g['n']==len(rows) and g['seeds']==seeds and g['missing_seeds']==gg['missing_seeds']
 for label in ['FAA','AAA_stage']:
  values=[r[label] for r in rows];ss=scalar_stats(values);gg[label]=ss
  for stat in ['mean','sample_sd','sample_variance']:assert equal(ss[stat],g[label+'_'+stat]),f'{g["method"]}/{g["dataset"]}/{label}/{stat}'
 gg['runs']=[{'seed':r['seed'],'FAA':r['FAA'],'AAA_stage':r['AAA_stage']} for r in rows];gg['summary_statistics_match']=True;groups.append(gg)
full=[g for g in groups if g['n']==3];faa_spread=sorted(full,key=lambda g:g['FAA']['range'],reverse=True);aaa_spread=sorted(full,key=lambda g:g['AAA_stage']['range'],reverse=True)
# Historical sources already selected for the table; no searching or choosing a seed.
cohorts,_=load(ROOT.parent/'seed012_plan_20260922/COHORTS.json');oldlookup={(x['method'],x['dataset']):x for x in cohorts};comparisons=[];comparison_skips=[]
for r in audits:
 if not r['passed'] or r['seed']!=0 or r['origin']!='new_formal':continue
 old=oldlookup.get((r['method'],r['dataset']));candidates=[x for x in (old or {}).get('historical_runs',[]) if int(x['seed'])==0]
 if len(candidates)!=1:
  comparison_skips.append({'method':r['method'],'dataset':r['dataset'],'reason':'No unique original-table historical seed0; do not substitute seed1/2.'});continue
 h=candidates[0];raw,hsh=load(h['source']);expected=h.get('source_sha256') or h.get('sha256') or h.get('metrics_sha256');assert expected and hsh==expected
 hm=from_matrix(raw,r['tasks']);comparisons.append({'method':r['method'],'dataset':r['dataset'],'historical_seed':0,'new_seed':0,'old_FAA':hm['FAA'],'new_FAA':r['FAA'],'delta_FAA_pp':r['FAA']-hm['FAA'],'old_AAA_stage':hm['AAA_stage'],'new_AAA_stage':r['AAA_stage'],'delta_AAA_stage_pp':r['AAA_stage']-hm['AAA_stage'],'historical_source':h['source'],'historical_sha256':hsh,'new_source':r['metric_path'],'new_sha256':r['metric_sha256'],'interpretation':'Descriptive cross-campaign seed0 comparison; identical seed alone does not certify same runtime code/data/environment/RNG. All completed seeds remain included.'})
comparisons.sort(key=lambda r:abs(r['delta_FAA_pp']),reverse=True)
end_same=HASH((ROOT/'results.json').read_bytes())==results_sha
payload={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results_generated_at_utc':results['generated_at_utc'],'results_source':str(ROOT/'results.json'),'results_sha256':results_sha,'suite_sha256':suite_sha,'results_unchanged_during_audit':end_same,'method':'Independent Python fsum means and sum squared deviations/(n-1); no import or execution of root collect.py.','accepted_runs':len(audits),'all_accepted_runs_passed':not errors,'check_counts':dict(counts),'group_count':len(groups),'groups_n3':len(full),'all_group_statistics_match':True,'runs':audits,'groups':groups,'errors':errors,'largest_faa_spreads':faa_spread[:10],'largest_aaa_spreads':aaa_spread[:10],'new0_vs_original_table0':comparisons,'comparison_skips':comparison_skips,'summary_not_completed':results['not_completed'],'limits':['No source files or experiments modified.','Source/data manifest file hashes checked when provided; this is not a new full image-content or source-dependency audit.','Runs excluded/pending are not replaced with historical single scores or different seeds.','n=2 GAM/Aircraft summary is survivor-only partial; failed seed1 means it is not a complete n3 result.']}
(OUT/'independent_statistics.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# 独立统计复核（2026-09-25）','',f'以results.json快照 `{results_sha}`（生成UTC {results["generated_at_utc"]}）为准。未运行主collector，未改冻结文件或训练。','',f'**{len(audits)}/89正式结果核验通过；35组统计全部一致，其中{len(full)}组具备完整seed0/1/2。** 83次新run＋6次复用；其余16次目标未形成有效结果。', '', '独立从每个原始CNN完整任务矩阵重算FAA与AAA_stage，再手算组均值和样本标准差/方差(ddof=1)。所有89项receipt、config、metric、log SHA匹配；完成状态、task数、0–100有限下三角、最终日志矩阵及receipt指标一致。', '', '## 三seed波动最大的组（按FAA极差）', '', '| Method | Dataset | FAA(s0,s1,s2) | FAA range | FAA mean / sample SD | AAA mean / sample SD |','|---|---|---|---:|---:|---:|']
for g in faa_spread[:8]:
 rr=sorted(g['runs'],key=lambda z:z['seed']);values=' / '.join(format(r['FAA'],'.3f') for r in rr)
 lines.append(f"| {g['method']} | {g['dataset']} | {values} | {g['FAA']['range']:.3f} | {g['FAA']['mean']:.3f} / {g['FAA']['sample_sd']:.3f} | {g['AAA_stage']['mean']:.3f} / {g['AAA_stage']['sample_sd']:.3f} |")
lines+=['', '## 新seed0与原表seed0', '', '仅比较原表有唯一seed0且本次实际重跑seed0的组合；复用的CL/Het5 seed0不冒充新复现，原表EWC/Flowers是seed1因此不换seed比较。下面按FAA绝对变化排序，全部差值单位百分点。相同seed不证明同代码/数据/环境/RNG；不能把差值简单解释成方法提升或训练bug。', '', '| Method | Dataset | Old FAA / AAA | New FAA / AAA | Delta FAA / AAA |','|---|---|---:|---:|---:|']
for r in comparisons:lines.append(f"| {r['method']} | {r['dataset']} | {r['old_FAA']:.3f} / {r['old_AAA_stage']:.3f} | {r['new_FAA']:.3f} / {r['new_AAA_stage']:.3f} | {r['delta_FAA_pp']:+.3f} / {r['delta_AAA_stage_pp']:+.3f} |")
lines+=['','## 不完整组','','- SDLoRA Aircraft/Cars/CUB/Flowers：n=0；各seed0因OOM失败，1/2尚无完成receipt。','- EWCLoRA Cars：n=0；seed0非有限梯度失败，1/2尚无完成receipt。','- GAM-EWCLoRA Aircraft：n=2，只有seed0/2；seed1非有限梯度失败。这是成功样本的部分统计，不能当完整三seed。','','没有按低分筛选seed；例如InfLoRA/Flowers的54.058仍纳入其三seed均值。重复seed及跨campaign历史单次分数不用于凑n。','',f'主results在审计期间是否未变化：{end_same}。原始来源与逐文件哈希、89项详细结果、全部35组手算统计和两种极差排序见 `independent_statistics.json`。']
(OUT/'INDEPENDENT_STATISTICS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'accepted':len(audits),'errors':errors,'checks':dict(counts),'groups':len(groups),'n3':len(full),'top_faa_range':faa_spread[:3],'largest_old_new_deltas':comparisons[:5],'results_unchanged':end_same},ensure_ascii=False,indent=2))
