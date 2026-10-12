import json,hashlib,statistics,math,subprocess
from pathlib import Path
OUT=Path('/data/140-1/users/liying/O-LoRA/refine-logs/finegrained_std_20260922')
ROOT=Path('/data/140-1/users/liying/CopyFlatCV/refine-logs')
DATASETS=['aircraft','cars196','cub200','flowers','het5']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(p):
 o=json.loads(Path(p).read_text()); mats=o['cnn']['matrices']; key=max(mats); mat=mats[key]; n=len(mat)
 assert all(len(row)==n for row in mat)
 assert all(math.isfinite(mat[i][j]) for i in range(n) for j in range(i+1))
 stage=[statistics.mean(mat[i][:i+1]) for i in range(n)]
 return {'faa':stage[-1],'aaa_stage':statistics.mean(stage),'tasks':n,'stage_means':stage,'source':str(p),'source_sha256':sha(p)}
rows=json.loads(Path('/data/140-1/users/liying/O-LoRA/refine-logs/cllora_comparison_20260918/results.json').read_text())
selected=[]
for r in rows:
 if r.get('method') not in ['CL-LoRA','GAM-EWCLoRA'] or r['dataset'] not in DATASETS:continue
 m=metrics(r['source']);assert abs(m['faa']-r['faa'])<1e-9;assert abs(m['aaa_stage']-r['aaa_stage'])<1e-9;assert m['source_sha256']==r['sha256']
 x={'method':r['method'],'dataset':r['dataset'],'seed':0,'n':1,'faa':m['faa'],'aaa_stage':m['aaa_stage'],'faa_sd':None,'aaa_stage_sd':None,'faa_variance':None,'aaa_stage_variance':None,'sd_reason':'Only one verified seed in the current table protocol','runs':[m],'config_path':r.get('config_path'),'recorded_params':r['params']}
 selected.append(x)
p=next((ROOT/'gam-ewc-fine-gapfill-20260918-attempt2/outputs').rglob('gam_ewc_het5_s0_vit_base_patch16_224_cl_metrics.json'))
m=metrics(p);rpath=ROOT/'gam-ewc-fine-gapfill-20260918-attempt2/receipts/gam_ewc_het5_s0.json';rec=json.loads(rpath.read_text())
assert m['tasks']==5 and rec['status']=='completed' and rec['exit_code']==0
selected.append({'method':'GAM-EWCLoRA','dataset':'het5','seed':0,'n':1,'faa':m['faa'],'aaa_stage':m['aaa_stage'],'faa_sd':None,'aaa_stage_sd':None,'faa_variance':None,'aaa_stage_variance':None,'sd_reason':'Only one verified seed in the current table protocol','runs':[m],'receipt':str(rpath),'receipt_sha256':sha(rpath)})
separate=[]
for d in ['cub200','flowers']:
 runs=[];configs=[]
 for seed in [1,2,3]:
  root=ROOT/('gam-ewc-gapfill-launch-20260917' if seed<3 else 'gam-ewc-main-seed3-20260917')
  name=f'{d}_wrapper_lam2000_s{seed}_run'
  cfgpath=root/'configs'/f'{name}.json';cfg=json.loads(cfgpath.read_text());configs.append(cfg)
  recpath=root/('attempt2/receipts' if seed<3 else 'receipts')/f'{name}.json';rec=json.loads(recpath.read_text())
  assert rec['status']=='completed' and rec['exit_code']==0 and len(rec['completed_tasks'])==10
  assert rec['config_sha256']==sha(cfgpath)
  paths=list((root/'outputs').rglob(f'*{name}_vit_base_patch16_224_cl_metrics.json'));assert len(paths)==1
  m=metrics(paths[0]);assert m['tasks']==10
  m.update(seed=seed,config=str(cfgpath),config_sha256=sha(cfgpath),receipt=str(recpath),receipt_sha256=sha(recpath),source_snapshot_sha256=rec['snapshot_sha256'],source_root=rec['source_root']);runs.append(m)
 ignored=['seed','loader_seed','prefix','outputs_root']
 protocols=[{k:v for k,v in c.items() if k not in ignored} for c in configs]
 assert protocols[0]==protocols[1]==protocols[2]
 assert len(set(r['source_snapshot_sha256'] for r in runs))==1
 result={'method':'GAM-EWCLoRA (RNG-controlled campaign)','dataset':d,'n':3,'seeds':[1,2,3],'runs':runs,'same_configuration_except':ignored,'same_source_snapshot':True,'protocol':protocols[0],'protocol_sha256':hashlib.sha256(json.dumps(protocols[0],sort_keys=True).encode()).hexdigest(),'eligible_for_current_table':False,'reason':'Current table is historical seed0, without documented clean/Fisher RNG isolation and deterministic loader settings; controlled campaign explicitly sets all three true. Historical Flowers also lacks GAM rho/norm_rho/gamma in logged config. Do not pool historical seed0 or attach this SD to its single-run result.'}
 for metric in ['faa','aaa_stage']:
  vals=[r[metric] for r in runs];result[metric]=statistics.mean(vals);result[metric+'_sd']=statistics.stdev(vals);result[metric+'_variance']=statistics.variance(vals)
 separate.append(result)
paths=json.loads(Path('/tmp/cl_gam_candidate_paths.json').read_text())
report={'audit_date':'2026-09-22','metric_definition':'CNN task-macro FAA and mean of task-macro stage accuracies AAA_stage; SD and variance across independent seeds, ddof=1; null for n=1.','search_roots':['/data/140-1/users/liying/CopyFlatCV','/data/140-1/users/liying/Flatness_CV'],'search_method':'rg --files -uu, filter cllora/normfisher/gam_ewc and requested fine-grained datasets; inspect matching raw metrics, configs, receipts and previous full workspace inventory.','candidate_metric_paths':paths,'current_table_groups':selected,'separate_protocol_groups':separate,'excluded_categories':['smoke outputs','copied evidence and duplicate seed0 paths','pure GAM','lambda0 GAM wrappers','different lambda trials','as1_normfisher_fishershape/FS-EWC/SAM/ratioFS campaigns','low-rank-cl Adam rank10 batch64/in21k model campaigns']}
(OUT/'cl_gam.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
lines=['# CL-LoRA / GAM-EWCLoRA 多种子审计','', '当前表的 CL-LoRA 和 GAM-EWCLoRA 各五个数据集均只有一个已核实同协议 seed，因此 SD 和样本方差均不可估计，不能填 0。','', '| 方法 | 数据集 | n | FAA | AAA_stage | SD |','|---|---|---:|---:|---:|---|']
for r in sorted(selected,key=lambda r:(r['method'],DATASETS.index(r['dataset']))):lines.append(f"| {r['method']} | {r['dataset']} | 1 | {r['faa']:.6f} | {r['aaa_stage']:.6f} | -- |")
lines+=['','另找到 RNG 控制协议的 GAM-EWCLoRA CUB/Flowers，三次运行只改变 seed、loader_seed、prefix、outputs_root，源代码快照相同，均完整结束。可以单独报告如下统计，不能给原表历史 seed0 的 74.22/76.47 或 86.75/91.76 添加这些 SD：','','| 新协议数据集 | seeds | FAA mean (sample SD) | AAA_stage mean (sample SD) |','|---|---|---:|---:|']
for r in separate:lines.append(f"| {r['dataset']} | 1,2,3 | {r['faa']:.6f} ({r['faa_sd']:.6f}) | {r['aaa_stage']:.6f} ({r['aaa_stage_sd']:.6f}) |")
lines+=['','协议差异：新 campaign 显式启用 `ewc_preserve_fisher_rng=true`、`ewc_isolate_clean_rng=true`、`deterministic_loader_rng=true`。历史 seed0 未固化这些设置；Flowers 还未记录 GAM rho/norm_rho/gamma，因此无证据证明只变 seed。','', 'CL-LoRA 搜索只找到正式 seed0 和 smoke，烟雾测试不能计为复现实验。未找到 Sep19 以后的新细粒度 CL/GAM-EWC 正式多 seed 结果。GAM-Het5 的 83.37/81.1556666667 已从 5×5 CNN 矩阵重新计算；不同方法、超参 sweep 与副本全部排除。','', '所有原始 metrics SHA256、配置及 receipt SHA256、独立种子统计保存在 `cl_gam.json`。']
(OUT/'cl_gam.md').write_text('\n'.join(lines)+'\n')
for r in separate:print(r['dataset'],r['faa'],r['faa_sd'],r['aaa_stage'],r['aaa_stage_sd'])
print('current groups',len(selected))
