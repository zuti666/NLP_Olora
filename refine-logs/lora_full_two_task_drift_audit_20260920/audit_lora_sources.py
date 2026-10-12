#!/usr/bin/env python3
"""Rebuild the six historical LoRA rows from their original log records (CPU only)."""
from pathlib import Path
import ast, csv, hashlib, json, math, statistics
ROOT=Path('/data/115-2/users/liying/low-rank-cl-new')
OUT=Path(__file__).resolve().parent
TAG='s012_rho0p05_normrho0p5_lam1e7'
REL=Path('verify_imagenet_r_1x1_table_metrics')/TAG
SRC=ROOT/'result_summary'/REL/'table_source.csv'
PAPER=Path('/data/140-1/users/liying/CopyFlatCV/newmethod_design_wiki/Making_EWC_Reliable__Stabilizing_Laplace_Posterior_Surrogates_for_Replay_Free_Continual_Learning (1)/sections/A_appendix.tex')
MAP={'baseline':'Vanilla LoRA','gam_only':'Shaping-only LoRA','control':'EWCLoRA','full':'Shaping + EWCLoRA','setting1':'EWCLoRA + Task-1 shaping','setting2':'EWCLoRA + Task-2 shaping'}
TARGET={'baseline':[90.73,19.885,84.85,89.59,-5.87,22.469,1.560],'gam_only':[93.34,10.946,84.80,93.81,-8.55,10.471,3.131],'control':[90.73,19.885,87.95,83.45,-2.78,24.599,.0106],'full':[93.34,10.946,89.10,88.33,-4.24,17.931,.0204],'setting1':[93.34,10.946,91.82,82.57,-1.52,18.734,.392],'setting2':[90.73,19.885,83.07,89.21,-7.66,41.316,.129]}
KEYS=['a11','F1','a21','a22','BWT','T2','r_eff']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def parse(p):
    a=[];m={};save=[]
    for i,l in enumerate(p.read_text().splitlines(),1):
        if 'save_ckp:' in l:save.append({'line':i,'text':l})
        if '=> Accuracy:' in l and 'Prototype' not in l:
            a.append({'line':i,'value':ast.literal_eval(l[l.index('{'):])})
        if '[Mechanism]' in l:
            d=ast.literal_eval(l[l.index('{'):]);m[str(d['task'])]={'line':i,'value':d}
    assert len(a)==2,(p,len(a))
    assert set(m)=={'0','1'},(p,m.keys())
    return {'accuracy':a,'mechanism':m,'checkpoint_config_lines':save}
rows=[];raw={};manifest={'source_repository':str(ROOT),'tag':TAG,'paper':ref(PAPER),'table_source_csv':ref(SRC),'generation_script':ref(ROOT/'scripts/run_verify_imagenet_r_1x1_table_metrics_s012.sh'),'runs':[],'summary':[]}
for idx,r in enumerate(csv.DictReader(SRC.open())):
    name=r['setting']
    if name not in MAP:continue
    p=Path(r['log_path']);cfgp=Path(r['config']);cfg=json.loads(cfgp.read_text());d=parse(p);uid=name+'_seed'+r['seed'];raw[uid]=d
    m0=d['mechanism']['0']['value'];m1=d['mechanism']['1']['value'];a0=d['accuracy'][0]['value'];a1=d['accuracy'][1]['value']
    fields={'a11':('R1_A_after_T1',a0['00-19'],f'/runs/{uid}/accuracy/0/value/00-19'), 'a21':('R2_A_after_T2',a1['00-19'],f'/runs/{uid}/accuracy/1/value/00-19'),'a22':('R2_B_after_T2',a1['20-39'],f'/runs/{uid}/accuracy/1/value/20-39'),'F1':('after_T1_fisher_total_mass',m0['endpoint_fisher_total_mass'],f'/runs/{uid}/mechanism/0/value/endpoint_fisher_total_mass'),'T2':('after_T2_fisher_total_mass',m1['endpoint_fisher_total_mass'],f'/runs/{uid}/mechanism/1/value/endpoint_fisher_total_mass'),'delta_norm_squared':('theta2_minus_theta1_norm_sq',m1['delta_norm'],f'/runs/{uid}/mechanism/1/value/delta_norm'),'F_accumulated_T2':('accumulated_fisher_total_mass',m1['accumulated_fisher_total_mass'],f'/runs/{uid}/mechanism/1/value/accumulated_fisher_total_mass')}
    fr={}
    for key,(col,v,pointer) in fields.items():
        assert abs(float(r[col])-v)<1e-8,(uid,key,r[col],v)
        fr[key]={'raw_value':v,'parsed_json_pointer':pointer,'csv_column':col,'csv_data_row_zero_based':idx}
    values={k:x[1] for k,x in fields.items()};values.update({'setting':name,'variant':MAP[name],'seed':int(r['seed']),'BWT':a1['00-19']-a0['00-19'],'r_eff':math.sqrt(m1['delta_norm'])})
    rows.append(values)
    manifest['runs'].append({'id':uid,'setting':name,'seed':int(r['seed']),'log':ref(p),'config':ref(cfgp),'config_values':cfg,'raw_log_line_references':{'a11':d['accuracy'][0]['line'],'a21_a22':d['accuracy'][1]['line'],'F1':d['mechanism']['0']['line'],'T2_delta_norm':d['mechanism']['1']['line']},'fields':fr,'derived':{'r_eff':values['r_eff'],'formula':'sqrt(mechanism[1].delta_norm)','BWT':values['BWT']},'checkpoint_save_flag':d['checkpoint_config_lines']})
for name in MAP:
    rr=[r for r in rows if r['setting']==name];means={k:statistics.mean(r[k] for r in rr) for k in KEYS};stds={k:statistics.stdev(r[k] for r in rr) for k in KEYS}
    decimals=[2,3,2,2,2,3,4 if name in {'control','full'} else 3]
    agree=[round(means[k],nd)==t for k,nd,t in zip(KEYS,decimals,TARGET[name])]
    assert all(agree),(name,means,TARGET[name])
    manifest['summary'].append({'setting':name,'variant':MAP[name],'seeds':[r['seed'] for r in rr],'mean':means,'sample_std_ddof1':stds,'reported':dict(zip(KEYS,TARGET[name])),'rounding_matches_all_fields':all(agree),'energy_mean':statistics.mean(r['delta_norm_squared'] for r in rr),'sqrt_mean_energy':math.sqrt(statistics.mean(r['delta_norm_squared'] for r in rr)),'accumulated_F_T2_mean':statistics.mean(r['F_accumulated_T2'] for r in rr)})
(OUT/'lora_raw_log_records.json').write_text(json.dumps({'runs':raw},indent=2,allow_nan=True)+'\n')
manifest['parsed_log_records']=ref(OUT/'lora_raw_log_records.json')
manifest['validation']={'runs_checked':len(rows),'every_selected_csv_value_matches_original_log':True,'all_42_paper_values_match_seed_means_after_rounding':True,'r_aggregation':'mean over seeds of sqrt(delta_norm), not sqrt of mean delta_norm','checkpoint_files_matching_experiment_tag':[],'limitations':['save_ckp=False in all original runs; no matching checkpoint files were found, so this is independent log/CSV recomputation, not tensor-level recomputation.','Current source files may have evolved after the May 2026 runs; logs provide runtime config but not a historical source commit hash.']}
(OUT/'lora_sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=True)+'\n')
with (OUT/'lora_recomputed_per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (OUT/'lora_recomputed_summary.csv').open('w') as f:
    data=[{'setting':x['setting'],**x['mean'],**{k+'_std':v for k,v in x['sample_std_ddof1'].items()},'energy_mean':x['energy_mean'],'sqrt_mean_energy':x['sqrt_mean_energy'],'accumulated_F_T2_mean':x['accumulated_F_T2_mean']} for x in manifest['summary']]
    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps({'runs':len(rows),'all_42_match':True,'output':str(OUT)},ensure_ascii=False))
