from pathlib import Path
import ast,csv,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parent
OLD=Path('/data/115-2/users/liying/low-rank-cl-new')
SOURCE=OLD/'result_summary/verify_imagenet_r_1x1_table_metrics/s012_rho0p05_normrho0p5_lam1e7/table_source.csv'
METHODS=['baseline','gam_only','control','full','setting1','setting2']
rows=[];evidence=[]
with SOURCE.open() as f: source=list(csv.DictReader(f))
for entry in source:
 if entry['setting'] not in METHODS: continue
 path=Path(entry['log_path']); records={};lns={}
 for lineno,line in enumerate(path.read_text(errors='replace').splitlines(),1):
  if '[Mechanism]' not in line:continue
  start=line.find('{')
  if start<0:continue
  rec=ast.literal_eval(line[start:])
  if 'task' in rec:records[int(rec['task'])]=rec;lns[int(rec['task'])]=lineno
 assert set(records)>={0,1},path
 r0,r1=records[0],records[1]
 raw_sq=float(r1['delta_norm']);radius=math.sqrt(raw_sq)
 mappings={'theta2_minus_theta1_norm_sq':raw_sq,'after_T1_fisher_total_mass':r0['endpoint_fisher_total_mass'],'after_T2_fisher_total_mass':r1['endpoint_fisher_total_mass'],'accumulated_fisher_total_mass':r1['accumulated_fisher_total_mass']}
 for field,value in mappings.items():assert math.isclose(float(entry[field]),value,rel_tol=1e-9,abs_tol=1e-10),(path,field)
 row={'setting':entry['setting'],'seed':int(entry['seed']),'a11':float(entry['R1_A_after_T1']),'a21':float(entry['R2_A_after_T2']),'a22':float(entry['R2_B_after_T2']),'BWT':float(entry['R2_A_after_T2'])-float(entry['R1_A_after_T1']),'delta_norm_squared':raw_sq,'r_effective':radius,'T1_fisher':r0['endpoint_fisher_total_mass'],'T2_current_fisher':r1['endpoint_fisher_total_mass'],'T2_accumulated_fisher':r1['accumulated_fisher_total_mass'],'lambda':float(r1['lambda_ewc']),'fisher_weighted_drift':r1['fisher_weighted_drift'],'normalized_fisher_drift':r1['normalized_fisher_drift'],'ewc_penalty_value':r1['ewc_penalty_value'],'train_ewc_ce_ratio_mean':r1.get('train_ewc_ce_ratio_mean'),'old_top1pct_update_energy_fraction':r1['delta_w_energy_in_t1_top1pct'],'old_top1pct_weighted_energy_fraction':r1['d_f_contribution_from_t1_top1pct'],'T1_emp_fisher_trace_separate_probe':r0.get('endpoint_emp_fisher_trace'),'T2_emp_fisher_trace_separate_probe':r1.get('endpoint_emp_fisher_trace'),'log_path':str(path),'mechanism_task0_line':lns[0],'mechanism_task1_line':lns[1]}
 assert math.isclose(row['fisher_weighted_drift']*row['lambda']/2,row['ewc_penalty_value'],rel_tol=1e-6,abs_tol=1e-8),(path,'penalty')
 rows.append(row);evidence.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'task_lines':lns,'csv_fields_verified':list(mappings)})
def save_csv(name,items):
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
save_csv('lora_per_seed_recomputed.csv',rows)
summaries=[]
for method in METHODS:
 rr=[r for r in rows if r['setting']==method];assert len(rr)==3
 o={'setting':method,'seeds':','.join(str(r['seed']) for r in rr),'n':3}
 for field in rows[0]:
  if field in ['setting','seed','log_path','mechanism_task0_line','mechanism_task1_line']:continue
  vals=[r[field] for r in rr]
  if all(v is not None for v in vals):o[field+'_mean']=statistics.mean(vals);o[field+'_sd']=statistics.stdev(vals)
 o['sqrt_mean_squared_drift']=math.sqrt(o['delta_norm_squared_mean'])
 summaries.append(o)
save_csv('lora_summary_recomputed.csv',summaries)
(ROOT/'lora_log_validation.json').write_text(json.dumps({'scope':'CPU text reanalysis; no model training/forward; target cohort save_ckp=False so no checkpoint-level LoRA recomputation','source_table':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'runs':evidence,'all_logged_fields_match_table_source':True,'all_penalties_match_half_lambda_weighted_drift':True},indent=2)+'\n')
for s in summaries:
 print(s['setting'],'r=',s['r_effective_mean'],'sd=',s['r_effective_sd'],'sq=',s['delta_norm_squared_mean'],'F1=',s['T1_fisher_mean'],'F2=',s['T2_current_fisher_mean'],'Fdir=',s['normalized_fisher_drift_mean'],'penalty=',s['ewc_penalty_value_mean'])
