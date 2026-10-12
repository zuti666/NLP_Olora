import json,pathlib,re,ast,statistics,collections,math
root=pathlib.Path('/data/140-1/users/liying/Flatness_CV')
rows=[]
for f in root.rglob('*_cl_metrics.json'):
 if '/imagenetr/' not in str(f): continue
 try: d=json.load(open(f));c=d.get('cnn',{});a=c.get('matrices',{}).get('final');fin=c.get('final',{})
 except:continue
 if not a:continue
 log=f.with_name(f.name.replace('_cl_metrics.json','.log'))
 s=log.read_text(errors='replace') if log.exists() else ''
 cfg={k:v.strip() for k,v in re.findall(r'\[trainer.py\] => ([A-Za-z_0-9]+): ([^\n]+)',s) if not k.startswith(('CNN','NME'))}
 p=f.parts;i=p.index('imagenetr');t=len(a)
 curves=[];last_dict=None
 for line in s.splitlines():
  if '=> CNN: ' in line:
   raw=line.split('=> CNN: ',1)[1];raw=re.sub(r'np\.float(?:32|64)\(([^()]*)\)',r'\1',raw)
   try:last_dict=ast.literal_eval(raw)
   except:last_dict=None
  if 'CNN top1 curve:' in line:
   raw=line.split('CNN top1 curve:',1)[1].strip();raw=re.sub(r'np\.float(?:32|64)\(([^()]*)\)',r'\1',raw)
   try:cur=ast.literal_eval(raw)
   except:continue
   if len(cur)==t:
    vec=[v for k,v in (last_dict or {}).items() if re.fullmatch(r'\d+-\d+',k)]
    match=len(vec)==t and all(abs(float(x)-float(y))<.0001 for x,y in zip(vec,a[-1]))
    curves.append({'curve':cur,'matrix_final_match':match,'task_dict':last_dict})
 matches=[v for v in curves if v['matrix_final_match']]
 chosen=(matches or curves)[-1] if curves else None
 curve=chosen['curve'] if chosen else None
 row={'method':p[i-2],'optimizer_path':p[i-1],'seed_path':p[i+1],'tasks':t,'config':cfg,'json_faa_task_macro':fin.get('FAA'),'json_aaa_triangle':fin.get('AAA'),'curve_faa_sample':curve[-1] if curve else None,'curve_aaa_stage_mean':statistics.mean(curve) if curve else None,'curve':curve,'curve_matrix_final_match':chosen['matrix_final_match'] if chosen else None,'full_curves_in_log':len(curves),'unique_full_curves_in_log':len({tuple(v['curve']) for v in curves}),'matched_curves':len(matches),'json_path':str(f),'log_path':str(log),'final_matrix':a}
 rows.append(row)
json.dump(rows,open('/tmp/imagenetr_flatness_inventory.json','w'),indent=2)
print('SAVED',len(rows),'rows')
for r in rows:
 cfg=r['config']
 if r['tasks']==10 and '/logs_inc_lora/' in r['json_path'] and cfg.get('lora_rank')=='10':
  print(r['method'],r['optimizer_path'],r['seed_path'],round(r['curve_faa_sample'] or 0,3),round(r['curve_aaa_stage_mean'] or 0,3),'match',r['curve_matrix_final_match'],'curves',r['unique_full_curves_in_log'],'LR',cfg.get('init_lr'),cfg.get('lrate'),'ep',cfg.get('epochs'),'sched',cfg.get('scheduler'),'shuffle',cfg.get('class_shuffle'))
