import json,pathlib,re,ast,statistics,collections
p='/tmp/imagenetr_flatness_inventory.json';rs=json.load(open(p))
for r in rs:
 s=pathlib.Path(r['log_path']).read_text(errors='replace') if pathlib.Path(r['log_path']).exists() else ''
 r['pretrained_sources']=sorted(set(re.findall(r'Loading pretrained weights from (.+)',s)))
 orders=re.findall(r'\[data_manager.py\] => (\[[^\n]*\])',s)
 r['logged_class_order']=ast.literal_eval(orders[-1]) if orders else None
 r['logged_order_identity']=r['logged_class_order']==list(range(200)) if orders else None
 if not r['curve']:continue
 matches=set();last_dict=None
 for line in s.splitlines():
  if '=> CNN: ' in line:
   raw=re.sub(r'np\.float(?:32|64)\(([^()]*)\)',r'\1',line.split('=> CNN: ',1)[1])
   try:last_dict=ast.literal_eval(re.sub(r"\bnan\b", "None", raw))
   except:last_dict=None
  if 'CNN top1 curve:' in line:
   raw=re.sub(r'np\.float(?:32|64)\(([^()]*)\)',r'\1',line.split('CNN top1 curve:',1)[1].strip())
   try:cur=ast.literal_eval(raw)
   except:continue
   n=len(cur)
   if n<=r['tasks'] and cur==r['curve'][:n]:
    vec=[v for k,v in (last_dict or {}).items() if re.fullmatch(r'\d+-\d+',k) and v is not None]
    if len(vec)==n and all(abs(float(x)-float(y))<.0001 for x,y in zip(vec,r['final_matrix'][n-1][:n])):matches.add(n)
 r['matrix_matched_curve_stages']=sorted(matches)
json.dump(rs,open(p,'w'),indent=2)
for r in rs:
 if r['tasks']==10 and r['config'].get('lora_rank')=='10' and r['method'] in ['seqlora','inclora','olora','sdlora']:
  print(r['method'],r['optimizer_path'],r['seed_path'],'matched_stages',len(r['matrix_matched_curve_stages']),'identity',r['logged_order_identity'],'pretrain',r['pretrained_sources'])
