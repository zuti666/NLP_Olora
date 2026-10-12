from pathlib import Path
import ast, re, json, hashlib, collections, csv, math

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/140-1/users/liying/CopyFlatCV')
files=[Path(x) for x in (HERE/'files.txt').read_text().splitlines()]
hits=[Path(x) for x in (HERE/'hits.txt').read_text().splitlines()]
ref=json.loads((ROOT/'refine-logs/s3-historical-reuse-20260920/reference_config_logged.json').read_text())
if 'params' in ref: ref=ref['params']
def val(x):
    try: return ast.literal_eval(x)
    except (SyntaxError,ValueError): return x
records=[]
for p in hits:
    if p.suffix not in ['.log','.out']: continue
    text=p.read_text(errors='replace')
    params={}
    for k,v in re.findall(r'\[[\w.]+\] => ([A-Za-z_]\w*): ([^\r\n]*)',text):
        if k not in params: params[k]=val(v)
    if str(params.get('dataset','')).replace('-','').replace('_','').lower()!='imageneta': continue
    ms=list(re.finditer(r'Accuracy Matrix \(CNN Evaluation \| time_by_task\):\s*(\[\[.*?\]\])',text,re.S))
    matrix=[]
    if ms:
        matrix=[[float(x) for x in row.split()] for row in re.findall(r'\[([^\[\]]+)\]',ms[-1].group(1))]
    n=len(matrix)
    finite=bool(n and all(len(r)==n for r in matrix) and all(math.isfinite(matrix[i][j]) for i in range(n) for j in range(i+1)))
    curves=re.findall(r'CNN top1 curve:\s*(\[[^\]\r\n]+\])',text)
    curve=val(re.sub(r'np\.float(?:32|64)\(([^)]+)\)',r'\1',curves[-1])) if curves else []
    if not isinstance(curve,list):curve=[]
    epoch_matches=re.findall(r'Task\s+(\d+),\s*Epoch\s+(\d+)/(\d+)',text)
    per_task={}
    for t,e,total in epoch_matches:
        key=str(t);cur=[int(e),int(total)]
        if key not in per_task or cur[0]>per_task[key][0]:per_task[key]=cur
    nonfinite=bool(re.search(r'(?:Loss|loss)[^\r\n]{0,50}\b(?:nan|inf)\b',text,re.I))
    rec={'path':str(p),'relative_path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'dataset':params.get('dataset'),'model_name':params.get('model_name',params.get('method')),'seed':params.get('seed'),
         'prefix':params.get('prefix'), 'params':params,'matrix_tasks':n,'matrix_lower_finite':finite,
         'cnn_curve_tasks':len(curve),'sample_final':curve[-1] if curve else None,
         'faa_macro':sum(matrix[-1])/n if finite else None,
         'aaa_stage':sum(sum(matrix[i][:i+1])/(i+1) for i in range(n))/n if finite else None,
         'epoch_coverage':per_task,'traceback': 'Traceback (most recent call last)' in text,'nonfinite_training':nonfinite,
         'status':'nonfinite_training' if nonfinite else ('complete_console_10tasks' if n==10 and finite else ('partial_matrix' if finite else 'no_complete_matrix')),
         's3_runtime_marker':'[AS1-NormFisher-Damp-WMergeThenFisher]' in text,
         'trainable_params':re.findall(r'Trainable params:\s*([\d,]+)',text),
         's3_reference_difference':{k:{'reference':v,'actual':params[k]} for k,v in ref.items() if k in params and params[k]!=v and k not in ['seed','prefix','device','filepath','config','outputs_root','optimizer_tag_override']}}
    rec['direct_s3_reuse']=False
    rec['not_direct_reason']='model or training protocol differs from the S3 LoRA 10x50 reference; inspect printed parameters'
    records.append(rec)

summaries=[]
for p in hits:
    if p.name!='run_summary.csv':continue
    with p.open(newline='') as f:
        for row in csv.DictReader(f):
            if str(row.get('dataset','')).replace('-','').replace('_','').lower()=='imageneta':
                summaries.append({'summary_path':str(p),'row':row})

groups=collections.defaultdict(list)
for r in records:
    key=(r.get('prefix') or r['path'],str(r['seed']),r['model_name'])
    groups[key].append(r)
dedup=[]
for key,g in groups.items():
    best=max(g,key=lambda x:(x['matrix_tasks'],len(x['epoch_coverage'])))
    best={**best,'related_log_paths':[r['path'] for r in g]}
    dedup.append(best)

bydir={}
for d in sorted(set(p.relative_to(ROOT).parts[0] for p in files)):
    fs=[p for p in files if p.relative_to(ROOT).parts[0]==d]
    hs=[p for p in hits if p.relative_to(ROOT).parts[0]==d]
    rs=[r for r in records if r['relative_path'].split('/')[0]==d]
    bydir[d]={'text_artifacts':len(fs),'bytes':sum(p.stat().st_size for p in fs),'keyword_hits':len(hs),'imageneta_log_records':len(rs),'models':dict(collections.Counter(r['model_name'] for r in rs)),'direct_s3_reuse_count':0}
out={'scope':'Nine explicitly enumerated additional sibling roots, all dates. Only text artifacts read. Console matrix values are rounded to 2 decimal per task. Internal/console duplicates linked by prefix+seed+model, not counted as independent experiments.',
     'total_text_artifacts':len(files),'keyword_hit_files':len(hits),'imageneta_log_records':len(records),
     'candidate_run_groups':len(dedup),'directory_coverage':bydir,'direct_s3_reuse_count':0,
     'records':records,'candidate_runs':dedup,'summary_rows':summaries}
(HERE/'audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
cols=['relative_path','model_name','seed','status','matrix_tasks','cnn_curve_tasks','faa_macro','aaa_stage','sample_final','traceback','nonfinite_training','direct_s3_reuse','params','s3_reference_difference','related_log_paths']
with (HERE/'candidate_runs.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
    for r in dedup:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in cols})
print(json.dumps({'directory_coverage':bydir,'candidate_run_groups':len(dedup),'status_count':dict(collections.Counter(r['status'] for r in dedup)),'model_counts':dict(collections.Counter(r['model_name'] for r in dedup)),'summary_rows':len(summaries)},indent=2))
