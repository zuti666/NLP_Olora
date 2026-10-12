"""Read-only bounded inventory of logs_our_joost. No imports of experiment code."""
import ast,collections,datetime,hashlib,json,math,re,subprocess
from pathlib import Path
ROOT=Path('/data/140-1/users/liying/CopyFlatCV/logs_our_joost')
OUT=Path(__file__).resolve().parent
COHORTS=Path('/data/140-1/users/liying/O-LoRA/refine-logs/seed012_plan_20260922/COHORTS.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
def num(x):
 try:return ast.literal_eval(x)
 except:return x

def stats(a,expected):
 if not a:return {'matrix_tasks':0,'matrix_complete':False,'FAA':None,'AAA_stage':None}
 n=len(a);finite=all(len(row)>=i+1 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in row[:i+1]) for i,row in enumerate(a));complete=finite and expected is not None and n==int(expected) and all(len(row)==n for row in a)
 return {'matrix_tasks':n,'expected_tasks':expected,'observed_lower_triangle_finite':finite,'matrix_complete':complete,'FAA':sum(a[-1][:n])/n if finite else None,'AAA_stage':sum(sum(row[:i+1])/(i+1) for i,row in enumerate(a))/n if finite else None,'lower_triangle':[[v if isinstance(v,(int,float)) and math.isfinite(v) else None for v in row[:i+1]] for i,row in enumerate(a)]}

def log_matrices(t):
 out=[]
 for m in re.finditer(r'Accuracy Matrix \(CNN Evaluation \| time_by_task\):\s*(\[\[.*?\]\])',t,re.S):
  try:out.append([[float(v) for v in row.split()] for row in re.findall(r'\[([^\[\]]+)\]',m.group(1))])
  except ValueError:continue
 return out

proc=subprocess.run(['rg','--files','-uuu',str(ROOT),'-g','*.log','-g','*.json','-g','*.yaml','-g','*.yml'],capture_output=True,text=True)
if proc.returncode not in (0,1):raise RuntimeError(proc.stderr)
paths=sorted(Path(p) for p in proc.stdout.splitlines());catalog=[];jsons={};records=[];unparsed=[]
for p in paths:
 st=p.stat();b=p.read_bytes();st2=p.stat();entry={'path':str(p),'relative':str(p.relative_to(ROOT)),'suffix':p.suffix,'bytes':len(b),'sha256':sha(b),'changed_while_read':(st.st_size,st.st_mtime_ns)!=(st2.st_size,st2.st_mtime_ns)};catalog.append(entry)
 if p.suffix=='.json':
  try:d=json.loads(b);entry['json_top_keys']=list(d) if isinstance(d,dict) else [];entry['kind']='cl_metrics' if isinstance(d,dict) and isinstance(d.get('cnn'),dict) else ('lora_metadata' if p.name.startswith('lora_meta_') else 'other_json');jsons[str(p)]=(d,entry)
  except Exception as e:entry['kind']='invalid_json';entry['parse_error']=str(e)
 elif p.suffix in ('.yaml','.yml'):entry['kind']='yaml_config'
 else:
  entry['kind']='log';text=b.decode(errors='replace');matches=list(re.finditer(r'^.*?=> prefix: (.*)$',text,re.M))
  if not matches:unparsed.append({'path':str(p),'sha256':entry['sha256'],'reason':'No printed prefix configuration block; may be queue/watcher/failed startup. Not assigned a dataset/method from filename.'});continue
  for i,m in enumerate(matches):
   start=m.start();end=matches[i+1].start() if i+1<len(matches) else len(text);block=text[start:end];pairs=re.findall(r'^.*?=> ([A-Za-z_][A-Za-z_0-9]*): (.*)$',block,re.M);params={}
   for k,v in pairs:
    if k not in params:params[k]=num(v.strip())
   if not params.get('dataset') or not params.get('model_name'):
    unparsed.append({'path':str(p),'sha256':entry['sha256'],'block':i,'reason':'Prefix block lacks actual dataset/model_name parameters.'});continue
   expected=params.get('nb_tasks',params.get('total_sessions'));matrixes=log_matrices(block);mat=matrixes[-1] if matrixes else None
   lossbad=re.search(r'\b(?:Loss|Train_loss|train_loss|CE|EWC)\s*(?:[=:]\s*)?(?:nan|[-+]?inf)\b',block,re.I)
   metricpath=str(p.with_name(p.stem+'_cl_metrics.json'))
   timestamp=re.search(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}',block)
   epochs=[{'task':int(a),'epoch':int(bb),'total':int(c)} for a,bb,c in re.findall(r'Task\s+(\d+),\s*Epoch\s+(\d+)/(\d+)',block)]
   rec={'id':len(records),'path':str(p),'log_sha256':entry['sha256'],'changed_while_read':entry['changed_while_read'],'block_index':i,'block_count':len(matches),'start_line':text[:start].count('\n')+1,'start_timestamp':timestamp.group() if timestamp else None,'method':params['model_name'],'dataset':params['dataset'],'seed':params.get('seed'),'parameters':params,'core':{k:params.get(k) for k in ['model_name','dataset','seed','init_lr','lrate','init_epoch','epochs','lora_rank','batch_size','backbone_type','class_shuffle','init_cls','increment','nb_tasks','total_sessions','optimizer','optimizer_type','momentum','weight_decay','ewc_lambda','fs_alpha','fs_direction_mode','fs_base_optimizer','fisher_mode']},'source_config_recorded':params.get('config'),'log_matrix':stats(mat,expected),'training_nonfinite_observed':bool(lossbad),'nonfinite_evidence':block[max(0,lossbad.start()-70):lossbad.end()+100].replace('\r',' ') if lossbad else None,'traceback':bool(re.search(r'Traceback \(most recent call last\)',block)),'last_epoch':epochs[-1] if epochs else None,'direct_metric_candidate':metricpath if i==len(matches)-1 else None,'evaluation_only_nan_upper_triangle_ignored':True}
   records.append(rec)
# Match metrics after reading every file; raw log matrices remain an independent consistency witness.
used=set()
for r in records:
 p=r['direct_metric_candidate'];entry=jsons.get(p)
 if entry and isinstance(entry[0],dict) and 'cnn' in entry[0]:
  d,j=entry;mm=d['cnn'].get('matrices',{});a=mm.get('final');key='final'
  if a is None and mm:
   keys=[k for k in mm if re.fullmatch(r't\d+',k)];key=max(keys,key=lambda x:int(x[1:])) if keys else None;a=mm.get(key)
  r['metrics_source']=p;r['metrics_sha256']=j['sha256'];r['metrics_matrix_key']=key;r['json_matrix']=stats(a,r['parameters'].get('nb_tasks',r['parameters'].get('total_sessions')));used.add(p)
  lm=r['log_matrix'];jm=r['json_matrix'];r['json_log_matrix_match']=(lm.get('lower_triangle')==jm.get('lower_triangle')) if lm['matrix_tasks'] else None
 else:r['json_matrix']=None
 metric=r['json_matrix'] if r['json_matrix'] and r.get('json_log_matrix_match') is not False else r['log_matrix'];r['results']=metric
 if r['training_nonfinite_observed']:state='invalid_nonfinite_training'
 elif r.get('json_log_matrix_match') is False:state='ambiguous_json_log_mismatch'
 elif metric['matrix_complete']:state='completed_finite_matrix'
 elif r['traceback']:state='failed_exception'
 elif metric['matrix_tasks']:state='partial_sequence'
 else:state='no_complete_result'
 r['status']=state;r['completion_caveat']='Completed means complete finite CNN matrix and no nonfinite training observed; does not claim process exit0 or full provenance audit.'
 # Config comparison fingerprint excludes only I/O, device and seed. Preserves every printed scientific field.
 ignore={'seed','prefix','config','filepath','outputs_root','device','feature_flat_save_path','CNN','NME'}
 pp={k:v for k,v in r['parameters'].items() if k not in ignore};r['recorded_nonseed_protocol_sha256']=sha(json.dumps(pp,sort_keys=True,default=str).encode())
 r['run_evidence_key']=sha(json.dumps([r['method'],r['dataset'],r['seed'],r['parameters'].get('prefix'),r['start_timestamp'][:19] if r['start_timestamp'] else None,r['recorded_nonseed_protocol_sha256']],sort_keys=True,default=str).encode())
# Exact evidence copies and repeated same-seed attempts remain separately represented; never counted as seeds.
evidencegroups=collections.defaultdict(list);protocolgroups=collections.defaultdict(list)
for r in records:evidencegroups[r['run_evidence_key']].append(r['id']);protocolgroups[(r['recorded_nonseed_protocol_sha256'],str(r['seed']))].append(r['id'])
duplicates=[{'run_evidence_key':k,'record_ids':v,'interpretation':'Same actual config/prefix/seed/start-second; probable stdout+trainer-log duplicates, not independent runs.'} for k,v in evidencegroups.items() if len(v)>1]
repeat=[{'protocol_sha256':k[0],'seed':k[1],'record_ids':v,'distinct_start_evidence_count':len({records[i]['run_evidence_key'] for i in v})} for k,v in protocolgroups.items() if len({records[i]['run_evidence_key'] for i in v})>1]
cohorts=json.loads(COHORTS.read_text());target=[]
for c in cohorts:
 candidates=[r for r in records if r['dataset']==c['dataset']]
 target.append({'cohort_id':c['cohort_id'],'method':c['method'],'dataset':c['dataset'],'target_seeds':[0,1,2],'same_dataset_records':len(candidates),'eligible_seed012':[],'reuse_count':0,'exclusion_reason':'No actual log dataset matches this cohort in the specified directory.' if not candidates else 'Needs method/config/source matching; not automatically reusable.'})
# Orphan metrics are visible, never silently accepted without logged identity.
orphans=[]
for p,(d,e) in jsons.items():
 if e.get('kind')=='cl_metrics' and p not in used:orphans.append({'path':p,'sha256':e['sha256'],'reason':'No matched last log block; no method/dataset inferred from path.'})
dist=collections.Counter(r['dataset'] for r in records);meth=collections.Counter(r['method'] for r in records);statuses=collections.Counter(r['status'] for r in records)
summary={}
for d in sorted(dist):
 rr=[r for r in records if r['dataset']==d];summary[d]={'log_blocks':len(rr),'methods':sorted({r['method'] for r in rr}),'seeds':sorted({str(r['seed']) for r in rr}),'statuses':dict(collections.Counter(r['status'] for r in rr)),'ranks':sorted({str(r['core']['lora_rank']) for r in rr}),'batch_sizes':sorted({str(r['core']['batch_size']) for r in rr}),'epochs':sorted({str(r['core']['epochs']) for r in rr}),'learning_rates':sorted({str(r['core']['lrate']) for r in rr})}
report={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scan_root':str(ROOT),'scope':'Only this directory; no training, no other source trees searched.','inventory_script':str(Path(__file__).resolve()),'file_count':len(paths),'file_types':dict(collections.Counter(p.suffix for p in paths)),'json_kinds':dict(collections.Counter(e.get('kind') for e in catalog if e['suffix']=='.json')),'top_directories':dict(collections.Counter(Path(e['relative']).parts[0] for e in catalog)),'log_record_count':len(records),'unparsed_log_blocks_or_files':unparsed,'dataset_summary':summary,'methods':dict(meth),'status_counts_log_evidence_not_independent_runs':dict(statuses),'distinct_run_evidence_keys':len(evidencegroups),'probable_duplicate_evidence_groups':duplicates,'same_protocol_same_seed_repeated_attempt_groups':repeat,'orphan_metrics':orphans,'target_cohorts':target,'target_cohort_reusable_count':sum(x['reuse_count'] for x in target),'records':records,'files':catalog,'statistics_policy':'FAA/AAA_stage recomputed from CNN lower triangle. Matrix upper NaN is expected and ignored; train Loss/CE/EWC nan/inf invalidates run even if accuracy matrix finite. No SD computed across different datasets/configs or duplicate evidence.','limitations':['Logs can be live; changed_while_read flags detect changes within each read, not whole-directory atomicity.','Protocol fingerprint uses recorded fields, does not assert missing defaults or source/data/checkpoint identity.','No exit0 receipts are asserted by this inventory.','Unknown/no-config logs and orphan metrics are retained, not identified by filenames.']}
(OUT/'joost_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
md=['# logs_our_joost 结果盘点（2026-09-22）','','**当前表35组没有可复用的新seed。** 完整扫描此指定目录后，实际打印的dataset只有ImageNet-A、ImageNet-R、cifar224(CIFAR100)、DomainNet；Aircraft/Cars/CUB/Flowers/Het5均未出现。本结论以日志参数为依据，没有用路径猜测模型或数据集。', '',f'共{len(paths)}文件，{sum(p.suffix==".log" for p in paths)}日志，{sum(p.suffix==".json" for p in paths)} JSON（{sum(e.get("kind")=="cl_metrics" for e in catalog)}任务指标JSON），无YAML配置。解析{len(records)}个带实际配置的日志块；另有{len(unparsed)}条缺配置/queue/watcher/失败启动日志。','','| Actual dataset | Log blocks | Seeds found | Ranks | Batch | Epochs |','|---|---:|---|---|---|---|']
for d,s in summary.items():md.append(f"| {d} | {s['log_blocks']} | {', '.join(s['seeds'])} | {', '.join(s['ranks'])} | {', '.join(s['batch_sizes'])} | {', '.join(s['epochs'])} |")
md+=['','日志块数包括stdout和trainer副本及重复attempt，不是独立实验次数。', '', '| Log-evidence status | Count |','|---|---:|']
for k,v in statuses.items():md.append(f'| {k} | {v} |')
md+=['',f'按实际配置、prefix、seed、启动秒生成{len(evidencegroups)}个run证据key；{len(duplicates)}组疑似重复日志，{len(repeat)}组同协议同seed不同启动attempt，均单独列出而不冒充多个seed。', '', '完成判定：完整预期任务数的finite CNN下三角矩阵，并且日志没有Loss/CE/EWC NaN/Inf；并不等同于已验证进程exit0。NaN训练即使最后能输出finite准确率也标invalid。矩阵上三角NaN是未见任务正常占位，不标发散。', '', 'FAA=最终一行任务宏均值；AAA_stage=各阶段任务宏均值的均值。JSON保存每次run的两指标、矩阵下三角、配置、source SHA、完成状态、重复关系，以及35组逐项不匹配原因。', '', '本目录其他方法均仅盘点、不加入当前七方法新实验：', '']
for m,n in sorted(meth.items()):md.append(f'- `{m}`：{n}个日志块。')
md+=['','复算：`python '+str(Path(__file__).resolve())+'`。只读取指定目录和既有35组配置摘要，输出写当前执行审计目录。源数据不会改动。','',f'指标未配对原日志数：{len(orphans)}；读取期间发生写入的文件数：{sum(e["changed_while_read"] for e in catalog)}。尚未直接认证任何其他协议可用于当前表，未计算跨协议SD。']
(OUT/'joost_inventory.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'files':len(paths),'logs':len(records),'datasets':summary,'statuses':dict(statuses),'duplicate_groups':len(duplicates),'repeated_attempts':len(repeat),'orphans':len(orphans),'target_eligible':sum(x['reuse_count'] for x in target)},ensure_ascii=False,indent=2))
