"""Standalone native-family contracts; never imports controller common.py."""
import ast, hashlib, json, math, re
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent.parent
MODULE=Path(__file__).resolve().parent

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def resolve(path): return Path(path).resolve()
def exclusive_json(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list):return [clean_json(v) for v in value]
    return value

DIRECT={'N01':(1.,1.,1.,.1),'N02':(1.,1.,1.,0.),'N04':(0.,1.,1.,0.),'N10':(.5,1.,.5,0.)}
SELECTORS={'N03':'g1_control','N05':'g0','N06':'g2','N07':'ug','N08':'c21_base','N09':'c21_perp','N11':'g1_control','N12':'g1_control','N13':'mean','N14':'gmatch',**{k:'native_beta' for k in DIRECT}}

def verify_config(job):
    path=resolve(job['config'])
    if sha256(path)!=job['config_sha256']:raise ValueError('Config hash differs')
    cfg=read_json(path);output=resolve(cfg['outputs_root'])
    if output!=resolve(job['output_dir']) or not output.is_relative_to(HERE/'outputs'):raise ValueError('Output path mismatch')
    if cfg['seed']!=[job['seed']] or job['seed'] not in (0,1):raise ValueError('Seed mismatch')
    source=read_json(job['source_config'])
    if sha256(job['source_config'])!=job['source_config_sha256']:raise ValueError('Historical config changed')
    allowed={'seed','prefix','outputs_root'} | ({'max_train_tasks'} if job['phase']=='smoke' else set())
    if {k:v for k,v in cfg.items() if k not in allowed}!={k:v for k,v in source.items() if k not in allowed}:raise ValueError('Unexpected scientific config change')
    condition=job['condition'];selector=SELECTORS[condition]
    if job['direction_selector']!=selector:raise ValueError('Selector/config identity mismatch')
    beta=tuple(cfg[k] for k in ['gam_grad_beta_1','gam_grad_beta_2','gam_grad_beta_3','gam_grad_gamma'])
    if beta!=DIRECT.get(condition,(1.,1.,0.,0.)):raise ValueError('Native beta/gamma template mismatch')
    if (cfg['init_epoch'],cfg['epochs'],cfg['nb_tasks'],cfg['gam_grad_norm_rho'],cfg['gam_grad_rho'],cfg['lora_rank'])!=(50,50,10,1.,.05,10):raise ValueError('Training protocol differs')
    if not (cfg['ewc_lambda']==0 and cfg['ewc_skip_task_fisher'] and not cfg['in_task_weight_merge'] and not cfg['task_endpoint_weight_merge'] and not cfg['approximate_fisher_in_task']):raise ValueError('Not pure GAM')
    if job['phase']=='smoke' and cfg.get('max_train_tasks')!=2:raise ValueError('Native smoke cap must be 2 tasks')
    if job['phase']=='formal' and cfg.get('max_train_tasks',10)!=10:raise ValueError('Formal task cap differs')
    multiplier={'N11':1.5,'N12':2.,'N13':2.}.get(condition,1.)
    if job['lr_multiplier']!=multiplier or cfg['init_lr']!=.04*multiplier or cfg['lrate']!=.02*multiplier:raise ValueError('LR multiplier differs')
    return path,cfg,output

NONFINITE_LOSS = re.compile(r'\bloss(?:[ _-]*(?:train|ce|ewc|flat|total))?\s*(?:[:=]\s*)?(?:tensor\(\s*)?[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z])',re.I)
def bad_loss_lines(path):
    if not Path(path).exists():return []
    return [line[-1000:] for line in Path(path).read_text(errors='replace').replace('\r','\n').splitlines() if NONFINITE_LOSS.search(line)][:10]
def read_curve(path):
    curve=None
    for line in Path(path).read_text(errors='replace').splitlines():
        if 'CNN top1 curve:' not in line:continue
        s=line.split('CNN top1 curve:',1)[1].strip();s=re.sub(r'(?:np|numpy)\.float(?:16|32|64)\(([^()]*)\)',r'\1',s)
        try: curve=[float(v) for v in ast.literal_eval(s)]
        except (ValueError,TypeError,SyntaxError):pass
    return curve

def inspect_metrics(suite, job, *, require_complete=True):
    _, _, output = verify_config(job)
    paths = sorted(output.rglob("*_cl_metrics.json")) if output.exists() else []
    if len(paths) != 1:
        raise ValueError(f"Expected one metrics JSON, found {len(paths)}")
    data = read_json(paths[0])
    cnn = data["cnn"]
    matrices = cnn["matrices"]
    matrix = matrices.get("final")
    if matrix is None:
        if require_complete:
            raise ValueError("Missing final CNN matrix")
        steps = sorted(k for k in matrices if re.fullmatch(r"t\d+", k))
        matrix = matrices[steps[-1]] if steps else []
    expected = int(job.get("expected_tasks", suite["expected_tasks"]))
    n = len(matrix)
    if not n or (require_complete and n != expected) or n > expected:
        raise ValueError(f"Wrong task count {n}, expected {expected}")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError("CNN matrix must be square")
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in row[:i+1]):
            raise ValueError(f"Nonfinite lower triangle in CNN matrix row {i}")
    # Native FWT/AFG and the first BWT entry are structurally undefined. We
    # validate observed accuracies and independently recompute reported metrics;
    # undefined native diagnostic fields are preserved as JSON null below.
    final = cnn.get("final", {})
    log = HERE / "runtime" / "jobs" / job["id"] / "console.log"
    bad = bad_loss_lines(log)
    if bad:
        raise ValueError("Nonfinite training loss: " + bad[0])
    curve = read_curve(log)
    if curve is None or (require_complete and len(curve) != expected):
        raise ValueError("Missing/incomplete sample-weighted CNN top1 curve")
    if any(not math.isfinite(x) for x in curve):
        raise ValueError("Nonfinite CNN top1 curve")
    faa = sum(matrix[-1]) / n
    aaa = sum(sum(row[:i+1]) / (i+1) for i, row in enumerate(matrix)) / n
    forgetting = (sum(max(matrix[i][j] for i in range(j, n)) - matrix[-1][j]
                      for j in range(n-1)) / (n-1)) if n > 1 else None
    bwt = (sum(matrix[-1][j] - matrix[j][j] for j in range(n-1)) / (n-1)) if n > 1 else None
    return clean_json({"metrics_path": str(paths[0]), "metrics_sha256": sha256(paths[0]),
                       "tasks_completed": n, "expected_tasks": expected,
                       "faa_task_macro": faa, "aaa_task_macro": aaa,
                       "forgetting": forgetting, "bwt": bwt,
                       "learning_accuracy": sum(matrix[i][i] for i in range(n))/n,
                       "faa_sample_weighted": curve[-1],
                       "aaa_sample_weighted": sum(curve) / len(curve),
                       "cnn_top1_curve": curve, "cnn_matrix": matrix, "native_final": final})
