"""Small, standard-library helpers for the pinned S3 experiment queue."""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ALLOWED_GPUS = {1, 2, 3}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path, base=HERE):
    value = Path(path)
    return (value if value.is_absolute() else base / value).resolve()


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n")


def exclusive_json(path, value):
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def proc_identity(pid):
    try:
        # comm may contain spaces and parentheses; fields after its LAST ')' are stable.
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "proc_starttime": fields[19],
                "hostname": socket.gethostname(),
                "boot_id": Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                "pid_namespace": os.readlink(f'/proc/{int(pid)}/ns/pid')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_matches(record):
    keys = {'pid', 'proc_starttime', 'hostname', 'boot_id', 'pid_namespace'}
    if not isinstance(record, dict) or set(record) != keys:
        return False
    identity = proc_identity(record["pid"])
    return bool(identity and identity == record)


def query_gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=30)
    result = {}
    for row in csv.reader(io.StringIO(output)):
        index, uuid, memory, utilization = [item.strip() for item in row]
        result[int(index)] = {"index": int(index), "uuid": uuid,
                              "memory_used_mb": int(memory), "utilization_percent": int(utilization)}
    return result


def assert_host(suite):
    if socket.gethostname() != suite["execution_hostname"]:
        raise RuntimeError(f"Host mismatch: {socket.gethostname()} != {suite['execution_hostname']}")


DIRECTIONS = [
    ('OE01_SAM_s2','g1_plus'),('OE02_Backward_s2','backward_reflected'),
    ('OE03_Central_s2','central'),('OE04_CE_s2','g0'),('OE05_Negative_s2','g1_minus'),
    ('OE06_Average_s2','even_average'),('OE07_ReflectedAverage_s2','even_reflected')]
PAIRING = 'seven_way_smoke_pairing'
FORMULAS = dict(g1_plus='g1_plus',backward_reflected='g0+(g0-g1_minus)',
    central='g0+0.5*(g1_plus-g1_minus)',g0='g0',g1_minus='g1_minus',
    even_average='0.5*(g1_plus+g1_minus)',even_reflected='g0+(g0-0.5*(g1_plus+g1_minus))')


def load_suite(path=HERE / 'suite.json'):
    suite = read_json(path)
    if suite['allowed_gpus'] != [1,2,3] or suite['max_parallel'] != 3:
        raise ValueError('Only physical GPUs 1,2,3 with max_parallel=3 are authorized')
    if suite['gpu_lock_dir'] != '/tmp/olora_gamfs_gpu_locks':
        raise ValueError('All campaigns must share the GPU lock directory')
    if suite['gpu_free_threshold_mib'] != 500 or suite['gpu_free_util_below_percent'] != 10:
        raise ValueError('Fresh idle criteria must remain <500 MiB and <10%')
    if suite['run_dir'] != str(HERE) or suite['run_id'] != HERE.name:
        raise ValueError('Run identity differs')
    if suite['method_name'] != 'odd_even_local_v1' or suite['outer_radius'] is not None or suite['no_external_notifications'] is not True:
        raise ValueError('Suite scientific identity or notification scope differs')
    if 'CUDA_VISIBLE_DEVICES' in suite.get('environment',{}):
        raise ValueError('Queue exclusively binds CUDA UUIDs')
    formal=[p[0] for p in DIRECTIONS];smokes=['smoke_'+p for p in formal]
    expected_ids=smokes+[PAIRING]+formal
    if [j['id'] for j in suite['jobs']] != expected_ids:
        raise ValueError('Exactly seven smokes, CPU pairing and seven formal jobs are authorized')
    for job in suite['jobs']:
        output=HERE/'outputs'/job['id']
        if job['output_dir'] != str(output):raise ValueError('Output path differs')
        if job['id']==PAIRING:
            required=dict(kind='pairing',phase='pairing',smoke=False,gpu_required=False,
                depends_on=smokes,expected_tasks=2,report_file='smoke_pairing_report.json',
                command=[suite['python'],str(HERE/'verify_pairing.py'),'--run-dir',str(HERE),'--output-dir',str(output)])
        else:
            smoke=job['id'].startswith('smoke_');ident=job['id'][6:] if smoke else job['id']
            direction=dict(DIRECTIONS)[ident];config=HERE/'configs'/(job['id']+'.json')
            command=[suite['python'],str(HERE/'trainer/odd_even_train.py'),'--source',suite['source_root'],
                '--config',str(config),'--output-dir',str(output)]
            if smoke:command+=['--max-tasks','2','--max-epochs','1','--max-batches','1']
            deps=[] if smoke else [PAIRING] if ident in formal[:3] else [PAIRING]+formal[:3]
            required=dict(kind='training',phase='smoke' if smoke else 'formal',branch='local_symmetric',
                direction=direction,mode='natural',backend='math',smoke=smoke,gpu_required=True,
                depends_on=deps,config=str(config),command=command,expected_tasks=2 if smoke else 10,
                expected_epochs_per_task=1 if smoke else 50,expected_steps=2 if smoke else 2550,
                expected_training_ce=6 if smoke else 7650,expected_isolated_replay_ce=12 if smoke else 60)
        differences=[k for k,v in required.items() if job.get(k)!=v]
        if differences:raise ValueError('Registered job differs: '+job['id']+' '+repr(differences))
    return suite


def verify_source(suite):
    root = resolve(suite["source_root"])
    manifest_path = resolve(suite["source_manifest"])
    manifest = read_json(manifest_path)
    files = manifest["files"]
    if len(files) != 190 or sha256(manifest_path) != suite["source_manifest_sha256"]:
        raise ValueError("Expected the unchanged pinned 190-file source manifest")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")
    return {"source_root": str(root), "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256(manifest_path), "source_files": files}


def verify_config(job):
    path = resolve(job["config"])
    actual = sha256(path)
    if actual != job["config_sha256"]:
        raise RuntimeError(f"Config hash mismatch: {job['id']}")
    config = read_json(path)
    if not Path(config["outputs_root"]).is_absolute():
        raise ValueError("outputs_root must be absolute so the native trainer uses the verified location")
    output = resolve(config["outputs_root"])
    if not output.is_relative_to(HERE) or output == HERE:
        raise ValueError(f"outputs_root must be a dedicated directory below campaign HERE: {output}")
    required = dict(probe_branch='local_symmetric',update_direction=job['direction'],method_name='odd_even_local_v1',step_mode='natural',
        attention_backend='math',seed=[2],gam_grad_rho=.05,gam_grad_norm_rho=0.0,lora_rank=10,
        class_shuffle=False,init_cls=20,increment=20,nb_tasks=10,total_sessions=10,
        init_epoch=50,epochs=50,batch_size=128,init_lr=.04,lrate=.02,
        optimizer='sgd',momentum=0,init_momentum=0,weight_decay=0.,init_weight_decay=0.,
        ewc_lambda=0.,in_task_weight_merge=False,task_endpoint_weight_merge=False,
        approximate_fisher_in_task=False,compute_full_fisher=False,ewc_skip_task_fisher=True,
        in_task_path_normalize_snapshots=False,max_train_tasks=10,scheduler='cosine',cosine_min_lr_ratio=.1,
        memory_size=0,memory_per_class=0,use_sgv=False,gam_use_sgv=False,sgv_lambda=0.,
        gam_grad_beta_1=1.,gam_grad_beta_2=1.,gam_grad_beta_3=0.,gam_grad_gamma=0.,
        gam_adaptive=False,lambda_flat=1.,backbone_type='vit_base_patch16_224',
        dataset='imageneta',data_root='/data/140-1/datasets',device=[0],
        component_rule='u0',component_scale='natural',optimizer_type='gam',gam_base_optimizer='sgd')
    if any(config.get(k)!=v for k,v in required.items()):
        raise ValueError('Pinned scientific configuration differs: '+job['id'])
    if config['prefix'] != job['id'] or output != resolve(job['output_dir'])/'native':
        raise ValueError('Native output or prefix differs from registered job')
    return path, config, output


def validate(suite):
    assert_host(suite)
    source = verify_source(suite)
    outputs = []
    for job in suite["jobs"]:
        output=resolve(job['output_dir'])
        if job['kind']=='training':verify_config(job)
        if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in outputs):
            raise ValueError("Job outputs_root directories overlap")
        outputs.append(output)
    if not resolve(suite["python"]).is_file():
        raise FileNotFoundError(suite["python"])
    return source


NONFINITE_LOSS = re.compile(
    r"\bloss(?:[ _-]*(?:train|ce|ewc|flat|total))?\s*(?:[:=]\s*)?"
    r"(?:tensor\(\s*)?[+-]?(?:nan|inf(?:inity)?)(?![A-Za-z])", re.I)


def bad_loss_lines(log_path):
    if not Path(log_path).is_file():
        return []
    bad = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for fragment in line.split("\r"):
                if NONFINITE_LOSS.search(fragment):
                    bad.append(fragment.strip()[-1000:])
                    if len(bad) >= 10:
                        return bad
    return bad


def read_curve(log_path):
    curve = None
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "CNN top1 curve:" not in line:
                continue
            payload = line.split("CNN top1 curve:", 1)[1].strip()
            payload = re.sub(r"(?:np|numpy)\.float(?:16|32|64)\(([^()]*)\)", r"\1", payload)
            try:
                candidate = ast.literal_eval(payload)
                if isinstance(candidate, list):
                    curve = [float(value) for value in candidate]
            except (ValueError, SyntaxError, TypeError):
                continue
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
                       "faa_sample_weighted": curve[-1],
                       "aaa_sample_weighted": sum(curve) / len(curve),
                       "cnn_top1_curve": curve, "cnn_matrix": matrix, "native_final": final})


def verify_preflight(suite):
    """Verify current odd/even-code evidence; copied G2 gates are context only."""
    for rel in ('gates/INPUT_ASSETS.json','trainer/ODD_EVEN_CPU_GATE.json','gates/TRAINER_ENTRYPOINTS.json','gates/INDEPENDENT_REVIEW.json'):
        if read_json(HERE/rel)['status'] != 'passed':
            raise RuntimeError('Required current preflight did not pass: '+rel)
    cpu = read_json(HERE/'trainer/ODD_EVEN_CPU_GATE.json')
    if cpu['cpu_only'] is not True or cpu['gpu_acceptance'] is not False or cpu['tests_run'] < 15 or cpu['errors'] or cpu['failures']:
        raise RuntimeError('Incomplete odd/even-engine CPU gate')
    for field,name in (('odd_even_step_sha256','odd_even_step.py'),('trainer_sha256','odd_even_train.py'),('tests_sha256','test_odd_even_cpu.py')):
        if cpu[field] != sha256(HERE/'trainer'/name):
            raise RuntimeError('Odd/even CPU evidence predates final code: '+name)
    entry = read_json(HERE/'gates/TRAINER_ENTRYPOINTS.json')
    if entry['trainer_sha256'] != sha256(HERE/'trainer/odd_even_train.py') or entry['cpu_only'] is not True or entry['gpu_calls'] != 0:
        raise RuntimeError('Actual trainer entrypoint identity differs')
    training_jobs=[j for j in suite['jobs'] if j['kind']=='training']
    if len(entry['jobs']) != 14 or {r['job'] for r in entry['jobs']} != {j['id'] for j in training_jobs}:
        raise RuntimeError('Exactly fourteen actual entrypoint receipts required')
    for job in training_jobs:
        row = next(r for r in entry['jobs'] if r['job'] == job['id'])
        if row['returncode'] != 0 or row['command'] != job['command']+['--dry-run'] or row['config_sha256'] != job['config_sha256']:
            raise RuntimeError('Actual trainer entrypoint command/config differs: '+job['id'])
        observed = json.loads(row['stdout'])
        required = dict(status='valid',source=suite['source_root'],config=job['config'],output_dir=job['output_dir'],branch=job['branch'],direction=job['direction'],mode='natural',backend='math',tasks=job['expected_tasks'],epochs=job['expected_epochs_per_task'],seed=2)
        if any(observed.get(k) != v for k,v in required.items()):
            raise RuntimeError('Actual trainer dry-run output differs: '+job['id'])
    spec = read_json(HERE/'environment_spec.json')
    assets = read_json(HERE/'gates/INPUT_ASSETS.json')
    canonical = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if canonical != '4df25e5efd1138e99d7388dd692dc7285fdab47e8b090c76d650c33cf32e74d8' or assets['canonical_spec_sha256'] != canonical:
        raise RuntimeError('Warm environment spec/receipt changed')
    if assets['gpu_calls'] != 0 or assets['package_rebuild'] is not False or assets['source']['source_file_count'] != 190 or assets['source']['manifest_sha256'] != suite['source_manifest_sha256'] or not assets['source']['all_source_hashes_exact']:
        raise RuntimeError('Input/source receipt is incomplete')
    if assets['source']['root'] != suite['source_root'] or assets['data']['asset_hashes'] != spec['data']['asset_hashes'] or assets['data']['all_expected_hashes_exact'] is not True:
        raise RuntimeError('Input asset identities differ')
    if assets['pretrained']['sha256'] != spec['weight_dirs']['vit_b16']['sha256'] or assets['pretrained']['verified'] is not True:
        raise RuntimeError('Pretrained identity differs')
    prior = read_json(suite['ordinary_gradient_receipt'])
    if sha256(suite['ordinary_gradient_receipt']) != suite['ordinary_gradient_receipt_sha256'] or prior['status'] != 'passed' or prior['case_count'] != 24 or prior['HVP_training_allowed'] is not False:
        raise RuntimeError('Prior ordinary math gradient acceptance differs')
