"""Run one queue-owned job, recording immutable provenance and final status."""
from __future__ import annotations

import argparse
import json
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import traceback

from common import (HERE, assert_host, exclusive_json, inspect_metrics, load_suite,
                    now, proc_identity, query_gpus, resolve, sha256, verify_config,
                    verify_source, verify_run_pins, verify_reuse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--uuid", required=True)
    parser.add_argument('--run-pins-sha256', required=True)
    args = parser.parse_args()
    suite = load_suite()
    job = next(job for job in suite["jobs"] if job["id"] == args.job)
    run = HERE / "runtime" / "jobs" / args.job
    record = {"id": args.job, "host": socket.gethostname(), "gpu": args.gpu,
              "gpu_uuid": args.uuid, "time": now(), **proc_identity(os.getpid())}
    try:
        assert_host(suite)
        record['run_pins_sha256'] = verify_run_pins(suite)
        if record['run_pins_sha256'] != args.run_pins_sha256:
            raise RuntimeError('Run pins do not match controller launch identity')
        verify_reuse(suite)
        if args.gpu not in suite["allowed_gpus"]:
            raise RuntimeError("Unauthorized GPU")
        if resolve(sys.executable) != resolve(suite["python"]):
            raise RuntimeError("Python interpreter mismatch")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.uuid:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly the assigned GPU UUID")
        gpu = query_gpus().get(args.gpu)
        if not gpu or gpu["uuid"] != args.uuid:
            raise RuntimeError("Physical GPU UUID mismatch")
        provenance = verify_source(suite)
        config_path, config, output = verify_config(job)
        record.update(provenance)
        record.update({"config": str(config_path), "config_sha256": sha256(config_path),
                       "suite_sha256": sha256(HERE / "suite.json"),
                       "runtime_sha256": {name: sha256(HERE / name) for name in
                                          ("common.py", "run_job.py", "launch_queue.py", "collect_results.py", "legacy_direction.py", "residual_selection.py", "effective_step.py", "step_reference.py", "validate_smokes.py", "summarize_controls.py")},
                       "python": sys.executable, "python_version": platform.python_version(),
                       "environment": {key: os.environ.get(key) for key in suite.get("environment", {})}})
        with open(config_path, "rb") as src, open(run / "config.json", "xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(run / "config.json") != job["config_sha256"]:
            raise RuntimeError("Copied config hash mismatch")
        reference_spec = None
        if job['direction_selector'] == 'gmatch':
            reference_job = next(j for j in suite['jobs'] if j['id'] == job['reference_job'])
            reference_run = HERE / 'runtime/jobs' / reference_job['id']
            from validate_smokes import validate_job_audit
            validate_job_audit(suite, reference_job)
            reference_receipt = __import__('common').read_json(reference_run/'completed.json')
            if (reference_receipt['config_sha256'] != reference_job['config_sha256']
                    or reference_receipt['run_pins_sha256'] != record['run_pins_sha256']):
                raise RuntimeError('Independent S reference differs from this frozen campaign')
            record['reference_dependency'] = {
                'id':reference_job['id'], 'phase':reference_job['phase'],
                'hashes':{name:sha256(reference_run/name) for name in
                          ['config.json','direction_audit.jsonl','direction_audit_summary.json','completed.json']}}
            reference_spec = {'job':reference_job, 'run_dir':str(reference_run),
                              'candidate_config':config, 'candidate_phase':job['phase']}
        output.mkdir(parents=True, exist_ok=False)
        source = resolve(suite["source_root"])
        os.chdir(source)
        sys.path.insert(0, str(source))
        import torch
        torch.set_num_threads(8)
        reuse = json.loads((HERE/'ENVIRONMENT_REUSE.json').read_text())
        spec = json.loads(Path(reuse['spec_path']).read_text())
        canonical = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if canonical != reuse['canonical_sha256']:
            raise RuntimeError('Warm environment specification changed')
        versions={'python':platform.python_version(),'torch':str(torch.__version__),
                  'torchvision':importlib.metadata.version('torchvision'),'timm':importlib.metadata.version('timm'),
                  'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        if versions != {'python':'3.11.15','torch':'2.4.1+cu121','torchvision':'0.19.1',
                        'timm':'1.0.9','cuda':'12.1','cudnn':90100}:
            raise RuntimeError('Warm runtime package versions differ: '+repr(versions))
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError('Historical TF32=false changed')
        record['warm_environment_versions']=versions
        record['warm_environment_spec_sha256']=canonical
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"Expected exactly one visible GPU; found {torch.cuda.device_count()}")
        record.update({"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                       "cudnn_version": torch.backends.cudnn.version(),
                       "visible_device_count": torch.cuda.device_count(),
                       "gpu_name": torch.cuda.get_device_name(0),
                       "native_flags_before_main": {"cudnn_benchmark": torch.backends.cudnn.benchmark,
                                                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                       "status": "started"})
        exclusive_json(run / "started.json", record)
        print(json.dumps({"runner_started": args.job, "gpu": args.gpu,
                          "uuid": args.uuid, "config_sha256": record["config_sha256"]}), flush=True)
        from src.main import main as train
        import torchvision
        if torchvision.__version__ != '0.19.1+cu121':
            raise RuntimeError('Warm torchvision runtime/CUDA build differs')
        record['torchvision_runtime_version']=torchvision.__version__
        from utils import factory
        from legacy_direction import install
        with install(factory, selector=job['direction_selector'],
                     audit_path=run / 'direction_audit.jsonl', source_root=source,
                     expected_radius=job['outer_radius'], reference_spec=reference_spec):
            train(["--config", str(run / "config.json")])
        sys.stdout.flush()
        sys.stderr.flush()
        ending_source = verify_source(suite)
        if ending_source["source_manifest_sha256"] != provenance["source_manifest_sha256"]:
            raise RuntimeError("Source manifest changed during training")
        result = inspect_metrics(suite, job)
        from validate_smokes import validate_job_audit
        proof = validate_job_audit(suite, job, require_receipt=False)
        record['direction_proof'] = proof
        if reference_spec:
            for name,digest in record['reference_dependency']['hashes'].items():
                if sha256(Path(reference_spec['run_dir'])/name) != digest:
                    raise RuntimeError('Independent S reference changed during Gmatch training')
        if verify_run_pins(suite) != args.run_pins_sha256:
            raise RuntimeError('Run pins changed during training')
        exclusive_json(run / "completed.json", {**record, "status": "completed", "ended_at": now(),
                                                "result": result})
        print(json.dumps({"runner_completed": args.job, "result": result}, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        record.update({"status": "failed", "ended_at": now(), "error": repr(exc),
                       "traceback": traceback.format_exc()})
        if not (run / "failed.json").exists():
            exclusive_json(run / "failed.json", record)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
