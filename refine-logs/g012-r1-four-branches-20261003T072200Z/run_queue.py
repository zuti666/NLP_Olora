"""Local idle-GPU queue with immutable pins, verified dependencies and no preemption."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import atomic_json,assert_host,load_suite,now,proc_identity,process_matches,query_gpus,sha256,verify_source
from validate_job import validate_job

H=Path(__file__).resolve().parent

def verify_pins(suite):
    pins=json.loads((H/"RUN_PINS.json").read_text())
    for name,expected in pins["files"].items():
        if sha256(H/name)!=expected:
            raise RuntimeError("Frozen execution artifact changed: "+name)
    if sha256(suite["source_manifest"])!=suite["source_manifest_sha256"]:
        raise RuntimeError("Frozen source manifest changed")
    from experiment_contract import verify_contract
    verify_contract(suite)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite",default=str(H/"suite.json"))
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    suite=load_suite(a.suite)
    assert_host(suite)
    verify_pins(suite)
    verify_source(suite)
    if a.dry_run:
        for j in suite["jobs"]:
            if not Path(j["command"][1]).is_file():
                raise FileNotFoundError(j["command"][1])
            if j["kind"]=="training" and sha256(j["config"])!=j["config_sha256"]:
                raise RuntimeError("Config mismatch")
        print(json.dumps(dict(status="valid",jobs=len(suite["jobs"]),
                              formal=sum(j["kind"]=="training" and not j["smoke"] for j in suite["jobs"]),
                              smokes=sum(j["smoke"] for j in suite["jobs"]),
                              allowed_gpus=suite["allowed_gpus"],max_parallel=suite["max_parallel"])))
        return 0
    runtime=H/"runtime";runtime.mkdir(exist_ok=True)
    controller=open(runtime/"controller.lock","a+")
    fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=runtime/"queue_state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(run_id=suite["run_id"],created_at=now(),
                     jobs={j["id"]:{"status":"pending"} for j in suite["jobs"]})
    if state["run_id"]!=suite["run_id"] or set(state["jobs"])!={j["id"] for j in suite["jobs"]}:
        raise RuntimeError("Persisted queue belongs to a different manifest")
    jobs={j["id"]:j for j in suite["jobs"]}
    for jid,r in state["jobs"].items():
        if r["status"]=="launching":
            ident=runtime/"jobs"/jid/"worker_identity.json"
            if ident.exists():
                proof=json.loads(ident.read_text())
                if proof["job_id"]!=jid:
                    raise RuntimeError("Recovery identity mismatch")
                r.update(status="running",process=proof["process"])
            else:
                # Unknown launch is quarantined; never duplicate a potentially live child.
                r.update(status="failed",error="launch interrupted without worker identity; inspect host before retry")
        if r["status"]=="completed":
            checked=validate_job(jobs[jid])
            if checked["summary_sha256"]!=r["summary_sha256"]:
                raise RuntimeError("Completed evidence changed: "+jid)
    state.update(controller=proc_identity(os.getpid()),status="running",suite_sha256=sha256(a.suite))
    atomic_json(runtime/"controller.json",dict(process=proc_identity(os.getpid()),started_at=now(),
                                              pid_namespace=os.readlink("/proc/self/ns/pid"),command=sys.argv))
    active={jid:None for jid,r in state["jobs"].items() if r["status"]=="running"}
    leases={}
    lockdir=Path(suite["gpu_lock_dir"]);lockdir.mkdir(exist_ok=True)
    for key in ("TMPDIR","TMP","TEMP"):
        Path(suite["environment"][key]).mkdir(parents=True,exist_ok=True)
    while True:
        for jid in list(active):
            r=state["jobs"][jid];worker=active[jid]
            if worker is not None:
                if worker.poll() is None:
                    continue
            elif process_matches(r["process"]):
                continue
            d=runtime/"jobs"/jid
            try:
                exit_record=json.loads((d/"exit.json").read_text())
                if exit_record["job_id"]!=jid or exit_record["returncode"]!=0:
                    raise RuntimeError("Worker failed: "+repr(exit_record))
                verify_pins(suite)
                checked=validate_job(jobs[jid])
                atomic_json(d/"completed.json",dict(job_id=jid,completed_at=now(),exit=exit_record,**checked))
                r.update(status="completed",completed_at=now(),summary_sha256=checked["summary_sha256"])
                print(json.dumps(dict(event="completed",job=jid)),flush=True)
            except Exception as exc:
                r.update(status="failed",failed_at=now(),error=repr(exc))
                atomic_json(d/"failed.json",r)
                print(json.dumps(dict(event="failed",job=jid,error=repr(exc))),flush=True)
            finally:
                active.pop(jid)
                if jid in leases:
                    for lease in leases.pop(jid):lease.close()
                atomic_json(state_path,state)
                subprocess.run([suite["python"],str(H/"collect_results.py")],cwd=H,check=True)
        failed=any(r["status"]=="failed" for r in state["jobs"].values())
        pending=[j for j in suite["jobs"] if state["jobs"][j["id"]]["status"]=="pending"]
        if failed and not active:
            state.update(status="halted_failure",updated_at=now())
            atomic_json(state_path,state)
            return 1
        if not pending and not active:
            state.update(status="completed",updated_at=now(),counts=dict(Counter(r["status"] for r in state["jobs"].values())))
            atomic_json(state_path,state)
            return 0
        paused=(runtime/"PAUSE").exists()
        if not paused and not failed:
            gpus=query_gpus()
            used={state["jobs"][jid]["gpu"] for jid in active}
            for job in pending:
                if len(active)>=suite["max_parallel"]:
                    break
                if any(state["jobs"][dep]["status"]!="completed" for dep in job["depends_on"]):
                    continue
                for dep in job["depends_on"]:
                    dep_job=jobs[dep]
                    dep_file=dep_job.get("report_file", "hvp_gate_report.json" if dep_job["kind"]=="hvp_gate" else "final_summary.json")
                    dep_path=Path(dep_job["output_dir"])/dep_file
                    if sha256(dep_path)!=state["jobs"][dep]["summary_sha256"]:
                        raise RuntimeError("Dependency receipt changed: "+dep)
                verify_pins(suite)
                if not job["smoke"]:
                    from validate_smokes import validate_smokes
                    validate_smokes(suite)
                if job["kind"]=="training" and sha256(job["config"])!=job["config_sha256"]:
                    raise RuntimeError("Config pin changed")
                chosen=None
                for index in suite["allowed_gpus"]:
                    gpu=gpus.get(index)
                    if not gpu or index in used or gpu["memory_used_mb"]>=500 or gpu["utilization_percent"]>=5:
                        continue
                    held=[]
                    for filename in (f"gpu{index}.lock",gpu["uuid"]+".lock"):
                        lock=open(lockdir/filename,"a+")
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:lock.close();break
                        held.append(lock)
                    if len(held)!=2:
                        for lock in held:lock.close()
                        continue
                    current=query_gpus().get(index)
                    if not current or current["uuid"]!=gpu["uuid"] or current["memory_used_mb"]>=500 or current["utilization_percent"]>=5:
                        for lock in held:lock.close()
                        continue
                    chosen=(index,current,held);break
                if chosen is None:
                    break
                index,gpu,held=chosen
                jid=job["id"];d=runtime/"jobs"/jid;d.mkdir(parents=True,exist_ok=False)
                out=Path(job["output_dir"])
                if out.exists() and any(out.iterdir()):
                    for lock in held:lock.close()
                    raise RuntimeError("Refusing to reuse a nonempty output: "+jid)
                atomic_json(d/"job.json",dict(job,cwd=suite["source_root"],
                                              cuda_witness_output=str(d/"cuda_witness.json")))
                r=state["jobs"][jid]
                r.update(status="launching",started_at=now(),gpu=index,gpu_uuid=gpu["uuid"],command=job["command"])
                atomic_json(state_path,state)
                env=os.environ.copy();env.update(suite["environment"])
                env["CUDA_VISIBLE_DEVICES"]=gpu["uuid"]
                log=open(d/"console.log","xb")
                try:
                    worker=subprocess.Popen([suite["python"],str(H/"job_worker.py"),"--job",str(d/"job.json"),
                                             "--lease-fd",str(held[0].fileno()),"--uuid-lease-fd",str(held[1].fileno())],
                                            cwd=H,env=env,stdout=log,stderr=subprocess.STDOUT,
                                            start_new_session=True,pass_fds=tuple(lock.fileno() for lock in held))
                finally:
                    log.close()
                r.update(status="running",started_at=now(),gpu=index,gpu_uuid=gpu["uuid"],
                         process=proc_identity(worker.pid),command=job["command"])
                active[jid]=worker;leases[jid]=held;used.add(index)
                atomic_json(d/"launched.json",r);atomic_json(state_path,state)
                print(json.dumps(dict(event="launched",job=jid,gpu=index,pid=worker.pid)),flush=True)
        ready=any(all(state["jobs"][d]["status"]=="completed" for d in j["depends_on"]) for j in pending)
        state.update(updated_at=now(),paused=paused,
                     status="halted_failure_draining" if failed else "paused" if paused else "running" if active else "waiting_for_idle_gpu" if ready else "waiting_for_dependencies",
                     counts=dict(Counter(r["status"] for r in state["jobs"].values())))
        atomic_json(state_path,state)
        time.sleep(suite["poll_seconds"])

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        (H/"runtime").mkdir(exist_ok=True)
        atomic_json(H/"runtime/controller_error.json",dict(at=now(),error=repr(exc)))
        raise
