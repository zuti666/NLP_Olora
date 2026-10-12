#!/usr/bin/env python3
"""CPU completion-validator contracts: positive schema must pass before negatives.

Numerical curvature fields come from real frozen-GAM tiny-model calculations.
The envelope/files are synthetic fixtures, never scientific experiment output.
KeyError/TypeError/missing implementation fields are test failures, not rejection
successes. No GPU, original output, or implementation file is modified.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

HERE=Path(__file__).resolve().parent.parent


def module_from(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=True)+"\n")


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curvature_fixture(campaign,work,validator):
    suite=json.loads((campaign/"suite.json").read_text())
    job=copy.deepcopy(next(j for j in suite["jobs"] if j["kind"]=="curvature" and j["task"]==0 and j["rule"]=="us"))
    job["max_batches"]=1
    source=Path(suite["source_root"])
    formation=campaign.parent/"formation-run-20261001T180712Z"/"trainer"
    sys.path[:0]=[str(source),str(formation),str(campaign/"diagnostics")]
    helper=module_from("curvature_parent_fixture",campaign/"review/horizon_validation_fixture.py")
    parent_job=helper.build_fixture(work,"us","matched")
    parent_id=parent_job["id"]
    cp=Path(parent_job["output_dir"])/"checkpoints/task_000"
    ca=module_from("validator_fixture_curvature",campaign/"diagnostics/curvature_audit.py")
    from optimer_PerturabtionType.gam import GAM
    from test_future_probe_cpu import make_learner
    # Recreate exactly the tiny parent checkpoint, so report and parent-model
    # hashes agree rather than merely sharing a fabricated identity envelope.
    ca.seed_all(2)
    model=helper.tiny.Toy()
    with torch.no_grad():
        for letter in ("a","b"):
            saved=torch.load(cp/("lora_w_%s_0.pt"%letter),map_location="cpu",weights_only=False)
            target=getattr(model.backbone,"w_"+letter.upper()+"s")
            for dst,src in zip(target,saved):dst.weight.copy_(src.weight)
    model.fc.load_state_dict(torch.load(cp/"fc_state_0.pt",map_location="cpu",weights_only=False))
    manifest=json.loads((cp/"manifest.json").read_text())
    assert ca.parameter_sha(model)==manifest["model_sha256"]
    learner=make_learner(model,source);learner._unwrap_network=lambda:model
    # The real runner uses a seeded shuffled loader: indices are not range(128).
    ca.seed_all(270000)
    data=torch.utils.data.TensorDataset(torch.arange(1259),torch.randn(1259,7),torch.arange(1259)%20)
    loader=torch.utils.data.DataLoader(data,batch_size=128,shuffle=True,num_workers=0,
                                      generator=torch.Generator().manual_seed(270000))
    ids,x,y=next(iter(loader))
    batch=ca.audit_batch(learner,GAM,x,y,0,20,.05,.25,(.05,.025,.0125),"native")
    batch.update(batch=0,samples=128,data_seed=270000,input_sha256=ca.tensor_sha(x),target_sha256=ca.tensor_sha(y),indices=ids.tolist())
    output=work/"outputs"/job["id"];output.mkdir(parents=True)
    source_pins=ca.pin_tree(source,"**/*.py")
    config=Path(parent_job["config"])
    job.update(config=str(config),config_sha256=ca.sha(config),output_dir=str(output),depends_on=[parent_id])
    command=job["command"]
    for flag,value in (("--config",str(config)),("--checkpoint-dir",str(cp)),("--output",str(output/"curvature_audit.json")),("--max-batches","1")):
        command[command.index(flag)+1]=value
    shutil.copyfile(campaign/"source_manifest.json",work/"source_manifest.json")
    fixture_suite=copy.deepcopy(suite)
    fixture_suite["source_manifest"]=str(work/"source_manifest.json")
    fixture_suite["jobs"]=[parent_job,job]
    save(work/"suite.json",fixture_suite)
    validator.HERE=work
    common=sys.modules[validator.inspect_metrics.__module__]
    common.HERE=work
    # No stubbed parent validator: establish a real valid recursive receipt.
    parent_actual=validator.validate_job(parent_job)
    assert parent_actual["summary"]["status"]=="completed"
    save(work/"runtime/jobs"/parent_id/"completed.json",{
        "job_id":parent_id,"exit":{"returncode":0},**parent_actual})
    report={"schema_version":1,"status":"completed","task":0,"seed":2,
            "checkpoint_dir":str(cp),"checkpoint_files_sha256":ca.pin_tree(cp),
            "checkpoint_manifest":manifest,"source_python_sha256":source_pins,
            "script_sha256":ca.sha(campaign/"diagnostics/curvature_audit.py"),
            "config_sha256":ca.sha(config),"class_range":[0,20],"restored_model_sha256":ca.parameter_sha(model),
            "coordinate_system":"trainable LoRA factors plus FC; not the effective-BA Hessian",
            "selection":"predeclared fixed batches, no high-alignment selection",
            "hvp_backend_requested":"auto","batches":[batch],"checkpoint_unchanged":True,
            "source_unchanged":True,"hvp_all_available":True,
            "interpretation":"CPU schema test fixture; never an experimental result"}
    path=output/"curvature_audit.json"
    save(path,report)
    return job,report,path,cp


def test_curvature(campaign,validator,root):
    root.mkdir(parents=True)
    job,valid,path,cp=curvature_fixture(campaign,root,validator)
    checks={}
    # An invalid schema or KeyError here cannot be misreported as a passed negative.
    accepted=validator.validate_job(job)
    assert accepted["summary"]["status"]=="completed"
    checks["legal_available_result"]="ACCEPTED"
    unsupported=copy.deepcopy(valid)
    unsupported["hvp_all_available"]=False
    for response in unsupported["batches"][0]["responses"].values():
        response["hvp"]={"status":"unsupported","backend":None,"direction_detached":True,
            "attempts":[{"backend":b,"reason":"derivative for test backward is not implemented",
                         "unsupported_double_backward":True} for b in ("native","math")],
            "interpretation":"Finite differences only; no autograd-HVP verification claim"}
        response["native_finite_response_vs_hvp"]=None
        for scan in response["finite_difference_scan"]:
            scan["forward_vs_hvp"]=scan["central_vs_hvp"]=None
    save(path,unsupported)
    assert validator.validate_job(job)["summary"]["hvp_all_available"] is False
    checks["legal_explicit_unsupported_result"]="ACCEPTED"

    def rejected(name,mutate,base=None):
        bad=copy.deepcopy(valid if base is None else base)
        mutate(bad)
        save(path,bad)
        try:
            validator.validate_job(job)
        except (ValueError,AssertionError,RuntimeError) as exc:
            checks[name]={"status":"REJECTED","reason":str(exc)}
        else:
            checks[name]={"status":"INCORRECTLY_ACCEPTED"}
        # Intentionally no catch of KeyError/TypeError/FileNotFoundError.

    rejected("nan",lambda s:s["batches"][0]["responses"]["original"]["native_finite_response_vs_hvp"].update(cosine=float("nan")))
    rejected("wrong_origin_rule",lambda s:s["checkpoint_manifest"].update(component_rule="uo"))
    alternate=root/"different_checkpoint";shutil.copytree(cp,alternate)
    rejected("wrong_origin_path",lambda s:s.update(checkpoint_dir=str(alternate)))
    source_key=next(iter(valid["source_python_sha256"]))
    rejected("wrong_source_hash",lambda s:s["source_python_sha256"].update({source_key:"0"*64}))
    rejected("wrong_closure_counter",lambda s:s["batches"][0]["native"].update(closure_count=3))
    rejected("wrong_ce_counter",lambda s:s["batches"][0]["native"].update(training_ce_gradient_passes=4))
    rejected("failed_restore",lambda s:s["batches"][0]["restore"].update(rng_exact=False))
    rejected("empty_restore",lambda s:s["batches"][0].update(restore={}))
    rejected("wrong_native_point",lambda s:s["batches"][0]["native"]["native_point_checks"].update(g2_native_roundtrip_outer_exact=False))
    rejected("missing_response",lambda s:s["batches"][0]["responses"].pop("outer"))
    rejected("wrong_script_hash",lambda s:s.update(script_sha256="0"*64))
    rejected("hvp_availability_mismatch",lambda s:s.update(hvp_all_available=True),unsupported)
    rejected("unsupported_without_evidence",lambda s:s["batches"][0]["responses"]["original"]["hvp"].update(attempts=[]),unsupported)
    original=(cp/"lora_w_a_0.pt").read_bytes()
    (cp/"lora_w_a_0.pt").write_bytes(original+b"changed")
    rejected("changed_native_file",lambda s:None)
    (cp/"lora_w_a_0.pt").write_bytes(original)
    save(path,valid)
    assert validator.validate_job(job)["summary"]["status"]=="completed"
    checks["legal_result_after_negative_cleanup"]="ACCEPTED"
    failures=[name for name,result in checks.items() if isinstance(result,dict) and result["status"]!="REJECTED"]
    return {"status":"PASS" if not failures else "FAIL","checks":checks,"incorrectly_accepted":failures,
            "fixture_scope":"Actual CPU native-GAM curvature numbers inside a synthetic completion envelope; not scientific data"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--campaign",default=str(HERE))
    p.add_argument("--output",required=True)
    p.add_argument("--curvature-only",action="store_true")
    args=p.parse_args()
    if torch.cuda.is_available():raise RuntimeError("CPU-only test: set CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    campaign=Path(args.campaign).resolve();sys.path.insert(0,str(campaign))
    pinned=("validate_job.py","common.py","diagnostics/curvature_audit.py",
            "review/horizon_validation_fixture.py","trainer/component_horizon.py",
            "trainer/effective_step.py","trainer/task0_trainer_base_v2.py","trainer/test_component_cpu.py")
    before={name:file_sha(campaign/name) for name in pinned}
    validator=module_from("horizon_completion_validator_under_test",campaign/"validate_job.py")
    with tempfile.TemporaryDirectory(prefix="horizon-validation-cpu-") as temp:
        work=Path(temp)
        result={"curvature":test_curvature(campaign,validator,work/"curvature")}
        if not args.curvature_only:
            helper_path=campaign/"review/horizon_validation_fixture.py"
            if not helper_path.is_file():raise FileNotFoundError("Horizon fixture helper not ready")
            helper=module_from("horizon_validation_fixture",helper_path)
            horizon=helper.run_tests(validator)
            horizon["status"]="PASS" if horizon["status"] in ("passed","PASS") else "FAIL"
            result["horizon"]=horizon
    after={name:file_sha(campaign/name) for name in pinned}
    assert before==after,"Imported implementation changed during CPU validation; rerun after freeze"
    result["status"]="PASS" if all(v["status"]=="PASS" for v in result.values()) else "FAIL"
    result["implementation_pins_sha256"]=before
    result["validator_sha256"]=validator.sha(campaign/"validate_job.py")
    result["test_script_sha256"]=validator.sha(__file__)
    save(args.output,result)
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise AssertionError("Completion validator accepted a corrupt result")


if __name__=="__main__":main()
