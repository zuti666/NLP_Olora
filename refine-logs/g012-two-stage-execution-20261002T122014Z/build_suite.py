"""Materialize only the authorized core; no GPU launch."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import socket
from common import atomic_json,sha256,load_suite

H=Path(__file__).resolve().parent
OLD=Path("/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r025-c21-ab-20261002T080000Z")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage2-backend",choices=("native","math"),required=True)
    args=ap.parse_args()
    previous=json.loads((OLD/"suite.json").read_text())
    raw=json.loads((OLD/"configs/screen_c21_base_natural_r025_s2.json").read_text())
    raw.update(compute_nme_class_means=False,train_num_workers=8,eval_num_workers=4)
    env=copy.deepcopy(previous["environment"])
    tmp="/tmp/g012-two-stage-20261002T122014Z"
    for k in ("TMPDIR","TMP","TEMP"):env[k]=tmp
    suite=dict(schema_version=1,run_id=H.name,run_dir=str(H),execution_hostname=socket.gethostname(),
               python=previous["python"],source_root=previous["source_root"],
               source_manifest=str(H/"source_manifest.json"),allowed_gpus=[2,3,4,5,6],max_parallel=5,
               poll_seconds=15,gpu_lock_dir="/tmp/olora_gamfs_gpu_locks",
               gpu_free_threshold_mib=500,gpu_free_max_util_percent=4,
               expected_tasks=10,environment=env,no_external_notifications=True,
               stage2_backend=args.stage2_backend,
               failure_policy="halt launches; preserve artifacts; active jobs finish",
               jobs=[])
    shutil.copy2(OLD/"source_manifest.json",H/"source_manifest.json")
    suite["source_manifest_sha256"]=sha256(H/"source_manifest.json")
    specs=[
      ("S1-01_CE_natural_s2","P","g0","natural","M1"),
      ("S1-02_Splus_natural_s2","P","g1_plus","natural","M1"),
      ("S1-03_Sminus_natural_s2","P","g1_minus","natural","M1"),
      ("S1-04_O_P_natural_s2","P","g2","natural","M1"),
      ("S1-05_O_D_natural_s2","D","g2","natural","M1"),
      ("S1-06_O_C_natural_s2","C","g2","natural","M1"),
      ("S1-07_A_P_natural_s2","P","A","natural","M1"),
      ("S1-08_A_D_natural_s2","D","A","natural","M1"),
      ("S1-09_A_C_natural_s2","C","A","natural","M1"),
      ("S1-10_O_R_natural_s2","R","g2","natural","M2"),
      ("S1-11_A_R_natural_s2","R","A","natural","M2"),
      ("S1-12_A_P_matched_s2","P","A","matched","M2"),
      ("S1-13_A_D_matched_s2","D","A","matched","M2"),
      ("S1-14_A_C_matched_s2","C","A","matched","M2"),
      ("S1-15_A_R_matched_s2","R","A","matched","M2"),
      ("S1-16_CE_matched_s2","P","g0","matched","M2")]
    stage2=[
      ("S2-01_Hstart_natural_s2","P","hstart","natural","M4"),
      ("S2-02_Hmid_natural_s2","P","hmid","natural","M4"),
      ("S2-03_Hstart_matched_s2","P","hstart","matched","M4"),
      ("S2-04_Hmid_matched_s2","P","hmid","matched","M4"),
      ("S2-05_Residual_natural_s2","P","residual_only","natural","M4"),
      ("S2-06_Residual_matched_s2","P","residual_only","matched","M4")]
    if args.stage2_backend=="math":
        stage2 += [
          ("S2-B01_Sec_math_natural_s2","P","A","natural","M4"),
          ("S2-B02_Sec_math_matched_s2","P","A","matched","M4")]
    first=[s[0] for s in specs if s[-1]=="M1"]
    all_stage1=[s[0] for s in specs]
    smoke_stage1=["smoke_"+s[0] for s in specs]
    smoke_stage2=["smoke_"+s[0] for s in stage2]
    def make(spec,backend,smoke,depends):
        ident,branch,direction,mode,phase=spec
        if smoke:ident="smoke_"+ident
        output=H/"outputs"/ident
        cfg=copy.deepcopy(raw)
        cfg.update(prefix=ident,outputs_root=str(output/"native"),
                   probe_branch=branch,update_direction=direction,step_mode=mode,
                   attention_backend=backend,component_rule="u0",component_scale="natural")
        config=H/"configs"/(ident+".json")
        atomic_json(config,cfg)
        command=[suite["python"],str(H/"trainer/four_probe_train.py"),"--source",suite["source_root"],
                 "--config",str(config),"--output-dir",str(output)]
        if smoke:command += ["--max-tasks","2","--max-epochs","1","--max-batches","1"]
        return dict(id=ident,kind="training",phase="smoke_"+phase if smoke else phase,
                    branch=branch,direction=direction,mode=mode,backend=backend,smoke=smoke,
                    command=command,config=str(config),config_sha256=sha256(config),
                    output_dir=str(output),depends_on=depends,
                    expected_tasks=2 if smoke else 10,
                    expected_epochs_per_task=1 if smoke else 50,
                    expected_steps=2 if smoke else 2550)
    for spec in specs:
        suite["jobs"].append(make(spec,"native",True,[]))
    for spec in specs:
        suite["jobs"].append(make(spec,"native",False,smoke_stage1+([] if spec[-1]=="M1" else first)))
    diag_out=H/"outputs/Stage1_mechanism_diagnostics"
    suite["jobs"].append(dict(id="Stage1_mechanism_diagnostics",kind="diagnostic",phase="M2_diagnostic",smoke=False,
       backend="native",response=False,report_file="stage1_mechanism_report.json",output_dir=str(diag_out),depends_on=all_stage1,
       command=[suite["python"],str(H/"trainer/mechanism_diagnostics.py"),"--source",suite["source_root"],
                "--reference-output",str(H/"outputs/S1-07_A_P_natural_s2"),"--output-dir",str(diag_out),"--device","cuda:0"]))
    gate_out=H/"outputs/HVP_full_gate"
    gate=dict(id="HVP_full_gate",kind="hvp_gate",phase="M3",smoke=False,expected_tasks=3,
              backend=args.stage2_backend,output_dir=str(gate_out),depends_on=all_stage1,
              command=[suite["python"],str(H/"trainer/hvp_gate.py"),"--source",suite["source_root"],
                       "--reference-output",str(H/"outputs/S1-07_A_P_natural_s2"),
                       "--output-dir",str(gate_out),"--backend",args.stage2_backend,
                       "--tasks","0","4","9","--max-batches","2"])
    suite["jobs"].append(gate)
    for spec in stage2:
        suite["jobs"].append(make(spec,args.stage2_backend,True,["HVP_full_gate"]))
    for spec in stage2:
        suite["jobs"].append(make(spec,args.stage2_backend,False,["HVP_full_gate"]+smoke_stage2))
    diag_out=H/"outputs/Response_mechanism_diagnostics"
    suite["jobs"].append(dict(id="Response_mechanism_diagnostics",kind="diagnostic",phase="M4_diagnostic",smoke=False,
       backend=args.stage2_backend,response=True,report_file="response_mechanism_report.json",output_dir=str(diag_out),
       depends_on=["HVP_full_gate"]+[s[0] for s in stage2],
       command=[suite["python"],str(H/"trainer/mechanism_diagnostics.py"),"--source",suite["source_root"],
                "--reference-output",str(H/"outputs/S1-07_A_P_natural_s2"),"--output-dir",str(diag_out),"--device","cuda:0",
                "--backend",args.stage2_backend,"--response-diagnostics"]))
    atomic_json(H/"suite.json",suite)
    load_suite(H/"suite.json")
    print(json.dumps(dict(status="materialized",formal=len(specs)+len(stage2),
                          smokes=len(specs)+len(stage2),hvp_gates=1,allowed_gpus=suite["allowed_gpus"])))

if __name__=="__main__":
    main()
