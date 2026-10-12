"""Separate natural/matched, native/math, smoke/formal, and partial/final."""
import csv
import io
import json
from pathlib import Path
from common import atomic_text,atomic_json,now

H=Path(__file__).resolve().parent

def main():
    suite=json.loads((H/"suite.json").read_text())
    state=json.loads((H/"runtime/queue_state.json").read_text()) if (H/"runtime/queue_state.json").exists() else {"jobs":{}}
    rows=[]
    for job in suite["jobs"]:
        if job["kind"]!="training" or job["smoke"]:
            continue
        record=state["jobs"].get(job["id"],{})
        matrix_path=Path(job["output_dir"])/"task_matrix.json"
        matrix=json.loads(matrix_path.read_text()) if matrix_path.exists() else {}
        row=dict(job=job["id"],phase=job["phase"],branch=job["branch"],direction=job["direction"],
                 mode=job["mode"],backend=job["backend"],seed=2,status=record.get("status","pending"),
                 tasks=len(matrix.get("matrix",[])),AAA=matrix.get("AAA_task_macro"),FAA=matrix.get("FAA_task_macro"),
                 LA=matrix.get("LA"),BWT=matrix.get("BWT"),forgetting=matrix.get("forgetting"),output_dir=job["output_dir"])
        rows.append(row)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(H/"summary.csv",stream.getvalue())
    atomic_json(H/"results.json",dict(updated_at=now(),rows=rows))
    lines=["# 两阶段实验运行结果","","更新时间："+now()+"。",
           "固定seed2、rank10、r=.25、rho=.05；AAA/FAA任务等权。未完成训练的阶段指标只能看进度，不能作为最终比较。",
           "","| 条件 | 后端 | 步长 | 状态 | 已完成任务 | AAA | FAA |","|---|---|---|---|---:|---:|---:|"]
    for r in rows:
        fmt=lambda v:"—" if v is None else "%.3f"%v
        lines.append("| "+r["job"]+" | "+r["backend"]+" | "+r["mode"]+" | "+r["status"]+" | "+str(r["tasks"])+" | "+fmt(r["AAA"])+" | "+fmt(r["FAA"])+" |")
    atomic_text(H/"RESULTS.md","\n".join(lines)+"\n")

if __name__=="__main__":
    main()
