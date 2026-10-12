"""Four new endpoints and two explicitly reused deterministic baselines."""
import csv
import io
import json
from pathlib import Path

from common import HERE, PROTOCOL, atomic_json, atomic_text, expected_budget, now, read_json, sha256, verify_cpu_gate
from validate_job import validate_job, validate_reuse


FACTOR_FIELDS = ("first_location_estimator", "second_residual_estimator", "residual_coefficient", "update_base")
METRIC_FIELDS = ("AAA_task_macro", "FAA_task_macro", "LA", "BWT", "forgetting")


def make_row(condition, variant, factors, formula, reused=False):
    budget = expected_budget(factors)
    return dict(condition=condition, variant=variant, radius=1., rho=.05, seed=2, rank=10,
                **factors, formula=formula, training_ce_per_step=budget["train_ce_per_update"],
                extra_diagnostic_gradient_calls=0 if not reused else 10,
                diagnostic_protocol="training_context_only_no_extra_gradient_v3" if not reused else "selected_training_g3_reuse_opposite_only_v2",
                step_mode="natural", protocol=PROTOCOL, source_protocol="", source_campaign="",
                status="pending", accepted=False, reused=reused, tasks_completed=0,
                AAA=None, FAA=None, LA=None, BWT=None, forgetting=None,
                progress_AAA=None, progress_FAA=None, progress_is_final=False,
                source_config="", source_receipt="", source_summary="", source_matrix="", source_trace="",
                config_sha256="", receipt_sha256="", summary_sha256="", matrix_sha256="", trace_sha256="", error="")


def fill_metrics(row, values):
    row.update(AAA=values["AAA_task_macro"], FAA=values["FAA_task_macro"],
               LA=values["LA"], BWT=values["BWT"], forgetting=values["forgetting"],
               accepted=True, tasks_completed=10, progress_is_final=True)


def main():
    suite = read_json(HERE / "suite.json")
    if suite.get("protocol") != PROTOCOL or len(suite.get("reused_results", [])) != 2:
        raise AssertionError("Expected four new v3 conditions and two explicit reused baselines")
    cpu_gate = verify_cpu_gate(suite)
    regression_verified = cpu_gate["checks"].get("ff_fb_same_state_regression") is True
    state_path = HERE / "runtime/queue_state.json"
    state = read_json(state_path) if state_path.exists() else {"jobs": {}}
    rows = []
    for job in suite["jobs"]:
        if job["kind"] != "training" or job["smoke"]:
            continue
        factors = {key:job[key] for key in FACTOR_FIELDS}
        row = make_row(job["condition_id"], job["variant"], factors, job["formula"])
        out = Path(job["output_dir"])
        row.update(source_protocol=suite["protocol"], source_campaign=str(HERE), source_config=job["config"],
                   config_sha256=job["config_sha256"],
                   source_receipt=str(HERE / "runtime/jobs" / job["id"] / "completed.json"),
                   source_summary=str(out / "final_summary.json"), source_matrix=str(out / "task_matrix.json"),
                   source_trace=str(out / "step_trace.jsonl"),
                   status=state["jobs"].get(job["id"], {}).get("status", "pending"))
        if Path(row["source_matrix"]).exists():
            matrix = read_json(row["source_matrix"])
            row.update(tasks_completed=len(matrix.get("matrix", [])), progress_AAA=matrix.get("AAA_task_macro"),
                       progress_FAA=matrix.get("FAA_task_macro"))
        if row["status"] == "completed":
            try:
                checked = validate_job(job)
                receipt = read_json(row["source_receipt"])
                if any(receipt[key] != checked[key] for key in ("summary_sha256", "trace_sha256", "matrix_sha256")):
                    raise AssertionError("Endpoint receipt differs from the validator")
                values = read_json(row["source_matrix"])
                fill_metrics(row, values)
                row.update(receipt_sha256=sha256(row["source_receipt"]), summary_sha256=checked["summary_sha256"],
                           matrix_sha256=checked["matrix_sha256"], trace_sha256=checked["trace_sha256"])
            except Exception as exc:
                row.update(status="rejected_evidence", error=repr(exc))
        rows.append(row)
    if len(rows) != 4 or {row["variant"] for row in rows} != {"CPLUS", "CMINUS", "BF", "BB"}:
        raise AssertionError("New formal factor grid is incomplete")
    for item in suite["reused_results"]:
        variant = item["variant"]
        second = "forward" if variant == "FF" else "backward_corrected"
        factors = dict(first_location_estimator="forward", second_residual_estimator=second,
                       residual_coefficient=1, update_base="g0")
        formula = "g0+g3Fplus-g2F" if variant == "FF" else "g0+g2F-g3Fminus"
        row = make_row(item["job_id"], variant, factors, formula, reused=True)
        paths, hashes = item["evidence_paths"], item["evidence_sha256"]
        row.update(status="reused_reference", source_protocol=item["source_protocol"], source_campaign=item["source_campaign"])
        for key in ("config", "receipt", "summary", "matrix", "trace"):
            row["source_"+key], row[key+"_sha256"] = paths[key], hashes[key]
        try:
            checked = validate_reuse(suite, item)
            fill_metrics(row, checked)
        except Exception as exc:
            row.update(status="rejected_reuse_evidence", error=repr(exc))
        rows.append(row)
    order = {key:index for index,key in enumerate(("FF", "FB", "CPLUS", "CMINUS", "BF", "BB"))}
    rows.sort(key=lambda row: order[row["variant"]])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(HERE / "summary.csv", buffer.getvalue())
    complete = {row["variant"]:row for row in rows if row["accepted"]}
    contrasts = []
    if regression_verified:
        for left,right in (("CPLUS", "FF"), ("CPLUS", "FB"), ("CMINUS", "CPLUS"),
                           ("BF", "FF"), ("BB", "FB"), ("BB", "BF"), ("FB", "FF")):
            if left in complete and right in complete:
                a,b = complete[left],complete[right]
                contrasts.append(dict(radius=1., candidate=left, reference=right, paired_seed=2, paired_n=1,
                    delta_AAA_pp=a["AAA"]-b["AAA"], delta_FAA_pp=a["FAA"]-b["FAA"], sample_sd=None,
                    amplitude_controlled=False, training_ce_budget_equal=a["training_ce_per_step"]==b["training_ce_per_step"],
                    candidate_training_ce=a["training_ce_per_step"], reference_training_ce=b["training_ce_per_step"],
                    contains_reused_reference=a["reused"] or b["reused"], ff_fb_same_state_regression_verified=True))
    atomic_json(HERE / "results.json", dict(updated_at=now(), expected_conditions=6, new_formal_expected=4,
                reused_expected=2, accepted_conditions=sum(row["accepted"] for row in rows),
                new_formal_accepted=sum(row["accepted"] and not row["reused"] for row in rows),
                reused_accepted=sum(row["accepted"] and row["reused"] for row in rows),
                ff_fb_same_state_regression_verified=regression_verified, rows=rows, contrasts=contrasts))
    lines = ["# ImageNet-A末端中心差分及联合后向替换结果", "",
             "seed2/rank10/rho=.05/r=1；当前学习率、自然SGD及严格CUDA确定性协议保持固定。新增CPLUS/CMINUS每步5训练CE，BF/BB每步4训练CE，均无额外梯度诊断。BB不读取任何局部上升侧梯度。关闭Fisher、实际权重融合、投影、HVP及最终步长匹配。", "",
             "FF/FB为已验收v2成绩的显式复用，保留原始来源；不是本轮新增训练。新旧FF/FB同状态点、梯度及实际更新回归检查已通过。旧v2每任务有一次隔离反侧诊断，新v3没有；完整状态/RNG恢复防止该诊断进入训练更新。", "",
             "| ID | 最终方向 | 训练CE/步 | 状态 | AAA | FAA | 来源 |",
             "|---|---|---:|---|---:|---:|---|"]
    fmt = lambda value:"—" if value is None else f"{value:.3f}"
    for row in rows:
        lines.append(f"| {row['variant']} | {row['formula']} | {row['training_ce_per_step']} | {row['status']} | {fmt(row['AAA'])} | {fmt(row['FAA'])} | {'已验收旧基线复用' if row['reused'] else '新独立训练'} |")
    lines += ["", "AAA/FAA只填写完整验收终点，部分任务观察独立写入CSV的progress字段；没有将短测当作正式结果。所有比较为单seed，样本标准差不可估计；自然步长未控制更新幅度，中心组与单侧组的梯度调用预算也不同。", "",
              "中心差分检验删除偶对称项后的符号作用；BF/BB与FF/FB形成第一定位估计×第二残差估计的联合对照。BB成功仅支持当前分量版本的两次局部侧别联合替换，不证明完整GAM等价或计算加速。"]
    atomic_text(HERE / "RESULTS.md", "\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
