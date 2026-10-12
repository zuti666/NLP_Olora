"""Collect 14 new endpoints plus two explicitly validated, pinned B1 reuses."""
import csv
import io
import json
from pathlib import Path

from common import HERE, atomic_json, atomic_text, now, read_json, sha256
from validate_job import validate_job, validate_reuse


def matrix_metrics(path):
    record = read_json(path)
    return {key: record[key] for key in ("AAA_task_macro", "FAA_task_macro", "LA", "BWT", "forgetting")}


def make_row(condition_id, variant, radius, base, probe_sign, residual_sign, formula):
    return dict(condition=condition_id, variant=variant, radius=radius, rho=.05, seed=2, rank=10,
                update_base=base, second_probe_sign=probe_sign, second_difference_sign=residual_sign,
                formula=formula, step_mode="natural", protocol="four_probe_second_probe_sign_exact_restore_rng_replay_v1",
                source_protocol="", status="pending", accepted=False, reused=False, tasks_completed=0,
                AAA=None, FAA=None, LA=None, BWT=None, forgetting=None,
                progress_AAA=None, progress_FAA=None, progress_is_final=False,
                source_config="", source_receipt="", source_summary="", source_matrix="", source_trace="",
                config_sha256="", receipt_sha256="", summary_sha256="", matrix_sha256="", trace_sha256="",
                error="")


def main():
    suite = read_json(HERE / "suite.json")
    state_path = HERE / "runtime/queue_state.json"
    state = read_json(state_path) if state_path.exists() else {"jobs": {}}
    rows = []
    for job in suite["jobs"]:
        if job["kind"] != "training" or job["smoke"]:
            continue
        row = make_row(job["condition_id"], job["variant"], job["radius"], job["update_base"],
                       job["second_probe_sign"], job["second_difference_sign"], job["formula"])
        out = Path(job["output_dir"])
        row.update(source_protocol=suite["protocol"], source_config=job["config"], config_sha256=job["config_sha256"],
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
                if receipt["summary_sha256"] != checked["summary_sha256"]:
                    raise AssertionError("Endpoint receipt differs from the validator")
                values = matrix_metrics(row["source_matrix"])
                row.update(accepted=True, tasks_completed=10, AAA=values["AAA_task_macro"], FAA=values["FAA_task_macro"],
                           LA=values["LA"], BWT=values["BWT"], forgetting=values["forgetting"],
                           receipt_sha256=sha256(row["source_receipt"]), summary_sha256=checked["summary_sha256"],
                           matrix_sha256=checked["matrix_sha256"], trace_sha256=checked["trace_sha256"],
                           progress_is_final=True)
            except Exception as exc:
                row.update(status="rejected_evidence", error=repr(exc))
        rows.append(row)
    for item in suite["reused_results"]:
        row = make_row(item["condition_id"], "B1", item["radius"], "g1", 1, 1, item["formula"])
        row.update(reused=True, status="reuse_pending_gate", source_protocol="four_probe_g1_plus_second_difference_exact_restore_rng_replay_v1",
                   **{key: item[key] for key in ("source_config", "source_receipt", "source_summary", "source_matrix", "source_trace")})
        try:
            checked = validate_reuse(suite, item)
            values = matrix_metrics(item["source_matrix"])
            row.update(status="completed_reused", accepted=True, tasks_completed=10,
                       AAA=values["AAA_task_macro"], FAA=values["FAA_task_macro"], LA=values["LA"],
                       BWT=values["BWT"], forgetting=values["forgetting"],
                       config_sha256=sha256(item["source_config"]), receipt_sha256=sha256(item["source_receipt"]),
                       summary_sha256=checked["summary_sha256"], matrix_sha256=checked["matrix_sha256"],
                       trace_sha256=checked["trace_sha256"])
        except Exception as exc:
            row.update(error=repr(exc))
        rows.append(row)
    rows.sort(key=lambda row: (-row["radius"], row["variant"]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(HERE / "summary.csv", buffer.getvalue())
    complete = [row for row in rows if row["accepted"]]
    contrasts = []
    for radius in (1., .25):
        by_variant = {row["variant"]: row for row in complete if row["radius"] == radius}
        for left, right in (("N5", "B0"), ("N6", "B1"), ("N1", "B0"), ("N2", "B1"),
                            ("N5", "N3"), ("N6", "N4")):
            if left in by_variant and right in by_variant:
                contrasts.append(dict(radius=radius, candidate=left, reference=right, paired_seed=2, paired_n=1,
                                      delta_AAA_pp=by_variant[left]["AAA"]-by_variant[right]["AAA"],
                                      delta_FAA_pp=by_variant[left]["FAA"]-by_variant[right]["FAA"],
                                      sample_sd=None, amplitude_controlled=False))
    atomic_json(HERE / "results.json", dict(updated_at=now(), expected_conditions=16,
                new_formal_expected=14, reused_expected=2, accepted_conditions=len(complete),
                new_formal_accepted=sum(row["accepted"] and not row["reused"] for row in rows),
                reused_accepted=sum(row["accepted"] and row["reused"] for row in rows), rows=rows, contrasts=contrasts))
    lines = ["# ImageNet-A第二内层探针符号消融结果", "",
             "seed2/rank10/rho=.05；前三个梯度沿正向P几何固定。每步4训练CE＋一次自然SGD，隔离每任务2次侧别诊断，不计入训练CE。无Fisher、实际融合、投影、HVP或最终匹配。", "",
             "| r | ID | 最终方向 | 状态 | AAA | FAA | 来源 |", "|---:|---|---|---|---:|---:|---|"]
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    for row in rows:
        lines.append(f"| {row['radius']} | {row['variant']} | {row['formula']} | {row['status']} | {fmt(row['AAA'])} | {fmt(row['FAA'])} | {'已验收旧B1复用' if row['reused'] else '新独立训练'} |")
    lines += ["", "只将完整验收终点填入AAA/FAA；部分任务值独立放在CSV progress字段。所有比较为单seed、自然步长，方向和有效更新幅度尚未分离。", "",
              "B1复用要求旧validator和receipt一致、旧证据哈希未改、CPU old/new同状态等价gate通过；旧五CE的g0+d32只在计划中作历史参考，不参与本表配对。"]
    atomic_text(HERE / "RESULTS.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
