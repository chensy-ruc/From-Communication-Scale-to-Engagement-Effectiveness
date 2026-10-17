"""Read-only screening snapshots and local statistical reports; never call a model."""
from __future__ import annotations

import errno
import gzip
import json
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import pandas as pd

from . import ai_agents as agents
from . import ai_analysis, ai_audit, ai_audit_figures, ai_outcomes
from . import ai_client as client
from . import ai_data as data
from . import ai_figures as chart
from . import ai_screening as screen
from .ai_stops import latest_stop
from .analysis import boot_correlation, correct_p
from .api import StopCollection
from .coding import COMMENT_LABELS, agreement_metrics
from .core import now, read_csv, stable_hash, write_csv, write_json, write_text
from .progress import event


def capture(p, experiment):
    root = data.folder(p, experiment)
    spec = screen.protocol(p, root)
    plan = screen.load_plan(root, spec)
    snapshot = ai_audit.capture(root)
    inv = ai_audit.inventory(root, spec, snapshot, plans={"coder": snapshot["jobs"], "checker": plan["jobs"] if plan else []})
    old = agents.load_protocol(root)["protocol"]["configurations"] if (root / "agent_protocol.json").exists() else {}
    inv["historical_versions"] = {client.version(old[r]): "historical_" + r for r in ["reviewer", "adjudicator"] if r in old}
    inv["final_ready"] = inv["final_ready"] and plan is not None
    initial = None
    if inv["final_ready"]:
        first, checked = inv["complete_parents"]["coder"], inv["complete_parents"]["checker"]
        # Regenerate deterministically in memory to detect modified labels or a forged plan.
        expected = screen.selection_plan(root, spec, snapshot["jobs"], first, persist=False)
        if plan != expected:
            raise ValueError("检查计划与当前初编来源不一致，拒绝生成最终报告")
        inv["final"], initial = screen.predictions(spec, snapshot["jobs"], first, plan, checked)
    return root, spec, plan, snapshot, inv, initial


def status(p, experiment):
    _, _, plan, snapshot, inv, _ = capture(p, experiment)
    units = ai_audit.unit_table(inv)
    summary = ai_audit.unit_summary(units)
    print(summary.to_string(index=False))
    print(f"方案：单轮初编＋检查；检查计划{'已冻结' if plan else '等待初编完成'}；累计记录请求 {len(snapshot['attempts'])}。")
    stopped = latest_stop(snapshot["executions"], snapshot["attempts"])
    if stopped:
        print(f"最近停止：{stopped['stop_reason']}；{stopped['description']}。"
              f"阶段 {stopped['phase'] or '未记录'}；最近请求错误 {stopped['last_request_error'] or '无'}。")
    return summary


def audit(p, experiment, captured=None):
    root, spec, plan, snapshot, inv, _ = captured or capture(p, experiment)
    attempts = ai_audit.request_table(snapshot, inv)
    units = ai_audit.unit_table(inv)
    batches = ai_audit.batch_table(inv, attempts)
    comments, videos = read_csv(root / "comments.csv"), read_csv(root / "videos.csv")
    tables = {
        "request_attempts.csv": attempts,
        "request_errors.csv": attempts.loc[~attempts.status.isin(["complete", "started"])].groupby(
            ["role", "kind", "status"]).size().rename("requests").reset_index(),
        "batch_status.csv": batches, "batch_summary.csv": ai_audit.state_summary(batches, ["role", "kind"]),
        "unit_status.csv.gz": units, "unit_summary.csv": ai_audit.unit_summary(units),
        "unavailable_reasons.csv": ai_audit.unavailable_summary(units),
        "evidence_quarantine.csv": pd.DataFrame(snapshot["evidence_quarantine"], columns=[
            "version", "kind", "target_id", "proof", "execution_id", "created_at"]),
        "comment_label_distribution.csv": ai_audit.label_distribution(units, comments, inv["roles"]),
        "completion_by_stratum.csv": ai_audit.completion_strata(units, comments, videos, ("coder", "checker")),
        "executions.csv": pd.DataFrame(snapshot["executions"]),
        "execution_events.csv": pd.DataFrame(snapshot["execution_events"])}
    dest = p.reports / "ai" / experiment / "screening" / "audit" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    dest.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        write_csv(frame, dest / name)
    with gzip.open(dest / "ledger.jsonl.gz", "wt", encoding="utf-8") as output:
        for name, rows in snapshot.items():
            for row in rows:
                output.write(json.dumps({"table": name, "row": row}, ensure_ascii=False) + "\n")
    write_json(dest / "screening_protocol.json", spec)
    if plan:
        write_csv(pd.DataFrame(plan["selection"]), dest / "selection.csv")
    figures = ai_audit_figures.generate(dest, tables, inv["final_ready"], screening=True)
    summary = {"captured_at": now(), "workflow": spec["protocol"]["workflow"], "final_ready": inv["final_ready"],
               "plan_hash": plan["plan_hash"] if plan else None, "invalid_cache": inv["invalid"],
               "figure_count": len(figures["figures"]), "model_requests_by_this_audit": 0,
               "request_count_including_historical_roles": len(attempts),
               "latest_stop": latest_stop(snapshot["executions"], snapshot["attempts"]),
               "units": tables["unit_summary.csv"].to_dict("records"),
               "unavailable_reasons": tables["unavailable_reasons.csv"].to_dict("records"),
               "evidence_quarantine_admissions": len(snapshot["evidence_quarantine"]),
               "execution_policy_records": ai_outcomes.policy_records(root),
               "scope": "checker describes only selected units; no accuracy claim; historical reviewer/adjudicator costs retained"}
    write_json(dest / "summary.json", summary)
    lines = ["# 检查方案执行审计", "", "初编为全量；检查仅覆盖重点与随机抽中条目，未抽中不等于已验证。",
             "检查标签分布是选择性子集的描述，不能直接代表全体；失败、未开始、不可编码与弃权分别保留。",
             "请求表包含切换前已发生的 historical_reviewer/adjudicator 消耗，不计为新方案的检查结果。",
             "计数去重至材料单元；逐次请求、拆批、初编标签、检查标签和最终标签分别留痕。", "",
             "不可编码包含服务拒绝和有上限的逐字证据失败隔离；两种原因单列，不是模型弃权、成功或负标签。隔离是测量缺失策略修订，不是语义标签修复。",
             "[缺失原因、数量与分母](unavailable_reasons.csv) · [隔离来源请求及原文哈希](evidence_quarantine.csv)",
             "[逐次请求](request_attempts.csv) · [去重条目](unit_status.csv.gz) · [标签分母](comment_label_distribution.csv)"]
    if summary["latest_stop"]:
        stopped = summary["latest_stop"]
        lines += ["", f"最近停止：`{stopped['stop_reason']}`；{stopped['description']}。",
                  f"最近请求错误：`{stopped['last_request_error']}`；阶段：`{stopped['phase']}`。"]
    for figure in figures["figures"]:
        lines += ["", f"![{figure['title']}](figures/{figure['png']})", "", figure["caption"]]
    write_text(dest / "审计报告.md", "\n".join(lines) + "\n")
    parent = p.reports / "ai" / experiment
    relative = str(dest.relative_to(parent))
    link = relative + "/审计报告.md"
    write_json(parent / "audit_latest.json", {"created_at": summary["captured_at"], "workflow": "screening",
               "path": relative, "report": link, "final_ready": inv["final_ready"], "latest_stop": summary["latest_stop"]})
    write_text(parent / "审计概览.md", f"# 当前检查方案审计\n\n[最近审计]({link})；最终标签就绪：{inv['final_ready']}。\n")
    event("检查审计", f"已输出去重成功/失败、标签比例及 8 张审计图：{dest}")
    return summary, dest, figures


def check_metrics(plan, inv):
    changes = []
    for selected in plan["selection"]:
        if not selected["selected"]:
            continue
        kind, key = selected["kind"], selected["id"]
        aa = inv["values"].get(("coder", kind, key))
        bb = inv["values"].get(("checker", kind, key))
        for label in data.VIDEO_FIELDS if kind == "videos" else COMMENT_LABELS:
            a, b = aa["labels"][label] if aa else None, bb["labels"][label] if bb else None
            paired = agents.known(a) and agents.known(b)
            changes.append({"kind": kind, "id": key, "group": selected["group"], "label": label,
                            "initial": a, "checked": b, "paired_known": paired,
                            "agrees": a == b if paired else None,
                            "weight": 1 / selected["inclusion_probability"],
                            "unavailable": aa is None or bb is None})
    detail = pd.DataFrame(changes, columns=["kind", "id", "group", "label", "initial", "checked", "paired_known", "agrees", "weight", "unavailable"])
    return metrics_from_comparisons(detail), detail


def metrics_from_comparisons(detail):
    """Compute check statistics from frozen per-label comparisons, without a model ledger."""
    rows = []
    for (kind, group, label), part in detail.groupby(["kind", "group", "label"]):
        pairs = part.loc[part.paired_known]
        metric = agreement_metrics(pairs.initial.astype(str).to_numpy(), pairs.checked.astype(str).to_numpy())
        rows.append({"kind": kind, "group": group, "label": label, "selected_units": len(part),
                     "paired_known": len(pairs), "pair_coverage": len(pairs) / len(part),
                     "unavailable_units": int(part.unavailable.sum()),
                     "disagreements": int((~pairs.agrees.astype(bool)).sum()),
                     "agreement_on_resolved": metric["agreement"], "sample_kappa_ai_vs_ai": metric["kappa"],
                     "weighted_agreement_on_resolved": (float((pairs.agrees.astype(float) * pairs.weight).sum() / pairs.weight.sum()) if len(pairs) and group == "random" else None),
                     "scope": "nonrisk_codable_domain_only" if group == "random" else "targeted_subset_only"})
    return pd.DataFrame(rows, columns=["kind", "group", "label", "selected_units", "paired_known", "pair_coverage", "unavailable_units", "disagreements", "agreement_on_resolved", "sample_kappa_ai_vs_ai", "weighted_agreement_on_resolved", "scope"])


def check_chart(metrics, dest, note):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, group in zip(axes, ["risk", "random"], strict=True):
        part = metrics.loc[metrics.group.eq(group)].reset_index(drop=True)
        if part.empty:
            chart.empty(ax)
        else:
            values = pd.to_numeric(part.agreement_on_resolved, errors="coerce")
            good = values.notna()
            ax.barh(part.index[good], values[good], color=chart.TEAL)
            ax.set_yticks(part.index, [f"{chart.label(r.label)} (n={r.paired_known}/{r.selected_units})" for r in part.itertuples()])
            for i in part.index[~good]:
                ax.text(.03, i, chart.text("不可估计", "Not estimable"), va="center")
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05)
        ax.set_title(chart.text("重点复查" if group == "risk" else "非重点部分随机抽查", "Targeted checks" if group == "risk" else "Random nonrisk audit"))
    return chart.save(fig, dest, "screening_agreement", chart.text("检查样本的一致性", "Agreement in check samples"),
                      "n 为两次均可判断数/抽中数；风险与随机部分分别统计，不是全量信度或准确率。随机加权指标另见源表。", note, n=len(metrics))


def analyze(p, experiment):
    dest = p.reports / "ai" / experiment / "screening"
    state = {"started_at": now(), "status": "running", "model_requests": 0, "errors": []}
    retries = p.cfg.get("ai", {}).get("report_retry_rounds", 1)
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("report_retry_rounds 须为 0..2")
    for attempt in range(retries + 1):
        try:
            write_json(dest / "report_state.json", state)
            result = analyze_once(p, experiment)
            state.update(status="complete", completed_at=now())
            write_json(dest / "report_state.json", state)
            return result
        except BaseException as exc:
            state["errors"].append({"type": type(exc).__name__})
            retry = isinstance(exc, OSError) and exc.errno in {errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT} and attempt < retries
            state["status"] = "retrying" if retry else "incomplete"
            try:
                write_json(dest / "report_state.json", state)
            except OSError:
                pass
            if not retry:
                raise
            event("检查报告", "本机临时 IO 错误，重试报告；不请求模型。")


def analyze_once(p, experiment):
    captured = capture(p, experiment)
    root, spec, plan, _snapshot, inv, initial = captured
    if not inv["final_ready"]:
        raise StopCollection("初编或检查尚未完成；不能生成最终实验结论，请同命令续传")
    dest = p.reports / "ai" / experiment / "screening"
    dest.mkdir(parents=True, exist_ok=True)
    metrics, detail = check_metrics(plan, inv)
    write_csv(pd.DataFrame(plan["selection"]), dest / "selection.csv")
    write_csv(metrics, dest / "check_metrics.csv")
    write_csv(detail, dest / "label_comparisons.csv")
    gate = {**{"videos." + c: True for c in data.VIDEO_FIELDS}, **{"comments." + c: True for c in COMMENT_LABELS}}
    report = {"version": spec["protocol_hash"], "reference_type": "none_ai_only", "gate": gate, "workflow": "screening"}
    ai_analysis.analyze_predictions(p, root, inv["final"], report, dest / "analysis")
    correlations = pd.read_csv(dest / "analysis/correlations.csv")
    initial_frame, requirements = ai_analysis.analysis_frame(root, initial, report)
    write_csv(initial_frame, dest / "analysis/sensitivity_initial_videos.csv")
    sensitivity = []
    for variant, frame in [("screened", pd.read_csv(dest / "analysis/analysis_videos.csv")), ("initial", initial_frame)]:
        for metric in requirements:
            stats = (correlations.set_index("y").loc[metric, ["n", "rho", "p", "ci_low", "ci_high", "bootstrap_valid"]].to_dict() if variant == "screened" else
                     boot_correlation(frame, "view_count", metric, p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"]))
            sensitivity.append({"variant": variant, "metric": metric, **stats})
    sensitivity = correct_p(pd.DataFrame(sensitivity), "fdr_bh")
    write_csv(sensitivity, dest / "analysis/sensitivity_correlations.csv")
    note = chart.text("单轮初编＋选择性 AI 检查；未抽中不等于已验证，区间不含编码误差。",
                      "Single coding + selective AI checks; unchecked is unvalidated; intervals exclude coding error.")
    folder = dest / "analysis/figures"
    with chart.style():
        frame = pd.read_csv(dest / "analysis/analysis_videos.csv")
        figures = [chart.overview(frame, folder, note=note), check_chart(metrics, folder, note),
                   chart.coverage_chart(pd.read_csv(dest / "analysis/measurement_coverage.csv"), folder, note),
                   chart.scatter_chart(frame, correlations, folder, note),
                   chart.correlation_chart(sensitivity, folder, note, variants=[
                       ("screened", -.15, chart.BLUE, "o", chart.text("检查后", "After checks")),
                       ("initial", .15, chart.ORANGE, "D", chart.text("初编", "Initial coding"))]),
                   chart.groups_chart(frame, folder, note, p.cfg["analysis"]["min_group_n"]),
                   chart.regression_chart(pd.read_csv(dest / "analysis/models_coefficients.csv"), pd.read_csv(dest / "analysis/models_status.csv"), folder, note)]
    manifest = {"created_at": now(), "figure_version": "screening-1", "figures": figures,
                "source_hashes": {str(path.relative_to(dest)): stable_hash(path.read_bytes()) for path in
                                  [dest / "selection.csv", dest / "check_metrics.csv", *sorted((dest / "analysis").glob('*.csv'))]}}
    write_json(folder / "manifest.json", manifest)
    summary, audit_dest, audit_figures = audit(p, experiment, captured)
    counts = pd.Series([r["group"] for r in plan["selection"]]).value_counts().to_dict()
    write_json(dest / "screening_report.json", {"protocol_hash": spec["protocol_hash"], "plan_hash": plan["plan_hash"],
        "selection_counts": counts, "checks": len(plan["jobs"]), "human_reference": False,
        "figure_count": len(figures) + len(audit_figures["figures"]), "full_population_reliability": False})
    relative = str(audit_dest.relative_to(dest))
    lines = ["# 单轮初编与抽样检查实验结果（探索性）", "",
             f"重点复查 {counts.get('risk', 0):,} 个，随机抽查 {counts.get('random', 0):,} 个，未抽中 {counts.get('remaining', 0):,} 个。",
             "被采信的初编标签均经过程序字段、ID、逐字证据和逻辑校验。服务拒绝和反复证据失败隔离均单列测量缺失，不计成功。检查只处理冻结抽中的条目，不看初编标签。",
             "未抽中使用初编标签；抽中且两次判断相同的已知标签保留，分歧或弃权转为未决 null，不执行第三轮裁决。",
             "没有人工参考。重点部分的检查结果不代表总体；随机部分的推断范围仅是初编可编码的非重点部分。",
             "同模型可能存在共同错误；一致率不是准确率，未抽中不是已验证。不能将本方案写成全量双编码信度检验。",
             "随机抽查保留各层实际入样概率与权重；加权一致率仍限于两次均已判断项，缺失覆盖另列。",
             "若检查分歧较多，报告保留未决及敏感性差异，不隐蔽扩大全量复核或自动提高输出上限。", "",
             "[抽样与风险原因](selection.csv) · [分组检查指标](check_metrics.csv) · [逐标签对照](label_comparisons.csv)",
             f"[执行与用量审计]({relative}/审计报告.md)", "",
             "统计区间不包含模型误分类；评论表达不证明真实学习、分享或因果机制。",
             "主分析与初编敏感性分析的差异也可能来自未决造成的可观测样本变化。"]
    lines += chart.markdown(manifest)
    for figure in audit_figures["figures"]:
        lines += ["", f"![{figure['title']}]({relative}/figures/{figure['png']})", "", figure["caption"]]
    write_text(dest / "实验结果.md", "\n".join(lines) + "\n")
    event("完成", f"初编、检查、统计和 15 张图已保存：{dest / '实验结果.md'}")
    return summary
