"""Zero-human coding: blind model passes, targeted adjudication and resumable orchestration."""
from __future__ import annotations

import copy
import errno
import json
import os
import sys
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd

from . import ai_client as client
from . import ai_data as data
from . import ai_outcomes as outcomes
from . import ai_prompts
from .ai_stops import ProviderActionRequired, latest_provider_block
from .api import StopCollection
from .coding import COMMENT_LABELS, agreement_metrics
from .core import now, stable_hash, write_csv, write_json
from .progress import BatchProgress, event
from .progress import label as role_label

ROLE_PROMPTS = ai_prompts.ROLE_PROMPTS


def prepare(p, experiment="agents_v1", threads_per_video=0, transcripts=None):
    data.prepare(p, experiment, threads_per_video=threads_per_video, transcripts=transcripts, automated=True)


def check_mode(root):
    m = data.manifest(root)
    if m.get("reference_type") != "none_ai_only":
        raise ValueError("此目录属于人工验证方案；请用 agent-prepare --experiment agents_v1 建立全 AI 实验")
    return m


def automated_codebook(text):
    """Retain label definitions while replacing instructions meant for human record keeping."""
    paragraphs = []
    for paragraph in text.split("\n\n"):
        if "reviewed=1" in paragraph and "coder_id" in paragraph:
            paragraph = ("自动编码按请求中的 JSON Schema 返回标签和原文证据，不生成人工 reviewed/coder_id。"
                         "0 是经过判断后的不符合；缺少证据或裁决后仍不确定时保留 null，不强制猜测。")
        paragraph = paragraph.replace("正式编码前做样本外试编码，固定版本与判定规则。",
                                      "本自动协议固定标签定义，不要求人工试编码。")
        paragraph = paragraph.replace("视频编码员需观看内容并结合标题、说明；评论编码员阅读原始西语、视频标题及 context 文件。",
                                      "模型只使用实际提供的原文和线程语境；视频主题可依据标题/说明，形式没有字幕时必须为 null。")
        paragraph = paragraph.replace("无法明确时记 `other` 并写备注", "无法明确时记 null；`other` 仅用于明确属于其他类别的内容")
        paragraph = paragraph.replace("主线不属于上述或无法唯一归类，备注说明", "主线明确不属于上述类别；不能唯一归类时为 null")
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def prompt_configurations(root):
    instructions = (root / "instructions.txt").read_text()
    codebook = automated_codebook((root / "codebook.md").read_text())
    return {role: {"instructions": instructions + "\n" + prompt, "agent_role": role, "codebook": codebook,
                   "task_instructions": dict(ai_prompts.TASK_INSTRUCTIONS), "prompt_version": ai_prompts.VERSION,
                   "agent_protocol_version": 3, "adaptive_batches": True} for role, prompt in ROLE_PROMPTS.items()}


def configurations(p, root):
    prompts = prompt_configurations(root)
    roles = p.cfg.get("ai", {}).get("roles", {})
    allowed = {"thinking", "reasoning_effort", "max_completion_tokens"}
    if (not isinstance(roles, dict) or set(roles) - set(prompts)
            or any(not isinstance(v, dict) or set(v) - allowed for v in roles.values())):
        raise ValueError("ai.roles 仅支持 coder/reviewer/adjudicator 的 thinking、reasoning_effort、max_completion_tokens")
    configs = {}
    for role, prompt in prompts.items():
        base = client.settings(p, root, roles.get(role))
        configs[role] = {**base, **prompt, "model": os.getenv(f"AI_{role.upper()}_MODEL", "").strip() or base["model"]}
    return configs


def coder_configuration(p, root, preview=False):
    """Build only the active coder; retain its exact historical cache identity."""
    roles = p.cfg.get("ai", {}).get("roles", {})
    if not isinstance(roles, dict):
        raise ValueError("ai.roles 须为角色配置表")  # noqa: TRY004 - CLI configuration errors are ValueError
    overrides = roles.get("coder", {})
    if not isinstance(overrides, dict) or set(overrides) - {"thinking", "reasoning_effort", "max_completion_tokens"}:
        raise ValueError("ai.roles.coder 仅支持 thinking、reasoning_effort、max_completion_tokens")
    base = client.settings(p, root, overrides, preview=preview)
    return {**base, **prompt_configurations(root)["coder"],
            "model": os.getenv("AI_CODER_MODEL", "").strip() or base["model"]}


def uses_screening(p, root):
    """All public Agent commands default to two stages, including fresh experiments.

    The explicit legacy value is retained for historical reproduction only.
    A frozen screening experiment can never fall back to the old pipeline.
    """
    workflow = p.cfg.get("ai", {}).get("workflow", "screening")
    if workflow not in {"screening", "legacy_three_pass"}:
        raise ValueError("ai.workflow 须为 screening；legacy_three_pass 仅用于历史复现")
    return workflow == "screening" or (root / "screening_protocol.json").exists()


def export_prompts(p, experiment="agents_v1"):
    """Export effective system messages and schemas, including saved frozen versions, without credentials."""
    root = data.folder(p, experiment)
    if uses_screening(p, root):
        from . import ai_screening
        return ai_screening.export_prompts(p, experiment)
    check_mode(root)
    saved = (root / "agent_protocol.json").exists()
    if saved:
        configs = load_protocol(root)["protocol"]["configurations"]
    else:
        cfg = p.cfg.get("ai", {})
        preview = {"model": "LOCAL_PREVIEW_ONLY", "base_url": "https://example.invalid/v1",
                   "response_format": cfg.get("response_format", "json_schema"),
                   "token_limit_field": cfg.get("token_limit_field", "max_completion_tokens"),
                   "max_completion_tokens": cfg.get("max_completion_tokens", 12000)}
        configs = {role: {**preview, **prompt} for role, prompt in prompt_configurations(root).items()}
    dest = p.reports / "ai" / experiment / "prompts"
    dest.mkdir(parents=True, exist_ok=True)
    hashes, versions = {}, {}
    for role, config in configs.items():
        versions[role] = config.get("prompt_version", "legacy")
        for kind in ["videos", "comments"]:
            job = {"kind": kind, "payload": "{}"}
            system = client.request_body(config, job)["messages"][0]["content"]
            name = f"{role}_{kind}.system.txt"
            (dest / name).write_text(system, encoding="utf-8")
            hashes[name] = stable_hash(system)
    for kind in ["videos", "comments"]:
        write_json(dest / f"{kind}.schema.json", client.schema(kind))
    write_json(dest / "manifest.json", {"created_at": now(), "source": "frozen_protocol" if saved else "current_unfrozen_preview",
        "prompt_versions": versions, "system_hashes": hashes, "model_requests": 0,
        "note": "system 消息原文；真实材料在独立 user 消息中注入。json_schema 模式下 schema 是独立接口参数。"})
    print(f"已导出六份实际 system 提示词及两个 JSON Schema：{dest}；未调用模型。")


def load_protocol(root):
    check_mode(root)
    path = root / "agent_protocol.json"
    if not path.exists():
        raise ValueError("尚未开始 agent-run；模型配置将在首次运行时自动冻结")
    record = json.loads(path.read_text())
    protocol = record["protocol"]
    if record["protocol_hash"] != client.version(protocol):
        raise ValueError("全 AI 协议记录已改变，请恢复文件")
    if protocol["manifest_hash"] != stable_hash((root / "manifest.json").read_bytes()):
        raise ValueError("全 AI 输入清单已改变，请恢复原实验")
    if protocol["instructions_hash"] != stable_hash((root / "instructions.txt").read_bytes()):
        raise ValueError("全 AI 冻结后的提示词已改变，请恢复文件或创建新实验")
    return record


def freeze(p, root):
    check_mode(root)
    configs = configurations(p, root)
    protocol = {"manifest_hash": stable_hash((root / "manifest.json").read_bytes()),
                "instructions_hash": stable_hash((root / "instructions.txt").read_bytes()),
                "configurations": configs, "workflow_version": 2,
                "policy": "blind_double_code_then_full_unit_adjudication; unresolved=null; no_human_reference"}
    path = root / "agent_protocol.json"
    if path.exists():
        record = load_protocol(root)
        if record["protocol"] != protocol:
            raise ValueError("模型/接口/提示词/输出参数与自动冻结协议不同；恢复原配置或另建 --experiment")
    else:
        record = {"created_at": now(), "protocol": protocol, "protocol_hash": client.version(protocol)}
        write_json(path, record)
    for config in configs.values():
        dest = root / "versions" / (client.version(config) + ".json")
        if dest.exists() and json.loads(dest.read_text()) != config:
            raise ValueError("保存的 Agent 配置已改变，请恢复记录")
        if not dest.exists():
            write_json(dest, config)
    return record


def completed(root, ver):
    db = data.connect(root)
    try:
        return {r[0] for r in db.execute("SELECT job_id FROM results WHERE version=? AND status='complete'", (ver,))}
    finally:
        db.close()


def settled(root, ver):
    db = data.connect(root)
    try:
        return {r[0] for r in db.execute("SELECT job_id FROM results WHERE version=? AND status IN ('complete','unavailable')", (ver,))}
    finally:
        db.close()


def records(root, ver, tasks):
    db = data.connect(root)
    try:
        rows = {r["job_id"]: dict(r) for r in db.execute(
            "SELECT job_id,status,labels,response,error FROM results WHERE version=?", (ver,))}
    finally:
        db.close()
    pending = [job for job in tasks if rows.get(job["job_id"], {}).get("status") not in outcomes.SETTLED]
    if pending:
        recovering = sum(rows.get(job["job_id"], {}).get("error", "").startswith("split_required:") for job in pending)
        raise StopCollection(f"Agent 尚有未完成/失败任务：{len(pending)} 个基础批次（其中待恢复 {recovering} 个）；"
                             "本阶段处理完毕后才能进入下一阶段。重跑同一条 agent-run 命令继续")
    result = outcomes.Records()
    for job in tasks:
        row = rows.get(job["job_id"], {})
        outcomes.add_record(result, job, row, client.validate_response)
    return result


def known(value):
    return value is not None and pd.notna(value) and value != "unknown"


def make_adjudication(root, tasks, first, second, protocol, persist=True):
    """One adjudication batch per original job at most; full source contexts are preserved."""
    pending = []
    for job in tasks:
        payload = json.loads(job["payload"])
        aa = {x["id"]: x for x in first[job["job_id"]]}
        bb = {x["id"]: x for x in second[job["job_id"]]}
        sources = {r["id"]: r for r in payload["contexts"]} if job["kind"] == "videos" else {}
        differences = {}
        for target in payload["target_ids"]:
            if any((job["kind"], target) in outcomes.unavailable(pass_) for pass_ in [first, second]):
                continue  # No further provider requests for a unit with an unavailable blind measurement.
            labels = []
            for label, a in aa[target]["labels"].items():
                b = bb[target]["labels"][label]
                if job["kind"] == "videos" and label == "form" and not sources[target]["transcript"].strip():
                    continue  # An unavailable transcript is not a disagreement an agent can solve.
                if a != b or not known(a) or not known(b):
                    labels.append(label)
            if labels:
                differences[target] = labels
        if not differences:
            continue
        payload = data.target_subset(payload, list(differences))
        payload["disputed_labels"] = differences
        payload["candidate_annotations"] = {"coder": [aa[x] for x in differences],
                                             "reviewer": [bb[x] for x in differences]}
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        pending.append({"job_id": stable_hash(content), "parent_job_id": job["job_id"], "kind": job["kind"],
                        "split": "adjudication", "payload": content, "targets": json.dumps(list(differences))})
    if not persist:
        return pending
    record = {"protocol_hash": protocol["protocol_hash"],
              "input_hash": client.version({"coder": first, "reviewer": second}), "jobs": pending}
    path = root / "adjudication_jobs.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("裁决输入与已冻结的两轮编码不同，请恢复原始结果，不可混用裁决")
    if not path.exists():
        write_json(path, record)
    return pending


def run_stage(runner, tasks, workers, role):
    """Resume transient pauses within the same command and budget; never retry arbitrary exceptions."""
    deferred = 0
    while True:
        try:
            return run_stage_once(runner, tasks, workers, role)
        except StopCollection:
            recovery = runner.automatic_recovery
            if recovery.resume(runner):
                continue
            if runner.stop_reason == "unfinished_annotation_tasks" and deferred < recovery.deferred_rounds:
                recovery.check_budget(runner)
                deferred += 1
                runner.execution_event("deferred_tasks_retry", runner.stop_reason, deferred)
                event(runner.stage_label(role), f"其余任务已处理；第 {deferred}/{recovery.deferred_rounds} 轮补试未完成批次，复用已通过子批。")
                runner.stop_reason = ""
                continue
            raise


def run_stage_once(runner, tasks, workers, role):
    if type(workers) is not int or not 1 <= workers <= client.MAX_WORKERS:
        raise ValueError(f"AI workers 须为 1..{client.MAX_WORKERS} 的整数")
    runner.phase, runner.workers = role, workers
    done = settled(runner.root, runner.version)
    valid_done = completed(runner.root, runner.version)
    remaining = [j for j in tasks if j["job_id"] not in done]
    if not remaining:
        result = records(runner.root, runner.version, tasks)
        event(runner.stage_label(role), f"复用已处理 {len(tasks):,}/{len(tasks):,} 批（含不可编码记录），不新增请求。")
        return result
    cached_unavailable = sum(j["job_id"] in done - valid_done for j in tasks)
    progress = BatchProgress(runner.stage_label(role), len(tasks), sum(j["job_id"] in valid_done for j in tasks),
                             runner.progress_every, runner.maximum, cached_unavailable=cached_unavailable)
    progress.start(runner.count, workers)
    event(runner.stage_label(role), f"全局派发上限 {runner.rps:g} 次/秒；HTTP 连接池上限 {client.MAX_WORKERS}；"
          "线程共享限速、冷却和总请求预算。")
    thinking = runner.config.get("thinking", {}).get("type")
    effort = runner.config.get("reasoning_effort", "服务默认")
    mode = "关闭" if thinking == "disabled" or effort == "none" else f"{thinking or '服务默认'} / {effort}"
    event(runner.stage_label(role), f"单次请求（含子批）输出上限 {runner.config['max_completion_tokens']:,} tokens；"
          f"思考 {mode}；输出上限不是独立的思维链额度。")
    if not runner.maximum and not runner.output_limit:
        event(runner.stage_label(role), "累计请求数与累计输出 token 不设停止阈值；继续记录用量，单次生成限制和故障保护仍有效。")
    if runner.output_limit:
        event(runner.stage_label(role), f"本次命令共享输出 token 派发预算 {runner.output_limit:,}；"
              "按每次输出上限预留，已知用量返还余量，未知用量保留预留；此预算不含输入费用。")
    if runner.maximum and runner.count >= runner.maximum:
        runner.stop_reason = "request_budget_exhausted"
        progress.finish(sum(j["job_id"] in valid_done for j in tasks), runner.count, saved_unavailable=cached_unavailable)
        raise StopCollection("本次 AI 总请求预算已到，重跑同一命令续传")
    pool = ThreadPoolExecutor(max_workers=workers)
    queue, pending, stopped = iter(remaining), {}, None
    try:
        while True:
            while not runner.stop.is_set() and len(pending) < workers:
                job = next(queue, None)
                if job is None:
                    break
                pending[pool.submit(runner.call, job)] = job
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                try:
                    handled = future.result() is True
                    cached = runner.cached(job)
                    unavailable = bool(cached and cached["status"] == "unavailable")
                    success = handled and not unavailable
                    recovery = bool(cached and cached["error"].startswith("split_required:"))
                    progress.update(success, runner.count, len(pending), f"{job['kind']} {job['job_id'][:12]}", recovery, unavailable)
                except (ValueError, StopCollection) as exc:
                    runner.stop.set()
                    saved = runner.cached(job)
                    budget_pause = isinstance(exc, StopCollection) and runner.stop_reason in {
                        "request_budget_exhausted", "output_token_budget_exhausted"}
                    if saved and saved["status"] == "error" and not budget_pause:
                        progress.update(False, runner.count, len(pending), f"停止 {job['kind']} {job['job_id'][:12]}",
                                        saved["error"].startswith("split_required:"))
                    # Concurrent generic stop messages must not hide the actionable account diagnosis.
                    if (stopped is None or isinstance(exc, ValueError) or
                            isinstance(exc, ProviderActionRequired) and not isinstance(stopped, ValueError)):
                        stopped = exc
    except BaseException:
        runner.stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        db = data.connect(runner.root)
        try:
            saved = {r["job_id"]: ("recovery_pending" if r["error"].startswith("split_required:") else r["status"]) for r in db.execute(
                "SELECT job_id,status,error FROM results WHERE version=?", (runner.version,))}
        finally:
            db.close()
        progress.finish(sum(saved.get(j["job_id"]) == "complete" for j in tasks), runner.count,
                        sum(saved.get(j["job_id"]) == "error" for j in tasks),
                        sum(saved.get(j["job_id"]) == "recovery_pending" for j in tasks),
                        sum(saved.get(j["job_id"]) == "unavailable" for j in tasks))
        event(runner.stage_label(role), f"本次命令实际 HTTP 并发峰值 {runner.peak_inflight}；当前在途 HTTP {runner.inflight}。")
        if runner.output_limit:
            event("输出预算", f"本次已计入 {runner.output_committed:,}/{runner.output_limit:,} token；"
                  f"其中 {runner.output_unknown_requests} 次用量未知，按该次输出上限估计；实际费用以服务账单为准。")
    if stopped:
        if isinstance(stopped, StopCollection) and runner.stop_reason == "output_token_budget_exhausted":
            raise StopCollection("本次输出 token 派发预算不足，已保存进度；预算包含在途预留和未知用量估计，重跑续传") from stopped
        if isinstance(stopped, StopCollection) and runner.stop_reason == "consecutive_annotation_failures":
            raise StopCollection(f"{runner.stage_label(role)}：连续 {runner.stop_failure_count or runner.max_failures} 个任务校验失败保护已触发；"
                                 f"最近错误 {runner.stop_detail or '见请求账本'}。有效结果与断点已保存；"
                                 "未进入下一阶段，排查后重跑同命令续传。") from stopped
        if isinstance(stopped, StopCollection) and runner.stop_reason == "request_budget_exhausted":
            raise StopCollection("本次 AI 总请求预算已到，已保存结果；重跑同一命令续传") from stopped
        raise stopped
    try:
        return records(runner.root, runner.version, tasks)
    except StopCollection:
        runner.stop_reason = "unfinished_annotation_tasks"
        raise


def predictions(root, protocol, tasks, first, second, adjudications, decisions):
    output, consensus = {"videos": [], "comments": []}, {"videos": [], "comments": []}
    resolved = {(j["kind"], item["id"]): item for j in adjudications for item in decisions[j["job_id"]]}
    for job in tasks:
        aa, bb = first[job["job_id"]], {r["id"]: r for r in second[job["job_id"]]}
        final = []
        for a in aa:
            b = bb[a["id"]]
            unit_key = (job["kind"], a["id"])
            item = copy.deepcopy(resolved.get(unit_key, a))
            missing = {role: outcomes.unavailable(pass_)[unit_key] for role, pass_ in
                       [("coder", first), ("reviewer", second), ("adjudicator", decisions)]
                       if unit_key in outcomes.unavailable(pass_)}
            if missing:
                item = outcomes.placeholder(job["kind"], a["id"], "+".join(sorted(set(missing.values()))))
            for label, value in item["labels"].items():
                if not known(value):
                    item["labels"][label], item["evidence"][label] = None, ""
            final.append(item)
            label_sources = {label: ("unresolved" if value is None else
                                    "ai_agreement" if a["labels"][label] == b["labels"][label] == value else
                                    "ai_adjudication") for label, value in item["labels"].items()}
            row = {"id": item["id"], **item["labels"], "evidence": json.dumps(item["evidence"], ensure_ascii=False),
                   "note": item["note"], "coding_source": "ai_agents_unvalidated",
                   "model_version_hash": protocol["protocol_hash"], "split": job["split"],
                   "coding_status": "unavailable" if missing else "validated",
                   "unavailable_reason": json.dumps(missing, sort_keys=True) if missing else "",
                   "decision_agent": "unavailable" if missing else "adjudicator" if unit_key in resolved else "blind_pair_agreement",
                   "label_sources": json.dumps(label_sources, sort_keys=True)}
            output[job["kind"]].append(row)
            strict = {**row, "coding_source": "ai_blind_agreement_only"}
            for label in item["labels"]:
                strict[label] = a["labels"][label] if not missing and known(a["labels"][label]) and a["labels"][label] == b["labels"][label] else None
            # These are sensitivity labels only, never present the adjudicator's evidence as their evidence.
            strict["evidence"], strict["note"] = json.dumps(a["evidence"], ensure_ascii=False), "Blind agreement only; disagreements are missing."
            consensus[job["kind"]].append(strict)
        client.validate_response(json.loads(job["payload"]), {"items": final})
    return ({k: pd.DataFrame(v) for k, v in output.items()}, {k: pd.DataFrame(v) for k, v in consensus.items()})


def agreement(p, root, tasks, first, second, final, dest):
    """Pre-adjudication AI consistency: neither pass is treated as ground truth."""
    rows, disagreements, confusion = [], [], []
    for kind, labels in [("videos", data.VIDEO_FIELDS), ("comments", COMMENT_LABELS)]:
        aa = {r["id"]: r for j in tasks if j["kind"] == kind for r in first[j["job_id"]]}
        bb = {r["id"]: r for j in tasks if j["kind"] == kind for r in second[j["job_id"]]}
        for label in labels:
            a = pd.Series({k: r["labels"][label] for k, r in aa.items()}, dtype=object)
            b = pd.Series({k: bb[k]["labels"][label] for k in aa}, dtype=object)
            a_known, b_known = a.map(known), b.map(known)
            both = a_known & b_known
            metric = agreement_metrics(a[both].astype(str).to_numpy(), b[both].astype(str).to_numpy())
            rows.append({"kind": kind, "label": label, "n_units": len(a), "n_both_resolved": int(both.sum()),
                         "coder_unavailable": sum((kind, key) in outcomes.unavailable(first) for key in a.index),
                         "reviewer_unavailable": sum((kind, key) in outcomes.unavailable(second) for key in a.index),
                         "coder_coverage": a_known.mean(), "reviewer_coverage": b_known.mean(),
                         "both_coverage": both.mean(), "agreement_on_resolved": metric["agreement"],
                         "kappa_ai_vs_ai": metric["kappa"], "alpha_nominal_ai_vs_ai": metric["alpha_nominal"],
                         "final_coverage": final[kind][label].notna().mean(),
                         "status": "no_paired_labels" if not both.any() else
                                   "constant_or_insufficient_pairs" if metric["kappa"] is None else "ai_consistency_only"})
            for key in a.index:
                if not both[key] or a[key] != b[key]:
                    disagreements.append({"kind": kind, "id": key, "label": label,
                                          "coder": a[key], "reviewer": b[key],
                                          "type": "unavailable" if any((kind, key) in outcomes.unavailable(pass_) for pass_ in [first, second])
                                                  else "abstention" if not both[key] else "disagreement"})
            table = pd.DataFrame({"coder": a.where(a_known, "__abstain__").astype(str),
                                  "reviewer": b.where(b_known, "__abstain__").astype(str)})
            for role, pass_ in [("coder", first), ("reviewer", second)]:
                for key in a.index:
                    if (kind, key) in outcomes.unavailable(pass_):
                        table.loc[key, role] = "__unavailable__"
            for (x, y), n in table.groupby(["coder", "reviewer"]).size().items():
                confusion.append({"kind": kind, "label": label, "coder": x, "reviewer": y, "n": int(n)})
    write_csv(pd.DataFrame(rows), dest / "ai_agreement.csv")
    write_csv(pd.DataFrame(disagreements, columns=["kind", "id", "label", "coder", "reviewer", "type"]), dest / "disagreements.csv")
    write_csv(pd.DataFrame(confusion), dest / "ai_confusion.csv")


def plan(p, experiment="agents_v1"):
    root = data.folder(p, experiment)
    if uses_screening(p, root):
        from . import ai_screening
        return ai_screening.plan_summary(p, experiment)
    m, tasks = check_mode(root), data.jobs(root)
    n = len(tasks)
    report = {"videos": m["n_videos"], "comments": m["n_comments"], "threads": m["n_threads"],
              "jobs_per_blind_pass": n, "requests_min_no_retries": 2 * n, "requests_max_no_retries": 3 * n,
              "human_coding_rows": 0, "youtube_api_units": 0, "request_range_excludes_adaptive_splits": True,
              "notes": "两轮全量编码；第三轮仅处理分歧/弃权，未提供字幕的形式缺失不请求裁决。"
                       "小批预检复用正式任务；上述区间不含重试及自动拆批的额外请求，并非费用上限。"}
    write_json(p.reports / "ai" / experiment / "plan.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def status(p, experiment="agents_v1"):
    root = data.folder(p, experiment)
    if uses_screening(p, root):
        from .ai_screening_report import status
        return status(p, experiment)
    check_mode(root)
    tasks = data.jobs(root)
    if not (root / "agent_protocol.json").exists():
        print(f"尚未调用 Agent；每轮 {len(tasks)} 个任务。")
        return
    spec = load_protocol(root)
    adj_path = root / "adjudication_jobs.json"
    adj = json.loads(adj_path.read_text())["jobs"] if adj_path.exists() else None
    db = data.connect(root)
    rows = []
    try:
        for role, config in spec["protocol"]["configurations"].items():
            ver = client.version(config)
            jj = tasks if role != "adjudicator" else adj
            results = {r["job_id"]: ("recovery_pending" if r["error"].startswith("split_required:") else r["status"]) for r in db.execute(
                "SELECT job_id,status,error FROM results WHERE version=?", (ver,))}
            attempts = list(db.execute("SELECT usage FROM attempts WHERE version=?", (ver,)))
            usage = [json.loads(r["usage"]) for r in attempts]
            rows.append({"agent": role, "model": config["model"], "jobs": len(jj) if jj is not None else None,
                         "complete": sum(results.get(j["job_id"]) == "complete" for j in jj or []),
                         "error": sum(results.get(j["job_id"]) == "error" for j in jj or []),
                         "recovery_pending": sum(results.get(j["job_id"]) == "recovery_pending" for j in jj or []),
                         "unavailable": sum(results.get(j["job_id"]) == "unavailable" for j in jj or []),
                         "cached_split_batches": sum(key.startswith("split_") and state == "complete"
                                                      for key, state in results.items()),
                         "cached_pilot_decisions": (sum(not key.startswith("split_") and state == "complete"
                                                       for key, state in results.items())
                                                    if role == "adjudicator" and adj is None else 0),
                         **client.usage_summary(usage)})
    finally:
        db.close()
    frame = pd.DataFrame(rows)
    write_csv(frame, p.reports / "ai" / experiment / "agent_status.csv")
    print(frame.to_string(index=False))
    db = data.connect(root)
    try:
        block = latest_provider_block([dict(r) for r in db.execute(
            "SELECT * FROM executions ORDER BY started_at DESC LIMIT 1")])
    finally:
        db.close()
    if block:
        event("最近账户暂停", f"{block['reason']}。{block['action']}。这是上次运行记录，不是实时余额查询。")


def analyze(p, experiment="agents_v1", _prepared=None):
    """Rebuild local reports with a bounded retry; never enter the model-calling path."""
    if uses_screening(p, data.folder(p, experiment)):
        from . import ai_screening
        return ai_screening.analyze(p, experiment)
    retries = p.cfg.get("ai", {}).get("report_retry_rounds", 1)
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError("ai.report_retry_rounds 须为 0..2 的整数")
    dest = p.reports / "ai" / experiment
    run_id = uuid.uuid4().hex
    state = {"report_run_id": run_id, "started_at": now(), "status": "running", "attempt": 0,
             "model_requests": 0, "errors": [], "note": "仅本机统计/制图；以 complete 状态确认本轮完整报告，旧文件可能仍在。"}

    def checkpoint(status):
        state.update(status=status, updated_at=now())
        write_json(dest / "report_runs" / f"{run_id}.json", state)
        write_json(dest / "report_state.json", state)

    for attempt in range(retries + 1):
        state["attempt"] = attempt + 1
        try:
            checkpoint("running")
            result = analyze_once(p, experiment, _prepared)
            checkpoint("complete")
            event("完成", f"全 AI 编码、复核、裁决、统计及图表报告已保存：{dest / '实验结果.md'}")
            return result
        except BaseException as exc:
            state["errors"].append({"attempt": attempt + 1, "type": type(exc).__name__,
                                    "errno": exc.errno if isinstance(exc, OSError) else None})
            retry = isinstance(exc, OSError) and exc.errno in {errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT} and attempt < retries
            try:
                checkpoint("retrying" if retry else "interrupted" if isinstance(exc, KeyboardInterrupt) else "incomplete")
            except OSError:
                event("报告", "报告状态文件未能写入；编码数据库保留，当前报告应视为未完成。")
            if retry:
                event("报告恢复", f"本机输出临时失败（{type(exc).__name__}）；重试 {attempt + 1}/{retries}，不请求模型。")
                continue
            if isinstance(exc, OSError):
                raise StopCollection(f"本机报告未完成（{type(exc).__name__}）；编码进度已保留，修复存储问题后运行 agent-analyze --experiment {experiment}") from None
            raise


def analyze_once(p, experiment="agents_v1", _prepared=None):
    from . import ai_analysis, ai_audit, ai_figures, ai_recovery
    from .analysis import boot_correlation, correct_p
    root = data.folder(p, experiment)
    event("分析", "开始读取已完成的编码结果并核对目标 ID、证据与协议；不调用模型。")
    spec, tasks = (load_protocol(root), data.jobs(root)) if _prepared is None else _prepared[:2]
    configs = spec["protocol"]["configurations"]
    first, second = ([records(root, client.version(configs[role]), tasks) for role in ["coder", "reviewer"]]
                     if _prepared is None else [_prepared[2][role] for role in ["coder", "reviewer"]])
    adjudications = make_adjudication(root, tasks, first, second, spec)
    decisions = (records(root, client.version(configs["adjudicator"]), adjudications) if _prepared is None else
                 _prepared[2]["adjudicator"])
    final, consensus = predictions(root, spec, tasks, first, second, adjudications, decisions)
    dest = p.reports / "ai" / experiment
    dest.mkdir(parents=True, exist_ok=True)
    event("分析", "编码结果校验完成；计算裁决前 AI 一致性与最终标签覆盖率。")
    agreement(p, root, tasks, first, second, final, dest)
    report = {"version": spec["protocol_hash"], "reference_type": "none_ai_only",
              "gate": {**{"videos." + c: True for c in data.VIDEO_FIELDS},
                       **{"comments." + c: True for c in COMMENT_LABELS}}}
    # Here gate means eligible for exploratory calculation, NOT a passed accuracy/validation test.
    ai_analysis.analyze_predictions(p, root, final, report, dest / "analysis")
    event("敏感性", "开始仅保留两轮一致标签的分析；主分析相关结果直接复用。")
    strict, requirements = ai_analysis.analysis_frame(root, consensus, report)
    strict["coding_source"] = "ai_blind_agreement_only"
    write_csv(strict, dest / "analysis" / "sensitivity_agreement_only_videos.csv")
    final_frame = pd.read_csv(dest / "analysis" / "analysis_videos.csv")
    correlations = pd.read_csv(dest / "analysis" / "correlations.csv")
    already_computed = correlations.set_index("y")
    sensitivity = []
    for mode, frame in [("adjudicated", final_frame), ("blind_agreement_only", strict)]:
        for metric in requirements:
            stats = (already_computed.loc[metric, ["n", "rho", "p", "ci_low", "ci_high", "bootstrap_valid"]].to_dict()
                     if mode == "adjudicated" else boot_correlation(frame, "view_count", metric,
                         p.cfg["analysis"]["bootstrap_reps"], p.cfg["study"]["seed"]))
            sensitivity.append({"variant": mode, "metric": metric, "n_observed": int(frame[metric].notna().sum()),
                "mean": frame[metric].mean(), **stats})
            if mode == "blind_agreement_only":
                event("敏感性", f"已完成 {metric}；有效视频 {int(stats['n']):,}。")
    write_csv(correct_p(pd.DataFrame(sensitivity), "fdr_bh"), dest / "analysis" / "sensitivity_correlations.csv")
    event("制图", "统计表已保存；开始生成 7 张报告图，每张提供 PNG 和 PDF。")
    figure_manifest = ai_figures.generate(dest, min_group_n=p.cfg["analysis"]["min_group_n"])
    audit_summary, audit_dest, audit_figures = ai_audit.generate(p, experiment, final=final)
    audit_relative = str(audit_dest.relative_to(dest))
    summary = {"created_at": now(), "protocol_hash": spec["protocol_hash"], "human_reference": False,
               "models": {k: v["model"] for k, v in configs.items()},
               "same_model_for_blind_passes": configs["coder"]["model"] == configs["reviewer"]["model"],
               "n_blind_jobs_each": len(tasks), "n_adjudication_jobs": len(adjudications),
               "n_adjudicated_units": sum(len(json.loads(j["targets"])) for j in adjudications),
               "unavailable_units": {kind: int(frame.coding_status.eq("unavailable").sum()) for kind, frame in final.items()},
               "prediction_hash": client.version({k: v.to_json(orient="records", force_ascii=False) for k, v in final.items()}),
               "figure_manifest": "analysis/figures/manifest.json", "figure_count": len(figure_manifest["figures"]),
               "figure_version": figure_manifest["figure_version"],
               "comment_recovery": ai_recovery.summary(root, data.connect),
               "execution_policies": outcomes.policy_records(root),
               "audit_report": audit_relative + "/审计报告.md", "audit_summary": audit_relative + "/summary.json",
               "audit_figure_count": audit_summary["figure_count"],
               "total_figure_count": len(figure_manifest["figures"]) + audit_summary["figure_count"],
               "scope": "exploratory_model_only; AI agreement is not human reliability or accuracy",
               "sensitivity_note": "仅保留两轮一致标签的指标使用完整视频分母；缺失可能导致选择性缩小，差异不只是标签变化。"}
    write_json(dest / "agent_report.json", summary)
    lines = ["# 全 AI 实验结果（探索性）", "", "编码、盲复核和裁决全部由模型完成，无人工标签或人工信度检验。",
             f"两轮各 {len(tasks)} 个任务；裁决 {summary['n_adjudicated_units']} 个分析单元。",
             f"不可编码单元：{summary['unavailable_units']['videos']} 个视频、{summary['unavailable_units']['comments']} 条评论；全部保留缺失，不计校验成功。",
             "同一模型使用不同提示词不代表误差独立；高一致性不能证明标签准确。", "",
             ("评论格式恢复、部分结果缓存和拆批的运行策略按请求登记，见 agent_report.json 的 comment_recovery。"
              "恢复策略与冻结的模型／基础提示词配置分别记录；未通过语义标签与正标签原文证据校验的内容不会作为成功结果。"), "",
             "## 传播规模与参与指标", ""]
    for row in correlations.itertuples():
        lines.append(f"- {row.y}：有效视频 n={int(row.n)}，Spearman ρ={row.rho:.4g}，BH 校正 p={row.p_adjusted:.4g}。")
    lines += ["", "## 阅读顺序", "", "1. ai_agreement.csv：裁决前的 AI 一致性与覆盖率。",
              "2. analysis/report.json：口径、缺失和适用范围。",
              "3. analysis/sensitivity_correlations.csv：裁决结果与仅保留一致标签的敏感性分析。",
              "4. analysis/groups_kruskal.csv、models_status.csv、models_coefficients.csv：组间与回归结果。",
              "", "主题来自标题/说明；没有字幕的形式保留缺失。学习表达和分享意向不等于实际认知改变与传播扩散。",
              "结果只描述可见数据中的关联，不能据此确认因果机制或预先给定的摘要结论。",
              "常量/样本不足模型会自动跳过并给出原因，无须补填人工编码。"]
    lines += ai_figures.markdown(figure_manifest)
    lines += ["", "## 执行质量与评论标签分布", "",
              (f"[完整审计、逐条标签与分母、请求和错误记录]({audit_relative}/审计报告.md)。"
              "请求数、原始批次数与去重评论数分别统计；成功通过校验不代表语义准确。")]
    for figure in audit_figures["figures"]:
        lines += ["", f"### {figure['title']}", "",
                  f"![{figure['title']}]({audit_relative}/figures/{figure['png']})", "",
                  figure["caption"] + f" [PDF]({audit_relative}/figures/{figure['pdf']})"]
    (dest / "实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(p, experiment="agents_v1", max_requests=None, workers=None):
    if uses_screening(p, data.folder(p, experiment)):
        from . import ai_screening
        return ai_screening.run(p, experiment, max_requests, workers)
    workers = client.worker_count(p, workers)
    max_requests = p.cfg.get("ai", {}).get("default_max_requests", 0) if max_requests is None else max_requests
    if type(max_requests) is not int or max_requests < 0:
        raise ValueError("max-requests 须为非负数")
    report_retries = p.cfg.get("ai", {}).get("report_retry_rounds", 1)
    if type(report_retries) is not int or not 0 <= report_retries <= 2:
        raise ValueError("ai.report_retry_rounds 须为 0..2 的整数")
    root = data.folder(p, experiment)
    event("准备", f"实验 {experiment}；本次请求上限 {max_requests or '不限'}，包含重试与拆批。")
    if not root.exists():
        prepare(p, experiment)
    pilot_n = p.cfg.get("ai", {}).get("pilot_jobs_per_kind", 1)
    if type(pilot_n) is not int or not 0 <= pilot_n <= 10:
        raise ValueError("ai.pilot_jobs_per_kind 须为 0..10 的整数")
    runner = None
    passes = {}
    try:
        if not (root / "agent_protocol.json").exists():
            # Validate credentials/runtime limits before freezing a new experiment.
            runner = client.Runner(p, root, configurations(p, root)["coder"], max_requests)
        spec, tasks = freeze(p, root), data.jobs(root)
        configs = spec["protocol"]["configurations"]
        event("计划", f"每轮 {len(tasks):,} 个基础批次；顺序执行初编、盲复核、裁决、统计和制图。"
              "裁决数量在两轮完成后确定；进度百分比仅对应当前阶段。")
        def execute(role, jobs, prefix=""):
            nonlocal runner
            config = configs[role]
            done = settled(root, client.version(config))
            if any(j["job_id"] not in done for j in jobs):
                if runner is None:
                    runner = client.Runner(p, root, config, max_requests)
                runner.config, runner.version = config, client.version(config)
                return run_stage(runner, jobs, workers, prefix + role)
            result = records(root, client.version(config), jobs)
            event(role_label(prefix + role), f"复用已处理 {len(jobs):,}/{len(jobs):,} 批（含不可编码记录），不新增请求。" if jobs else "无待处理任务，进入下一阶段。")
            return result
        # Both models process actual large batches early; cached results are reused in the full passes.
        pilot = [j for kind in ["videos", "comments"]
                 for j in sorted((t for t in tasks if t["kind"] == kind),
                                 key=lambda t: (-len(t["payload"]), t["job_id"]))[:pilot_n]]
        if pilot:
            event("预检", f"先处理 {len(pilot)} 个正式批次，两轮结果会复用到后续全量阶段。")
            first = execute("coder", pilot, "预检/")
            second = execute("reviewer", pilot, "预检/")
            adjudications = make_adjudication(root, pilot, first, second, spec, persist=False)
            pilot_decisions = execute("adjudicator", adjudications, "预检/")
            pilot_unavailable = {role: len(outcomes.unavailable(value)) for role, value in
                                 [("coder", first), ("reviewer", second), ("adjudicator", pilot_decisions)]}
            write_json(p.reports / "ai" / experiment / "pilot.json", {
                "job_ids": [j["job_id"] for j in pilot], "adjudication_jobs": len(adjudications),
                "status": "completed_with_unavailable_inputs" if any(pilot_unavailable.values()) else "outputs_parse_and_match_schema",
                "unavailable_units": pilot_unavailable, "semantic_accuracy_validated": False,
                "note": "真实任务小批预检，结果复用；无分歧时不请求裁决，预检不证明真实语义准确率。"})
        for role in ["coder", "reviewer", "adjudicator"]:
            jobs = tasks if role != "adjudicator" else make_adjudication(root, tasks, passes["coder"], passes["reviewer"], spec)
            if role == "adjudicator":
                event("裁决计划", f"需要裁决 {len(jobs):,} 批、{sum(len(json.loads(j['targets'])) for j in jobs):,} 个单元。")
            # Stage barriers ensure no in-flight request sees another role's configuration.
            passes[role] = execute(role, jobs)
    finally:
        original_error = sys.exc_info()[1]
        if runner is not None:
            runner.stop.set()
            try:
                runner.client.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must not overwrite the actual stop reason
                event("连接清理", f"清理未完成（{type(exc).__name__}）；已落盘编码保留。")
            try:
                runner.record_end(original_error)
            except Exception as exc:  # Retain the primary error; fail visibly if journaling alone failed.
                event("运行记录", f"结束状态未能保存（{type(exc).__name__}）；已落盘编码保留。")
                if original_error is None:
                    raise
        if (root / "agent_protocol.json").exists():
            try:
                status(p, experiment)
            except Exception as exc:  # noqa: BLE001 - a derived status CSV is not a reason to discard valid coding
                event("状态汇总", f"输出未完成（{type(exc).__name__}）；编码记录已保留，可单独运行 agent-status。")
            if len(passes) < 3 and p.cfg.get("ai", {}).get("audit_on_pause", True):
                from . import ai_audit
                try:
                    ai_audit.generate(p, experiment)
                except Exception as exc:  # noqa: BLE001 - preserve the original exception after saving coding results
                    # Audit/rendering failure must not mask the original pause or discard saved labels.
                    event("审计", f"暂停后的报告生成未完成（{type(exc).__name__}）；编码记录已保留，可单独运行 agent-audit。")
    analyze(p, experiment, _prepared=(spec, tasks, passes))
