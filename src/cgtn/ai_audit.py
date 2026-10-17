"""Offline, point-in-time operational and coding audits; no provider or research DB writes."""
from __future__ import annotations

import gzip
import json
import sqlite3
import tomllib
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import ai_agents as agents
from . import ai_client as client
from . import ai_data as data
from . import ai_outcomes as outcomes
from .ai_stops import latest_provider_block
from .coding import COMMENT_LABELS
from .core import now, read_csv, stable_hash, write_csv, write_json, write_text
from .progress import event

ROLES = ["coder", "reviewer", "adjudicator", "final"]
STATES = ["complete", "unavailable", "failed", "recovery_pending", "started_unconfirmed", "not_started", "not_ready", "conflicting_results"]
TABLES = ["jobs", "attempts", "results", "job_registry", "attempt_outputs", "result_history", "executions", "execution_events", "evidence_quarantine"]


def readonly_project(config):
    path = Path(config).resolve()
    cfg = tomllib.loads(path.read_text())
    study = cfg["study"]
    return SimpleNamespace(base=path.parent, cfg=cfg,
        root=path.parent / study["data_dir"] / study["run_id"],
        reports=path.parent / study["reports_dir"] / study["run_id"])


def capture(root):
    """SQLite backup supplies one consistent snapshot even while a collector writes WAL."""
    with closing(sqlite3.connect(f"{(root / 'ai.sqlite').resolve().as_uri()}?mode=ro", uri=True)) as source, closing(sqlite3.connect(":memory:")) as snapshot:
        source.backup(snapshot)
        snapshot.row_factory = sqlite3.Row
        present = {r[0] for r in snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {name: [dict(r) for r in snapshot.execute(f"SELECT * FROM {name} ORDER BY 1,2")]
                if name in present else [] for name in TABLES}


def subset_job(parent, targets):
    part = data.target_subset(json.loads(parent["payload"]), targets)
    content = json.dumps(part, ensure_ascii=False, sort_keys=True)
    return {**parent, "job_id": "split_" + stable_hash(content), "payload": content,
            "targets": json.dumps(targets), "parent_job_id": parent["job_id"]}


def inventory(root, spec, snapshot, plans=None):
    """Reconstruct legacy child inputs, validate cached output, and never double-count caches."""
    configs = spec["protocol"]["configurations"]
    versions = {role: client.version(cfg) for role, cfg in configs.items()}
    results = {(r["version"], r["job_id"]): r for r in snapshot["results"]}
    base = snapshot["jobs"]
    data.validate_jobs(data.manifest(root), base)
    custom = plans is not None
    complete_parents, invalid = {}, []
    for role in plans if custom else ["coder", "reviewer"]:
        complete_parents[role] = outcomes.Records()
        for job in plans[role] if custom else base:
            saved = results.get((versions[role], job["job_id"]))
            if saved and saved["status"] in outcomes.SETTLED:
                try:
                    outcomes.add_record(complete_parents[role], job, saved, client.validate_response)
                except (ValueError, TypeError, KeyError):
                    invalid.append({"role": role, "job_id": job["job_id"], "issue": "invalid_cached_parent"})
    paired = [] if custom else [j for j in base if all(j["job_id"] in complete_parents[r] for r in ["coder", "reviewer"])]
    adj = [] if custom else agents.make_adjudication(root, paired, complete_parents["coder"], complete_parents["reviewer"], spec, persist=False)
    plans = plans if custom else {"coder": base, "reviewer": base, "adjudicator": adj}
    registry = {(versions[role], j["job_id"]): j for role, jobs in plans.items() for j in jobs}
    for registered in snapshot["job_registry"]:
        key = (registered["version"], registered["job_id"])
        payload = json.loads(registered["payload"])
        if payload["kind"] != registered["kind"] or payload["target_ids"] != json.loads(registered["targets"]):
            raise ValueError("请求登记的类型/目标与实际输入不一致")
        prior = registry.get(key)
        if prior and json.loads(prior["payload"]) != json.loads(registered["payload"]):
            raise ValueError("请求登记输入与冻结任务不一致，拒绝输出可能错误的统计")
        if registered["job_id"].startswith("split_"):
            canonical = json.dumps(json.loads(registered["payload"]), sort_keys=True, ensure_ascii=False)
            if registered["job_id"] != "split_" + stable_hash(canonical):
                raise ValueError("拆批输入哈希不一致，拒绝输出可能错误的统计")
        registry[key] = registered
    # Pre-upgrade completed child caches retain target IDs even though inputs were not registered.
    owners = {role: {key: j for j in jobs for key in json.loads(j["targets"])} for role, jobs in plans.items()}
    for role, ver in versions.items():
        for saved in snapshot["results"]:
            key = (ver, saved["job_id"])
            if saved["version"] != ver or key in registry or saved["status"] != "complete":
                continue
            labels = json.loads(saved["labels"])
            ids = [r["id"] for r in labels]
            candidates = {owners[role][x]["job_id"] for x in ids if x in owners[role]}
            if ids and len(candidates) == 1 and all(x in owners[role] for x in ids):
                reconstructed = subset_job(owners[role][ids[0]], ids)
                if reconstructed["job_id"] == saved["job_id"]:
                    registry[key] = reconstructed
    # Recovery plans identify children that have no output yet, recursively.
    changed = True
    while changed:
        changed = False
        for key, job in list(registry.items()):
            saved = results.get(key)
            plan = json.loads(saved["response"]) if saved and saved.get("response") else None
            if not isinstance(plan, dict) or plan.get("source") != "pending_recovery":
                continue
            for group in plan["target_groups"]:
                child = subset_job(job, group)
                child_key = (key[0], child["job_id"])
                if child_key not in registry:
                    registry[child_key] = child
                    changed = True
    values, flags, provenance, copies, unavailable_reasons = {}, defaultdict(set), {}, Counter(), {}
    for key, job in registry.items():
        role = next((r for r, ver in versions.items() if ver == key[0]), None)
        if role is None:
            continue
        saved = results.get(key)
        if not saved:
            continue
        unit_keys = [(role, job["kind"], x) for x in json.loads(job["targets"])]
        if saved["status"] in outcomes.SETTLED:
            try:
                labels, missing = outcomes.unpack(json.loads(job["payload"]), saved, client.validate_response)
            except (ValueError, TypeError, KeyError):
                for unit in unit_keys:
                    flags[unit].add("conflicting_results")
                invalid.append({"role": role, "job_id": key[1], "issue": "invalid_cached_output"})
                continue
            for target, reason in missing.items():
                unit = (role, job["kind"], target)
                flags[unit].add("unavailable")
                unavailable_reasons[unit] = reason
                provenance[unit] = key[1]
            for item in labels:
                unit = (role, job["kind"], item["id"])
                copies[unit] += 1
                if unit in values and values[unit] != item:
                    flags[unit].add("conflicting_results")
                    invalid.append({"role": role, "job_id": key[1], "issue": "conflicting_duplicate_output"})
                else:
                    values[unit] = item
                    if unit not in provenance or key[1].startswith("split_"):
                        provenance[unit] = key[1]
        else:
            state = "recovery_pending" if saved["error"].startswith("split_required:") else "failed"
            for unit in unit_keys:
                flags[unit].add(state)
    for attempt in snapshot["attempts"]:
        key = (attempt["version"], attempt["job_id"])
        job = registry.get(key)
        role = next((r for r, ver in versions.items() if ver == key[0]), None)
        if job and role and key not in results:
            for unit in json.loads(job["targets"]):
                flags[(role, job["kind"], unit)].add("started_unconfirmed" if attempt["status"] == "started" else "failed")
    for unit in set(values) & set(unavailable_reasons):
        flags[unit].add("conflicting_results")
        invalid.append({"role": unit[0], "job_id": provenance[unit], "issue": "validated_and_unavailable_conflict"})
    decisions = outcomes.Records()
    for job in adj:
        row = results.get((versions["adjudicator"], job["job_id"]))
        if row and row["status"] in outcomes.SETTLED:
            try:
                outcomes.add_record(decisions, job, row, client.validate_response)
            except (ValueError, TypeError, KeyError):
                pass  # Already listed as an invalid cache; no final results will be produced.
    ready = ((all(len(complete_parents[r]) == len(jobs) for r, jobs in plans.items()) if custom else
              len(paired) == len(base) and len(decisions) == len(adj)) and not invalid)
    final = agents.predictions(root, spec, base, complete_parents["coder"], complete_parents["reviewer"], adj, decisions)[0] if ready and not custom else None
    return {"versions": versions, "plans": plans, "registry": registry, "results": results, "values": values,
            "flags": flags, "provenance": provenance, "copies": copies, "invalid": invalid, "final": final,
            "unavailable_reasons": unavailable_reasons,
            "final_ready": ready, "paired_parent_jobs": len(paired), "complete_parents": complete_parents,
            "roles": list(plans) + ["final"]}


def request_table(snapshot, inventory):
    version_roles = {v: k for k, v in inventory["versions"].items()}
    version_roles.update(inventory.get("historical_versions", {}))
    archived_outputs = {r["attempt_id"]: r for r in snapshot["attempt_outputs"]}
    rows = []
    for attempt in snapshot["attempts"]:
        usage = json.loads(attempt["usage"])
        job = inventory["registry"].get((attempt["version"], attempt["job_id"]))
        n = len(json.loads(job["targets"])) if job else usage.get("_client_target_count")
        accepted = n if attempt["status"] == "complete" else usage.get("_client_valid_items")
        outcome = ("fully_valid" if attempt["status"] == "complete" else "started_unconfirmed"
                   if attempt["status"] == "started" else "partially_valid" if accepted else "invalid")
        tokens = client.usage_summary([usage])
        output = archived_outputs.get(attempt["id"], {})
        rows.append({"attempt_id": attempt["id"], "role": version_roles.get(attempt["version"], "unknown_version"),
            "version": attempt["version"], "job_id": attempt["job_id"],
            "kind": job["kind"] if job else usage.get("_client_kind", "unknown"),
            "parent_job_id": job.get("parent_job_id") if job else None,
            "execution_id": usage.get("_client_execution_id", "legacy_run_unknown"),
            "automatic_recovery_round": usage.get("_client_recovery_round"),
            "phase": usage.get("_client_phase"), "retry_index": usage.get("_client_retry_index"),
            "runtime_settings": json.dumps(usage.get("_client_runtime"), sort_keys=True),
            "json_error": json.dumps(usage.get("_client_json_error"), sort_keys=True),
            "network_error": json.dumps(usage.get("_client_network_error"), sort_keys=True),
            "output_budget": json.dumps(usage.get("_client_output_budget"), sort_keys=True),
            "assistant_content_archived": bool(output), "assistant_content_redacted": output.get("redacted"),
            "assistant_content_sha256": output.get("content_sha256"),
            "started_at": usage.get("_client_started_at"), "updated_at": attempt["updated_at"],
            "request_sequence": usage.get("_client_request_sequence"),
            "dispatch_wait_seconds": usage.get("_client_dispatch_wait_seconds"),
            "http_started_at": usage.get("_client_http_started_at"),
            "http_finished_at": usage.get("_client_http_finished_at"),
            "http_inflight_at_start": usage.get("_client_http_inflight_at_start"),
            "peak_http_inflight": usage.get("_client_peak_http_inflight"),
            "http_status": usage.get("_client_http_status"), "finish_reason": usage.get("_client_finish_reason"),
            "provider_refusal_reason": usage.get("_client_provider_refusal_reason"),
            "status": attempt["status"], "outcome": outcome, "target_count": n, "valid_items_in_response": accepted,
            "reported_tokens": tokens["reported_tokens"] if not tokens["unknown_usage_attempts"] else None,
            "elapsed_seconds": usage.get("_client_elapsed_seconds"), "format_repairs": len(usage.get("_client_repairs", [])),
            "policy_hash": usage.get("_client_comment_recovery", {}).get("policy_hash", "legacy_or_noncomment"),
            "request_sha256": usage.get("_client_request_sha256"), "input_reconstructed": job is not None})
    return pd.DataFrame(rows, columns=["attempt_id", "role", "version", "job_id", "kind", "parent_job_id", "execution_id", "automatic_recovery_round",
        "phase", "retry_index", "runtime_settings", "json_error", "network_error", "output_budget", "provider_refusal_reason", "assistant_content_archived", "assistant_content_redacted",
        "assistant_content_sha256", "started_at", "updated_at", "http_status", "finish_reason", "status", "outcome", "target_count", "valid_items_in_response", "reported_tokens",
        "elapsed_seconds", "format_repairs", "policy_hash", "request_sha256", "input_reconstructed",
        "request_sequence", "dispatch_wait_seconds", "http_started_at", "http_finished_at",
        "http_inflight_at_start", "peak_http_inflight"])


def unit_table(inventory):
    rows = []
    for role in inventory.get("roles", ROLES):
        jobs = inventory["plans"]["coder" if role == "final" else role]
        final_rows = ({kind: df.set_index("id").to_dict("index") for kind, df in inventory["final"].items()}
                      if role == "final" and inventory["final"] is not None else {})
        for job in jobs:
            for key in json.loads(job["targets"]):
                unit = (role, job["kind"], key)
                flag = inventory["flags"].get(unit, set())
                item = inventory["values"].get(unit)
                if role == "final":
                    item = final_rows.get(job["kind"], {}).get(key)
                    state = ("unavailable" if item is not None and item.get("coding_status") == "unavailable" else
                             "complete" if item is not None else "not_ready")
                else:
                    state = ("conflicting_results" if "conflicting_results" in flag else "unavailable" if "unavailable" in flag else "complete" if item else
                             next((s for s in ["failed", "recovery_pending", "started_unconfirmed"] if s in flag), "not_started"))
                columns = COMMENT_LABELS if job["kind"] == "comments" else data.VIDEO_FIELDS
                labels = item if role == "final" else item["labels"] if item else {}
                rows.append({"role": role, "kind": job["kind"], "id": key, "base_job_id": job["job_id"], "status": state,
                    "source_job_id": inventory["provenance"].get(unit), "cache_copies": inventory["copies"].get(unit, 0),
                    "unavailable_reason": (item.get("unavailable_reason") if role == "final" and item else
                                           inventory["unavailable_reasons"].get(unit)),
                    **{c: labels.get(c) if state == "complete" else None for c in columns},
                    "resolved_labels": sum(agents.known(labels.get(c)) for c in columns) if state == "complete" else None,
                    "label_count": len(columns),
                    "evidence": (item.get("evidence") if role == "final" else json.dumps(item["evidence"], ensure_ascii=False)) if item else None,
                    "note": item.get("note") if item else None})
    frame = pd.DataFrame(rows)
    for name in COMMENT_LABELS + data.VIDEO_FIELDS:
        if name not in frame:
            frame[name] = None
    return frame


def state_summary(frame, groups):
    rows = []
    for key, part in frame.groupby(groups, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        counts = part.status.value_counts()
        rows.append({**dict(zip(groups, key, strict=True)), "total": len(part),
                     **{state: int(counts.get(state, 0)) for state in STATES},
                     "completion_rate": part.status.eq("complete").mean()})
    return pd.DataFrame(rows)


def unit_summary(units):
    summary = state_summary(units, ["role", "kind"])
    complete = units.status.eq("complete")
    resolved = pd.to_numeric(units.resolved_labels, errors="coerce")
    semantic = units.assign(
        complete_all_labels_resolved=complete & resolved.eq(units.label_count),
        complete_with_abstentions=complete & resolved.lt(units.label_count),
        complete_all_labels_abstained=complete & resolved.eq(0)).groupby(["role", "kind"])[
            ["complete_all_labels_resolved", "complete_with_abstentions", "complete_all_labels_abstained"]].sum().reset_index()
    return summary.merge(semantic, on=["role", "kind"], validate="one_to_one")


def unavailable_summary(units):
    missing = units.loc[units.status.eq("unavailable")]
    summary = missing.groupby(["role", "kind", "unavailable_reason"], dropna=False).size().rename("units").reset_index()
    totals = units.groupby(["role", "kind"]).size().rename("planned_units").reset_index()
    summary = summary.merge(totals, on=["role", "kind"], validate="many_to_one")
    summary["share_of_planned"] = summary.units / summary.planned_units
    return summary


def batch_table(inventory, attempts):
    rows = []
    attempted = set(zip(attempts.version, attempts.job_id, strict=True))
    for role, jobs in inventory["plans"].items():
        ver = inventory["versions"][role]
        for job in jobs:
            result = inventory["results"].get((ver, job["job_id"]))
            targets = json.loads(job["targets"])
            conflict = any("conflicting_results" in inventory["flags"].get((role, job["kind"], key), set())
                           for key in targets)
            state = ("conflicting_results" if conflict else result["status"] if result and result["status"] in outcomes.SETTLED else
                     "recovery_pending" if result and result["error"].startswith("split_required:") else
                     "failed" if result else "started_unconfirmed" if (ver, job["job_id"]) in attempted else "not_started")
            rows.append({"role": role, "kind": job["kind"], "job_id": job["job_id"], "status": state,
                "target_count": len(targets),
                "scope": "adjudication_from_completed_pairs" if role == "adjudicator" else "original_tasks"})
    return pd.DataFrame(rows)


def label_distribution(units, comments, roles=ROLES):
    meta = comments.rename(columns={"comment_id": "id"}).copy()
    meta["is_channel_author"] = pd.to_numeric(meta.is_channel_author, errors="raise").eq(1)
    meta["is_reply"] = pd.to_numeric(meta.is_reply, errors="raise").eq(1)
    coded = units.loc[units.kind.eq("comments") & units.status.eq("complete")].merge(
        meta[["id", "is_channel_author", "is_reply"]], on="id", validate="many_to_one")
    rows = []
    for role in roles:
        group = coded.loc[coded.role.eq(role)]
        for scope in ["all_targets", "audience", "audience_nonspam"]:
            scoped = group if scope == "all_targets" else group.loc[~group.is_channel_author]
            if scope == "audience_nonspam":
                scoped = scoped.loc[pd.to_numeric(scoped.spam, errors="coerce").eq(0)]
            for position in ["all", "root", "reply"]:
                part = scoped if position == "all" else scoped.loc[scoped.is_reply.eq(position == "reply")]
                for label in COMMENT_LABELS:
                    values = pd.to_numeric(part[label], errors="coerce")
                    positive, negative = int(values.eq(1).sum()), int(values.eq(0).sum())
                    resolved, total = positive + negative, len(values)
                    rows.append({"role": role, "scope": scope, "position": position, "label": label,
                        "successful_comments": total, "positive": positive, "negative": negative,
                        "abstained": total - resolved, "resolved": resolved,
                        "positive_over_resolved": positive / resolved if resolved else None,
                        "positive_over_successful": positive / total if total else None,
                        "abstention_rate": (total - resolved) / total if total else None})
    return pd.DataFrame(rows)


def completion_strata(units, comments, videos, roles=("coder", "reviewer")):
    meta = comments.rename(columns={"comment_id": "id"}).copy()
    meta["language_group"] = meta.language.where(meta.language.isin(["es", "und"]), "other")
    meta["length_band"] = pd.cut(pd.to_numeric(meta.char_count), [-1, 0, 20, 80, 240, float("inf")],
                                 labels=["empty", "1-20", "21-80", "81-240", "241+"]).astype(str)
    meta["position"] = pd.to_numeric(meta.is_reply).map({0: "root", 1: "reply"})
    meta = meta.merge(videos[["video_id", "month"]], on="video_id", validate="many_to_one")
    frame = units.loc[units.kind.eq("comments") & units.role.isin(roles)].merge(
        meta[["id", "language_group", "length_band", "position", "month"]], on="id", validate="many_to_one")
    rows = []
    for column in ["language_group", "length_band", "position", "month"]:
        grouped = state_summary(frame, ["role", column]).rename(columns={column: "stratum"})
        grouped["dimension"] = column
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def generate(p, experiment="agents_v1", final=None):
    from . import ai_audit_figures
    root = data.folder(p, experiment)
    if agents.uses_screening(p, root):
        from . import ai_screening
        return ai_screening.audit(p, experiment)
    spec = agents.load_protocol(root)
    event("审计", "读取一致性快照；按请求、原始批次及去重单元统计，不调用模型。")
    snapshot = capture(root)
    captured_at = now()
    inv = inventory(root, spec, snapshot)
    if final is not None and not inv["final_ready"]:
        raise ValueError("完整分析要求的编码结果尚未完成，拒绝发布最终占比")
    attempts = request_table(snapshot, inv)
    units = unit_table(inv)
    batches = batch_table(inv, attempts)
    comments, videos = read_csv(root / "comments.csv"), read_csv(root / "videos.csv")
    labels = label_distribution(units, comments)
    strata = completion_strata(units, comments, videos)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = p.reports / "ai" / experiment / "audit" / timestamp
    dest.mkdir(parents=True, exist_ok=False)
    tables = {"request_attempts.csv": attempts,
        "executions.csv": pd.DataFrame(snapshot["executions"], columns=["execution_id", "started_at", "ended_at", "status", "stop_reason", "request_count"]),
        "execution_events.csv": pd.DataFrame(snapshot["execution_events"], columns=["id", "execution_id", "phase", "event", "reason", "recovery_round", "wait_seconds", "request_count", "created_at"]),
        "request_summary.csv": attempts.groupby(["role", "kind", "outcome"], dropna=False).size().rename("requests").reset_index(),
        "request_errors.csv": attempts.loc[~attempts.status.isin(["complete", "started"])].groupby(
            ["role", "kind", "status"]).size().rename("requests").reset_index(),
        "batch_status.csv": batches, "batch_summary.csv": state_summary(batches, ["role", "kind"]),
        "unit_status.csv.gz": units, "unit_summary.csv": unit_summary(units),
        "unavailable_reasons.csv": unavailable_summary(units),
        "evidence_quarantine.csv": pd.DataFrame(snapshot["evidence_quarantine"], columns=[
            "version", "kind", "target_id", "proof", "execution_id", "created_at"]),
        "comment_label_distribution.csv": labels, "completion_by_stratum.csv": strata}
    for name, table in tables.items():
        write_csv(table, dest / name)
    with gzip.open(dest / "ledger.jsonl.gz", "wt", encoding="utf-8") as stream:
        for name in TABLES:
            for row in snapshot[name]:
                stream.write(json.dumps({"table": name, "row": row}, ensure_ascii=False) + "\n")
    # The immutable journal also keeps legacy inferred child inputs for later reproducibility.
    with gzip.open(dest / "resolved_task_inputs.jsonl.gz", "wt", encoding="utf-8") as stream:
        for (version, _), job in inv["registry"].items():
            stream.write(json.dumps({"version": version, **job}, ensure_ascii=False) + "\n")
    write_json(dest / "agent_protocol.json", spec)
    execution_policies = outcomes.policy_records(root)
    write_json(dest / "execution_policies.json", execution_policies)
    provider_block = latest_provider_block(snapshot["executions"])
    write_json(dest / "provider_block.json", provider_block)
    note = "流程已处理完毕；不可编码记录保留缺失，成功标签单列。AI 一致性不等于准确率。" if inv["final_ready"] else "实验尚未完成：阶段性描述，不代表全体评论的最终分布。"
    figures = ai_audit_figures.generate(dest, tables, final_ready=inv["final_ready"])
    outcome_counts = attempts.outcome.value_counts().to_dict()
    summary = {"created_at": now(), "snapshot_captured_at": captured_at, "snapshot_id": timestamp, "protocol_hash": spec["protocol_hash"],
        "final_ready": inv["final_ready"], "note": note, "request_count": len(attempts), "request_outcomes": outcome_counts,
        "execution_policy_hashes": sorted(execution_policies),
        "provider_block": provider_block,
        "recovery_event_counts": dict(Counter(row["event"] for row in snapshot["execution_events"])),
        "reported_tokens": float(attempts.reported_tokens.sum()), "unknown_token_requests": int(attempts.reported_tokens.isna().sum()),
        "unmapped_requests": int((~attempts.input_reconstructed).sum()), "cache_validation_issues": inv["invalid"],
        "unit_summary": tables["unit_summary.csv"].to_dict("records"),
        "archived_rows": {name: len(rows) for name, rows in snapshot.items()},
        "assistant_output_archive_note": "新版本保存 HTTP 200 的模型正文（含格式失败），并脱敏当前密钥；旧失败正文无法追回。HTTP 错误正文不保存。",
        "history_note": "attempts 是每次请求；results 是最新状态；result_history 从本次升级后的运行开始追加，不伪造旧历史。",
        "unavailable_note": "服务拒绝为 unavailable，不计校验成功或模型弃权；最终该单元标签保持缺失，不重试绕过拒绝，不缩小输入总数。",
        "execution_note": "executions 保存新版每次运行的结束状态与安全停止原因；旧轮次缺少此记录，不反推具体网络异常类型。",
        "denominator_note": "成功是结构与证据校验通过，不代表语义准确。多标签可重叠；每类阳性/已判断与阳性/成功评论分别报告，null 不计为 0。",
        "strata_note": "未完成时覆盖差异可能来自任务顺序；不能据此认定缺失随机或非随机。",
        "source_hashes": {name: stable_hash((root / name).read_bytes()) for name in ["manifest.json", "videos.csv", "comments.csv", "agent_protocol.json"]},
        "file_hashes": {f.name: stable_hash(f.read_bytes()) for f in dest.iterdir() if f.is_file()},
        "figure_count": len(figures["figures"]), "figure_manifest": "figures/manifest.json", "real_model_requests": 0}
    write_json(dest / "summary.json", summary)
    lines = ["# 实验执行与评论编码审计", "", note, "",
        f"快照：{timestamp}；协议：`{spec['protocol_hash']}`。", "",
        (f"累计 {len(attempts):,} 次请求：整批通过 {outcome_counts.get('fully_valid', 0):,}，"
        f"部分项通过 {outcome_counts.get('partially_valid', 0):,}，未采信输出 {outcome_counts.get('invalid', 0):,}，"
        f"在途或中断未确认 {outcome_counts.get('started_unconfirmed', 0):,}。"),
        "历史失败请求可能已经恢复，不能当成最终失败评论数；拆批缓存先按角色和目标 ID 去重。", "",
        "| 角色 | 单元 | 计划数 | 校验通过 | 不可编码 | 待恢复 | 当前失败 | 未开始 | 未确认 | 等待全流程 | 缓存异常 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["unit_summary.csv"].itertuples():
        lines.append(f"| {row.role} | {row.kind} | {row.total} | {row.complete} | {row.unavailable} | {row.recovery_pending} | {row.failed} | {row.not_started} | {row.started_unconfirmed} | {row.not_ready} | {row.conflicting_results} |")
    lines += ["", (f"无法映射输入的请求：{summary['unmapped_requests']}；缓存校验/冲突记录：{len(inv['invalid'])}。"
              "非零时须先核对 summary.json；不能将不完整审计视为完整覆盖。")]
    if provider_block:
        lines += ["", "## 最近账户暂停", "",
                  f"{provider_block['reason']}。{provider_block['action']}。",
                  f"记录时间：{provider_block['recorded_at']}；这是服务当时的返回，不是实时余额查询。",
                  "[账户阻断与恢复动作](provider_block.json)。全部成功结果仍保留，暂停目标不作为不可编码或负标签。"]
    lines += ["", "裁决计划仅覆盖当前两轮均完成的原始批次；全量计划在两轮全部结束后确定。final 仅在全流程完成后出现有效标签。",
        "", "## 统计口径与实验限制", "",
        "- 每条评论可同时属于多类，各类占比之和不必为 100%，不使用类别饼图。",
        "- unavailable 是服务拒绝或有限证据失败隔离造成的测量缺失，与模型返回 null 的弃权分开。原因、计数与分母见 unavailable_reasons.csv；不能记为成功或 0。",
        "- `positive_over_resolved = positive / (positive + negative)`；`positive_over_successful = positive / successful_comments`。弃权单列，分母为 0 时为空。",
        "- all_targets 含频道回复；audience 排除频道发言；audience_nonspam 还要求 spam 明确为 0。垃圾标签未决者不混入该子集，所有原始记录仍保留。",
        "- 标签图默认展示 audience；实质回复使用 audience 中的回复作为分母。非垃圾受众、根/回复细分均在源表中。",
        "- 相同模型的两轮盲编码可能存在共同偏差；格式与原文证据通过不等于语义准确。AI 一致性不能作为人工信度。",
        "- 早期预检偏向长输入，任务未完成时的类别比例和覆盖差异不代表全体评论。",
        "- 视频形式缺少字幕时仍缺失；观看量为采集时累计值，学习和分享标签是文本表达，不能证明真实认知改变、扩散或因果机制。",
        "", "## 数据与追溯", "",
        "[逐次请求](request_attempts.csv) · [请求类型](request_summary.csv) · [错误类型](request_errors.csv) · [原始批次](batch_status.csv) · [批次汇总](batch_summary.csv)", "",
        "[逐单元状态、标签与证据（压缩 CSV）](unit_status.csv.gz) · [单元汇总](unit_summary.csv) · [标签分布与分母](comment_label_distribution.csv) · [分层覆盖](completion_by_stratum.csv)", "",
        "[账本快照（压缩 JSONL）](ledger.jsonl.gz) · [任务实际输入及恢复输入](resolved_task_inputs.jsonl.gz) · [冻结协议](agent_protocol.json) · [哈希及摘要](summary.json)", "",
        summary["assistant_output_archive_note"], summary["history_note"], summary["unavailable_note"], summary["execution_note"], "",
        "[编码运行结束状态与停止原因](executions.csv) · [自动恢复与延后重试事件](execution_events.csv) · [执行恢复策略](execution_policies.json)。网络诊断仅保留异常类型和系统错误码，不记录异常正文或代理地址。", "",
        "旧请求缺少执行轮次或准确开始时间时保留未知，不猜造；started 状态也不强行当作失败。"]
    for figure in figures["figures"]:
        lines += ["", f"## {figure['title']}", "", f"![{figure['title']}](figures/{figure['png']})", "",
                  figure["caption"] + f" [PDF](figures/{figure['pdf']})"]
    (dest / "审计报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    parent = p.reports / "ai" / experiment
    relative = str(dest.relative_to(parent))
    write_json(parent / "audit_latest.json", {"created_at": summary["created_at"], "path": relative,
                                             "report": relative + "/审计报告.md", "final_ready": inv["final_ready"]})
    write_text(parent / "审计概览.md", f"# 实验执行与编码审计\n\n{note}\n\n[打开最近完整审计]({relative}/审计报告.md)\n\n"
        "每次审计另存时间戳目录，保留之前的快照；审计不调用模型。\n")
    event("审计", f"已保存 {len(attempts):,} 次请求、去重单元统计及 {len(figures['figures'])} 张图：{dest / '审计报告.md'}")
    return summary, dest, figures
