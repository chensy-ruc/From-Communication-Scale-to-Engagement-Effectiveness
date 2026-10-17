"""Default two-stage coding with targeted checks and a frozen probability audit.

The original three-pass protocol and its caches remain intact. A separate protocol
records the changed measurement procedure; no selected disagreement is silently
converted into a negative label or presented as independently verified truth.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from collections import defaultdict

import pandas as pd

from . import ai_agents as agents
from . import ai_client as client
from . import ai_data as data
from . import ai_outcomes as outcomes
from .api import StopCollection
from .coding import COGNITIVE
from .core import now, read_csv, stable_hash, write_json, write_text
from .progress import event

PROTOCOL_FILE = "screening_protocol.json"
PLAN_FILE = "screening_plan.json"
CHECKER_PROMPT = """ROLE: checker.
你负责抽样质量检查。只依据给出的原始材料、完整相关线程及编码手册重新编码。
你看不到初编标签，不推测它的选择；不因为材料被抽中而假定原编码错误。
逐标签判断，兼顾漏标与多标，核对否定、反问、引语归属和跨文化比较边界。
仅返回全部目标的完整 JSON 标签、逐字证据和简短备注；无证据时保留 null。
不生成一致率、准确率或研究结论，不声称人工核验；无需输出思考过程。
"""
CRITICAL = ["fact_check", "cross_cultural", "substantive_response",
            "learning_or_reconsideration", "sharing_intent"]


def active(root):
    return (root / PROTOCOL_FILE).exists()


def protocol(p, root, create=False):
    agents.check_mode(root)
    path = root / PROTOCOL_FILE
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and saved["protocol_hash"] != client.version(saved["protocol"]):
        raise ValueError("检查冻结协议哈希不一致，请恢复文件")
    # Existing experiments keep their exact protocol and paid coder cache. Fresh
    # two-stage experiments have no dependency on a three-role protocol file.
    inherited = ("source_protocol_hash" in saved["protocol"] if saved else
                 (root / "agent_protocol.json").exists())
    original = agents.load_protocol(root) if inherited else None
    cfg = p.cfg.get("ai", {}).get("screening", {})
    rate = cfg.get("random_fraction", .1)
    batch = cfg.get("batch_size", 20)
    limit = cfg.get("max_completion_tokens", 6000)
    if (type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1 or
            type(batch) is not int or not 1 <= batch <= 20 or type(limit) is not int or limit < 1):
        raise ValueError("screening：random_fraction 须为 (0,1]，batch_size 为 1..20，输出上限为正整数")
    coder = (original["protocol"]["configurations"]["coder"] if original else
             saved["protocol"]["configurations"]["coder"] if saved else agents.coder_configuration(p, root, preview=not create))
    checker = {**copy.deepcopy(coder), "agent_role": "checker", "thinking": {"type": "disabled"},
               "instructions": (root / "instructions.txt").read_text() + "\n" + CHECKER_PROMPT,
               "prompt_version": "screening-checker-v1", "max_completion_tokens": limit}
    checker.pop("reasoning_effort", None)
    provenance = ({"source_protocol_hash": original["protocol_hash"]} if original else
                  {"manifest_hash": stable_hash((root / "manifest.json").read_bytes()),
                   "instructions_hash": stable_hash((root / "instructions.txt").read_bytes())})
    spec = {"workflow": "single_code_targeted_and_random_check_v1" if original else "single_code_targeted_and_random_check_v2",
            **provenance, "configurations": {"coder": coder, "checker": checker},
            "random_fraction": rate, "seed": p.cfg["study"]["seed"], "batch_size": batch,
            "risk_rule_version": 1, "critical_positive_labels": CRITICAL,
            "sampling": "stratified_SRS_without_replacement_in_nonrisk_codable_units; ceil(n*fraction); seed_hash_order",
            "decision": "keep unselected coder labels; selected disagreements/abstentions=null; no adjudication",
            "reference_type": "none_ai_only"}
    if saved:
        if saved["protocol"] != spec:
            raise ValueError("检查规则/输入/模型与 screening 冻结协议不同；恢复配置或另建实验")
        return saved
    if not create:
        return {"created_at": None, "protocol": spec, "protocol_hash": client.version(spec)}
    if agents.coder_configuration(p, root) != coder:
        raise ValueError("当前初编配置与已保存标签不兼容；不能用检查流程绕过冻结保护")
    saved = {"created_at": now(), "protocol": spec, "protocol_hash": client.version(spec)}
    for config in spec["configurations"].values():
        version_path = root / "versions" / (client.version(config) + ".json")
        if version_path.exists() and json.loads(version_path.read_text()) != config:
            raise ValueError("保存的 Agent 配置已改变，请恢复记录")
        if not version_path.exists():
            write_json(version_path, config)
    write_json(path, saved)
    return saved


def source_lookup(tasks):
    sources, contexts = {}, {}
    for job in tasks:
        payload = json.loads(job["payload"])
        for context in payload["contexts"]:
            entries = [context] if job["kind"] == "videos" else context["comments"]
            for item in entries:
                key = (job["kind"], item["id"])
                if key in contexts and contexts[key] != context:
                    raise ValueError("同一目标的研究上下文不一致")
                sources[key], contexts[key] = item, context
    return sources, contexts


def risk_reasons(kind, item, source):
    labels = item["labels"]
    eligible = [k for k in labels if not (kind == "videos" and k == "form" and not source["transcript"].strip())]
    reasons = ["unresolved:" + k for k in eligible if not agents.known(labels[k])]
    if kind == "comments":
        reasons += ["critical_positive:" + k for k in CRITICAL if labels[k] == 1]
        if len(source["text"].strip()) <= 20 and any(labels[k] == 1 for k in COGNITIVE):
            reasons.append("short_text_cognitive_positive")
    elif labels["topic"] == "other":
        reasons.append("other_topic")
    return reasons


def selection_plan(root, spec, tasks, first, persist=True):
    """Select before viewing checker responses; retain probabilities for the random domain."""
    sources, contexts = source_lookup(tasks)
    videos = read_csv(root / "videos.csv").set_index("video_id").to_dict("index")
    comments = read_csv(root / "comments.csv").set_index("comment_id").to_dict("index")
    rows, strata = [], defaultdict(list)
    rule = spec["protocol"]
    for job in sorted(tasks, key=lambda j: (j["kind"], j["job_id"])):
        for item in first[job["job_id"]]:
            kind, key = job["kind"], item["id"]
            source = sources[(kind, key)]
            unavailable = (kind, key) in outcomes.unavailable(first)
            reasons = [] if unavailable else risk_reasons(kind, item, source)
            if kind == "videos":
                stratum = json.dumps([kind, videos[key]["month"]])
            else:
                meta = comments[key]
                size = len(source["text"])
                length = "empty" if not size else "1-20" if size <= 20 else "21-80" if size <= 80 else "81-240" if size <= 240 else "241+"
                language = meta["language"] if meta["language"] in {"es", "und"} else "other"
                stratum = json.dumps([kind, language, int(source["is_reply"]), length])
            row = {"kind": kind, "id": key, "base_job_id": job["job_id"], "stratum": stratum,
                   "group": "unavailable" if unavailable else "risk" if reasons else "remaining",
                   "reasons": reasons, "selected": bool(reasons), "inclusion_probability": 1.0 if reasons else 0.0,
                   "stratum_population": None, "stratum_sample": None}
            rows.append(row)
            if row["group"] == "remaining":
                strata[stratum].append(row)
    for stratum, values in strata.items():
        n = len(values)
        k = math.ceil(n * rule["random_fraction"])
        ordered = sorted(values, key=lambda r: stable_hash(json.dumps([rule["seed"], stratum, r["id"]])))
        selected = {r["id"] for r in ordered[:k]}
        for row in values:
            row.update(selected=row["id"] in selected, inclusion_probability=k / n,
                       stratum_population=n, stratum_sample=k)
            if row["selected"]:
                row["group"] = "random"
    # Repack selected targets across original batches. Keeping one sparse check per
    # original batch would repeat the system prompt nearly as often as a full pass.
    jobs = []
    for kind in ["videos", "comments"]:
        selected = [r for r in rows if r["kind"] == kind and r["selected"]]
        size = min(rule["batch_size"], 10 if kind == "videos" else 20)
        for offset in range(0, len(selected), size):
            chunk = selected[offset:offset + size]
            targets = [r["id"] for r in chunk]
            unique = {}
            for target in targets:
                context = contexts[(kind, target)]
                unique[context["id" if kind == "videos" else "root_id"]] = context
            payload = {"kind": kind, "contexts": list(unique.values()), "target_ids": targets}
            text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            jobs.append({"job_id": "check_" + stable_hash(text), "kind": kind, "split": "screening",
                         "payload": text, "targets": json.dumps(targets)})
    record = {"protocol_hash": spec["protocol_hash"],
              "coder_results_hash": client.version({"labels": first, "unavailable": sorted(outcomes.unavailable(first).items())}),
              "selection": rows, "jobs": jobs}
    record["plan_hash"] = client.version(record)
    path = root / PLAN_FILE
    if persist:
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError("已冻结的检查抽样或初编结果发生变化，拒绝重抽或混用")
        if not path.exists():
            write_json(path, record)
    return record


def load_plan(root, spec):
    path = root / PLAN_FILE
    if not path.exists():
        return None
    plan = json.loads(path.read_text())
    if (plan["protocol_hash"] != spec["protocol_hash"] or
            plan["plan_hash"] != client.version({k: v for k, v in plan.items() if k != "plan_hash"})):
        raise ValueError("检查计划哈希与冻结协议不一致")
    return plan


def plan_summary(p, experiment):
    root = data.folder(p, experiment)
    spec = protocol(p, root)
    plan = load_plan(root, spec)
    result = {"workflow": spec["protocol"]["workflow"], "coder_base_jobs": len(data.jobs(root)),
              "stages": ["coder", "checker"], "protocol_frozen": active(root),
              "random_fraction_in_nonrisk_domain": spec["protocol"]["random_fraction"],
              "checker_jobs": len(plan["jobs"]) if plan else None,
              "selected_units": sum(r["selected"] for r in plan["selection"]) if plan else None,
              "full_reviewer_requests": 0, "adjudicator_requests": 0,
              "note": "检查量在初编完成后按风险及分层抽样确定；拆批/重试另计；不是全量双编码。"}
    write_json(p.reports / "ai" / experiment / "screening/plan.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def export_prompts(p, experiment):
    root = data.folder(p, experiment)
    spec = protocol(p, root)
    dest = p.reports / "ai" / experiment / "screening/prompts"
    hashes = {}
    for role, config in spec["protocol"]["configurations"].items():
        for kind in ["videos", "comments"]:
            content = client.request_body(config, {"kind": kind, "payload": "{}"})["messages"][0]["content"]
            name = f"{role}_{kind}.system.txt"
            write_text(dest / name, content)
            hashes[name] = stable_hash(content)
    write_json(dest / "manifest.json", {"protocol_hash": spec["protocol_hash"], "system_hashes": hashes, "model_requests": 0})
    for kind in ["videos", "comments"]:
        write_json(dest / f"{kind}.schema.json", client.schema(kind))
    print(f"已导出初编/检查四份 system 提示词及两个 JSON Schema：{dest}；未调用模型。")
    return hashes


def predictions(spec, tasks, first, plan, checked):
    checked_items = {(j["kind"], x["id"]): x for j in plan["jobs"] for x in checked[j["job_id"]]}
    selection = {(r["kind"], r["id"]): r for r in plan["selection"]}
    final, initial = {"videos": [], "comments": []}, {"videos": [], "comments": []}
    for job in tasks:
        valid_final = []
        for a in first[job["job_id"]]:
            key = job["kind"], a["id"]
            choice, b = selection[key], checked_items.get(key)
            missing = {role: outcomes.unavailable(records)[key] for role, records in
                       [("coder", first), ("checker", checked)] if key in outcomes.unavailable(records)}
            item = copy.deepcopy(a)
            sources = {label: "coder_only" if agents.known(value) else "unresolved" for label, value in a["labels"].items()}
            if choice["selected"]:
                if b is None:
                    raise StopCollection("检查尚未完成，不能把未检查条目当作通过")
                for label, value in item["labels"].items():
                    if agents.known(value) and value == b["labels"][label]:
                        sources[label] = "sampled_check_agreement"
                    else:
                        item["labels"][label], item["evidence"][label] = None, ""
                        sources[label] = "screening_unresolved"
            if missing:
                item = outcomes.placeholder(job["kind"], a["id"], "+".join(sorted(set(missing.values()))))
                sources = dict.fromkeys(item["labels"], "unavailable")
            # Masking opinion on disagreement must also mask dependent reasoned_opinion.
            if job["kind"] == "comments" and item["labels"]["reasoned_opinion"] == 1 and item["labels"]["opinion"] != 1:
                item["labels"]["reasoned_opinion"], item["evidence"]["reasoned_opinion"] = None, ""
                sources["reasoned_opinion"] = "screening_unresolved_dependency"
            def row(value, unavailable, source, label_sources, job=job, choice=choice):
                return {"id": value["id"], **value["labels"], "evidence": json.dumps(value["evidence"], ensure_ascii=False),
                        "note": value["note"], "coding_source": source, "model_version_hash": spec["protocol_hash"],
                        "split": job["split"], "coding_status": "unavailable" if unavailable else "validated",
                        "unavailable_reason": json.dumps(unavailable), "check_group": choice["group"],
                        "decision_agent": "screening" if choice["selected"] and source == "ai_screening_unvalidated" else "coder",
                        "label_sources": json.dumps(label_sources, sort_keys=True)}
            initial[job["kind"]].append(row(a, {"coder": missing["coder"]} if "coder" in missing else {}, "ai_single_code_unvalidated",
                                            {k: "coder_only" if agents.known(v) else "unresolved" for k, v in a["labels"].items()}))
            final[job["kind"]].append(row(item, missing, "ai_screening_unvalidated", sources))
            valid_final.append(item)
        client.validate_response(json.loads(job["payload"]), {"items": valid_final})
    return ({k: pd.DataFrame(v) for k, v in final.items()}, {k: pd.DataFrame(v) for k, v in initial.items()})


def run(p, experiment, max_requests=None, workers=None):
    workers = client.worker_count(p, workers)
    maximum = p.cfg.get("ai", {}).get("default_max_requests", 0) if max_requests is None else max_requests
    if type(maximum) is not int or maximum < 0:
        raise ValueError("max-requests 须为非负整数")
    root = data.folder(p, experiment)
    if not root.exists():
        agents.prepare(p, experiment)
    spec = protocol(p, root, create=True)
    if agents.coder_configuration(p, root) != spec["protocol"]["configurations"]["coder"]:
        raise ValueError("当前初编配置与冻结版本不同，请恢复配置")
    configs, runner = spec["protocol"]["configurations"], None
    event("双阶段计划", "初编 1/2 → 检查 2/2 → 本机统计与制图；复用有效结果，检查覆盖重点及其余分层随机抽样。")
    try:
        tasks = data.jobs(root)
        def execute(role, jobs):
            nonlocal runner
            config = configs[role]
            done = agents.settled(root, client.version(config))
            if all(j["job_id"] in done for j in jobs):
                event("初编 1/2" if role == "coder" else "检查 2/2", f"复用已处理 {len(jobs):,} 批，无新增请求。")
                return agents.records(root, client.version(config), jobs)
            if runner is None:
                runner = client.Runner(p, root, config, maximum)
                runner.stage_labels = {"coder": "初编 1/2", "checker": "检查 2/2"}
            runner.config, runner.version = config, client.version(config)
            return agents.run_stage(runner, jobs, workers, role)
        first = execute("coder", tasks)
        plan = selection_plan(root, spec, tasks, first)
        counts = pd.Series([r["group"] for r in plan["selection"]]).value_counts().to_dict()
        event("检查计划", f"重点 {counts.get('risk', 0):,} 个；随机 {counts.get('random', 0):,} 个；重组为 {len(plan['jobs']):,} 批。")
        execute("checker", plan["jobs"])
    finally:
        error = sys.exc_info()[1]
        if runner is not None:
            runner.stop.set()
            try:
                runner.client.close()
            except Exception as exc:  # noqa: BLE001 - preserve the primary exception
                event("连接清理", f"未完成（{type(exc).__name__}），已保存编码保留。")
            try:
                runner.record_end(error)
            except Exception as exc:
                event("运行记录", f"结束状态未保存（{type(exc).__name__}）。")
                if error is None:
                    raise
        if error is not None and p.cfg.get("ai", {}).get("audit_on_pause", True):
            try:
                audit(p, experiment)
            except Exception as exc:  # noqa: BLE001 - audit errors must not hide the primary cause
                event("检查审计", f"报告未完成（{type(exc).__name__}）；数据库与断点保留。")
    analyze(p, experiment)


def audit(p, experiment):
    from . import ai_screening_report
    return ai_screening_report.audit(p, experiment)


def analyze(p, experiment):
    from . import ai_screening_report
    return ai_screening_report.analyze(p, experiment)
