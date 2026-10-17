"""Bounded provider checks, isolated from production labels and protocol freezing."""
from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from . import ai_agents as agents
from . import ai_client as client
from . import ai_data as data
from .api import StopCollection
from .core import now, stable_hash, write_json


def check(p, experiment="agents_v1", online=False):
    root = data.folder(p, experiment)
    from . import ai_screening
    screening = agents.uses_screening(p, root)
    agents.check_mode(root)
    tasks = data.jobs(root)
    if screening:
        spec = ai_screening.protocol(p, root)
        configs = spec["protocol"]["configurations"]
        frozen = ai_screening.active(root)
        if agents.coder_configuration(p, root) != configs["coder"]:
            raise ValueError("当前初编配置与冻结版本不同，请恢复配置")
    else:
        configs = agents.configurations(p, root)
        frozen = (root / "agent_protocol.json").exists()
        if frozen and agents.load_protocol(root)["protocol"]["configurations"] != configs:
            raise ValueError("配置与正式实验冻结版本不同；恢复原配置或使用新实验后再检查")
    limit = 4 if screening else 6
    selected = [max((j for j in tasks if j["kind"] == kind), key=lambda j: (len(j["payload"]), j["job_id"]))
                for kind in ["videos", "comments"] if any(j["kind"] == kind for j in tasks)]
    parent = p.reports / "ai" / experiment
    if screening:
        parent = parent / "screening"
    report = {"created_at": now(), "experiment": experiment, "online": online,
              "status": "local_checks_only", "ready_for_full_experiment": False,
              "workflow": "screening" if screening else "legacy_three_pass",
              "api_key_present": bool(os.getenv("AI_API_KEY", "").strip()),
              "base_url": configs["coder"]["base_url"], "models": {r: c["model"] for r, c in configs.items()},
              "request_format": configs["coder"]["response_format"],
              "token_limit_field": configs["coder"]["token_limit_field"],
              "max_output_tokens": configs["coder"]["max_completion_tokens"],
              "reasoning_effort": configs["coder"].get("reasoning_effort"),
              "thinking": configs["coder"].get("thinking"),
              "configured_workers": client.worker_count(p),
              "role_settings": {role: {k: c.get(k) for k in ["thinking", "reasoning_effort", "max_completion_tokens"]}
                                for role, c in configs.items()},
              "default_max_requests": p.cfg.get("ai", {}).get("default_max_requests", 0),
              "max_output_tokens_per_run": p.cfg.get("ai", {}).get("max_output_tokens_per_run", 0),
              "requests_per_second": p.cfg.get("ai", {}).get("requests_per_second", 1),
              "configuration_hashes": {r: client.version(c) for r, c in configs.items()},
              "formal_protocol_preexisting": frozen, "formal_results_written": 0,
              "base_jobs": len(tasks), "request_limit": limit if online else 0, "checks": [],
              "semantic_accuracy_validated": False,
              "note": "大输入批次连通性/结构/ID/逐字证据自检，不测量准确率；结果仅存检查目录，不启动统计或全量实验。"}
    if not online:
        write_json(parent / "check_latest.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    dest = parent / "checks" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    dest.mkdir(parents=True, exist_ok=False)
    report["check_directory"] = str(dest)
    write_json(dest / "configurations.json", configs)
    runner, failure = None, None
    passes = {}
    try:
        runner = client.Runner(p, dest, configs["coder"], max_requests=limit)
        runner.stage_labels = {"coder": "初编 1/2", "checker": "检查 2/2"} if screening else {}
        runner.retries = 0  # Fail visibly; the check must never launch a long retry sequence.
        def execute(role, jobs):
            runner.config, runner.version = configs[role], client.version(configs[role])
            output = {}
            for job in jobs:
                db = data.connect(dest)
                with db:
                    db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?)", tuple(job[k] for k in
                               ["job_id", "kind", "split", "payload", "targets"]))
                db.close()
                print(f"接口检查 {role}/{job['kind']}：{len(json.loads(job['targets']))} 个目标；最多 {limit} 次总请求。", flush=True)
                if not runner.call(job):
                    raise StopCollection("接口检查输出未通过校验；查看 check_latest.json 和检查账本")
                saved = runner.cached(job)
                if saved["status"] != "complete":
                    raise StopCollection("接口检查遇到服务拒绝，已记录不可编码；不能当作接口预检通过")
                items = json.loads(saved["labels"])
                output[job["job_id"]] = items
                report["checks"].append({"role": role, "kind": job["kind"], "job_id": job["job_id"],
                    "targets": len(items), "status": "schema_ids_evidence_valid",
                    "unresolved_labels": sum(value is None for item in items for value in item["labels"].values())})
            return output
        for role in (["coder", "checker"] if screening else ["coder", "reviewer"]):
            passes[role] = execute(role, selected)
        if not screening:
            # Check the adjudicator interface even if the two blind passes agree on every label.
            adjudications = []
            for job in selected:
                payload = data.target_subset(json.loads(job["payload"]), json.loads(job["targets"])[:1])
                target = payload["target_ids"][0]
                payload["candidate_annotations"] = {r: [x for x in passes[r][job["job_id"]] if x["id"] == target]
                                                    for r in ["coder", "reviewer"]}
                a, b = [payload["candidate_annotations"][r][0]["labels"] for r in ["coder", "reviewer"]]
                payload["disputed_labels"] = {target: [k for k in a if a[k] != b[k] or a[k] is None]}
                payload["diagnostic_only"] = "Interface check; includes agreements; not production adjudication."
                content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                adjudications.append({"job_id": stable_hash(content), "kind": job["kind"], "split": "diagnostic",
                                      "payload": content, "targets": json.dumps(payload["target_ids"])})
            execute("adjudicator", adjudications)
        report["status"], report["ready_for_full_experiment"] = "online_checks_passed", True
    except (ValueError, StopCollection) as exc:
        failure = exc
        report["status"], report["error"] = "online_check_failed", str(exc)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    finally:
        if runner is not None:
            runner.stop.set()
            runner.client.close()
            runner.record_end(sys.exc_info()[1] or failure)
        db = data.connect(dest)
        try:
            report["usage"] = client.usage_summary([json.loads(r[0]) for r in db.execute("SELECT usage FROM attempts")])
            report["attempt_statuses"] = {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status")}
            report["returned_models"] = sorted({m for row in db.execute("SELECT response FROM results WHERE status='complete'")
                                                if isinstance((m := json.loads(row[0]).get("model")), str)})
        finally:
            db.close()
        write_json(dest / "report.json", report)
        write_json(parent / "check_latest.json", report)
        print(f"检查记录：{parent / 'check_latest.json'}；正式实验未启动。", flush=True)
    if failure:
        raise failure
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
