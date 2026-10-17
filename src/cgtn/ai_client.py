"""Bounded, resumable OpenAI-compatible HTTP annotation. No YouTube credentials used."""
from __future__ import annotations

import json
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx
import pandas as pd

from . import ai_data as data
from . import ai_outcomes as outcomes
from . import ai_quarantine as quarantine
from . import ai_recovery as recovery
from .ai_resilience import AutomaticRecovery
from .ai_stops import ProviderActionRequired, deepseek_content_refusal, provider_action
from .api import StopCollection
from .coding import COMMENT_LABELS, VIDEO_LABELS
from .core import now, stable_hash, write_csv, write_json
from .progress import event
from .progress import label as role_label

MAX_WORKERS = 64


def worker_count(p, override=None):
    workers = p.cfg.get("ai", {}).get("workers", 8) if override is None else override
    if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"AI workers 须为 1..{MAX_WORKERS} 的整数")
    return workers


def settings(p, root, overrides=None, preview=False):
    cfg = {**p.cfg.get("ai", {}), **(overrides or {})}
    base = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    model = os.getenv("AI_MODEL", "").strip()
    parsed = urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("AI_BASE_URL 不得包含密钥、用户名或查询参数")
    if not parsed.hostname or (parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("远程 AI 服务须使用 HTTPS；HTTP 仅允许本机回环地址")
    if not model and preview:
        model = "LOCAL_PREVIEW_ONLY"
    if not model:
        raise ValueError("请在项目 .env 配置 AI_MODEL；YouTube API key 不能用于模型编码")
    out = {"base_url": base, "model": model, "response_format": cfg.get("response_format", "json_schema"),
           "max_completion_tokens": cfg.get("max_completion_tokens", 12000),
           "token_limit_field": cfg.get("token_limit_field", "max_completion_tokens"),
           "instructions": (root / "instructions.txt").read_text(),
           "codebook": (root / "codebook.md").read_text(), "protocol_version": 3, "response_repair_version": 1}
    if out["response_format"] not in {"json_schema", "json_object", "none"}:
        raise ValueError("ai.response_format 必须为 json_schema/json_object/none")
    if out["token_limit_field"] not in {"max_completion_tokens", "max_tokens"}:
        raise ValueError("ai.token_limit_field 必须为 max_completion_tokens 或 max_tokens")
    if type(out["max_completion_tokens"]) is not int or out["max_completion_tokens"] < 1:
        raise ValueError("AI 输出 token 上限须为正整数")
    if "temperature" in cfg:
        out["temperature"] = cfg["temperature"]
    if "reasoning_effort" in cfg:
        out["reasoning_effort"] = cfg["reasoning_effort"]
    if "thinking" in cfg:
        if type(cfg["thinking"]) is not str or cfg["thinking"] not in {"enabled", "disabled"}:
            raise ValueError("ai.thinking 须为 enabled 或 disabled")
        out["thinking"] = {"type": cfg["thinking"]}
    return out


def version(config):
    return stable_hash(json.dumps(config, sort_keys=True, ensure_ascii=False))


def frozen(root, config):
    f = root / "frozen.json"
    if not f.exists():
        raise ValueError("验证集与正式批量编码前须运行 ai-freeze")
    result = json.loads(f.read_text())
    if result["version"] != version(config):
        raise ValueError("模型/地址/提示词/请求参数与冻结版本不同；请恢复或创建新实验")
    if result["reference_hashes"] != data.reference_hashes(root, "validation"):
        raise ValueError("冻结后的验证参考标签已改变，不可沿用原验证结果")
    if result["manifest_hash"] != stable_hash((root / "manifest.json").read_bytes()):
        raise ValueError("冻结实验清单已改变")
    return result


def freeze(p, experiment):
    root = data.folder(p, experiment)
    if data.manifest(root).get("reference_type") == "none_ai_only":
        raise ValueError("全 AI 实验由 agent-run 自动冻结，不需要人工 ai-freeze")
    config = settings(p, root)
    if (root / "frozen.json").exists():
        frozen(root, config)
        print("模型、提示词与验证参考已经冻结；保持不变。")
        return
    dev_report = root / "evaluation_dev.json"
    if not dev_report.exists():
        raise ValueError("先完成开发集 AI 编码、小规模人工参考与 ai-evaluate --split dev")
    report = json.loads(dev_report.read_text())
    if report["version"] != version(config) or report["reference_hashes"] != data.reference_hashes(root, "dev"):
        raise ValueError("开发集模型或参考标签已更新，请重新 ai-evaluate --split dev")
    for kind in ["videos", "comments"]:
        data.reference(root, "validation", kind)
    record = {"version": version(config), "created_at": now(), "configuration": config,
              "manifest_hash": stable_hash((root / "manifest.json").read_bytes()),
              "reference_hashes": data.reference_hashes(root, "validation"),
              "thresholds": thresholds(p)}
    write_json(root / "frozen.json", record)
    print("已冻结模型、提示词、评价标准和独立验证参考。现在可运行 ai-code --split validation。")


def thresholds(p):
    cfg = p.cfg.get("ai", {})
    result = {"min_reference_positives": cfg.get("min_reference_positives", 10),
            "min_reference_negatives": cfg.get("min_reference_negatives", 10),
            "min_precision": cfg.get("min_precision", .8), "min_recall": cfg.get("min_recall", .8),
            "min_prediction_coverage": cfg.get("min_prediction_coverage", .95)}
    for key, value in result.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("AI 验证筛选标准须为有限数值")
        if key.startswith("min_reference"):
            if not isinstance(value, int) or value < 1:
                raise ValueError("验证正/负例下限须为正整数")
        elif not 0 <= value <= 1:
            raise ValueError("验证比例标准须在 0..1")
    return result


def schema(kind):
    labels = data.VIDEO_FIELDS if kind == "videos" else COMMENT_LABELS
    properties = {c: {"type": ["string", "null"], "enum": VIDEO_LABELS[c] + [None]} if kind == "videos"
                  else {"type": ["integer", "null"], "enum": [0, 1, None]} for c in labels}
    def obj(props):
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return obj({"items": {"type": "array", "items": obj({"id": {"type": "string"}, "labels": obj(properties),
                  "evidence": obj({c: {"type": "string"} for c in labels}), "note": {"type": "string"}})}})


def validate_response(payload, body):
    kind = payload["kind"]
    labels = data.VIDEO_FIELDS if kind == "videos" else COMMENT_LABELS
    if not isinstance(body, dict) or set(body) != {"items"} or not isinstance(body["items"], list):
        raise ValueError("invalid_json_shape")
    records = body["items"]
    ids = [x.get("id") if isinstance(x, dict) else None for x in records]
    if any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)) or set(ids) != set(payload["target_ids"]):
        raise ValueError("missing_duplicate_or_invented_ids")
    context = ({r["id"]: r for r in payload["contexts"]} if kind == "videos" else
               {r["id"]: r for thread in payload["contexts"] for r in thread["comments"]})
    result = []
    for r in records:
        if set(r) != {"id", "labels", "evidence", "note"} or not isinstance(r["note"], str):
            raise ValueError("invalid_item_shape")
        if not isinstance(r["labels"], dict) or not isinstance(r["evidence"], dict):
            raise ValueError("invalid_labels_or_evidence")  # noqa: TRY004 - malformed annotation data
        if set(r["labels"]) != set(labels) or set(r["evidence"]) != set(labels):
            raise ValueError("missing_or_extra_label")
        source = context[r["id"]]
        for name, value in r["labels"].items():
            valid = value is None or (value in VIDEO_LABELS[name] if kind == "videos" else type(value) is int and value in {0, 1})
            if not valid:
                raise ValueError("invalid_label_value")
            quote = r["evidence"][name]
            if not isinstance(quote, str):
                raise ValueError("invalid_evidence")  # noqa: TRY004 - malformed annotation data
            if kind == "videos":
                text = "\n".join(source[c] for c in ["title", "description", "transcript"])
                if name == "form":
                    text = source["transcript"]
                    if not text.strip() and value is not None:
                        raise ValueError("form_requires_transcript")
            else:
                text = source["text"]
            if value is not None and (kind == "videos" or value == 1) and (not quote.strip() or quote not in text):
                raise ValueError("positive_label_requires_exact_source_excerpt")
        if kind == "comments":
            lab = r["labels"]
            if not source["is_reply"] and lab["substantive_response"] not in {0, None}:
                raise ValueError("root_cannot_be_substantive_reply")
            if lab["reasoned_opinion"] == 1 and lab["opinion"] != 1:
                raise ValueError("reasoned_opinion_requires_opinion")
        result.append(r)
    return result


def task_codebook(text, kind):
    """Send common definitions and the relevant label section, not unrelated reporting guidance."""
    video_start, comment_start, derived_start = "## 视频标签", "## 评论标签", "## 派生参与质量"
    if not all(marker in text for marker in [video_start, comment_start, derived_start]):
        return text
    common = text[:text.index(video_start)]
    section = (text[text.index(video_start):text.index(comment_start)] if kind == "videos" else
               text[text.index(comment_start):text.index(derived_start)])
    return common + section


def request_body(config, job):
    task_instructions = config.get("task_instructions", {}).get(job["kind"], "")
    body = {"model": config["model"], "messages": [
        {"role": "system", "content": config["instructions"] +
         ("\n" + task_instructions if task_instructions else "") + "\nCODEBOOK:\n" + task_codebook(config["codebook"], job["kind"])},
        {"role": "user", "content": job["payload"]}],
        config["token_limit_field"]: config["max_completion_tokens"]}
    if config["response_format"] == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "research_annotations", "strict": True, "schema": schema(job["kind"])}}
    elif config["response_format"] == "json_object":
        body["response_format"] = {"type": "json_object"}
        body["messages"][0]["content"] += "\nRequired JSON schema:\n" + json.dumps(schema(job["kind"]))
    else:
        body["messages"][0]["content"] += "\nRequired JSON schema:\n" + json.dumps(schema(job["kind"]))
    if urlsplit(config["base_url"]).hostname == "api.openai.com":
        body["store"] = False
    for name in ["temperature", "reasoning_effort", "thinking"]:
        if name in config:
            if name == "reasoning_effort" and config.get("thinking", {}).get("type") == "disabled":
                continue
            body[name] = config[name]
    return body


def retry_after_seconds(value):
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            stamp = parsedate_to_datetime(value)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            delay = max(0.0, (stamp - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None
    return delay if math.isfinite(delay) and delay >= 0 else None


def usage_summary(usages):
    def number(value):
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    totals, latencies, unknown = [], [], 0
    for usage in usages:
        total = usage.get("total_tokens")
        if not number(total):
            incoming = usage.get("prompt_tokens", usage.get("input_tokens"))
            outgoing = usage.get("completion_tokens", usage.get("output_tokens"))
            total = incoming + outgoing if number(incoming) and number(outgoing) else None
        if total is None:
            unknown += 1
        else:
            totals.append(total)
        if number(usage.get("_client_elapsed_seconds")):
            latencies.append(usage["_client_elapsed_seconds"])
    return {"request_attempts": len(usages), "reported_tokens": sum(totals), "unknown_usage_attempts": unknown,
            "mean_request_seconds": sum(latencies) / len(latencies) if latencies else None}


ANNOTATION_ERRORS = {"invalid_json_shape", "missing_duplicate_or_invented_ids", "invalid_item_shape",
                     "invalid_labels_or_evidence", "missing_or_extra_label", "invalid_label_value", "invalid_evidence",
                     "form_requires_transcript", "positive_label_requires_exact_source_excerpt",
                     "root_cannot_be_substantive_reply", "reasoned_opinion_requires_opinion"}

BIGMODEL_FATAL_CODES = {
    "1000": "authentication_failed", "1001": "authentication_missing", "1003": "authentication_expired",
    "1005": "second_factor_required", "1113": "insufficient_quota", "1210": "invalid_parameters",
    "1211": "model_not_found", "1212": "unsupported_method", "1213": "missing_parameter",
    "1214": "invalid_parameter", "1215": "incompatible_parameters", "1220": "access_denied",
    "1221": "endpoint_retired", "1222": "endpoint_not_found", "1308": "period_quota_exhausted",
    "1309": "subscription_expired", "1310": "period_quota_exhausted", "1311": "model_access_denied",
    "1313": "account_restricted", "1314": "enterprise_plan_expired", "1315": "key_product_mismatch",
    "1316": "period_quota_and_balance_exhausted", "1317": "period_quota_and_balance_exhausted",
    "1318": "account_spend_limit", "1319": "account_spend_limit", "1320": "enterprise_spend_limit",
    "1321": "enterprise_spend_limit",
}


def transport_diagnostic(exc):
    """Retain useful types/errno only; exception messages may contain URLs or credentials."""
    names = ["ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError",
             "WriteError", "CloseError", "LocalProtocolError", "RemoteProtocolError", "ProxyError",
             "UnsupportedProtocol", "DecodingError", "TooManyRedirects", "HTTPError"]
    kind = next((name for name in names if isinstance(exc, getattr(httpx, name))), "HTTPError")
    causes, numbers, seen = [], [], set()
    current = exc
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        name = type(current).__name__
        if name in names + ["gaierror", "OSError", "ConnectionRefusedError", "ConnectionResetError",
                            "TimeoutError", "SSLError", "SSLCertVerificationError"]:
            causes.append(name)
        if isinstance(current, OSError) and type(current.errno) is int:
            numbers.append(current.errno)
        current = current.__cause__ or current.__context__
    return {"type": kind, "cause_types": causes, "os_errno": numbers}


class Runner:
    def __init__(self, p, root, config, max_requests):
        cfg = p.cfg.get("ai", {})
        self.root, self.config, self.version = root, config, version(config)
        self.diagnostics_root = p.reports / "ai" / root.name / "diagnostics"
        self.maximum, self.count = max_requests, 0
        self.output_limit = cfg.get("max_output_tokens_per_run", 0)
        if type(self.output_limit) is not int or self.output_limit < 0:
            raise ValueError("ai.max_output_tokens_per_run 须为非负整数，0 表示不限")
        self.output_committed, self.output_reserved, self.output_unknown_requests = 0, 0, 0
        self.output_reservations = {}
        self.execution_id = uuid.uuid4().hex
        self.stop_reason = ""
        self.stop_detail = ""
        self.stop_failure_count = 0
        self.stage_labels = {}
        self.resume_not_before = 0.0
        self.automatic_recovery = AutomaticRecovery(cfg)
        self.phase, self.workers = None, worker_count(p)
        self.stop, self.lock = threading.Event(), threading.Lock()
        # Dispatch waiting and SQLite transactions must never hold the control-state lock.
        self.dispatch_lock, self.write_lock = threading.Lock(), threading.Lock()
        self.inflight = self.peak_inflight = 0
        self.last = self.cooldown_until = 0.0
        self.failure_streak = 0
        self.max_failures = int(cfg.get("max_consecutive_failures", 5))
        self.max_retry_delay = float(cfg.get("max_retry_delay_seconds", 30))
        self.network_retry_delay = float(cfg.get("network_retry_delay_seconds", 10))
        self.trust_env = cfg.get("trust_env", True)
        self.rps = float(cfg.get("requests_per_second", 1))
        self.retries = int(cfg.get("retries", 2))
        self.max_chars = int(cfg.get("max_input_characters", 200000))
        self.progress_every = int(cfg.get("progress_every_jobs", 10))
        self.comment_recovery = cfg.get("comment_recovery", True)
        self.schema_retries = cfg.get("comment_schema_retries", 1)
        if type(self.comment_recovery) is not bool or type(self.schema_retries) is not int or not 0 <= self.schema_retries <= 2:
            raise ValueError("comment_recovery 须为布尔值，comment_schema_retries 须为 0..2 整数")
        self.comment_recovery = self.comment_recovery and config.get("adaptive_batches", False)
        self.quarantine_limit = cfg.get("evidence_quarantine_limit", 0)
        if type(self.quarantine_limit) is not int or not 0 <= self.quarantine_limit <= 100:
            raise ValueError("evidence_quarantine_limit 须为 0..100 整数；0 禁用有限隔离")
        if not self.comment_recovery:
            self.quarantine_limit = 0
        timeout = float(cfg.get("timeout_seconds", 120))
        self.timeout = timeout
        if (max_requests < 0 or not math.isfinite(self.rps) or self.rps <= 0 or self.retries < 0 or self.max_chars < 1
                or self.max_failures < 1 or self.progress_every < 1 or not math.isfinite(timeout) or timeout <= 0
                or not math.isfinite(self.max_retry_delay) or not 0 < self.max_retry_delay <= 60
                or not math.isfinite(self.network_retry_delay) or not 0 < self.network_retry_delay <= self.max_retry_delay
                or type(self.trust_env) is not bool):
            raise ValueError("AI 请求预算、限速、重试次数或输入上限不合法")
        key = os.getenv("AI_API_KEY", "").strip()
        self._credential = key
        if not key and urlsplit(config["base_url"]).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("请在项目 .env 填写 AI_API_KEY；不读取或复用 YOUTUBE_API_KEY")
        self.recovery_runtime = recovery.policy_record(root, True, self.schema_retries) if self.comment_recovery else None
        self.execution_policy = outcomes.policy_record(root)
        self.client = self.new_client()
        db = data.connect(root)
        with db:
            db.execute("INSERT INTO executions VALUES (?,?,?,?,?,?)", (self.execution_id, now(), None, "running", "", 0))
        db.close()
        if self.quarantine_limit:
            self.execution_event("evidence_quarantine_policy", f"limit={self.quarantine_limit};min_attempts={quarantine.MIN_ATTEMPTS}")

    def new_client(self):
        return httpx.Client(timeout=self.timeout,
                            limits=httpx.Limits(max_connections=MAX_WORKERS, max_keepalive_connections=MAX_WORKERS,
                                                keepalive_expiry=60),
                            headers={"Authorization": "Bearer " + self._credential} if self._credential else {},
                            follow_redirects=False, trust_env=self.trust_env)

    def renew_client(self):
        # The controller calls this only after the old worker pool has fully drained.
        self.client.close()
        self.client = self.new_client()

    def execution_event(self, kind, reason, recovery_round=0, wait_seconds=0):
        with self.write_lock, data.transaction(self.root) as db:
            db.execute("INSERT INTO execution_events(execution_id,phase,event,reason,recovery_round,wait_seconds,request_count,created_at) "
                       "VALUES (?,?,?,?,?,?,?,?)", (self.execution_id, self.phase, kind, reason, recovery_round,
                                                   wait_seconds, self.count, now()))

    def pause_service(self, reason):
        with self.lock:
            # A later in-flight transport failure must not overwrite an earlier fatal/configuration stop.
            if not self.stop.is_set():
                self.stop_reason = reason
                self.stop.set()

    def record_end(self, error=None):
        state = ("coding_complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else
                 "awaiting_provider_action" if isinstance(error, ProviderActionRequired) else
                 "paused" if isinstance(error, StopCollection) else "error")
        reason = "interrupted" if isinstance(error, KeyboardInterrupt) else self.stop_reason or ("completed" if error is None
                                     else "unfinished_tasks" if isinstance(error, StopCollection) else "local_or_configuration_error")
        with self.write_lock, data.transaction(self.root) as db:
            db.execute("UPDATE executions SET ended_at=?,status=?,stop_reason=?,request_count=? WHERE execution_id=?",
                       (now(), state, reason, self.count, self.execution_id))

    def save_outcome(self, job, values, missing, response=None):
        if missing:
            raw = {"source": "local_unavailable", "unavailable_targets": missing,
                   "client_execution_policy": self.execution_policy,
                   "child_job_ids": (response or {}).get("child_job_ids", [])}
            if response and response.get("refusal_evidence"):
                raw["refusal_evidence"] = response["refusal_evidence"]
            if response and response.get("quarantine_evidence"):
                raw["quarantine_evidence"] = response["quarantine_evidence"]
            self.save(job, "unavailable", values, raw, "+".join(sorted(set(missing.values()))), local_recovery=True)
        else:
            self.save(job, "complete", values, response, "")

    def cached(self, job):
        db = data.connect(self.root, initialize=False)
        try:
            row = db.execute("SELECT status,labels,error,response FROM results WHERE version=? AND job_id=?",
                             (self.version, job["job_id"])).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def restore_attempt(self, job):
        """Replay a committed response after a crash between the request journal and result cache."""
        db = data.connect(self.root, initialize=False)
        try:
            row = db.execute(
                "SELECT a.id,a.status,a.usage,o.content,o.content_sha256,o.redacted,j.payload "
                "FROM attempts a LEFT JOIN attempt_outputs o ON o.attempt_id=a.id "
                "JOIN job_registry j ON j.version=a.version AND j.job_id=a.job_id "
                "WHERE a.version=? AND a.job_id=? ORDER BY a.id DESC LIMIT 1",
                (self.version, job["job_id"])).fetchone()
        finally:
            db.close()
        if row is None:
            return None
        if json.loads(row["payload"]) != json.loads(job["payload"]):
            raise ValueError("请求账本输入与续传任务不同，拒绝复用或覆盖")
        if row["status"] == "provider_refusal" and self.config.get("adaptive_batches", False):
            self.save_outcome(job, [], dict.fromkeys(json.loads(job["targets"]), "provider_refusal"))
            self.notice(job, "账本恢复", "复用已记录的服务拒绝，保留不可编码；不重复请求")
            return True
        usage = json.loads(row["usage"])
        if (row["status"] == "http_400" and self.config.get("adaptive_batches", False)
                and self.restore_diagnosed_refusal(job, row["id"], usage)):
            return True
        complete = row["status"] == "complete"
        partial = self.comment_recovery and row["status"] in recovery.REPLAY_ERRORS
        if not (complete or partial) or row["content"] is None or row["redacted"]:
            return None
        if stable_hash(row["content"]) != row["content_sha256"]:
            raise ValueError("已归档模型正文哈希不一致，请恢复原始账本；不自动覆盖")
        payload = json.loads(job["payload"])
        repairs = []
        try:
            if self.comment_recovery:
                decoded, repairs = recovery.decode(row["content"])
                decoded, shape_repairs = recovery.normalize(decoded, payload["target_ids"], payload)
                repairs += shape_repairs
            else:
                decoded = recovery.load_json(row["content"])
            job["_identity_split_eligible"] = recovery.identity_split_eligible(payload, decoded)
            if complete:
                labels = validate_response(payload, decoded)
            else:
                labels, issues = recovery.inspect_items(payload, decoded, validate_response, allow_partial_ids=True)
        except (ValueError, TypeError, KeyError):
            # Never fabricate a missing semantic label or evidence excerpt.
            return None
        if complete or labels and len(labels) == len(payload["target_ids"]):
            validate_response(payload, {"items": labels})
            self.save(job, "complete", labels, {"source": "recovered_from_attempt", "origin_attempt_id": row["id"],
                      "usage": usage, "client_comment_recovery": self.recovery_runtime, "format_repairs": repairs}, "")
            self.notice(job, "账本恢复", f"从已落盘请求 {row['id']} 恢复 {len(labels)} 个目标；本次不新增请求")
            return True
        if not labels:
            # Carry the archived, locally recomputed diagnosis into the FIRST
            # resumed request rather than repeating the original failing prompt.
            job["_recovery_feedback"] = {"error": row["status"], "issues": issues}
            return None
        valid = {item["id"] for item in labels}
        groups = [[key for key in payload["target_ids"] if (key in valid) == keep] for keep in [True, False]]
        self.notice(job, "账本恢复", f"请求 {row['id']} 已有 {len(labels)}/{len(payload['target_ids'])} 项通过；仅补未通过项")
        return self.partition(job, groups, row["status"], labels, issues, row["id"])

    def restore_diagnosed_refusal(self, job, attempt_id, usage):
        """Reuse an independent diagnosis only for the exact latest failed request in this experiment."""
        request_hash = stable_hash(json.dumps(request_body(self.config, job), ensure_ascii=False, sort_keys=True))
        if usage.get("_client_http_status") != 400 or usage.get("_client_request_sha256") != request_hash:
            return False
        for path in sorted(self.diagnostics_root.glob("*/diagnosis.json")):
            content = path.read_bytes()
            record = json.loads(content)
            if (not isinstance(record, dict) or record.get("kind") != "single_failed_request_diagnosis"
                    or record.get("source_attempt_id") != attempt_id or record.get("job_id") != job["job_id"]
                    or record.get("request_sha256") != request_hash or record.get("model") != self.config["model"]
                    or record.get("parameters_unchanged") is not True or record.get("request_count") != 1
                    or not deepseek_content_refusal(self.config["base_url"], record.get("http_status"),
                                                   record.get("provider_error"))):
                continue
            evidence = {"source": "independent_diagnostic", "source_attempt_id": attempt_id,
                        "diagnostic_sha256": stable_hash(content), "request_sha256": request_hash,
                        "provider": "deepseek", "reason": "content_exists_risk"}
            self.save_outcome(job, [], dict.fromkeys(json.loads(job["targets"]), "provider_refusal"),
                              {"refusal_evidence": evidence})
            self.notice(job, "不可编码", "复用同一请求的独立诊断：DeepSeek 内容拒绝；保留缺失，本次不新增请求")
            return True
        return False

    def reserve_request(self):
        """Called under dispatch_lock; rate/cooldown waits leave stop and result handlers unblocked."""
        while True:
            with self.lock:
                if self.stop.is_set():
                    raise StopCollection("AI 任务已停止，已保存结果可续传")
                if self.maximum and self.count >= self.maximum:
                    self.stop_reason = "request_budget_exhausted"
                    self.stop.set()
                    raise StopCollection("本次 AI 请求预算已到，重跑同一命令续传")
                allowance = self.config["max_completion_tokens"]
                ready_at = max(self.last + 1 / self.rps, self.cooldown_until)
                delay = max(0, ready_at - time.monotonic())
            if self.stop.wait(delay):
                raise StopCollection("AI 任务已停止，已保存结果可续传")
            with self.lock:
                if self.stop.is_set():
                    raise StopCollection("AI 任务已停止，已保存结果可续传")
                # An in-flight response can extend the shared cooldown while this worker waits.
                if self.cooldown_until > ready_at:
                    continue
                if self.output_limit and self.output_committed + self.output_reserved + allowance > self.output_limit:
                    self.stop_reason = "output_token_budget_exhausted"
                    self.stop.set()
                    raise StopCollection("本次输出 token 派发预算不足，已保存进度；包含在途预留和未知用量估计，重跑续传")
                self.count += 1
                self.output_reserved += allowance
                self.output_reservations[self.count] = allowance
                self.last = time.monotonic()
                return self.count

    def settle_output_budget(self, sequence, usage):
        """Account once, refund known unused output; unknown usage retains its full reservation."""
        with self.lock:
            allowance = self.output_reservations.pop(sequence)
            tokens = usage.get("completion_tokens")
            known = type(tokens) is int and tokens >= 0
            self.output_reserved -= allowance
            self.output_committed += tokens if known else allowance
            self.output_unknown_requests += not known
            return {"limit": self.output_limit, "request_reserved": allowance,
                    "request_accounted": tokens if known else allowance, "usage_known": known,
                    "committed": self.output_committed, "inflight_reserved": self.output_reserved,
                    "unknown_usage_requests": self.output_unknown_requests}

    def post(self, request, metadata):
        """Count actual overlapping HTTP calls, not the queued/cached parent jobs."""
        with self.lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
            metadata["_client_http_inflight_at_start"] = self.inflight
            metadata["_client_peak_http_inflight"] = self.peak_inflight
        metadata["_client_http_started_at"] = now()
        try:
            return self.client.post(self.config["base_url"] + "/chat/completions", json=request)
        finally:
            metadata["_client_http_finished_at"] = now()
            with self.lock:
                self.inflight -= 1

    def fail(self, job, error):
        if error == quarantine.ERROR and self.quarantine(job):
            return True
        self.save(job, "error", [], None, error)
        self.notice(job, "失败", error + "；保留未完成状态供续传")
        with self.lock:
            self.failure_streak += 1
            if self.failure_streak >= self.max_failures:
                if not self.stop.is_set():
                    self.stop_reason = "consecutive_annotation_failures"
                    self.stop_detail = error
                    self.stop_failure_count = self.failure_streak
                    self.stop.set()
                raise StopCollection(f"连续 {self.failure_streak} 个任务失败（{error}），已停止新请求以控制消耗；修复后同命令续传")
        return False

    def quarantine(self, job):
        if not self.quarantine_limit or job["kind"] != "comments" or len(json.loads(job["targets"])) != 1:
            return False
        db = data.connect(self.root, initialize=False)
        try:
            proof = quarantine.candidate(db, self.version, job, validate_response)
        finally:
            db.close()
        if proof is None:
            return False
        quarantine.validate_proof(json.loads(job["payload"]), proof["id"], proof)
        with self.write_lock, data.transaction(self.root) as db:
            proof = quarantine.admit(db, proof, self.quarantine_limit, self.execution_id)
        if proof is None:
            self.pause_service("evidence_quarantine_limit_reached")
            raise StopCollection(f"已达到全实验 {self.quarantine_limit} 个角色×评论的编码失败隔离上限；"
                                 "保留失败条目，请检查是否存在系统性证据错误，不再重复请求")
        self.save_outcome(job, [], {proof["id"]: quarantine.REASON}, {"quarantine_evidence": {proof["id"]: proof}})
        self.notice(job, "编码失败隔离", "单条引文经两次请求返回及字段反馈仍不合格；保留原文与失败账本，"
                    "登记测量缺失，不计成功；继续其他任务")
        with self.lock:
            self.failure_streak = 0
        return True

    def stage_label(self, role):
        return self.stage_labels.get(role, role_label(role))

    def notice(self, job, action, detail):
        event(self.stage_label(self.phase or self.config.get("agent_role", "AI")) + "/" + action,
              f"{job['kind']} 批次 {job['job_id'][:12]}：{detail}")

    def split(self, job, reason):
        payload = json.loads(job["payload"])
        ids = payload["target_ids"]
        if len(ids) < 2:
            return self.fail(job, reason + "_single_target_no_truncation")
        return self.partition(job, [ids[:len(ids) // 2], ids[len(ids) // 2:]], reason)

    def partition(self, job, groups, reason, accepted=None, issues=None, origin_attempt=None):
        payload = json.loads(job["payload"])
        flat = [key for group in groups for key in group]
        if (len(groups) < 2 or any(not group for group in groups) or len(flat) != len(set(flat))
                or set(flat) != set(payload["target_ids"])):
            raise ValueError("保存的恢复分组不完整或存在重复，请恢复任务记录")
        marker = "split_required:" + reason
        plan = {"source": "pending_recovery", "target_groups": groups, "validation_issues": issues or [],
                "recovery_reason": reason, "client_comment_recovery": self.recovery_runtime}
        self.save(job, "error", [], plan, marker, local_recovery=True)
        self.notice(job, "拆批", f"{reason}；{len(flat)} 个目标分为 {' + '.join(str(len(g)) for g in groups)}；"
                    "完整相关线程保留，成功子批次复用")
        values, child_ids, missing, unfinished, quarantine_evidence = [], [], {}, False, {}
        accepted_map = {r["id"]: r for r in accepted or []}
        for targets in groups:
            part = data.target_subset(payload, targets)
            content = json.dumps(part, ensure_ascii=False, sort_keys=True)
            child = {**job, "job_id": "split_" + stable_hash(content), "payload": content, "targets": json.dumps(targets),
                     "parent_job_id": job["job_id"]}
            child.pop("_recovery_feedback", None)
            if issues:
                child["_recovery_feedback"] = {"error": reason, "issues": issues}
            child_ids.append(child["job_id"])
            if accepted_map and set(targets) <= set(accepted_map):
                good = validate_response(part, {"items": [accepted_map[key] for key in targets]})
                prior = self.cached(child)
                if prior and prior["status"] in outcomes.SETTLED:
                    previous, absent = outcomes.unpack(part, prior, validate_response)
                    if absent or {r["id"]: r for r in previous} != {r["id"]: r for r in good}:
                        raise ValueError("已保存子批与归档有效项不一致，拒绝覆盖")
                else:
                    self.save(child, "complete", good, {"source": "validated_subset", "parent_job_id": job["job_id"],
                              "origin_attempt_id": origin_attempt, "client_comment_recovery": self.recovery_runtime}, "")
            if not self.call(child):
                unfinished = True
                continue  # A bad child must not starve the remaining siblings; keep the parent's recovery plan.
            saved = self.cached(child)
            valid, unavailable = outcomes.unpack(part, saved, validate_response)
            values.extend(valid)
            missing.update(unavailable)
            if unavailable:
                quarantine_evidence.update(json.loads(saved["response"]).get("quarantine_evidence", {}))
        if unfinished:
            return False
        validate_response(data.target_subset(payload, [key for key in flat if key not in missing]), {"items": values})
        self.save_outcome(job, values, missing, {"source": "locally_assembled_split", "child_job_ids": child_ids,
                                               "client_comment_recovery": self.recovery_runtime,
                                               "quarantine_evidence": quarantine_evidence})
        return True

    def call(self, job):
        if self.stop.is_set():
            raise StopCollection("AI 任务已停止，已保存结果可续传")
        job.pop("_identity_split_eligible", None)  # Recompute only from this job's verified archive.
        body = request_body(self.config, job)
        payload = json.loads(job["payload"])
        cached = self.cached(job)
        if cached and cached["status"] in outcomes.SETTLED:
            outcomes.unpack(payload, cached, validate_response)
            return True
        restored = self.restore_attempt(job)
        if restored is not None:
            return restored
        if self.quarantine(job):
            return True
        adaptive = self.config.get("adaptive_batches", False)
        recover_items = self.comment_recovery
        if adaptive and cached and cached["error"] == "provider_refusal":
            self.save_outcome(job, [], dict.fromkeys(payload["target_ids"], "provider_refusal"))
            self.notice(job, "不可编码", "复用服务拒绝记录；保留缺失与原因，不再次请求")
            return True
        if adaptive and cached and cached["error"].startswith("split_required:"):
            plan = json.loads(cached["response"]) if cached.get("response") else None
            if isinstance(plan, dict) and plan.get("source") == "pending_recovery":
                return self.partition(job, plan["target_groups"], plan["recovery_reason"], issues=plan.get("validation_issues"))
            return self.split(job, cached["error"].split(":", 1)[1])
        if (recover_items and cached and cached["error"] in recovery.SPLIT_ERRORS and len(payload["target_ids"]) > 1
                and (cached["error"] != "missing_duplicate_or_invented_ids" or job.get("_identity_split_eligible", False))):
            return self.split(job, cached["error"])  # Do not repeat an already exhausted large failing batch.
        error = "unknown_error"
        repair_info = job.get("_recovery_feedback", {}) if recover_items else {}
        issues = repair_info.get("issues", [])
        schema_failures = 0
        if len(json.dumps(body, ensure_ascii=False)) > self.max_chars:
            if adaptive:
                return self.split(job, "input_exceeds_configured_limit")
            return self.fail(job, "input_exceeds_configured_limit_no_truncation")
        for attempt in range(self.retries + 1):
            request = {**body, "messages": [dict(m) for m in body["messages"]]}
            repair_error = error if attempt else repair_info.get("error", "")
            if repair_error in ANNOTATION_ERRORS or repair_error in {"invalid_json", "output_truncated"}:
                request["messages"][0]["content"] += (
                    "\n<output_repair>上次输出未通过程序校验，错误代码：" + repair_error +
                    "。请重新核对全部目标 ID、JSON 字段、整数/枚举及逐字证据；不能确认的标签用 null，"
                    "note 与引文保持简短。仅输出完整 JSON，不输出解释或代码围栏。</output_repair>")
                if recover_items:
                    request["messages"][0]["content"] += recovery.feedback(issues, payload["target_ids"], job["kind"])
                    focused = recovery.focus_payload(payload, issues)
                    if focused is not payload:
                        original_content = request["messages"][1]["content"]
                        request["messages"][1]["content"] = json.dumps(focused, ensure_ascii=False, sort_keys=True)
                        if len(json.dumps(request, ensure_ascii=False)) > self.max_chars:
                            # Optional copies must not make otherwise valid full contexts unfit.
                            request["messages"][1]["content"] = original_content
            if len(json.dumps(request, ensure_ascii=False)) > self.max_chars:
                if adaptive:
                    return self.split(job, "input_exceeds_configured_limit")
                return self.fail(job, "input_exceeds_configured_limit_no_truncation")
            queued_at = time.monotonic()
            with self.dispatch_lock:
                sequence = self.reserve_request()
                metadata = {"_client_execution_id": self.execution_id, "_client_started_at": now(),
                            "_client_request_sequence": sequence,
                            "_client_dispatch_wait_seconds": round(time.monotonic() - queued_at, 6),
                            "_client_kind": job["kind"], "_client_target_count": len(payload["target_ids"]),
                            "_client_phase": self.phase, "_client_retry_index": attempt,
                            "_client_runtime": {"workers": self.workers, "max_requests": self.maximum,
                                "max_output_tokens_per_run": self.output_limit,
                                "requests_per_second": self.rps, "timeout_seconds": self.timeout, "retries": self.retries,
                                "max_input_characters": self.max_chars, "max_consecutive_failures": self.max_failures,
                                "max_retry_delay_seconds": self.max_retry_delay, "comment_recovery": self.comment_recovery,
                                "comment_schema_retries": self.schema_retries, "network_retry_delay_seconds": self.network_retry_delay,
                                "evidence_quarantine_limit": self.quarantine_limit,
                                "trust_env": self.trust_env, "http_max_connections": MAX_WORKERS,
                                **self.automatic_recovery.settings()},
                            "_client_recovery_round": self.automatic_recovery.used,
                            "_client_execution_policy": self.execution_policy,
                            "_client_request_sha256": stable_hash(json.dumps(request, ensure_ascii=False, sort_keys=True))}
                with self.write_lock, data.transaction(self.root) as db:
                    data.register_job(db, self.version, job)
                    cursor = db.execute("INSERT INTO attempts(version,job_id,status,usage,updated_at) VALUES (?,?,?,?,?)",
                                        (self.version, job["job_id"], "started", json.dumps(metadata), now()))
                    aid = cursor.lastrowid
            usage, raw, labels, fatal, terminal = {}, None, None, False, False
            decoded, repairs, accepted = None, [], []
            assistant_content, json_error, network_error = None, None, None
            http_status, finish_reason = None, None
            sent_feedback = [i for i in issues if i["id"] in payload["target_ids"]][:4] if repair_error else []
            issues = []
            backoff, transient = None, False
            started = time.monotonic()
            try:
                response = self.post(request, metadata)
                http_status = response.status_code
                if response.status_code != 200:
                    error = f"http_{response.status_code}"
                    transient = response.status_code in {408, 429, 500, 502, 503, 504}
                    fatal = not transient
                    backoff = retry_after_seconds(response.headers.get("Retry-After")) if transient else None
                    provider_error = None
                    try:
                        provider_error = response.json().get("error", {})
                        code = provider_error.get("code") if isinstance(provider_error, dict) else None
                    except (ValueError, AttributeError):
                        code = None
                    # Only recognized codes are retained; provider error prose may echo credentials.
                    if isinstance(code, str) and code in {"insufficient_quota", "billing_hard_limit_reached"}:
                        error, fatal = code, True
                    elif isinstance(code, str) and code in {"context_length_exceeded", "context_window_exceeded"}:
                        error, fatal, transient = "context_length_exceeded", False, False
                    if urlsplit(self.config["base_url"]).hostname == "api.deepseek.com" and http_status == 402:
                        error, fatal, transient = "deepseek_insufficient_balance", True, False
                    if deepseek_content_refusal(self.config["base_url"], http_status, provider_error):
                        error, fatal, transient, terminal = "provider_refusal", False, False, True
                        metadata["_client_provider_refusal_reason"] = "deepseek_content_exists_risk"
                    if urlsplit(self.config["base_url"]).hostname == "open.bigmodel.cn":
                        code = str(code) if isinstance(code, (str, int)) else ""
                        if code in BIGMODEL_FATAL_CODES:
                            error, fatal = "bigmodel_" + code + "_" + BIGMODEL_FATAL_CODES[code], True
                        elif code == "1261":
                            error, fatal, transient = "context_length_exceeded", False, False
                        elif code == "1301":
                            error, fatal, transient, terminal = "provider_refusal", False, False, True
                        elif code in {"1302", "1305"}:
                            error, fatal, transient = "bigmodel_" + code + "_rate_limit", False, True
                else:
                    raw = recovery.load_json(response.content)
                    if not isinstance(raw, dict):
                        raise ValueError("invalid_response_envelope")
                    usage = raw.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    choice = raw["choices"][0]
                    reason = choice.get("finish_reason")
                    finish_reason = reason if reason in {"stop", "length", "content_filter", "tool_calls", "function_call", None} else "other"
                    content = choice["message"].get("content")
                    if isinstance(content, str):
                        assistant_content = content
                    if choice["message"].get("refusal") or choice.get("finish_reason") == "content_filter":
                        error, terminal = "provider_refusal", True
                    elif choice.get("finish_reason") == "length":
                        error = "output_truncated"
                    elif choice.get("finish_reason") != "stop":
                        error = "incomplete_response"
                    else:
                        if recover_items:
                            decoded, repairs = recovery.decode(choice["message"]["content"])
                            decoded, shape_repairs = recovery.normalize(decoded, payload["target_ids"], payload)
                            repairs += shape_repairs
                        else:
                            decoded = recovery.load_json(choice["message"]["content"])
                        labels = validate_response(payload, decoded)
            except httpx.HTTPError as exc:
                error, transient = "network_error", True
                network_error = transport_diagnostic(exc)
                if network_error["type"] in {"LocalProtocolError", "UnsupportedProtocol", "TooManyRedirects"}:
                    error, transient, fatal = "invalid_transport_configuration", False, True
            except json.JSONDecodeError as exc:
                error = "invalid_json"
                json_error = {"line": exc.lineno, "column": exc.colno, "position": exc.pos,
                              "source": "assistant_content" if assistant_content is not None else "response_envelope"}
            except ValueError as exc:
                error = str(exc) if str(exc) in ANNOTATION_ERRORS else "invalid_response_envelope"
            except (KeyError, IndexError, TypeError, AttributeError):
                error = "invalid_response_envelope"
            if (http_status == 200 and assistant_content is None and
                    error in {"invalid_json", "invalid_response_envelope"} and not terminal):
                # Distinct error code also prevents a resumed gateway failure from triggering comment splits.
                error, transient = "invalid_response_envelope", True
            if recover_items and labels is None and decoded is not None:
                accepted, issues = recovery.inspect_items(payload, decoded, validate_response, allow_partial_ids=True)
            usage = {**usage, **metadata, "_client_elapsed_seconds": round(time.monotonic() - started, 6),
                     "_client_http_status": http_status, "_client_finish_reason": finish_reason}
            usage["_client_output_budget"] = self.settle_output_budget(sequence, usage)
            if json_error:
                usage["_client_json_error"] = json_error
            if network_error:
                usage["_client_network_error"] = network_error
            if recover_items:
                usage.update(_client_comment_recovery=self.recovery_runtime, _client_repairs=repairs,
                             _client_validation_issues=issues, _client_valid_items=len(labels) if labels is not None else len(accepted),
                             _client_feedback=sent_feedback[:4],
                             _client_request_sha256=stable_hash(json.dumps(request, ensure_ascii=False, sort_keys=True)))
            with self.write_lock, data.transaction(self.root) as db:
                if assistant_content is not None:
                    # Only HTTP-200 assistant text, never HTTP error bodies, headers or reasoning traces.
                    # Preserve malformed annotations for diagnosis; exclude any literal configured credential.
                    credential = self._credential
                    redacted = bool(credential and credential in assistant_content)
                    content = assistant_content.replace(credential, "[REDACTED_API_KEY]") if redacted else assistant_content
                    db.execute("INSERT INTO attempt_outputs VALUES (?,?,?,?)", (aid, content, stable_hash(content), int(redacted)))
                db.execute("UPDATE attempts SET status=?,usage=?,updated_at=? WHERE id=?",
                           ("complete" if labels is not None else error, json.dumps(usage), now(), aid))
            if labels is not None:
                if repairs:
                    self.notice(job, "格式恢复", f"完成 {len(repairs)} 处格式修复，语义标签和正标签证据均未代填")
                if recover_items:
                    raw = {**raw, "client_comment_recovery": self.recovery_runtime, "format_repairs": repairs,
                           "origin_attempt_id": aid}
                self.save(job, "complete", labels, raw, "")
                with self.lock:
                    self.failure_streak = 0
                return True
            self.save(job, "error", [], raw, error)
            if issues:
                first_issue = issues[0]
                details = json.dumps({k: v for k, v in first_issue.items() if k not in {"id", "code"}}, ensure_ascii=False)
                self.notice(job, "字段诊断", f"{len(issues)} 条未通过；首条 {first_issue['id']}：{first_issue['code']} {details}；详情已入请求账本")
            if fatal:
                with self.lock:
                    self.stop_reason = error
                    self.stop.set()
                if provider_action(error):
                    self.execution_event("provider_action_required", error)
                    self.notice(job, "账户暂停", provider_action(error)["reason"])
                    raise ProviderActionRequired(error)
                self.notice(job, "停止", error)
                raise ValueError(f"AI 接口 {error}，已停止；核对服务、模型、参数与密钥。密钥及服务错误正文不会输出。")
            if terminal:
                if adaptive and error == "provider_refusal":
                    self.save_outcome(job, [], dict.fromkeys(payload["target_ids"], error))
                    self.notice(job, "不可编码", f"服务拒绝 {len(payload['target_ids'])} 个目标；单列缺失，不计成功，不阻塞其他任务")
                    return True
                return self.fail(job, error)
            if error == quarantine.ERROR and self.quarantine(job):
                return True
            if recover_items and accepted and len(accepted) < len(payload["target_ids"]):
                valid_ids = {r["id"] for r in accepted}
                groups = [[key for key in payload["target_ids"] if (key in valid_ids) == keep] for keep in [True, False]]
                self.notice(job, "部分保存", f"{len(accepted)}/{len(payload['target_ids'])} 条已通过；仅补未通过项")
                return self.partition(job, groups, error, accepted, issues, aid)
            if (recover_items and error in recovery.SPLIT_ERRORS
                    and (error != "missing_duplicate_or_invented_ids" or recovery.identity_split_eligible(payload, decoded))):
                schema_failures += 1
                if len(payload["target_ids"]) > 1 and (schema_failures > self.schema_retries or attempt == self.retries):
                    ids = payload["target_ids"]
                    return self.partition(job, [ids[:len(ids) // 2], ids[len(ids) // 2:]], error, issues=issues)
            if adaptive and error in {"output_truncated", "context_length_exceeded"}:
                return self.split(job, error)
            delay = backoff if backoff is not None else min(2 ** attempt + random.random() * .25, self.max_retry_delay)
            if error == "network_error":
                delay = min(self.network_retry_delay * 2 ** attempt, self.max_retry_delay)
                if attempt == self.retries:
                    self.pause_service("network_retries_exhausted")
                    self.notice(job, "网络暂停", f"{network_error['type']}；有限重试已耗尽，停止新请求，缓存可续传")
                    raise StopCollection(f"网络连接/传输连续失败（{network_error['type']}），已保存进度。"
                                         "检查网络、VPN/代理后重跑同命令；详情见请求账本。")
            if transient:
                with self.lock:
                    self.resume_not_before = max(self.resume_not_before, time.monotonic() + delay)
                if delay > self.max_retry_delay:
                    self.pause_service("retry_after_exceeds_wait_limit")
                    self.notice(job, "暂缓", f"{error}；Retry-After={delay:.1f} 秒超过等待上限")
                    raise StopCollection("服务要求的 Retry-After 超过单次重试等待上限；已暂停派发，由自动恢复检查等待预算")
                if attempt == self.retries:
                    self.pause_service("transient_retries_exhausted")
                    self.notice(job, "服务暂缓", f"{error}；本批有限重试已耗尽，暂停派发后尝试自动恢复")
                    raise StopCollection(f"临时服务错误 {error}；已保存进度，等待自动恢复或同命令续传")
                with self.lock:
                    self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)
                detail = error + (f"/{network_error['type']}" if network_error else "")
                self.notice(job, "冷却", f"{detail}；所有线程暂停新请求至少 {delay:.1f} 秒")
            if attempt < self.retries:
                self.notice(job, "待重试", f"{error}；下一次为第 {attempt + 2}/{self.retries + 1} 次尝试，计入总请求预算")
            if attempt < self.retries and not transient and self.stop.wait(delay):
                raise StopCollection("AI 编码已停止，重跑可续传")
        return self.fail(job, error)

    def save(self, job, status, labels, response, error, local_recovery=False):
        # Never persist provider error bodies or headers, which may echo credentials.
        safe_response = None
        if status == "complete" and response:
            safe_response = {k: response.get(k) for k in ["id", "model", "usage", "choices", "source", "child_job_ids",
                "parent_job_id", "origin_attempt_id", "client_comment_recovery", "format_repairs"]}
        elif status == "unavailable" and local_recovery and response and response.get("source") == "local_unavailable":
            safe_response = {k: response[k] for k in ["source", "unavailable_targets", "client_execution_policy", "child_job_ids"]}
            if response.get("refusal_evidence"):
                safe_response["refusal_evidence"] = response["refusal_evidence"]
            if response.get("quarantine_evidence"):
                safe_response["quarantine_evidence"] = response["quarantine_evidence"]
            outcomes.unpack(json.loads(job["payload"]), {"status": status, "labels": json.dumps(labels),
                            "response": json.dumps(safe_response)}, validate_response)
        elif status == "unavailable":
            raise ValueError("不可编码结果只能由本机执行策略登记")
        elif status == "error" and local_recovery and response and response.get("source") == "pending_recovery":
            safe_response = {k: response[k] for k in ["source", "target_groups", "validation_issues",
                                                     "recovery_reason", "client_comment_recovery"]}
        with self.write_lock, data.transaction(self.root) as db:
            data.register_job(db, self.version, job)
            values = (
                self.version, job["job_id"], status, json.dumps(labels, ensure_ascii=False),
                json.dumps(safe_response, ensure_ascii=False), error, now())
            db.execute("INSERT INTO result_history(version,job_id,status,labels,response,error,updated_at) VALUES (?,?,?,?,?,?,?)", values)
            db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?)", values)


def predictions(root, ver, split="all", require_complete=True):
    tasks = data.jobs(root, split)
    db = data.connect(root)
    rows = {r["job_id"]: dict(r) for r in db.execute("SELECT * FROM results WHERE version=?", (ver,))}
    db.close()
    missing = [j for j in tasks if rows.get(j["job_id"], {}).get("status") != "complete"]
    if missing and require_complete:
        raise ValueError(f"{split} 还有 {len(missing)} 个 AI 任务未完成；运行 ai-code 续传")
    result = {"videos": [], "comments": []}
    for job in tasks:
        row = rows.get(job["job_id"], {})
        if row.get("status") != "complete":
            continue
        values = validate_response(json.loads(job["payload"]), {"items": json.loads(row["labels"])})
        for item in values:
            result[job["kind"]].append({"id": item["id"], **item["labels"],
                "evidence": json.dumps(item["evidence"], ensure_ascii=False), "note": item["note"],
                "coding_source": "model", "model_version_hash": ver, "split": job["split"]})
    return {kind: pd.DataFrame(values, columns=["id"] + (data.VIDEO_FIELDS if kind == "videos" else COMMENT_LABELS)
             + ["evidence", "note", "coding_source", "model_version_hash", "split"]) for kind, values in result.items()}


def status(p, experiment):
    root = data.folder(p, experiment)
    tasks = data.jobs(root)
    db = data.connect(root)
    versions = [r[0] for r in db.execute("SELECT DISTINCT version FROM attempts UNION SELECT DISTINCT version FROM results")]
    summary = []
    for ver in versions:
        results = {r["job_id"]: r["status"] for r in db.execute("SELECT job_id,status FROM results WHERE version=?", (ver,))}
        attempts = list(db.execute("SELECT usage,status FROM attempts WHERE version=?", (ver,)))
        usages = [json.loads(r["usage"]) for r in attempts]
        usage_totals = usage_summary(usages)
        for split in ["dev", "validation", "production"]:
            jj = [j for j in tasks if j["split"] == split]
            summary.append({"version": ver, "split": split, "jobs": len(jj),
                            "complete": sum(results.get(j["job_id"]) == "complete" for j in jj),
                            "error": sum(results.get(j["job_id"]) == "error" for j in jj)})
        print(f"版本 {ver[:12]}：累计请求尝试 {len(attempts)}；服务报告 token "
              f"{usage_totals['reported_tokens']}；用量未知请求 {usage_totals['unknown_usage_attempts']}。")
    db.close()
    dest = p.reports / "ai" / experiment
    dest.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(summary, columns=["version", "split", "jobs", "complete", "error"])
    write_csv(frame, dest / "status.csv")
    print(frame.to_string(index=False) if len(frame) else f"尚未调用模型；共 {len(tasks)} 个任务。")


def run(p, experiment, split, max_requests=500, workers=None):
    workers = worker_count(p, workers)
    root = data.folder(p, experiment)
    if data.manifest(root).get("reference_type") == "none_ai_only":
        raise ValueError("全 AI 实验请使用 agent-run；原 ai-code 保留人工验证方案")
    tasks = data.jobs(root, split)
    config = settings(p, root)
    if split != "dev" or (root / "frozen.json").exists():
        frozen(root, config)
    if split in {"all", "production"}:
        from .ai_eval import checked
        checked(p, experiment)
    config_path = root / "versions" / (version(config) + ".json")
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("已保存的模型配置版本被修改，请恢复原配置记录")
    if not config_path.exists():
        write_json(config_path, config)
    db = data.connect(root)
    done = {r[0] for r in db.execute("SELECT job_id FROM results WHERE version=? AND status='complete'", (version(config),))}
    db.close()
    remaining = [j for j in tasks if j["job_id"] not in done]
    if not remaining:
        print(f"{split} 全部已完成；复用保存结果，不请求接口。")
        return
    runner = Runner(p, root, config, max_requests)
    runner.workers = workers
    pool, pending = ThreadPoolExecutor(max_workers=workers), {}
    queue, stopped, completed = iter(remaining), None, 0
    print(f"{split} 待处理 {len(remaining)} 个任务；本次最多 {max_requests or '不限'} 次请求（含重试）。", flush=True)
    try:
        while True:
            while not runner.stop.is_set() and len(pending) < workers:
                job = next(queue, None)
                if job is None:
                    break
                pending[pool.submit(runner.call, job)] = job["job_id"]
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                try:
                    future.result()
                    completed += 1
                    if completed % 10 == 0:
                        print(f"本次已处理 {completed}/{len(remaining)} 个任务。", flush=True)
                except (ValueError, StopCollection) as exc:
                    runner.stop.set()
                    if stopped is None or isinstance(exc, ValueError):
                        stopped = exc
    finally:
        runner.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        runner.client.close()
        status(p, experiment)
    if stopped:
        raise stopped
    try:
        predictions(root, version(config), split)
    except ValueError as exc:
        raise StopCollection(str(exc)) from None
    print("当前阶段已完成。输出中的 null 表示模型未判断，不会当作 0。")
