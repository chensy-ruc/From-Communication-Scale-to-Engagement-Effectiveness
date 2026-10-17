"""Separate validated model output from explicit, locally recorded measurement unavailability."""
from __future__ import annotations

import json

from . import ai_data as data
from . import ai_quarantine as quarantine
from .coding import COMMENT_LABELS
from .core import stable_hash, write_json

SETTLED = {"complete", "unavailable"}
POLICY = {
    "version": "execution-recovery-v9",
    "provider_refusal": "retain original inputs; no automatic retry or bypass; record unavailable, never successful",
    "analysis": "any unavailable blind pass or adjudication leaves the final unit unavailable; no replacement labels",
    "network": "bounded shared cooldown and automatic stage recovery after workers drain; honor Retry-After and global request/wait budgets; journal safe diagnostics and recovery events",
    "annotation": "continue remaining split siblings after isolated failures; bounded deferred retries; unresolved labels remain failed, never replaced",
    "evidence_quarantine": "opt-in bounded missingness amendment: only single comments with two consecutive archived HTTP-200/stop verbatim-evidence failures; latest request feedback must cover the PREVIOUS failed attempt fields, even if new fields fail next; store feedback basis and fields, preserve legacy proofs; revalidate input/output hashes; mark unavailable, never successful/model-null labels; durable unchanged cap across roles and restarts; never quarantine network, account, refusal, truncated or ID failures",
    "reports": "retry transient local report errors without model requests; explicit report state; optional status/audit errors do not mask the primary failure",
    "provider_account": "recognized billing/quota blocks pause awaiting provider action; no automatic retry, fabricated measurement or balance claim; preserve resume position and action guidance",
    "concurrency": "1..64 bounded workers; global dispatch rate and request budget; cooldown waits release the control lock; serialize durable short WAL writes; record HTTP overlap and dispatch wait; retain blind-stage barriers",
    "committed_response_recovery": "revalidate the latest matching archived response before issuing a new request; reuse valid complete/partial items and recorded refusals; verify source hashes; never overwrite conflicting settled child results",
    "json_validation": "reject duplicate object keys and nonstandard JSON constants; never select the last of conflicting label values",
    "output_budget": "command-wide output-token admission budget across roles, retries and recovery; reserve each request maximum under the dispatch lock, reconcile reported completion tokens, retain maximum estimates for unknown usage; never alter frozen response limits to fit remaining budget; excludes input cost and is not a billing guarantee",
    "deepseek_content_refusal": "recognize only the explicit HTTP 400 invalid_request_error / Content Exists Risk envelope; retain unavailable targets without retry or splitting; reuse independent diagnostic evidence only when linked to the same latest attempt and exact request hash; retain provenance without error prose",
}


def policy_record(root):
    digest = stable_hash(json.dumps(POLICY, sort_keys=True, ensure_ascii=False))
    path = root / "execution_policies" / f"{digest}.json"
    if path.exists() and json.loads(path.read_text()) != POLICY:
        raise ValueError("执行恢复策略记录已改变，请恢复记录")
    if not path.exists():
        write_json(path, POLICY)
    return {"version": POLICY["version"], "policy_hash": digest}


def policy_records(root):
    records = {}
    for path in sorted((root / "execution_policies").glob("*.json")):
        value = json.loads(path.read_text())
        if path.stem != stable_hash(json.dumps(value, sort_keys=True, ensure_ascii=False)):
            raise ValueError("执行恢复策略文件哈希不一致")
        records[path.stem] = value
    return records


class Records(dict):
    """The mapping remains backward-compatible; unavailability is explicit metadata, not a model label."""
    def __init__(self):
        super().__init__()
        self.unavailable = {}


def unavailable(records):
    return getattr(records, "unavailable", {})


def placeholder(kind, key, reason):
    fields = data.VIDEO_FIELDS if kind == "videos" else COMMENT_LABELS
    return {"id": key, "labels": dict.fromkeys(fields), "evidence": dict.fromkeys(fields, ""),
            "note": f"LOCAL_MEASUREMENT_UNAVAILABLE:{reason}; no model annotation supplied"}


def unpack(payload, row, validator):
    """Validate every real item and every unavailable ID; require complete, disjoint accounting."""
    if row["status"] == "complete":
        return validator(payload, {"items": json.loads(row["labels"])}), {}
    if row["status"] != "unavailable":
        raise ValueError("任务尚未完成或标记不可编码")
    response = json.loads(row["response"])
    if not isinstance(response, dict) or response.get("source") != "local_unavailable":
        raise ValueError("不可编码任务缺少本机来源记录")
    reasons = response.get("unavailable_targets")
    if (not isinstance(reasons, dict) or not reasons or
            any(reason not in {"provider_refusal", quarantine.REASON} for reason in reasons.values())):
        raise ValueError("不可编码任务的原因记录不合法")
    targets = payload["target_ids"]
    if not set(reasons) <= set(targets):
        raise ValueError("不可编码记录包含错误目标 ID")
    proofs = response.get("quarantine_evidence", {})
    isolated = {key for key, reason in reasons.items() if reason == quarantine.REASON}
    if not isinstance(proofs, dict) or set(proofs) != isolated:
        raise ValueError("编码失败隔离的逐项证据不完整")
    for key in isolated:
        quarantine.validate_proof(payload, key, proofs[key])
    resolved = [key for key in targets if key not in reasons]
    part = data.target_subset(payload, resolved)
    rows = validator(part, {"items": json.loads(row["labels"])})
    return rows, reasons


def add_record(records, job, row, validator):
    payload = json.loads(job["payload"])
    values, missing = unpack(payload, row, validator)
    by_id = {r["id"]: r for r in values}
    records[job["job_id"]] = [by_id[key] if key in by_id else placeholder(job["kind"], key, missing[key])
                               for key in payload["target_ids"]]
    records.unavailable.update({(job["kind"], key): reason for key, reason in missing.items()})
