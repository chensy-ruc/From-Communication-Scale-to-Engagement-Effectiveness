"""Bounded measurement missingness, never a repair or a substitute model annotation."""
from __future__ import annotations

import json

from . import ai_data as data
from . import ai_recovery as recovery
from .coding import COMMENT_LABELS
from .core import now, stable_hash

ERROR = "positive_label_requires_exact_source_excerpt"
REASON = "evidence_validation_exhausted"
MIN_ATTEMPTS = 2


def payload_hash(payload):
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def candidate(db, version, job, validator):
    """Read-only: prove two consecutive failed single-target responses, including corrective feedback."""
    payload = json.loads(job["payload"])
    if len(payload["target_ids"]) != 1 or payload["kind"] != "comments":
        return None
    key = payload["target_ids"][0]
    rows = list(db.execute(
        "SELECT a.id,a.status,a.usage,o.content,o.content_sha256,o.redacted,j.payload "
        "FROM attempts a LEFT JOIN attempt_outputs o ON o.attempt_id=a.id "
        "JOIN job_registry j ON j.version=a.version AND j.job_id=a.job_id "
        "WHERE a.version=? AND a.job_id=? ORDER BY a.id DESC LIMIT ?",
        (version, job["job_id"], MIN_ATTEMPTS)))
    if len(rows) != MIN_ATTEMPTS:
        return None
    proof = []
    for row in rows:
        if json.loads(row["payload"]) != payload:
            raise ValueError("隔离核查发现请求输入改变，拒绝处理")
        usage = json.loads(row["usage"])
        if (row["status"] != ERROR or usage.get("_client_http_status") != 200
                or usage.get("_client_finish_reason") != "stop" or row["content"] is None or row["redacted"]):
            return None
        if stable_hash(row["content"]) != row["content_sha256"]:
            raise ValueError("隔离核查发现归档正文哈希不一致，拒绝处理")
        try:
            body, _ = recovery.decode(row["content"])
            # Ambiguous, missing, extra or invented IDs are not eligible for this policy.
            if not recovery.exact_items(payload, body):
                return None
            body, _ = recovery.normalize(body, payload["target_ids"], payload)
            valid, issues = recovery.inspect_items(payload, body, validator)
        except (ValueError, TypeError, KeyError):
            return None
        if (valid or len(issues) != 1 or issues[0]["code"] != ERROR
                or not issues[0].get("evidence_fields_not_verbatim")):
            return None
        feedback = usage.get("_client_feedback", [])
        feedback_fields = sorted({field for item in feedback if item.get("id") == key and item.get("code") == ERROR
                                  for field in item.get("evidence_fields_not_verbatim", []) if field in COMMENT_LABELS})
        proof.append({"attempt_id": row["id"], "content_sha256": row["content_sha256"],
                      "validation_issue": issues[0], "feedback_fields": feedback_fields,
                      "corrective_feedback_sent": False})
    # Feedback is composed BEFORE the next response exists: it must match the
    # preceding failure, not whichever other field fails in the next response.
    proof[0]["corrective_feedback_sent"] = set(proof[1]["validation_issue"]["evidence_fields_not_verbatim"]) <= set(proof[0]["feedback_fields"])
    if not proof[0]["corrective_feedback_sent"]:
        return None
    return {"reason": REASON, "kind": "comments", "id": key, "version": version,
            "job_id": job["job_id"], "payload_sha256": payload_hash(payload), "attempts": proof,
            "feedback_basis": "previous_failed_attempt"}


def validate_proof(payload, key, proof):
    """Also used on assembled parents: verify that evidence refers to this exact single-target input."""
    if (not isinstance(proof, dict) or proof.get("reason") != REASON or proof.get("kind") != "comments"
            or payload["kind"] != "comments" or proof.get("id") != key
            or proof.get("payload_sha256") != payload_hash(data.target_subset(payload, [key]))):
        raise ValueError("编码失败隔离缺少匹配的输入溯源")
    attempts = proof.get("attempts")
    if (not isinstance(attempts, list) or len(attempts) != MIN_ATTEMPTS
            or any(not isinstance(a, dict) or type(a.get("attempt_id")) is not int
                   or not isinstance(a.get("content_sha256"), str) or len(a["content_sha256"]) != 64
                   or a.get("validation_issue", {}).get("id") != key
                   or a.get("validation_issue", {}).get("code") != ERROR
                   or not a.get("validation_issue", {}).get("evidence_fields_not_verbatim") for a in attempts)
            or attempts[0]["attempt_id"] <= attempts[1]["attempt_id"]
            or attempts[0].get("corrective_feedback_sent") is not True):
        raise ValueError("编码失败隔离缺少两次请求失败及字段反馈证据")
    # Older settled v8 proofs remain valid; new proofs also store and verify the
    # actual feedback fields against the independently revalidated prior attempt.
    basis = proof.get("feedback_basis")
    if basis is not None and (basis != "previous_failed_attempt"
            or not set(attempts[1]["validation_issue"]["evidence_fields_not_verbatim"]) <= set(attempts[0].get("feedback_fields", []))):
        raise ValueError("编码失败隔离的反馈未覆盖前一次请求的错误字段")


def admit(db, proof, limit, execution_id):
    """Called inside BEGIN IMMEDIATE: a durable experiment-wide cap across roles and restarts."""
    identity = (proof["version"], proof["kind"], proof["id"])
    existing = db.execute("SELECT proof FROM evidence_quarantine WHERE version=? AND kind=? AND target_id=?",
                          identity).fetchone()
    if existing:
        prior = json.loads(existing[0])
        if prior["payload_sha256"] != proof["payload_sha256"]:
            raise ValueError("已隔离条目的输入发生变化，拒绝覆盖")
        return prior
    if db.execute("SELECT COUNT(*) FROM evidence_quarantine").fetchone()[0] >= limit:
        return None
    db.execute("INSERT INTO evidence_quarantine VALUES (?,?,?,?,?,?)", (
        *identity, json.dumps(proof, ensure_ascii=False), execution_id, now()))
    return proof
