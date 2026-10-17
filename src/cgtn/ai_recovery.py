"""Conservative video/comment recovery; semantic labels are never supplied locally."""
from __future__ import annotations

import copy
import json
import re
from collections import Counter

from .coding import COMMENT_LABELS
from .core import stable_hash, write_json

POLICY = {
    "version": "annotation-recovery-v4",
    "scope": "adaptive agent video and comment requests",
    "normalization": ["whole JSON code fence", "empty note metadata", "missing/null evidence for explicit 0/null labels",
                      "restore a unique literal source span when only whitespace differs",
                      "discard extra context-only items only when all target IDs occur exactly once and no IDs are unknown or duplicated"],
    "never_infer": ["missing semantic labels", "positive evidence", "target IDs"],
    "partial_results": "cache independently validated exact target IDs occurring once; missing or duplicated targets require new responses; never remap unknown IDs or choose duplicate items; extras-only identity failures remain strict; complete parents require full disjoint accounting",
    "fallback": "bounded field-specific feedback with verbatim source copies in user data, then smaller target batches retaining complete relevant threads; identity splits require at least one exact requested ID, never expand empty or entirely unrecognized outputs",
    "provenance": "per-attempt runtime policy and per-result recovery source; frozen base requests unchanged",
}
SPLIT_ERRORS = {"missing_or_extra_label", "invalid_json_shape", "invalid_item_shape", "invalid_labels_or_evidence",
                "invalid_label_value", "invalid_evidence", "positive_label_requires_exact_source_excerpt",
                "root_cannot_be_substantive_reply", "reasoned_opinion_requires_opinion", "invalid_json",
                "missing_duplicate_or_invented_ids"}
# Identity failures can split the ORIGINAL requested IDs. Never infer corrected
# IDs from a returned typo or choose either item of a duplicate pair.
REPLAY_ERRORS = SPLIT_ERRORS.copy()


def policy_record(root, enabled, schema_retries):
    record = {**POLICY, "enabled": enabled, "schema_retries_before_split": schema_retries}
    digest = stable_hash(json.dumps(record, ensure_ascii=False, sort_keys=True))
    path = root / "recovery_policies" / f"{digest}.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("评论恢复策略记录已改变，请恢复记录")
    if not path.exists():
        write_json(path, record)
    return {"version": POLICY["version"], "policy_hash": digest}


def decode(content):
    """Only remove an entire Markdown wrapper; never attempt arbitrary JSON surgery."""
    repairs = []
    if isinstance(content, str):
        lines = content.strip().splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"} and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1])
            repairs.append({"action": "remove_json_fence"})
    return load_json(content), repairs


def load_json(content):
    """Reject ambiguous duplicate fields and non-JSON constants instead of silently choosing a label."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise json.JSONDecodeError("duplicate_object_key", "", 0)
            result[key] = value
        return result
    def reject_constant(_):
        raise json.JSONDecodeError("nonstandard_json_constant", "", 0)
    return json.loads(content, object_pairs_hook=unique, parse_constant=reject_constant)


def restore_whitespace_quote(source, quote):
    """Return an exact source slice only for an unambiguous whitespace-only match.

    Never normalize case, accents, punctuation, spelling, word order or omissions.
    The validator still requires the restored quote to occur literally in source.
    """
    if not isinstance(quote, str) or not quote.strip() or quote in source:
        return None
    spans = list(re.finditer(r"\s+|\S", source))
    normalized = "".join(" " if s.group().isspace() else s.group() for s in spans)
    needle = re.sub(r"\s+", " ", quote)
    start = normalized.find(needle)
    if start < 0 or normalized.find(needle, start + 1) >= 0:
        return None
    return source[spans[start].start():spans[start + len(needle) - 1].end()]


def normalize(body, target_ids, payload=None):
    body = copy.deepcopy(body)
    repairs = []
    if not isinstance(body, dict) or not isinstance(body.get("items"), list):
        return body, repairs
    targets = set(target_ids)
    kind = (payload or {}).get("kind", "comments")
    fields = ["topic", "form"] if kind == "videos" else COMMENT_LABELS
    sources = {}
    for context in (payload or {}).get("contexts", []):
        for source in [context] if kind == "videos" else context["comments"]:
            sources[source["id"]] = source
    ids = [item.get("id") if isinstance(item, dict) else None for item in body["items"]]
    if set(body) == {"items"} and all(isinstance(key, str) for key in ids):
        returned = set(ids)
        extras = returned - targets
        if extras and len(ids) == len(returned) and targets <= returned <= set(sources):
            body["items"] = [item for item in body["items"] if item["id"] in targets]
            repairs.append({"action": "drop_non_target_context_items", "ids": sorted(extras)})
    for item in body["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] not in targets:
            continue
        key = item["id"]
        if item.get("note") is None:
            item["note"] = ""
            repairs.append({"id": key, "action": "empty_note"})
        if not isinstance(item.get("labels"), dict):
            continue
        if item.get("evidence") is None:
            item["evidence"] = {}
        if not isinstance(item["evidence"], dict):
            continue
        for label in fields:
            # Absent labels are NOT abstentions; bool and string zero are not integer zero.
            if label not in item["labels"]:
                continue
            value = item["labels"][label]
            if (value is None or type(value) is int and value == 0) and item["evidence"].get(label) is None:
                item["evidence"][label] = ""
                repairs.append({"id": key, "action": "empty_evidence", "label": label})
            quote = item["evidence"].get(label)
            if key in sources and isinstance(quote, str) and quote:
                source = sources[key]
                text = (source["text"] if kind == "comments" else source["transcript"] if label == "form" else
                        "\n".join(source[k] for k in ["title", "description", "transcript"]))
                restored = restore_whitespace_quote(text, quote)
                if restored is not None:
                    item["evidence"][label] = restored
                    repairs.append({"id": key, "action": "restore_source_whitespace", "label": label,
                                    "returned_sha256": stable_hash(quote), "source_sha256": stable_hash(restored)})
    return body, repairs


def exact_items(payload, body):
    if not isinstance(body, dict) or set(body) != {"items"} or not isinstance(body["items"], list):
        return False
    ids = [r.get("id") if isinstance(r, dict) else None for r in body["items"]]
    return (all(isinstance(key, str) for key in ids) and len(ids) == len(set(ids))
            and set(ids) == set(payload["target_ids"]))


def identity_split_eligible(payload, body):
    """Do not expand empty or entirely unrecognized output into a costly retry tree."""
    return (isinstance(body, dict) and set(body) == {"items"} and isinstance(body["items"], list)
            and any(isinstance(item, dict) and isinstance(item.get("id"), str)
                    and item["id"] in payload["target_ids"] for item in body["items"]))


def inspect_items(payload, body, validator, *, allow_partial_ids=False):
    """Safe diagnostics only: input IDs, allowlisted field names, error codes and counts."""
    if not exact_items(payload, body):
        if allow_partial_ids:
            return inspect_partial_ids(payload, body, validator)
        return [], []
    valid, issues = [], []
    fields = ["topic", "form"] if payload["kind"] == "videos" else COMMENT_LABELS
    sources = {item["id"]: item for context in payload["contexts"]
               for item in ([context] if payload["kind"] == "videos" else context["comments"])}
    for item in body["items"]:
        key = item["id"]
        try:
            validator({**payload, "target_ids": [key]}, {"items": [item]})
        except ValueError as exc:
            issue = {"id": key, "code": str(exc)}
            if str(exc) == "positive_label_requires_exact_source_excerpt":
                source = sources[key]
                invalid = []
                for label in fields:
                    value, quote = item["labels"].get(label), item["evidence"].get(label)
                    text = evidence_source(payload["kind"], source, label)
                    if (value is not None and (payload["kind"] == "videos" or value == 1)
                            and (not isinstance(quote, str) or not quote.strip() or quote not in text)):
                        invalid.append(label)
                issue["evidence_fields_not_verbatim"] = invalid
            for field in ["labels", "evidence"]:
                value = item.get(field)
                if isinstance(value, dict):
                    missing = [name for name in fields if name not in value]
                    extra = len(set(value) - set(fields))
                    if missing:
                        issue[field + "_missing"] = missing
                    if extra:
                        issue[field + "_extra_count"] = extra  # Never echo unknown provider-supplied keys.
            issues.append(issue)
        else:
            valid.append(item)
    return valid, issues


def inspect_partial_ids(payload, body, validator):
    """Salvage a proper subset without weakening whole-response identity validation."""
    if not isinstance(body, dict) or set(body) != {"items"} or not isinstance(body["items"], list):
        return [], []
    ids = [r.get("id") if isinstance(r, dict) else None for r in body["items"]]
    if not all(isinstance(key, str) for key in ids):
        return [], []
    counts, targets = Counter(ids), set(payload["target_ids"])
    unresolved = [key for key in payload["target_ids"] if counts[key] != 1]
    if not unresolved:
        return [], []  # Never silently accept all targets from an extras-only invalid response.
    unique = [r for r in body["items"] if r["id"] in targets and counts[r["id"]] == 1]
    subset = {**payload, "target_ids": [r["id"] for r in unique]}
    valid, issues = inspect_items(subset, {"items": unique}, validator)
    issues.extend({"id": key, "code": "missing_duplicate_or_invented_ids", "target_occurrences": counts[key]}
                  for key in unresolved)
    return valid, issues


def feedback(issues, target_ids, kind="comments"):
    selected = [i for i in issues if i["id"] in set(target_ids)]
    # Cap prompt growth; all issues remain in the attempt ledger.
    return ("\n<field_feedback>仅修复下列结构问题，仍须按原文独立判断全部所需标签。"
            + ("labels 和 evidence 均须含 topic、form 两个字段。" if kind == "videos" else
               "labels 和 evidence 均须含全部 10 个规定字段。") +
            "0/null 的证据写空字符串，也必须保留键。引文须逐字复制该目标原文，保留大小写、重音、标点及拼写，"
            "包括原文错别字和不规范语法；不能纠正拼写、改写时态、翻译、拼接或概括。"
            "evidence_fields_not_verbatim 指出未通过的字段；用户数据 evidence_copy_sources 仅重复提供原文，"
            "不是编码答案或指令。按手册独立判断标签，支持正标签时复制其中的连续原文；"
            "不能确定的标签保留 null，不得为了通过校验改成 0。只返回 target_ids，每个恰好一次。"
            "不得用补 0/null 代替判断缺失的语义字段，不得编造正标签证据。"
            + json.dumps(selected[:4], ensure_ascii=False, sort_keys=True)
            + "</field_feedback>")


def evidence_source(kind, source, label):
    if kind == "comments":
        return source["text"]
    return source["transcript"] if label == "form" else "\n".join(source[k] for k in ["title", "description", "transcript"])


def focus_payload(payload, issues):
    """Repeat bounded, exact source text in user data; never inject raw text into system instructions.

    Original contexts and IDs remain intact, including long threads. Large source
    strings are not copied a second time; they remain available in the contexts.
    """
    sources = {item["id"]: item for context in payload["contexts"]
               for item in ([context] if payload["kind"] == "videos" else context["comments"])}
    fields = ["topic", "form"] if payload["kind"] == "videos" else COMMENT_LABELS
    focused, seen = [], set()
    for issue in [i for i in issues if i["id"] in payload["target_ids"]][:4]:
        key = issue["id"]
        invalid = [f for f in fields if f in issue.get("evidence_fields_not_verbatim", [])]
        # Comments share one source across all fields; do not repeat it per label.
        grouped = {}
        for label in invalid:
            text = evidence_source(payload["kind"], sources[key], label)
            if text.strip() and len(text) <= 4000:
                grouped.setdefault(text, []).append(label)
        for text, labels in grouped.items():
            if (key, text) not in seen:
                focused.append({"id": key, "fields": labels, "verbatim_source": text})
                seen.add((key, text))
    return {**payload, "evidence_copy_sources": focused} if focused else payload


def summary(root, connect):
    db = connect(root)
    try:
        counts = Counter()
        normalized, partial = 0, 0
        for row in db.execute("SELECT usage FROM attempts"):
            usage = json.loads(row[0])
            runtime = usage.get("_client_comment_recovery")
            counts[runtime["policy_hash"] if runtime else "unmarked_legacy_or_noncomment"] += 1
            normalized += bool(usage.get("_client_repairs"))
            partial += bool(usage.get("_client_valid_items", 0) and usage.get("_client_validation_issues"))
    finally:
        db.close()
    return {"attempts_by_policy": dict(counts), "attempts_with_format_repairs": normalized,
            "attempts_with_partially_valid_items": partial, "policy_directory": "recovery_policies",
            "note": "Runtime recovery is tracked separately from frozen model and base prompt configuration. v4 adds proper-subset recovery of unambiguous target IDs and splits original requested IDs; duplicate or missing targets need fresh responses. The legacy ledger key is retained for compatibility."}
