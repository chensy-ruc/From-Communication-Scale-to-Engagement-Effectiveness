from __future__ import annotations

import html
import json
import re
import unicodedata

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs

from .collect import target_year
from .core import write_csv, write_json

DetectorFactory.seed = 20250917


def clean_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", html.unescape(text or ""))).strip()


def duration_seconds(value):
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", value or "")
    return sum(float(n or 0) * scale for n, scale in zip(m.groups(), [86400, 3600, 60, 1])) if m else None


def language(text):
    if len(re.findall(r"[^\W\d_]", text)) < 20:
        return "und", None
    try:
        x = detect_langs(text)[0]
        return (x.lang, x.prob) if x.prob >= .8 else ("und", x.prob)
    except LangDetectException:
        return "und", None


def nullable_count(s, key):
    value = s.get(key)
    return int(value) if value is not None else None


VIDEO_COLUMNS = ["video_id", "title", "description", "published_at", "fetched_at", "view_count", "like_count",
                 "comment_count", "duration_seconds", "live_status", "url", "month", "duration_band",
                 "age_days", "likes_per_1000_views", "comments_per_1000_views"]
COMMENT_COLUMNS = ["comment_id", "video_id", "root_id", "is_reply", "author_hash", "is_channel_author",
                   "text_original", "text_clean", "published_at", "updated_at", "fetched_at", "like_count",
                   "total_reply_count", "language", "language_probability", "char_count", "empty_text",
                   "has_url", "repeat_by_same_author", "within_year", "within_first30", "before_cutoff",
                   "eligible_window"]


def clean(p):
    rows, excluded = [], []
    for r in p.store.rows("SELECT * FROM videos"):
        d = json.loads(r["payload"])
        if not r["available"]:
            excluded.append({"video_id": r["video_id"], "reason": "metadata_unavailable"})
            continue
        s, st, cd = d["snippet"], d.get("statistics", {}), d.get("contentDetails", {})
        if not target_year(p, s["publishedAt"]):
            excluded.append({"video_id": r["video_id"], "reason": "outside_year"})
            continue
        if d.get("status", {}).get("privacyStatus", "public") != "public":
            excluded.append({"video_id": r["video_id"], "reason": "not_public"})
            continue
        pub = pd.Timestamp(s["publishedAt"])
        dur = duration_seconds(cd.get("duration"))
        live = "recorded_live" if d.get("liveStreamingDetails", {}).get("actualStartTime") else s.get("liveBroadcastContent", "none")
        row = {"video_id": r["video_id"], "title": clean_text(s["title"]), "description": s.get("description", ""),
               "published_at": s["publishedAt"], "fetched_at": r["fetched_at"],
               "view_count": nullable_count(st, "viewCount"), "like_count": nullable_count(st, "likeCount"),
               "comment_count": nullable_count(st, "commentCount"), "duration_seconds": dur,
               "live_status": live, "url": f"https://www.youtube.com/watch?v={r['video_id']}",
               "month": pub.tz_convert(p.cfg["study"]["timezone"]).strftime("%Y-%m"),
               "duration_band": "unknown" if dur is None else "0-180s" if dur <= 180 else "181-600s" if dur <= 600 else "601s+",
               "age_days": max((pd.Timestamp(r["fetched_at"]) - pub).total_seconds() / 86400, 0)}
        for name, col in [("likes", "like_count"), ("comments", "comment_count")]:
            row[f"{name}_per_1000_views"] = 1000 * row[col] / row["view_count"] if row["view_count"] and row[col] is not None else None
        rows.append(row)
    videos = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
    p.store.save_frame("clean_videos", videos)
    p.store.set_meta("clean_video_source_count", p.store.rows("SELECT COUNT(*) AS n FROM videos")[0]["n"])
    write_csv(videos, p.root / "processed/videos.csv")
    write_csv(pd.DataFrame(excluded, columns=["video_id", "reason"]), p.reports / "excluded_videos.csv")
    dates = {r.video_id: pd.Timestamp(r.published_at) for r in videos.itertuples()}
    cutoff = pd.Timestamp(p.store.meta("comment_cutoff") or p.store.meta("created_at"))
    channel_hash = p.author_hash(p.cfg["study"]["channel_id"])
    rows = []
    for r in p.store.rows("SELECT * FROM comments"):
        if r["video_id"] not in dates:
            continue
        d = json.loads(r["payload"])
        s = d["snippet"]
        text = s.get("textOriginal", s.get("textDisplay", ""))
        text_clean = clean_text(text)
        lang, prob = language(text_clean)
        pub = pd.Timestamp(s["publishedAt"])
        delta = (pub - dates[r["video_id"]]).total_seconds() / 86400
        wy, w30, before = target_year(p, s["publishedAt"]), 0 <= delta < 30, pub <= cutoff
        window = p.cfg["study"]["comment_window"]
        eligible = before and (window == "all" or (window == "year" and wy) or (window == "first30" and w30))
        rows.append({"comment_id": r["comment_id"], "video_id": r["video_id"], "root_id": r["root_id"],
                     "is_reply": r["is_reply"], "author_hash": s.get("author_hash"),
                     "is_channel_author": int(s.get("author_hash") == channel_hash),
                     "text_original": text, "text_clean": text_clean, "published_at": s["publishedAt"],
                     "updated_at": s.get("updatedAt"), "fetched_at": r["fetched_at"],
                     "like_count": s.get("likeCount"), "total_reply_count": d.get("total_reply_count"),
                     "language": lang, "language_probability": prob, "char_count": len(text_clean),
                     "empty_text": int(not text_clean), "has_url": int(bool(re.search(r"https?://|www\.", text_clean))),
                     "within_year": int(wy), "within_first30": int(w30), "before_cutoff": int(before),
                     "eligible_window": int(eligible)})
    comments = pd.DataFrame(rows)
    if not comments.empty:
        comments["repeat_by_same_author"] = (comments.author_hash.notna() &
            comments.duplicated(["video_id", "author_hash", "text_clean"], keep=False)).astype(int)
        comments = comments.reindex(columns=COMMENT_COLUMNS)
    else:
        comments = pd.DataFrame(columns=COMMENT_COLUMNS)
    p.store.save_frame("clean_comments", comments)
    write_csv(comments, p.root / "processed/comments.csv")
    report = {"videos": len(videos), "comments": len(comments), "video_exclusions": len(excluded),
              "window": p.cfg["study"]["comment_window"], "eligible_comments": int(comments.eligible_window.sum()),
              "text_rule": "按 ID 去重；保留情绪、表情、短评、非西语和不同作者的相同文本；仅添加质量标记。",
              "missing_statistics": {c: int(videos[c].isna().sum()) for c in ["view_count", "like_count", "comment_count"]},
              "duration_note": "时长分层不是 Shorts 分类；平台形态由人工复核。"}
    write_json(p.reports / "cleaning_report.json", report)
    print(f"已清洗：{len(videos)} 个视频，{len(comments)} 条评论。")
