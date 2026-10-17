from __future__ import annotations

import itertools
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .api import API, APIError, StopCollection
from .core import load_sample, now, stable_hash, write_csv, write_json


def target_year(p, stamp):
    if not stamp:
        return False
    return datetime.fromisoformat(stamp).astimezone(
        ZoneInfo(p.cfg["study"]["timezone"])).year == p.cfg["study"]["year"]


def scan_inventory(p, api, uploads_id):
    """Persist each page cursor after its rows; older runs can replay their page cache."""
    cursor = p.store.meta("inventory_cursor") or {}
    token = cursor.get("next_page_token")
    seen = set(cursor.get("seen_tokens", []))
    pages = cursor.get("pages", 0)
    restarted = False
    while not cursor.get("done"):
        try:
            data, _ = api.get("playlistItems", part="snippet,contentDetails", maxResults=50,
                              playlistId=uploads_id, pageToken=token)
        except APIError as e:
            if e.reason != "invalidPageToken" or restarted:
                raise
            # Keep old responses/records as provenance, restart only this expired chain once.
            p.store.archive_pages([("playlistItems", "playlistId", {uploads_id})], "invalidPageToken")
            cursor = {}
            p.store.set_meta("inventory_cursor", cursor)
            p.store.set_meta("inventory_cursor_restarts", (p.store.meta("inventory_cursor_restarts") or 0) + 1)
            token, seen, pages, restarted = None, set(), 0, True
            print("上传目录分页凭据失效；已归档旧响应，重新遍历目录并按视频 ID 合并。", flush=True)
            continue
        rows = []
        for item in data.get("items", []):
            d, s = item.get("contentDetails", {}), item.get("snippet", {})
            vid = d.get("videoId") or s.get("resourceId", {}).get("videoId")
            if vid:
                rows.append((vid, d.get("videoPublishedAt"), s.get("title")))
        p.store.many("INSERT OR REPLACE INTO inventory VALUES (?,?,?)", rows)
        pages += 1
        token = data.get("nextPageToken")
        if token and token in seen:
            raise ValueError("上传目录分页 token 循环；不能宣称目录完整")
        if token:
            seen.add(token)
        cursor = {"next_page_token": token, "pages": pages, "seen_tokens": sorted(seen), "done": not token}
        p.store.set_meta("inventory_cursor", cursor)
        p.store.set_meta("inventory_pages", pages)
        p.store.set_meta("inventory_reported_total", data.get("pageInfo", {}).get("totalResults"))
        if pages % 100 == 0:
            print(f"已遍历上传目录 {pages} 页；目录游标已保存", flush=True)
    p.store.set_meta("inventory_complete", True)


def collect_videos(p, hours=0):
    if p.store.meta("videos_complete"):
        print("本快照的视频目录与元数据已完成，直接复用。", flush=True)
        return
    api = API(p, hours)
    channel, _ = api.get("channels", part="snippet,contentDetails,statistics",
                         id=p.cfg["study"]["channel_id"])
    if len(channel.get("items", [])) != 1:
        raise ValueError("频道未找到，请核对 config.toml 的 channel_id")
    ch = channel["items"][0]
    p.store.set_meta("channel", ch)
    print(f"频道：{ch['snippet']['title']} ({ch['id']})", flush=True)
    if not p.store.meta("inventory_complete"):
        scan_inventory(p, api, ch["contentDetails"]["relatedPlaylists"]["uploads"])
    inventory = p.store.rows("SELECT * FROM inventory ORDER BY video_id")
    ids = [r["video_id"] for r in inventory if not r["published_hint"] or target_year(p, r["published_hint"])]
    done = {r["video_id"] for r in p.store.rows("SELECT video_id FROM videos")}
    pending = [v for v in ids if v not in done]
    for start in range(0, len(pending), 50):
        batch = pending[start:start + 50]
        data, stamp = api.get("videos", part="snippet,contentDetails,statistics,status,liveStreamingDetails",
                              id=",".join(batch), maxResults=50)
        items = {v["id"]: v for v in data.get("items", [])}
        p.store.many("INSERT OR REPLACE INTO videos VALUES (?,?,?,?)",
                     [(v, stamp, int(v in items), json.dumps(items.get(v, {}), ensure_ascii=False)) for v in batch])
    p.store.set_meta("videos_complete", True)
    p.store.set_meta("candidate_count", len(ids))
    p.store.set_meta("videos_finished_at", now())
    print(f"上传目录 {len(inventory)} 个 ID；{p.cfg['study']['year']}/日期未知候选 {len(ids)} 个。下一步 clean。")


def select_sample(p, n, strata):
    if not p.store.meta("videos_complete"):
        raise ValueError("请先完成 collect-videos 和 clean")
    path = p.root / "sample.csv"
    if path.exists():
        raise ValueError("sample.csv 已存在，抽样框已冻结。续传无需重新抽样；新设计请另建 run_id")
    df = p.store.frame("clean_videos")
    if df.empty:
        raise ValueError("没有符合年份的公开视频")
    if any(c not in df for c in strata):
        raise ValueError("可用分层列包含 month,duration_band,live_status；主题须先自行完成全框内容编码")
    if n < 0:
        raise ValueError("n=0 表示普查，否则必须为正数")
    df["stratum"] = df[strata].fillna("unknown").astype(str).agg("|".join, axis=1)
    groups = [(s, g.sort_values("video_id")) for s, g in df.groupby("stratum", sort=True)]
    rng = np.random.default_rng(p.cfg["study"]["seed"])
    target = len(df) if n == 0 else min(n, len(df))
    if target < len(groups):
        raise ValueError(f"样本量至少 {len(groups)} 才能覆盖每个非空层；建议 n=600")
    # Minimum one per non-empty stratum, then proportional allocation with capacity limits.
    counts = np.array([len(g) for _, g in groups])
    allocated = np.ones(len(groups), dtype=int)
    while allocated.sum() < target:
        score = counts * target / counts.sum() - allocated
        score[allocated >= counts] = -np.inf
        allocated[np.argmax(score)] += 1
    parts = []
    for (s, g), k in zip(groups, allocated):
        chosen = g.iloc[rng.choice(len(g), k, replace=False)].copy()
        chosen["stratum_population_n"], chosen["stratum_sample_n"] = len(g), k
        chosen["video_probability"] = k / len(g)
        chosen["video_weight"] = len(g) / k
        parts.append(chosen)
    result = pd.concat(parts).sample(frac=1, random_state=p.cfg["study"]["seed"]).reset_index(drop=True)
    cols = ["video_id", "stratum", "stratum_population_n", "stratum_sample_n", "video_probability", "video_weight"]
    write_csv(result[cols], path)
    write_json(p.root / "sample_design.json", {"n": len(result), "N": len(df), "strata": strata,
               "seed": p.cfg["study"]["seed"], "created_at": now(), "sample_hash": stable_hash(path.read_bytes()),
               "mode": "census" if target == len(df) else "stratified_srs"})
    print(f"抽样框已冻结：{len(result)}/{len(df)} 个视频，{len(groups)} 层。")


def plan(p, all_videos=False):
    v = p.store.frame("clean_videos")
    sample = p.root / "sample.csv"
    scope = p.store.meta("comment_target_frame") or {}
    if not all_videos and scope.get("scope") == "all":
        v = v[v.video_id.isin(scope["ids"])]
    elif not all_videos and sample.exists():
        v = v.merge(load_sample(p)[["video_id"]], on="video_id")
    c = pd.to_numeric(v["comment_count"], errors="coerce")
    # Counts include replies; reply pagination cannot be precisely predicted before collection.
    lower = int(np.maximum(1, np.ceil(c.fillna(0) / 100)).sum())
    conservative = int(len(v) + np.ceil(c.fillna(0) / 100).sum() + c.fillna(0).sum())
    out = {"videos": len(v), "known_comment_count_sum": int(c.sum()), "unknown_comment_count_videos": int(c.isna().sum()),
           "optimistic_comment_requests": lower, "conservative_comment_requests": conservative,
           "configured_daily_budget": p.cfg["api"]["daily_budget"],
           "optimistic_rate_limit_hours": round(lower / p.cfg["api"]["requests_per_second"] / 3600, 3),
           "note": "仅为计划值。计数变化、回复线程、重试及目录消耗会影响实际配额；不是完成保证。"}
    write_json(p.reports / "collection_plan.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def save_comment(p, item, vid, root_id, reply, stamp, total_reply_count=None):
    item = dict(item)
    if total_reply_count is not None:
        item["total_reply_count"] = total_reply_count
    p.store.execute("INSERT OR REPLACE INTO comments VALUES (?,?,?,?,?,?)",
                    (item["id"], vid, root_id, int(reply), stamp, json.dumps(item, ensure_ascii=False)))


def _collect_one_pages(p, api, vid):
    def state(status, reason=""):
        p.store.execute("INSERT OR REPLACE INTO collection VALUES (?,?,?,?)", (vid, status, reason, now()))
        return status
    state("running")
    token, seen, gaps = None, set(), []
    try:
        while True:
            api.check()
            data, stamp = api.get("commentThreads", part="snippet,replies", videoId=vid,
                                  maxResults=100, order="time", textFormat="plainText", pageToken=token)
            for thread in data.get("items", []):
                s = thread["snippet"]
                root = s["topLevelComment"]
                root_id = root["id"]
                expected = s.get("totalReplyCount", 0)
                save_comment(p, root, vid, root_id, False, stamp, expected)
                inline = thread.get("replies", {}).get("comments", [])
                for reply in inline:
                    save_comment(p, reply, vid, root_id, True, stamp)
                if len({r["id"] for r in inline}) < expected:
                    rt, rseen, returned = None, set(), set()
                    while True:
                        api.check()
                        try:
                            replies, rstamp = api.get("comments", part="snippet", parentId=root_id,
                                                     maxResults=100, textFormat="plainText", pageToken=rt)
                        except APIError as e:
                            if e.reason in {"commentNotFound", "forbidden"}:
                                gaps.append(f"{root_id}:{e.reason}")
                                break
                            raise
                        for reply in replies.get("items", []):
                            returned.add(reply["id"])
                            save_comment(p, reply, vid, root_id, True, rstamp)
                        rt = replies.get("nextPageToken")
                        if not rt:
                            if len(returned) < expected:
                                gaps.append(f"{root_id}:reply_count_changed")
                            break
                        if rt in rseen:
                            raise APIError("reply_pagination_loop")
                        rseen.add(rt)
            token = data.get("nextPageToken")
            if not token:
                return state("reply_gap" if gaps else "complete", ";".join(gaps))
            if token in seen:
                raise APIError("thread_pagination_loop")
            seen.add(token)
    except StopCollection as e:
        state("partial", str(e))
        raise
    except APIError as e:
        if e.reason in {"commentsDisabled", "videoNotFound", "forbidden"}:
            return state(e.reason, str(e))
        state("error", str(e))
        if e.reason == "invalidPageToken":
            raise
        if e.status == 400 and e.reason == "processingFailure":
            # A failing video must remain resumable without stopping unrelated videos.
            print(f"{vid}：评论 processingFailure 重试仍失败，已保存为 error 待续传；继续其他视频。", flush=True)
            return "error"
        # Configuration/auth errors should stop the run, not generate thousands of identical failures.
        if e.status in {400, 401, 403} and e.reason != "invalidPageToken":
            api.stop.set()
            raise
        return "error"


def archive_video_pages(p, vid, reason):
    roots = {r["root_id"] for r in p.store.rows("SELECT DISTINCT root_id FROM comments WHERE video_id=?", (vid,))}
    p.store.archive_pages([("commentThreads", "videoId", {vid}), ("comments", "parentId", roots)], reason)


def collect_one(p, api, vid):
    prior = p.store.rows("SELECT status FROM collection WHERE video_id=?", (vid,))
    if prior and prior[0]["status"] == "reply_gap":
        # Replaying the same cached count mismatch can never recover a reply gap.
        # Refresh this video's chain once per invocation and preserve the old responses.
        archive_video_pages(p, vid, f"reply_gap_retry:{vid}")
        print(f"{vid} 存在回复缺口；已归档旧分页，重新核对该视频可见线程和回复。", flush=True)
    for attempt in range(2):
        try:
            return _collect_one_pages(p, api, vid)
        except APIError as e:
            if e.reason != "invalidPageToken":
                raise
            if attempt:
                return "error"
            archive_video_pages(p, vid, f"invalidPageToken:{vid}")
            print(f"{vid} 的评论分页凭据失效；已保留旧响应，重取该视频分页并按评论 ID 合并。", flush=True)
    raise AssertionError("unreachable")


def comment_targets(p, all_videos=False):
    saved = p.store.meta("comment_target_frame")
    # Once expanded to the census, a later resume without --all must never shrink the target.
    if saved and saved["scope"] == "all":
        return pd.DataFrame({"video_id": saved["ids"]})
    if all_videos:
        if not p.store.meta("videos_complete"):
            raise ValueError("全量评论采集须先完成 collect-videos 和 clean")
        if p.store.meta("clean_video_source_count") != p.store.rows("SELECT COUNT(*) AS n FROM videos")[0]["n"]:
            raise ValueError("清洗视频框尚未更新到完整目录；请先运行 clean，再采集全量评论")
        selected = p.store.frame("clean_videos")[["video_id"]].sort_values("video_id")
        selected = selected.sample(frac=1, random_state=p.cfg["study"]["seed"])
    else:
        if not (p.root / "sample.csv").exists():
            raise ValueError("先运行 sample，或使用 collect-all / collect-comments --all 采集全部视频")
        selected = load_sample(p)
    if selected.empty:
        raise ValueError("目标年份没有可采集视频；请核对视频目录与清洗结果")
    p.store.set_meta("comment_target_frame", {"scope": "all" if all_videos else "sample",
        "ids": selected.video_id.tolist(), "frozen_at": now()})
    return selected


def collect_comments(p, hours=0, all_videos=False):
    sample = comment_targets(p, all_videos)
    p.store.set_meta("comment_cutoff", p.store.meta("comment_cutoff") or now())
    states = {r["video_id"]: r["status"] for r in p.store.rows("SELECT * FROM collection")}
    terminal = {"complete", "commentsDisabled", "videoNotFound", "forbidden"}
    pending = [v for v in sample.video_id if states.get(v) not in terminal]
    print(f"评论目标 {len(sample)} 个视频；已处理 {len(sample) - len(pending)}，本次待处理 {len(pending)}。", flush=True)
    if not pending:
        audit(p)
        return
    api = API(p, hours)
    workers = p.cfg["api"]["workers"]
    if workers < 1:
        raise ValueError("workers 必须至少为 1")
    stopped = None
    pool = ThreadPoolExecutor(max_workers=workers)
    jobs = {}
    remaining = iter(pending)
    try:
        for vid in itertools.islice(remaining, workers):
            jobs[pool.submit(collect_one, p, api, vid)] = vid
        done = 0
        while jobs:
            finished, _ = wait(jobs, return_when=FIRST_COMPLETED)
            for job in finished:
                jobs.pop(job)
                try:
                    status = job.result()
                except (StopCollection, APIError) as e:
                    if stopped is None or isinstance(e, APIError):
                        stopped = e
                    api.stop.set()
                else:
                    done += 1
                    if done % 25 == 0 or done == len(pending):
                        print(f"本次已处理 {done}/{len(pending)}；最近状态 {status}", flush=True)
            if not api.stop.is_set():
                for vid in itertools.islice(remaining, workers - len(jobs)):
                    jobs[pool.submit(collect_one, p, api, vid)] = vid
    finally:
        # Also runs on Ctrl+C: stop workers before waiting, never drain a full-year queued workload.
        api.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        api.client.close()
        audit(p)
    if stopped:
        raise stopped
    states = p.store.frame("collection")
    states = states[states.video_id.isin(sample.video_id)]
    if states.status.isin(["partial", "error", "running", "reply_gap"]).any():
        raise StopCollection("部分视频尚未采集完成（含回复缺口），请查看 audit.json 并重跑相同命令；"
                             "持续不可见的回复不能通过重试保证找回")


def collect_all(p):
    """One resumable entry point; the study year is unchanged and no wall-clock cap is imposed."""
    from .clean import clean
    try:
        collect_videos(p, hours=0)
    except (StopCollection, APIError, KeyboardInterrupt):
        audit(p)
        raise
    # Reuse the cleaned, frozen video frame on later comment-only resumes.
    if (not p.store.rows("SELECT name FROM sqlite_master WHERE type='table' AND name='clean_videos'")
            or p.store.rows("SELECT COUNT(*) AS n FROM videos")[0]["n"] != p.store.meta("clean_video_source_count")):
        clean(p)
    if not (p.root / "sample.csv").exists():
        select_sample(p, 0, ["month", "duration_band"])
    else:
        load_sample(p)
        print("保留原 sample.csv 和人工编码；全量采集目标单独记录，不受该研究样本限制。", flush=True)
    collect_comments(p, hours=0, all_videos=True)
    clean(p)
    print("全量采集任务已结束。请在 audit.json 核对不可访问视频及回复缺口；元数据和评论已清洗导出。")


def audit(p):
    counts = {}
    for table in ["inventory", "videos", "comments", "raw_pages", "raw_page_history"]:
        counts[table] = p.store.rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
    states = p.store.frame("collection")
    research_sample = load_sample(p) if (p.root / "sample.csv").exists() else pd.DataFrame(columns=["video_id"])
    target = p.store.meta("comment_target_frame")
    selected = pd.DataFrame({"video_id": target["ids"]}) if target else research_sample
    coverage = selected.merge(states, on="video_id", how="left")
    coverage["status"] = coverage["status"].fillna("not_started")
    observed = pd.DataFrame(p.store.rows("""SELECT video_id,COUNT(*) AS collected_total,
        SUM(CASE WHEN is_reply=0 THEN 1 ELSE 0 END) AS collected_top_level,
        SUM(CASE WHEN is_reply=1 THEN 1 ELSE 0 END) AS collected_replies
        FROM comments GROUP BY video_id"""),
        columns=["video_id", "collected_total", "collected_top_level", "collected_replies"])
    coverage = coverage.merge(observed, on="video_id", how="left")
    for col in ["collected_total", "collected_top_level", "collected_replies"]:
        coverage[col] = pd.to_numeric(coverage[col], errors="raise").fillna(0).astype(int)
    platform_counts = {}
    for row in p.store.rows("SELECT video_id,payload FROM videos WHERE available=1"):
        value = json.loads(row["payload"]).get("statistics", {}).get("commentCount")
        platform_counts[row["video_id"]] = int(value) if value is not None else np.nan
    coverage["platform_comment_count_snapshot"] = coverage.video_id.map(platform_counts)
    denominator = coverage.platform_comment_count_snapshot.where(coverage.platform_comment_count_snapshot.gt(0))
    coverage["retrieved_to_platform_count_ratio"] = (coverage.collected_total / denominator).where(coverage.status.eq("complete"))
    write_csv(coverage, p.reports / "collection_coverage.csv")
    c = p.store.rows("""SELECT MIN(fetched_at) AS first,MAX(fetched_at) AS last FROM
        (SELECT fetched_at FROM raw_pages UNION ALL SELECT fetched_at FROM raw_page_history)""")[0]
    result = {"counts": counts, "inventory_complete": bool(p.store.meta("inventory_complete")),
              "videos_complete": bool(p.store.meta("videos_complete")),
              "candidate_count": p.store.meta("candidate_count"),
              "inventory_reported_total": p.store.meta("inventory_reported_total"),
              "inventory_cursor_restarts": p.store.meta("inventory_cursor_restarts") or 0,
              "comment_scope": target["scope"] if target else "sample",
              "research_sample_videos": len(research_sample),
              "selected_videos": len(selected), "comment_status": coverage.status.value_counts().to_dict(),
              "request_time_range": c, "comment_cutoff": p.store.meta("comment_cutoff"),
              "count_ratio_note": "抓取/平台计数之比仅供核查，可能超过 1；不同时间与可见性口径使它不是严格完整率。",
              "scope": "仅采集时仍公开、可检索的内容；目录分页结束不等于历史全集无缺失。"}
    write_json(p.reports / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
