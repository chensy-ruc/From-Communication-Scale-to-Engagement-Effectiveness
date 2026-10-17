from __future__ import annotations

import argparse
import os
import platform
import shutil
import sqlite3
import sys

from .api import API, APIError, StopCollection
from .core import Project, write_json


def parser():
    p = argparse.ArgumentParser(description="CGTN Español 2025：可续传采集、全 AI/人工编码与可复现统计")
    p.add_argument("--config", default="config.toml", help="配置文件路径")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="初始化 SQLite 与工作目录，不联网")
    doctor = sub.add_parser("doctor", help="检查本机环境；--online 额外验证一次官方 API")
    doctor.add_argument("--online", action="store_true")
    v = sub.add_parser("collect-videos", help="遍历上传目录并采集目标年份视频元数据")
    v.add_argument("--hours", type=float, default=0, help="运行时限（小时）；默认 0=不限时长")
    sub.add_parser("collect-all", help="不限运行时长，采集目标年份全部公开视频元数据和全部可见评论；同命令续传")
    sub.add_parser("clean", help="清洗视频与评论，保留原始快照和清洗标记")
    plan_cmd = sub.add_parser("plan", help="按元数据计数估算评论采集请求量")
    plan_cmd.add_argument("--all", action="store_true", help="按目标年份全部视频估算，不受已有研究样本限制")
    s = sub.add_parser("sample", help="冻结视频采集样本，n=0 为全视频普查")
    s.add_argument("--n", type=int, default=0, help="默认 0=全视频；大于 0 时分层抽样")
    s.add_argument("--strata", nargs="+", default=["month", "duration_band"])
    c = sub.add_parser("collect-comments", help="获取抽样视频的全部可见评论与回复")
    c.add_argument("--hours", type=float, default=0, help="运行时限（小时）；默认 0=不限时长")
    c.add_argument("--all", action="store_true", help="采集目标年份所有视频的评论，保留已有研究抽样文件")
    sub.add_parser("audit", help="输出目录、时间、采集状态与覆盖率审计")
    prep = sub.add_parser("prepare-coding", help="生成两个盲编码表，不自动填入语义标签")
    prep.add_argument("--kind", choices=["videos", "comments"], required=True)
    prep.add_argument("--threads-per-video", type=int, default=20, help="随机抽取线程数；0=全部")
    prep.add_argument("--reliability-fraction", type=float, default=.2)
    prep.add_argument("--allow-incomplete", action="store_true", help="显式接受采集缺失，仅做探索性分析")
    for name in ["reliability", "finalize-coding"]:
        cmd = sub.add_parser(name, help="独立编码信度/导入最终裁决")
        cmd.add_argument("--kind", choices=["videos", "comments"], required=True)
    a = sub.add_parser("analyze", help="描述、相关、组间、回归与敏感性分析")
    a.add_argument("--descriptive-only", action="store_true")
    sub.add_parser("analyze-structure", help="无需语义编码的全量互动结构、相关、组间与回归分析")
    for name in ["ai-prepare", "ai-plan", "ai-code", "ai-status", "ai-freeze", "ai-evaluate", "ai-analyze"]:
        cmd = sub.add_parser(name, help="独立的 AI 辅助编码/盲验证流程，见 README.md")
        cmd.add_argument("--experiment", default="hybrid_v1", help="AI 实验版本，与采集 run_id 独立")
        if name == "ai-prepare":
            cmd.add_argument("--dev-videos", type=int, default=30)
            cmd.add_argument("--validation-videos", type=int, default=150)
            cmd.add_argument("--dev-threads", type=int, default=50)
            cmd.add_argument("--validation-threads", type=int, default=250)
            cmd.add_argument("--threads-per-video", type=int, default=0, help="0=全部受众线程；不改原人工抽样")
            cmd.add_argument("--transcripts", help="可选的本机字幕 CSV：video_id,transcript,transcript_source")
        if name == "ai-code":
            cmd.add_argument("--split", choices=["dev", "validation", "production", "all"], default="dev")
            cmd.add_argument("--max-requests", type=int, default=500, help="本次请求预算，含重试；0 为显式不限")
            cmd.add_argument("--workers", type=int, help="并发 1..64；默认读取 [ai].workers（未配置时为 8）")
        if name == "ai-evaluate":
            cmd.add_argument("--split", choices=["dev", "validation"], required=True)
    for name in ["agent-prepare", "agent-plan", "agent-prompts", "agent-check", "agent-run", "agent-screen", "agent-status", "agent-analyze", "agent-audit"]:
        cmd = sub.add_parser(name, help="零人工双阶段：初编＋检查，见 README.md")
        cmd.add_argument("--experiment", help="默认读取 ai.default_experiment，未配置则为 agents_v1")
        if name == "agent-prepare":
            cmd.add_argument("--threads-per-video", type=int, default=0, help="0=全部受众线程")
            cmd.add_argument("--transcripts", help="可选现有字幕 CSV，不提供则形式保留缺失")
        if name in {"agent-run", "agent-screen"}:
            cmd.add_argument("--max-requests", type=int, help="各阶段共享；默认读取 ai.default_max_requests，未配置则为 0（不限）；不改变单次输出上限，累计 token 限制由 ai.max_output_tokens_per_run 单独配置")
            cmd.add_argument("--workers", type=int, help="每阶段并发 1..64；默认读取 [ai].workers（未配置时为 8）")
        if name == "agent-check":
            cmd.add_argument("--online", action="store_true", help="双阶段最多 4 次真实接口请求，检查结果独立保存，不冻结或运行正式实验")
    return p


def doctor(p, online):
    import importlib.metadata
    versions = {x: importlib.metadata.version(x) for x in ["httpx", "pandas", "numpy", "scipy", "statsmodels", "matplotlib"]}
    report = {"python": sys.version.split()[0], "machine": platform.machine(), "system": platform.system(),
              "disk_free_gb": round(shutil.disk_usage(p.base).free / 1024 ** 3, 1), "packages": versions,
              "api_key_configured": bool(os.getenv("YOUTUBE_API_KEY")),
              "proxy_env_configured": any(os.getenv(x) for x in ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"])}
    if online:
        api = API(p, .1)
        data, stamp = api.get("channels", part="snippet,contentDetails,statistics", id=p.cfg["study"]["channel_id"])
        if len(data.get("items", [])) != 1:
            raise ValueError("频道 ID 无法解析")
        report["channel_title"] = data["items"][0]["snippet"]["title"]
        report["api_verified_at"] = stamp
    write_json(p.reports / "environment.json", report)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "agent-audit":
            from . import ai_audit
            # This command neither opens the research database for writing nor loads credentials.
            # SQLite supplies a consistent snapshot; an active agent-run need not be interrupted.
            p = ai_audit.readonly_project(args.config)
            ai_audit.generate(p, args.experiment or p.cfg.get("ai", {}).get("default_experiment", "agents_v1"))
            return 0
        p = Project(args.config)
        with p.lock():
            from . import analysis, clean, coding, collect
            if args.command == "init":
                print(f"已初始化：{p.root}；尚未请求 YouTube 数据。")
            elif args.command == "doctor":
                doctor(p, args.online)
            elif args.command == "collect-videos":
                collect.collect_videos(p, args.hours)
            elif args.command == "collect-all":
                collect.collect_all(p)
            elif args.command == "clean":
                clean.clean(p)
            elif args.command == "plan":
                collect.plan(p, all_videos=args.all)
            elif args.command == "sample":
                collect.select_sample(p, args.n, args.strata)
            elif args.command == "collect-comments":
                collect.collect_comments(p, args.hours, all_videos=args.all)
            elif args.command == "audit":
                collect.audit(p)
            elif args.command == "prepare-coding":
                coding.prepare(p, args.kind, args.threads_per_video, args.reliability_fraction, args.allow_incomplete)
            elif args.command == "reliability":
                coding.reliability(p, args.kind)
            elif args.command == "finalize-coding":
                coding.finalize(p, args.kind)
            elif args.command == "analyze":
                analysis.analyze(p, args.descriptive_only)
            elif args.command == "analyze-structure":
                from . import structure
                structure.analyze(p)
            elif args.command.startswith("agent-"):
                from . import ai_agents
                args.experiment = args.experiment or p.cfg.get("ai", {}).get("default_experiment", "agents_v1")
                if args.command == "agent-prepare":
                    ai_agents.prepare(p, args.experiment, args.threads_per_video, args.transcripts)
                elif args.command == "agent-plan":
                    ai_agents.plan(p, args.experiment)
                elif args.command == "agent-prompts":
                    ai_agents.export_prompts(p, args.experiment)
                elif args.command == "agent-check":
                    from . import ai_check
                    ai_check.check(p, args.experiment, args.online)
                elif args.command == "agent-run":
                    ai_agents.run(p, args.experiment, args.max_requests, args.workers)
                elif args.command == "agent-screen":
                    from . import ai_screening
                    ai_screening.run(p, args.experiment, args.max_requests, args.workers)
                elif args.command == "agent-status":
                    ai_agents.status(p, args.experiment)
                elif args.command == "agent-analyze":
                    ai_agents.analyze(p, args.experiment)
            elif args.command.startswith("ai-"):
                from . import ai_analysis, ai_client, ai_data, ai_eval
                if args.command == "ai-prepare":
                    ai_data.prepare(p, args.experiment, args.dev_videos, args.validation_videos,
                                    args.dev_threads, args.validation_threads, args.threads_per_video, args.transcripts)
                elif args.command == "ai-plan":
                    ai_data.plan(p, args.experiment)
                elif args.command == "ai-code":
                    ai_client.run(p, args.experiment, args.split, args.max_requests, args.workers)
                elif args.command == "ai-status":
                    ai_client.status(p, args.experiment)
                elif args.command == "ai-freeze":
                    ai_client.freeze(p, args.experiment)
                elif args.command == "ai-evaluate":
                    ai_eval.evaluate(p, args.experiment, args.split)
                elif args.command == "ai-analyze":
                    ai_analysis.analyze(p, args.experiment)
    except StopCollection as e:
        print(f"可续传停止：{e}", file=sys.stderr)
        return 3
    except (ValueError, FileNotFoundError, APIError, sqlite3.OperationalError) as e:
        print(f"未完成：{e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("已中断；已落盘分页可复用。重跑同一命令继续。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
