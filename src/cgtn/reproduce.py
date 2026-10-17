"""Recompute the existing study from curated CSVs without databases or API access."""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import tomllib
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import ai_analysis, ai_data, ai_screening_report, analysis, structure
from .coding import COMMENT_LABELS
from .core import write_csv, write_json


def verify_dataset(root):
    """Verify every retained source against its frozen byte-level checksum."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Invalid or duplicate dataset manifest path")
        seen.add(str(relative))
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external dataset file: {relative}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {relative}")
    return manifest


class CSVStore:
    """The small dataframe interface used by the original descriptive/structure code."""

    def __init__(self, root):
        self.root = root

    def frame(self, table):
        names = {"clean_videos": "videos.csv", "clean_comments": "comments.csv",
                 "collection": "collection_coverage.csv"}
        if table not in names:
            raise ValueError(f"Unsupported CSV table: {table}")
        return pd.read_csv(self.root / "processed" / names[table], low_memory=False)


def read_comparisons(path):
    # Preserve heterogeneous numeric/video labels as strings, as in the original check metrics.
    frame = pd.read_csv(path, dtype={"id": str, "initial": str, "checked": str}, keep_default_na=False)
    for column in ["paired_known", "unavailable"]:
        mapped = frame[column].astype(str).map({"True": True, "False": False})
        if mapped.isna().any():
            raise ValueError(f"Invalid boolean in comparison column: {column}")
        frame[column] = mapped.astype(bool)
    frame["agrees"] = frame.agrees.astype(str).map({"True": True, "False": False, "": None})
    if frame.loc[frame.paired_known, "agrees"].isna().any():
        raise ValueError("Paired comparisons must declare agreement")
    frame["weight"] = pd.to_numeric(frame.weight, errors="raise")
    if not np.isfinite(frame.weight).all() or frame.weight.le(0).any():
        raise ValueError("Comparison weights must be positive and finite")
    return frame


def compare_tables(reference, output):
    """Compare all recomputable reference tables; report mismatched columns without row contents."""
    checks = []
    for source in sorted(reference.rglob("*.csv")):
        relative = source.relative_to(reference)
        if relative.parts[0] == "audit" or source.name == "excluded_videos.csv":
            checks.append({"table": relative.as_posix(), "status": "archived_only",
                           "reason": "requires original API/request history, not statistical inputs"})
            continue
        target = output / relative
        if not target.exists():
            checks.append({"table": relative.as_posix(), "status": "missing"})
            continue
        expected = pd.read_csv(source, low_memory=False)
        actual = pd.read_csv(target, low_memory=False)
        mismatch = []
        if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
            mismatch.append("shape_or_columns")
        else:
            for column in expected:
                try:
                    pd.testing.assert_series_equal(expected[column], actual[column], check_dtype=False,
                                                   check_exact=False, rtol=1e-7, atol=1e-9)
                except AssertionError:
                    mismatch.append(column)
        checks.append({"table": relative.as_posix(), "rows": len(expected),
                       "status": "different" if mismatch else "matched", "different_columns": mismatch})
    return checks


def _run(root, output, cfg):
    source_report = json.loads((root / "metadata/analysis_report.json").read_text(encoding="utf-8"))
    if (cfg["analysis"] != source_report["analysis_configuration"]
            or cfg["study"]["seed"] != source_report["analysis_seed"]):
        raise ValueError("Study configuration does not match the frozen analysis report")
    context = SimpleNamespace(cfg=cfg, reports=output / "descriptive", store=CSVStore(root))
    context.reports.mkdir()
    print("Recomputing descriptive and interaction-structure statistics...", flush=True)
    analysis.analyze(context, descriptive_only=True)
    context.reports = output
    structure.analyze(context)
    prediction = {kind: pd.read_csv(root / "labels" / f"predictions_{kind}.csv", low_memory=False)
                  for kind in ["comments", "videos"]}
    gate = {**{"videos." + field: True for field in ai_data.VIDEO_FIELDS},
            **{"comments." + field: True for field in COMMENT_LABELS}}
    report = {"version": source_report["model_version"], "reference_type": "none_ai_only",
              "workflow": "screening", "gate": gate}
    print("Recomputing final annotation analysis...", flush=True)
    ai_analysis.analyze_predictions(context, root / "inputs", prediction, report, output / "analysis")
    comparisons = read_comparisons(root / "checks/label_comparisons.csv")
    write_csv(ai_screening_report.metrics_from_comparisons(comparisons), output / "checks/check_metrics.csv")
    final = pd.read_csv(output / "analysis/analysis_videos.csv", low_memory=False)
    initial = pd.read_csv(root / "inputs/initial_analysis_videos.csv", low_memory=False)
    if not initial.video_id.is_unique or set(initial.video_id) != set(final.video_id):
        raise ValueError("Initial/final sensitivity frames have inconsistent video IDs")
    write_csv(initial, output / "analysis/sensitivity_initial_videos.csv")
    correlations = pd.read_csv(output / "analysis/correlations.csv").set_index("y")
    sensitivity = []
    for variant, frame in [("screened", final), ("initial", initial)]:
        for metric in correlations.index:
            stats = (correlations.loc[metric, ["n", "rho", "p", "ci_low", "ci_high", "bootstrap_valid"]].to_dict()
                     if variant == "screened" else analysis.boot_correlation(
                         frame, "view_count", metric, cfg["analysis"]["bootstrap_reps"], cfg["study"]["seed"]))
            sensitivity.append({"variant": variant, "metric": metric, **stats})
    write_csv(analysis.correct_p(pd.DataFrame(sensitivity), "fdr_bh"), output / "analysis/sensitivity_correlations.csv")
    checks = compare_tables(root / "reference", output)
    result = {"network_requests": 0, "database_required": False,
              "reference_tolerance": {"relative": 1e-7, "absolute": 1e-9},
              "matched_tables": sum(r["status"] == "matched" for r in checks),
              "archived_tables": sum(r["status"] == "archived_only" for r in checks),
              "failed_tables": sum(r["status"] not in {"matched", "archived_only"} for r in checks),
              "tables": checks}
    write_json(output / "verification.json", result)
    if result["failed_tables"]:
        raise ValueError(f"{result['failed_tables']} reference tables differ; inspect {output / 'verification.json'}")
    print(f"Reproduction passed: {result['matched_tables']} tables match; "
          f"{result['archived_tables']} provenance-only tables retained. See {output / 'verification.json'}", flush=True)
    return result


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    verify_dataset(root)
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the frozen dataset directory")
    cfg = tomllib.loads((root / "study.toml").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)

    def no_network(*args, **kwargs):
        raise RuntimeError("CSV reproduction must not access the network")

    with ExitStack() as stack:
        for owner, attr in [(socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "getaddrinfo")]:
            stack.enter_context(patch.object(owner, attr, no_network))
        return _run(root, output, cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="Curated dataset directory")
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"), help="New output directory")
    parser.add_argument("--verify-only", action="store_true", help="Check frozen file hashes without recomputing")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_dataset(args.data)
            print(f"Dataset verified: {len(manifest['files'])} files")
        else:
            run(args.data, args.output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Reproduction failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
