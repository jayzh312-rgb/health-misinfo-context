"""Inspect the dated CoAID releases without modifying raw data."""

from __future__ import annotations

import csv
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable, TextIO

import pandas as pd


RANDOM_SEED = 42  # Reserved for future operations that require randomness.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COAID_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "coaid"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
SUMMARY_PATH = OUTPUT_DIR / "coaid_dataset_summary.csv"
REPORT_PATH = OUTPUT_DIR / "coaid_inspection_report.txt"
MANIFEST_PATH = OUTPUT_DIR / "coaid_file_manifest.csv"
RELEASE_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")
PREVIEW_ROWS = 5
LOW_CARDINALITY_MAX = 20

COLUMN_HINTS = {
    "likely_label_columns": (
        "label", "class", "category", "veracity", "credibility", "fake", "real",
        "misinformation", "truth", "rating",
    ),
    "likely_datetime_columns": (
        "date", "time", "timestamp", "created_at", "published", "publish_date",
    ),
    "likely_text_columns": (
        "text", "title", "content", "article", "body", "claim", "description",
        "headline", "statement", "abstract",
    ),
    "likely_social_columns": (
        "tweet", "twitter", "reply", "retweet", "favorite", "favourite", "like",
        "share", "comment", "user", "follower", "following", "engagement",
        "social", "screen_name", "status_id",
    ),
}

SUMMARY_FIELDS = [
    "release_period", "source_directory", "file", "relative_path", "status", "rows",
    "columns", "column_names", "data_types", "duplicate_rows", "missing_cells",
    "missing_percent", "likely_primary_key", "related_id_columns", "likely_label_columns",
    "likely_datetime_columns", "likely_text_columns", "likely_social_columns", "error",
]

MANIFEST_FIELDS = [
    "release_period", "filename", "rows", "columns", "content_type",
    "likely_primary_key", "related_id_columns", "missing_percent", "notes",
]


class Tee:
    """Write the same text to the terminal and a persistent report."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def release_period(path: Path) -> str:
    """Extract the dated release folder from a raw file path."""
    for part in reversed(path.relative_to(COAID_RAW_DIR).parts[:-1]):
        if RELEASE_PATTERN.fullmatch(part):
            return part
    return "unknown"


def matching_columns(columns: Iterable[object], keywords: tuple[str, ...]) -> list[str]:
    """Return columns whose normalized names contain supplied keywords."""
    matches: list[str] = []
    for column in columns:
        name = str(column)
        normalized = name.lower().strip().replace(" ", "_").replace("-", "_")
        if any(keyword in normalized for keyword in keywords):
            matches.append(name)
    return matches


def likely_id_columns(columns: Iterable[object]) -> list[str]:
    """Identify ID-like columns conservatively from their names."""
    matches: list[str] = []
    exact_names = {"id", "index", "tweetid", "tweet_id", "article_id", "news_id", "claim_id"}
    for column in columns:
        name = str(column)
        normalized = name.lower().strip().replace(" ", "_").replace("-", "_")
        if (
            normalized in exact_names
            or normalized.startswith("unnamed:")
            or normalized.endswith("_id")
            or normalized.startswith("id_")
        ):
            matches.append(name)
    return matches


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str]:
    """Load a CSV using common encodings while preserving all original values."""
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False), encoding
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Could not parse CSV. " + " | ".join(errors))


def content_type(filename: str) -> str:
    """Classify table role from the official filename without changing labels."""
    stem = Path(filename).stem
    if stem.endswith("_tweets_replies"):
        return "reply_mapping"
    if stem.endswith("_tweets"):
        return "tweet_mapping"
    return "content"


def primary_key(frame: pd.DataFrame, role: str) -> tuple[str, list[str], str]:
    """Return an evidence-based candidate key and related ID columns."""
    ids = likely_id_columns(frame.columns)
    preferred = {
        "content": ["Unnamed: 0"],
        "tweet_mapping": ["tweet_id"],
        "reply_mapping": ["reply_id"],
    }[role]
    for column in preferred:
        if column in frame and len(frame) > 0 and frame[column].notna().all() and frame[column].is_unique:
            return column, ids, "single complete unique ID column"

    composites = {
        "content": ["Unnamed: 0"],
        "tweet_mapping": ["index", "tweet_id"],
        "reply_mapping": ["news_id", "tweet_id", "reply_id"],
    }[role]
    if composites and all(column in frame for column in composites):
        complete = frame[composites].notna().all(axis=1).all()
        unique = not frame.duplicated(subset=composites).any()
        if len(frame) > 0 and complete and unique:
            return f"({', '.join(composites)})", ids, "composite fields are complete and unique"
    return "undetermined", ids, "no tested ID candidate is complete and unique"


def inspect_file(path: Path) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    """Print diagnostics and return summary/manifest rows plus the loaded frame."""
    relative = path.relative_to(COAID_RAW_DIR)
    period = release_period(path)
    frame, encoding = read_csv_safely(path)
    rows, columns = frame.shape
    role = content_type(path.name)
    key, ids, key_note = primary_key(frame, role)
    duplicate_rows = int(frame.duplicated().sum())
    missing_counts = frame.isna().sum()
    missing_percentages = (missing_counts / rows * 100) if rows else missing_counts.astype(float)
    total_cells = rows * columns
    missing_cells = int(missing_counts.sum())
    total_missing_percent = round(missing_cells / total_cells * 100, 2) if total_cells else 0.0
    hints = {
        category: matching_columns(frame.columns, keywords)
        for category, keywords in COLUMN_HINTS.items()
    }

    print("\n" + "=" * 88)
    print(f"Release period: {period}")
    print(f"Source directory: {relative.parent.as_posix()}")
    print(f"File: {path.name}")
    print(f"Relative path: {relative.as_posix()}")
    print(f"Encoding used: {encoding}")
    print(f"Content type: {role}")
    print(f"Shape: {rows:,} rows x {columns:,} columns")
    print(f"Column names: {list(frame.columns)}")
    print(f"\nData types:\n{frame.dtypes.to_string()}")
    missing_report = pd.DataFrame(
        {"missing_count": missing_counts, "missing_percent": missing_percentages.round(2)}
    )
    print(f"\nMissing values:\n{missing_report.to_string()}")
    print(f"\nDuplicate rows: {duplicate_rows:,}")
    print(f"Likely primary key: {key} ({key_note})")
    print(f"Related ID columns: {ids or 'none detected'}")
    for column in ids:
        print(
            f"  {column}: {frame[column].nunique(dropna=True):,} unique non-missing "
            f"value(s), {int(frame[column].isna().sum()):,} missing"
        )
    print("Column-name hints (candidates only; original meanings are preserved):")
    for category, matches in hints.items():
        print(f"  {category}: {matches or 'none detected'}")
    low_columns = [
        column for column in frame.columns
        if 0 < frame[column].nunique(dropna=True) <= LOW_CARDINALITY_MAX
    ]
    print(f"Low-cardinality values (<= {LOW_CARDINALITY_MAX} unique non-missing):")
    if not low_columns:
        print("  none detected")
    for column in low_columns:
        print(f"  {column}: {frame[column].dropna().unique().tolist()}")
    print(f"\nFirst {min(PREVIEW_ROWS, rows)} rows:\n{frame.head(PREVIEW_ROWS).to_string(index=False)}")

    summary = {
        "release_period": period,
        "source_directory": relative.parent.as_posix(),
        "file": path.name,
        "relative_path": relative.as_posix(),
        "status": "loaded",
        "rows": rows,
        "columns": columns,
        "column_names": " | ".join(map(str, frame.columns)),
        "data_types": " | ".join(f"{column}:{dtype}" for column, dtype in frame.dtypes.items()),
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "missing_percent": total_missing_percent,
        "likely_primary_key": key,
        "related_id_columns": " | ".join(ids),
        **{category: " | ".join(matches) for category, matches in hints.items()},
        "error": "",
    }
    filename_categories = []
    for value in ("News", "Claim", "Fake", "Real"):
        if value in path.name:
            filename_categories.append(value)
    manifest = {
        "release_period": period,
        "filename": path.name,
        "rows": rows,
        "columns": columns,
        "content_type": role,
        "likely_primary_key": key,
        "related_id_columns": " | ".join(ids),
        "missing_percent": total_missing_percent,
        "notes": f"Filename categories: {', '.join(filename_categories)}; {key_note}",
    }
    return summary, manifest, frame


def base_stem(filename: str) -> str:
    """Return the content-table stem shared by its tweet and reply tables."""
    stem = Path(filename).stem
    return stem.removesuffix("_tweets_replies").removesuffix("_tweets")


def print_relationship_audit(frames: dict[tuple[str, str], pd.DataFrame]) -> None:
    """Test within-release links and cross-release identifier overlap."""
    print("\n" + "#" * 88)
    print("RELATIONSHIP AND RELEASE-OVERLAP AUDIT")
    periods = sorted({period for period, _ in frames})
    print("\nRelease-level row totals (mapping rows are engagements, not unique content items):")
    for period in periods:
        totals = {"content": 0, "tweet_mapping": 0, "reply_mapping": 0}
        for (frame_period, filename), frame in frames.items():
            if frame_period == period:
                totals[content_type(filename)] += len(frame)
        print(
            f"  {period}: content={totals['content']:,}, "
            f"tweet_mapping={totals['tweet_mapping']:,}, "
            f"reply_mapping={totals['reply_mapping']:,}"
        )
    for period in periods:
        print(f"\nRelease {period}:")
        stems = sorted({base_stem(name) for p, name in frames if p == period})
        for stem in stems:
            content = frames.get((period, f"{stem}.csv"))
            tweets = frames.get((period, f"{stem}_tweets.csv"))
            replies = frames.get((period, f"{stem}_tweets_replies.csv"))
            print(f"  {stem}:")
            if content is not None and tweets is not None and "Unnamed: 0" in content and "index" in tweets:
                content_ids = set(content["Unnamed: 0"].dropna())
                linked = int(tweets["index"].isin(content_ids).sum())
                print(f"    content ID -> tweet index: {linked:,}/{len(tweets):,} tweet rows linked")
            else:
                print("    content ID -> tweet index: not testable (table or expected column absent)")
            if tweets is not None and replies is not None and "tweet_id" in tweets and "tweet_id" in replies:
                tweet_ids = set(tweets["tweet_id"].dropna())
                linked = int(replies["tweet_id"].isin(tweet_ids).sum())
                print(f"    tweet ID -> reply tweet_id: {linked:,}/{len(replies):,} reply rows linked")
            else:
                print("    tweet ID -> reply tweet_id: not testable (table or expected column absent)")
            if content is not None and replies is not None and "Unnamed: 0" in content and "news_id" in replies:
                content_ids = set(content["Unnamed: 0"].dropna())
                linked = int(replies["news_id"].isin(content_ids).sum())
                print(f"    content ID -> reply news_id: {linked:,}/{len(replies):,} reply rows linked")
            else:
                print("    content ID -> reply news_id: not testable (table or expected column absent)")

    print("\nCross-release overlaps (same table family; exact identifier values):")
    for stem in sorted({base_stem(name) for _, name in frames}):
        for suffix, id_column in ((".csv", "Unnamed: 0"), ("_tweets.csv", "tweet_id"), ("_tweets_replies.csv", "reply_id")):
            available = [(period, frames[(period, f"{stem}{suffix}")]) for period in periods if (period, f"{stem}{suffix}") in frames]
            overlaps = []
            for index, (left_period, left) in enumerate(available):
                if id_column not in left:
                    continue
                left_ids = set(left[id_column].dropna())
                for right_period, right in available[index + 1:]:
                    if id_column in right:
                        count = len(left_ids & set(right[id_column].dropna()))
                        if count:
                            overlaps.append(f"{left_period}/{right_period}={count}")
            print(f"  {stem}{suffix} [{id_column}]: {', '.join(overlaps) if overlaps else 'no overlap'}")

    print("\nExact content-item overlaps across releases (ID column excluded):")
    for stem in sorted({base_stem(name) for _, name in frames}):
        available = [(period, frames[(period, f"{stem}.csv")]) for period in periods if (period, f"{stem}.csv") in frames]
        results = []
        for index, (left_period, left) in enumerate(available):
            for right_period, right in available[index + 1:]:
                shared = [column for column in left.columns if column in right.columns and column != "Unnamed: 0"]
                if not shared:
                    continue
                left_rows = set(map(tuple, left[shared].fillna("<MISSING>").astype(str).to_numpy()))
                right_rows = set(map(tuple, right[shared].fillna("<MISSING>").astype(str).to_numpy()))
                count = len(left_rows & right_rows)
                results.append(f"{left_period}/{right_period}={count}")
        if results:
            print(f"  {stem}: {', '.join(results)}")

    print("\nContent overlaps using stable-looking fields (evidence of repeated items):")
    print("  Note: matching URLs/titles are indicators, not a final deduplication rule.")
    for stem in sorted({base_stem(name) for _, name in frames}):
        available = [(period, frames[(period, f"{stem}.csv")]) for period in periods if (period, f"{stem}.csv") in frames]
        for index, (left_period, left) in enumerate(available):
            for right_period, right in available[index + 1:]:
                checks = []
                for columns in (("news_url",), ("title",), ("fact_check_url", "title")):
                    if all(column in left and column in right for column in columns):
                        left_rows = set(map(tuple, left[list(columns)].dropna().astype(str).to_numpy()))
                        right_rows = set(map(tuple, right[list(columns)].dropna().astype(str).to_numpy()))
                        checks.append(f"{'+'.join(columns)}={len(left_rows & right_rows)}")
                if checks:
                    print(f"  {stem} {left_period}/{right_period}: {', '.join(checks)}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    """Write a generated table outside the raw-data directory."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_inspection() -> int:
    """Discover, inspect, summarize, and audit all dated CoAID CSVs."""
    csv_files = sorted(path for path in COAID_RAW_DIR.rglob("*.csv") if path.is_file())
    if not csv_files:
        print(f"No CoAID CSV files found beneath: {COAID_RAW_DIR}")
        write_csv(SUMMARY_PATH, SUMMARY_FIELDS, [])
        write_csv(MANIFEST_PATH, MANIFEST_FIELDS, [])
        return 0

    print(f"Found {len(csv_files)} CSV file(s) beneath {COAID_RAW_DIR}")
    summaries: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    errors = 0
    for path in csv_files:
        try:
            summary, manifest_row, frame = inspect_file(path)
            summaries.append(summary)
            manifest.append(manifest_row)
            frames[(str(summary["release_period"]), path.name)] = frame
        except Exception as exc:
            errors += 1
            relative = path.relative_to(COAID_RAW_DIR)
            print(f"\nERROR loading {relative.as_posix()}: {type(exc).__name__}: {exc}")
            summaries.append({
                **{field: "" for field in SUMMARY_FIELDS},
                "release_period": release_period(path),
                "source_directory": relative.parent.as_posix(),
                "file": path.name,
                "relative_path": relative.as_posix(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })

    print_relationship_audit(frames)
    write_csv(SUMMARY_PATH, SUMMARY_FIELDS, summaries)
    write_csv(MANIFEST_PATH, MANIFEST_FIELDS, manifest)
    print(f"\nSummary saved to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Manifest saved to: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Inspection complete: {len(csv_files) - errors} loaded, {errors} error(s).")
    return 1 if errors else 0


def main() -> int:
    """Write a fresh report while also streaming progress to the terminal."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            result = run_inspection()
    print(f"Full report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
