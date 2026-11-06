"""Shared deterministic helpers for provenance-preserving content tables."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

SNOWFLAKE_EPOCH_MS = 1288834974657
COMMON_COLUMNS = [
    "dataset", "source_release_or_subset", "content_uid", "content_category",
    "veracity_original", "title", "text", "publish_date_raw", "publish_date_utc",
    "publish_date_quality", "observed_post_count", "unique_post_count", "reply_count",
    "unique_reply_count", "engagement_present", "first_observed_post_time",
    "last_observed_post_time", "observed_lifespan", "publication_to_first_post_delay",
    "provenance",
]

def nonempty(value: Any) -> bool:
    return value is not None and not pd.isna(value) and (not isinstance(value, str) or bool(value.strip()))

def clean_scalar(value: Any) -> str:
    return "" if not nonempty(value) else str(value).strip()

def stable_uid(*parts: Any) -> str:
    raw = "\x1f".join(clean_scalar(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

def normalize_text(value: Any) -> str:
    text = clean_scalar(value).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()

def normalize_url(value: Any) -> str:
    text = clean_scalar(value).casefold().strip().rstrip("/")
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    return text

def parse_date(value: Any) -> tuple[str, str]:
    if not nonempty(value):
        return "", "missing"
    if isinstance(value, (int, float)):
        parsed = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return "", "unparsable"
    if parsed.year < 1990 or parsed.year > 2026:
        return parsed.isoformat(), "implausible_range"
    return parsed.isoformat(), "valid"

def snowflake_time(identifier: Any) -> tuple[str, str]:
    try:
        number = int(str(identifier).strip())
        parsed = pd.to_datetime((number >> 22) + SNOWFLAKE_EPOCH_MS, unit="ms", utc=True, errors="coerce")
    except (ValueError, TypeError, OverflowError):
        return "", "decode_error"
    if pd.isna(parsed):
        return "", "decode_error"
    quality = "valid" if 2010 <= parsed.year <= 2026 else "impossible_date"
    return parsed.isoformat(), quality

def timing(post_ids: Iterable[Any], publish_utc: str) -> dict[str, Any]:
    decoded = [pd.Timestamp(ts) for identifier in post_ids if (ts := snowflake_time(identifier)[0])]
    if not decoded:
        return {"first_observed_post_time":"", "last_observed_post_time":"", "observed_lifespan":"", "publication_to_first_post_delay":""}
    first, last = min(decoded), max(decoded)
    delay = ""
    if publish_utc:
        delay = (first - pd.Timestamp(publish_utc)).total_seconds()
    return {"first_observed_post_time":first.isoformat(), "last_observed_post_time":last.isoformat(), "observed_lifespan":(last-first).total_seconds(), "publication_to_first_post_delay":delay}

def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def project_root(script_file: str) -> Path:
    return Path(script_file).resolve().parents[1]
