#!/usr/bin/env python3
"""
01_scrape_truth_social.py
=========================

Collect Trump's Truth Social posts for the study window.

Sources:
  - Truth Social public API (Mastodon-compatible, no auth required)
  - CNN ix.cnn.io archive (backfill / baseline)
  - Stiles S3 archive (additional backfill, may be unavailable)

Outputs:
  data/raw/truth_archive.json   (all posts, deduplicated)
  data/raw/truth_archive.csv    (same in CSV)

Usage:
    python code/01_scrape_truth_social.py [--max-pages 0] [--no-backfill] [--delay 2.0]

Note for replication:
  The cached `data/raw/posts_60d.parquet` shipped with this repo is the
  topic-tagged 60-day analysis subset that all downstream stages consume.
  Re-running this script overwrites `truth_archive.{json,csv}` but does NOT
  regenerate posts_60d.parquet — that is the output of the topic classifier
  used in our prior work (Graham et al., 2026), which is not redistributed
  here. To redo the topic tagging from scratch, see docs/methodology.md.
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

import requests

from _paths import RAW_DIR, TRUTH_ARCHIVE_JSON, TRUTH_ARCHIVE_CSV, ensure_dirs

# ─── Configuration ────────────────────────────────────────────────────────────

TRUMP_ACCOUNT_ID = "107780257626128497"
TRUTH_SOCIAL_API = f"https://truthsocial.com/api/v1/accounts/{TRUMP_ACCOUNT_ID}/statuses"
CNN_ARCHIVE_JSON = "https://ix.cnn.io/data/truth-social/truth_archive.json"
STILES_ARCHIVE = "https://stilesdata.com/trump-truth-social-archive/truth_archive.json"

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://truthsocial.com/@realDonaldTrump",
}

POSTS_PER_PAGE = 40
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF = 5


# ─── HTML / Text Cleaning ────────────────────────────────────────────────────

def clean_html(raw_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", raw_html)
    text = re.sub(r"<.*?>", "", text)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&#39;", "'").replace("&apos;", "'"))
    return text.strip()


def normalize_unicode(text: str) -> str:
    try:
        import ftfy
        return ftfy.fix_text(text)
    except ImportError:
        try:
            if any(ord(c) > 127 for c in text):
                return text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        return text


def clean_content(raw: str) -> str:
    return normalize_unicode(clean_html(raw))


# ─── Data Extraction ─────────────────────────────────────────────────────────

def extract_post(raw_post: dict) -> dict:
    if "media_attachments" in raw_post:
        media_urls = [
            m.get("url", "") or m.get("preview_url", "")
            for m in raw_post.get("media_attachments", [])
            if m.get("url") or m.get("preview_url")
        ]
    else:
        media = raw_post.get("media", [])
        if isinstance(media, str):
            media_urls = [u.strip() for u in media.split(";") if u.strip()]
        elif isinstance(media, list):
            media_urls = media
        else:
            media_urls = []

    content = raw_post.get("content", "")
    if "<" in content and ">" in content:
        content = clean_content(content)
    else:
        content = normalize_unicode(content).strip()

    post_url = raw_post.get("url", "")
    if not post_url and raw_post.get("id"):
        post_url = f"https://truthsocial.com/@realDonaldTrump/{raw_post['id']}"

    return {
        "id": str(raw_post.get("id", "")),
        "created_at": raw_post.get("created_at", ""),
        "content": content,
        "url": post_url,
        "media": media_urls,
        "replies_count": int(raw_post.get("replies_count", 0)),
        "reblogs_count": int(raw_post.get("reblogs_count", 0)),
        "favourites_count": int(raw_post.get("favourites_count", 0)),
        "is_reblog": raw_post.get("reblog") is not None,
        "reblog_of": (
            raw_post["reblog"].get("url", "")
            if isinstance(raw_post.get("reblog"), dict) else None
        ),
    }


# ─── Backfill ────────────────────────────────────────────────────────────────

def fetch_archive_backfill(session: requests.Session) -> list[dict]:
    print("\n[backfill] fetching upstream archives...")
    for url, label in [(CNN_ARCHIVE_JSON, "CNN"), (STILES_ARCHIVE, "Stiles S3")]:
        try:
            print(f"   trying {label}: {url}")
            r = session.get(url, timeout=60)
            r.raise_for_status()
            posts = [extract_post(p) for p in r.json()]
            print(f"   {label} OK: {len(posts):,} posts")
            return posts
        except Exception as e:
            print(f"   {label} failed: {e}")
    print("   all backfill sources failed; starting fresh")
    return []


# ─── API scraping ────────────────────────────────────────────────────────────

def fetch_api_page(session: requests.Session, max_id: Optional[str] = None) -> list[dict]:
    params = {"exclude_replies": "true", "with_muted": "true", "limit": str(POSTS_PER_PAGE)}
    if max_id:
        params["max_id"] = max_id
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(TRUTH_SOCIAL_API, params=params, headers=DEFAULT_HEADERS,
                            timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                print(f"   rate-limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = RETRY_BACKOFF * attempt
            print(f"   attempt {attempt}/{MAX_RETRIES} failed: {e}; retrying in {wait}s")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    return []


def scrape_all(session, existing_ids, max_pages, delay):
    print(f"\n[api] {TRUTH_SOCIAL_API}")
    new_posts = []
    max_id = None
    page = 0
    empty = dupes = 0
    while True:
        page += 1
        if max_pages > 0 and page > max_pages:
            break
        raw = fetch_api_page(session, max_id=max_id)
        if not raw:
            empty += 1
            if empty >= 3:
                print("   3 empty pages; stopping")
                break
            time.sleep(delay)
            continue
        empty = 0
        page_new = 0
        for r in raw:
            p = extract_post(r)
            if p["id"] not in existing_ids:
                new_posts.append(p)
                existing_ids.add(p["id"])
                page_new += 1
        print(f"   page {page}: {len(raw)} fetched, {page_new} new")
        if page_new == 0:
            dupes += 1
            if dupes >= 5:
                print("   5 dupe-only pages; stopping")
                break
        else:
            dupes = 0
        max_id = str(int(raw[-1]["id"]) - 1)
        time.sleep(delay)
    return new_posts


# ─── Output ──────────────────────────────────────────────────────────────────

def merge_posts(*sources):
    merged = {}
    for s in sources:
        for p in s:
            pid = str(p.get("id", ""))
            if pid:
                merged[pid] = p
    return sorted(merged.values(), key=lambda p: p.get("created_at", ""), reverse=True)


def save(posts):
    with open(TRUTH_ARCHIVE_JSON, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    print(f"   wrote {TRUTH_ARCHIVE_JSON} ({len(posts):,} posts, "
          f"{os.path.getsize(TRUTH_ARCHIVE_JSON)/1e6:.1f} MB)")
    fields = ["id", "created_at", "content", "url", "media",
              "replies_count", "reblogs_count", "favourites_count",
              "is_reblog", "reblog_of"]
    with open(TRUTH_ARCHIVE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for p in posts:
            row = dict(p)
            row["media"] = "; ".join(row.get("media") or [])
            w.writerow(row)
    print(f"   wrote {TRUTH_ARCHIVE_CSV}")


def main():
    ap = argparse.ArgumentParser(description="Scrape Trump's Truth Social posts.")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--no-backfill", action="store_true")
    ap.add_argument("--api-only", action="store_true")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args()

    ensure_dirs()
    print("=" * 60)
    print(f"  Truth Social scrape   started {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    existing = []
    if TRUTH_ARCHIVE_JSON.exists():
        with open(TRUTH_ARCHIVE_JSON, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"loaded {len(existing):,} existing posts")

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    backfill = []
    if not args.no_backfill and not args.api_only:
        backfill = fetch_archive_backfill(session)

    known = {str(p.get("id", "")) for p in (existing + backfill)}
    api_posts = scrape_all(session, known, args.max_pages, args.delay)

    print("\n[merge]")
    all_posts = merge_posts(existing, backfill, api_posts)
    print(f"   total unique: {len(all_posts):,}")

    save(all_posts)


if __name__ == "__main__":
    main()
