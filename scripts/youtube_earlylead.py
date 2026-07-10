#!/usr/bin/env python3
"""
YouTube early-lead persistence -- snapshot crawler + analyzer (skeleton).

Tests the dislike-removal natural experiment of youtube_dislike_protocol.md.
The YouTube Data API returns only a video's CURRENT statistics, not its historical
view trajectory, so rho(tau) (early vs final ordering) needs a LONGITUDINAL panel.
This script builds one two ways:

  --crawl   : select a genre/date cohort and APPEND a timestamped snapshot of
              (view_count, like_count) per video to a panel CSV. Run repeatedly
              (e.g. daily via cron) to accumulate each video's trajectory going
              forward. Needs YOUTUBE_API_KEY.
  --analyze : from an accumulated panel CSV, reconstruct each video's views at
              early fractions tau of a horizon and compute rho(tau) / tau90 for the
              cohort. Works on any panel in the same schema -- including an
              ARCHIVED 2021-2022 panel loaded into it, which is what the
              retrospective test needs (a fresh crawl cannot reconstruct 2021).

Panel schema (one row per snapshot per video):
  snapshot_iso, video_id, category_id, published_iso, view_count, like_count

Usage:
  export YOUTUBE_API_KEY=...
  python scripts/youtube_earlylead.py --crawl --category 27 \
      --published-after 2024-01-01 --published-before 2024-01-08 --cohort 200
  python scripts/youtube_earlylead.py --analyze --horizon-days 180
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import spearmanr

API = "https://www.googleapis.com/youtube/v3"
ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "output" / "youtube_panel.csv"
TAUS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.375, 0.5, 0.75, 1.0])
SESSION = requests.Session()


def _key():
    k = os.environ.get("YOUTUBE_API_KEY")
    if not k:
        sys.exit("ERROR: set YOUTUBE_API_KEY (YouTube Data API v3 key).")
    return k


def _get(endpoint, params):
    params = dict(params, key=_key())
    for attempt in range(5):
        r = SESSION.get(f"{API}/{endpoint}", params=params, timeout=30)
        if r.status_code == 403 and "quota" in r.text.lower():
            sys.exit("YouTube API quota exceeded; resume tomorrow (quota is daily).")
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {endpoint}")


# --------------------------------------------------------------------------- #
# Crawl: build cohort + append a snapshot
# --------------------------------------------------------------------------- #
def search_cohort(category, after, before, cohort):
    """Video ids published in [after, before] in a category, most-viewed first."""
    ids, token = [], None
    after_iso = f"{after}T00:00:00Z"
    before_iso = f"{before}T00:00:00Z"
    while len(ids) < cohort:
        data = _get("search", {
            "part": "id", "type": "video", "order": "viewCount",
            "videoCategoryId": category, "publishedAfter": after_iso,
            "publishedBefore": before_iso, "maxResults": 50,
            **({"pageToken": token} if token else {})})
        ids += [it["id"]["videoId"] for it in data.get("items", [])
                if it["id"].get("videoId")]
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.2)
    return ids[:cohort]


def fetch_stats(ids):
    """Current statistics + snippet for up to 50 ids per call."""
    rows = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _get("videos", {"part": "statistics,snippet", "id": ",".join(chunk)})
        for it in data.get("items", []):
            st, sn = it.get("statistics", {}), it.get("snippet", {})
            rows.append({
                "video_id": it["id"],
                "category_id": sn.get("categoryId", ""),
                "published_iso": sn.get("publishedAt", ""),
                "view_count": int(st.get("viewCount", 0)),
                "like_count": int(st.get("likeCount", 0)),
            })
        time.sleep(0.2)
    return rows


def crawl(args):
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    new = not PANEL.exists()
    # existing cohort ids (so repeat runs re-snapshot the SAME videos)
    ids = []
    if not new:
        with open(PANEL) as fh:
            ids = sorted({r["video_id"] for r in csv.DictReader(fh)})
    if not ids:
        print(f"selecting cohort: category={args.category} "
              f"{args.published_after}..{args.published_before}")
        ids = search_cohort(args.category, args.published_after,
                            args.published_before, args.cohort)
    print(f"snapshotting {len(ids)} videos")
    rows = fetch_stats(ids)
    now = datetime.now(timezone.utc).isoformat()
    with open(PANEL, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["snapshot_iso", "video_id", "category_id",
                        "published_iso", "view_count", "like_count"])
        for r in rows:
            w.writerow([now, r["video_id"], r["category_id"],
                        r["published_iso"], r["view_count"], r["like_count"]])
    print(f"appended {len(rows)} snapshot rows to {PANEL}")
    print("run again later (e.g. daily) to accumulate trajectories; then --analyze")


# --------------------------------------------------------------------------- #
# Analyze: rho(tau) from an accumulated panel
# --------------------------------------------------------------------------- #
def _parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def analyze(args):
    if not PANEL.exists():
        sys.exit(f"no panel at {PANEL}; --crawl first (or load an archived panel).")
    # group snapshots by video: list of (age_days, views)
    per_video, pub = {}, {}
    with open(PANEL) as fh:
        for r in csv.DictReader(fh):
            vid = r["video_id"]
            try:
                t0 = _parse_iso(r["published_iso"])
                age = (_parse_iso(r["snapshot_iso"]) - t0).total_seconds() / 86400
                per_video.setdefault(vid, []).append((age, int(r["view_count"])))
                pub[vid] = t0
            except (ValueError, KeyError):
                continue
    H = args.horizon_days
    cols = []          # per video: views at each tau (interpolated), + final
    for vid, pts in per_video.items():
        pts = sorted(pts)
        if len(pts) < 2 or pts[-1][0] < H * TAUS[3]:   # too little coverage
            continue
        ages = np.array([a for a, _ in pts])
        views = np.array([v for _, v in pts], dtype=float)
        row = [float(np.interp(tau * H, ages, views)) for tau in TAUS]
        cols.append(row)
    if len(cols) < 30:
        sys.exit(f"only {len(cols)} videos with enough coverage; crawl longer "
                 f"or lower --horizon-days.")
    C = np.array(cols)
    final = C[:, -1]
    rho = np.array([spearmanr(C[:, j], final).statistic for j in range(len(TAUS))])
    hit = np.where(rho >= 0.9)[0]
    tau90 = float(TAUS[hit[0]]) if len(hit) else float("nan")

    print(f"\nN videos = {len(cols)}   horizon = {H} days")
    print(f"{'tau':>7} {'rho':>8} {'sqrt':>8}")
    for t, rv in zip(TAUS, rho):
        print(f"{t:>7.3f} {rv:>8.3f} {np.sqrt(t):>8.3f}")
    print(f"tau90 = {tau90}")
    print("For the natural experiment, compare tau90 / rho(0.1) between a "
          "pre-2021-11 and a post-2021-11 cohort, and DiD across genres by "
          "dislike-informativeness (youtube_dislike_protocol.md).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crawl", action="store_true", help="append a snapshot")
    ap.add_argument("--analyze", action="store_true", help="rho(tau) from panel")
    ap.add_argument("--category", default="27", help="videoCategoryId (27=Education)")
    ap.add_argument("--published-after", default="2024-01-01")
    ap.add_argument("--published-before", default="2024-01-08")
    ap.add_argument("--cohort", type=int, default=200)
    ap.add_argument("--horizon-days", type=float, default=180)
    args = ap.parse_args()
    if args.crawl:
        crawl(args)
    elif args.analyze:
        analyze(args)
    else:
        ap.error("choose --crawl or --analyze")


if __name__ == "__main__":
    main()
