import os
import json
import math
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from moviepy import VideoFileClip, concatenate_videoclips
from yt_dlp import YoutubeDL


# -----------------------------
# Config / thresholds
# -----------------------------
@dataclass
class Config:
    resize_w: int = 320
    resize_h: int = 240

    # Threshold used on per-frame "table" (abs(table) >= threshold)
    luminance_segment_threshold: float = 25.0

    # Trigger counting: hits > 3 within ~1 second window
    min_hits_in_1s: int = 3

    # Optional: extend removals forward by N seconds (kept for compatibility)
    extend_level_seconds: float = 0.5

    # Pad cuts before/after in seconds
    pad_before_sec: float = 0.05
    pad_after_sec: float = 0.08

    # Merge intervals that are close together
    merge_gap_sec: float = 0.08

    # NEW: require at least this many consecutive flagged frames to cut
    min_run_frames: int = 3

    # NEW: safety cap - never cut more than this fraction of the whole video
    max_cut_fraction: float = 0.25


# -----------------------------
# YouTube download
# -----------------------------
def download_youtube(url: str, out_dir: Optional[str] = None, cookies_file: Optional[str] = None) -> str:
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="flashsafe_")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        # Spoof a real browser so YouTube doesn't 403
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com/",
        },
        "extractor_args": {"youtube": {"player_client": ["web", "android"]}},
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
    }

    # If a cookies.txt file exists, use it (greatly improves success rate)
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file
    elif os.path.exists("cookies.txt"):
        ydl_opts["cookiefile"] = "cookies.txt"

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            base, _ = os.path.splitext(filename)
            mp4 = base + ".mp4"
            if os.path.exists(mp4):
                filename = mp4
        return filename


# -----------------------------
# Video -> frames
# -----------------------------
def load_video_frames(path: str, cfg: Config) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frames_rgb = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.resize(
            frame_bgr, (cfg.resize_w, cfg.resize_h), interpolation=cv2.INTER_AREA
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)

    cap.release()

    if len(frames_rgb) < 2:
        raise RuntimeError("Video too short or no frames read.")

    return np.stack(frames_rgb, axis=0), float(fps)


# -----------------------------
# Core model (vectorized)
# -----------------------------
def run_model(frames_rgb: np.ndarray) -> np.ndarray:
    """
    Returns:
      table: float array [T-1] per-frame score
    """
    mean_rgb = frames_rgb.mean(axis=3)
    avg_lum_frames = 413.435 * (0.002745 * mean_rgb + 0.0189623) ** 2.2

    change_lum = avg_lum_frames[1:] - avg_lum_frames[:-1]  # [T-1,H,W]
    flat = change_lum.reshape(change_lum.shape[0], -1)

    pos = flat.copy()
    pos[pos < 0] = 0
    pos.sort(axis=1)
    pos = np.flip(pos, axis=1)

    neg = (-flat).copy()
    neg[neg < 0] = 0
    neg.sort(axis=1)
    neg = np.flip(neg, axis=1)

    QUARTER = math.ceil(pos.shape[1] / 4)
    p_avgL = pos[:, :QUARTER].mean(axis=1)
    n_avgL = neg[:, :QUARTER].mean(axis=1)

    table = p_avgL - n_avgL
    out = np.empty_like(table)
    out[table > 0] = p_avgL[table > 0]
    out[table <= 0] = -n_avgL[table <= 0]
    return out


# -----------------------------
# Adaptive threshold computation
# -----------------------------
def compute_adaptive_threshold(
    table: np.ndarray,
    base_threshold: float = 25.0,
    percentile: float = 97.5,
    sensitivity_fraction: float = 0.60,
    min_threshold: float = 5.0,
    max_threshold: float = 80.0,
) -> float:
    """
    Derive a threshold relative to THIS video's flash score distribution.

    - Anchors to the high-percentile tail of abs(table) scores.
    - Sets threshold at sensitivity_fraction of that anchor.
    - Clamps to [min_threshold, max_threshold].
    - Falls back to base_threshold for calm videos with no activity.
    """
    abs_scores = np.abs(table)

    active = abs_scores[abs_scores > 1.0]
    if len(active) < 10:
        return base_threshold

    high_pct = float(np.percentile(active, percentile))
    adaptive = high_pct * sensitivity_fraction
    adaptive = max(min_threshold, min(max_threshold, adaptive))

    return round(adaptive, 2)


# -----------------------------
# Trigger counting (kept from your model)
# -----------------------------
def collapse_segments(table: np.ndarray) -> Tuple[List[int], List[float]]:
    def sign(x: float) -> int:
        return 1 if x > 0 else -1

    fin: List[float] = []
    fin_frames: List[int] = []

    cum = float(table[0])
    fin_frame = 1

    for i in range(len(table) - 1):
        if sign(table[i]) == sign(table[i + 1]):
            cum += float(table[i + 1])
            fin_frame += 1
        else:
            fin.append(cum)
            fin_frames.append(fin_frame)
            cum = float(table[i + 1])
            fin_frame = i + 2

    fin.append(cum)
    fin_frames.append(fin_frame)
    return fin_frames, fin


def get_ep_and_remove_frames(
    fin_frames: List[int], fin: List[float], threshold: float
) -> Tuple[List[int], List[int]]:
    ep_frm: List[int] = []
    rem_frm: List[int] = []
    prev = 0

    for x in range(len(fin)):
        if abs(fin[x]) >= threshold:
            frame_inc = fin_frames[x] - prev
            prev = fin_frames[x]
            rem_frm.append(fin_frames[x])
            ep_frm.append(frame_inc)

    return ep_frm, rem_frm


def possible_triggers(ep_frm: List[int], fps: float, min_hits_in_1s: int) -> int:
    ext = 0
    score = 0
    hits = 0

    for inc in ep_frm:
        if score < fps:
            score += inc
            hits += 1
        else:
            if hits > min_hits_in_1s:
                ext += 1
            score = 0
            hits = 0

    if hits > min_hits_in_1s:
        ext += 1

    return ext


# -----------------------------
# NEW: table-based cut intervals (much more precise)
# -----------------------------
def table_to_cut_intervals(
    table: np.ndarray,
    fps: float,
    threshold: float,
    min_run_frames: int,
    pad_before: float,
    pad_after: float,
    merge_gap: float,
    extend_level_seconds: float,
) -> List[Tuple[float, float]]:
    """
    Build cut intervals from runs where abs(table) >= threshold.
    - min_run_frames prevents cutting on single-frame noise
    - pad_before/after add a little margin
    - merge_gap merges close intervals
    - extend_level_seconds optionally extends the cut forward
    """
    mask = np.abs(table) >= threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    # find consecutive runs in idx
    runs: List[Tuple[int, int]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = i
            prev = i
    runs.append((start, prev))

    # convert runs to padded time intervals
    intervals: List[Tuple[float, float]] = []
    extend = max(0.0, float(extend_level_seconds))
    for a, b in runs:
        run_len = (b - a + 1)
        if run_len < min_run_frames:
            continue

        t1 = (a / fps) - pad_before
        t2 = (b / fps) + pad_after + extend

        t1 = max(0.0, t1)
        t2 = max(t1, t2)
        intervals.append((t1, t2))

    if not intervals:
        return []

    # merge close/overlapping intervals
    intervals.sort()
    merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(x[0], x[1]) for x in merged]


# -----------------------------
# Analyze + report writing
# -----------------------------
def analyze(video_path: str, cfg: Config) -> Dict[str, Any]:
    frames_rgb, fps = load_video_frames(video_path, cfg)
    table = run_model(frames_rgb)

    # Compute adaptive threshold from this video's own score distribution
    adaptive_threshold = compute_adaptive_threshold(
        table,
        base_threshold=cfg.luminance_segment_threshold,
    )
    effective_threshold = adaptive_threshold

    # trigger estimate (kept from original approach)
    fin_frames, fin = collapse_segments(table)
    ep_frm, rem_frm = get_ep_and_remove_frames(fin_frames, fin, effective_threshold)
    triggers = possible_triggers(ep_frm, fps, cfg.min_hits_in_1s)
    rem_times = [f / fps for f in rem_frm]

    # NEW: precise cut intervals from table itself
    cut_intervals = table_to_cut_intervals(
        table=table,
        fps=fps,
        threshold=effective_threshold,
        min_run_frames=cfg.min_run_frames,
        pad_before=cfg.pad_before_sec,
        pad_after=cfg.pad_after_sec,
        merge_gap=cfg.merge_gap_sec,
        extend_level_seconds=cfg.extend_level_seconds,
    )

    return {
        "video_path": video_path,
        "fps": fps,
        "num_frames_processed": int(frames_rgb.shape[0]),
        "adaptive_threshold": adaptive_threshold,
        "effective_threshold": effective_threshold,
        "user_configured_threshold": cfg.luminance_segment_threshold,
        "threshold_abs_table": effective_threshold,
        "min_run_frames": cfg.min_run_frames,
        "extend_level_seconds": cfg.extend_level_seconds,
        "pad_before_sec": cfg.pad_before_sec,
        "pad_after_sec": cfg.pad_after_sec,
        "merge_gap_sec": cfg.merge_gap_sec,
        "max_cut_fraction": cfg.max_cut_fraction,
        "possible_triggers": int(triggers),
        "harmful_segment_end_frames": rem_frm,
        "harmful_segment_end_times_sec": rem_times,
        "cut_intervals_sec": [{"start": a, "end": b} for (a, b) in cut_intervals],
        "table": table,
    }


def write_outputs(result: Dict[str, Any], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    report = dict(result)
    table = report.pop("table", None)

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if table is None:
        raise RuntimeError("Missing table in result.")

    df = pd.DataFrame({"frame_index_delta": np.arange(len(table), dtype=int), "score": table})
    csv_path = os.path.join(out_dir, "events.csv")
    df.to_csv(csv_path, index=False)

    return report_path, csv_path


# -----------------------------
# Safe video creation
# -----------------------------
def sanitize_video(video_path: str, report_json_path: str, out_path: str) -> str:
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    cuts = report.get("cut_intervals_sec", [])
    max_cut_fraction = float(report.get("max_cut_fraction", 0.25))

    clip = VideoFileClip(video_path)

    # normalize + clamp cut intervals
    intervals: List[Tuple[float, float]] = []
    for seg in cuts:
        t1 = float(seg["start"])
        t2 = float(seg["end"])
        if t2 <= t1:
            continue
        if t1 >= clip.duration:
            continue
        t1 = max(0.0, t1)
        t2 = min(clip.duration, t2)
        intervals.append((t1, t2))

    intervals.sort()

    total_cut = sum((b - a) for a, b in intervals)
    cut_fraction = (total_cut / clip.duration) if clip.duration > 0 else 1.0

    print(f"Video duration: {clip.duration:.2f}s")
    print(f"Cut intervals ({len(intervals)}), total cut: {total_cut:.2f}s ({100*cut_fraction:.1f}%)")

    # Safety cap: keep only the longest cuts until we fit budget
    if intervals and cut_fraction > max_cut_fraction:
        budget = max_cut_fraction * clip.duration
        longest_first = sorted(intervals, key=lambda x: (x[1] - x[0]), reverse=True)

        chosen: List[Tuple[float, float]] = []
        used = 0.0
        for a, b in longest_first:
            dur = b - a
            if dur <= 0:
                continue
            if used + dur <= budget:
                chosen.append((a, b))
                used += dur
            if used >= budget:
                break

        chosen.sort()
        print(f"Auto-capped cuts to <= {100*max_cut_fraction:.0f}%: using {len(chosen)} intervals, {used:.2f}s total")
        intervals = chosen

    # build keep intervals (everything NOT in cuts)
    keep: List[Tuple[float, float]] = []
    cur = 0.0
    for t1, t2 in intervals:
        if t1 > cur:
            keep.append((cur, t1))
        cur = max(cur, t2)
    if cur < clip.duration:
        keep.append((cur, clip.duration))

    if not keep:
        clip.close()
        raise RuntimeError("All content was removed by cut intervals; nothing left to write.")

    # moviepy v2: subclipped
    kept_clips = [clip.subclipped(a, b) for (a, b) in keep if (b - a) > 0.05]
    final = concatenate_videoclips(kept_clips, method="compose")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    final.close()
    clip.close()
    return out_path
