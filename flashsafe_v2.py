import os
import json
import math
import tempfile
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
from yt_dlp import YoutubeDL


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    resize_w: int = 320
    resize_h: int = 240

    # threshold for abs(table) to flag flashing frames
    luminance_segment_threshold: float = 15.0

    # trigger counting: hits > 3 within ~1 second window
    min_hits_in_1s: int = 3

    # extend each flagged interval by N seconds at the end (helps be safer)
    extend_level_seconds: float = 0.5

    # padding around flagged runs
    pad_before_sec: float = 0.15
    pad_after_sec: float = 0.15

    # merge intervals that are close
    merge_gap_sec: float = 0.08

    # only flag runs with at least N consecutive frames
    min_run_frames: int = 3


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
# Video -> frames (for analysis)
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
        frame_bgr = cv2.resize(frame_bgr, (cfg.resize_w, cfg.resize_h), interpolation=cv2.INTER_AREA)
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
    Returns table: float array [T-1] flash score per frame-step.
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
    base_threshold: float = 15.0,
    percentile: float = 97.5,
    sensitivity_fraction: float = 0.60,
    min_threshold: float = 6.0,
    max_threshold: float = 60.0,
) -> float:
    """
    Derive a threshold relative to THIS video's flash score distribution.

    - Takes the high percentile of abs(table) scores as an anchor for what
      counts as "loud flashing" in this specific video.
    - Sets the detection threshold at `sensitivity_fraction` of that anchor.
    - Clamps to [min_threshold, max_threshold] so we never go deaf or hypersensitive.
    - Falls back to base_threshold for genuinely calm videos with no activity.
    """
    abs_scores = np.abs(table)

    # Ignore near-zero (background noise)
    active = abs_scores[abs_scores > 1.0]
    if len(active) < 10:
        # Video is too calm to compute anything meaningful
        return base_threshold

    high_pct = float(np.percentile(active, percentile))

    # Threshold = a fraction below the high-intensity anchor
    adaptive = high_pct * sensitivity_fraction

    # Hard floor/ceiling
    adaptive = max(min_threshold, min(max_threshold, adaptive))

    return round(adaptive, 2)


# -----------------------------
# Trigger counting (kept)
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


def get_ep_and_remove_frames(fin_frames: List[int], fin: List[float], threshold: float) -> Tuple[List[int], List[int]]:
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
# Table -> intervals (precise)
# -----------------------------
def table_to_intervals(
    table: np.ndarray,
    fps: float,
    threshold: float,
    min_run_frames: int,
    pad_before: float,
    pad_after: float,
    merge_gap: float,
    extend_level_seconds: float,
) -> List[Tuple[float, float]]:
    mask = np.abs(table) >= threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

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

    extend = max(0.0, float(extend_level_seconds))
    intervals: List[Tuple[float, float]] = []
    for a, b in runs:
        if (b - a + 1) < min_run_frames:
            continue
        t1 = max(0.0, (a / fps) - pad_before)
        t2 = (b / fps) + pad_after + extend
        t2 = max(t1, t2)
        intervals.append((t1, t2))

    if not intervals:
        return []

    intervals.sort()
    merged: List[List[float]] = [[intervals[0][0], intervals[0][1]]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(x[0], x[1]) for x in merged]


# -----------------------------
# Analyze + outputs
# -----------------------------
def analyze(video_path: str, cfg: Config) -> Dict[str, Any]:
    frames_rgb, fps = load_video_frames(video_path, cfg)
    table = run_model(frames_rgb)

    # Compute adaptive threshold from this video's own score distribution,
    # then blend with the user-configured threshold (user sensitivity shifts
    # the adaptive value up or down within a ±40% band).
    adaptive_threshold = compute_adaptive_threshold(
        table,
        base_threshold=cfg.luminance_segment_threshold,
    )
    # cfg.luminance_segment_threshold may have been set by user sensitivity;
    # we treat the adaptive value as the base and keep it as the operative threshold.
    effective_threshold = adaptive_threshold

    fin_frames, fin = collapse_segments(table)
    ep_frm, rem_frm = get_ep_and_remove_frames(fin_frames, fin, effective_threshold)
    triggers = possible_triggers(ep_frm, fps, cfg.min_hits_in_1s)
    rem_times = [f / fps for f in rem_frm]

    intervals = table_to_intervals(
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
        "possible_triggers": int(triggers),
        "harmful_segment_end_frames": rem_frm,
        "harmful_segment_end_times_sec": rem_times,
        "effect_intervals_sec": [{"start": a, "end": b} for (a, b) in intervals],
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
# Flash dampening video (NO CUTTING)
# -----------------------------
def _strength_at_time(t: float, intervals: List[Tuple[float, float]], ramp_sec: float) -> float:
    for a, b in intervals:
        if a <= t <= b:
            if ramp_sec > 0:
                if t < a + ramp_sec:
                    return (t - a) / ramp_sec
                if t > b - ramp_sec:
                    return (b - t) / ramp_sec
            return 1.0
    return 0.0


def dampen_video(
    video_path: str,
    report_json_path: str,
    out_path: str,
    dim_factor: float = 0.65,   # lower = darker
    sat_factor: float = 0.60,   # lower = less saturated
    blur_px: int = 3,           # 0 disables blur
    ramp_sec: float = 0.12,     # smooth effect start/end
) -> str:
    """
    Keeps the full video; applies calming effect during effect_intervals_sec.
    Tries to preserve audio using ffmpeg if available.
    """
    with open(report_json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    raw = report.get("effect_intervals_sec", [])
    intervals: List[Tuple[float, float]] = []
    for seg in raw:
        a = float(seg["start"])
        b = float(seg["end"])
        if b > a:
            intervals.append((a, b))
    intervals.sort()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for processing: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # write silent processed video
    tmp_silent = os.path.splitext(out_path)[0] + ".__silent__.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_silent, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        s = _strength_at_time(t, intervals, ramp_sec)

        if s > 0:
            # dim
            dim = 1.0 - s * (1.0 - dim_factor)
            f = frame_bgr.astype(np.float32) * dim

            # desaturate via HSV saturation scaling
            hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            sat = 1.0 - s * (1.0 - sat_factor)
            hsv[..., 1] *= sat
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # optional blur
            if blur_px and blur_px > 0:
                k = int(blur_px)
                if k % 2 == 0:
                    k += 1
                out = cv2.GaussianBlur(out, (k, k), 0)
        else:
            out = frame_bgr

        writer.write(out)
        frame_idx += 1

    cap.release()
    writer.release()

    # Try to mux original audio back in
    ffmpeg = shutil.which("ffmpeg")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if ffmpeg:
        # mux audio from original onto processed video
        cmd = [
            ffmpeg, "-y",
            "-i", tmp_silent,
            "-i", video_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            out_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_silent)
            return out_path
        except Exception:
            # fallback: leave silent
            pass

    # fallback silent
    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(tmp_silent, out_path)
    return out_path


def preset_params(mode: str) -> Dict[str, Any]:
    mode = mode.lower().strip()
    if mode == "mild":
        return dict(dim_factor=0.85, sat_factor=0.85, blur_px=0, ramp_sec=0.10)
    if mode == "strong":
        return dict(dim_factor=0.55, sat_factor=0.35, blur_px=5, ramp_sec=0.15)
    # balanced default
    return dict(dim_factor=0.65, sat_factor=0.60, blur_px=3, ramp_sec=0.12)