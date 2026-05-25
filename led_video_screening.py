# -*- coding: utf-8 -*-
"""
Video-first screening tool for urban LED display screens.

This script implements the video-derived screening layer only. It does not
calculate calibrated illuminance, luminance, mEDI, MDER, BLER, or health risk.

Subcommands
-----------
1. annotate
   Manually annotate target-screen, other-screen, and excluded polygons.

2. extract
   Extract video-derived proxy indicators from annotated videos.

3. screen
   Convert extracted indicators into percentile-based screening flags,
   profile suggestions, and management suggestions.

Typical workflow
----------------
python led_video_screening.py annotate --video-dir "F:/videos" --annotation-dir "./annotations"
python led_video_screening.py extract --video-dir "F:/videos" --annotation-dir "./annotations" --out-dir "./outputs"
python led_video_screening.py screen --indicator-csv "./outputs/video_indicator_summary.csv" --out-dir "./outputs"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Dashboard generation falls back to OpenCV text.
    Image = ImageDraw = ImageFont = None


# =============================================================================
# Simple PyCharm settings
# =============================================================================
# If you run this script directly in PyCharm without command-line parameters,
# the script will use the settings below.
#
# SIMPLE_VIDEO_PATH can be either:
#   1. a video folder, such as r".\example_videos"
#   2. one video file, such as r".\example_videos\example.mp4"
#
# SIMPLE_MODE:
#   "annotate"              = only draw/save ROI polygons
#   "extract"               = extract indicators from existing annotations
#   "screen"                = create screening flags from extracted indicators
#   "annotate_extract_screen" = annotate first, then extract and screen
#
# For the first run, use "annotate_extract_screen" or "annotate".
# For later reruns after annotations already exist, use "extract" or "screen".
SIMPLE_VIDEO_PATH = r".\example_videos"
SIMPLE_MODE = "annotate_extract_screen"
# "auto" keeps the original aspect ratio and scales the annotation window so
# the whole frame fits on screen. You can also use a number, such as 0.5 or 0.8.
SIMPLE_DISPLAY_SCALE = "auto"
SIMPLE_DISPLAY_MAX_WIDTH = 1600
SIMPLE_DISPLAY_MAX_HEIGHT = 900
SIMPLE_FRAME_INTERVAL_SEC = 3.0
# Study-compatible extraction uses approximately 0.5-second sampling.
SIMPLE_SAMPLE_INTERVAL_SEC = 0.5
SIMPLE_RANDOM_SEED = 20260420
SIMPLE_OVERWRITE_ANNOTATION = False
# Use study-derived reference thresholds from the 85-video dataset. This makes
# single-video screening meaningful. Set to False if you want thresholds to be
# calculated from the currently processed batch.
SIMPLE_USE_STUDY_REFERENCE_THRESHOLDS = True
# Keep this as "none" unless the OpenCV frame is visibly rotated incorrectly.
# Your earlier extraction had the correct direction, so the default is no rotation.
SIMPLE_ROTATE_DEGREES = "none"
# Background ROI: first use a 120-pixel screen-expansion ring; if the ring has
# too few valid pixels, use the non-screen part of the frame instead.
SIMPLE_BACKGROUND_RING_MAX_PX = 120
SIMPLE_MIN_BACKGROUND_RING_PIXELS = 5000
SIMPLE_LOCAL_BG_TARGET_AREA_THRESHOLD = 0.30
SIMPLE_SCREEN_DARK_RGB_THRESHOLD = 50
SIMPLE_FILTER_BACKGROUND_BY_SCREEN_SIGNATURE = True
SIMPLE_SCREEN_SIGNATURE_BIN_SIZE = 0.01
SIMPLE_SCREEN_SIGNATURE_MIN_PROP = 0.005
SIMPLE_SCREEN_SIGNATURE_CUM_PROP = 0.85
SIMPLE_SCREEN_SIGNATURE_NEIGHBOR_RADIUS = 1
SIMPLE_SCREEN_SIGNATURE_SAMPLE_MAX = 8000


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

ROLE_KEYS = {
    ord("1"): "target_screen",
    ord("2"): "other_screen",
    ord("3"): "exclude",
}

ROLE_COLOR = {
    "target_screen": (0, 165, 255),
    "other_screen": (255, 180, 60),
    "exclude": (140, 140, 140),
}

ROLE_LABEL = {
    "target_screen": "target_screen",
    "other_screen": "other_screen",
    "exclude": "exclude",
}


# -----------------------------
# General helpers
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_videos(video_dir: Path) -> list[Path]:
    if video_dir.is_file() and video_dir.suffix.lower() in VIDEO_EXTS:
        return [video_dir]
    return sorted(
        [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda x: x.name.lower(),
    )


def resolve_video_path(video_input: Path, video_file: str, annotation_stem: str) -> Path | None:
    if video_input.is_file():
        if video_input.exists() and video_input.stem == annotation_stem:
            return video_input
        return None

    candidate = video_input / video_file
    if candidate.exists():
        return candidate

    matches = [
        p for p in video_input.iterdir()
        if p.is_file() and p.stem == annotation_stem and p.suffix.lower() in VIDEO_EXTS
    ]
    return matches[0] if matches else None


def read_rotation_degrees(cap: cv2.VideoCapture, requested: Any) -> int:
    """Return clockwise rotation degrees for frames read by OpenCV."""
    if isinstance(requested, str) and requested.lower() == "auto":
        try:
            meta = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        except Exception:
            meta = 0
        if not np.isfinite(meta):
            return 0
        # OpenCV reports orientation metadata as clockwise degrees for many
        # phone videos. Unsupported backends usually return 0.
        deg = int(round(float(meta))) % 360
        return deg if deg in {0, 90, 180, 270} else 0

    if isinstance(requested, str):
        requested = requested.lower().strip()
        aliases = {
            "none": 0,
            "no": 0,
            "clockwise": 90,
            "cw": 90,
            "counterclockwise": 270,
            "ccw": 270,
        }
        if requested in aliases:
            return aliases[requested]
    try:
        deg = int(requested) % 360
    except Exception:
        deg = 0
    return deg if deg in {0, 90, 180, 270} else 0


def rotate_frame_clockwise(frame: np.ndarray, degrees: int) -> np.ndarray:
    degrees = int(degrees) % 360
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int, rotate_degrees: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return rotate_frame_clockwise(frame, rotate_degrees)


def annotation_display_size(width: int, height: int, args: argparse.Namespace) -> tuple[int, int, float]:
    requested = getattr(args, "display_scale", "auto")
    if isinstance(requested, str) and requested.lower().strip() == "auto":
        max_w = int(getattr(args, "display_max_width", 1600))
        max_h = int(getattr(args, "display_max_height", 900))
        scale = min(1.0, max_w / max(width, 1), max_h / max(height, 1))
    else:
        scale = float(requested)
    scale = max(scale, 0.05)
    aw = max(1, int(round(width * scale)))
    ah = max(1, int(round(height * scale)))
    return aw, ah, scale


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_image(path: Path, image: np.ndarray) -> bool:
    """Write image files safely to Windows paths containing non-ASCII text."""
    ensure_dir(path.parent)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def safe_file_stem(text: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in str(text))
    return out.strip().strip(".") or "output"


def safe_divide(a: float, b: float, eps: float = 1e-9) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < eps:
        return np.nan
    return float(a / b)


def polygon_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    arr = np.asarray(poly, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convert_poly_to_original(
    poly: list[list[float]],
    annotation_width: int,
    annotation_height: int,
    original_width: int,
    original_height: int,
) -> list[list[float]]:
    sx = original_width / annotation_width
    sy = original_height / annotation_height
    return [[float(x) * sx, float(y) * sy] for x, y in poly]


def clip_poly(poly: list[list[float]], width: int, height: int) -> list[list[int]]:
    out = []
    for x, y in poly:
        xi = int(round(min(max(x, 0), width - 1)))
        yi = int(round(min(max(y, 0), height - 1)))
        out.append([xi, yi])
    return out


def make_mask(shape_hw: tuple[int, int], polygons: list[list[list[float]]]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3 or polygon_area(poly) < 1:
            continue
        pts = np.asarray(clip_poly(poly, w, h), dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    return mask > 0


def expand_screen_to_background_ring(
    screen_mask: np.ndarray,
    exclude_mask: np.ndarray,
    lower_px: int = 30,
    upper_px: int = 120,
    ratio: float = 0.15,
) -> np.ndarray:
    ys, xs = np.where(screen_mask)
    if len(xs) == 0:
        return np.zeros_like(screen_mask, dtype=bool)

    k = int(upper_px)
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(screen_mask.astype(np.uint8), kernel, iterations=1) > 0
    ring = dilated & (~screen_mask) & (~exclude_mask)
    return ring


def screen_dark_mask(rgb: np.ndarray, threshold: int) -> np.ndarray:
    return (
        (rgb[..., 0] < threshold)
        & (rgb[..., 1] < threshold)
        & (rgb[..., 2] < threshold)
    )


def xy_bin_pairs_from_pixels(rgb_pixels: np.ndarray, bin_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = rgb_to_xy(rgb_pixels)
    valid = np.isfinite(x) & np.isfinite(y)
    valid &= x > 1e-6
    valid &= y > 1e-6
    valid &= x >= 0.0
    valid &= x <= 0.8
    valid &= y >= 0.0
    valid &= y <= 0.9

    x_bin = np.full(len(rgb_pixels), -999999, dtype=np.int32)
    y_bin = np.full(len(rgb_pixels), -999999, dtype=np.int32)
    x_bin[valid] = np.floor(x[valid] / bin_size).astype(np.int32)
    y_bin[valid] = np.floor(y[valid] / bin_size).astype(np.int32)
    return x_bin, y_bin, valid


def build_screen_signature_bins(
    frame_rgb: np.ndarray,
    screen_mask: np.ndarray,
    threshold: int,
    bin_size: float,
    min_prop: float,
    cum_prop: float,
    neighbor_radius: int,
    sample_max: int,
) -> set[tuple[int, int]]:
    pixels = frame_rgb[screen_mask]
    if pixels.size == 0:
        return set()

    keep = ~(
        (pixels[:, 0] < threshold)
        & (pixels[:, 1] < threshold)
        & (pixels[:, 2] < threshold)
    )
    pixels = pixels[keep]
    if len(pixels) == 0:
        return set()
    if len(pixels) > sample_max:
        pixels = pixels[:sample_max]

    x_bin, y_bin, valid = xy_bin_pairs_from_pixels(pixels, bin_size)
    pairs = list(zip(x_bin[valid], y_bin[valid]))
    if not pairs:
        return set()

    counts: dict[tuple[int, int], int] = {}
    for pair in pairs:
        counts[pair] = counts.get(pair, 0) + 1

    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    selected: set[tuple[int, int]] = set()
    cumulative = 0
    for pair, count in ranked:
        prop = count / total
        if prop >= min_prop:
            selected.add(pair)
        if cumulative / total < cum_prop:
            selected.add(pair)
            cumulative += count

    expanded: set[tuple[int, int]] = set()
    for xb, yb in selected:
        for dx in range(-neighbor_radius, neighbor_radius + 1):
            for dy in range(-neighbor_radius, neighbor_radius + 1):
                expanded.add((xb + dx, yb + dy))
    return expanded


def filter_mask_by_screen_signature(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    signature_bins: set[tuple[int, int]],
    bin_size: float,
) -> np.ndarray:
    if not signature_bins or int(mask.sum()) == 0:
        return mask
    ys, xs = np.where(mask)
    pixels = frame_rgb[ys, xs, :]
    x_bin, y_bin, valid = xy_bin_pairs_from_pixels(pixels, bin_size)
    keep = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        if valid[i] and (int(x_bin[i]), int(y_bin[i])) in signature_bins:
            keep[i] = False
    out = np.zeros(mask.shape, dtype=bool)
    out[ys[keep], xs[keep]] = True
    return out


def sample_values(arr: np.ndarray, mask: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    vals = arr[mask]
    if vals.size == 0:
        return vals
    if vals.shape[0] <= max_points:
        return vals
    idx = rng.choice(vals.shape[0], size=max_points, replace=False)
    return vals[idx]


# -----------------------------
# Color and image metrics
# -----------------------------


def luminance_proxy_yprime(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def blue_channel_proportion(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    denom = rgb[..., 0] + rgb[..., 1] + rgb[..., 2]
    out = np.full(denom.shape, np.nan, dtype=np.float32)
    valid = denom > 1.0
    out[valid] = rgb[..., 2][valid] / denom[valid]
    return out


def srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    srgb = rgb_u8.astype(np.float32) / 255.0
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def rgb_to_xy(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lin = srgb_to_linear(rgb_u8)
    r = lin[..., 0]
    g = lin[..., 1]
    b = lin[..., 2]

    x_tr = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y_tr = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z_tr = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    denom = x_tr + y_tr + z_tr

    x = np.full(denom.shape, np.nan, dtype=np.float32)
    y = np.full(denom.shape, np.nan, dtype=np.float32)
    valid = denom > 1e-9
    x[valid] = x_tr[valid] / denom[valid]
    y[valid] = y_tr[valid] / denom[valid]
    return x, y


def chromaticity_metrics(
    x: np.ndarray,
    y: np.ndarray,
    bin_size: float = 0.005,
    luminance_proxy: np.ndarray | None = None,
    luminance_min: float = 10.0,
) -> dict[str, float]:
    # Match the original chromaticity-frequency workflow:
    # valid CIE x-y range, no origin points, and low-luminance pixels removed.
    valid = np.isfinite(x) & np.isfinite(y) & (x > 1e-6) & (y > 1e-6) & (x <= 0.8) & (y <= 0.9)
    if luminance_proxy is not None:
        valid = valid & np.isfinite(luminance_proxy) & (luminance_proxy >= luminance_min)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return {
            "chromaticity_50_occupancy": np.nan,
            "chromaticity_90_occupancy": np.nan,
            "chromaticity_50_bins": np.nan,
            "chromaticity_90_bins": np.nan,
            "chromaticity_entropy": np.nan,
            "chromaticity_x_mean": np.nan,
            "chromaticity_y_mean": np.nan,
        }

    xb = np.floor(x / bin_size).astype(int)
    yb = np.floor(y / bin_size).astype(int)
    keys = xb.astype(np.int64) * 100000 + yb.astype(np.int64)
    _, counts = np.unique(keys, return_counts=True)
    counts = np.sort(counts)[::-1]
    total = counts.sum()
    probs = counts / total
    cum = np.cumsum(probs)

    n50 = int(np.searchsorted(cum, 0.50) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    # Occupancy is reported as bin area on the CIE x-y grid, scaled by 1000
    # for readability, matching the manuscript tables.
    occupancy_scale = bin_size * bin_size * 1000.0
    return {
        "chromaticity_50_occupancy": float(n50 * occupancy_scale),
        "chromaticity_90_occupancy": float(n90 * occupancy_scale),
        "chromaticity_50_bins": float(n50),
        "chromaticity_90_bins": float(n90),
        "chromaticity_entropy": entropy,
        "chromaticity_x_mean": float(np.mean(x)),
        "chromaticity_y_mean": float(np.mean(y)),
    }


def saturation_proxy(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    out = np.zeros(mx.shape, dtype=np.float32)
    valid = mx > 1e-9
    out[valid] = (mx[valid] - mn[valid]) / mx[valid]
    return out


# -----------------------------
# Annotation GUI
# -----------------------------


class AnnotationState:
    def __init__(self) -> None:
        self.current_points: list[list[int]] = []
        self.polygons: dict[str, list[list[list[int]]]] = {
            "target_screen": [],
            "other_screen": [],
            "exclude": [],
        }
        self.current_role = "target_screen"
        self.img_show: np.ndarray | None = None

    def reset_frame(self) -> None:
        self.current_points = []
        self.polygons = {
            "target_screen": [],
            "other_screen": [],
            "exclude": [],
        }
        self.current_role = "target_screen"

    def commit_polygon(self) -> None:
        if len(self.current_points) >= 3:
            self.polygons[self.current_role].append([p.copy() for p in self.current_points])
            self.current_points = []


def annotate_videos(args: argparse.Namespace) -> None:
    video_dir = Path(args.video_dir)
    annotation_dir = Path(args.annotation_dir)
    ensure_dir(annotation_dir)

    videos = list_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found: {video_dir}")

    print(f"[INFO] Found {len(videos)} videos")
    print("[INFO] Left click: add point | Enter/Space: confirm polygon")
    print("[INFO] 1 target_screen | 2 other_screen | 3 exclude")
    print("[INFO] N: save frame and next | Q: save video and next | ESC: stop")

    for video_path in videos:
        out_path = annotation_dir / f"{video_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] Existing annotation: {out_path}")
            save_annotation_check_overlays(video_path, out_path, annotation_dir / "annotation_check_overlays")
            continue
        result = annotate_one_video(video_path, args)
        if result is None:
            print("[STOP] User interrupted")
            break
        write_json(out_path, result)
        save_annotation_check_overlays(video_path, out_path, annotation_dir / "annotation_check_overlays")
        print(f"[SAVE] {out_path} | annotated_frames={len(result['frames'])}")


def annotate_one_video(video_path: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return {
            "video_file": video_path.name,
            "screen_name": video_path.stem,
            "frames": [],
            "error": "cannot_open_video",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0 or not np.isfinite(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    rotate_degrees = read_rotation_degrees(cap, getattr(args, "rotate_degrees", "none"))

    step = max(1, int(round(fps * args.frame_interval_sec)))
    frame_indices = list(range(0, max(total_frames, 1), step))
    if not frame_indices:
        frame_indices = [0]

    state = AnnotationState()
    win = "LED screen polygon annotator"

    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.current_points.append([int(x), int(y)])

    def draw_overlay() -> np.ndarray:
        if state.img_show is None:
            raise RuntimeError("No display image loaded.")
        img = state.img_show.copy()
        for role, polys in state.polygons.items():
            color = ROLE_COLOR[role]
            for poly in polys:
                arr = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [arr], True, color, 2, cv2.LINE_AA)
                overlay = img.copy()
                cv2.fillPoly(overlay, [arr], color)
                img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

        if state.current_points:
            color = ROLE_COLOR[state.current_role]
            for p in state.current_points:
                cv2.circle(img, tuple(p), 4, (0, 255, 255), -1, cv2.LINE_AA)
            for i in range(1, len(state.current_points)):
                cv2.line(img, tuple(state.current_points[i - 1]), tuple(state.current_points[i]), color, 2, cv2.LINE_AA)
            if len(state.current_points) >= 3:
                cv2.line(img, tuple(state.current_points[-1]), tuple(state.current_points[0]), color, 2, cv2.LINE_AA)

        lines = [
            f"Role: {ROLE_LABEL[state.current_role]} | 1 target | 2 other | 3 exclude",
            "Left click: add point | Enter/Space: confirm polygon | Z: undo point | X: undo polygon",
            "N: save frame and next | Q: save video and next | ESC: exit",
        ]
        y = 28
        for line in lines:
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            y += 28
        return img

    def save_frame(frame_idx: int, ann_frames: list[dict[str, Any]]) -> None:
        state.commit_polygon()
        if not any(len(v) for v in state.polygons.values()):
            return
        ann_frames.append(
            {
                "frame_idx": int(frame_idx),
                "target_screen": state.polygons["target_screen"],
                "other_screen": state.polygons["other_screen"],
                "exclude": state.polygons["exclude"],
            }
        )

    # Use AUTOSIZE so the displayed annotation image is not stretched by a
    # manually resized OpenCV window. A stretched window would make rectangular
    # videos appear square and would make mouse coordinates misleading.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse_callback)

    ann_frames: list[dict[str, Any]] = []
    annotation_width = None
    annotation_height = None
    analysis_width = None
    analysis_height = None
    display_scale_value = None

    print(f"\n[VIDEO] {video_path.name}")
    print(f"[VIDEO_META] raw={raw_width}x{raw_height}, rotate_clockwise={rotate_degrees}")
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_bgr = rotate_frame_clockwise(frame_bgr, rotate_degrees)

        state.reset_frame()
        h, w = frame_bgr.shape[:2]
        analysis_width = w
        analysis_height = h
        aw, ah, actual_display_scale = annotation_display_size(w, h, args)
        display_scale_value = actual_display_scale
        annotation_width = aw
        annotation_height = ah
        state.img_show = cv2.resize(frame_bgr, (aw, ah), interpolation=cv2.INTER_AREA)
        print(f"[FRAME] analysis={w}x{h}, annotation_display={aw}x{ah}, display_scale={actual_display_scale:.4f}")

        while True:
            vis = draw_overlay()
            cv2.setWindowTitle(
                win,
                f"{video_path.name} | frame {frame_idx}/{max(total_frames - 1, 0)} | {i + 1}/{len(frame_indices)}",
            )
            cv2.imshow(win, vis)
            k = cv2.waitKey(20) & 0xFF

            if k in ROLE_KEYS:
                state.commit_polygon()
                state.current_role = ROLE_KEYS[k]
            elif k in (13, 32):
                state.commit_polygon()
            elif k in (ord("z"), ord("Z")):
                if state.current_points:
                    state.current_points.pop()
            elif k in (ord("x"), ord("X")):
                role = state.current_role
                if state.polygons[role]:
                    state.polygons[role].pop()
            elif k in (ord("n"), ord("N")):
                save_frame(frame_idx, ann_frames)
                break
            elif k in (ord("q"), ord("Q")):
                save_frame(frame_idx, ann_frames)
                cap.release()
                cv2.destroyWindow(win)
                return {
                    "video_file": video_path.name,
                    "screen_name": video_path.stem,
                    "fps": fps,
                    "total_frames": total_frames,
                    "raw_width": raw_width,
                    "raw_height": raw_height,
                    "rotate_degrees_clockwise": rotate_degrees,
                    "original_width": int(analysis_width or 0),
                    "original_height": int(analysis_height or 0),
                    "annotation_width": int(annotation_width or 0),
                    "annotation_height": int(annotation_height or 0),
                    "display_scale": actual_display_scale,
                    "coordinate_system": "annotation_display",
                    "frames": ann_frames,
                }
            elif k == 27:
                cap.release()
                cv2.destroyWindow(win)
                return None

    cap.release()
    cv2.destroyWindow(win)
    return {
        "video_file": video_path.name,
        "screen_name": video_path.stem,
        "fps": fps,
        "total_frames": total_frames,
        "raw_width": raw_width,
        "raw_height": raw_height,
        "rotate_degrees_clockwise": rotate_degrees,
        "original_width": int(analysis_width or raw_width),
        "original_height": int(analysis_height or raw_height),
        "annotation_width": int(annotation_width or (analysis_width or raw_width)),
        "annotation_height": int(annotation_height or (analysis_height or raw_height)),
        "display_scale": display_scale_value,
        "coordinate_system": "annotation_display",
        "frames": ann_frames,
    }


# -----------------------------
# Annotation interpolation
# -----------------------------


@dataclass
class FramePolygons:
    target_screen: list[list[list[float]]]
    other_screen: list[list[list[float]]]
    exclude: list[list[list[float]]]


def normalize_annotation_to_original(ann: dict[str, Any]) -> dict[str, Any]:
    original_width = int(ann.get("original_width") or 0)
    original_height = int(ann.get("original_height") or 0)
    annotation_width = int(ann.get("annotation_width") or original_width)
    annotation_height = int(ann.get("annotation_height") or original_height)

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Annotation JSON lacks valid original_width/original_height.")
    if annotation_width <= 0 or annotation_height <= 0:
        raise ValueError("Annotation JSON lacks valid annotation_width/annotation_height.")

    coord = ann.get("coordinate_system", "annotation_display")
    out = dict(ann)
    frames = []
    for fr in ann.get("frames", []):
        item = {"frame_idx": int(fr["frame_idx"])}
        for role in ("target_screen", "other_screen", "exclude"):
            polys = fr.get(role, []) or []
            if coord == "original_video":
                item[role] = [[list(map(float, p)) for p in poly] for poly in polys]
            else:
                item[role] = [
                    convert_poly_to_original(poly, annotation_width, annotation_height, original_width, original_height)
                    for poly in polys
                ]
        frames.append(item)
    frames = sorted(frames, key=lambda x: x["frame_idx"])
    out["frames_original"] = frames
    return out


def compatible_polygons(a: list[list[list[float]]], b: list[list[list[float]]]) -> bool:
    if len(a) != len(b):
        return False
    for pa, pb in zip(a, b):
        if len(pa) != len(pb):
            return False
    return True


def interpolate_polys(
    a: list[list[list[float]]],
    b: list[list[list[float]]],
    t: float,
) -> list[list[list[float]]]:
    if not compatible_polygons(a, b):
        return a if t < 0.5 else b
    out = []
    for pa, pb in zip(a, b):
        arr_a = np.asarray(pa, dtype=float)
        arr_b = np.asarray(pb, dtype=float)
        arr = arr_a * (1.0 - t) + arr_b * t
        out.append(arr.tolist())
    return out


def polygons_for_frame(ann_norm: dict[str, Any], frame_idx: int) -> FramePolygons:
    frames = ann_norm["frames_original"]
    if not frames:
        return FramePolygons([], [], [])
    if frame_idx <= frames[0]["frame_idx"]:
        fr = frames[0]
        return FramePolygons(fr["target_screen"], fr["other_screen"], fr["exclude"])
    if frame_idx >= frames[-1]["frame_idx"]:
        fr = frames[-1]
        return FramePolygons(fr["target_screen"], fr["other_screen"], fr["exclude"])

    left = frames[0]
    right = frames[-1]
    for i in range(len(frames) - 1):
        if frames[i]["frame_idx"] <= frame_idx <= frames[i + 1]["frame_idx"]:
            left = frames[i]
            right = frames[i + 1]
            break

    denom = max(1, right["frame_idx"] - left["frame_idx"])
    t = (frame_idx - left["frame_idx"]) / denom
    return FramePolygons(
        target_screen=interpolate_polys(left["target_screen"], right["target_screen"], t),
        other_screen=interpolate_polys(left["other_screen"], right["other_screen"], t),
        exclude=interpolate_polys(left["exclude"], right["exclude"], t),
    )


def save_annotation_check_overlays(video_path: Path, ann_path: Path, out_dir: Path) -> None:
    """Save overlay images to verify whether JSON polygon coordinates match frames."""
    try:
        ann_raw = read_json(ann_path)
        ann = normalize_annotation_to_original(ann_raw)
    except Exception as exc:
        print(f"[WARN] Cannot read annotation for overlay check: {ann_path.name} | {exc}")
        return

    frames = ann.get("frames_original", [])
    if not frames:
        print(f"[WARN] No annotated frames for overlay check: {ann_path.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video for overlay check: {video_path}")
        return

    ensure_dir(out_dir)
    rotate_degrees = int(ann.get("rotate_degrees_clockwise", 0) or 0)
    saved = 0
    for fr in frames:
        frame_idx = int(fr["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = rotate_frame_clockwise(frame, rotate_degrees)
        overlay = frame.copy()

        for role in ("target_screen", "other_screen", "exclude"):
            color = ROLE_COLOR[role]
            for poly in fr.get(role, []) or []:
                if len(poly) < 3:
                    continue
                pts = np.asarray(clip_poly(poly, overlay.shape[1], overlay.shape[0]), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, color, 4, cv2.LINE_AA)
                fill = overlay.copy()
                cv2.fillPoly(fill, [pts], color)
                overlay = cv2.addWeighted(fill, 0.18, overlay, 0.82, 0)

        label = f"{video_path.name} | frame {frame_idx}"
        cv2.putText(overlay, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overlay, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = out_dir / f"{video_path.stem}_frame_{frame_idx:06d}_overlay.jpg"
        if write_image(out_path, overlay):
            saved += 1

    cap.release()
    print(f"[CHECK] annotation overlays saved: {saved} -> {out_dir}")


# -----------------------------
# Extraction
# -----------------------------


def extract_batch(args: argparse.Namespace) -> None:
    video_dir = Path(args.video_dir)
    annotation_dir = Path(args.annotation_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "pixel_samples")
    ensure_dir(out_dir / "quality_overlays")

    annotations = sorted(annotation_dir.glob("*.json"))
    if not annotations:
        raise FileNotFoundError(f"No annotation JSON files found: {annotation_dir}")

    all_summary = []
    all_quality = []
    for ann_path in annotations:
        ann = read_json(ann_path)
        video_file = ann.get("video_file") or f"{ann_path.stem}.mp4"
        video_path = resolve_video_path(video_dir, video_file, ann_path.stem)
        if video_path is None:
            print(f"[WARN] Missing video for annotation: {ann_path.name}")
            continue

        print(f"[RUN] {video_path.name}")
        summary, quality = extract_one_video(video_path, ann_path, out_dir, args)
        all_summary.append(summary)
        all_quality.append(quality)

    if all_summary:
        pd.DataFrame(all_summary).to_csv(out_dir / "video_indicator_summary.csv", index=False, encoding="utf-8-sig")
    if all_quality:
        pd.DataFrame(all_quality).to_csv(out_dir / "roi_quality_summary.csv", index=False, encoding="utf-8-sig")

    print("[DONE] Extraction finished")
    print("[OUTPUT]", out_dir)


def extract_one_video(
    video_path: Path,
    ann_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ann = normalize_annotation_to_original(read_json(ann_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or ann.get("fps") or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or ann.get("total_frames") or 0)
    rotate_degrees = int(ann.get("rotate_degrees_clockwise", read_rotation_degrees(cap, getattr(args, "rotate_degrees", "none"))))
    step = max(1, int(round(fps * args.sample_interval_sec)))
    frame_indices = list(range(0, max(total_frames, 1), step))

    rng = np.random.default_rng(args.random_seed)

    screen_y_series = []
    bg_y_series = []
    non_screen_y_series = []
    ratio_series = []
    diff_series = []
    frame_diff_series = []
    sampled_pixel_change_series = []
    blue_series = []
    sat_series = []
    valid_frame_indices = []
    screen_area_series = []
    bg_area_series = []
    saturation_pixel_ratio_series = []

    pixel_rows = []
    prev_sampled_screen_y_map = None
    prev_sampled_screen_mask = None
    prev_screen_mean = None
    overlay_saved = False

    for frame_idx in frame_indices:
        frame_bgr = read_frame_at(cap, frame_idx, rotate_degrees)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        yprime = luminance_proxy_yprime(frame_rgb)
        blue = blue_channel_proportion(frame_rgb)
        sat = saturation_proxy(frame_rgb)
        chroma_x, chroma_y = rgb_to_xy(frame_rgb)

        polys = polygons_for_frame(ann, frame_idx)
        target_mask = make_mask((height, width), polys.target_screen)
        other_mask = make_mask((height, width), polys.other_screen)
        exclude_mask = make_mask((height, width), polys.exclude) | other_mask
        screen_mask = target_mask & (~exclude_mask)
        valid_full_mask = ~exclude_mask
        non_screen_mask = valid_full_mask & (~target_mask)

        valid_pixels = max(1, int(valid_full_mask.sum()))
        target_area_ratio = float(screen_mask.sum() / valid_pixels)
        if target_area_ratio >= float(getattr(args, "local_bg_target_area_threshold", 0.30)):
            bg_mask = non_screen_mask
        else:
            bg_mask = expand_screen_to_background_ring(
                screen_mask,
                exclude_mask,
                upper_px=int(getattr(args, "background_ring_max_px", 120)),
            )
            if int(bg_mask.sum()) < int(getattr(args, "min_background_ring_pixels", args.min_background_pixels)):
                bg_mask = non_screen_mask

        dark = screen_dark_mask(frame_rgb, int(getattr(args, "screen_dark_rgb_threshold", 50)))
        screen_stat_mask = screen_mask & (~dark)

        if bool(getattr(args, "filter_background_by_screen_signature", True)):
            signature_bins = build_screen_signature_bins(
                frame_rgb,
                screen_mask,
                int(getattr(args, "screen_dark_rgb_threshold", 50)),
                float(getattr(args, "screen_signature_bin_size", 0.01)),
                float(getattr(args, "screen_signature_min_prop", 0.005)),
                float(getattr(args, "screen_signature_cum_prop", 0.85)),
                int(getattr(args, "screen_signature_neighbor_radius", 1)),
                int(getattr(args, "screen_signature_sample_max", 8000)),
            )
            bg_mask = filter_mask_by_screen_signature(
                frame_rgb,
                bg_mask,
                signature_bins,
                float(getattr(args, "screen_signature_bin_size", 0.01)),
            )
            non_screen_mask = filter_mask_by_screen_signature(
                frame_rgb,
                non_screen_mask,
                signature_bins,
                float(getattr(args, "screen_signature_bin_size", 0.01)),
            )

        screen_n = int(screen_stat_mask.sum())
        bg_n = int(bg_mask.sum())
        if screen_n < args.min_screen_pixels:
            continue

        screen_y = yprime[screen_stat_mask]
        bg_y = yprime[bg_mask] if bg_n > 0 else np.asarray([], dtype=float)
        non_screen_y = yprime[non_screen_mask] if int(non_screen_mask.sum()) > 0 else np.asarray([], dtype=float)

        screen_mean = float(np.mean(screen_y))
        bg_mean = float(np.mean(bg_y)) if bg_y.size else np.nan
        non_screen_mean = float(np.mean(non_screen_y)) if non_screen_y.size else np.nan
        ratio = safe_divide(screen_mean, bg_mean, eps=1.0)
        diff = screen_mean - bg_mean if np.isfinite(bg_mean) else np.nan

        screen_y_series.append(screen_mean)
        bg_y_series.append(bg_mean)
        non_screen_y_series.append(non_screen_mean)
        ratio_series.append(ratio)
        diff_series.append(diff)
        valid_frame_indices.append(frame_idx)
        screen_area_series.append(screen_n)
        bg_area_series.append(bg_n)

        screen_blue = blue[screen_stat_mask]
        screen_sat = sat[screen_stat_mask]
        blue_series.append(float(np.nanmean(screen_blue)))
        sat_series.append(float(np.nanmean(screen_sat)))
        saturation_pixel_ratio_series.append(float(np.mean(np.max(frame_rgb[screen_stat_mask], axis=1) >= args.saturation_u8)))

        # Match the revised manuscript/intermediate-table definition:
        # frame difference is the absolute change in screen ROI mean luminance
        # between adjacent sampled time points, not native-frame pixel change.
        if prev_screen_mean is not None:
            frame_diff_series.append(abs(screen_mean - prev_screen_mean))
        prev_screen_mean = screen_mean

        if prev_sampled_screen_y_map is not None and prev_sampled_screen_mask is not None:
            common = screen_stat_mask & prev_sampled_screen_mask
            if int(common.sum()) >= args.min_screen_pixels:
                sampled_change = float(np.mean(np.abs(yprime[common] - prev_sampled_screen_y_map[common])))
                sampled_pixel_change_series.append(sampled_change)
        prev_sampled_screen_y_map = yprime
        prev_sampled_screen_mask = screen_stat_mask

        if args.export_pixel_samples:
            add_pixel_sample_rows(
                pixel_rows,
                video_path.stem,
                frame_idx,
                "screen",
                frame_rgb,
                yprime,
                blue,
                sat,
                chroma_x,
                chroma_y,
                screen_stat_mask,
                args.max_pixels_per_region_frame,
                rng,
            )
            add_pixel_sample_rows(
                pixel_rows,
                video_path.stem,
                frame_idx,
                "background",
                frame_rgb,
                yprime,
                blue,
                sat,
                chroma_x,
                chroma_y,
                bg_mask,
                args.max_pixels_per_region_frame,
                rng,
            )
            add_pixel_sample_rows(
                pixel_rows,
                video_path.stem,
                frame_idx,
                "non_screen",
                frame_rgb,
                yprime,
                blue,
                sat,
                chroma_x,
                chroma_y,
                non_screen_mask,
                args.max_pixels_per_region_frame,
                rng,
            )

        if args.save_quality_overlay and not overlay_saved:
            save_overlay(out_dir / "quality_overlays" / f"{video_path.stem}_overlay.jpg", frame_bgr, screen_mask, bg_mask, exclude_mask)
            overlay_saved = True

    cap.release()

    if not valid_frame_indices:
        summary = {
            "video_id": video_path.stem,
            "video_file": video_path.name,
            "valid_frames": 0,
            "error": "no_valid_frames",
        }
        quality = dict(summary)
        return summary, quality

    screen_y_arr = np.asarray(screen_y_series, dtype=float)
    bg_y_arr = np.asarray(bg_y_series, dtype=float)
    ratio_arr = np.asarray(ratio_series, dtype=float)
    diff_arr = np.asarray(diff_series, dtype=float)
    fd_arr = np.asarray(frame_diff_series, dtype=float)
    sampled_pixel_change_arr = np.asarray(sampled_pixel_change_series, dtype=float)
    blue_arr = np.asarray(blue_series, dtype=float)
    sat_arr = np.asarray(sat_series, dtype=float)

    if args.export_pixel_samples and pixel_rows:
        pixel_df = pd.DataFrame(pixel_rows)
        pixel_df.to_csv(out_dir / "pixel_samples" / f"{video_path.stem}_pixel_samples.csv", index=False, encoding="utf-8-sig")
        screen_pixels = pixel_df[pixel_df["region_type"] == "screen"]
        chroma = chromaticity_metrics(
            screen_pixels["x"].to_numpy(),
            screen_pixels["y"].to_numpy(),
            args.chromaticity_bin_size,
            screen_pixels["luminance_proxy"].to_numpy(),
        )
        screen_luminance_p95 = float(screen_pixels["luminance_proxy"].quantile(0.95)) if not screen_pixels.empty else np.nan
    else:
        screen_luminance_p95 = float(np.nanpercentile(screen_y_arr, 95))
        chroma = {
            "chromaticity_50_occupancy": np.nan,
            "chromaticity_90_occupancy": np.nan,
            "chromaticity_50_bins": np.nan,
            "chromaticity_90_bins": np.nan,
            "chromaticity_entropy": np.nan,
            "chromaticity_x_mean": np.nan,
            "chromaticity_y_mean": np.nan,
        }

    summary = {
        "video_id": video_path.stem,
        "video_file": video_path.name,
        "valid_frames": len(valid_frame_indices),
        "fps": fps,
        "sample_interval_sec": args.sample_interval_sec,
        "screen_luminance_mean": float(np.nanmean(screen_y_arr)),
        "screen_luminance_p95": screen_luminance_p95,
        "screen_luminance_CV": safe_divide(float(np.nanstd(screen_y_arr, ddof=1)), float(np.nanmean(screen_y_arr)), eps=1.0),
        "background_luminance_mean": float(np.nanmean(bg_y_arr)),
        "screen_background_luminance_difference_mean": float(np.nanmean(diff_arr)),
        "screen_background_luminance_ratio_mean": float(np.nanmean(ratio_arr)),
        "screen_background_luminance_ratio_CV": safe_divide(float(np.nanstd(ratio_arr, ddof=1)), float(np.nanmean(ratio_arr)), eps=1e-6),
        "screen_frame_diff_mean": float(np.nanmean(fd_arr)) if fd_arr.size else np.nan,
        "screen_frame_diff_p95": float(np.nanpercentile(fd_arr, 95)) if fd_arr.size else np.nan,
        "screen_sampled_pixel_change_mean": float(np.nanmean(sampled_pixel_change_arr)) if sampled_pixel_change_arr.size else np.nan,
        "screen_sampled_pixel_change_p95": float(np.nanpercentile(sampled_pixel_change_arr, 95)) if sampled_pixel_change_arr.size else np.nan,
        "screen_blue_ratio_mean": float(np.nanmean(blue_arr)),
        "screen_blue_ratio_p95": float(np.nanpercentile(blue_arr, 95)),
        "screen_saturation_mean": float(np.nanmean(sat_arr)),
        "screen_saturation_p95": float(np.nanpercentile(sat_arr, 95)),
        **chroma,
    }

    quality = {
        "video_id": video_path.stem,
        "video_file": video_path.name,
        "valid_frames": len(valid_frame_indices),
        "median_screen_pixels": float(np.median(screen_area_series)),
        "median_background_pixels": float(np.median(bg_area_series)),
        "min_screen_pixels": int(np.min(screen_area_series)),
        "min_background_pixels": int(np.min(bg_area_series)),
        "mean_saturated_pixel_ratio_screen": float(np.nanmean(saturation_pixel_ratio_series)),
        "annotation_json": str(ann_path),
        "coordinate_system": ann.get("coordinate_system", "annotation_display"),
        "display_scale": ann.get("display_scale", np.nan),
    }
    return summary, quality


def add_pixel_sample_rows(
    rows: list[dict[str, Any]],
    video_id: str,
    frame_idx: int,
    region_type: str,
    rgb: np.ndarray,
    yprime: np.ndarray,
    blue: np.ndarray,
    saturation: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    if len(xs) > max_points:
        idx = rng.choice(len(xs), size=max_points, replace=False)
        xs = xs[idx]
        ys = ys[idx]
    pix = rgb[ys, xs, :]
    for i in range(len(xs)):
        rows.append(
            {
                "video_id": video_id,
                "frame_idx": int(frame_idx),
                "region_type": region_type,
                "pixel_x": int(xs[i]),
                "pixel_y": int(ys[i]),
                "R": int(pix[i, 0]),
                "G": int(pix[i, 1]),
                "B": int(pix[i, 2]),
                "luminance_proxy": float(yprime[ys[i], xs[i]]),
                "blue_ratio": float(blue[ys[i], xs[i]]) if np.isfinite(blue[ys[i], xs[i]]) else np.nan,
                "saturation": float(saturation[ys[i], xs[i]]),
                "x": float(x[ys[i], xs[i]]) if np.isfinite(x[ys[i], xs[i]]) else np.nan,
                "y": float(y[ys[i], xs[i]]) if np.isfinite(y[ys[i], xs[i]]) else np.nan,
            }
        )


def save_overlay(path: Path, frame_bgr: np.ndarray, screen_mask: np.ndarray, bg_mask: np.ndarray, exclude_mask: np.ndarray) -> None:
    overlay = frame_bgr.copy()
    color = np.zeros_like(frame_bgr)
    color[screen_mask] = (0, 165, 255)
    color[bg_mask] = (0, 255, 0)
    color[exclude_mask] = (120, 120, 120)
    overlay = cv2.addWeighted(color, 0.35, overlay, 0.65, 0)
    write_image(path, overlay)


# -----------------------------
# Screening and profile diagnosis
# -----------------------------


PROFILE_SUGGESTIONS = {
    "exposure_intensity": "Review peak/mean screen luminance proxy, high-luminance content share, visible area, and local background balance.",
    "temporal_variability": "Reduce abrupt transitions, rapid scene changes, and unstable screen-background contrast dynamics.",
    "blue_channel_tendency": "Reduce blue-rich or cold-color content during nighttime periods; consider optional spectral validation in high-priority cases.",
    "screen_background_contrast": "Coordinate screen content and surrounding facade/shopfront lighting to reduce contrast-related visual salience.",
    "chromatic_spread": "Control highly saturated, broadly distributed, or rapidly changing color content.",
    "integrated_priority": "Prioritize field review; optional calibrated spectral measurement can be used for selected high-priority or disputed cases.",
}


# Study-derived upper-quartile thresholds from the 85-video dataset.
# These are relative video-derived proxy thresholds, not regulatory standards.
STUDY_REFERENCE_THRESHOLDS = {
    "exposure_intensity": {
        "indicator": "screen_luminance_p95",
        "threshold": 238.8601,
        "basis": "85-video upper quartile",
    },
    "temporal_variability": {
        "indicator": "screen_frame_diff_mean",
        "threshold": 11.4383335754886,
        "secondary_indicators": {
            "screen_luminance_CV": 0.240627969794186,
            "screen_background_luminance_ratio_CV": 0.463211381335869,
        },
        "basis": "85-video upper quartile",
    },
    "blue_channel_tendency": {
        "indicator": "screen_blue_ratio_mean",
        "threshold": 0.397443148517498,
        "basis": "85-video upper quartile",
    },
    "screen_background_contrast": {
        "indicator": "screen_background_luminance_difference_mean",
        "threshold": 87.913524417509,
        "secondary_indicator": "screen_background_luminance_ratio_mean",
        "secondary_threshold": 3.96420008635695,
        "basis": "85-video upper quartile",
    },
    "chromatic_spread": {
        "indicator": "chromaticity_90_occupancy",
        "threshold": 46.0062472110662,
        "basis": "85-video upper quartile",
    },
}


def screen_indicators(args: argparse.Namespace) -> None:
    indicator_csv = Path(args.indicator_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(indicator_csv)
    if df.empty:
        raise ValueError("Indicator CSV is empty.")

    q = float(args.quantile)
    metric_map = {
        "exposure_intensity": "screen_luminance_p95",
        "temporal_variability": "screen_frame_diff_mean",
        "blue_channel_tendency": "screen_blue_ratio_mean",
        "screen_background_contrast": "screen_background_luminance_difference_mean",
        "chromatic_spread": "chromaticity_90_occupancy",
    }

    available = {k: v for k, v in metric_map.items() if v in df.columns}
    if not available:
        raise ValueError("No expected screening indicators were found in the input CSV.")

    out = df.copy()
    thresholds = {}
    threshold_rows = []
    active = {}
    use_reference = bool(getattr(args, "use_reference_thresholds", False))
    for dim, col in available.items():
        reference = STUDY_REFERENCE_THRESHOLDS.get(dim)
        if use_reference and reference is not None and reference["indicator"] == col:
            active[dim] = col
            threshold = float(reference["threshold"])
            thresholds[dim] = threshold
            if dim == "temporal_variability" and reference.get("secondary_indicators"):
                comparisons = [out[col] / threshold]
                flags = out[col] >= threshold
                indicator_parts = [col]
                threshold_parts = [str(threshold)]
                for sec_col, sec_threshold in reference["secondary_indicators"].items():
                    if sec_col in out.columns:
                        sec_threshold = float(sec_threshold)
                        comparisons.append(out[sec_col] / sec_threshold)
                        flags = flags | (out[sec_col] >= sec_threshold)
                        indicator_parts.append(sec_col)
                        threshold_parts.append(str(sec_threshold))
                out[f"flag_{dim}"] = flags
                out[f"score_{dim}"] = pd.concat(comparisons, axis=1).max(axis=1)
            elif dim == "screen_background_contrast" and reference.get("secondary_indicator") in out.columns:
                sec_col = str(reference["secondary_indicator"])
                sec_threshold = float(reference["secondary_threshold"])
                out[f"flag_{dim}"] = (out[col] >= threshold) | (out[sec_col] >= sec_threshold)
                out[f"score_{dim}"] = pd.concat(
                    [out[col] / threshold, out[sec_col] / sec_threshold],
                    axis=1,
                ).max(axis=1)
            else:
                out[f"flag_{dim}"] = out[col] >= threshold
                out[f"score_{dim}"] = out[col] / threshold
            threshold_rows.append(
                {
                    "dimension": dim,
                    "indicator": " OR ".join(indicator_parts) if dim == "temporal_variability" else (col if "secondary_indicator" not in reference else f"{col} OR {reference['secondary_indicator']}"),
                    "threshold_type": reference["basis"],
                    "quantile": q,
                    "threshold": " OR ".join(threshold_parts) if dim == "temporal_variability" else (threshold if "secondary_threshold" not in reference else f"{threshold} OR {reference['secondary_threshold']}"),
                }
            )
        elif use_reference:
            continue
        else:
            active[dim] = col
            threshold = float(out[col].quantile(q))
            thresholds[dim] = threshold
            out[f"flag_{dim}"] = out[col] >= threshold
            out[f"score_{dim}"] = percentile_rank(out[col])
            threshold_rows.append(
                {
                    "dimension": dim,
                    "indicator": col,
                    "threshold_type": "current-batch quantile",
                    "quantile": q,
                    "threshold": threshold,
                }
            )

    if not active:
        raise ValueError("No active screening indicators were available.")

    score_cols = [f"score_{dim}" for dim in active]
    flag_cols = [f"flag_{dim}" for dim in active]
    out["integrated_video_screening_score"] = out[score_cols].mean(axis=1, skipna=True)
    out["n_dimension_flags"] = out[flag_cols].sum(axis=1)
    if use_reference:
        integrated_threshold = np.nan
        out["flag_integrated_priority"] = out["n_dimension_flags"] >= 2
        threshold_rows.append(
            {
                "dimension": "integrated_priority",
                "indicator": "n_dimension_flags",
                "threshold_type": "rule based on 85-video reference flags",
                "quantile": np.nan,
                "threshold": 2,
            }
        )
    else:
        integrated_threshold = float(out["integrated_video_screening_score"].quantile(q))
        out["flag_integrated_priority"] = out["integrated_video_screening_score"] >= integrated_threshold
        threshold_rows.append(
            {
                "dimension": "integrated_priority",
                "indicator": "integrated_video_screening_score",
                "threshold_type": "current-batch quantile",
                "quantile": q,
                "threshold": integrated_threshold,
            }
        )
    out["dominant_screening_dimension"] = out.apply(lambda r: dominant_dimension(r, active), axis=1)
    out["screening_profile"] = out.apply(profile_label, axis=1)
    out["management_suggestion"] = out.apply(management_suggestion, axis=1)

    out.to_csv(out_dir / "screening_results.csv", index=False, encoding="utf-8-sig")

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(out_dir / "screening_thresholds.csv", index=False, encoding="utf-8-sig")
    create_dashboard_images(out, threshold_df, out_dir)

    print("[DONE] Screening finished")
    print("[OUTPUT]", out_dir / "screening_results.csv")
    print("[OUTPUT]", out_dir / "screening_thresholds.csv")


def create_dashboard_images(results: pd.DataFrame, thresholds: pd.DataFrame, out_dir: Path) -> None:
    quality_path = out_dir / "roi_quality_summary.csv"
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    threshold_map = {
        str(r["dimension"]): r
        for _, r in thresholds.iterrows()
        if str(r.get("indicator", "")) not in {"n_dimension_flags", "integrated_video_screening_score"}
    }

    dashboard_dir = out_dir / "dashboards"
    ensure_dir(dashboard_dir)
    for _, row in results.iterrows():
        video_id = str(row.get("video_id", "video"))
        qrow = pd.Series(dtype=object)
        if not quality.empty and "video_id" in quality.columns:
            sub = quality[quality["video_id"].astype(str) == video_id]
            if not sub.empty:
                qrow = sub.iloc[0]
        img = build_single_dashboard(row, qrow, threshold_map)
        out_path = dashboard_dir / f"{safe_file_stem(video_id)}_screening_dashboard.png"
        write_image(out_path, img)
    print(f"[OUTPUT] dashboards -> {dashboard_dir}")


def build_single_dashboard(row: pd.Series, qrow: pd.Series, threshold_map: dict[str, pd.Series]) -> np.ndarray:
    w, h = 1600, 1180

    if Image is None:
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Pillow is not installed; dashboard rendering is unavailable.", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        return img

    def load_font(size: int, bold: bool = False):
        candidates = [
            r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        ]
        for font_path in candidates:
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    font_title = load_font(42, True)
    font_h1 = load_font(28, True)
    font_h2 = load_font(23, True)
    font_body = load_font(21)
    font_small = load_font(18)
    font_tiny = load_font(15)
    font_badge = load_font(26, True)

    bg = (247, 249, 250)
    card = (255, 255, 255)
    ink = (31, 41, 55)
    muted = (101, 116, 139)
    line = (226, 232, 240)
    green = (46, 125, 82)
    red = (211, 69, 51)
    orange = (230, 145, 35)
    light_bar = (235, 239, 243)

    im = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(im)

    def text(x: int, y: int, s: Any, font=font_body, fill=ink):
        draw.text((x, y), str(s), font=font, fill=fill)

    def rounded_rect(box, radius=18, fill=card, outline=None, width=1):
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def wrap_words(s: str, max_chars: int) -> list[str]:
        words = str(s).split()
        lines, cur = [], ""
        for word in words:
            nxt = (cur + " " + word).strip()
            if len(nxt) > max_chars and cur:
                lines.append(cur)
                cur = word
            else:
                cur = nxt
        if cur:
            lines.append(cur)
        return lines

    def safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return np.nan

    def fmt(value: Any, digits: int = 3) -> str:
        value = safe_float(value)
        return f"{value:.{digits}f}" if np.isfinite(value) else "NA"

    video_label = row.get("video_file", row.get("video_id", ""))
    profile = str(row.get("screening_profile", ""))
    n_flags = int(row.get("n_dimension_flags", 0)) if pd.notna(row.get("n_dimension_flags", np.nan)) else 0
    priority = "HIGH PRIORITY" if bool(row.get("flag_integrated_priority", False)) else "ROUTINE REVIEW"
    priority_color = red if priority == "HIGH PRIORITY" else green

    # Header
    draw.rectangle((0, 0, w, 118), fill=(241, 245, 249))
    text(44, 30, "LED Display Screen Video-First Screening", font_title, ink)
    text(46, 82, f"Video: {video_label}", font_small, muted)
    rounded_rect((1150, 28, 1538, 88), radius=12, fill=priority_color)
    text(1210, 43, priority, font_badge, (255, 255, 255))

    # Summary card
    rounded_rect((40, 145, 1560, 285), radius=18, fill=card, outline=line)
    text(68, 168, "Profile summary", font_h1, ink)
    text(70, 208, f"Profile: {profile}", font_body, ink)
    text(70, 242, f"Elevated dimensions: {n_flags}", font_body, ink)
    if not qrow.empty:
        q_text = (
            f"Valid frames: {qrow.get('valid_frames', '')}    "
            f"Median screen pixels: {fmt(qrow.get('median_screen_pixels', np.nan), 1)}    "
            f"Saturated-pixel ratio: {fmt(qrow.get('mean_saturated_pixel_ratio_screen', np.nan), 3)}"
        )
        text(570, 242, q_text, font_small, muted)

    metrics = [
        ("exposure_intensity", "High luminance", "screen_luminance_p95", "flag_exposure_intensity"),
        ("temporal_variability", "Temporal variability", "screen_frame_diff_mean", "flag_temporal_variability"),
        ("blue_channel_tendency", "Blue-channel tendency", "screen_blue_ratio_mean", "flag_blue_channel_tendency"),
        ("screen_background_contrast", "Screen-background contrast", "screen_background_luminance_difference_mean", "flag_screen_background_contrast"),
        ("chromatic_spread", "Chromatic spread", "chromaticity_90_occupancy", "flag_chromatic_spread"),
    ]

    text(48, 325, "Indicators compared with 85-video reference thresholds", font_h1, ink)
    y = 370
    for dim, label, col, flag_col in metrics:
        if col not in row.index or dim not in threshold_map:
            continue
        val = safe_float(row[col])
        if dim in {"temporal_variability", "screen_background_contrast"}:
            thr = float(STUDY_REFERENCE_THRESHOLDS[dim]["threshold"])
        else:
            thr = safe_float(threshold_map[dim]["threshold"])
        ratio = val / thr if np.isfinite(val) and thr else np.nan
        is_flag = bool(row.get(flag_col, False))
        color = red if is_flag else green
        status = "FLAG" if is_flag else "OK"

        rounded_rect((40, y, 1560, y + 88), radius=14, fill=card, outline=line)
        text(68, y + 20, label, font_h2, ink)

        if dim == "temporal_variability":
            lum_cv = safe_float(row.get("screen_luminance_CV", np.nan))
            ratio_cv = safe_float(row.get("screen_background_luminance_ratio_CV", np.nan))
            detail = f"FD={fmt(val)}/{fmt(thr)}; lum CV={fmt(lum_cv)}/0.241; ratio CV={fmt(ratio_cv)}/0.463"
            ratio = max(
                ratio if np.isfinite(ratio) else 0,
                lum_cv / 0.240627969794186 if np.isfinite(lum_cv) else 0,
                ratio_cv / 0.463211381335869 if np.isfinite(ratio_cv) else 0,
            )
        elif dim == "screen_background_contrast":
            ratio_val = safe_float(row.get("screen_background_luminance_ratio_mean", np.nan))
            ratio_thr = float(STUDY_REFERENCE_THRESHOLDS[dim]["secondary_threshold"])
            detail = f"diff={fmt(val)}/{fmt(thr)}; ratio={fmt(ratio_val)}/{fmt(ratio_thr)}"
            ratio = max(ratio if np.isfinite(ratio) else 0, ratio_val / ratio_thr if np.isfinite(ratio_val) else 0)
        else:
            detail = f"value={fmt(val)}; reference={fmt(thr)}; ratio={fmt(ratio, 2)}"

        text(430, y + 22, detail, font_tiny if dim in {"temporal_variability", "screen_background_contrast"} else font_small, muted)
        bar_x, bar_y, bar_w, bar_h = 1015, y + 30, 395, 18
        rounded_rect((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), radius=8, fill=light_bar)
        fill_w = int(min(bar_w, max(0.0, ratio if np.isfinite(ratio) else 0.0) * bar_w))
        rounded_rect((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h), radius=8, fill=color)
        draw.line((bar_x + bar_w, bar_y - 8, bar_x + bar_w, bar_y + bar_h + 8), fill=orange, width=3)
        text(1440, y + 20, status, font_h2, color)
        y += 102

    # Suggestions card
    rounded_rect((40, 902, 1560, 1087), radius=18, fill=card, outline=line)
    text(68, 925, "Management suggestions", font_h1, ink)
    suggestions = str(row.get("management_suggestion", ""))
    sy = 970
    for line_text in wrap_words(suggestions, 120)[:4]:
        text(72, sy, "- " + line_text, font_small, ink)
        sy += 28

    rounded_rect((40, 1110, 1560, 1150), radius=10, fill=(241, 245, 249), outline=None)
    text(
        62,
        1120,
        "Note: These are video-derived screening proxies, not calibrated photometric, spectral, or health-risk thresholds.",
        font_tiny,
        muted,
    )

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def percentile_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def dominant_dimension(row: pd.Series, available: dict[str, str]) -> str:
    best_dim = None
    best_score = -np.inf
    for dim in available:
        score = row.get(f"score_{dim}", np.nan)
        if np.isfinite(score) and score > best_score:
            best_dim = dim
            best_score = score
    return best_dim or "undetermined"


def profile_label(row: pd.Series) -> str:
    if bool(row.get("flag_integrated_priority", False)) and int(row.get("n_dimension_flags", 0)) >= 2:
        return "integrated high-priority profile"
    dim = row.get("dominant_screening_dimension", "undetermined")
    return dim.replace("_", "-") + " dominant profile"


def management_suggestion(row: pd.Series) -> str:
    suggestions = []
    for dim, text in PROFILE_SUGGESTIONS.items():
        flag_name = f"flag_{dim}" if dim != "integrated_priority" else "flag_integrated_priority"
        if bool(row.get(flag_name, False)):
            suggestions.append(text)
    if not suggestions:
        dim = row.get("dominant_screening_dimension", "undetermined")
        suggestions.append(PROFILE_SUGGESTIONS.get(dim, "No high-priority flag; routine review is sufficient."))
    return " ".join(suggestions)


# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video-first screening tool for urban LED display screen light-pollution exposure proxies."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_annotate = sub.add_parser("annotate", help="Manually annotate video ROIs.")
    p_annotate.add_argument("--video-dir", required=True, help="Directory containing video files.")
    p_annotate.add_argument("--annotation-dir", required=True, help="Output directory for annotation JSON files.")
    p_annotate.add_argument("--frame-interval-sec", type=float, default=3.0, help="Interval between annotation keyframes.")
    p_annotate.add_argument("--display-scale", default="auto", help='Display scale used in annotation GUI, such as "auto", 0.5, or 0.8.')
    p_annotate.add_argument("--display-max-width", type=int, default=1600, help="Maximum annotation-window width when --display-scale is auto.")
    p_annotate.add_argument("--display-max-height", type=int, default=900, help="Maximum annotation-window height when --display-scale is auto.")
    p_annotate.add_argument("--rotate-degrees", default="none", help='Clockwise rotation: "none", "auto", 0, 90, 180, or 270.')
    p_annotate.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation JSON files.")
    p_annotate.set_defaults(func=annotate_videos)

    p_extract = sub.add_parser("extract", help="Extract video-derived proxy indicators from annotated videos.")
    p_extract.add_argument("--video-dir", required=True, help="Directory containing video files.")
    p_extract.add_argument("--annotation-dir", required=True, help="Directory containing annotation JSON files.")
    p_extract.add_argument("--out-dir", required=True, help="Output directory.")
    p_extract.add_argument("--sample-interval-sec", type=float, default=0.5, help="Video sampling interval for extraction.")
    p_extract.add_argument("--rotate-degrees", default="none", help='Clockwise rotation if annotation has no stored value: "none", "auto", 0, 90, 180, or 270.')
    p_extract.add_argument("--min-screen-pixels", type=int, default=500, help="Minimum valid target-screen pixels per sampled frame.")
    p_extract.add_argument("--min-background-pixels", type=int, default=500, help="Minimum valid background pixels per sampled frame.")
    p_extract.add_argument("--background-ring-max-px", type=int, default=120, help="Maximum screen-expansion distance for adjacent background ROI.")
    p_extract.add_argument("--min-background-ring-pixels", type=int, default=5000, help="Fallback to non-screen area when the background ring has fewer pixels.")
    p_extract.add_argument("--local-bg-target-area-threshold", type=float, default=0.30, help="Use non-screen background when target screen occupies at least this fraction of valid pixels.")
    p_extract.add_argument("--screen-dark-rgb-threshold", type=int, default=50, help="Remove target-screen pixels when all RGB channels are below this threshold.")
    p_extract.add_argument("--filter-background-by-screen-signature", action="store_true", default=True, help="Remove background pixels matching high-frequency screen chromaticity bins.")
    p_extract.add_argument("--no-filter-background-by-screen-signature", dest="filter_background_by_screen_signature", action="store_false", help="Disable screen-signature background filtering.")
    p_extract.add_argument("--screen-signature-bin-size", type=float, default=0.01, help="CIE x-y bin size for screen-signature filtering.")
    p_extract.add_argument("--screen-signature-min-prop", type=float, default=0.005, help="Minimum bin proportion for screen-signature selection.")
    p_extract.add_argument("--screen-signature-cum-prop", type=float, default=0.85, help="Cumulative high-frequency bin proportion for screen-signature selection.")
    p_extract.add_argument("--screen-signature-neighbor-radius", type=int, default=1, help="Neighbor radius around selected screen-signature bins.")
    p_extract.add_argument("--screen-signature-sample-max", type=int, default=8000, help="Maximum screen pixels used to build per-frame chromaticity signature.")
    p_extract.add_argument("--export-pixel-samples", action="store_true", help="Export sampled pixel-level data for chromaticity analysis.")
    p_extract.add_argument("--max-pixels-per-region-frame", type=int, default=2000, help="Maximum sampled pixels per region per frame.")
    p_extract.add_argument("--chromaticity-bin-size", type=float, default=0.005, help="CIE x-y bin size for chromaticity occupancy.")
    p_extract.add_argument("--saturation-u8", type=int, default=250, help="U8 channel threshold for saturated-pixel quality flag.")
    p_extract.add_argument("--save-quality-overlay", action="store_true", help="Save one ROI overlay image per video for quality checks.")
    p_extract.add_argument("--random-seed", type=int, default=20260420, help="Random seed for pixel sampling.")
    p_extract.set_defaults(func=extract_batch)

    p_screen = sub.add_parser("screen", help="Create screening flags and management suggestions from video indicators.")
    p_screen.add_argument("--indicator-csv", required=True, help="CSV generated by the extract subcommand.")
    p_screen.add_argument("--out-dir", required=True, help="Output directory.")
    p_screen.add_argument("--quantile", type=float, default=0.75, help="Percentile threshold for high-priority screening flags.")
    p_screen.add_argument("--use-reference-thresholds", action="store_true", help="Use study-derived 85-video reference thresholds instead of current-batch quantiles.")
    p_screen.set_defaults(func=screen_indicators)

    return parser


def run_simple_pycharm_workflow() -> None:
    video_input = Path(SIMPLE_VIDEO_PATH)
    base_dir = video_input.parent if video_input.is_file() else video_input
    annotation_dir = base_dir / "led_video_screening_annotations"
    out_dir = base_dir / "led_video_screening_results"

    print("[SIMPLE MODE]")
    print(f"[VIDEO_INPUT] {video_input}")
    print(f"[ANNOTATIONS] {annotation_dir}")
    print(f"[OUTPUTS] {out_dir}")
    print(f"[MODE] {SIMPLE_MODE}")

    if SIMPLE_MODE in {"annotate", "annotate_extract_screen"}:
        annotate_args = argparse.Namespace(
            video_dir=str(video_input),
            annotation_dir=str(annotation_dir),
            frame_interval_sec=float(SIMPLE_FRAME_INTERVAL_SEC),
            display_scale=SIMPLE_DISPLAY_SCALE,
            display_max_width=int(SIMPLE_DISPLAY_MAX_WIDTH),
            display_max_height=int(SIMPLE_DISPLAY_MAX_HEIGHT),
            rotate_degrees=SIMPLE_ROTATE_DEGREES,
            overwrite=bool(SIMPLE_OVERWRITE_ANNOTATION),
        )
        annotate_videos(annotate_args)

    if SIMPLE_MODE in {"extract", "annotate_extract_screen"}:
        extract_args = argparse.Namespace(
            video_dir=str(video_input),
            annotation_dir=str(annotation_dir),
            out_dir=str(out_dir),
            sample_interval_sec=float(SIMPLE_SAMPLE_INTERVAL_SEC),
            rotate_degrees=SIMPLE_ROTATE_DEGREES,
            min_screen_pixels=500,
            min_background_pixels=500,
            background_ring_max_px=int(SIMPLE_BACKGROUND_RING_MAX_PX),
            min_background_ring_pixels=int(SIMPLE_MIN_BACKGROUND_RING_PIXELS),
            local_bg_target_area_threshold=float(SIMPLE_LOCAL_BG_TARGET_AREA_THRESHOLD),
            screen_dark_rgb_threshold=int(SIMPLE_SCREEN_DARK_RGB_THRESHOLD),
            filter_background_by_screen_signature=bool(SIMPLE_FILTER_BACKGROUND_BY_SCREEN_SIGNATURE),
            screen_signature_bin_size=float(SIMPLE_SCREEN_SIGNATURE_BIN_SIZE),
            screen_signature_min_prop=float(SIMPLE_SCREEN_SIGNATURE_MIN_PROP),
            screen_signature_cum_prop=float(SIMPLE_SCREEN_SIGNATURE_CUM_PROP),
            screen_signature_neighbor_radius=int(SIMPLE_SCREEN_SIGNATURE_NEIGHBOR_RADIUS),
            screen_signature_sample_max=int(SIMPLE_SCREEN_SIGNATURE_SAMPLE_MAX),
            export_pixel_samples=True,
            max_pixels_per_region_frame=2000,
            chromaticity_bin_size=0.005,
            saturation_u8=250,
            save_quality_overlay=True,
            random_seed=int(SIMPLE_RANDOM_SEED),
        )
        extract_batch(extract_args)

    if SIMPLE_MODE in {"screen", "extract", "annotate_extract_screen"}:
        indicator_csv = out_dir / "video_indicator_summary.csv"
        if not indicator_csv.exists():
            print(f"[WARN] Indicator CSV not found, skip screening: {indicator_csv}")
            return
        screen_args = argparse.Namespace(
            indicator_csv=str(indicator_csv),
            out_dir=str(out_dir),
            quantile=0.75,
            use_reference_thresholds=bool(SIMPLE_USE_STUDY_REFERENCE_THRESHOLDS),
        )
        screen_indicators(screen_args)


def main() -> None:
    if len(sys.argv) == 1:
        run_simple_pycharm_workflow()
        return

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

