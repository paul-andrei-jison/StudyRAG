"""Image cleanup for phone photos of printed pages.

Works on in-memory copies: callers pass bytes or a path, the original file is never written.
Every decision is recorded in CleanResult so a bad page can be explained later.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .orientation import LineProbe, detect_rotation, rotate_ccw

DETECT_EDGE = 1000          # page detection runs on a downscaled copy for speed and noise
MIN_PAGE_AREA = 0.25        # a detected page must cover at least this share of the frame
SKIP_FLATTEN_AREA = 0.97    # a page filling the frame is already cropped: do not warp
MAX_SKEW_DEG = 15.0
MIN_SKEW_DEG = 0.3          # smaller angles are noise, leave the image alone


@dataclass
class CleanResult:
    image: Image.Image                    # grayscale, ready for the READER / OCR
    original_size: tuple[int, int]        # after EXIF rotation, before any cleanup
    exif_rotated: bool = False
    page_found: bool = False
    skew_deg: float = 0.0                 # rotation applied to straighten text (degrees)
    glare_fraction: float = 0.0           # share of pixels blown out in the source
    steps: list[str] = field(default_factory=list)
    auto_rotation: int = 0                # degrees ccw applied by orientation detection (no EXIF tag)


def load_image(source: bytes | Path | str | Image.Image) -> tuple[Image.Image, bool]:
    """Open an image, apply EXIF rotation, return RGB and whether it was rotated."""
    if isinstance(source, Image.Image):  # e.g. a rendered PDF page: nothing to rotate
        return source.convert("RGB"), False
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else Path(source))
    img.load()
    orientation = img.getexif().get(0x0112, 1)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB"), orientation not in (0, 1)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)  # tl, tr, br, bl


def find_page_corners(bgr: np.ndarray) -> np.ndarray | None:
    """Return the page quadrilateral (tl, tr, br, bl) in full-image coordinates, or None."""
    h, w = bgr.shape[:2]
    scale = min(1.0, DETECT_EDGE / max(h, w))
    small = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)  # text/holes inside the page become solid
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area_share = cv2.contourArea(contour) / float(small.shape[0] * small.shape[1])
    if area_share < MIN_PAGE_AREA or area_share > SKIP_FLATTEN_AREA:
        return None
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32)) / scale
    return None


def flatten_page(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    m = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, m, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _profile_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST)
    return float(np.var(rotated.sum(axis=1, dtype=np.float64)))


def estimate_skew(gray: np.ndarray) -> float:
    """Angle (degrees, cv2 convention) that straightens text lines; 0 when unsure."""
    scale = min(1.0, 800 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    if binary.sum() == 0:
        return 0.0
    best = max(np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 0.01, 0.5),
               key=lambda a: _profile_score(binary, float(a)))
    fine = max(np.arange(best - 0.5, best + 0.51, 0.1), key=lambda a: _profile_score(binary, float(a)))
    return float(fine)


def rotate_keep_size(img: np.ndarray, angle: float, fill: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=fill)


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the paper background to remove shadows and gradients."""
    scale = min(1.0, 1000 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = max(3, int(small.shape[1] * 0.03) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, max(3, int(small.shape[1] * 0.06) | 1))
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def boost_contrast(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (0.5, 99.5))
    if hi - lo > 10:
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def clean_image(source: bytes | Path | str | Image.Image, max_edge: int = 2048,
                rotation_probe: LineProbe | None = None) -> CleanResult:
    """Clean a page photo. With a rotation_probe, pages lacking an EXIF rotation are auto-uprighted."""
    pil, rotated = load_image(source)
    auto = 0
    if rotation_probe is not None and not rotated:
        auto = detect_rotation(pil, rotation_probe)
        pil = rotate_ccw(pil, auto)
    result = CleanResult(image=pil, original_size=pil.size, exif_rotated=rotated, auto_rotation=auto)
    if rotated:
        result.steps.append("exif-rotate")
    if auto:
        result.steps.append(f"auto-rotate-{auto}")

    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    result.glare_fraction = float((gray0 >= 250).mean())

    corners = find_page_corners(bgr)
    if corners is not None:
        bgr = flatten_page(bgr, corners)
        result.page_found = True
        result.steps.append("flatten-page")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray = flatten_illumination(gray)
    result.steps.append("flatten-illumination")

    angle = estimate_skew(gray)
    if MIN_SKEW_DEG <= abs(angle) <= MAX_SKEW_DEG:
        gray = rotate_keep_size(gray, angle, fill=int(np.median(gray)))
        result.skew_deg = angle
        result.steps.append("deskew")

    gray = boost_contrast(gray)
    result.steps.append("contrast")

    out = Image.fromarray(gray)
    longest = max(out.size)
    if longest > max_edge:  # never upscale
        ratio = max_edge / longest
        out = out.resize((round(out.width * ratio), round(out.height * ratio)), Image.LANCZOS)
        result.steps.append(f"downscale-{max_edge}")
    result.image = out
    return result


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
