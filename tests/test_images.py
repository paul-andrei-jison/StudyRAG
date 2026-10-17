import hashlib
import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from studybuddy.reader.images import (
    clean_image, estimate_skew, find_page_corners, load_image, to_png_bytes,
)

LINES = [
    "Photosynthesis is the process by which green plants convert light energy",
    "into chemical energy. It takes place mainly in the chloroplasts of leaf",
    "cells, where chlorophyll absorbs sunlight and drives the reactions.",
] * 6


def make_page(w=1200, h=1700):
    img = Image.new("RGB", (w, h), (250, 248, 244))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    y = 120
    for line in LINES:
        d.text((90, y), line, fill=(20, 20, 20), font=font)
        y += 70
    return img


def to_bytes(img, fmt="PNG", **kw):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def photo_of(page, size=(1800, 2400), corners=((200, 150), (1650, 250), (1580, 2250), (130, 2150)),
             shadow=False):
    """Warp the page onto a dark table like a phone photo."""
    w, h = page.size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, np.float32(corners))
    arr = np.array(page)
    bg = np.full((size[1], size[0], 3), (55, 65, 75), np.uint8)
    warp = cv2.warpPerspective(arr, m, size, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, size)
    comp = np.where(mask[..., None] > 0, warp, bg).astype(np.float32)
    if shadow:
        comp *= np.linspace(0.45, 1.05, size[0])[None, :, None]
    return Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8))


def row_profile_variance(img):
    arr = np.array(img.convert("L"))
    binary = (arr < 128).astype(np.float64)
    return binary.sum(axis=1).var()


def test_exif_rotation_applied():
    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 cw to display
    data = to_bytes(img, "JPEG", exif=exif)
    rotated, flag = load_image(data)
    assert flag and rotated.size == (200, 400)
    assert clean_image(data).exif_rotated


def test_no_exif_not_flagged():
    _, flag = load_image(to_bytes(Image.new("RGB", (50, 50))))
    assert not flag


def test_deskew_straightens_text():
    page = make_page()
    tilted = page.rotate(4.0, resample=Image.BICUBIC, fillcolor=(250, 248, 244))
    result = clean_image(to_bytes(tilted), max_edge=4000)
    assert "deskew" in result.steps
    assert abs(abs(result.skew_deg) - 4.0) < 1.0
    assert row_profile_variance(result.image) > 1.5 * row_profile_variance(tilted)


def test_straight_page_is_not_rotated():
    result = clean_image(to_bytes(make_page()), max_edge=4000)
    assert "deskew" not in result.steps
    assert abs(result.skew_deg) < 0.3


def test_estimate_skew_blank_is_zero():
    assert estimate_skew(np.full((300, 300), 255, np.uint8)) == 0.0


def test_perspective_page_flattened():
    page = make_page()
    photo = photo_of(page)
    arr = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)
    assert find_page_corners(arr) is not None
    result = clean_image(to_bytes(photo), max_edge=4000)
    assert result.page_found and "flatten-page" in result.steps
    out = np.array(result.image)
    # dark table must be gone: border pixels are paper-bright, not table-dark
    border = np.concatenate([out[:5].ravel(), out[-5:].ravel(), out[:, :5].ravel(), out[:, -5:].ravel()])
    assert np.median(border) > 200
    # aspect ratio close to the real page (1200x1700), allowing for the photo's own distortion
    assert abs(result.image.width / result.image.height - 1200 / 1700) < 0.12


def test_shadow_gradient_reduced():
    page = make_page()
    shaded = np.array(page).astype(np.float32) * np.linspace(0.4, 1.0, page.width)[None, :, None]
    shaded_img = Image.fromarray(shaded.astype(np.uint8))
    result = clean_image(to_bytes(shaded_img), max_edge=4000)
    out = np.array(result.image)
    left, right = np.median(out[:, :200]), np.median(out[:, -200:])
    assert abs(float(left) - float(right)) < 15


def test_downscale_caps_long_edge_and_never_upscales():
    big = clean_image(to_bytes(make_page(2400, 3400)), max_edge=2048)
    assert max(big.image.size) == 2048 and any(s.startswith("downscale") for s in big.steps)
    small = clean_image(to_bytes(make_page(600, 850)), max_edge=2048)
    assert small.image.size == (600, 850)


def test_plain_page_without_background_reports_no_page():
    result = clean_image(to_bytes(make_page()), max_edge=4000)
    assert not result.page_found  # page fills the frame: nothing to flatten, no crash


def test_output_is_grayscale_and_png_encodes():
    result = clean_image(to_bytes(make_page()))
    assert result.image.mode == "L"
    assert to_png_bytes(result.image).startswith(b"\x89PNG")


def test_glare_fraction_recorded():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    assert clean_image(to_bytes(img)).glare_fraction > 0.9


def test_original_file_never_modified(tmp_path):
    p = tmp_path / "scan.png"
    make_page().save(p)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    clean_image(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert [f.name for f in tmp_path.iterdir()] == ["scan.png"]


def test_unreadable_image_raises():
    with pytest.raises(Exception):
        clean_image(b"not an image")
