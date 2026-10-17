import hashlib
import io

import pytest
from PIL import Image

from studybuddy.errors import PermanentIngestError
from studybuddy.ingest import files as files_mod
from studybuddy.ingest.files import (
    classify, document_hash, heif_supported, load_document, sha256_file, sniff, validate_image,
)
from test_pdf import make_scanned_pdf, make_text_pdf


def save(path, fmt, size=(120, 160), color=(200, 200, 200)):
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


@pytest.mark.parametrize("fmt,name,kind,ext", [
    ("JPEG", "a.jpg", "jpeg", ".jpg"), ("PNG", "a.png", "png", ".png"), ("WEBP", "a.webp", "webp", ".webp"),
    ("BMP", "a.bmp", "bmp", ".bmp"), ("TIFF", "a.tif", "tiff", ".tiff"),
])
def test_images_are_recognised_by_content(tmp_path, fmt, name, kind, ext):
    info = classify(save(tmp_path / name, fmt))
    assert (info.kind, info.ext, info.is_pdf) == (kind, ext, False)


def test_type_comes_from_bytes_not_from_the_name(tmp_path):
    disguised = tmp_path / "scan.jpg"
    make_scanned_pdf(disguised, 1)
    info = classify(disguised)
    assert info.kind == "pdf" and info.ext == ".pdf" and info.is_pdf
    png_named_jpg = save(tmp_path / "x.jpg", "PNG")
    assert classify(png_named_jpg).ext == ".png"


def test_pdf_is_recognised(tmp_path):
    make_text_pdf(tmp_path / "t.pdf", ["Enough text on this page to count"])
    assert classify(tmp_path / "t.pdf").kind == "pdf"


@pytest.mark.parametrize("data,match", [
    (b"", "empty"), (b"GIF89a....", "GIF"), (b"just some text, not an image", "unsupported or damaged"),
    (b"PK\x03\x04docx-or-zip", "unsupported or damaged"), (b"\x00" * 32, "unsupported or damaged"),
])
def test_unsupported_files_are_permanent_errors(tmp_path, data, match):
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    with pytest.raises(PermanentIngestError, match=match):
        classify(p)


def test_missing_file_is_a_permanent_error(tmp_path):
    with pytest.raises(PermanentIngestError, match="cannot be read"):
        classify(tmp_path / "gone.jpg")


def test_sniff_recognises_heic_brands():
    assert sniff(b"\x00\x00\x00\x1cftypheic\x00\x00\x00\x00") == "heic"
    assert sniff(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00") == "heic"
    assert sniff(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00") == "unknown"  # an MP4 video is not a HEIC photo


def test_heic_without_support_is_a_clear_permanent_error(tmp_path, monkeypatch):
    p = tmp_path / "IMG_0001.HEIC"
    p.write_bytes(b"\x00\x00\x00\x1cftypheic\x00\x00\x00\x00" + b"\x00" * 100)
    monkeypatch.setattr(files_mod, "_heif_state", False)
    with pytest.raises(PermanentIngestError, match="pillow-heif.*send a JPEG"):
        classify(p)


def test_real_heic_works_when_pillow_heif_is_installed(tmp_path):
    pytest.importorskip("pillow_heif")
    assert heif_supported()
    p = tmp_path / "photo.heic"
    Image.new("RGB", (64, 48), (200, 30, 30)).save(p, format="HEIF")
    info = classify(p)
    assert info.kind == "heic" and info.ext == ".heic"
    doc = load_document([p])
    assert doc.pages[0].source_kind == "image" and not doc.is_pdf


def test_corrupt_and_truncated_images_are_permanent_errors(tmp_path):
    good = save(tmp_path / "good.jpg", "JPEG", size=(400, 400))
    truncated = tmp_path / "cut.jpg"
    truncated.write_bytes(good.read_bytes()[:300])
    with pytest.raises(PermanentIngestError, match="damaged or unreadable"):
        validate_image(classify(truncated))
    header_only = tmp_path / "hdr.png"
    header_only.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
    with pytest.raises(PermanentIngestError, match="damaged or unreadable"):
        validate_image(classify(header_only))


def test_tiny_images_are_rejected(tmp_path):
    with pytest.raises(PermanentIngestError, match="too small"):
        validate_image(classify(save(tmp_path / "dot.png", "PNG", size=(2, 2))))


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f.jpg"
    save(p, "JPEG")
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_document_hash_for_single_files_and_albums():
    assert document_hash(["aa"], False) == "aa" and document_hash(["aa"], True) == "aa"
    ab, ba = document_hash(["aa", "bb"], True), document_hash(["bb", "aa"], True)
    assert ab != ba and ab != "aa" and len(ab) == 64
    assert ab == document_hash(["aa", "bb"], True)  # stable
    assert ab == hashlib.sha256(b"aa\nbb").hexdigest()


def test_single_photo_loads_as_one_page(tmp_path):
    p = save(tmp_path / "a.jpg", "JPEG")
    doc = load_document([p])
    assert [(pg.page_number, pg.source_path, pg.source_kind) for pg in doc.pages] == [(1, str(p), "image")]
    assert doc.pages[0].image == p and not doc.is_pdf


def test_album_pages_keep_the_given_order(tmp_path):
    paths = [save(tmp_path / f"{n}.jpg", "JPEG") for n in ("z", "a", "m")]
    doc = load_document(paths, is_album=True)
    assert [Path_name(pg.source_path) for pg in doc.pages] == ["z.jpg", "a.jpg", "m.jpg"]
    assert [pg.page_number for pg in doc.pages] == [1, 2, 3]


def Path_name(p):
    return p.replace("\\", "/").rsplit("/", 1)[-1]


def test_several_files_must_be_an_album(tmp_path):
    paths = [save(tmp_path / f"{n}.jpg", "JPEG") for n in "ab"]
    with pytest.raises(PermanentIngestError, match="not as an album"):
        load_document(paths)


def test_pdf_cannot_join_an_album(tmp_path):
    make_scanned_pdf(tmp_path / "s.pdf", 1)
    img = save(tmp_path / "a.jpg", "JPEG")
    with pytest.raises(PermanentIngestError, match="cannot be combined"):
        load_document([tmp_path / "s.pdf", img], is_album=True)


def test_scanned_pdf_pages_are_rendered(tmp_path):
    make_scanned_pdf(tmp_path / "s.pdf", 3)
    doc = load_document([tmp_path / "s.pdf"], dpi=72)
    assert doc.is_pdf and [p.page_number for p in doc.pages] == [1, 2, 3]
    assert all(p.source_kind == "pdf-scan" and p.text_layer is None and p.image is not None for p in doc.pages)
    assert all(p.source_path == str(tmp_path / "s.pdf") for p in doc.pages)


def test_text_layer_pages_are_not_rendered(tmp_path):
    make_text_pdf(tmp_path / "t.pdf", ["A real text layer with enough characters", "Second page has real text too ok"])
    doc = load_document([tmp_path / "t.pdf"])
    assert all(p.source_kind == "pdf-text" and p.image is None and p.text_layer for p in doc.pages)
    assert "real text layer" in doc.pages[0].text_layer


def test_corrupt_pdf_is_permanent(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    with pytest.raises(PermanentIngestError):
        load_document([p])


def test_no_files_is_an_error():
    with pytest.raises(PermanentIngestError, match="no files"):
        load_document([])
