import pytest
from PIL import Image

from studybuddy.errors import ReaderError
from studybuddy.reader.pdf import inspect_pdf, render_pdf_pages
from studybuddy.reader.pipeline import PageInput, read_pages
from studybuddy.reader.quality import PageRecord, illegible_ratio, score_page
from studybuddy.reader.transcribe import PageTranscript
from test_pdf import make_scanned_pdf, make_text_pdf

GOOD = "Photosynthesis converts light energy into chemical energy in the chloroplasts. " * 40


def test_score_clean_page_is_high():
    q, flags, ratio, cpm = score_page(GOOD, (1000, 1400), 0.95)
    assert q > 0.9 and flags == [] and ratio == 0 and cpm > 1000


def test_empty_output_is_flagged_and_zero():
    assert score_page("   \n", (1000, 1400), 0.9) == (0.0, ["empty"], 0.0, 0.0)


def test_illegible_share():
    assert illegible_ratio("a [illegible] c [illegible]") == 0.5
    assert illegible_ratio("") == 0.0
    q, flags, ratio, _ = score_page(("word " * 8 + "[illegible] ") * 10, (500, 500), 1.0)
    assert "many_illegible" in flags and ratio > 0.1 and q < 0.8


def test_too_short_for_image_size():
    q, flags, _, cpm = score_page("Title", (3000, 4000), 1.0)
    assert "too_short" in flags and cpm < 20 and q <= 0.5


def test_low_agreement_flagged():
    q, flags, _, _ = score_page(GOOD, (1000, 1400), 0.3)
    assert "low_agreement" in flags and q < 0.4


def test_no_ocr_means_no_agreement_penalty():
    q, flags, _, _ = score_page(GOOD, (1000, 1400), None)
    assert q == 1.0 and "low_agreement" not in flags


def test_glare_lowers_quality():
    q, flags, _, _ = score_page(GOOD, (1000, 1400), 1.0, glare_fraction=0.3)
    assert "glare" in flags and q == 0.8


def test_needs_attention():
    assert PageRecord(1, "a", "image", quality=0.9).needs_attention is False
    assert PageRecord(1, "a", "image", quality=0.4).needs_attention is True
    assert PageRecord(1, "a", "image", quality=0.9, flags=["empty"]).needs_attention is True
    assert PageRecord(1, "a", "image", quality=0.9, flags=["glare"]).needs_attention is False


# ---- pipeline ----------------------------------------------------------------

class FakeReader:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def transcribe(self, image):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return PageTranscript(text=reply, image_size=image.size, attempts=1)


class FakeOcr:
    name = "fake"

    def __init__(self, text=GOOD, fail=False):
        self.text, self.fail = text, fail

    def read(self, image):
        if self.fail:
            raise RuntimeError("ocr exploded")
        return self.text


def img_page(n, src="scan.jpg"):
    return PageInput(n, src, image=Image.new("RGB", (800, 1000), (200, 200, 200)))


def test_every_page_gets_a_record_even_when_one_fails():
    reader = FakeReader([GOOD, ReaderError("cut off"), GOOD])
    records = read_pages([img_page(1), img_page(2), img_page(3)], reader, FakeOcr())
    assert [r.page_number for r in records] == [1, 2, 3]
    assert records[1].error and "cut off" in records[1].error
    assert records[1].flags == ["error"] and records[1].quality == 0.0 and records[1].needs_attention
    assert records[0].quality > 0.9 and records[2].quality > 0.9


def test_original_path_and_cleanup_details_recorded():
    (rec,) = read_pages([img_page(1, src="C:/lib/Science/scan.jpg")], FakeReader([GOOD]), FakeOcr())
    assert rec.source_path == "C:/lib/Science/scan.jpg" and rec.source_kind == "image"
    assert rec.image_size == (800, 1000) and "contrast" in rec.steps and rec.sent_size
    assert rec.ocr_agreement == 1.0


def test_ocr_failure_flags_but_keeps_the_page():
    (rec,) = read_pages([img_page(1)], FakeReader([GOOD]), FakeOcr(fail=True))
    assert rec.text == GOOD and "ocr_failed" in rec.flags and rec.ocr_agreement is None and rec.error is None


def test_low_agreement_page_needs_attention():
    (rec,) = read_pages([img_page(1)], FakeReader([GOOD]), FakeOcr(text="unrelated words entirely here"))
    assert "low_agreement" in rec.flags and rec.needs_attention


def test_blank_page_is_flagged_empty():
    (rec,) = read_pages([img_page(1)], FakeReader([""]), FakeOcr(text=""))
    assert rec.flags == ["empty"] and rec.needs_attention


def test_unreadable_image_becomes_error_record():
    (rec,) = read_pages([PageInput(1, "bad.jpg", image=b"not an image")], FakeReader([]), None)
    assert rec.error and rec.flags == ["error"]


def test_page_without_image_or_text_is_error_record():
    (rec,) = read_pages([PageInput(1, "x.pdf")], FakeReader([]), None)
    assert rec.error and "neither" in rec.error


def test_text_layer_pages_skip_the_reader():
    reader = FakeReader([])
    (rec,) = read_pages([PageInput(1, "t.pdf", text_layer=GOOD, source_kind="pdf-text")], reader, FakeOcr())
    assert reader.calls == 0 and rec.source_kind == "pdf-text" and rec.text == GOOD and rec.quality > 0.9


def test_duplicate_page_numbers_rejected():
    with pytest.raises(ReaderError, match="Duplicate"):
        read_pages([img_page(1), img_page(1)], FakeReader([GOOD, GOOD]), None)


def test_mixed_pdf_pages_in_equals_pages_out(tmp_path):
    text_pdf, scan_pdf = tmp_path / "t.pdf", tmp_path / "s.pdf"
    make_text_pdf(text_pdf, ["A real text layer with enough characters", "Another page of real text here ok"])
    make_scanned_pdf(scan_pdf, 3)
    pages = [PageInput(i.page_number, str(text_pdf), text_layer=i.text, source_kind="pdf-text")
             for i in inspect_pdf(text_pdf)]
    pages += [PageInput(n, str(scan_pdf), image=im, source_kind="pdf-scan")
              for n, im in render_pdf_pages(scan_pdf, dpi=72)]
    pages = [PageInput(n, p.source_path, p.image, p.text_layer, p.source_kind) for n, p in enumerate(pages, 1)]
    reader = FakeReader([GOOD, GOOD, GOOD])
    records = read_pages(pages, reader, None)
    assert len(records) == 5 and reader.calls == 3
    assert [r.source_kind for r in records] == ["pdf-text"] * 2 + ["pdf-scan"] * 3
