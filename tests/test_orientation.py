import io

import pytest
from PIL import Image

from studybuddy.reader.images import clean_image
from studybuddy.reader.ocr import RapidOcrEngine
from studybuddy.reader.orientation import Line, detect_rotation, probe_of, rotate_ccw
from test_images import make_page, to_bytes


@pytest.fixture(scope="module")
def engine():
    return RapidOcrEngine()


# ---- real OCR engine on synthetic pages ------------------------------------------------

@pytest.mark.parametrize("turned", [90, 180, 270])
def test_sideways_and_upside_down_pages_are_uprighted(engine, turned):
    page = make_page()
    sideways = page.rotate(turned, expand=True)  # PIL rotates counter-clockwise
    expected = (360 - turned) % 360              # the rotation that undoes it
    assert detect_rotation(sideways, engine) == expected
    result = clean_image(to_bytes(sideways), max_edge=4000, rotation_probe=engine)
    assert result.auto_rotation == expected and f"auto-rotate-{expected}" in result.steps
    assert result.image.height > result.image.width  # portrait again
    lines = engine.probe(result.image)
    assert any("photosynthesis" in ln.text.lower() for ln in lines)
    assert all(ln.width > ln.height for ln in lines)  # lines are horizontal


def test_upright_page_is_left_alone(engine):
    result = clean_image(to_bytes(make_page()), max_edge=4000, rotation_probe=engine)
    assert result.auto_rotation == 0 and not any(s.startswith("auto-rotate") for s in result.steps)


def test_blank_page_is_not_rotated_by_guesswork(engine):
    assert detect_rotation(Image.new("RGB", (1200, 1700), "white"), engine) == 0


def test_exif_rotated_photo_skips_detection():
    class Boom:
        def probe(self, image):
            raise AssertionError("must not probe when EXIF already rotated the image")

    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    result = clean_image(buf.getvalue(), rotation_probe=Boom())
    assert result.exif_rotated and result.auto_rotation == 0


def test_no_probe_means_no_detection():
    sideways = make_page().rotate(90, expand=True)
    assert clean_image(to_bytes(sideways), max_edge=4000).auto_rotation == 0


# ---- detection logic with a fake probe -------------------------------------------------

class FakeProbe:
    """Scripted probe: returns the next canned line list per call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def probe(self, image):
        self.calls += 1
        return self.results.pop(0)


def wide(text="some words", conf=0.95):
    return Line(400, 30, text, conf)


def tall(text="some words", conf=0.95):
    return Line(30, 400, text, conf)


def test_too_few_lines_returns_zero_without_further_probing():
    probe = FakeProbe([wide(), wide()])
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 0 and probe.calls == 1


def test_upright_wins_when_flipped_reads_worse():
    probe = FakeProbe([wide()] * 5, [wide()] * 5, [wide("xx", 0.3)] * 5)
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 0


def test_flipped_wins_gives_180():
    probe = FakeProbe([wide("zz", 0.3)] * 5, [wide("a long recognised line of text")] * 5)
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 180


def test_sideways_pages_get_90_or_270():
    # tall boxes -> rotate 90 first; then upright reads fine -> 90
    probe = FakeProbe([tall()] * 5, [wide()] * 5, [wide("x", 0.2)] * 5)
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 90
    # ... or upside-down after that rotation -> 270
    probe = FakeProbe([tall()] * 5, [wide("x", 0.2)] * 5, [wide()] * 5)
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 270


def test_near_tie_does_not_flip():
    probe = FakeProbe([wide()] * 5, [wide()] * 5, [wide(conf=0.97)] * 5)
    assert detect_rotation(Image.new("RGB", (100, 100)), probe) == 0


def test_rotate_ccw_is_exact_and_zero_is_identity():
    img = Image.new("RGB", (30, 10))
    assert rotate_ccw(img, 0) is img
    assert rotate_ccw(img, 90).size == (10, 30) and rotate_ccw(img, 180).size == (30, 10)


def test_probe_of_only_accepts_engines_that_probe():
    assert probe_of(object()) is None and probe_of(None) is None
    assert probe_of(FakeProbe()) is not None


def test_pipeline_records_auto_rotation(engine):
    from studybuddy.reader.pipeline import PageInput, read_pages
    from studybuddy.reader.transcribe import PageTranscript

    class Reader:
        def __init__(self):
            self.sizes = []

        def transcribe(self, image):
            self.sizes.append(image.size)
            return PageTranscript("Photosynthesis is the process by which green plants convert light " * 5,
                                  image.size, 1)

    reader = Reader()
    sideways = to_bytes(make_page().rotate(90, expand=True))
    (rec,) = read_pages([PageInput(1, "scan.png", image=sideways)], reader, engine)
    assert rec.auto_rotation == 270 and "auto-rotate-270" in rec.steps
    assert reader.sizes[0][1] > reader.sizes[0][0]  # the READER receives a portrait page
