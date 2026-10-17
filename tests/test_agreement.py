from PIL import Image, ImageDraw, ImageFont

from studybuddy.reader.agreement import agreement, normalise
from studybuddy.reader.ocr import RapidOcrEngine

TEXT = "Photosynthesis converts light energy into chemical energy in the chloroplasts"


def test_normalise_strips_markup_only_one_engine_makes():
    text = "| Name | Mass |\n| --- | --- |\n| Car | 1200 |\n[figure: a block with an arrow]\n$F = m a$ and [illegible]"
    assert normalise(text) == ["name", "mass", "car", "1200", "f", "m", "a", "and"]


def test_identical_text_scores_one():
    a = agreement(TEXT, TEXT)
    assert a.score == 1.0 and a.word_f1 == 1.0


def test_case_and_punctuation_do_not_matter():
    assert agreement("Hello, World!", "hello world").score == 1.0


def test_column_order_difference_still_agrees():
    left, right = "alpha beta gamma delta", "epsilon zeta eta theta"
    assert agreement(f"{left} {right}", f"{right} {left}").score == 1.0


def test_one_misread_word_lowers_but_stays_high():
    a = agreement(TEXT, TEXT.replace("chloroplasts", "chloroplast5"))
    assert 0.8 < a.score < 1.0


def test_unrelated_text_scores_low():
    assert agreement(TEXT, "completely different words about history and kings").score < 0.3


def test_empty_cases():
    assert agreement("", "").score == 1.0          # blank page seen the same by both engines
    assert agreement("some text", "").score == 0.0
    assert agreement("", "some text").score == 0.0
    assert agreement("[figure: x]", "").score == 1.0  # only markup on one side counts as empty


def test_rapidocr_reads_rendered_text():
    img = Image.new("RGB", (1400, 200), "white")
    ImageDraw.Draw(img).text((30, 50), "Photosynthesis Worksheet Chloroplast", fill="black",
                             font=ImageFont.load_default(size=70))
    text = RapidOcrEngine().read(img)
    assert agreement("Photosynthesis Worksheet Chloroplast", text).word_f1 > 0.6
