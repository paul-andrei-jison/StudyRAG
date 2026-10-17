from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from studybuddy.reader.evaluate import Report, evaluate, format_report, summarise, write_report, find_pairs
from studybuddy.reader.regions import load_regions, normalise_with_mask, split_error_rates
from studybuddy.reader.transcribe import PageTranscript

REF = "Name Juan Cruz Section 10-A1\nB 1. What is a key reason?\nA. presence of moons\nA 2. Which gases?\n"


def sidecar(tmp_path, *lines):
    p = tmp_path / "page.hw"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def spans_text(ref, spans):
    return [ref[s:e] for s, e in spans]


def test_literal_and_regex_entries(tmp_path):
    side = sidecar(tmp_path, "# header", "", "Juan Cruz", r"re:^[A-D](?= \d+\.)")
    assert spans_text(REF, load_regions(REF, side)) == ["Juan Cruz", "B", "A"]


def test_no_sidecar_means_no_regions(tmp_path):
    assert load_regions(REF, tmp_path / "missing.hw") == []
    assert load_regions(REF, None) == []


def test_dead_entry_is_loud(tmp_path):
    with pytest.raises(ValueError, match="matches nothing"):
        load_regions(REF, sidecar(tmp_path, "Not on the page"))
    with pytest.raises(ValueError, match="matches nothing"):
        load_regions(REF, sidecar(tmp_path, r"re:^Z\d"))


def test_bad_regex_is_loud(tmp_path):
    with pytest.raises(ValueError, match="bad regex"):
        load_regions(REF, sidecar(tmp_path, "re:(unclosed"))


def test_mask_survives_whitespace_collapse():
    text = "a  b\n\nc d"
    spans = [(0, 4)]  # "a  b"
    norm, mask = normalise_with_mask(text, spans)
    assert norm == "a b c d"
    assert [c for c, m in zip(norm, mask) if m] == ["a", " ", "b"]


def test_split_separates_handwritten_from_printed_errors(tmp_path):
    spans = load_regions(REF, sidecar(tmp_path, "Juan Cruz", r"re:^[A-D](?= \d+\.)"))
    # one handwritten error (Cruz -> Kruz, wrong answer letter B -> D) and one printed error (moons -> mooms)
    hyp = REF.replace("Cruz", "Kruz").replace("B 1.", "D 1.").replace("moons", "mooms")
    s = split_error_rates(REF, hyp, spans)
    assert s.hw_words == 4  # Juan, Cruz, B, A
    assert s.hw_wer == pytest.approx(2 / 4)
    assert s.printed_wer == pytest.approx(1 / s.printed_words)
    assert s.hw_cer > 0 and s.printed_cer > 0


def test_perfect_hypothesis_is_zero_everywhere(tmp_path):
    spans = load_regions(REF, sidecar(tmp_path, "Juan Cruz"))
    s = split_error_rates(REF, REF, spans)
    assert (s.printed_cer, s.printed_wer, s.hw_cer, s.hw_wer) == (0, 0, 0, 0)


def test_insertions_count_against_printed_text(tmp_path):
    spans = load_regions(REF, sidecar(tmp_path, "Juan Cruz"))
    s = split_error_rates(REF, REF + "an invented extra line\n", spans)
    assert s.printed_wer > 0 and s.hw_wer == 0


def test_no_handwriting_gives_none(tmp_path):
    s = split_error_rates(REF, REF.replace("moons", "mooms"), [])
    assert s.hw_cer is None and s.hw_wer is None and s.hw_words == 0 and s.printed_wer > 0


def test_empty_hypothesis_misses_everything(tmp_path):
    spans = load_regions(REF, sidecar(tmp_path, "Juan Cruz"))
    s = split_error_rates(REF, "  ", spans)
    assert s.hw_wer == 1.0 and s.printed_wer == 1.0 and s.printed_cer == 1.0


def test_quote_styles_do_not_count_in_split():
    ref = "Earth's crust\nB 1. Earth's core"
    s = split_error_rates(ref, "Earth\u2019s crust\nB 1. Earth\u2019s core", [(14, 15)])
    assert s.printed_wer == 0 and s.hw_wer == 0


# ---- through evaluate() ---------------------------------------------------------

class FakeReader:
    def __init__(self, text):
        self.text = text

    def transcribe(self, image):
        return PageTranscript(self.text, image.size, 1)


def make_pair(tmp_path, reference, hw_lines=None):
    img = Image.new("RGB", (900, 1200), (250, 250, 250))
    ImageDraw.Draw(img).text((50, 100), "page", fill="black", font=ImageFont.load_default(size=40))
    img.save(tmp_path / "p.png")
    (tmp_path / "p.txt").write_text(reference, encoding="utf-8")
    if hw_lines is not None:
        (tmp_path / "p.hw").write_text("\n".join(hw_lines), encoding="utf-8")
    return [(tmp_path / "p.png", tmp_path / "p.txt")]


def test_evaluate_reports_the_split_when_a_sidecar_exists(tmp_path):
    pairs = make_pair(tmp_path, REF, ["Juan Cruz"])
    rows = evaluate(pairs, {"m": FakeReader(REF.replace("Cruz", "Kruz"))}, None)
    row = rows[0]
    assert row.hw_words == 2 and row.reader_hw_wer == 0.5 and row.reader_printed_wer == 0
    report = Report(rows, summarise(rows), find_pairs(None))
    assert "Printed vs handwritten" in format_report(report)
    assert report.summary["m"]["reader_hw_wer"] == 0.5
    json_path, csv_path = write_report(report, tmp_path / "out", stamp="x")
    assert "reader_hw_wer" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_evaluate_without_sidecar_has_no_split(tmp_path):
    rows = evaluate(make_pair(tmp_path, REF), {"m": FakeReader(REF)}, None)
    assert rows[0].hw_words == 0 and rows[0].reader_hw_wer is None
    assert "Printed vs handwritten" not in format_report(Report(rows, summarise(rows), find_pairs(None)))


def test_dead_sidecar_entry_becomes_an_error_row(tmp_path):
    rows = evaluate(make_pair(tmp_path, REF, ["Not on the page"]), {"m": FakeReader(REF)}, None)
    assert "matches nothing" in rows[0].error


def test_raw_wer_is_recorded_with_compare_raw(tmp_path):
    rows = evaluate(make_pair(tmp_path, REF), {"m": FakeReader(REF)}, None, compare_raw=True)
    assert rows[0].raw_reader_cer == 0.0 and rows[0].raw_reader_wer == 0.0
