import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from studybuddy.reader.evaluate import (
    Report, error_rates, evaluate, find_pairs, format_report, summarise, write_report,
)
from studybuddy.reader.transcribe import PageTranscript

REF = "Photosynthesis converts light energy into chemical energy"
ROOT = Path(__file__).resolve().parents[1]


def test_error_rates_known_values():
    cer, wer = error_rates("abcd", "abxd")
    assert cer == pytest.approx(0.25) and wer == 1.0
    cer, wer = error_rates("a b c", "a x c")
    assert wer == pytest.approx(1 / 3)
    assert error_rates(REF, REF) == (0.0, 0.0)


def test_error_rates_ignore_whitespace_only():
    assert error_rates("a  b\n c", "a b c") == (0.0, 0.0)
    assert error_rates("Hello", "hello")[0] > 0  # case counts


def test_empty_hypothesis_is_total_error_and_empty_reference_rejected():
    assert error_rates(REF, "  ") == (1.0, 1.0)
    with pytest.raises(ValueError):
        error_rates("", "text")


def test_find_pairs_not_set_or_missing_or_empty(tmp_path):
    assert find_pairs(None).pairs == [] and "not set" in find_pairs(None).note
    assert "does not exist" in find_pairs(tmp_path / "nope").note
    empty = find_pairs(tmp_path)
    assert empty.pairs == [] and "No image" in empty.note


def test_find_pairs_matches_by_stem_and_reports_orphans(tmp_path):
    for name in ("a.jpg", "B.PNG", "c.jpg"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "orphan.txt").write_text("o")
    (tmp_path / "notes.md").write_text("ignored")
    found = find_pairs(tmp_path)
    assert [(i.name, r.name) for i, r in found.pairs] == [("a.jpg", "a.txt"), ("B.PNG", "b.txt")]
    assert [p.name for p in found.images_without_reference] == ["c.jpg"]
    assert [p.name for p in found.references_without_image] == ["orphan.txt"]


class FakeReader:
    def __init__(self, text="", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def transcribe(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model fell over")
        return PageTranscript(self.text, image.size, 1)


class FakeOcr:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def read(self, image):
        return self.text


@pytest.fixture
def sample(tmp_path):
    img = Image.new("RGB", (900, 1200), (250, 250, 250))
    ImageDraw.Draw(img).text((50, 100), REF, fill="black", font=ImageFont.load_default(size=40))
    img.save(tmp_path / "p1.png")
    (tmp_path / "p1.txt").write_text(REF, encoding="utf-8")
    return [(tmp_path / "p1.png", tmp_path / "p1.txt")]


def test_evaluate_two_models_and_ocr(sample):
    readers = {"good": FakeReader(REF), "bad": FakeReader("Photosynthesis converts lite energy")}
    rows = evaluate(sample, readers, FakeOcr(REF.replace("light", "ligh")), compare_raw=True)
    by = {r.model: r for r in rows}
    assert len(rows) == 2 and by["good"].reader_cer == 0.0 and by["bad"].reader_cer > 0.2
    assert by["good"].ocr_cer == by["bad"].ocr_cer > 0  # OCR is model independent
    assert by["good"].agreement > by["bad"].agreement
    assert by["good"].raw_reader_cer == 0.0 and by["good"].reader_seconds is not None
    s = summarise(rows)
    assert s["good"]["reader_cer"] == 0.0 and s["bad"]["reader_cer"] > 0.2 and s["good"]["errors"] == 0


def test_failing_model_gives_error_row_not_silence(sample):
    rows = evaluate(sample, {"ok": FakeReader(REF), "broken": FakeReader(fail=True)}, None)
    assert [r.model for r in rows] == ["ok", "broken"]
    assert rows[0].error == "" and "fell over" in rows[1].error
    assert summarise(rows)["broken"]["errors"] == 1


def test_unreadable_image_gives_error_row_for_every_model(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"junk")
    (tmp_path / "x.txt").write_text(REF)
    rows = evaluate([(tmp_path / "x.jpg", tmp_path / "x.txt")], {"m1": FakeReader(REF), "m2": FakeReader(REF)}, None)
    assert len(rows) == 2 and all(r.error for r in rows)


def test_report_files_and_text(sample, tmp_path):
    rows = evaluate(sample, {"m": FakeReader(REF)}, FakeOcr(REF))
    report = Report(rows, summarise(rows), find_pairs(None))
    text = format_report(report)
    assert "p1.png" in text and "READER CER=0.000" in text and "OCR CER=0.000" in text
    json_path, csv_path = write_report(report, tmp_path / "out", stamp="t")
    assert json.loads(json_path.read_text())["rows"][0]["model"] == "m"
    with csv_path.open(newline="", encoding="utf-8") as f:
        assert next(csv.DictReader(f))["image"] == "p1.png"


def run_script(tmp_path, samples):
    env = {**os.environ, "SB_READER_MODEL": "r", "SB_CHAT_MODEL": "c", "SB_EMBED_MODEL": "e",
           "SB_LIBRARY_DIR": str(tmp_path / "lib"), "SB_DATA_DIR": str(tmp_path / "data"),
           "SB_SAMPLES_DIR": str(samples)}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_reader.py")],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)


def test_script_with_missing_samples_folder_exits_cleanly(tmp_path):
    out = run_script(tmp_path, tmp_path / "does-not-exist")
    assert out.returncode == 0 and "No samples to evaluate" in out.stdout and "does not exist" in out.stdout


def test_script_with_empty_samples_folder_exits_cleanly(tmp_path):
    (tmp_path / "s").mkdir()
    out = run_script(tmp_path, tmp_path / "s")
    assert out.returncode == 0 and "No samples to evaluate" in out.stdout
