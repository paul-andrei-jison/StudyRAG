"""Bake-off: character/word error rate of READER models and classic OCR against hand-corrected text.

Samples are image + same-name .txt pairs in a folder OUTSIDE the repo (SB_SAMPLES_DIR). The .txt
must be the true text of the page in the same conventions the READER uses (Markdown tables, LaTeX,
[figure: ...], [illegible]); only whitespace is normalised before comparing. Case and punctuation count.
"""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol

import jiwer
from PIL import Image

from .agreement import agreement
from .images import clean_image
from .ocr import OcrEngine
from .quality import score_page

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_WS = re.compile(r"\s+")


class TranscribesImages(Protocol):
    def transcribe(self, image: Image.Image): ...  # returns an object with .text


@dataclass
class SamplePairs:
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    images_without_reference: list[Path] = field(default_factory=list)
    references_without_image: list[Path] = field(default_factory=list)
    note: str = ""


@dataclass
class Row:
    image: str
    model: str
    reader_cer: float | None = None
    reader_wer: float | None = None
    ocr_cer: float | None = None
    ocr_wer: float | None = None
    agreement: float | None = None
    coverage: float | None = None         # share of OCR words present in the READER text
    quality: float | None = None
    flags: str = ""
    reader_seconds: float | None = None
    raw_reader_cer: float | None = None   # only with compare_raw: READER on the uncleaned image
    error: str = ""


@dataclass
class Report:
    rows: list[Row]
    summary: dict[str, dict]
    skipped: SamplePairs


def normalise_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    """(CER, WER). An empty hypothesis scores 1.0 (everything missing); the reference must be non-empty."""
    ref, hyp = normalise_ws(reference), normalise_ws(hypothesis)
    if not ref:
        raise ValueError("reference text is empty")
    if not hyp:
        return 1.0, 1.0
    return jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)


def find_pairs(samples_dir: Path | None) -> SamplePairs:
    out = SamplePairs()
    if samples_dir is None:
        out.note = "SB_SAMPLES_DIR is not set."
        return out
    if not samples_dir.is_dir():
        out.note = f"Samples folder does not exist: {samples_dir}"
        return out
    files = sorted(p for p in samples_dir.iterdir() if p.is_file())
    refs = {p.stem.lower(): p for p in files if p.suffix.lower() == ".txt"}
    used = set()
    for img in (p for p in files if p.suffix.lower() in IMAGE_EXTS):
        ref = refs.get(img.stem.lower())
        if ref is None:
            out.images_without_reference.append(img)
        else:
            out.pairs.append((img, ref))
            used.add(ref)
    out.references_without_image = [r for r in refs.values() if r not in used]
    if not out.pairs:
        out.note = f"No image + .txt pairs found in {samples_dir}."
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(rows: list[Row]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for model in dict.fromkeys(r.model for r in rows):
        mine = [r for r in rows if r.model == model]
        ok = [r for r in mine if not r.error]
        summary[model] = {
            "pages": len(mine),
            "errors": len(mine) - len(ok),
            "reader_cer": _mean([r.reader_cer for r in ok if r.reader_cer is not None]),
            "reader_wer": _mean([r.reader_wer for r in ok if r.reader_wer is not None]),
            "ocr_cer": _mean([r.ocr_cer for r in mine if r.ocr_cer is not None]),
            "ocr_wer": _mean([r.ocr_wer for r in mine if r.ocr_wer is not None]),
            "agreement": _mean([r.agreement for r in ok if r.agreement is not None]),
            "raw_reader_cer": _mean([r.raw_reader_cer for r in ok if r.raw_reader_cer is not None]),
        }
    return summary


def evaluate(pairs: list[tuple[Path, Path]], readers: Mapping[str, TranscribesImages],
             ocr: OcrEngine | None, max_edge: int = 2048, compare_raw: bool = False) -> list[Row]:
    """One Row per (model, page). A failing page is a row with an error, never silently dropped."""
    rows: list[Row] = []
    for image_path, ref_path in pairs:
        reference = ref_path.read_text(encoding="utf-8")
        shared = Row(image=image_path.name, model="")
        cleaned = None
        ocr_text = ""
        try:
            cleaned = clean_image(image_path, max_edge=max_edge)
            if ocr is not None:
                ocr_text = ocr.read(cleaned.image)
                shared.ocr_cer, shared.ocr_wer = error_rates(reference, ocr_text)
        except Exception as exc:
            shared.error = f"{type(exc).__name__}: {exc}"

        for model, reader in readers.items():
            row = Row(**{**asdict(shared), "model": model})
            if cleaned is None:
                rows.append(row)
                continue
            try:
                start = time.monotonic()
                text = reader.transcribe(cleaned.image).text
                row.reader_seconds = round(time.monotonic() - start, 1)
                row.reader_cer, row.reader_wer = error_rates(reference, text)
                if ocr is not None:
                    found = agreement(text, ocr_text)
                    row.agreement, row.coverage = found.score, found.ocr_coverage
                quality, flags, _, _ = score_page(text, cleaned.original_size, row.agreement,
                                                  cleaned.glare_fraction, row.coverage)
                row.quality, row.flags = quality, ",".join(flags)
                if compare_raw:
                    raw = Image.open(image_path)
                    raw.load()
                    raw = raw.convert("L")
                    if max(raw.size) > max_edge:
                        r = max_edge / max(raw.size)
                        raw = raw.resize((round(raw.width * r), round(raw.height * r)), Image.LANCZOS)
                    row.raw_reader_cer = error_rates(reference, reader.transcribe(raw).text)[0]
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_report(report: Report) -> str:
    lines = [f"{'image':32} {'model':24} {'rCER':>6} {'rWER':>6} {'oCER':>6} {'oWER':>6} {'agree':>6}  flags/error"]
    for r in report.rows:
        lines.append(f"{r.image[:32]:32} {r.model[:24]:24} {_fmt(r.reader_cer):>6} {_fmt(r.reader_wer):>6} "
                     f"{_fmt(r.ocr_cer):>6} {_fmt(r.ocr_wer):>6} {_fmt(r.agreement):>6}  {r.error or r.flags}")
    lines.append("")
    lines.append("Summary (mean over pages; r = READER, o = classic OCR; lower CER/WER is better)")
    for model, s in report.summary.items():
        raw = f", READER CER on uncleaned images {_fmt(s['raw_reader_cer'])}" if s["raw_reader_cer"] is not None else ""
        lines.append(f"  {model}: pages={s['pages']} errors={s['errors']} READER CER={_fmt(s['reader_cer'])} "
                     f"WER={_fmt(s['reader_wer'])} | OCR CER={_fmt(s['ocr_cer'])} WER={_fmt(s['ocr_wer'])} "
                     f"| agreement={_fmt(s['agreement'])}{raw}")
    for label, items in (("images without a .txt reference", report.skipped.images_without_reference),
                         (".txt references without an image", report.skipped.references_without_image)):
        if items:
            lines.append(f"Skipped {len(items)} {label}: " + ", ".join(p.name for p in items))
    return "\n".join(lines)


def write_report(report: Report, out_dir: Path, stamp: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path, csv_path = out_dir / f"reader-eval-{stamp}.json", out_dir / f"reader-eval-{stamp}.csv"
    json_path.write_text(json.dumps({"summary": report.summary, "rows": [asdict(r) for r in report.rows]},
                                    indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(Row("", "")).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in report.rows)
    return json_path, csv_path
