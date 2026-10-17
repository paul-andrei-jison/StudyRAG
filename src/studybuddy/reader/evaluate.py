"""Bake-off: character/word error rate of READER models and classic OCR against hand-corrected text.

Samples are image + same-name .txt pairs in a folder OUTSIDE the repo (SB_SAMPLES_DIR). The .txt
must be the true text of the page in the same conventions the READER uses (Markdown tables, LaTeX,
[figure: ...], [illegible]); only whitespace and quote style (curly vs straight) are normalised before comparing. Case and other punctuation count.
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
from .orientation import probe_of
from .quality import score_page
from .regions import HW_SUFFIX, SplitRates, load_regions, split_error_rates
from .textnorm import normalise_text

DRAFT_MARKER = "DRAFT - proofread against the page, then delete this line"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class TranscribesImages(Protocol):
    def transcribe(self, image: Image.Image): ...  # returns an object with .text


@dataclass
class SamplePairs:
    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    images_without_reference: list[Path] = field(default_factory=list)
    references_without_image: list[Path] = field(default_factory=list)
    unproofed_drafts: list[Path] = field(default_factory=list)  # references still carrying the DRAFT line
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
    raw_reader_wer: float | None = None
    # Printed vs handwritten split (needs a <stem>.hw sidecar; hw_* stay None on pages without one)
    hw_words: int = 0
    reader_printed_cer: float | None = None
    reader_printed_wer: float | None = None
    reader_hw_cer: float | None = None
    reader_hw_wer: float | None = None
    ocr_printed_cer: float | None = None
    ocr_printed_wer: float | None = None
    ocr_hw_cer: float | None = None
    ocr_hw_wer: float | None = None
    error: str = ""


@dataclass
class Report:
    rows: list[Row]
    summary: dict[str, dict]
    skipped: SamplePairs


def error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    """(CER, WER). An empty hypothesis scores 1.0 (everything missing); the reference must be non-empty."""
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    if not ref:
        raise ValueError("reference text is empty")
    if not hyp:
        return 1.0, 1.0
    return jiwer.cer(ref, hyp), jiwer.wer(ref, hyp)


def is_unproofed_draft(reference_path: Path) -> bool:
    """True while the DRAFT line is still anywhere in the file: the text is the model's own output."""
    text = reference_path.read_text(encoding="utf-8", errors="replace")
    return any(line.strip() == DRAFT_MARKER for line in text.splitlines())


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
            used.add(ref)
            if is_unproofed_draft(ref):
                out.unproofed_drafts.append(ref)  # never scored: it would measure the model against itself
            else:
                out.pairs.append((img, ref))
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
            "raw_reader_wer": _mean([r.raw_reader_wer for r in ok if r.raw_reader_wer is not None]),
        }
        for who in ("reader", "ocr"):
            for part in ("printed", "hw"):
                for metric in ("cer", "wer"):
                    name = f"{who}_{part}_{metric}"
                    summary[model][name] = _mean([getattr(r, name) for r in ok if getattr(r, name) is not None])
    return summary


def _set_split(row: Row, who: str, split: SplitRates) -> None:
    row.hw_words = split.hw_words
    setattr(row, f"{who}_printed_cer", split.printed_cer)
    setattr(row, f"{who}_printed_wer", split.printed_wer)
    setattr(row, f"{who}_hw_cer", split.hw_cer)
    setattr(row, f"{who}_hw_wer", split.hw_wer)


def evaluate(pairs: list[tuple[Path, Path]], readers: Mapping[str, TranscribesImages],
             ocr: OcrEngine | None, max_edge: int = 2048, compare_raw: bool = False) -> list[Row]:
    """One Row per (model, page). A failing page is a row with an error, never silently dropped."""
    rows: list[Row] = []
    for image_path, ref_path in pairs:
        if is_unproofed_draft(ref_path):
            raise ValueError(f"{ref_path.name} is still an unproofed DRAFT; proofread it and delete the DRAFT line")
        reference = ref_path.read_text(encoding="utf-8")
        shared = Row(image=image_path.name, model="")
        cleaned = None
        ocr_text = ""
        spans: list[tuple[int, int]] = []
        try:
            spans = load_regions(reference, ref_path.with_suffix(HW_SUFFIX))
            cleaned = clean_image(image_path, max_edge=max_edge, rotation_probe=probe_of(ocr))
            if ocr is not None:
                ocr_text = ocr.read(cleaned.image)
                shared.ocr_cer, shared.ocr_wer = error_rates(reference, ocr_text)
                if spans:
                    _set_split(shared, "ocr", split_error_rates(reference, ocr_text, spans))
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
                if spans:
                    _set_split(row, "reader", split_error_rates(reference, text, spans))
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
                    row.raw_reader_cer, row.raw_reader_wer = error_rates(reference, reader.transcribe(raw).text)
            except Exception as exc:
                row.error = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def write_drafts(images: list[Path], reader: TranscribesImages, max_edge: int = 2048
                 ) -> tuple[list[Path], dict[Path, str]]:
    """Write READER output beside each image as <stem>.txt, headed by the DRAFT line.

    Never overwrites an existing file. Returns (written, {image: error}); a failing image does not stop the rest.
    """
    written: list[Path] = []
    failed: dict[Path, str] = {}
    for image_path in images:
        target = image_path.with_suffix(".txt")
        if target.exists():
            failed[image_path] = f"{target.name} already exists; not overwritten"
            continue
        try:
            text = reader.transcribe(clean_image(image_path, max_edge=max_edge).image).text
        except Exception as exc:
            failed[image_path] = f"{type(exc).__name__}: {exc}"
            continue
        target.write_text(f"{DRAFT_MARKER}\n{text}\n", encoding="utf-8")
        written.append(target)
    return written, failed


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_report(report: Report) -> str:
    lines = [f"{'image':32} {'model':24} {'rCER':>6} {'rWER':>6} {'oCER':>6} {'oWER':>6} {'agree':>6}  flags/error"]
    for r in report.rows:
        lines.append(f"{r.image[:32]:32} {r.model[:24]:24} {_fmt(r.reader_cer):>6} {_fmt(r.reader_wer):>6} "
                     f"{_fmt(r.ocr_cer):>6} {_fmt(r.ocr_wer):>6} {_fmt(r.agreement):>6}  {r.error or r.flags}")
    if any(r.hw_words for r in report.rows):
        lines.append("")
        lines.append("Printed vs handwritten (from <stem>.hw sidecars; insertions count as printed errors)")
        lines.append(f"{'image':32} {'hw words':>8} {'rCERp':>6} {'rWERp':>6} {'rCERh':>6} {'rWERh':>6} "
                     f"{'oCERp':>6} {'oWERp':>6} {'oCERh':>6} {'oWERh':>6}")
        for r in report.rows:
            if r.hw_words:
                lines.append(f"{r.image[:32]:32} {r.hw_words:>8} {_fmt(r.reader_printed_cer):>6} "
                             f"{_fmt(r.reader_printed_wer):>6} {_fmt(r.reader_hw_cer):>6} {_fmt(r.reader_hw_wer):>6} "
                             f"{_fmt(r.ocr_printed_cer):>6} {_fmt(r.ocr_printed_wer):>6} "
                             f"{_fmt(r.ocr_hw_cer):>6} {_fmt(r.ocr_hw_wer):>6}")
    lines.append("")
    lines.append("Summary (mean over pages; r = READER, o = classic OCR; lower CER/WER is better)")
    for model, s in report.summary.items():
        raw = (f", READER on uncleaned images CER {_fmt(s['raw_reader_cer'])} WER {_fmt(s['raw_reader_wer'])}"
               if s["raw_reader_cer"] is not None else "")
        lines.append(f"  {model}: pages={s['pages']} errors={s['errors']} READER CER={_fmt(s['reader_cer'])} "
                     f"WER={_fmt(s['reader_wer'])} | OCR CER={_fmt(s['ocr_cer'])} WER={_fmt(s['ocr_wer'])} "
                     f"| agreement={_fmt(s['agreement'])}{raw}")
    for label, items in (("DRAFT references (proofread and delete the DRAFT line to score them)",
                          report.skipped.unproofed_drafts),
                         ("images without a .txt reference", report.skipped.images_without_reference),
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
