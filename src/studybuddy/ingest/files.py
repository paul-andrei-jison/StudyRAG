"""File type detection, hashing and page loading for ingest.

Type is decided from the file's first bytes, not its name (a `.jpg` that is really a PDF is a PDF). Anything that
can never work is a PermanentIngestError raised BEFORE any model is called.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image

from ..errors import PermanentIngestError, ReaderError
from ..reader.pdf import inspect_pdf, render_pdf_pages
from ..reader.pipeline import PageInput

HASH_CHUNK = 1024 * 1024
EXTENSION = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp", "tiff": ".tiff", "heic": ".heic", "pdf": ".pdf"}
HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"heif"}

_heif_state: bool | None = None


def heif_supported() -> bool:
    """True when pillow-heif is installed (registered once). Without it HEIC is a clear permanent failure."""
    global _heif_state
    if _heif_state is None:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
            _heif_state = True
        except Exception:  # not installed or no wheel for this platform
            _heif_state = False
    return _heif_state


@dataclass(frozen=True)
class FileInfo:
    path: Path
    kind: str          # jpeg | png | webp | bmp | tiff | heic | pdf
    ext: str           # the right extension for the real type, e.g. ".jpg"

    @property
    def is_pdf(self) -> bool:
        return self.kind == "pdf"


def sniff(head: bytes) -> str:
    """The real file type from its first bytes, or 'unknown'."""
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[4:8] == b"ftyp" and head[8:12] in HEIC_BRANDS:
        return "heic"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    return "unknown"


def classify(path: Path) -> FileInfo:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(16)
    except OSError as exc:
        raise PermanentIngestError(f"{path.name}: cannot be read ({exc})") from exc
    if size == 0:
        raise PermanentIngestError(f"{path.name}: the file is empty")
    kind = sniff(head)
    if kind == "gif":
        raise PermanentIngestError(f"{path.name}: GIF images are not supported; send a JPEG, PNG or PDF")
    if kind == "unknown":
        raise PermanentIngestError(f"{path.name}: unsupported or damaged file type (send a photo or a PDF)")
    if kind == "heic" and not heif_supported():
        raise PermanentIngestError(f"{path.name}: HEIC photos need the pillow-heif package; send a JPEG instead")
    return FileInfo(path, kind, EXTENSION[kind])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def document_hash(file_hashes: Sequence[str], is_album: bool) -> str:
    """One file: its own hash. An album: the hash of its page hashes in order (same photos, same order = same document)."""
    if not is_album or len(file_hashes) == 1:
        return file_hashes[0]
    return hashlib.sha256("\n".join(file_hashes).encode("ascii")).hexdigest()


def validate_image(info: FileInfo) -> None:
    """Fully decode the image so a truncated or corrupt photo fails now, not in the middle of the reader."""
    try:
        with Image.open(info.path) as img:
            img.load()
            if img.width < 8 or img.height < 8:
                raise PermanentIngestError(f"{info.path.name}: the image is too small to read ({img.width}x{img.height})")
    except PermanentIngestError:
        raise
    except Exception as exc:  # UnidentifiedImageError, truncated data, decompression bomb, ...
        raise PermanentIngestError(f"{info.path.name}: the image is damaged or unreadable ({type(exc).__name__}: {exc})") from exc


@dataclass
class LoadedDocument:
    pages: list[PageInput]
    files: list[FileInfo]
    is_pdf: bool


def load_document(files: Sequence[Path], is_album: bool = False, dpi: int = 200) -> LoadedDocument:
    """Validate the files and turn them into reader inputs, one PageInput per page, numbered from 1."""
    if not files:
        raise PermanentIngestError("no files were given")
    infos = [classify(Path(f)) for f in files]
    has_pdf = any(i.is_pdf for i in infos)
    if has_pdf and len(infos) > 1:
        raise PermanentIngestError("A PDF cannot be combined with other files in one document.")
    if len(infos) > 1 and not is_album:
        raise PermanentIngestError("Several files were submitted as one document but not as an album.")

    if has_pdf:
        path = infos[0].path
        try:
            layers = inspect_pdf(path)
            scanned = {p.page_number for p in layers if not p.has_text_layer}
            rendered = dict(render_pdf_pages(path, dpi=dpi, only=scanned)) if scanned else {}
        except ReaderError as exc:
            raise PermanentIngestError(f"{path.name}: {exc}") from exc
        if not layers:
            raise PermanentIngestError(f"{path.name}: the PDF has no pages")
        pages = [PageInput(p.page_number, str(path), image=rendered.get(p.page_number),
                           text_layer=p.text if p.has_text_layer else None,
                           source_kind="pdf-text" if p.has_text_layer else "pdf-scan") for p in layers]
        return LoadedDocument(pages, infos, True)

    pages = []
    for number, info in enumerate(infos, start=1):
        validate_image(info)
        pages.append(PageInput(number, str(info.path), image=info.path, source_kind="image"))
    return LoadedDocument(pages, infos, False)
