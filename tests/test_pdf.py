import pytest
from PIL import Image

from studybuddy.errors import ReaderError
from studybuddy.reader.pdf import inspect_pdf, page_count, render_pdf_pages


def make_text_pdf(path, pages):
    """Hand-built PDF with a Helvetica text layer, one string per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>")
        stream = f"BT /F1 14 Tf 72 700 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)


def make_scanned_pdf(path, n):
    imgs = [Image.new("RGB", (400, 560), (200 + i, 200, 200)) for i in range(n)]
    imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:])


def test_text_layer_detected_per_page(tmp_path):
    p = tmp_path / "t.pdf"
    make_text_pdf(p, ["Photosynthesis converts light into chemical energy", "Chloroplasts contain chlorophyll pigments"])
    infos = inspect_pdf(p)
    assert [i.page_number for i in infos] == [1, 2]
    assert all(i.has_text_layer for i in infos)
    assert "Photosynthesis" in infos[0].text and "Chloroplasts" in infos[1].text


def test_scanned_pdf_has_no_text_layer(tmp_path):
    p = tmp_path / "s.pdf"
    make_scanned_pdf(p, 3)
    infos = inspect_pdf(p)
    assert len(infos) == 3 and not any(i.has_text_layer for i in infos)
    assert all(i.text == "" for i in infos)


def test_mixed_pdf_short_text_is_not_a_layer(tmp_path):
    p = tmp_path / "m.pdf"
    make_text_pdf(p, ["12", "This page has a proper sentence of text"])
    assert [i.has_text_layer for i in inspect_pdf(p)] == [False, True]


def test_render_yields_every_page_in_order(tmp_path):
    p = tmp_path / "s.pdf"
    make_scanned_pdf(p, 3)
    pages = list(render_pdf_pages(p, dpi=72))
    assert [n for n, _ in pages] == [1, 2, 3]
    assert page_count(p) == len(pages)
    assert all(img.mode == "RGB" and img.width > 100 for _, img in pages)


def test_render_dpi_scales_size(tmp_path):
    p = tmp_path / "t.pdf"
    make_text_pdf(p, ["A page with enough text to count here"])
    (_, small), = render_pdf_pages(p, dpi=72)
    (_, big), = render_pdf_pages(p, dpi=144)
    assert small.size == (612, 792) and big.size == (1224, 1584)


def test_corrupt_pdf_raises(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(ReaderError, match="Cannot open"):
        inspect_pdf(p)
    with pytest.raises(ReaderError):
        list(render_pdf_pages(p))
