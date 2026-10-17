import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from studybuddy.ingest.files import classify, sha256_file
from studybuddy.ingest.filing import Filer, FilingError, safe_move

WHEN = datetime(2026, 10, 17, 14, 25)


def photo(path, size=(300, 400), color=(180, 190, 200), fmt="JPEG"):
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


@pytest.fixture
def lib(tmp_path):
    return tmp_path / "library"


@pytest.fixture
def inbox(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def file_one(lib, src, subject="Science", title="Photosynthesis worksheet", copy_fn=None, hashes=None):
    filer = Filer(lib, copy_fn) if copy_fn else Filer(lib)
    return filer.file_document([src], [classify(src)], hashes or [sha256_file(src)], subject, title, WHEN, False)


def tree(root):
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")) if root.exists() else []


# ---- single files --------------------------------------------------------------------------

def test_single_photo_is_filed_with_the_readable_name(lib, inbox):
    src = photo(inbox / "IMG_2041.JPG")
    before = src.read_bytes()
    filed = file_one(lib, src)
    assert filed.final_path == lib / "Science" / "2026-10-17_1425_Science_photosynthesis-worksheet.jpg"
    assert filed.final_path.read_bytes() == before and filed.page_files == [filed.final_path]
    assert filed.pdf_path is None and filed.created == [filed.final_path]
    assert src.read_bytes() == before  # the original is untouched and still there
    assert tree(lib) == ["Science", "Science/2026-10-17_1425_Science_photosynthesis-worksheet.jpg"]  # no temp, no placeholder


def test_extension_follows_the_real_file_type(lib, inbox):
    src = photo(inbox / "scan.jpg", fmt="PNG")  # really a PNG
    assert file_one(lib, src).final_path.suffix == ".png"


def test_name_collisions_get_a_number_and_never_overwrite(lib, inbox):
    a, b = photo(inbox / "a.jpg", color=(1, 2, 3)), photo(inbox / "b.jpg", color=(9, 9, 9))
    first, second = file_one(lib, a), file_one(lib, b)
    assert second.final_path.name == first.final_path.name.replace(".jpg", "_2.jpg")
    assert first.final_path.read_bytes() == a.read_bytes() and second.final_path.read_bytes() == b.read_bytes()


def test_windows_hostile_subject_and_title_are_made_safe(lib, inbox):
    src = photo(inbox / "a.jpg")
    filed = file_one(lib, src, subject="CON", title='Notes: "Forces/Motion"?? ...')
    assert filed.final_path.parent.name == "_CON" and filed.final_path.name.startswith("2026-10-17_1425__CON_notes-forces-motion")
    assert filed.final_path.exists()


def test_very_long_titles_keep_the_path_within_limits(lib, inbox):
    filed = file_one(lib, photo(inbox / "a.jpg"), title="word " * 200)
    assert len(str(filed.final_path)) < 240 and filed.final_path.suffix == ".jpg"


def test_a_corrupted_copy_is_detected_and_leaves_nothing_behind(lib, inbox):
    src = photo(inbox / "a.jpg")
    original = src.read_bytes()

    def corrupting_copy(s, d):
        d.write_bytes(s.read_bytes()[:-5] + b"XXXXX")

    with pytest.raises(FilingError, match="does not match") as info:
        file_one(lib, src, copy_fn=corrupting_copy)
    assert info.value.transient is True
    assert [p for p in tree(lib) if p != "Science"] == []  # no placeholder, no .partial
    assert src.read_bytes() == original


def test_a_source_that_changed_since_it_was_hashed_is_not_filed(lib, inbox):
    src = photo(inbox / "a.jpg")
    stale_hash = hashlib.sha256(b"what the file used to contain").hexdigest()
    with pytest.raises(FilingError, match="does not match"):
        file_one(lib, src, hashes=[stale_hash])
    assert [p for p in tree(lib) if p != "Science"] == []


def test_copy_oserror_becomes_a_filing_error(lib, inbox):
    def failing(s, d):
        raise OSError("disk full")

    with pytest.raises(FilingError, match="disk full"):
        file_one(lib, photo(inbox / "a.jpg"), copy_fn=failing)
    assert [p for p in tree(lib) if p != "Science"] == []


def test_pdf_is_filed_as_a_pdf(lib, inbox):
    pdf = inbox / "doc.pdf"
    Image.new("RGB", (200, 280), "white").save(pdf, "PDF")
    filed = file_one(lib, pdf, title="Chapter notes")
    assert filed.final_path.suffix == ".pdf" and filed.final_path.read_bytes() == pdf.read_bytes()


# ---- albums --------------------------------------------------------------------------------

def file_album(lib, paths, subject="Science", title="Photosynthesis worksheet", copy_fn=None):
    filer = Filer(lib, copy_fn) if copy_fn else Filer(lib)
    return filer.file_document(paths, [classify(p) for p in paths], [sha256_file(p) for p in paths],
                               subject, title, WHEN, True)


def test_album_gets_its_own_folder_with_every_page_and_a_combined_pdf(lib, inbox):
    pages = [photo(inbox / f"{i}.jpg", size=(300 + i, 400), color=(i * 30, 50, 90)) for i in range(1, 4)]
    filed = file_album(lib, pages)
    folder = lib / "Science" / "2026-10-17_1425_Science_photosynthesis-worksheet"
    assert filed.final_path == folder and folder.is_dir()
    assert sorted(p.name for p in folder.iterdir()) == [
        "2026-10-17_1425_Science_photosynthesis-worksheet.pdf", "page-01.jpg", "page-02.jpg", "page-03.jpg"]
    for n, src in enumerate(pages, 1):  # byte-identical, in the given order
        assert (folder / f"page-{n:02d}.jpg").read_bytes() == src.read_bytes()
    assert filed.page_files == [folder / f"page-0{n}.jpg" for n in (1, 2, 3)]
    pdf = pdfium.PdfDocument(str(filed.pdf_path))
    assert len(pdf) == 3
    pdf.close()


def test_album_pdf_downscales_huge_photos_but_the_page_photo_keeps_full_resolution(lib, inbox):
    big = photo(inbox / "big.jpg", size=(5000, 3000))
    small = photo(inbox / "small.jpg", size=(400, 300))
    filed = file_album(lib, [big, small])
    with Image.open(filed.page_files[0]) as img:
        assert img.size == (5000, 3000)
    pdf = pdfium.PdfDocument(str(filed.pdf_path))
    width, height = pdf[0].get_size()
    assert round(max(width, height)) == 2500 and round(min(width, height)) == 1500  # 72 dpi: points = pixels
    assert len(pdf) == 2
    pdf.close()


def test_album_with_an_exif_rotated_photo_builds_a_pdf(lib, inbox):
    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6
    rotated = inbox / "rot.jpg"
    img.save(rotated, "JPEG", exif=exif)
    filed = file_album(lib, [rotated, photo(inbox / "b.jpg")])
    assert filed.pdf_path.exists() and filed.page_files[0].read_bytes() == rotated.read_bytes()


def test_album_folder_collision_gets_a_number(lib, inbox):
    pages = [photo(inbox / f"{i}.jpg", color=(i, i, i)) for i in (1, 2)]
    first, second = file_album(lib, pages), file_album(lib, pages)
    assert second.final_path.name == first.final_path.name + "_2"
    assert (second.final_path / "page-01.jpg").exists()


def test_album_failure_removes_the_whole_folder_it_created(lib, inbox):
    pages = [photo(inbox / f"{i}.jpg", color=(i, i, i)) for i in (1, 2, 3)]
    calls = {"n": 0}

    def flaky(s, d):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        shutil.copyfile(s, d)

    with pytest.raises(FilingError):
        file_album(lib, pages, copy_fn=flaky)
    assert [p for p in tree(lib) if p != "Science"] == []
    assert all(p.exists() for p in pages)


def test_a_failed_album_does_not_touch_an_earlier_one(lib, inbox):
    ok = file_album(lib, [photo(inbox / "a.jpg", color=(1, 1, 1)), photo(inbox / "b.jpg", color=(2, 2, 2))])
    with pytest.raises(FilingError):
        file_album(lib, [photo(inbox / "c.jpg", color=(3, 3, 3)), photo(inbox / "d.jpg", color=(4, 4, 4))],
                   copy_fn=lambda s, d: (_ for _ in ()).throw(OSError("boom")))
    assert (ok.final_path / "page-01.jpg").exists() and len(list((lib / "Science").iterdir())) == 1


# ---- rollback and source removal -----------------------------------------------------------

def test_rollback_deletes_only_what_the_job_created(lib, inbox):
    (lib / "Science").mkdir(parents=True)
    bystander = lib / "Science" / "someone-elses.jpg"
    bystander.write_bytes(b"keep")
    filed = file_one(lib, photo(inbox / "a.jpg"))
    album = file_album(lib, [photo(inbox / "b.jpg", color=(5, 5, 5)), photo(inbox / "c.jpg", color=(6, 6, 6))])
    Filer.rollback(filed.created + album.created)
    assert tree(lib) == ["Science", "Science/someone-elses.jpg"] and bystander.read_bytes() == b"keep"
    Filer.rollback(filed.created)  # idempotent


def test_sources_are_removed_only_after_the_filed_copy_verifies(lib, inbox):
    src = photo(inbox / "a.jpg")
    filed = file_one(lib, src)
    assert Filer.remove_sources([src], filed.page_files, [sha256_file(filed.final_path)]) == []
    assert not src.exists() and filed.final_path.exists()


def test_a_source_is_kept_when_its_copy_is_missing_or_wrong(lib, inbox):
    src = photo(inbox / "a.jpg")
    digest = sha256_file(src)
    missing = lib / "nope.jpg"
    problems = Filer.remove_sources([src], [missing], [digest])
    assert src.exists() and len(problems) == 1 and "could not be verified" in problems[0]
    wrong = lib / "Science"
    wrong.mkdir(parents=True)
    bad_copy = wrong / "bad.jpg"
    bad_copy.write_bytes(b"not the same")
    assert "could not be verified" in Filer.remove_sources([src], [bad_copy], [digest])[0] and src.exists()


def test_a_locked_source_is_reported_not_raised(lib, inbox, monkeypatch):
    src = photo(inbox / "a.jpg")
    filed = file_one(lib, src)
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self.name == "a.jpg":
            raise PermissionError("in use by another process")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked)
    problems = Filer.remove_sources([src], filed.page_files, [sha256_file(filed.final_path)])
    assert len(problems) == 1 and "could not be removed" in problems[0]
    monkeypatch.undo()
    assert src.exists()


def test_album_sources_are_removed_page_by_page(lib, inbox):
    pages = [photo(inbox / f"{i}.jpg", color=(i, 9, 9)) for i in (1, 2)]
    filed = file_album(lib, pages)
    assert Filer.remove_sources(pages, filed.page_files, [sha256_file(p) for p in filed.page_files]) == []
    assert not any(p.exists() for p in pages)


# ---- safe_move (duplicates/, failed/) ------------------------------------------------------

def test_safe_move_verifies_and_removes_the_source(tmp_path, inbox):
    src = photo(inbox / "dup.jpg")
    data = src.read_bytes()
    moved = safe_move(src, tmp_path / "duplicates", name="abc123_dup.jpg")
    assert moved.read_bytes() == data and not src.exists() and moved.name == "abc123_dup.jpg"


def test_safe_move_never_overwrites(tmp_path, inbox):
    a, b = photo(inbox / "x.jpg", color=(1, 1, 1)), photo(inbox / "y.jpg", color=(2, 2, 2))
    first = safe_move(a, tmp_path / "dest", name="same.jpg")
    second = safe_move(b, tmp_path / "dest", name="same.jpg")
    assert first.name == "same.jpg" and second.name == "same_2.jpg" and first.read_bytes() != second.read_bytes()


def test_safe_move_keeps_the_source_when_the_copy_cannot_be_verified(tmp_path, inbox, monkeypatch):
    src = photo(inbox / "dup.jpg")
    import studybuddy.ingest.filing as filing

    def bad_copy(s, d):
        d.write_bytes(b"corrupt")

    monkeypatch.setattr(filing, "_copy_bytes", bad_copy)
    with pytest.raises(FilingError, match="does not match"):
        safe_move(src, tmp_path / "dest")
    assert src.exists() and list((tmp_path / "dest").iterdir()) == []
