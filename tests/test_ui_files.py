import os
import subprocess
import sys

import pytest
from PIL import Image

from ingest_helpers import make_photo
from studybuddy.ui.files import LibraryGuard, PageUnavailableError, PathNotAllowedError


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "lib"
    (root / "Science").mkdir(parents=True)
    return root


@pytest.fixture
def guard(lib):
    return LibraryGuard(lib)


def make_pdf(path, pages=2):
    Image.new("RGB", (200, 300), "white").save(path, "PDF", save_all=True,
                                                 append_images=[Image.new("RGB", (200, 300), "gray")] * (pages - 1))
    return path


def test_a_photo_inside_the_library_is_shown(guard, lib):
    photo = make_photo(lib / "Science" / "a.jpg", size=(300, 400))
    image = guard.render_page(photo)
    assert image.size == (300, 400) and image.mode == "RGB"
    assert guard.render_page(str(photo)).size == (300, 400)  # str paths work too


def test_big_photos_are_shrunk_and_the_original_is_untouched(guard, lib):
    photo = make_photo(lib / "Science" / "big.jpg", size=(4000, 2000))
    before = photo.read_bytes()
    assert max(guard.render_page(photo).size) == 1800
    assert photo.read_bytes() == before


def test_exif_rotation_is_applied(guard, lib):
    path = lib / "Science" / "rot.jpg"
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise to view
    Image.new("RGB", (400, 200), "white").save(path, exif=exif)
    assert guard.render_page(path).size == (200, 400)


def test_a_relative_path_is_taken_from_the_library(guard, lib):
    make_photo(lib / "Science" / "a.jpg")
    assert guard.render_page("Science/a.jpg").size == (300, 400)


def test_pdf_pages_are_rendered_and_out_of_range_is_refused(guard, lib):
    pdf = make_pdf(lib / "Science" / "d.pdf", pages=2)
    assert guard.render_page(pdf, 2).width > 100
    for bad in (0, 3):
        with pytest.raises(PageUnavailableError, match="2 page"):
            guard.render_page(pdf, bad)


@pytest.mark.parametrize("make_path", [
    lambda tmp, lib: tmp / "secret.jpg",                      # a file beside the library
    lambda tmp, lib: lib / ".." / "secret.jpg",               # dot-dot out of it
    lambda tmp, lib: lib / "Science" / ".." / ".." / "secret.jpg",
    lambda tmp, lib: str(lib) + "/../secret.jpg",
    lambda tmp, lib: "../secret.jpg",                         # relative traversal
    lambda tmp, lib: "..\\secret.jpg",
    lambda tmp, lib: tmp / "lib-evil" / "x.jpg",              # sibling that merely starts with the library name
])
def test_paths_outside_the_library_are_refused(guard, lib, tmp_path, make_path):
    make_photo(tmp_path / "secret.jpg")
    make_photo(tmp_path / "lib-evil" / "x.jpg")
    with pytest.raises(PathNotAllowedError, match="outside the library"):
        guard.render_page(make_path(tmp_path, lib))


@pytest.mark.parametrize("raw", ["", "   ", "a\x00.jpg"])
def test_empty_and_nul_paths_are_refused(guard, raw):
    with pytest.raises(PathNotAllowedError):
        guard.render_page(raw)


def test_a_symlink_that_leaves_the_library_is_refused(guard, lib, tmp_path):
    outside = make_photo(tmp_path / "outside" / "o.jpg")
    link = lib / "Science" / "link.jpg"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here (Windows needs Developer Mode or an administrator shell); the junction test below covers the same escape")
    with pytest.raises(PathNotAllowedError, match="outside the library"):
        guard.render_page(link)


@pytest.mark.skipif(sys.platform != "win32", reason="directory junctions are Windows only")
def test_a_junction_that_leaves_the_library_is_refused(guard, lib, tmp_path):
    outside = tmp_path / "outside"
    make_photo(outside / "o.jpg")
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(lib / "Science" / "jump"), str(outside)], capture_output=True)
    if done.returncode != 0:
        pytest.skip("cannot create a junction here")
    with pytest.raises(PathNotAllowedError, match="outside the library"):
        guard.render_page(lib / "Science" / "jump" / "o.jpg")


@pytest.mark.skipif(sys.platform != "win32", reason="drive and stream syntax is Windows only")
@pytest.mark.parametrize("raw", ["C:secret.jpg", "\\\\server\\share\\x.jpg", "Science\\a.jpg:hidden"])
def test_windows_only_path_forms_are_refused(guard, lib, raw):
    make_photo(lib / "Science" / "a.jpg")
    with pytest.raises((PathNotAllowedError, PageUnavailableError)):  # drive-relative "C:x.jpg" lands inside the library, on a file that is not there
        guard.render_page(raw)


def test_the_library_folder_itself_and_non_images_are_refused(guard, lib):
    with pytest.raises((PathNotAllowedError, PageUnavailableError)):
        guard.render_page(lib)
    text = lib / "Science" / "notes.txt"
    text.write_text("not an image", encoding="utf-8")
    with pytest.raises(PathNotAllowedError, match="not a photo or PDF"):
        guard.render_page(text)
    renamed = lib / "Science" / "fake.jpg"
    renamed.write_text("<html>script</html>", encoding="utf-8")
    with pytest.raises(PathNotAllowedError, match="not a photo or PDF"):  # judged by its bytes, not its name
        guard.render_page(renamed)


def test_a_moved_file_and_a_damaged_photo_are_explained(guard, lib):
    with pytest.raises(PageUnavailableError, match="no longer in the library"):
        guard.render_page(lib / "Science" / "gone.jpg")
    bad = lib / "Science" / "cut.jpg"
    bad.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
    with pytest.raises(PageUnavailableError, match="cannot be shown"):
        guard.render_page(bad)
