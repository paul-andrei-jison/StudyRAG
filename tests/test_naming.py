import threading
from datetime import datetime
from pathlib import Path

import pytest

from studybuddy.ingest.naming import (
    MAX_PATH_CHARS, NamingError, claim_unique_dir, claim_unique_file, fit_name, is_windows_reserved,
    library_stem, normalise_extension, sanitize_component, slugify, subject_folder, subject_token,
)

WHEN = datetime(2026, 10, 17, 14, 25)


@pytest.mark.parametrize("name", ["CON", "con", "Prn", "AUX", "NUL", "COM1", "com9", "LPT1", "lpt9", "con.txt", "NUL.jpg"])
def test_reserved_device_names_are_detected(name):
    assert is_windows_reserved(name)


@pytest.mark.parametrize("name", ["console", "com10", "lpt0", "science", "aux-notes", "nullable.jpg"])
def test_ordinary_names_are_not_reserved(name):
    assert not is_windows_reserved(name)


def test_sanitize_replaces_invalid_characters_and_collapses_spaces():
    assert sanitize_component('a<b>c:d"e/f\\g|h?i*j', 40) == "a b c d e f g h i j"
    assert sanitize_component("tab\tand\nnewline\x00nul", 40) == "tab and newline nul"
    assert sanitize_component("  lots   of   space  ", 40) == "lots of space"


def test_sanitize_strips_trailing_dots_and_spaces_which_windows_drops():
    assert sanitize_component("notes. . ", 40) == "notes"
    assert sanitize_component("...", 40) == "untitled"
    assert sanitize_component("name.", 40) == "name"


def test_sanitize_escapes_reserved_names():
    assert sanitize_component("CON", 40) == "_CON"
    assert sanitize_component("nul.txt", 40) == "_nul.txt"
    assert sanitize_component("Console", 40) == "Console"


def test_sanitize_truncates_without_leaving_a_trailing_dot_or_space():
    out = sanitize_component("abcd. efgh", 5)
    assert out == "abcd" and len(out) <= 5
    assert len(sanitize_component("x" * 500, 40)) == 40


def test_sanitize_empty_and_all_invalid_fall_back():
    assert sanitize_component("", 40) == "untitled"
    assert sanitize_component('<>:"/\\|?*', 40) == "untitled"
    assert sanitize_component("", 40, fallback="Unsorted") == "Unsorted"


def test_sanitize_keeps_unicode_letters():
    assert sanitize_component("Pagsusuri ng Wika - Filipino", 40) == "Pagsusuri ng Wika - Filipino"
    assert sanitize_component("Café Géographie", 40) == "Café Géographie"


@pytest.mark.parametrize("text,slug", [
    ("Photosynthesis Worksheet", "photosynthesis-worksheet"),
    ("  Earth's  Magnetic Field!! ", "earths-magnetic-field"),
    ("Earth’s Atmosphere", "earths-atmosphere"),
    ("Newton's 2nd Law: F=ma", "newtons-2nd-law-f-ma"),
    ("Café Géographie: l'été", "cafe-geographie-lete"),
    ("Chapter 4 - Forces & Motion (Part 2)", "chapter-4-forces-motion-part-2"),
    ("日本語のノート", "untitled"),
    ("", "untitled"),
    ("***", "untitled"),
    ("CON", "con"),
])
def test_slugify(text, slug):
    assert slugify(text) == slug


def test_slugify_truncates_at_a_word_boundary():
    slug = slugify("the habitable zone and the protective magnetic field of earth explained simply", 30)
    assert len(slug) <= 30 and not slug.endswith("-") and slug.startswith("the-habitable-zone")
    assert slug in "the-habitable-zone-and-the-protective-magnetic"
    assert slugify("x" * 200, 50) == "x" * 50  # no boundary to cut at


def test_subject_token_and_folder():
    assert subject_token("Earth Science") == "Earth-Science"
    assert subject_folder("Earth Science") == "Earth Science"
    assert subject_folder("a/b") == "a b" and subject_token("CON") == "_CON"
    assert len(subject_folder("S" * 100)) == 40


def test_library_stem_matches_the_documented_shape():
    stem = library_stem(WHEN, "Science", "Photosynthesis worksheet")
    assert stem == "2026-10-17_1425_Science_photosynthesis-worksheet"
    assert library_stem(WHEN, "Earth Science", "Plates") == "2026-10-17_1425_Earth-Science_plates"
    assert library_stem(WHEN, "Science", "") == "2026-10-17_1425_Science_untitled"
    assert library_stem(datetime(2026, 1, 2, 3, 4), "Math", "x") == "2026-01-02_0304_Math_x"


def test_extensions():
    assert normalise_extension(".JPEG") == ".jpg" and normalise_extension("PNG") == ".png"
    assert normalise_extension(".tif") == ".tiff" and normalise_extension("") == ""


def test_fit_name_shortens_long_stems_but_keeps_the_extension(tmp_path):
    directory = Path("C:/lib") / ("d" * 100)
    name = fit_name(directory, "stem-" + "x" * 300, ".jpg")
    assert name.endswith(".jpg") and len(str(directory / name)) <= MAX_PATH_CHARS
    assert fit_name(tmp_path, "short", ".pdf") == "short.pdf"


def test_fit_name_rejects_a_folder_path_that_leaves_no_room():
    with pytest.raises(NamingError, match="too long"):
        fit_name(Path("C:/" + "d" * 235), "stem", ".jpg")


def test_claim_unique_file_numbers_collisions(tmp_path):
    first = claim_unique_file(tmp_path, "2026-10-17_1425_Science_notes", ".jpg")
    second = claim_unique_file(tmp_path, "2026-10-17_1425_Science_notes", ".jpg")
    third = claim_unique_file(tmp_path, "2026-10-17_1425_Science_notes", ".jpg")
    assert [p.name for p in (first, second, third)] == [
        "2026-10-17_1425_Science_notes.jpg", "2026-10-17_1425_Science_notes_2.jpg",
        "2026-10-17_1425_Science_notes_3.jpg"]
    assert all(p.exists() and p.stat().st_size == 0 for p in (first, second, third))


def test_claim_unique_file_does_not_touch_existing_files(tmp_path):
    existing = tmp_path / "a.jpg"
    existing.write_bytes(b"precious")
    claimed = claim_unique_file(tmp_path, "a", ".jpg")
    assert claimed.name == "a_2.jpg" and existing.read_bytes() == b"precious"


def test_claim_unique_file_creates_the_folder(tmp_path):
    claimed = claim_unique_file(tmp_path / "Science", "n", ".png")
    assert claimed.parent.is_dir() and claimed.exists()


def test_claim_unique_dir(tmp_path):
    a = claim_unique_dir(tmp_path, "album")
    b = claim_unique_dir(tmp_path, "album")
    assert (a.name, b.name) == ("album", "album_2") and a.is_dir() and b.is_dir()


def test_concurrent_claims_never_return_the_same_path(tmp_path):
    results, lock = [], threading.Lock()

    def claim():
        path = claim_unique_file(tmp_path, "same-name", ".jpg")
        with lock:
            results.append(path)

    threads = [threading.Thread(target=claim) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 30 and len(set(results)) == 30
    assert len(list(tmp_path.iterdir())) == 30
