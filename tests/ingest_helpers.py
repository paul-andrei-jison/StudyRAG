"""Shared fixtures for the ingest tests: fake reader / chat / embedder around real SQLite, LanceDB and the filesystem."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image

from conftest import FakeEmbedder
from studybuddy.index.index import Index
from studybuddy.ingest.jobs import JobStore
from studybuddy.ingest.processor import Processor
from studybuddy.ingest.subjects import SubjectList
from studybuddy.reader.transcribe import PageTranscript

PAGE_TEXT = ("Photosynthesis converts light energy into chemical energy inside the chloroplasts of green plants. "
             "The overall equation uses carbon dioxide and water to make glucose and oxygen.")
GOOD_JSON = json.dumps({"title": "Photosynthesis Worksheet",
                        "summary": "Explains how plants make sugar from light. Includes the overall equation.",
                        "subject": "Science"})
WHEN = datetime(2026, 10, 17, 14, 25)


class FakeReader:
    """transcribe() returns scripted text per call (a list), or the same text; an Exception entry is raised."""

    def __init__(self, texts=None, default=PAGE_TEXT):
        self.texts, self.default, self.calls = list(texts or []), default, 0

    def transcribe(self, image):
        self.calls += 1
        item = self.texts.pop(0) if self.texts else self.default
        if isinstance(item, Exception):
            raise item
        return PageTranscript(text=item, image_size=image.size, attempts=1)


class FakeChat:
    def __init__(self, *replies, default=GOOD_JSON):
        self.replies, self.default, self.calls = list(replies), default, 0

    def chat(self, messages, temperature=0.0, max_tokens=None):
        self.calls += 1
        item = self.replies.pop(0) if self.replies else self.default
        if isinstance(item, Exception):
            raise item
        return item


def make_photo(path: Path, size=(300, 400), color=(150, 160, 170), fmt="JPEG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)
    return path


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")) if root.exists() else []


@dataclass
class Env:
    settings: object
    index: Index
    reader: FakeReader
    chat: FakeChat
    embedder: FakeEmbedder
    subjects: SubjectList
    jobs: JobStore
    processor: Processor
    inbox: Path
    persisted: list = field(default_factory=list)

    @property
    def library(self) -> Path:
        return self.settings.library_dir

    @property
    def data(self) -> Path:
        return self.settings.data_dir

    def job(self, *paths, subject=None, album=False, kinds=None, cleanup=True, origin="api"):
        kinds = kinds or ["photo-file"] * len(paths)
        return self.jobs.create([str(p) for p in paths], kinds, WHEN, origin=origin, subject=subject, is_album=album,
                                cleanup_sources=cleanup)

    def counts(self):
        return self.index.store.counts()


def make_env(tmp_path, settings, reader=None, chat=None, embedder=None, subjects=("Math", "Science", "English")) -> Env:
    s = dataclasses.replace(settings, library_dir=tmp_path / "library", data_dir=tmp_path / "data",
                            subjects_file=tmp_path / "data" / "subjects.json")
    s.library_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    reader = reader or FakeReader()
    chat = chat or FakeChat()
    embedder = embedder or FakeEmbedder()
    index = Index.open(s, embedder)
    subject_list = SubjectList(s.data_dir / "subjects.json", list(subjects))
    jobs = JobStore(s.data_dir / "queue")
    inbox = s.data_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    env = Env(s, index, reader, chat, embedder, subject_list, jobs, None, inbox)
    env.processor = Processor(s, index, reader, None, chat, subject_list, inbox_dir=inbox,
                              persist=lambda job: (jobs.save(job), env.persisted.append(list(job.staged))))
    return env
