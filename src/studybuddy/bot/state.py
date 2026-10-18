"""What the bot still owes the student, kept on disk so a restart or a sleeping computer forgets nothing.

`SB_DATA_DIR/bot/items/<id>.json`      one per file that arrived (stage, spool path, file id for a retry, the job it became)
`SB_DATA_DIR/bot/questions/<id>.json`  one per "which subject?" question (the files it covers, the buttons' subject list, its deadline)

Every write is atomic (temporary file, then rename). A file that cannot be read is set aside in `corrupt/`, never fatal.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Generic, TypeVar

# stages of an item
DOWNLOADING, ASKING, SUBMITTED, DONE = "downloading", "asking", "submitted", "done"
# stages of a question
OPEN, AWAITING_OTHER, CLOSED = "open", "awaiting_other", "closed"


def new_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Item:
    id: str
    chat_id: int
    message_id: int
    kind: str                              # photo | photo-file | pdf | other
    file_id: str
    file_unique_id: str = ""
    file_size: int | None = None
    original_name: str | None = None
    media_group_id: str | None = None
    caption: str | None = None
    received_at: str = ""                  # ISO local time the message was sent (becomes the document's date in its file name)
    spool_path: str | None = None
    sha256: str | None = None
    stage: str = DOWNLOADING
    attempts: int = 0                      # download attempts so far
    next_attempt_at: float = 0.0
    question_id: str | None = None
    subject: str | None = None             # the chosen subject, None = auto-detect
    job_id: str | None = None
    created_at: float = 0.0
    finished_at: float | None = None


@dataclass
class Question:
    id: str
    chat_id: int
    item_ids: list[str]
    subjects: list[str] = field(default_factory=list)    # snapshot of the subject list the buttons index into
    message_id: int | None = None                        # the question message, edited as the student answers
    deadline: float = 0.0                                # epoch seconds: after this the files go to auto-detect
    album: bool = False                                  # "Make these one document" was tapped
    stage: str = OPEN
    created_at: float = 0.0
    settle_until: float = 0.0                            # a photo group is asked about only after no more photos arrive
    asked: bool = False                                  # the question message has been sent
    group_key: str | None = None                         # media_group_id: late photos of the same group join an open question
    awaiting_since: float = 0.0                          # when "Other..." was tapped (the next text message is the subject)
    caption_subject: str | None = None                   # a caption that names a known subject: used without asking
    caption_suggestion: str | None = None                # a caption that is a valid NEW subject name: offered as one button


T = TypeVar("T", Item, Question)


class _Dir(Generic[T]):
    def __init__(self, root: Path, cls: type[T]):
        self.root, self.cls = root, cls
        self._names = {f.name for f in fields(cls)}
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, ident: str) -> Path:
        return self.root / f"{ident}.json"

    def save(self, obj: T) -> None:
        path = self._path(obj.id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(obj), indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, ident: str) -> T | None:
        path = self._path(ident)
        try:
            return self._load(path)
        except FileNotFoundError:
            return None

    def _load(self, path: Path) -> T | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self.cls(**{k: v for k, v in data.items() if k in self._names})
        except FileNotFoundError:
            raise
        except (OSError, ValueError, TypeError):
            corrupt = self.root / "corrupt"
            corrupt.mkdir(exist_ok=True)
            os.replace(path, corrupt / path.name)
            return None

    def all(self) -> list[T]:
        out = []
        for path in sorted(self.root.glob("*.json")):
            try:
                obj = self._load(path)
            except FileNotFoundError:
                continue
            if obj is not None:
                out.append(obj)
        return sorted(out, key=lambda o: (o.created_at, o.id))

    def delete(self, ident: str) -> None:
        self._path(ident).unlink(missing_ok=True)


class StateStore:
    """Thread-safe (the bot's event loop and the ingest worker's callback both touch it)."""

    def __init__(self, root: Path):
        self.lock = threading.RLock()
        self.items = _Dir(Path(root) / "items", Item)
        self.questions = _Dir(Path(root) / "questions", Question)

    # items
    def save_item(self, item: Item) -> None:
        with self.lock:
            self.items.save(item)

    def item(self, ident: str) -> Item | None:
        with self.lock:
            return self.items.get(ident)

    def all_items(self) -> list[Item]:
        with self.lock:
            return self.items.all()

    def find_item(self, predicate: Callable[[Item], bool]) -> list[Item]:
        return [i for i in self.all_items() if predicate(i)]

    def already_seen(self, chat_id: int, message_id: int, file_unique_id: str) -> bool:
        """Telegram can deliver an update again after a crash; the same message + file is not taken twice."""
        return any(i.chat_id == chat_id and i.message_id == message_id and i.file_unique_id == file_unique_id for i in self.all_items())

    # questions
    def save_question(self, question: Question) -> None:
        with self.lock:
            self.questions.save(question)

    def question(self, ident: str) -> Question | None:
        with self.lock:
            return self.questions.get(ident)

    def all_questions(self) -> list[Question]:
        with self.lock:
            return self.questions.all()

    def delete_question(self, ident: str) -> None:
        with self.lock:
            self.questions.delete(ident)

    def sweep_done(self, now: float, keep_seconds: float = 48 * 3600) -> int:
        """Forget finished items after a while (they are kept that long so a re-delivered update is recognised)."""
        with self.lock:
            old = [i for i in self.items.all() if i.stage == DONE and i.finished_at is not None and now - i.finished_at > keep_seconds]
            for item in old:
                self.items.delete(item.id)
            for question in self.questions.all():
                if question.stage == CLOSED and now - question.created_at > keep_seconds:
                    self.questions.delete(question.id)
            return len(old)
