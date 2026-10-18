"""Everything the app runs in one process: the LM Studio client, the index, the ingest service, the inbox watcher and chat.

`Runtime` takes its parts as arguments so tests can build it from fakes; `build_runtime` builds the real ones.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

from ..chat.gate import GateConfig
from ..chat.session import ChatConfig, ChatSession
from ..config import Settings, validate_dirs
from ..index.index import Index
from ..ingest.service import IngestService
from ..ingest.subjects import SubjectList
from ..ingest.watcher import InboxWatcher
from .files import LibraryGuard
from .library import LibraryView
from .status import StatusBoard
from .uploads import sweep_empty_uploads, uploads_dir

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    client: Any                      # LMStudioClient (or a fake with the same methods)
    index: Index
    subjects: SubjectList
    service: IngestService
    watcher: InboxWatcher | None = None
    chat_config: ChatConfig | None = None
    gate: GateConfig | None = None   # None: the calibrated default

    def __post_init__(self) -> None:
        self.guard = LibraryGuard(self.settings.library_dir)
        self.library = LibraryView(self.service, self.guard)
        self.board = StatusBoard(self.service)

    def new_session(self) -> ChatSession:
        """One conversation per browser tab; they share the index and the model client."""
        config = dataclasses.replace(self.chat_config) if self.chat_config else ChatConfig.from_settings(self.settings)
        return ChatSession(self.index, self.client, self.settings, self.subjects, config, self.gate)

    def start(self) -> "Runtime":
        sweep_empty_uploads(uploads_dir(self.service))
        self.service.start()
        self.service.run_exclusive(self._settle_interrupted_moves).add_done_callback(self._log_settled)
        if self.watcher is not None:
            self.watcher.start()
        return self

    def _settle_interrupted_moves(self, index: Index) -> list[str]:
        """A subject change that a crash interrupted is undone or finished before anything else touches the library."""
        return self.library.relocator.recover()

    @staticmethod
    def _log_settled(future) -> None:
        try:
            for note in future.result():
                log.warning("subject change settled after a restart: %s", note)
        except Exception:
            log.exception("could not settle interrupted subject changes")

    def close(self) -> None:
        """Orderly shutdown: stop looking for files, let the document being read finish, then close the stores."""
        if self.watcher is not None:
            self.watcher.stop()
        self.service.stop()
        self.index.close()
        close = getattr(self.client, "close", None)
        if close:
            close()


def build_runtime(settings: Settings, use_ocr: bool = True) -> Runtime:
    """The real thing. Raises a StudyBuddyError (bad folders, embedding model changed, ...) for the caller to print."""
    from ..llm import LMStudioClient
    from ..reader.ocr import RapidOcrEngine
    from ..reader.transcribe import Reader

    validate_dirs(settings)
    client = LMStudioClient(settings)
    try:
        index = Index.open(settings, client)
        subjects = SubjectList.from_settings(settings)
    except BaseException:
        client.close()
        raise
    service = IngestService(settings, index, Reader(client), RapidOcrEngine() if use_ocr else None, client, subjects)
    return Runtime(settings, client, index, subjects, service, InboxWatcher(service))
