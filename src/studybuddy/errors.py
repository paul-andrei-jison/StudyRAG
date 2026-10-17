class StudyBuddyError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(StudyBuddyError):
    """Invalid or missing configuration."""


class LLMError(StudyBuddyError):
    """LM Studio request failed or returned something unusable."""


class TransientLLMError(LLMError):
    """LM Studio is off, busy or has no model loaded: the same request is expected to work later."""


class ContextOverflowError(LLMError):
    """The prompt (including any image) does not fit the model's loaded context."""


class ReaderError(StudyBuddyError):
    """A page could not be read; never swallowed into an empty transcript."""


class PermanentIngestError(StudyBuddyError):
    """A document can never be ingested as it is (corrupt, unsupported type, ...): retrying will not help."""


class ChatContextError(StudyBuddyError):
    """The model's context is too small to hold a chat prompt (system prompt, question, a source and the answer)."""


class IndexStoreError(StudyBuddyError):
    """The search index is unusable or was asked to do something inconsistent."""


class IndexMismatchError(IndexStoreError):
    """The embedding model or document prefix no longer matches the stored vectors."""
