class StudyBuddyError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(StudyBuddyError):
    """Invalid or missing configuration."""


class LLMError(StudyBuddyError):
    """LM Studio request failed or returned something unusable."""


class ContextOverflowError(LLMError):
    """The prompt (including any image) does not fit the model's loaded context."""


class ReaderError(StudyBuddyError):
    """A page could not be read; never swallowed into an empty transcript."""
