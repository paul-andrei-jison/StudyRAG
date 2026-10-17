class StudyBuddyError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(StudyBuddyError):
    """Invalid or missing configuration."""


class LLMError(StudyBuddyError):
    """LM Studio request failed or returned something unusable."""
