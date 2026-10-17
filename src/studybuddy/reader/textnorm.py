"""Text normalisation shared by the scoring code."""
import re

_WS = re.compile(r"\s+")

# Typographic quotes the READER sometimes emits where the page (and so the reference) uses plain ones.
_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
})


def normalise_text(text: str) -> str:
    """Unify curly/straight quotes and collapse whitespace. Case and the rest of the punctuation still count."""
    return _WS.sub(" ", text.translate(_QUOTES)).strip()
