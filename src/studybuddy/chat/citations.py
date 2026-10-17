"""Citation markers: find `[n]` in an answer, keep only the numbers that name a real source, strip the rest.

`[1]`, `[1, 2]`, `[1;2]`, `[1-3]` and the full-width `【1】` are citations; `[1][2]` is two of them. A number outside
1..N (or 0) is invalid: it is removed from the text and recorded, never shown as if it pointed at something.
Anything else in brackets (`[sic]`, `[a]`, `[1.5]`) is ordinary text and left alone.

`CitationFilter` does the same job on a stream: text is released as soon as it cannot be part of a marker, a possible
marker is held back until it closes, so the output equals the one-shot result for every way the text was split.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_OPEN, _CLOSE = "[【", "]】"
_GROUP = re.compile(r"^[\[【]\s*(\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–]\s*\d+)?)*)\s*[\]】]$")
_INCOMPLETE_AT_END = re.compile(r"[\[【]\s*[\d\s,;\-–]*$")
MAX_MARKER_CHARS = 24   # a bracket that stays open longer than this is not a citation
MAX_RANGE = 20          # [1-999999] must not expand into a million numbers


def _numbers(spec: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[,;]", spec):
        part = part.strip()
        if re.search(r"[-–]", part):
            low, high = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            out.extend(range(low, min(high, low + MAX_RANGE) + 1) if low <= high else [low, high])
        else:
            out.append(int(part))
    return out


@dataclass
class CitationFilter:
    """Streaming cleaner. `feed` returns the text that is safe to show now; call `flush` at the end."""
    n_sources: int
    cited: list[int] = field(default_factory=list)       # valid numbers, in order of first use
    invalid: list[int] = field(default_factory=list)     # numbers that named no source (stripped)
    _buffer: str = ""

    def _resolve(self, group: str) -> str:
        """The cleaned form of one citation group: valid numbers kept, invalid ones recorded and dropped."""
        numbers = _numbers(_GROUP.match(group).group(1))
        valid = [n for n in numbers if 1 <= n <= self.n_sources]
        for n in numbers:
            if 1 <= n <= self.n_sources:
                if n not in self.cited:
                    self.cited.append(n)
            elif n not in self.invalid:
                self.invalid.append(n)
        return "[" + ", ".join(str(n) for n in dict.fromkeys(valid)) + "]" if valid else ""

    def feed(self, piece: str) -> str:
        self._buffer += piece
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out: list[str] = []
        buf = self._buffer
        while buf:
            start = next((i for i, ch in enumerate(buf) if ch in _OPEN), -1)
            if start < 0:
                keep = len(buf) if final else len(buf.rstrip(" 	"))  # trailing spaces might precede a marker: hold them
                out.append(buf[:keep])
                buf = buf[keep:]
                break
            hold_from = len(buf[:start].rstrip(" \t"))  # spaces right before a marker belong to it
            if hold_from > 0:
                out.append(buf[:hold_from])
            lead = buf[hold_from:start]
            rest = buf[start:]
            close = next((i for i, ch in enumerate(rest) if ch in _CLOSE), -1)
            next_open = next((i for i, ch in enumerate(rest[1:], 1) if ch in _OPEN), -1)
            if close >= 0 and (next_open < 0 or close < next_open):
                group = rest[: close + 1]
                if _GROUP.match(group):
                    cleaned = self._resolve(group)
                    out.append(lead + cleaned if cleaned else "")
                    buf = rest[close + 1:]
                    continue
                out.append(lead + rest[0])      # not a citation: show the bracket and look again after it
                buf = rest[1:]
                continue
            if next_open >= 0 or len(rest) > MAX_MARKER_CHARS:
                out.append(lead + rest[0])      # another bracket opened first, or it stayed open too long
                buf = rest[1:]
                continue
            if final:
                out.append("" if _INCOMPLETE_AT_END.fullmatch(rest) else lead + rest)  # a cut-off "[1" is dropped
                buf = ""
                break
            buf = lead + rest                   # might still become a marker: wait for more text
            break
        self._buffer = buf
        return "".join(out)


@dataclass
class CleanedText:
    text: str
    cited: list[int]
    invalid: list[int]


def clean_citations(text: str, n_sources: int) -> CleanedText:
    """One-shot version of CitationFilter (it is the same code, so streaming and one-shot always agree)."""
    f = CitationFilter(n_sources)
    cleaned = f.feed(text) + f.flush()
    return CleanedText(cleaned, f.cited, f.invalid)
