"""Segmentation of raw sources into embeddable spans.

Every chunk produced here is a *contiguous span of the original text*, and its
locator records the character offsets of that span, such that

    source_text[locator["start"]:locator["end"]] == chunk.text

holds exactly, byte for byte, for every chunk. That is not a nicety. A node
extracted from a chunk cites the chunk's span, and the citation is what makes
an extraction error recoverable: given offsets you can show a reviewer the
exact sentence a claim came from, and re-extract from it. An approximate
offset — off by the length of a stripped newline, or pointing into a
reconstructed string rather than the source — degrades a citation from
evidence to a hint, and it degrades silently, because nothing in the pipeline
downstream ever compares the two again.

So the implementation never builds a chunk by joining pieces together. It only
ever chooses two offsets and slices. Trimming whitespace moves an offset; it
does not edit a string.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any

from ..config import ChunkConfig

# A blank line, i.e. a paragraph boundary. Deliberately not \n\s*\n: that also
# matches a run of blank lines as one separator, which would let the cut point
# land past the start of the following paragraph.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

# A structural cut is only taken if it fills at least this fraction of the
# window. Without the floor, the greedy "last paragraph break that fits" rule
# emits a 20-character chunk whenever a short heading is followed by a long
# paragraph — common in markdown, and the resulting chunk embeds to noise.
_MIN_STRUCTURAL_FILL = 0.5


@dataclass(frozen=True)
class Chunk:
    """One embeddable span, with the offsets needed to cite it."""

    text: str
    index: int
    # {"start": int, "end": int, "unit": "char"}. Stored verbatim into
    # cg_segments.locator, which is JSONB, so it stays open for hosts that want
    # to add page/line/message-id coordinates of their own.
    locator: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> int:
        return int(self.locator["start"])

    @property
    def end(self) -> int:
        return int(self.locator["end"])

    def verify(self, source_text: str) -> bool:
        """True if this chunk still cites exactly what it says it cites.

        Cheap enough to assert in tests and after any change to a source's
        stored text. If it returns False the citation is pointing at the wrong
        span, and every node derived from this chunk is mis-attributed.
        """
        return source_text[self.start : self.end] == self.text


def chunk_text(text: str, config: ChunkConfig) -> list[Chunk]:
    """Split ``text`` into overlapping spans, preferring structural boundaries.

    Deterministic: identical input and config always yield identical output,
    including indices and offsets. Re-chunking a source therefore produces
    stable citations, which is what allows a re-run to update segments in place
    instead of orphaning the nodes that cite them.
    """
    size = config.chunk_size
    overlap = config.chunk_overlap

    if size <= 0:
        raise ValueError(f"chunk_size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {overlap}")
    if overlap >= size:
        # The window advances by (size - overlap) per step. At parity it never
        # advances, and the loop below would emit chunks until memory runs out.
        raise ValueError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({size})"
        )

    n = len(text)
    if n == 0:
        return []

    breaks = [m.start() for m in _PARAGRAPH_BREAK.finditer(text)]
    min_fill = int(size * _MIN_STRUCTURAL_FILL)

    chunks: list[Chunk] = []
    index = 0
    start = _next_word_start(text, 0)

    while start < n:
        if n - start <= size:
            end = n
        else:
            limit = start + size
            end = (
                _paragraph_cut(breaks, start, limit, min_fill)
                or _line_cut(text, start, limit, min_fill)
                or _word_cut_at_or_before(text, start, limit)
                # No whitespace anywhere in the window: a single token longer
                # than chunk_size (a base64 blob, a long URL). Overrun the
                # window rather than cut the token — the overrun is bounded by
                # one token, whereas a mid-word cut corrupts both halves and
                # the offsets can no longer be used to quote a readable span.
                or _word_end_after(text, limit)
            )

        # text[start] is non-whitespace by construction, so only the tail needs
        # trimming, and it is trimmed by moving the offset — not by rstrip()ing
        # a copy, which would leave the locator describing a longer span than
        # the text it is attached to.
        span_end = end
        while span_end > start and text[span_end - 1].isspace():
            span_end -= 1

        if span_end > start:
            chunks.append(
                Chunk(
                    text=text[start:span_end],
                    index=index,
                    locator={"start": start, "end": span_end, "unit": "char"},
                )
            )
            index += 1

        if end >= n:
            break

        # Step back by the overlap, then forward to a word start. Snapping
        # forward can land past `end` when the step-back lands inside the
        # chunk's final word; that yields zero overlap for this pair but never
        # skips text, because cuts only ever fall on word boundaries.
        start = _next_word_start(text, max(end - overlap, start + 1))

    return chunks


def _next_word_start(text: str, i: int) -> int:
    """First index at or after ``i`` that begins a word."""
    n = len(text)
    j = max(i, 0)
    # Landing mid-word means the leading fragment would be a partial word;
    # discard it by advancing past the word it belongs to.
    if 0 < j < n and not text[j].isspace() and not text[j - 1].isspace():
        while j < n and not text[j].isspace():
            j += 1
    while j < n and text[j].isspace():
        j += 1
    return j


def _paragraph_cut(
    breaks: list[int], start: int, limit: int, min_fill: int
) -> int | None:
    """Largest paragraph boundary in ``(start, limit]``, if it fills enough."""
    i = bisect_right(breaks, limit) - 1
    if i < 0:
        return None
    b = breaks[i]
    # Only the largest candidate is worth testing: every earlier break fills
    # less, so if this one fails the floor they all do.
    if b <= start or b - start < min_fill:
        return None
    return b


def _line_cut(text: str, start: int, limit: int, min_fill: int) -> int | None:
    """Largest single newline in ``(start, limit]``, if it fills enough.

    Second tier because markdown lists and tables carry one item per line with
    no blank lines between them; without this, list-heavy sources get cut
    mid-row at an arbitrary space.
    """
    floor = start + min_fill
    if floor >= limit:
        return None
    b = text.rfind("\n", floor, limit)
    return b if b > start else None


def _word_cut_at_or_before(text: str, start: int, limit: int) -> int | None:
    """Largest cut in ``(start, limit]`` that does not fall inside a word."""
    n = len(text)
    e = min(limit, n)
    while e > start:
        if e == n or text[e].isspace() or text[e - 1].isspace():
            return e
        e -= 1
    return None


def _word_end_after(text: str, i: int) -> int:
    """End of the word that index ``i`` falls inside."""
    n = len(text)
    e = min(i, n)
    while e < n and not text[e].isspace():
        e += 1
    return e


__all__ = ["Chunk", "chunk_text"]
