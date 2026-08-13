"""Segmentation, and the one invariant that makes a citation evidence.

Every chunk must satisfy

    source_text[locator["start"]:locator["end"]] == chunk.text

byte for byte. A node extracted from a chunk cites that chunk's span, and the
citation is what makes an extraction error recoverable: given offsets you can
show a reviewer the exact sentence a claim came from. An offset that is off by
a stripped newline degrades a citation from evidence to a hint — and it degrades
*silently*, because nothing downstream ever compares the two again.

So most of this file is one assertion applied to awkward input.
"""

from __future__ import annotations

import pytest

from contextgraph.config import ChunkConfig
from contextgraph.embeddings.chunker import chunk_text

CORPUS = {
    "empty": "",
    "one word": "hello",
    "short sentence": "The quick brown fox jumps over the lazy dog.",
    "paragraphs": "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(8)),
    "markdown list": "\n".join(f"- item {i} with some trailing text" for i in range(200)),
    "no whitespace at all": "x" * 5000,
    "one enormous token": "short intro " + ("y" * 3000) + " short outro",
    "blank line runs": "First.\n\n\n\n\nSecond." + (" tail" * 400),
    "trailing whitespace": ("word " * 500) + "\n\n\n   \t  \n",
    "leading whitespace": "\n\n\t   " + ("word " * 500),
    "unicode": ("héllo wörld — naïve café 日本語テキスト ✨ " * 120),
    "windows newlines": "\r\n".join(f"line {i} of text" for i in range(400)),
}

CONFIGS = [
    ChunkConfig(chunk_size=1200, chunk_overlap=150),
    ChunkConfig(chunk_size=100, chunk_overlap=10),
    ChunkConfig(chunk_size=64, chunk_overlap=0),
    ChunkConfig(chunk_size=5000, chunk_overlap=1000),
]


@pytest.mark.parametrize("name", list(CORPUS))
@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"size{c.chunk_size}")
class TestTheCitationInvariant:
    def test_every_chunk_slices_back_to_itself(self, name, config):
        source = CORPUS[name]
        for chunk in chunk_text(source, config):
            assert chunk.verify(source), (
                f"{name}: locator {chunk.locator} does not describe its own text"
            )
            assert source[chunk.start : chunk.end] == chunk.text

    def test_no_text_is_silently_dropped(self, name, config):
        """Chunks may overlap, but between them they must cover every word.

        A gap here loses content with no error — the source is stored intact
        and simply becomes unsearchable, which looks identical to "nothing in
        this source matched".
        """
        source = CORPUS[name]
        chunks = chunk_text(source, config)
        if not source.strip():
            assert chunks == []
            return
        covered = set()
        for c in chunks:
            covered.update(range(c.start, c.end))
        missing = [
            i for i, ch in enumerate(source) if not ch.isspace() and i not in covered
        ]
        assert not missing, f"{name}: {len(missing)} characters covered by no chunk"

    def test_chunks_are_ordered_and_indexed_contiguously(self, name, config):
        chunks = chunk_text(CORPUS[name], config)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        starts = [c.start for c in chunks]
        assert starts == sorted(starts)

    def test_no_chunk_is_empty_or_pure_whitespace(self, name, config):
        for c in chunks_of(CORPUS[name], config):
            assert c.text.strip(), "an empty chunk embeds to noise and cites nothing"

    def test_determinism(self, name, config):
        """Re-chunking must produce identical citations.

        This is what lets a re-run update segments in place instead of
        orphaning every node that cites them.
        """
        source = CORPUS[name]
        first = chunk_text(source, config)
        second = chunk_text(source, config)
        assert [(c.text, c.locator, c.index) for c in first] == [
            (c.text, c.locator, c.index) for c in second
        ]


def chunks_of(source, config):
    return chunk_text(source, config)


class TestRejectedConfiguration:
    """Bad config fails loudly rather than looping forever."""

    def test_overlap_equal_to_size_is_refused(self):
        # The window advances by (size - overlap). At parity it never advances
        # and the loop emits chunks until memory runs out.
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            chunk_text("some text", ChunkConfig(chunk_size=100, chunk_overlap=100))

    def test_overlap_larger_than_size_is_refused(self):
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            chunk_text("some text", ChunkConfig(chunk_size=100, chunk_overlap=200))

    def test_zero_size_is_refused(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("some text", ChunkConfig(chunk_size=0, chunk_overlap=0))

    def test_negative_overlap_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            chunk_text("some text", ChunkConfig(chunk_size=100, chunk_overlap=-1))


class TestBoundaryPreference:
    def test_a_paragraph_break_is_preferred_over_an_arbitrary_cut(self):
        para = "word " * 100
        source = (para + "\n\n") * 4
        chunks = chunk_text(source, ChunkConfig(chunk_size=700, chunk_overlap=50))
        assert len(chunks) > 1
        # A structural cut leaves no dangling partial word at the boundary.
        for c in chunks[:-1]:
            assert not c.text.endswith(" ")

    def test_a_short_heading_does_not_produce_a_tiny_chunk(self):
        """Without a fill floor, greedy "last break that fits" emits a
        20-character chunk whenever a heading precedes a long paragraph —
        common in markdown, and the result embeds to noise."""
        source = "# Heading\n\n" + ("word " * 400)
        chunks = chunk_text(source, ChunkConfig(chunk_size=600, chunk_overlap=50))
        assert all(len(c.text) > 50 for c in chunks[:-1])

    def test_a_token_longer_than_the_window_is_not_split(self):
        """Overrunning by one token beats corrupting both halves — the offsets
        must still quote a readable span."""
        blob = "z" * 900
        source = f"intro text {blob} outro text"
        chunks = chunk_text(source, ChunkConfig(chunk_size=200, chunk_overlap=20))
        assert any(blob in c.text for c in chunks), "the token was cut in half"
        for c in chunks:
            assert c.verify(source)
