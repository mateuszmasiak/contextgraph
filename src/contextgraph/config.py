"""Tunables, with the reasoning that produced each default.

Every threshold here was measured against a real corpus rather than guessed,
and the measurements are recorded because the numbers are only defensible
alongside them. If you change one, re-measure — do not reason about it from
first principles, because the first-principles answer is wrong in an
instructive way (see MERGE_THRESHOLD).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionConfig:
    """How aggressively to fold new claims into existing nodes."""

    # Cosine at/above which two SAME-TYPE nodes are auto-merged with no LLM call.
    #
    # Measured, and worth internalising before touching: on a real 100-node
    # corpus this fired ZERO times. The highest same-project/same-type pair was
    # 0.804, and two nodes with byte-identical titles scored 0.798 — because the
    # embedded text is "title\nsummary" and differing summaries pull identical
    # titles apart.
    #
    # The instinct is to lower it. Do not. True duplicates measured 0.70-0.80,
    # and genuinely distinct siblings ("Login screen" vs "Dashboard screen",
    # 0.679) occupy the SAME band — only ~0.035 separated the highest true
    # negative from the lowest true positive. Cosine ranks candidates here; it
    # cannot decide them. That is why the adjudicator exists.
    merge_threshold: float = 0.90

    # Lower bound of the band worth an LLM adjudication call. Below this, a
    # candidate is simply new.
    related_low: float = 0.74

    # How many nearest same-kind nodes to consider. Emphatically not 1: a top-1
    # lookup lets a cross-type neighbour mask the true same-type match. In real
    # data the nearest neighbour to a "Landing Page" screen was the "Landing
    # Page Journey" flow at 0.832 — the highest-scoring pair in the corpus, and
    # a merge that must never happen.
    candidate_limit: int = 5

    # Adjudicator behaviour on error. False (fail-open to "distinct") produces a
    # duplicate; True routes the decision to the review gate instead. Duplicates
    # are recoverable and merges are not, so open is the safer default for an
    # unattended pipeline.
    fail_closed: bool = False


@dataclass(frozen=True)
class RosterConfig:
    """The canonical-name roster shown to the extractor.

    This is the highest-leverage part of the whole pipeline and the least
    obvious. Measured on real data, 10 of 11 near-duplicate pairs were
    cross-run: the extractor had simply never been told what already existed,
    so each turn re-invented a surface form ("Task Status Tracking" became
    "Task Status Tracking UI") and resolution was left to recover identity from
    a signal that cannot carry it.

    Showing the extractor the existing titles moves de-duplication from
    detection to prevention. In an A/B over two sources where the second
    restated the first in drifted wording, roster-on produced 0 new nodes and
    5 merges; roster-off invented spurious extra nodes both runs.
    """

    enabled: bool = True
    # Entries offered. Measured at ~9 tokens/entry for short entity titles and
    # ~23 for full-sentence knowledge claims, so 200 lands between ~1.1k and
    # ~4.5k tokens. Truncation is entity-first, then most-recently-updated.
    limit: int = 200
    max_aliases_per_entry: int = 3


@dataclass(frozen=True)
class ChunkConfig:
    """Segmentation of raw sources for the retrieval index."""

    chunk_size: int = 1200
    chunk_overlap: int = 150


@dataclass(frozen=True)
class ExtractionConfig:
    max_tokens: int = 4096
    temperature: float = 0.1
    # Hard ceiling on source text sent to the model. Unbounded input against a
    # bounded output budget fails as a truncated object, which is
    # indistinguishable from "this source contained nothing" — permanently, and
    # with no signal. Sources beyond this are segmented and extracted in
    # batches instead.
    max_source_chars: int = 200_000


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    resolution: ResolutionConfig = ResolutionConfig()
    roster: RosterConfig = RosterConfig()
    chunking: ChunkConfig = ChunkConfig()
    extraction: ExtractionConfig = ExtractionConfig()

    # Must match the vector column width created by the migration. Changing it
    # requires a migration and a full re-embed; it is validated at startup so
    # the failure is loud rather than a silent dimension mismatch.
    embedding_dimensions: int = 1536


DEFAULT_CONFIG = Config()
