"""
chunker.py — Three chunking strategies for the Novus Bank RAG pipeline.

Week 2 replaces the naive fixed-size baseline (ingest.py) with two
improved strategies. All three are compared via eval_harness.py A/B test.

Usage:
    from scripts.chunker import chunk_text
    chunks = chunk_text(text, strategy="sliding_window")

Strategies:
    fixed_size      — 500-char slices, no overlap (Week 1 baseline)
    sliding_window  — fixed-size chunks with configurable overlap
    sentence_aware  — paragraph-boundary-preserving merge
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# A1.1 baseline (preserved for A/B comparison)
# ---------------------------------------------------------------------------

def fixed_size_chunk(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into fixed-size character slices with no overlap.

    This is the Week 1 baseline. Hard cuts at every chunk_size boundary
    frequently bisect numbered policy lists, causing boundary-split failures
    on multi-sentence rules (debt: no-chunk-overlap).
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# A1.1 sliding window
# ---------------------------------------------------------------------------

def sliding_window_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fixed-size chunks with overlap between adjacent windows.

    Each new chunk starts (chunk_size - overlap) characters after the
    previous one, so adjacent chunks share `overlap` characters of context.
    This eliminates hard boundary cuts for sentences that straddle a chunk
    boundary — the sentence appears in full in at least one chunk.

    Design choice: overlap=100 (20% of chunk_size) balances coverage against
    embedding cost. Larger overlap means more redundant chunks and higher
    ingest cost; smaller overlap reduces the benefit.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap  # the overlap formula
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# A1.2 sentence-aware / paragraph-preserving
# ---------------------------------------------------------------------------

def sentence_aware_chunk(text: str, chunk_size: int = 500) -> list[str]:
    """Split on paragraph breaks then merge small paragraphs up to chunk_size.

    Steps:
    1. Split on double-newlines (markdown paragraph boundaries).
    2. Accumulate paragraphs into a current chunk until adding the next
       paragraph would exceed chunk_size.
    3. Flush the current chunk and start a new one.

    This keeps logical units together — numbered policy lists and conditions
    stay in the same chunk rather than being split mid-item. The tradeoff is
    variable chunk sizes: very long paragraphs become their own oversized
    chunks, and very short documents may yield only one chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2  # +2 for the "\n\n" separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

STRATEGIES = {
    "fixed_size": fixed_size_chunk,
    "sliding_window": sliding_window_chunk,
    "sentence_aware": sentence_aware_chunk,
}


def chunk_text(text: str, strategy: str = "fixed_size", **kwargs) -> list[str]:
    """Route to the requested chunking strategy.

    Args:
        text:     Full document text.
        strategy: One of "fixed_size", "sliding_window", "sentence_aware".
        **kwargs: Forwarded to the strategy function (e.g. chunk_size, overlap).

    Raises:
        ValueError: If strategy is not one of the known keys.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[strategy](text, **kwargs)
