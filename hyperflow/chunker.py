"""Semantic-aware chunking: fixed token target + semantic boundary splitting."""

import numpy as np
import tiktoken
import logging

logger = logging.getLogger(__name__)

_enc = tiktoken.encoding_for_model("gpt-4")


def _token_count(text):
    """Return token count for a string."""
    return len(_enc.encode(text))


def chunk_corpus_by_tokens(text, chunk_size=1200, overlap=100,
                           nlp_model=None, embedding_model=None,
                           search_window_ratio=0.2):
    """Split text into chunks of ~chunk_size tokens, cutting at semantic boundaries.

    When nlp_model and embedding_model are provided, the function:
      1. Splits the entire text into sentences via spaCy
      2. Accumulates sentences until approaching chunk_size tokens
      3. Within a search window (chunk_size ± search_window_ratio), finds the
         sentence boundary with the largest semantic gap (cosine distance)
      4. Cuts there, producing chunks that respect both token budget and semantics

    Falls back to pure fixed-window splitting if models are not provided.

    Args:
        text: Full corpus text
        chunk_size: Target token count per chunk (default 1200)
        overlap: Number of overlap sentences carried into the next chunk
        nlp_model: Loaded spaCy model for sentence segmentation (optional)
        embedding_model: SentenceTransformer model for embeddings (optional)
        search_window_ratio: Fraction of chunk_size to search around target (default 0.2)
    """
    if nlp_model is None or embedding_model is None:
        return _chunk_fixed(text, chunk_size, overlap)

    return _chunk_semantic(text, chunk_size, overlap, nlp_model, embedding_model,
                           search_window_ratio)


def _chunk_fixed(text, chunk_size, overlap):
    """Pure fixed-window token chunking (legacy fallback)."""
    tokens = _enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _chunk_semantic(text, chunk_size, overlap_sents, nlp_model, embedding_model,
                    search_window_ratio):
    """Fixed token target + semantic boundary chunking."""
    # 1. Sentence segmentation
    doc = nlp_model(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]

    if len(sentences) == 0:
        return [text.strip()] if text.strip() else []

    # 2. Compute token count for each sentence
    sent_tokens = [_token_count(s) for s in sentences]

    # 3. Embed all sentences (for cosine distance computation)
    embeddings = embedding_model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # 4. Compute adjacent cosine distances
    distances = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        distances.append(1.0 - sim)
    # Sentinel for last sentence (no split after last)
    distances.append(0.0)

    # 5. Greedily build chunks, cutting at best semantic boundary near chunk_size
    lo = int(chunk_size * (1 - search_window_ratio))  # e.g. 960
    hi = int(chunk_size * (1 + search_window_ratio))  # e.g. 1440

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        # Accumulate sentences until we exceed hi or run out
        cumulative = 0
        end_idx = start_idx
        while end_idx < len(sentences) and cumulative + sent_tokens[end_idx] <= hi:
            cumulative += sent_tokens[end_idx]
            end_idx += 1

        # If we've consumed all remaining sentences, emit final chunk
        if end_idx >= len(sentences):
            chunk_text = " ".join(sentences[start_idx:end_idx])
            chunks.append(chunk_text)
            break

        # Find candidate split points: sentence boundaries where cumulative tokens ∈ [lo, hi]
        # A split "after sentence i" means the chunk is sentences[start_idx:i+1]
        cumsum = 0
        candidates = []  # (sentence_index, cumulative_tokens, semantic_distance)
        for i in range(start_idx, end_idx):
            cumsum += sent_tokens[i]
            if lo <= cumsum <= hi and i + 1 < len(sentences):
                candidates.append((i, cumsum, distances[i]))

        if candidates:
            # Pick the boundary with the largest semantic gap
            best = max(candidates, key=lambda c: c[2])
            split_after = best[0]
        else:
            # No candidate in window — split at the last sentence that fits under chunk_size
            cumsum = 0
            split_after = start_idx
            for i in range(start_idx, end_idx):
                cumsum += sent_tokens[i]
                if cumsum <= chunk_size:
                    split_after = i

        chunk_text = " ".join(sentences[start_idx:split_after + 1])
        chunks.append(chunk_text)

        # Overlap: carry last `overlap_sents` sentences into next chunk
        start_idx = max(split_after + 1 - overlap_sents, split_after + 1)
        # Ensure forward progress
        if start_idx <= split_after:
            start_idx = split_after + 1

    logger.info("Semantic chunking: %d sentences -> %d chunks (target %d tokens)",
                len(sentences), len(chunks), chunk_size)
    return chunks


def extract_sentences(text, nlp_model):
    """Extract sentences from text using spaCy sentence segmentation."""
    doc = nlp_model(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]


def kamradt_semantic_units(sentences, embeddings, percentile=60):
    """
    Kamradt Percentile semantic chunking.

    Compute cosine distance between adjacent sentence embeddings.
    Split at distances above the given percentile threshold.
    Returns list of lists of sentence indices (groups).
    """
    if len(sentences) <= 1:
        return [list(range(len(sentences)))]

    # Compute adjacent cosine distances
    distances = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        distances.append(1.0 - sim)

    threshold = np.percentile(distances, percentile)

    units = []
    current = [0]
    for i in range(len(distances)):
        if distances[i] > threshold:
            units.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    units.append(current)
    return units


def create_semantic_units(chunk_text, nlp_model, embedding_model, percentile=60):
    """
    Split a chunk into semantic units using Kamradt Percentile method.

    1. Extract sentences via spaCy
    2. Embed sentences with BGE
    3. Run Kamradt to group adjacent sentences
    4. Return list of semantic unit texts

    Args:
        chunk_text: Text of a single chunk (~1200 tokens)
        nlp_model: Loaded spaCy model for sentence segmentation
        embedding_model: SentenceTransformer model for embeddings
        percentile: Kamradt breakpoint percentile (default 60)

    Returns:
        list[str]: List of semantic unit texts (each is 1+ joined sentences)
    """
    sentences = extract_sentences(chunk_text, nlp_model)

    if len(sentences) == 0:
        # Fallback: treat the whole chunk as one semantic unit
        return [chunk_text.strip()] if chunk_text.strip() else []

    if len(sentences) == 1:
        return sentences

    # Embed all sentences
    embeddings = embedding_model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )

    # Run Kamradt grouping
    groups = kamradt_semantic_units(sentences, embeddings, percentile)

    # Join sentences in each group into a semantic unit text
    su_texts = []
    for group in groups:
        su_text = " ".join(sentences[i] for i in group)
        su_texts.append(su_text)

    return su_texts
