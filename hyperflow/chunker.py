"""Semantic unit chunking: 1200-token chunking + Kamradt Percentile merging."""

import numpy as np
import tiktoken
import logging

logger = logging.getLogger(__name__)

_enc = tiktoken.encoding_for_model("gpt-4")


def chunk_corpus_by_tokens(text, chunk_size=1200, overlap=100):
    """Split text into chunks of ~chunk_size tokens with overlap."""
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


def extract_sentences(text, nlp_model):
    """Extract sentences from text using spaCy sentence segmentation."""
    doc = nlp_model(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]


def kamradt_semantic_units(sentences, embeddings, percentile=80):
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


def create_semantic_units(chunk_text, nlp_model, embedding_model, percentile=80):
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
        percentile: Kamradt breakpoint percentile (default 80)

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
