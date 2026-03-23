"""Entity text normalization for deduplication across semantic units.

Applies deterministic canonicalization rules so that surface variants
of the same entity (plural, article prefix, hyphen, parenthetical)
collapse to a single canonical form before the seen-set check.

Also provides embedding-based merging for semantically equivalent
entities that survive text-level canonicalization.
"""

import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

_ARTICLE_RE = re.compile(r"^(?:the|an|a)\s+", re.IGNORECASE)
_SPACED_PAREN_RE = re.compile(r"\s+\([^)]*\)")  # only match parentheticals preceded by whitespace
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


class EntityNormalizer:
    """Stateless text canonicalizer backed by a spaCy model for lemmatization."""

    def __init__(self, spacy_model):
        self._nlp = spacy_model

    def canonicalize(self, text: str) -> str:
        """Apply canonicalization rules in fixed order.

        Order:
            1. Strip leading articles  (the/a/an)
            2. Remove parenthetical content
            3. Normalize hyphens to spaces
            4. Lemmatize via spaCy
            5. Collapse whitespace and strip
        """
        # Step 1: strip articles
        text = _ARTICLE_RE.sub("", text)

        # Step 2: remove parenthetical content
        # Handle "(s)" suffix first: drug(s) → drug, sentinel lymph node(s) → sentinel lymph node
        text = text.replace("(s)", "")
        # Remove parenthetical annotations preceded by a space: "squamous cell (epidermoid) carcinoma"
        # but preserve identifiers like "t(11;22)" or "del(5q)" where "(" directly follows text
        text = _SPACED_PAREN_RE.sub("", text)

        # Step 3: hyphens → spaces
        text = text.replace("-", " ")

        # Step 4: lemmatize
        text = self._lemmatize(text)

        # Step 5: collapse whitespace
        text = _MULTI_SPACE_RE.sub(" ", text).strip()

        return text

    def _lemmatize(self, text: str) -> str:
        doc = self._nlp(text)
        parts = []
        for token in doc:
            lemma = token.lemma_ if token.pos_ == "NOUN" else token.text
            parts.append(lemma + token.whitespace_)
        return "".join(parts)


def normalize_entity_list(normalizer: EntityNormalizer,
                          raw_entities: list[dict],
                          min_length: int = 3) -> list[str]:
    """Normalize and deduplicate a list of GLiNER-style entity dicts.

    Each dict must have a ``"text"`` key.  Returns deduplicated canonical
    entity strings in first-seen order.
    """
    normalized = []
    seen: set[str] = set()
    for entity in raw_entities:
        text = entity["text"].lower().strip()
        text = normalizer.canonicalize(text)
        if len(text) < min_length or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def normalize_entity_text(normalizer: EntityNormalizer, text: str) -> str:
    """Normalize a single already-lowered entity string."""
    return normalizer.canonicalize(text)


def merge_similar_entities(entity_texts, embedding_model,
                           passage_entities: dict, su_entities: dict,
                           threshold: float = 0.93, batch_size: int = 128):
    """Merge semantically similar entities using embedding cosine similarity.

    Entities with similarity >= threshold are grouped via union-find.
    Within each group the shortest text (ties broken alphabetically)
    becomes the canonical representative.

    This function operates on raw text lists *before* store insertion,
    so only canonical entities need to be inserted into the EmbeddingStore.

    Args:
        entity_texts: list of unique entity strings to check.
        embedding_model: SentenceTransformer used for encoding.
        passage_entities: {passage_hash_id: [entity_text, ...]}
        su_entities: {su_text: [entity_text, ...]}
        threshold: cosine similarity threshold for merging.
        batch_size: encoding batch size.

    Returns:
        (canonical_entity_texts, updated_passage_entities, updated_su_entities, alias_map)
    """
    texts = list(entity_texts)
    if len(texts) < 2:
        return texts, passage_entities, su_entities, {}

    embeddings = embedding_model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
        batch_size=batch_size,
    )
    embeddings = np.asarray(embeddings)

    # Pairwise cosine similarity
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, 0)

    # Union-Find
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    rows, cols = np.where(sim >= threshold)
    for i, j in zip(rows, cols):
        if i < j:
            union(i, j)

    # Group entities by connected component
    groups = {}
    for idx in range(len(texts)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    # Build alias map: old_text -> canonical_text
    alias_map = {}
    for members in groups.values():
        if len(members) == 1:
            continue
        member_texts = [(texts[i], i) for i in members]
        member_texts.sort(key=lambda x: (len(x[0]), x[0]))
        canonical = member_texts[0][0]
        for text, _ in member_texts[1:]:
            alias_map[text] = canonical

    if not alias_map:
        logger.info("Entity merge: no pairs above threshold %.2f", threshold)
        return texts, passage_entities, su_entities, alias_map

    logger.info("Entity merge: %d entities merged into %d groups (threshold=%.2f)",
                len(alias_map), len([m for m in groups.values() if len(m) > 1]),
                threshold)

    # Filter to canonical entities only
    canonical_texts = [t for t in texts if t not in alias_map]

    # Apply alias map to passage_entities and su_entities
    def _apply(entity_dict):
        updated = {}
        for key, ents in entity_dict.items():
            seen = set()
            new_ents = []
            for e in ents:
                canonical = alias_map.get(e, e)
                if canonical not in seen:
                    seen.add(canonical)
                    new_ents.append(canonical)
            updated[key] = new_ents
        return updated

    updated_passage = _apply(passage_entities)
    updated_su = _apply(su_entities)

    return canonical_texts, updated_passage, updated_su, alias_map
