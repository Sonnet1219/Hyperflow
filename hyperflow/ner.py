import logging
import re

import spacy

from hyperflow.entity_normalization import EntityNormalizer, normalize_entity_list


logger = logging.getLogger(__name__)


MEDICAL_GLINER_LABELS = (
    "disease or cancer type",
    "disease abbreviation",
    "symptom or clinical sign",
    "risk factor or exposure",
    "anatomy or body site",
    "diagnostic test or imaging",
    "pathology finding or histology",
    "stage or grade",
    "treatment or procedure",
    "drug or regimen",
    "biomarker or receptor",
    "gene or mutation",
)

NOVEL_GLINER_LABELS = (
    "person",
    "civilization",
    "location",
    "artifact",
    "deity",
    "language",
    "historical event",
    "ancient text or scripture",
    "symbol or totem",
    "architectural structure",
)


def get_gliner_labels_for_corpus(corpus_name: str) -> list[str]:
    normalized_name = corpus_name.lower()
    if "novel" in normalized_name:
        return list(NOVEL_GLINER_LABELS)
    return list(MEDICAL_GLINER_LABELS)


class SpacyNER:
    def __init__(self, spacy_model):
        spacy.prefer_gpu()
        self.spacy_model = spacy.load(spacy_model)
        self.normalizer = EntityNormalizer(self.spacy_model)
        logger.info("spaCy NER loaded with model: %s (GPU enabled)", spacy_model)

    def extract_entities_from_text(self, text):
        """Extract entities from a single text string (semantic unit)."""
        doc = self.spacy_model(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                continue
            entities.add(self.normalizer.canonicalize(ent.text.lower()))
        return list(entities)

    def question_ner(self, question: str):
        doc = self.spacy_model(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(self.normalizer.canonicalize(ent.text.lower()))
        return question_entities


class GLiNERExtractor:
    """Zero-shot NER using GLiNER bi-encoder for domain-specific entity extraction."""

    def __init__(self, model_name="knowledgator/gliner-bi-large-v2.0",
                 labels=None, threshold=0.3, min_entity_length=3,
                 enable_long_text_windowing=True, window_overlap_sentences=1,
                 normalizer=None, device=None):
        import torch
        from gliner import GLiNER
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained(model_name).to(self.device)
        self.labels = labels or []
        self.threshold = threshold
        self.min_length = min_entity_length
        self.normalizer = normalizer
        self.enable_long_text_windowing = enable_long_text_windowing
        self.window_overlap_sentences = max(0, window_overlap_sentences)
        self.max_length = getattr(self.model.config, "max_len", None)
        self.sentencizer = self._build_sentencizer()
        logger.info(
            "GLiNER loaded: model=%s, device=%s, labels=%s, threshold=%s, max_len=%s, long_text_windowing=%s, overlap_sentences=%s",
            model_name,
            self.device,
            self.labels,
            self.threshold,
            self.max_length,
            self.enable_long_text_windowing,
            self.window_overlap_sentences,
        )

    def _build_sentencizer(self):
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception as exc:
            logger.warning("Falling back to regex sentence splitting for GLiNER windowing: %s", exc)
            return None

    def _predict_entities(self, text):
        return self.model.predict_entities(text, self.labels, threshold=self.threshold)

    def _normalize_entity_texts(self, entities):
        if self.normalizer is not None:
            return normalize_entity_list(self.normalizer, entities, self.min_length)
        normalized_entities = []
        seen = set()
        for entity in entities:
            entity_text = entity["text"].lower().strip()
            if len(entity_text) < self.min_length or entity_text in seen:
                continue
            seen.add(entity_text)
            normalized_entities.append(entity_text)
        return normalized_entities

    def _tokenize_text(self, text):
        tokens, _, _ = self.model.prepare_inputs([text])
        return tokens[0]

    def _split_into_sentences(self, text):
        if self.sentencizer is not None:
            doc = self.sentencizer(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if sentences:
                return sentences
        return [segment.strip() for segment in re.split(r'(?<=[.!?])\s+', text) if segment.strip()]

    def _hard_split_by_tokens(self, text):
        tokens, starts, ends = self.model.prepare_inputs([text])
        tokens = tokens[0]
        starts = starts[0]
        ends = ends[0]
        if len(tokens) == 0:
            return [text.strip()] if text.strip() else []

        windows = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.max_length, len(tokens))
            window_text = text[starts[start_idx]:ends[end_idx - 1]].strip()
            if window_text:
                windows.append(window_text)
            if end_idx >= len(tokens):
                break
            start_idx = end_idx
        return windows

    def _split_long_text_into_windows(self, text):
        sentences = self._split_into_sentences(text)
        if not sentences:
            return self._hard_split_by_tokens(text)

        sentence_tokens, _, _ = self.model.prepare_inputs(sentences)
        filtered_sentences = [
            (sentence, len(tokens))
            for sentence, tokens in zip(sentences, sentence_tokens)
            if len(tokens) > 0
        ]
        if not filtered_sentences:
            return self._hard_split_by_tokens(text)

        sentences = [sentence for sentence, _ in filtered_sentences]
        token_counts = [token_count for _, token_count in filtered_sentences]

        windows = []
        start_sentence_idx = 0
        while start_sentence_idx < len(sentences):
            if token_counts[start_sentence_idx] > self.max_length:
                windows.extend(self._hard_split_by_tokens(sentences[start_sentence_idx]))
                start_sentence_idx += 1
                continue

            end_sentence_idx = start_sentence_idx
            token_count = token_counts[start_sentence_idx]
            while (
                end_sentence_idx + 1 < len(sentences)
                and token_count + token_counts[end_sentence_idx + 1] <= self.max_length
            ):
                end_sentence_idx += 1
                token_count += token_counts[end_sentence_idx]

            window_text = " ".join(sentences[start_sentence_idx:end_sentence_idx + 1]).strip()
            if window_text:
                windows.append(window_text)

            if end_sentence_idx >= len(sentences) - 1:
                break

            covered_sentence_count = end_sentence_idx - start_sentence_idx + 1
            overlap = min(self.window_overlap_sentences, covered_sentence_count - 1)
            start_sentence_idx = end_sentence_idx - overlap + 1

        return windows or self._hard_split_by_tokens(text)

    def extract_entities_from_text(self, text):
        """Extract entities from a single text string (semantic unit)."""
        if not self.enable_long_text_windowing or not self.max_length:
            return self._normalize_entity_texts(self._predict_entities(text))

        token_count = len(self._tokenize_text(text))
        if token_count <= self.max_length:
            return self._normalize_entity_texts(self._predict_entities(text))

        window_texts = self._split_long_text_into_windows(text)
        extracted_entities = []
        seen = set()
        for window_text in window_texts:
            for entity_text in self._normalize_entity_texts(self._predict_entities(window_text)):
                if entity_text in seen:
                    continue
                seen.add(entity_text)
                extracted_entities.append(entity_text)

        logger.debug(
            "Windowed GLiNER extraction split a %s-token semantic unit into %s windows",
            token_count,
            len(window_texts),
        )
        return extracted_entities

    def question_ner(self, question: str):
        """Extract entities from a question for seed matching."""
        entities = self._predict_entities(question)
        return set(self._normalize_entity_texts(entities))
