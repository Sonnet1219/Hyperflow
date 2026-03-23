from dataclasses import dataclass, field


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
    "organization",
    "architectural structure",
)


def get_gliner_labels_for_corpus(corpus_name: str) -> list[str]:
    normalized_name = corpus_name.lower()
    if "novel" in normalized_name:
        return list(NOVEL_GLINER_LABELS)
    return list(MEDICAL_GLINER_LABELS)


@dataclass
class HyperflowConfig:
    save_dir: str = "./index_store"
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    llm_model_name: str = "gpt-4o-mini"
    query_instruction_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    # Frontier expansion parameters
    diffusion_max_hops: int = 3         # max BFS hops from seed entities
    diffusion_top_k: int = 15           # new entities discovered per hop
    hop_decay: float = 0.7              # score decay per hop (score × decay^hop)
    # SU conductance gating
    conductance_floor: float = 0.3           # query-SU sim below this -> zero conductance
    conductance_gamma: float = 0.5           # power exponent for conductance curve (< 1 broadens mid-range)
    # Semantic unit chunking
    semantic_unit_percentile: int = 80  # Kamradt percentile for semantic unit boundary detection
    # Attribute fallback
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    # Reranker
    use_reranker: bool = True
    reranker_model_name: str = "Qwen/Qwen3-Reranker-4B"
    reranker_candidate_top_k: int = 30
    reranker_batch_size: int = 8
    reranker_max_length: int = 4096
    reranker_instruction: str = (
        "Given a multi-hop question, judge whether the document contains evidence "
        "that helps answer the question, either directly or as an intermediate bridge."
    )
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
    # NER backend
    ner_backend: str = "gliner"  # "gliner" or "spacy"
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_threshold: float = 0.3
    gliner_labels: list[str] = field(default_factory=lambda: list(MEDICAL_GLINER_LABELS))
    enable_gliner_long_text_windowing: bool = True
    gliner_window_overlap_sentences: int = 1
    min_entity_length: int = 3
