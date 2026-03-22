from dataclasses import dataclass, field

from hyperflow.utils import LLM_Model


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
    dataset_name: str
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    query_instruction_prefix: str = "Represent this sentence for searching relevant passages: "
    llm_model: LLM_Model = None
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    # Spectral diffusion parameters
    diffusion_alpha: float = 0.85       # diffusion weight (higher = more exploration, less teleport)
    diffusion_max_iter: int = 10        # convergence iteration limit
    convergence_tol: float = 1e-4       # L2 norm convergence threshold
    semantic_unit_gate_threshold: float = 0.5  # (legacy) binary gate threshold; unused by anisotropic diffusion
    diffusion_top_k: int = 10           # activate top-K entities per diffusion round
    # Anisotropic conductance parameters
    conductance_floor: float = 0.3           # query-SU sim below this → zero conductance
    conductance_gamma: float = 0.5           # power exponent for conductance curve (< 1 broadens mid-range)
    conductance_diversity_beta: float = 0.3  # diversity penalty strength (0 = disabled)
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
