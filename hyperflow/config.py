from dataclasses import dataclass, field

from hyperflow.ner import MEDICAL_GLINER_LABELS


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
    # Dual-channel passage scoring
    scoring_lambda: float = 0.5             # fusion weight (1.0 = pure dense, 0.0 = pure entity coverage)
    # Frontier expansion parameters
    expansion_max_hops: int = 3             # max BFS hops from seed entities
    expansion_top_k: int = 15              # new entities discovered per hop
    hop_decay: float = 0.5                  # score decay per hop (score × decay^hop)
    # SU conductance gating
    conductance_floor: float = 0.5          # query-SU sim below this -> zero conductance
    conductance_gamma: float = 1.0          # power exponent (1.0 = linear, < 1 broadens, > 1 sharpens)
    # Progressive Query Steering
    steering_alpha: float = 0.7             # retain ratio of original query (1.0 = no steering)
    steering_top_k: int = 3                 # top-K activated entities for centroid computation
    # Semantic unit chunking
    semantic_unit_percentile: int = 60      # Kamradt percentile for SU boundary detection
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
    # NER backend
    ner_backend: str = "gliner"             # "gliner" or "spacy"
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_threshold: float = 0.3
    gliner_labels: list[str] = field(default_factory=lambda: list(MEDICAL_GLINER_LABELS))
    enable_gliner_long_text_windowing: bool = True
    gliner_window_overlap_sentences: int = 1
    min_entity_length: int = 3
