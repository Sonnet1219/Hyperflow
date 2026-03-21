from dataclasses import dataclass, field
from hyperflow.utils import LLM_Model


@dataclass
class HyperflowConfig:
    dataset_name: str
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    query_instruction_prefix: str = "Represent this sentence for searching relevant passages: "
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
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
    sentence_gate_threshold: float = 0.5  # block sentences with query-similarity below this
    diffusion_top_k: int = 10           # activate top-K entities per diffusion round
    # Attribute fallback
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    # Reranker
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
