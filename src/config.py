from dataclasses import dataclass, field
from src.utils import LLM_Model
@dataclass
class LinearRAGConfig:
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    bridge_diversity_weight: float = 0.3  # β: info-theoretic diversity penalty (0=pure similarity, 1=full MMR)
    use_vectorized_retrieval: bool = False  # True for vectorized matrix computation, False for BFS iteration
    use_query_evolution: bool = False  # Enable query evolution / semantic steering across hops
    query_evolution_inertia: float = 0.7  # α: retention of original query intent (0=full evolution, 1=no evolution)
    query_evolution_steering: float = 0.5  # γ: bridge sentence steering strength
    # Hypergraph spectral diffusion
    use_hypergraph_diffusion: bool = False  # Third retrieval mode: hypergraph spectral diffusion
    hypergraph_alpha: float = 0.85  # diffusion weight (higher = more exploration, less teleport)
    hypergraph_max_iterations: int = 10  # diffusion convergence iteration limit
    hypergraph_convergence_tol: float = 1e-4  # L2 norm convergence threshold
    hypergraph_damping_gamma: float = 0.5  # hyperedge flow damping rate (0=hard dedup, 1=no damping)
    hypergraph_activation_ratio: float = 0.05  # adaptive threshold: activate entities with score >= top_score * ratio
    use_context_modulation: bool = False  # context-modulated incidence matrix H_ctx
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])