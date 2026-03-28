from dataclasses import dataclass


@dataclass
class HyperSUConfig:
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
    scoring_lambda: float = 0.5
    expansion_max_hops: int = 3
    expansion_top_k: int = 15
    hop_decay: float = 0.5
    conductance_floor: float = 0.5
    conductance_gamma: float = 1.0
    steering_alpha: float = 0.7
    steering_top_k: int = 3
    semantic_unit_percentile: int = 60
    use_reranker: bool = True
    reranker_model_name: str = "Qwen/Qwen3-Reranker-4B"
    reranker_candidate_top_k: int = 30
    reranker_batch_size: int = 8
    reranker_max_length: int = 4096
    reranker_instruction: str = (
        "Given a multi-hop question, judge whether the document contains evidence "
        "that helps answer the question, either directly or as an intermediate bridge."
    )
    langextract_model_id: str = "gpt-4o-mini"
    langextract_api_key: str | None = None
    langextract_model_url: str | None = None
    langextract_max_char_buffer: int = 1000
    langextract_extraction_passes: int = 1
    langextract_use_schema_constraints: bool = True
    entity_merge_threshold: float = 0.90
