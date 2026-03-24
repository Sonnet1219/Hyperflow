"""Hyperflow: Hypergraph-based retrieval-augmented generation engine."""

from hyperflow.config import HyperflowConfig
from hyperflow.embedding_store import EmbeddingStore
from hyperflow.entity_normalization import merge_similar_entities
from hyperflow.ner import SpacyNER, GLiNERExtractor
from hyperflow.reranker import QwenReranker
from hyperflow.chunker import create_semantic_units
from hyperflow.frontier import frontier_expansion
from hyperflow.knowledge_graph import KnowledgeGraph, dense_retrieval
from hyperflow.utils import LLM_Model

import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Hyperflow:
    def __init__(self, save_dir="./output", llm_model_name="gpt-4o-mini",
                 embedding_model_name="BAAI/bge-large-en-v1.5", **kwargs):
        self.config = HyperflowConfig(
            save_dir=save_dir,
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            **kwargs,
        )
        logger.info("Initializing Hyperflow with config: %s", self.config)
        logger.info("Retrieval method: Frontier Expansion")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", self.device)

        # Load embedding model (needed for indexing and retrieval)
        self.embedding_model = SentenceTransformer(
            embedding_model_name, device=str(self.device)
        )
        # LLM is lazy-loaded on first use (only needed for rag_qa, not indexing/retrieval)
        self._llm_model = None

        # Initialize stores
        self._init_embedding_stores()

        # Load NER
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        if self.config.ner_backend == "gliner":
            self.ner_extractor = GLiNERExtractor(
                model_name=self.config.gliner_model,
                labels=self.config.gliner_labels,
                threshold=self.config.gliner_threshold,
                min_entity_length=self.config.min_entity_length,
                enable_long_text_windowing=self.config.enable_gliner_long_text_windowing,
                window_overlap_sentences=self.config.gliner_window_overlap_sentences,
                normalizer=self.spacy_ner.normalizer,
                device=str(self.device),
            )
            logger.info("NER backend: GLiNER")
        else:
            self.ner_extractor = self.spacy_ner
            logger.info("NER backend: spaCy")

        # Graph and reranker
        self.knowledge_graph = KnowledgeGraph()
        self.reranker = None

    @property
    def llm_model(self):
        if self._llm_model is None:
            self._llm_model = LLM_Model(self.config.llm_model_name)
        return self._llm_model

    def _init_embedding_stores(self):
        save_dir = self.config.save_dir
        self.passage_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "passage_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="passage"
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "entity_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="entity"
        )
        self.su_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "su_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="su"
        )

    # ====== Indexing ======

    def index(self, docs):
        """Build the hypergraph index from documents.

        Args:
            docs: list of passage strings (e.g. ["0:chunk_text", "1:chunk_text", ...])
        """
        self.passage_embedding_store.insert_text(docs)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        cached_passage_entities, cached_su_entities, uncached_passage_ids = \
            self._load_cached_ner(hash_id_to_passage.keys())
        logger.info(
            "NER cache status: cached_passages=%s, uncached_passages=%s",
            len(cached_passage_entities), len(uncached_passage_ids),
        )

        if len(uncached_passage_ids) > 0:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in uncached_passage_ids}

            # Step 1: Create semantic units via Kamradt Percentile
            logger.info("Creating semantic units via Kamradt Percentile (p=%s)...",
                        self.config.semantic_unit_percentile)
            all_su_texts = []
            passage_to_su_texts = {}
            for p_hash_id, p_text in tqdm(new_hash_id_to_passage.items(), desc="Semantic Unit Chunking"):
                su_texts = create_semantic_units(
                    p_text, self.spacy_ner.spacy_model,
                    self.embedding_model,
                    self.config.semantic_unit_percentile
                )
                passage_to_su_texts[p_hash_id] = su_texts
                all_su_texts.extend(su_texts)
            logger.info("Created %s semantic units from %s passages (avg %.1f SU/passage)",
                        len(all_su_texts), len(new_hash_id_to_passage),
                        len(all_su_texts) / max(len(new_hash_id_to_passage), 1))

            # Step 2: Run NER on each semantic unit
            new_passage_entities = {}
            new_su_entities = {}
            for p_hash_id, su_texts in tqdm(passage_to_su_texts.items(), desc="Entity Extraction"):
                passage_entities = set()
                for su_text in su_texts:
                    su_ents = self.ner_extractor.extract_entities_from_text(su_text)
                    if su_ents:
                        new_su_entities[su_text] = su_ents
                        passage_entities.update(su_ents)
                new_passage_entities[p_hash_id] = list(passage_entities)

            # Merge new with cached
            cached_passage_entities.update(new_passage_entities)
            cached_su_entities.update(new_su_entities)
        else:
            logger.info("All passages already have cached NER results; skipping NER recomputation.")

        self._persist_ner_data(cached_passage_entities, cached_su_entities)

        # Build graph structures
        entity_nodes, su_nodes, passage_to_entities, entity_to_su, _ = \
            self.knowledge_graph.build_node_edge_maps(cached_passage_entities, cached_su_entities)

        self.su_embedding_store.insert_text(list(su_nodes))

        # Merge semantically similar entities before inserting into store
        entity_nodes, cached_passage_entities, cached_su_entities, alias_map = \
            merge_similar_entities(
                list(entity_nodes), self.embedding_model,
                cached_passage_entities, cached_su_entities,
                threshold=0.93, batch_size=self.config.batch_size,
            )
        if alias_map:
            self._persist_ner_data(cached_passage_entities, cached_su_entities)
            _, su_nodes, passage_to_entities, entity_to_su, _ = \
                self.knowledge_graph.build_node_edge_maps(cached_passage_entities, cached_su_entities)

        self.entity_embedding_store.insert_text(list(entity_nodes))

        self.knowledge_graph.build_entity_su_mapping(
            entity_to_su, self.entity_embedding_store, self.su_embedding_store
        )
        self.knowledge_graph.link_entities_to_passages(
            passage_to_entities, self.passage_embedding_store, self.entity_embedding_store
        )
        self.knowledge_graph.link_adjacent_passages(self.passage_embedding_store)

    def _load_cached_ner(self, passage_hash_ids):
        self._ner_cache_path = os.path.join(self.config.save_dir, "ner_results.json")
        if os.path.exists(self._ner_cache_path):
            with open(self._ner_cache_path) as f:
                cached = json.load(f)
            cached_passage_entities = cached["passage_hash_id_to_entities"]
            cached_su_entities = cached.get("su_to_entities",
                                            cached.get("sentence_to_entities", {}))
            uncached_ids = set(passage_hash_ids) - set(cached_passage_entities.keys())
            return cached_passage_entities, cached_su_entities, uncached_ids
        return {}, {}, set(passage_hash_ids)

    def _persist_ner_data(self, passage_entities, su_entities):
        os.makedirs(os.path.dirname(self._ner_cache_path), exist_ok=True)
        with open(self._ner_cache_path, "w") as f:
            json.dump({
                "passage_hash_id_to_entities": passage_entities,
                "su_to_entities": su_entities,
            }, f)

    # ====== Retrieval ======

    def retrieve(self, queries, num_to_retrieve=None):
        """Retrieve relevant passages for a list of queries.

        Args:
            queries: list of query strings
            num_to_retrieve: number of passages to return per query (default: config.retrieval_top_k)

        Returns:
            list of dicts: [{"query": str, "passages": list[str], "scores": list[float]}, ...]
        """
        top_k = num_to_retrieve or self.config.retrieval_top_k
        use_rerank = self.config.use_reranker

        if use_rerank:
            self._lazy_load_reranker()
            logger.info("Reranking enabled: top %s candidates -> top %s final passages",
                        self.config.reranker_candidate_top_k, top_k)
        else:
            logger.info("Reranking disabled. Using dual-channel scoring (top %s).", top_k)

        self._prepare_retrieval_cache()

        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving"):
            query_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True, show_progress_bar=False,
                batch_size=self.config.batch_size,
                prompt=self.config.query_instruction_prefix
            )
            _, seed_indices, seed_texts, seed_hash_ids, seed_scores = \
                self.extract_seed_entities(query)

            if len(seed_texts) != 0:
                sorted_hash_ids, sorted_scores = self._diffuse_from_seeds(
                    query, query_embedding, seed_indices, seed_hash_ids, seed_scores
                )
                if use_rerank:
                    candidate_hash_ids = sorted_hash_ids[:self.config.reranker_candidate_top_k]
                    candidate_passages = [
                        self.passage_embedding_store.hash_id_to_text[h]
                        for h in candidate_hash_ids
                    ]
                    _, final_scores, final_passages = self._rerank_passages(
                        query, candidate_hash_ids, candidate_passages, top_k
                    )
                else:
                    top_hash_ids = sorted_hash_ids[:top_k]
                    final_scores = [float(s) for s in sorted_scores[:top_k]]
                    final_passages = [
                        self.passage_embedding_store.hash_id_to_text[h]
                        for h in top_hash_ids
                    ]
            else:
                # Fallback: dense retrieval when no entities found
                sorted_indices, sorted_scores = dense_retrieval(
                    self._passage_embeddings, query_embedding
                )
                if use_rerank:
                    candidate_indices = sorted_indices[:self.config.reranker_candidate_top_k]
                    candidate_hash_ids = [self.passage_embedding_store.hash_ids[idx] for idx in candidate_indices]
                    candidate_passages = [self.passage_embedding_store.texts[idx] for idx in candidate_indices]
                    _, final_scores, final_passages = self._rerank_passages(
                        query, candidate_hash_ids, candidate_passages, top_k
                    )
                else:
                    top_indices = sorted_indices[:top_k]
                    final_scores = [float(sorted_scores[i]) for i in range(len(top_indices))]
                    final_passages = [self.passage_embedding_store.texts[idx] for idx in top_indices]

            retrieval_results.append({
                "query": query,
                "passages": final_passages,
                "scores": final_scores,
            })
        return retrieval_results

    def _prepare_retrieval_cache(self):
        """Cache embeddings and graph lookups. Called once before retrieval loop."""
        self._entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self._entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self._passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self._passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self._su_embeddings = np.array(self.su_embedding_store.embeddings)

        # Build hypergraph
        logger.info("Building hypergraph...")
        self.knowledge_graph.build_hypergraph(
            self.entity_embedding_store, self.su_embedding_store, self.device,
        )

        # Precompute entity IDF for dual-channel scoring
        kg = self.knowledge_graph
        num_passages = len(self._passage_hash_ids)
        self._entity_idf = {}
        for p_hid, entity_weights in kg.edge_weights.items():
            for e_hid in entity_weights:
                if e_hid not in self._entity_idf:
                    self._entity_idf[e_hid] = 0
                self._entity_idf[e_hid] += 1
        for e_hid in self._entity_idf:
            self._entity_idf[e_hid] = np.log(num_passages / max(self._entity_idf[e_hid], 1))

    def extract_seed_entities(self, query):
        """Extract named entities from query and find closest indexed entities."""
        query_entities = list(self.ner_extractor.question_ner(query))
        if len(query_entities) == 0:
            return [], [], [], [], []
        query_entity_embeddings = self.embedding_model.encode(
            query_entities, normalize_embeddings=True, show_progress_bar=False,
            batch_size=self.config.batch_size,
            prompt=self.config.query_instruction_prefix
        )
        similarities = np.dot(self._entity_embeddings, query_entity_embeddings.T)
        seed_indices = []
        seed_texts = []
        seed_hash_ids = []
        seed_scores = []
        for q_idx in range(len(query_entities)):
            scores = similarities[:, q_idx]
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_hash_id = self._entity_hash_ids[best_idx]
            best_text = self.entity_embedding_store.hash_id_to_text[best_hash_id]
            seed_indices.append(best_idx)
            seed_texts.append(best_text)
            seed_hash_ids.append(best_hash_id)
            seed_scores.append(best_score)
        return query_entities, seed_indices, seed_texts, seed_hash_ids, seed_scores

    def _diffuse_from_seeds(self, query, query_embedding, seed_indices,
                            seed_hash_ids, seed_scores):
        """Run frontier expansion + dual-channel evidence fusion.

        Channel 1 (semantic): cosine(passage_embedding, query_embedding)
        Channel 2 (coverage): activation-weighted entity coverage with IDF

        final_score(p) = λ · semantic(p) + (1-λ) · coverage(p)
        """
        kg = self.knowledge_graph

        activated_entities = frontier_expansion(
            config=self.config,
            hypergraph=kg._hypergraph,
            entity_hash_ids=self._entity_hash_ids,
            su_embeddings=self._su_embeddings,
            question_embedding=query_embedding,
            seed_entity_indices=seed_indices,
            seed_entity_hash_ids=seed_hash_ids,
            seed_entity_scores=seed_scores,
        )

        # Channel 1: Dense semantic similarity
        dense_sims = np.dot(self._passage_embeddings, query_embedding)

        # Channel 2: Activation-weighted entity coverage with IDF
        coverage_scores = np.zeros(len(self._passage_hash_ids))
        total_activation = sum(v[1] for v in activated_entities.values())
        if total_activation > 0:
            for p_idx, p_hid in enumerate(self._passage_hash_ids):
                if p_hid in kg.edge_weights:
                    cov = 0.0
                    for e_hid in kg.edge_weights[p_hid]:
                        if e_hid in activated_entities:
                            act_score = activated_entities[e_hid][1]
                            idf = self._entity_idf.get(e_hid, 1.0)
                            cov += act_score * idf
                    coverage_scores[p_idx] = cov / total_activation

        # Normalize both to [0, 1]
        d_max = dense_sims.max()
        dense_norm = dense_sims / d_max if d_max > 0 else dense_sims
        c_max = coverage_scores.max()
        coverage_norm = coverage_scores / c_max if c_max > 0 else coverage_scores

        # Fuse
        lam = self.config.scoring_lambda
        final_scores = lam * dense_norm + (1 - lam) * coverage_norm

        # Sort and return
        sorted_indices = np.argsort(final_scores)[::-1]
        sorted_hash_ids = [self._passage_hash_ids[i] for i in sorted_indices]
        sorted_scores = final_scores[sorted_indices].tolist()
        return sorted_hash_ids, sorted_scores

    def _lazy_load_reranker(self):
        if self.reranker is not None:
            return
        self.reranker = QwenReranker(
            model_name=self.config.reranker_model_name,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
            instruction=self.config.reranker_instruction,
        )

    def _rerank_passages(self, query, candidate_hash_ids, candidate_passages, top_k):
        """Rerank candidate passages using the reranker model."""
        if len(candidate_passages) == 0:
            return [], [], []
        rerank_scores = self.reranker.score(query, candidate_passages)
        reranked_indices = np.argsort(np.asarray(rerank_scores))[::-1]
        selected_indices = reranked_indices[:top_k]

        final_hash_ids = [candidate_hash_ids[idx] for idx in selected_indices]
        final_scores = [float(rerank_scores[idx]) for idx in selected_indices]
        final_passages = [candidate_passages[idx] for idx in selected_indices]
        return final_hash_ids, final_scores, final_passages

    # ====== QA ======

    def rag_qa(self, queries, num_to_retrieve=None):
        """Retrieve passages and generate answers for a list of queries.

        Args:
            queries: list of query strings
            num_to_retrieve: number of passages per query (default: config.retrieval_top_k)

        Returns:
            list of dicts: [{"query": str, "answer": str, "passages": list[str]}, ...]
        """
        retrieval_results = self.retrieve(queries, num_to_retrieve)
        system_prompt = (
            "As an advanced reading comprehension assistant, your task is to analyze text passages "
            "and corresponding questions meticulously. Your response start after \"Thought: \", "
            "where you will methodically break down the reasoning process, illustrating how you "
            "arrive at conclusions. Conclude with \"Answer: \" to present a concise, definitive "
            "response, devoid of additional elaborations."
        )
        all_messages = []
        for result in retrieval_results:
            prompt_user = ""
            for passage in result["passages"]:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {result['query']}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result, result in zip(all_qa_results, retrieval_results):
            try:
                answer = qa_result.split('Answer:')[1].strip()
            except Exception:
                answer = qa_result
            result["answer"] = answer
        return retrieval_results

