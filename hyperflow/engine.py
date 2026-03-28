"""Hyperflow: Hypergraph-based retrieval-augmented generation engine."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from hyperflow.chunker import create_semantic_units
from hyperflow.config import HyperflowConfig
from hyperflow.embedding_store import EmbeddingStore
from hyperflow.entity_normalization import merge_entity_mentions
from hyperflow.frontier import frontier_expansion
from hyperflow.knowledge_graph import KnowledgeGraph, dense_retrieval
from hyperflow.ner import LangExtractExtractor
from hyperflow.reranker import QwenReranker
from hyperflow.utils import LLM_Model, compute_mdhash_id


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        self.embedding_model = SentenceTransformer(
            embedding_model_name, device=str(self.device)
        )
        self._llm_model = None

        self._init_embedding_stores()

        spacy.prefer_gpu()
        self.spacy_model = spacy.load(self.config.spacy_model)
        logger.info("spaCy sentence splitter loaded with model: %s", self.config.spacy_model)

        self.ner_extractor = LangExtractExtractor(
            model_id=self.config.langextract_model_id,
            api_key=self.config.langextract_api_key,
            model_url=self.config.langextract_model_url,
            max_char_buffer=self.config.langextract_max_char_buffer,
            extraction_passes=self.config.langextract_extraction_passes,
            max_workers=self.config.max_workers,
            use_schema_constraints=self.config.langextract_use_schema_constraints,
        )

        self.knowledge_graph = KnowledgeGraph()
        self.reranker = None
        self.entity_nodes_by_hash = {}

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
            batch_size=self.config.batch_size,
            namespace="passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "entity_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="entity",
        )
        self.su_embedding_store = EmbeddingStore(
            self.embedding_model,
            db_filename=os.path.join(save_dir, "su_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="su",
        )

    # ====== Indexing ======

    def index(self, docs):
        """Build the hypergraph index from documents."""
        self.passage_embedding_store.insert_text(docs)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        cached_passage_to_su_ids, cached_su_data, uncached_passage_ids = \
            self._load_cached_extractions(hash_id_to_passage.keys())
        logger.info(
            "Extraction cache status: cached_passages=%s, uncached_passages=%s",
            len(cached_passage_to_su_ids),
            len(uncached_passage_ids),
        )

        if uncached_passage_ids:
            new_hash_id_to_passage = {
                hash_id: hash_id_to_passage[hash_id]
                for hash_id in uncached_passage_ids
            }
            logger.info(
                "Creating semantic units via Kamradt Percentile (p=%s)...",
                self.config.semantic_unit_percentile,
            )

            total_su_count = 0
            for passage_hash_id, passage_text in tqdm(
                new_hash_id_to_passage.items(),
                desc="Semantic Unit Chunking",
            ):
                su_texts = create_semantic_units(
                    passage_text,
                    self.spacy_model,
                    self.embedding_model,
                    self.config.semantic_unit_percentile,
                )
                su_items = []
                su_hash_ids = []
                for su_text in su_texts:
                    su_hash_id = compute_mdhash_id(su_text, prefix="su-")
                    su_items.append((su_hash_id, su_text))
                    su_hash_ids.append(su_hash_id)
                cached_passage_to_su_ids[passage_hash_id] = su_hash_ids
                total_su_count += len(su_items)

                su_mentions = self.ner_extractor.extract_mentions_from_su_batch(
                    su_items, passage_hash_id=passage_hash_id
                )
                for su_hash_id, su_text in su_items:
                    cached_su_data[su_hash_id] = {
                        "text": su_text,
                        "mentions": su_mentions.get(su_hash_id, []),
                    }

            logger.info(
                "Created %s semantic units from %s passages (avg %.1f SU/passage)",
                total_su_count,
                len(new_hash_id_to_passage),
                total_su_count / max(len(new_hash_id_to_passage), 1),
            )
        else:
            logger.info("All passages already have cached extraction results; skipping extraction.")

        self._persist_extraction_data(cached_passage_to_su_ids, cached_su_data)

        # Rebuild SU/entity stores from the extraction cache so the graph stays consistent.
        self.su_embedding_store.clear()
        self.entity_embedding_store.clear()
        self.knowledge_graph = KnowledgeGraph()

        su_text_by_hash = {
            su_hash_id: payload["text"]
            for su_hash_id, payload in cached_su_data.items()
            if payload.get("text")
        }
        self.su_embedding_store.insert_text(list(su_text_by_hash.values()))

        all_mentions = []
        for payload in cached_su_data.values():
            all_mentions.extend(payload.get("mentions", []))

        entity_nodes, passage_entities, su_entities, passage_entity_counts = \
            merge_entity_mentions(
                all_mentions,
                su_text_by_hash,
                self.embedding_model,
                similarity_threshold=self.config.entity_merge_threshold,
                batch_size=self.config.batch_size,
            )

        entity_texts = [node["embedding_text"] for node in entity_nodes]
        self.entity_embedding_store.insert_text(entity_texts)
        self._persist_entity_nodes(entity_nodes)

        _, _, passage_to_entities, entity_to_su, _ = \
            self.knowledge_graph.build_node_edge_maps(passage_entities, su_entities)

        self.knowledge_graph.build_entity_su_mapping(
            entity_to_su,
            self.entity_embedding_store,
            self.su_embedding_store,
        )
        self.knowledge_graph.link_entities_to_passages(
            passage_to_entities,
            self.passage_embedding_store,
            self.entity_embedding_store,
            passage_entity_counts=passage_entity_counts,
        )
        self.knowledge_graph.link_adjacent_passages(self.passage_embedding_store)

    def _load_cached_extractions(self, passage_hash_ids):
        self._ner_cache_path = os.path.join(self.config.save_dir, "ner_results.json")
        if not os.path.exists(self._ner_cache_path):
            return {}, {}, set(passage_hash_ids)

        with open(self._ner_cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)

        if cached.get("schema_version") != 2:
            logger.warning(
                "Ignoring legacy extraction cache at %s because schema_version != 2",
                self._ner_cache_path,
            )
            return {}, {}, set(passage_hash_ids)

        passage_to_su_ids = cached.get("passage_hash_id_to_su_hash_ids", {})
        su_to_data = cached.get("su_to_data", {})
        uncached_ids = set(passage_hash_ids) - set(passage_to_su_ids.keys())
        return passage_to_su_ids, su_to_data, uncached_ids

    def _persist_extraction_data(self, passage_to_su_ids, su_to_data):
        os.makedirs(os.path.dirname(self._ner_cache_path), exist_ok=True)
        with open(self._ner_cache_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 2,
                    "passage_hash_id_to_su_hash_ids": passage_to_su_ids,
                    "su_to_data": su_to_data,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def _persist_entity_nodes(self, entity_nodes):
        entity_nodes_path = os.path.join(self.config.save_dir, "entity_nodes.json")
        self.entity_nodes_by_hash = {}
        serializable_nodes = {}
        for entity_node in entity_nodes:
            entity_text = entity_node["embedding_text"]
            entity_hash_id = self.entity_embedding_store.text_to_hash_id.get(entity_text)
            if entity_hash_id is None:
                continue
            stored_node = dict(entity_node)
            stored_node["hash_id"] = entity_hash_id
            self.entity_nodes_by_hash[entity_hash_id] = stored_node
            serializable_nodes[entity_hash_id] = stored_node

        with open(entity_nodes_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "entities": serializable_nodes,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    # ====== Retrieval ======

    def retrieve(self, queries, num_to_retrieve=None):
        """Retrieve relevant passages for a list of queries."""
        top_k = num_to_retrieve or self.config.retrieval_top_k
        use_rerank = self.config.use_reranker

        if use_rerank:
            self._lazy_load_reranker()
            logger.info(
                "Reranking enabled: top %s candidates -> top %s final passages",
                self.config.reranker_candidate_top_k,
                top_k,
            )
        else:
            logger.info("Reranking disabled. Using dual-channel scoring (top %s).", top_k)

        self._prepare_retrieval_cache()

        retrieval_results = []
        for query in tqdm(queries, desc="Retrieving"):
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
                prompt=self.config.query_instruction_prefix,
            )
            _, seed_indices, seed_texts, seed_hash_ids, seed_scores = \
                self.extract_seed_entities(query)

            if seed_texts:
                sorted_hash_ids, sorted_scores = self._diffuse_from_seeds(
                    query,
                    query_embedding,
                    seed_indices,
                    seed_hash_ids,
                    seed_scores,
                )
                if use_rerank:
                    candidate_hash_ids = sorted_hash_ids[:self.config.reranker_candidate_top_k]
                    candidate_passages = [
                        self.passage_embedding_store.hash_id_to_text[hash_id]
                        for hash_id in candidate_hash_ids
                    ]
                    _, final_scores, final_passages = self._rerank_passages(
                        query,
                        candidate_hash_ids,
                        candidate_passages,
                        top_k,
                    )
                else:
                    top_hash_ids = sorted_hash_ids[:top_k]
                    final_scores = [float(score) for score in sorted_scores[:top_k]]
                    final_passages = [
                        self.passage_embedding_store.hash_id_to_text[hash_id]
                        for hash_id in top_hash_ids
                    ]
            else:
                sorted_indices, sorted_scores = dense_retrieval(
                    self._passage_embeddings,
                    query_embedding,
                )
                if use_rerank:
                    candidate_indices = sorted_indices[:self.config.reranker_candidate_top_k]
                    candidate_hash_ids = [
                        self.passage_embedding_store.hash_ids[idx]
                        for idx in candidate_indices
                    ]
                    candidate_passages = [
                        self.passage_embedding_store.texts[idx]
                        for idx in candidate_indices
                    ]
                    _, final_scores, final_passages = self._rerank_passages(
                        query,
                        candidate_hash_ids,
                        candidate_passages,
                        top_k,
                    )
                else:
                    top_indices = sorted_indices[:top_k]
                    final_scores = [
                        float(sorted_scores[idx]) for idx in range(len(top_indices))
                    ]
                    final_passages = [
                        self.passage_embedding_store.texts[idx]
                        for idx in top_indices
                    ]

            retrieval_results.append({
                "query": query,
                "passages": final_passages,
                "scores": final_scores,
            })
        return retrieval_results

    def _prepare_retrieval_cache(self):
        """Cache embeddings and graph lookups. Called once before retrieval loop."""
        self._entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self._entity_embeddings = np.asarray(self.entity_embedding_store.embeddings)
        self._passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self._passage_embeddings = np.asarray(self.passage_embedding_store.embeddings)
        self._su_embeddings = np.asarray(self.su_embedding_store.embeddings)

        logger.info("Building hypergraph...")
        self.knowledge_graph.build_hypergraph(
            self.entity_embedding_store,
            self.su_embedding_store,
            self.device,
        )

        entity_hash_id_set = set(self._entity_hash_ids)
        kg = self.knowledge_graph
        num_passages = len(self._passage_hash_ids)
        self._entity_idf = {}
        for passage_hash_id, entity_weights in kg.edge_weights.items():
            for entity_hash_id in entity_weights:
                if entity_hash_id not in entity_hash_id_set:
                    continue
                self._entity_idf[entity_hash_id] = self._entity_idf.get(entity_hash_id, 0) + 1
        for entity_hash_id in list(self._entity_idf.keys()):
            self._entity_idf[entity_hash_id] = np.log(
                num_passages / max(self._entity_idf[entity_hash_id], 1)
            )

    def extract_seed_entities(self, query):
        """Extract query entities and map them onto indexed canonical entities."""
        if len(self._entity_hash_ids) == 0 or self._entity_embeddings.size == 0:
            return [], [], [], [], []

        query_mentions = self.ner_extractor.extract_query_entities(query)
        if not query_mentions:
            return [], [], [], [], []

        query_entity_embeddings = self.embedding_model.encode(
            [mention["embedding_text"] for mention in query_mentions],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
            prompt=self.config.query_instruction_prefix,
        )
        similarities = np.dot(self._entity_embeddings, query_entity_embeddings.T)
        seed_indices = []
        seed_texts = []
        seed_hash_ids = []
        seed_scores = []
        for query_idx in range(len(query_mentions)):
            scores = similarities[:, query_idx]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_hash_id = self._entity_hash_ids[best_idx]
            best_metadata = self.entity_nodes_by_hash.get(best_hash_id, {})
            best_text = best_metadata.get(
                "canonical_name",
                self.entity_embedding_store.hash_id_to_text[best_hash_id],
            )
            seed_indices.append(best_idx)
            seed_texts.append(best_text)
            seed_hash_ids.append(best_hash_id)
            seed_scores.append(best_score)
        return query_mentions, seed_indices, seed_texts, seed_hash_ids, seed_scores

    def _diffuse_from_seeds(self, query, query_embedding, seed_indices,
                            seed_hash_ids, seed_scores):
        """Run frontier expansion + dual-channel evidence fusion."""
        activated_entities = frontier_expansion(
            config=self.config,
            hypergraph=self.knowledge_graph._hypergraph,
            entity_hash_ids=self._entity_hash_ids,
            su_embeddings=self._su_embeddings,
            question_embedding=query_embedding,
            seed_entity_indices=seed_indices,
            seed_entity_hash_ids=seed_hash_ids,
            seed_entity_scores=seed_scores,
            entity_embeddings=self._entity_embeddings,
        )

        dense_sims = np.dot(self._passage_embeddings, query_embedding)

        coverage_scores = np.zeros(len(self._passage_hash_ids))
        total_activation = sum(value[1] for value in activated_entities.values())
        if total_activation > 0:
            for passage_idx, passage_hash_id in enumerate(self._passage_hash_ids):
                if passage_hash_id not in self.knowledge_graph.edge_weights:
                    continue
                coverage = 0.0
                for entity_hash_id in self.knowledge_graph.edge_weights[passage_hash_id]:
                    if entity_hash_id not in activated_entities:
                        continue
                    act_score = activated_entities[entity_hash_id][1]
                    idf = self._entity_idf.get(entity_hash_id, 1.0)
                    coverage += act_score * idf
                coverage_scores[passage_idx] = coverage / total_activation

        dense_max = dense_sims.max()
        dense_norm = dense_sims / dense_max if dense_max > 0 else dense_sims
        coverage_max = coverage_scores.max()
        coverage_norm = coverage_scores / coverage_max if coverage_max > 0 else coverage_scores

        lam = self.config.scoring_lambda
        final_scores = lam * dense_norm + (1 - lam) * coverage_norm

        sorted_indices = np.argsort(final_scores)[::-1]
        sorted_hash_ids = [self._passage_hash_ids[idx] for idx in sorted_indices]
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
        if not candidate_passages:
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
        """Retrieve passages and generate answers for a list of queries."""
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
                {"role": "user", "content": prompt_user},
            ]
            all_messages.append(messages)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)",
            ))

        for qa_result, result in zip(all_qa_results, retrieval_results):
            try:
                answer = qa_result.split("Answer:")[1].strip()
            except Exception:
                answer = qa_result
            result["answer"] = answer
        return retrieval_results
