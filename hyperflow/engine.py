"""Hyperflow: Hypergraph-based retrieval-augmented generation engine."""

from hyperflow.embedding_store import EmbeddingStore
from hyperflow.utils import min_max_normalize
from hyperflow.ner import SpacyNER
from hyperflow.reranker import QwenReranker
from hyperflow import diffusion as diff
from hyperflow import graph as gr

import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import igraph as ig
import re
import logging
import torch

logger = logging.getLogger(__name__)


class Hyperflow:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing Hyperflow with config: {self.config}")
        logger.info("Retrieval method: Hypergraph Spectral Diffusion")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.dataset_name = global_config.dataset_name
        self.init_embedding_stores()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)
        self.reranker = None

    def init_embedding_stores(self):
        self.passage_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "passage_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="passage"
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "entity_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="entity"
        )
        self.sentence_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"),
            batch_size=self.config.batch_size, namespace="sentence"
        )

    def load_cached_ner(self, passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_results = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_results["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_results["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = (
            "As an advanced reading comprehension assistant, your task is to analyze text passages "
            "and corresponding questions meticulously. Your response start after \"Thought: \", "
            "where you will methodically break down the reasoning process, illustrating how you "
            "arrive at conclusions. Conclude with \"Answer: \" to present a concise, definitive "
            "response, devoid of additional elaborations."
        )
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = ""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
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

        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results

    def retrieve(self, questions):
        self._lazy_load_reranker()
        logger.info(
            "Reranking enabled: top %s candidates -> top %s final passages",
            self.config.reranker_candidate_top_k,
            self.config.retrieval_top_k,
        )
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        # Build incidence matrices for spectral diffusion
        logger.info("Building sparse incidence matrices...")
        self.entity_to_sentence_sparse, self.sentence_to_entity_sparse = diff.build_incidence_matrix(
            self.entity_hash_id_to_sentence_hash_ids, self.sentence_hash_id_to_entity_hash_ids,
            self.entity_embedding_store, self.sentence_embedding_store, self.device
        )
        e2s_shape = self.entity_to_sentence_sparse.shape
        s2e_shape = self.sentence_to_entity_sparse.shape
        e2s_nnz = self.entity_to_sentence_sparse._nnz()
        s2e_nnz = self.sentence_to_entity_sparse._nnz()
        logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
        logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
        logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
        logger.info(f"Device: {self.device}")

        # Precompute degree vectors for Laplacian normalization
        logger.info("Computing hypergraph degree vectors...")
        self.d_v_inv_sqrt, self.d_e_inv = diff.compute_degree_vectors(
            self.entity_to_sentence_sparse, self.device
        )

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(
                question, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size,
                prompt=self.config.query_instruction_prefix
            )
            _, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.extract_seed_entities(question)
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = self.diffuse_from_seeds(
                    question, question_embedding, seed_entity_indices, seed_entities,
                    seed_entity_hash_ids, seed_entity_scores
                )
                candidate_passage_hash_ids = sorted_passage_hash_ids[:self.config.reranker_candidate_top_k]
                candidate_passages = [
                    self.passage_embedding_store.hash_id_to_text[passage_hash_id]
                    for passage_hash_id in candidate_passage_hash_ids
                ]
                _, final_passage_scores, final_passages = self.rerank_passages(
                    question=question,
                    candidate_passage_hash_ids=candidate_passage_hash_ids,
                    candidate_passages=candidate_passages,
                )
            else:
                sorted_passage_indices, sorted_passage_scores = gr.dense_retrieval(self.passage_embeddings, question_embedding)
                candidate_passage_indices = sorted_passage_indices[:self.config.reranker_candidate_top_k]
                candidate_passage_hash_ids = [self.passage_embedding_store.hash_ids[idx] for idx in candidate_passage_indices]
                candidate_passages = [self.passage_embedding_store.texts[idx] for idx in candidate_passage_indices]
                _, final_passage_scores, final_passages = self.rerank_passages(
                    question=question,
                    candidate_passage_hash_ids=candidate_passage_hash_ids,
                    candidate_passages=candidate_passages,
                )
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        return retrieval_results

    def _lazy_load_reranker(self):
        if self.reranker is not None:
            return
        self.reranker = QwenReranker(
            model_name=self.config.reranker_model_name,
            batch_size=self.config.reranker_batch_size,
            max_length=self.config.reranker_max_length,
            instruction=self.config.reranker_instruction,
        )

    def rerank_passages(self, question, candidate_passage_hash_ids, candidate_passages):
        if len(candidate_passages) == 0:
            return [], [], []

        rerank_scores = self.reranker.score(question, candidate_passages)
        reranked_indices = np.argsort(np.asarray(rerank_scores))[::-1]
        selected_indices = reranked_indices[:self.config.retrieval_top_k]

        final_passage_hash_ids = [candidate_passage_hash_ids[idx] for idx in selected_indices]
        final_passage_scores = [float(rerank_scores[idx]) for idx in selected_indices]
        final_passages = [candidate_passages[idx] for idx in selected_indices]
        return final_passage_hash_ids, final_passage_scores, final_passages

    def diffuse_from_seeds(self, question, question_embedding, seed_entity_indices, seed_entities,
                           seed_entity_hash_ids, seed_entity_scores):
        """Run spectral diffusion, score passages, and rank via PPR."""
        entity_weights, actived_entities = diff.spectral_flow_propagation(
            config=self.config,
            entity_hash_ids=self.entity_hash_ids,
            entity_embeddings=self.entity_embeddings,
            sentence_embeddings=self.sentence_embeddings,
            question_embedding=question_embedding,
            seed_entity_indices=seed_entity_indices,
            seed_entity_hash_ids=seed_entity_hash_ids,
            seed_entity_scores=seed_entity_scores,
            entity_to_sentence_sparse=self.entity_to_sentence_sparse,
            node_name_to_vertex_idx=self.node_name_to_vertex_idx,
            d_v_inv_sqrt=self.d_v_inv_sqrt,
            d_e_inv=self.d_e_inv,
            graph_node_count=len(self.graph.vs["name"]),
            device=self.device,
        )
        passage_weights = gr.score_passages(
            config=self.config,
            question=question,
            question_embedding=question_embedding,
            actived_entities=actived_entities,
            passage_embeddings=self.passage_embeddings,
            passage_embedding_store=self.passage_embedding_store,
            entity_embedding_store=self.entity_embedding_store,
            node_name_to_vertex_idx=self.node_name_to_vertex_idx,
            graph_node_count=len(self.graph.vs["name"]),
        )
        node_weights = entity_weights + passage_weights
        sorted_passage_hash_ids, sorted_passage_scores = gr.personalized_pagerank(
            graph=self.graph,
            node_weights=node_weights,
            damping=self.config.damping,
            node_name_to_vertex_idx=self.node_name_to_vertex_idx,
            vertex_idx_to_node_name=self.vertex_idx_to_node_name,
            passage_node_indices=self.passage_node_indices,
        )
        return sorted_passage_hash_ids, sorted_passage_scores

    def extract_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], [], []
        question_entity_embeddings = self.config.embedding_model.encode(
            question_entities, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size,
            prompt=self.config.query_instruction_prefix
        )
        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
            best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return question_entities, seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_cached_ner(hash_id_to_passage.keys())
        logger.info(
            "NER cache status: cached_passages=%s, uncached_passages=%s",
            len(existing_passage_hash_id_to_entities),
            len(new_passage_hash_ids),
        )
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_data(existing_passage_hash_id_to_entities, existing_sentence_to_entities,
                               new_passage_hash_id_to_entities, new_sentence_to_entities)
        else:
            logger.info("All passages already have cached spaCy NER results; skipping NER recomputation.")
        self.persist_ner_data(existing_passage_hash_id_to_entities, existing_sentence_to_entities)

        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = \
            gr.build_node_edge_maps(existing_passage_hash_id_to_entities, existing_sentence_to_entities)

        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))

        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [
                self.sentence_embedding_store.text_to_hash_id[s] for s in sentence
            ]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [
                self.entity_embedding_store.text_to_hash_id[e] for e in entities
            ]

        gr.link_entities_to_passages(passage_hash_id_to_entities, self.passage_embedding_store,
                                     self.entity_embedding_store, self.node_to_node_stats)
        gr.link_adjacent_passages(self.passage_embedding_store, self.node_to_node_stats)

        self.node_name_to_vertex_idx, self.passage_node_indices = gr.finalize_graph(
            self.graph, self.entity_embedding_store, self.passage_embedding_store, self.node_to_node_stats
        )

        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "Hyperflow.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.graph.write_graphml(output_graphml_path)

    def merge_ner_data(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities,
                       new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def persist_ner_data(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({
                "passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                "sentence_to_entities": existing_sentence_to_entities
            }, f)
