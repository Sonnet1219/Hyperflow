"""Knowledge graph: entity-passage bipartite graph + entity-SU hyperedge mapping."""

import re
import math
import numpy as np
from collections import defaultdict
from hyperflow.utils import min_max_normalize
from hyperflow.diffusion import Hypergraph
import igraph as ig
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Encapsulates both the entity-passage graph (for PPR) and
    the entity-SU hyperedge mapping (for spectral diffusion)."""

    def __init__(self):
        # --- Entity-Passage bipartite graph (igraph) ---
        self.graph = ig.Graph(directed=False)
        self.edge_weights = defaultdict(dict)
        self.node_name_to_vertex_idx = {}
        self.vertex_idx_to_node_name = {}
        self.passage_node_indices = []

        # --- Entity-SU hyperedge mapping ---
        self.entity_to_su_ids = {}
        self._hypergraph = None

    # ====== Indexing ======

    def build_node_edge_maps(self, passage_entities, su_entities):
        """Extract entity/SU node sets and build bidirectional edge maps.

        Args:
            passage_entities: dict mapping passage_hash_id -> list of entity texts
            su_entities: dict mapping su_text -> list of entity texts

        Returns:
            entity_nodes, su_nodes, passage_to_entities, entity_to_su, su_to_entity
        """
        entity_nodes = set()
        su_nodes = set()
        passage_to_entities = defaultdict(set)
        entity_to_su = defaultdict(set)
        su_to_entity = defaultdict(set)
        for passage_hash_id, entities in passage_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_to_entities[passage_hash_id].add(entity)
        for su, entities in su_entities.items():
            su_nodes.add(su)
            for entity in entities:
                entity_to_su[entity].add(su)
                su_to_entity[su].add(entity)
        return entity_nodes, su_nodes, passage_to_entities, entity_to_su, su_to_entity

    def build_entity_su_mapping(self, entity_to_su, entity_store, su_store):
        """Convert text-level entity->SU mapping to hash_id-level and store.

        Args:
            entity_to_su: dict mapping entity_text -> set of su_texts
            entity_store: EmbeddingStore for entities
            su_store: EmbeddingStore for semantic units
        """
        self.entity_to_su_ids = {}
        for entity, sus in entity_to_su.items():
            entity_hash_id = entity_store.text_to_hash_id[entity]
            self.entity_to_su_ids[entity_hash_id] = [
                su_store.text_to_hash_id[s] for s in sus
            ]

    def link_entities_to_passages(self, passage_to_entities, passage_store, entity_store):
        """Create weighted edges between passages and their contained entities."""
        passage_entity_count = {}
        passage_total_count = defaultdict(int)
        for passage_hash_id, entities in passage_to_entities.items():
            passage_text = passage_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = entity_store.text_to_hash_id[entity]
                count = passage_text.count(entity)
                passage_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_total_count[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_entity_count.items():
            weight = count / passage_total_count[passage_hash_id]
            self.edge_weights[passage_hash_id][entity_hash_id] = weight

    def link_adjacent_passages(self, passage_store):
        """Connect sequentially adjacent passages based on their index prefix."""
        passage_id_to_text = passage_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.edge_weights[current_node][next_node] = 1.0

    def finalize(self, entity_store, passage_store):
        """Register all nodes and edges into the igraph structure."""
        # Register nodes
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = entity_store.get_hash_id_to_text()
        passage_hash_id_to_text = passage_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}
        passage_hash_ids = set(passage_hash_id_to_text.keys())

        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)

        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[pid]
            for pid in passage_hash_ids
            if pid in self.node_name_to_vertex_idx
        ]

        # Register edges
        edges = []
        weights = []
        for node_hash_id, neighbors in self.edge_weights.items():
            for neighbor_hash_id, weight in neighbors.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def save(self, path):
        """Save graph to GraphML file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.graph.write_graphml(path)

    # ====== Retrieval ======

    def build_hypergraph(self, entity_store, su_store, device):
        """Build Hypergraph incidence matrix from entity-SU mapping (called once per retrieval session)."""
        self._hypergraph = Hypergraph(self.entity_to_su_ids, entity_store, su_store, device)
        return self._hypergraph

    def personalized_pagerank(self, node_weights, damping):
        """Run personalized PageRank and return sorted passage hash_ids with scores."""
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices = np.argsort(doc_scores)[::-1]
        sorted_scores = doc_scores[sorted_indices]

        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]]
            for i in sorted_indices
        ]

        return sorted_passage_hash_ids, sorted_scores.tolist()

    def score_passages(self, config, query, query_embedding, activated_entities,
                       passage_embeddings, passage_store, entity_store):
        """Compute passage relevance scores combining dense retrieval and entity bonuses."""
        graph_node_count = len(self.graph.vs)
        passage_weights = np.zeros(graph_node_count)
        dpr_indices, dpr_scores = dense_retrieval(passage_embeddings, query_embedding)
        dpr_scores = min_max_normalize(dpr_scores)
        apply_attribute_boost = (
            config.enable_hybrid_attribute_fallback
            and _is_attribute_query(query, config.attribute_query_keywords)
        )
        query_lower = query.lower()

        for i, dpr_idx in enumerate(dpr_indices):
            total_entity_bonus = 0
            passage_hash_id = passage_store.hash_ids[dpr_idx]
            dpr_score = dpr_scores[i]
            passage_text_lower = passage_store.hash_id_to_text[passage_hash_id].lower()
            for entity_hash_id, (entity_idx, entity_score, tier) in activated_entities.items():
                entity_lower = entity_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = config.passage_ratio * dpr_score + math.log(1 + total_entity_bonus)

            if apply_attribute_boost:
                overlap = _attribute_keyword_overlap(query_lower, passage_text_lower, config.attribute_query_keywords)
                if overlap > 0:
                    passage_score += config.attribute_keyword_boost * math.log(1 + overlap)

            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * config.passage_node_weight
        return passage_weights


def dense_retrieval(passage_embeddings, query_embedding):
    """Rank all passages by cosine similarity to the query."""
    query_emb = query_embedding.reshape(1, -1)
    similarities = np.dot(passage_embeddings, query_emb.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_scores = similarities[sorted_indices].tolist()
    return sorted_indices, sorted_scores


def _is_attribute_query(query, keywords):
    tokens = set(re.findall(r"\w+", query.lower()))
    return any(keyword in tokens for keyword in keywords)


def _attribute_keyword_overlap(query_lower, passage_text_lower, keywords):
    overlap = 0
    for keyword in keywords:
        if keyword in query_lower and keyword in passage_text_lower:
            overlap += 1
    return overlap
