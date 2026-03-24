"""Knowledge graph: entity-passage edge weights and entity-SU hyperedge mapping."""

import re
import numpy as np
from collections import defaultdict
from hyperflow.frontier import Hypergraph
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Entity-passage edge weights and entity-SU hyperedge mapping
    for frontier expansion and dual-channel passage scoring."""

    def __init__(self):
        self.edge_weights = defaultdict(dict)
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
            weight = count / passage_total_count[passage_hash_id] if passage_total_count[passage_hash_id] > 0 else 0
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

    # ====== Retrieval ======

    def build_hypergraph(self, entity_store, su_store, device):
        """Build Hypergraph incidence matrix from entity-SU mapping."""
        self._hypergraph = Hypergraph(self.entity_to_su_ids, entity_store, su_store, device)
        return self._hypergraph


def dense_retrieval(passage_embeddings, query_embedding):
    """Rank all passages by cosine similarity to the query."""
    query_emb = query_embedding.reshape(1, -1)
    similarities = np.dot(passage_embeddings, query_emb.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_scores = similarities[sorted_indices].tolist()
    return sorted_indices, sorted_scores
