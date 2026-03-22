"""Graph construction, PPR scoring, and passage relevance computation."""

import re
import math
import numpy as np
from collections import defaultdict
from hyperflow.utils import min_max_normalize
import logging

logger = logging.getLogger(__name__)


def build_node_edge_maps(existing_passage_hash_id_to_entities, existing_su_to_entities):
    """Extract entity/semantic-unit node sets and build bidirectional edge maps."""
    entity_nodes = set()
    su_nodes = set()
    passage_hash_id_to_entities = defaultdict(set)
    entity_to_su = defaultdict(set)
    su_to_entity = defaultdict(set)
    for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
        for entity in entities:
            entity_nodes.add(entity)
            passage_hash_id_to_entities[passage_hash_id].add(entity)
    for su, entities in existing_su_to_entities.items():
        su_nodes.add(su)
        for entity in entities:
            entity_to_su[entity].add(su)
            su_to_entity[su].add(entity)
    return entity_nodes, su_nodes, passage_hash_id_to_entities, entity_to_su, su_to_entity


def link_entities_to_passages(passage_hash_id_to_entities, passage_embedding_store, entity_embedding_store, node_to_node_stats):
    """Create weighted edges between passages and their contained entities."""
    passage_to_entity_count = {}
    passage_to_all_score = defaultdict(int)
    for passage_hash_id, entities in passage_hash_id_to_entities.items():
        passage = passage_embedding_store.hash_id_to_text[passage_hash_id]
        for entity in entities:
            entity_hash_id = entity_embedding_store.text_to_hash_id[entity]
            count = passage.count(entity)
            passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
            passage_to_all_score[passage_hash_id] += count
    for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
        score = count / passage_to_all_score[passage_hash_id]
        node_to_node_stats[passage_hash_id][entity_hash_id] = score


def link_adjacent_passages(passage_embedding_store, node_to_node_stats):
    """Connect sequentially adjacent passages based on their index prefix."""
    passage_id_to_text = passage_embedding_store.get_hash_id_to_text()
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
        node_to_node_stats[current_node][next_node] = 1.0


def register_nodes(graph, entity_embedding_store, passage_embedding_store):
    """Add entity and passage nodes to the graph, return passage node indices."""
    existing_nodes = {v["name"]: v for v in graph.vs if "name" in v.attributes()}
    entity_hash_id_to_text = entity_embedding_store.get_hash_id_to_text()
    passage_hash_id_to_text = passage_embedding_store.get_hash_id_to_text()
    all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

    passage_hash_ids = set(passage_hash_id_to_text.keys())

    for hash_id, text in all_hash_id_to_text.items():
        if hash_id not in existing_nodes:
            graph.add_vertex(name=hash_id, content=text)

    node_name_to_vertex_idx = {v["name"]: v.index for v in graph.vs if "name" in v.attributes()}
    passage_node_indices = [
        node_name_to_vertex_idx[passage_id]
        for passage_id in passage_hash_ids
        if passage_id in node_name_to_vertex_idx
    ]
    return node_name_to_vertex_idx, passage_node_indices


def register_edges(graph, node_to_node_stats):
    """Add weighted edges to the graph from the accumulated stats."""
    edges = []
    weights = []
    for node_hash_id, neighbors in node_to_node_stats.items():
        for neighbor_hash_id, weight in neighbors.items():
            if node_hash_id == neighbor_hash_id:
                continue
            edges.append((node_hash_id, neighbor_hash_id))
            weights.append(weight)
    graph.add_edges(edges)
    graph.es['weight'] = weights


def finalize_graph(graph, entity_embedding_store, passage_embedding_store, node_to_node_stats):
    """Register all nodes and edges in the graph."""
    node_name_to_vertex_idx, passage_node_indices = register_nodes(graph, entity_embedding_store, passage_embedding_store)
    register_edges(graph, node_to_node_stats)
    return node_name_to_vertex_idx, passage_node_indices


def personalized_pagerank(graph, node_weights, damping, node_name_to_vertex_idx, vertex_idx_to_node_name, passage_node_indices):
    """Run personalized PageRank and return sorted passage hash_ids with scores."""
    reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
    pagerank_scores = graph.personalized_pagerank(
        vertices=range(len(node_name_to_vertex_idx)),
        damping=damping,
        directed=False,
        weights='weight',
        reset=reset_prob,
        implementation='prpack'
    )

    doc_scores = np.array([pagerank_scores[idx] for idx in passage_node_indices])
    sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
    sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

    sorted_passage_hash_ids = [
        vertex_idx_to_node_name[passage_node_indices[i]]
        for i in sorted_indices_in_doc_scores
    ]

    return sorted_passage_hash_ids, sorted_passage_scores.tolist()


def dense_retrieval(passage_embeddings, question_embedding):
    """Rank all passages by cosine similarity to the question."""
    question_emb = question_embedding.reshape(1, -1)
    similarities = np.dot(passage_embeddings, question_emb.T).flatten()
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_scores = similarities[sorted_indices].tolist()
    return sorted_indices, sorted_scores


def score_passages(config, question, question_embedding, actived_entities,
                   passage_embeddings, passage_embedding_store, entity_embedding_store,
                   node_name_to_vertex_idx, graph_node_count):
    """Compute passage relevance scores combining dense retrieval and entity bonuses."""
    passage_weights = np.zeros(graph_node_count)
    dpr_passage_indices, dpr_passage_scores = dense_retrieval(passage_embeddings, question_embedding)
    dpr_passage_scores = min_max_normalize(dpr_passage_scores)
    apply_attribute_boost = (
        config.enable_hybrid_attribute_fallback
        and _is_attribute_query(question, config.attribute_query_keywords)
    )
    question_lower = question.lower()

    for i, dpr_passage_index in enumerate(dpr_passage_indices):
        total_entity_bonus = 0
        passage_hash_id = passage_embedding_store.hash_ids[dpr_passage_index]
        dpr_passage_score = dpr_passage_scores[i]
        passage_text_lower = passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
        for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
            entity_lower = entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
            entity_occurrences = passage_text_lower.count(entity_lower)
            if entity_occurrences > 0:
                denom = tier if tier >= 1 else 1
                entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                total_entity_bonus += entity_bonus

        passage_score = config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

        if apply_attribute_boost:
            overlap = _attribute_keyword_overlap(question_lower, passage_text_lower, config.attribute_query_keywords)
            if overlap > 0:
                passage_score += config.attribute_keyword_boost * math.log(1 + overlap)

        passage_node_idx = node_name_to_vertex_idx[passage_hash_id]
        passage_weights[passage_node_idx] = passage_score * config.passage_node_weight
    return passage_weights


def _is_attribute_query(question, keywords):
    tokens = set(re.findall(r"\w+", question.lower()))
    return any(keyword in tokens for keyword in keywords)


def _attribute_keyword_overlap(question_lower, passage_text_lower, keywords):
    overlap = 0
    for keyword in keywords:
        if keyword in question_lower and keyword in passage_text_lower:
            overlap += 1
    return overlap
