"""Hop-wise frontier expansion on a hypergraph for multi-hop entity discovery.

Vertices  = entities
Hyperedges = semantic units (each hyperedge connects all entities that co-occur in one SU)

Algorithm: explicit BFS-like hop-by-hop exploration.  Each hop discovers
NEW entities through query-relevant SUs, then those new entities become the
frontier for the next hop.
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class Hypergraph:
    """Sparse hypergraph backed by an incidence matrix H.

    H[v, e] = 1  iff vertex v belongs to hyperedge e.
    """

    def __init__(self, entity_hash_id_to_su_hash_ids,
                 entity_embedding_store, su_embedding_store, device):
        self.device = device
        self.num_vertices = len(entity_embedding_store.hash_id_to_text)
        self.num_hyperedges = len(su_embedding_store.hash_id_to_text)

        # Build incidence matrix H  (|V| x |E|)
        indices = []
        for entity_hash_id, su_hash_ids in entity_hash_id_to_su_hash_ids.items():
            v_idx = entity_embedding_store.hash_id_to_idx[entity_hash_id]
            for su_hash_id in su_hash_ids:
                e_idx = su_embedding_store.hash_id_to_idx[su_hash_id]
                indices.append([v_idx, e_idx])

        if indices:
            idx_tensor = torch.tensor(indices, dtype=torch.long).t()
            val_tensor = torch.ones(idx_tensor.shape[1], dtype=torch.float32)
            self.H = torch.sparse_coo_tensor(
                idx_tensor, val_tensor,
                (self.num_vertices, self.num_hyperedges), device=device,
            ).coalesce()
        else:
            self.H = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
                (self.num_vertices, self.num_hyperedges), device=device,
            )

        nnz = self.H._nnz()
        ones_e = torch.ones(self.num_hyperedges, 1, device=device)
        d_v = torch.sparse.mm(self.H, ones_e).squeeze()
        ones_v = torch.ones(self.num_vertices, 1, device=device)
        d_e = torch.sparse.mm(self.H.t(), ones_v).squeeze()

        logger.info(
            "Hypergraph built: %d vertices, %d hyperedges, nnz=%d, "
            "sparsity=%.2f%%, D_v range [%.0f, %.0f], D_e range [%.0f, %.0f]",
            self.num_vertices, self.num_hyperedges, nnz,
            (1 - nnz / max(self.num_vertices * self.num_hyperedges, 1)) * 100,
            d_v.min(), d_v.max(), d_e.min(), d_e.max(),
        )


def frontier_expansion(config, hypergraph, entity_hash_ids,
                       su_embeddings, question_embedding,
                       seed_entity_indices, seed_entity_hash_ids,
                       seed_entity_scores):
    """Hop-wise frontier expansion on a hypergraph.

    Each hop:
      1. Find SUs reachable from the current frontier, gated by query relevance.
      2. Through those SUs, discover new (not yet activated) entities.
      3. Score each candidate via max-aggregation:
            score(v) = max over reachable SUs e containing v:
                       w[e] × max(frontier_score of frontier entities in e)
      4. Keep top-K new entities; they become the next frontier.

    Returns:
        activated_entities: dict of entity_hash_id → (entity_idx, score, 1)
    """
    device = hypergraph.device
    num_vertices = hypergraph.num_vertices
    num_hyperedges = hypergraph.num_hyperedges
    top_k = config.expansion_top_k
    max_hops = config.expansion_max_hops
    hop_decay = config.hop_decay

    H = hypergraph.H

    # ── Phase A: Query-conditioned SU conductance ──
    question_emb = (question_embedding.reshape(-1, 1)
                    if len(question_embedding.shape) == 1
                    else question_embedding)
    su_sims = np.dot(su_embeddings, question_emb).flatten()
    w_q = torch.from_numpy(su_sims).float().to(device).clamp(min=0)

    floor = config.conductance_floor
    gamma = config.conductance_gamma
    w_base = torch.clamp(w_q - floor, min=0) / max(1.0 - floor, 1e-8)
    w_conductance = w_base.pow(gamma)

    num_active = int((w_conductance > 0).sum().item())
    logger.info(
        "Conductance (floor=%.2f, gamma=%.2f): %d/%d active SUs, %d suppressed",
        floor, gamma, num_active, num_hyperedges, num_hyperedges - num_active,
    )

    # ── Phase B: Initialise seeds ──
    activated = {}
    for idx, score in zip(seed_entity_indices, seed_entity_scores):
        activated[idx] = float(score)

    frontier = set(activated.keys())

    # Precompute COO structure once
    H_coo = H.coalesce()
    h_indices = H_coo.indices()
    h_v_idx = h_indices[0]
    h_e_idx = h_indices[1]

    # ── Phase C: Hop-by-hop frontier expansion ──
    for hop in range(1, max_hops + 1):
        if not frontier:
            break

        # 1) Build frontier score vector
        frontier_scores = torch.zeros(num_vertices, device=device)
        for idx in frontier:
            frontier_scores[idx] = activated[idx]

        # 2) Scatter max frontier scores to SUs via COO
        is_frontier = frontier_scores[h_v_idx] > 0
        frontier_signal = torch.where(is_frontier, frontier_scores[h_v_idx],
                                       torch.tensor(0.0, device=device))
        max_frontier_per_su = torch.zeros(num_hyperedges, device=device)
        max_frontier_per_su.scatter_reduce_(0, h_e_idx, frontier_signal, reduce="amax")

        # 3) Bridge score = conductance × max frontier score
        bridge_scores = w_conductance * max_frontier_per_su

        # 4) Scatter max bridge scores to candidate entities via COO
        candidate_scores = torch.full((num_vertices,), -1.0, device=device)
        edge_bridge = bridge_scores[h_e_idx]
        candidate_scores.scatter_reduce_(0, h_v_idx, edge_bridge, reduce="amax")

        # Zero out already-activated entities
        activated_tensor = torch.zeros(num_vertices, dtype=torch.bool, device=device)
        for idx in activated:
            activated_tensor[idx] = True
        candidate_scores[activated_tensor] = -1.0

        # 5) Select top-K new entities with positive scores
        num_positive = int((candidate_scores > 0).sum().item())
        if num_positive == 0:
            logger.debug("Hop %d: no new candidates found, stopping.", hop)
            break

        k = min(top_k, num_positive)
        top_vals, top_idx = torch.topk(candidate_scores, k)

        decay = hop_decay ** hop
        new_frontier = set()
        for idx_t, val_t in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist()):
            if val_t <= 0:
                break
            activated[idx_t] = val_t * decay
            new_frontier.add(idx_t)

        logger.debug(
            "Hop %d: %d new entities (top score=%.4f, decay=%.3f), "
            "%d total activated",
            hop, len(new_frontier),
            top_vals[0].item() if len(top_vals) > 0 else 0, decay,
            len(activated),
        )

        frontier = new_frontier

    logger.info(
        "Frontier expansion finished: %d hops, %d activated entities",
        min(hop, max_hops) if frontier else hop - 1, len(activated),
    )

    # ── Phase D: Collect results ──
    activated_entities = {}
    for idx, hash_id, score in zip(seed_entity_indices, seed_entity_hash_ids, seed_entity_scores):
        activated_entities[hash_id] = (idx, score, 1)

    for entity_idx, score in activated.items():
        hash_id = entity_hash_ids[entity_idx]
        if hash_id not in activated_entities:
            activated_entities[hash_id] = (entity_idx, score, 1)

    return activated_entities
