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


def _compute_conductance(su_embeddings, query_emb, floor, gamma, device):
    """Compute SU conductance weights from a query embedding."""
    su_sims = np.dot(su_embeddings, query_emb).flatten()
    w_q = torch.from_numpy(su_sims).float().to(device).clamp(min=0)
    w_base = torch.clamp(w_q - floor, min=0) / max(1.0 - floor, 1e-8)
    return w_base.pow(gamma)


def frontier_expansion(config, hypergraph, entity_hash_ids,
                       su_embeddings, question_embedding,
                       seed_entity_indices, seed_entity_hash_ids,
                       seed_entity_scores,
                       entity_embeddings=None):
    """Hop-wise frontier expansion on a hypergraph.

    Each hop:
      1. Find SUs reachable from the current frontier, gated by query relevance.
      2. Through those SUs, discover new (not yet activated) entities.
      3. Score each candidate via max-aggregation:
            score(v) = max over reachable SUs e containing v:
                       w[e] × max(frontier_score of frontier entities in e)
      4. Keep top-K new entities; they become the next frontier.
      5. (Progressive Steering) Update query embedding using top-K activated
         entity centroid, then recompute conductance for the next hop.

    Args:
        entity_embeddings: numpy array (num_entities, dim). Required for
            progressive steering (steering_alpha < 1.0). If None or
            steering_alpha == 1.0, steering is disabled.

    Returns:
        activated_entities: dict of entity_hash_id → (entity_idx, score)
    """
    device = hypergraph.device
    num_vertices = hypergraph.num_vertices
    num_hyperedges = hypergraph.num_hyperedges
    top_k = config.expansion_top_k
    max_hops = config.expansion_max_hops
    hop_decay = config.hop_decay
    floor = config.conductance_floor
    gamma = config.conductance_gamma

    # Steering parameters
    alpha = config.steering_alpha
    steering_top_k = config.steering_top_k
    use_steering = alpha < 1.0 and entity_embeddings is not None

    H = hypergraph.H

    # ── Phase A: Initial query-conditioned SU conductance ──
    q_original = (question_embedding.reshape(-1, 1)
                  if len(question_embedding.shape) == 1
                  else question_embedding)
    q_current = q_original

    w_conductance = _compute_conductance(su_embeddings, q_current, floor, gamma, device)

    num_active = int((w_conductance > 0).sum().item())
    logger.info(
        "Conductance (floor=%.2f, gamma=%.2f): %d/%d active SUs, %d suppressed",
        floor, gamma, num_active, num_hyperedges, num_hyperedges - num_active,
    )
    if use_steering:
        logger.info("Progressive steering enabled: alpha=%.2f, top_k=%d", alpha, steering_top_k)

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

        # ── Progressive Query Steering ──
        # After discovering new entities, steer the query toward what was found
        # so the next hop's conductance favours SUs relevant to the evolving topic.
        if use_steering and frontier:
            # Take top-K activated entities (excluding seeds) by score
            non_seed = {idx: sc for idx, sc in activated.items()
                        if idx not in set(seed_entity_indices)}
            if non_seed:
                sorted_ents = sorted(non_seed.items(), key=lambda x: x[1], reverse=True)
                top_ent_indices = [idx for idx, _ in sorted_ents[:steering_top_k]]
                centroid = np.mean(entity_embeddings[top_ent_indices], axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-8:
                    centroid = centroid / norm
                    q_steered = alpha * q_original.flatten() + (1 - alpha) * centroid
                    q_steered = q_steered / (np.linalg.norm(q_steered) + 1e-8)
                    q_current = q_steered.reshape(-1, 1)

                    w_conductance = _compute_conductance(
                        su_embeddings, q_current, floor, gamma, device
                    )
                    new_active = int((w_conductance > 0).sum().item())
                    logger.debug(
                        "Steering after hop %d: %d active SUs (was %d)",
                        hop, new_active, num_active,
                    )

    logger.info(
        "Frontier expansion finished: %d hops, %d activated entities",
        min(hop, max_hops) if frontier else hop - 1, len(activated),
    )

    # ── Phase D: Collect results ──
    activated_entities = {}
    for idx, hash_id, score in zip(seed_entity_indices, seed_entity_hash_ids, seed_entity_scores):
        activated_entities[hash_id] = (idx, score)

    for entity_idx, score in activated.items():
        hash_id = entity_hash_ids[entity_idx]
        if hash_id not in activated_entities:
            activated_entities[hash_id] = (entity_idx, score)

    return activated_entities
