"""Hypergraph spectral diffusion for entity score propagation."""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def build_incidence_matrix(entity_hash_id_to_sentence_hash_ids, sentence_hash_id_to_entity_hash_ids,
                           entity_embedding_store, sentence_embedding_store, device):
    """
    Build sparse incidence matrices H (entity-to-sentence) and S2E (sentence-to-entity)
    using PyTorch COO tensors. Called once per retrieve() session.
    """
    entity_hash_ids = list(entity_embedding_store.hash_id_to_text.keys())
    sentence_hash_ids = list(sentence_embedding_store.hash_id_to_text.keys())
    num_entities = len(entity_hash_ids)
    num_sentences = len(sentence_hash_ids)

    # Entity-to-sentence (incidence matrix H)
    e2s_indices = []
    e2s_values = []
    for entity_hash_id, sent_hash_ids in entity_hash_id_to_sentence_hash_ids.items():
        entity_idx = entity_embedding_store.hash_id_to_idx[entity_hash_id]
        for sent_hash_id in sent_hash_ids:
            sentence_idx = sentence_embedding_store.hash_id_to_idx[sent_hash_id]
            e2s_indices.append([entity_idx, sentence_idx])
            e2s_values.append(1.0)

    # Sentence-to-entity (transpose direction)
    s2e_indices = []
    s2e_values = []
    for sent_hash_id, ent_hash_ids in sentence_hash_id_to_entity_hash_ids.items():
        sentence_idx = sentence_embedding_store.hash_id_to_idx[sent_hash_id]
        for entity_hash_id in ent_hash_ids:
            entity_idx = entity_embedding_store.hash_id_to_idx[entity_hash_id]
            s2e_indices.append([sentence_idx, entity_idx])
            s2e_values.append(1.0)

    if len(e2s_indices) > 0:
        idx_tensor = torch.tensor(e2s_indices, dtype=torch.long).t()
        val_tensor = torch.tensor(e2s_values, dtype=torch.float32)
        e2s_sparse = torch.sparse_coo_tensor(
            idx_tensor, val_tensor, (num_entities, num_sentences), device=device
        ).coalesce()
    else:
        e2s_sparse = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
            (num_entities, num_sentences), device=device
        )

    if len(s2e_indices) > 0:
        idx_tensor = torch.tensor(s2e_indices, dtype=torch.long).t()
        val_tensor = torch.tensor(s2e_values, dtype=torch.float32)
        s2e_sparse = torch.sparse_coo_tensor(
            idx_tensor, val_tensor, (num_sentences, num_entities), device=device
        ).coalesce()
    else:
        s2e_sparse = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
            (num_sentences, num_entities), device=device
        )

    return e2s_sparse, s2e_sparse


def compute_degree_vectors(H, device):
    """Compute vertex and hyperedge degree vectors for Laplacian normalization."""
    ones_e = torch.ones(H.shape[1], 1, device=device)
    d_v = torch.sparse.mm(H, ones_e).squeeze()         # vertex degree
    ones_v = torch.ones(H.shape[0], 1, device=device)
    d_e = torch.sparse.mm(H.t(), ones_v).squeeze()      # hyperedge degree
    d_v_inv_sqrt = torch.where(d_v > 0, d_v.pow(-0.5), torch.zeros_like(d_v))
    d_e_inv = torch.where(d_e > 0, d_e.pow(-1.0), torch.zeros_like(d_e))
    logger.info(f"Hypergraph degrees: D_v range [{d_v.min():.0f}, {d_v.max():.0f}], "
                f"D_e range [{d_e.min():.0f}, {d_e.max():.0f}]")
    return d_v_inv_sqrt, d_e_inv


def spectral_flow_propagation(config, entity_hash_ids, entity_embeddings, sentence_embeddings,
                              question_embedding, seed_entity_indices, seed_entity_hash_ids,
                              seed_entity_scores, entity_to_sentence_sparse, node_name_to_vertex_idx,
                              d_v_inv_sqrt, d_e_inv, graph_node_count, device):
    """
    Wavefront hypergraph spectral diffusion.

    Operator: Theta = D_v^{-1/2} H G D_e^{-1} H^T D_v^{-1/2}
    Update:   f^{t+1} = alpha * Theta @ f^{t} + (1-alpha) * f^{0}
    Gate G:   binary mask — block sentences with query-similarity below threshold
    Wavefront: each round keeps only top-K entities and re-normalizes,
               so the frontier advances K entities at a time.
    """
    num_entities = len(entity_hash_ids)
    num_sentences = entity_to_sentence_sparse.shape[1]
    entity_weights = np.zeros(graph_node_count)
    top_k = config.diffusion_top_k

    # Phase A: Initialize f^(0) from seed entities
    f0 = torch.zeros(num_entities, device=device)
    for idx, score in zip(seed_entity_indices, seed_entity_scores):
        f0[idx] = float(score)
    f0_norm = f0.norm()
    if f0_norm > 0:
        f0 = f0 / f0_norm

    # Phase B: Sentence gate — block sentences with low query-similarity
    question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
    sentence_sims = np.dot(sentence_embeddings, question_emb).flatten()
    w_q = torch.from_numpy(sentence_sims).float().to(device).clamp(min=0)
    gate_mask = (w_q >= config.sentence_gate_threshold).float()
    num_pass = int(gate_mask.sum().item())
    logger.info(f"Sentence gate (threshold={config.sentence_gate_threshold}): "
                f"{num_pass}/{num_sentences} pass, {num_sentences - num_pass} blocked")

    H = entity_to_sentence_sparse
    H_T = H.t()

    # Phase C: Wavefront diffusion
    alpha = config.diffusion_alpha
    f = f0.clone()
    activated_indices = set()

    for t in range(config.diffusion_max_iter):
        step1 = d_v_inv_sqrt * f
        step2 = torch.sparse.mm(H_T, step1.unsqueeze(1)).squeeze(1)
        step3 = d_e_inv * step2
        step4 = gate_mask * step3
        step5 = torch.sparse.mm(H, step4.unsqueeze(1)).squeeze(1)
        theta_f = d_v_inv_sqrt * step5

        f_new = alpha * theta_f + (1 - alpha) * f0
        delta = (f_new - f).norm().item()

        # Wavefront: keep top-K, zero rest, re-normalize
        top_indices = torch.topk(f_new, min(top_k, num_entities)).indices
        activated_indices.update(top_indices.cpu().tolist())
        mask = torch.zeros(num_entities, device=device)
        mask[top_indices] = 1.0
        f = f_new * mask
        f_norm = f.norm()
        if f_norm > 0:
            f = f / f_norm

        if delta < config.convergence_tol:
            logger.debug(f"Wavefront diffusion converged at iteration {t+1} (delta={delta:.6f})")
            break

    # Phase D: Collect activated entities (seeds + all wavefront discoveries)
    actived_entities = {}
    for idx, hash_id, score in zip(seed_entity_indices, seed_entity_hash_ids, seed_entity_scores):
        actived_entities[hash_id] = (idx, score, 1)
        if hash_id in node_name_to_vertex_idx:
            entity_weights[node_name_to_vertex_idx[hash_id]] = score

    if f0_norm > 0:
        final_scores = (f * f0_norm).cpu().numpy()
    else:
        final_scores = f.cpu().numpy()

    for entity_idx in activated_indices:
        hash_id = entity_hash_ids[entity_idx]
        if hash_id not in actived_entities:
            score = max(float(final_scores[entity_idx]), 0.01)
            actived_entities[hash_id] = (entity_idx, score, 1)
        if hash_id in node_name_to_vertex_idx:
            node_idx = node_name_to_vertex_idx[hash_id]
            entity_weights[node_idx] = max(entity_weights[node_idx], float(final_scores[entity_idx]))

    return entity_weights, actived_entities
