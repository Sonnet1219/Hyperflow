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


def build_context_aware_incidence(H, entity_embeddings, sentence_embeddings, query_embedding, device):
    """Build context-modulated incidence matrix: H_ctx[v,e] = cos_sim(entity_v * sentence_e, query)."""
    H_coalesced = H.coalesce()
    indices = H_coalesced.indices()
    entity_indices = indices[0].cpu().numpy()
    sentence_indices = indices[1].cpu().numpy()

    entity_embs = entity_embeddings[entity_indices]
    sentence_embs = sentence_embeddings[sentence_indices]
    context_embs = entity_embs * sentence_embs                    # Hadamard product

    norms = np.linalg.norm(context_embs, axis=1, keepdims=True)
    context_embs = context_embs / np.maximum(norms, 1e-8)
    query_emb = query_embedding.reshape(-1, 1) if len(query_embedding.shape) == 1 else query_embedding
    sims = np.dot(context_embs, query_emb).flatten()
    sims = np.maximum(sims, 0.0)

    values = torch.from_numpy(sims.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(indices, values, H_coalesced.shape, device=device).coalesce()


def spectral_flow_propagation(config, entity_hash_ids, entity_embeddings, sentence_embeddings,
                              question_embedding, seed_entity_indices, seed_entity_hash_ids,
                              seed_entity_scores, entity_to_sentence_sparse, node_name_to_vertex_idx,
                              d_v_inv_sqrt, d_e_inv, graph_node_count, device):
    """
    Hypergraph spectral diffusion with flow damping and adaptive activation.

    Operator: Theta = D_v^{-1/2} H W_q^{(t)} D_e^{-1} H^T D_v^{-1/2}
    Update:   f^{t+1} = alpha * Theta @ f^{t} + (1-alpha) * f^{0}
    Flow damping: W_q^{(t)} = W_q * gamma^flow_cumulative
    Activation: adaptive threshold = top_score * activation_ratio
    """
    num_entities = len(entity_hash_ids)
    num_sentences = entity_to_sentence_sparse.shape[1]
    entity_weights = np.zeros(graph_node_count)

    # Phase A: Initialize f^(0) from seed entities
    f0 = torch.zeros(num_entities, device=device)
    for idx, score in zip(seed_entity_indices, seed_entity_scores):
        f0[idx] = float(score)
    f0_norm = f0.norm()
    if f0_norm > 0:
        f0 = f0 / f0_norm

    # Phase B: Query-adaptive hyperedge weights W_q
    question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
    sentence_sims = np.dot(sentence_embeddings, question_emb).flatten()
    w_q = torch.from_numpy(sentence_sims).float().to(device).clamp(min=0)

    # Phase C: Select H and degree matrices (context modulation or binary)
    if config.use_context_modulation:
        H = build_context_aware_incidence(entity_to_sentence_sparse, entity_embeddings,
                                          sentence_embeddings, question_embedding, device)
        H_T = H.t()
        d_v = torch.sparse.mm(H, torch.ones(H.shape[1], 1, device=device)).squeeze()
        d_e = torch.sparse.mm(H_T, torch.ones(H.shape[0], 1, device=device)).squeeze()
        d_v_inv_sqrt_local = torch.where(d_v > 0, d_v.pow(-0.5), torch.zeros_like(d_v))
        d_e_inv_local = torch.where(d_e > 0, d_e.pow(-1.0), torch.zeros_like(d_e))
    else:
        H = entity_to_sentence_sparse
        H_T = H.t()
        d_v_inv_sqrt_local = d_v_inv_sqrt
        d_e_inv_local = d_e_inv

    # Phase D: Iterative spectral diffusion with hyperedge flow damping
    alpha = config.diffusion_alpha
    gamma = config.flow_damping
    f = f0.clone()
    flow_cumulative = torch.zeros(num_sentences, device=device)

    for t in range(config.diffusion_max_iter):
        w_q_damped = w_q * (gamma ** flow_cumulative)

        step1 = d_v_inv_sqrt_local * f                                          # D_v^{-1/2} @ f
        step2 = torch.sparse.mm(H_T, step1.unsqueeze(1)).squeeze(1)            # H^T @ step1
        step3 = d_e_inv_local * step2                                           # D_e^{-1} @ step2
        step4 = w_q_damped * step3                                              # W_q^{(t)} @ step3
        step5 = torch.sparse.mm(H, step4.unsqueeze(1)).squeeze(1)              # H @ step4
        theta_f = d_v_inv_sqrt_local * step5                                    # D_v^{-1/2} @ step5

        f_new = alpha * theta_f + (1 - alpha) * f0

        flow_cumulative += step4.abs()

        delta = (f_new - f).norm().item()
        f = f_new
        if delta < config.convergence_tol:
            logger.debug(f"Spectral diffusion converged at iteration {t+1} (delta={delta:.6f})")
            break

    # Phase E: Convert to entity_weights and actived_entities
    if f0_norm > 0:
        f = f * f0_norm
    entity_scores_np = f.cpu().numpy()

    top_score = entity_scores_np.max()
    adaptive_threshold = top_score * config.activation_ratio

    actived_entities = {}
    for idx, hash_id, score in zip(seed_entity_indices, seed_entity_hash_ids, seed_entity_scores):
        actived_entities[hash_id] = (idx, score, 1)
        if hash_id in node_name_to_vertex_idx:
            node_idx = node_name_to_vertex_idx[hash_id]
            entity_weights[node_idx] = score

    for entity_idx in range(num_entities):
        score = entity_scores_np[entity_idx]
        if score >= adaptive_threshold:
            hash_id = entity_hash_ids[entity_idx]
            if hash_id in node_name_to_vertex_idx:
                node_idx = node_name_to_vertex_idx[hash_id]
                entity_weights[node_idx] = max(entity_weights[node_idx], float(score))
                if hash_id not in actived_entities:
                    actived_entities[hash_id] = (entity_idx, float(score), 1)
    return entity_weights, actived_entities
