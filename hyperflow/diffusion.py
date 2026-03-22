"""Hypergraph spectral diffusion for entity score propagation.

Vertices  = entities
Hyperedges = semantic units (each hyperedge connects all entities that co-occur in one SU)
"""

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class Hypergraph:
    """Sparse hypergraph backed by an incidence matrix H.

    H[v, e] = 1  iff vertex v belongs to hyperedge e.

    Precomputes degree-normalisation vectors so that repeated propagation
    calls (one per query) only pay the construction cost once.
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

        # Degree vectors
        ones_e = torch.ones(self.num_hyperedges, 1, device=device)
        d_v = torch.sparse.mm(self.H, ones_e).squeeze()           # vertex degree
        ones_v = torch.ones(self.num_vertices, 1, device=device)
        d_e = torch.sparse.mm(self.H.t(), ones_v).squeeze()       # hyperedge degree

        self._d_v_inv_sqrt = torch.where(d_v > 0, d_v.pow(-0.5), torch.zeros_like(d_v))
        self._d_e_inv = torch.where(d_e > 0, d_e.pow(-1.0), torch.zeros_like(d_e))

        nnz = self.H._nnz()
        logger.info(
            "Hypergraph built: %d vertices, %d hyperedges, nnz=%d, "
            "sparsity=%.2f%%, D_v range [%.0f, %.0f], D_e range [%.0f, %.0f]",
            self.num_vertices, self.num_hyperedges, nnz,
            (1 - nnz / max(self.num_vertices * self.num_hyperedges, 1)) * 100,
            d_v.min(), d_v.max(), d_e.min(), d_e.max(),
        )

    def propagate(self, f, hyperedge_weights=None):
        """One-step spectral propagation on the hypergraph.

        Operator:  Θ_W f = D_v^{-1/2}  H  W  D_e^{-1}  H^T  D_v^{-1/2}  f

        Args:
            f: vertex signal vector  (num_vertices,)
            hyperedge_weights: optional per-hyperedge weight/gate vector (num_hyperedges,).
                               If None, all weights are 1 (standard Laplacian).

        Returns:
            Propagated vertex signal  (num_vertices,)
        """
        x = self._d_v_inv_sqrt * f                                         # D_v^{-1/2} f
        x = torch.sparse.mm(self.H.t(), x.unsqueeze(1)).squeeze(1)        # H^T x  → hyperedge signals
        x = self._d_e_inv * x                                              # D_e^{-1}
        if hyperedge_weights is not None:
            x = hyperedge_weights * x                                       # W (gate)
        x = torch.sparse.mm(self.H, x.unsqueeze(1)).squeeze(1)            # H x    → back to vertices
        x = self._d_v_inv_sqrt * x                                         # D_v^{-1/2}
        return x


def spectral_flow_propagation(config, hypergraph, entity_hash_ids,
                              su_embeddings, question_embedding,
                              seed_entity_indices, seed_entity_hash_ids,
                              seed_entity_scores, node_name_to_vertex_idx,
                              graph_node_count):
    """Anisotropic hypergraph spectral diffusion with novelty-seeking conductance.

    Operator:  Θ_{W(t)} = D_v^{-1/2}  H  W(t)  D_e^{-1}  H^T  D_v^{-1/2}
    Update:    f^{t+1} = α · Θ_{W(t)} f^t  +  (1-α) · f^0

    Conductance W(t) is query-conditioned and diversity-aware:

        w_e^(t) = [ (sim(e, q) - τ)_+ / (1 - τ) ]^γ          (base relevance)
                  × (1 - β · max_{e' ∈ covered} sim(e, e'))    (novelty factor)

    The base conductance is a continuous soft gate: hyperedges with query
    similarity below τ receive zero conductance; above τ the weight is
    power-compressed (γ < 1 broadens mid-range contributions).
    The novelty factor penalises hyperedges similar to those already
    explored, encouraging the diffusion frontier to spread into novel
    regions of the hypergraph — analogous to anisotropic diffusion
    where the medium's conductivity adapts to the diffusing signal.

    Wavefront: each round keeps only top-K vertices and re-normalises,
    so the frontier advances K vertices at a time.
    """
    device = hypergraph.device
    num_vertices = hypergraph.num_vertices
    num_hyperedges = hypergraph.num_hyperedges
    entity_weights = np.zeros(graph_node_count)
    top_k = config.diffusion_top_k

    # Phase A: Initialise f^(0) from seed entities
    f0 = torch.zeros(num_vertices, device=device)
    for idx, score in zip(seed_entity_indices, seed_entity_scores):
        f0[idx] = float(score)
    f0_norm = f0.norm()
    if f0_norm > 0:
        f0 = f0 / f0_norm

    # Phase B: Query-conditioned base conductance (continuous soft gate)
    question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
    su_sims = np.dot(su_embeddings, question_emb).flatten()
    w_q = torch.from_numpy(su_sims).float().to(device).clamp(min=0)

    floor = config.conductance_floor
    gamma = config.conductance_gamma
    w_base = torch.clamp(w_q - floor, min=0) / max(1.0 - floor, 1e-8)
    w_conductance = w_base.pow(gamma)

    num_active = int((w_conductance > 0).sum().item())
    logger.info(
        "Anisotropic conductance (floor=%.2f, gamma=%.2f): "
        "%d/%d active hyperedges, %d suppressed",
        floor, gamma, num_active, num_hyperedges, num_hyperedges - num_active,
    )

    # Phase C: Anisotropic diffusion with diversity-aware conductance
    alpha = config.diffusion_alpha
    beta = config.conductance_diversity_beta
    f = f0.clone()
    activated_indices = set()

    # Precompute SU embedding tensor for diversity computation
    su_emb_tensor = torch.from_numpy(su_embeddings).float().to(device) if beta > 0 else None
    H_t = hypergraph.H.t()  # (|E|, |V|) sparse — maps vertices to hyperedges
    covered_su_indices = set()

    for t in range(config.diffusion_max_iter):
        # Compute per-round conductance: base relevance × novelty
        if beta > 0 and len(covered_su_indices) > 0:
            covered_list = sorted(covered_su_indices)
            covered_embs = su_emb_tensor[covered_list]               # (|covered|, dim)
            sim_matrix = torch.mm(su_emb_tensor, covered_embs.t())   # (|E|, |covered|)
            max_sim = sim_matrix.max(dim=1).values.clamp(min=0)      # (|E|,)
            novelty = (1.0 - beta * max_sim).clamp(min=0)
            w_t = w_conductance * novelty
        else:
            w_t = w_conductance

        theta_f = hypergraph.propagate(f, hyperedge_weights=w_t)
        f_new = alpha * theta_f + (1 - alpha) * f0
        delta = (f_new - f).norm().item()

        # Wavefront: keep top-K, zero rest, re-normalise
        top_indices = torch.topk(f_new, min(top_k, num_vertices)).indices
        activated_indices.update(top_indices.cpu().tolist())

        # Update covered hyperedges: find SUs containing any top-K vertex
        if beta > 0:
            indicator = torch.zeros(num_vertices, 1, device=device)
            indicator[top_indices] = 1.0
            su_activation = torch.sparse.mm(H_t, indicator).squeeze(1)  # (|E|,)
            newly_covered = (su_activation > 0).nonzero(as_tuple=True)[0].cpu().tolist()
            covered_su_indices.update(newly_covered)

        mask = torch.zeros(num_vertices, device=device)
        mask[top_indices] = 1.0
        f = f_new * mask
        f_norm = f.norm()
        if f_norm > 0:
            f = f / f_norm

        if delta < config.convergence_tol:
            logger.debug("Anisotropic diffusion converged at iteration %d (delta=%.6f)", t + 1, delta)
            break

    if beta > 0:
        logger.info("Diffusion finished: %d rounds, %d covered hyperedges, %d activated vertices",
                     t + 1, len(covered_su_indices), len(activated_indices))

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
