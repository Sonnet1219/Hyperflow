# Hyperflow: Description-Aware Hypergraph Indexing for Single-Step Multi-Hop Retrieval

> Hyperflow turns a document collection into a reasoning-ready hypergraph index. It builds a semantic-unit hypergraph with LLM-extracted canonical entities and frontier expansion to support single-step multi-hop retrieval for complex question answering.

## Overview

Many existing GraphRAG systems rely on LLMs in the offline indexing stage to extract relations, summarize communities, or generate graph structure. This makes indexing expensive and often introduces noisy, highly connected hub nodes that can dominate downstream reasoning.

Hyperflow is designed to address both issues:

- It builds a retrieval-oriented hypergraph directly from the corpus, without asking an LLM to generate explicit relations.
- It uses LangExtract to produce semantic-unit mentions with grounded descriptions, then merges them into canonical entity nodes for hypergraph construction.
- At query time, it performs hop-wise frontier expansion with conductance gating and progressive query steering, activating multiple related evidence chains in one retrieval pass.
- This enables single-step multi-hop retrieval: instead of retrieving hop by hop, Hyperflow can connect scattered evidence across entities, semantic units, and passages in one shot.

## Core Idea

Hyperflow has two stages: offline indexing and online retrieval.

### 1. Offline Indexing: Build a Hypergraph from Semantic Units and Canonical Entities

Given a corpus of passages, Hyperflow:

1. Chunks the corpus into ~1200-token passages with overlap.
2. Splits each passage into semantic units via Kamradt Percentile merging (embedding-based sentence grouping).
3. Runs LangExtract on each semantic unit to extract grounded entity mentions with short descriptions.
4. Merges mention-level entities into canonical global entity nodes using normalized names plus embedding similarity.
5. Treats each semantic unit as a hyperedge connecting all canonical entities that appear in it.
6. Stores embeddings for passages, entities, and semantic units in Parquet files.
7. Builds an auxiliary passage-entity graph for final ranking.

Hyperedges come directly from canonical entity-semantic unit co-occurrence.

### 2. Online Retrieval: Single-Step Multi-Hop Reasoning

For each question, Hyperflow:

1. Extracts seed entities from the query via LangExtract.
2. Matches them to the nearest indexed entities by embedding similarity.
3. Runs frontier expansion on the hypergraph:
   - Each hop discovers new entities through query-relevant semantic units, gated by conductance.
   - Progressive query steering shifts the query embedding toward discovered entities between hops.
4. Scores passages via dual-channel fusion:
   - Channel 1 (semantic): cosine similarity between passage and query embeddings.
   - Channel 2 (coverage): activation-weighted entity coverage with IDF.
5. Optionally reranks candidates with `Qwen/Qwen3-Reranker-4B`.
6. Sends the retrieved evidence to a reader LLM to generate the answer.

## Core Algorithms

### Hypergraph Structure

Hyperflow models the corpus as an entity-semantic unit hypergraph:

- **Vertices**: canonical entities (built from mention-level entity + description pairs)
- **Hyperedges**: semantic units (sentence groups produced by Kamradt Percentile chunking)
- **Incidence matrix H**: `H[v, e] = 1` iff entity `v` appears in semantic unit `e`

### Frontier Expansion

During retrieval, Hyperflow propagates relevance from seed entities through the hypergraph using explicit BFS-like hop-by-hop exploration:

1. **Conductance gating**: each semantic unit receives a conductance weight based on its embedding similarity to the query. Units below a floor threshold are suppressed entirely.
2. **Hop propagation**: at each hop, frontier entity scores flow through high-conductance semantic units to discover new entities. Candidates are scored via max-aggregation and top-K selection.
3. **Hop decay**: scores decay exponentially with hop distance (`score × decay^hop`).
4. **Progressive query steering**: after each hop, the query embedding is shifted toward the centroid of top activated entities, updating conductance for the next hop.

### Dual-Channel Passage Scoring

After frontier expansion, passages are scored by fusing two signals:

```
final_score(p) = λ · semantic(p) + (1-λ) · coverage(p)
```

- `semantic(p)`: normalized cosine similarity between passage and query embeddings
- `coverage(p)`: sum of (activation score × IDF) for each activated entity in the passage
- `λ`: configurable fusion weight (default 0.5)

### Reranking

Optionally, top candidates are reranked using a Qwen3-based reranker that scores query-document relevance via yes/no logit comparison.

## Repository Structure

```text
hyperflow/
  config.py               # HyperflowConfig dataclass
  engine.py               # Main engine: indexing, retrieval, QA pipeline
  frontier.py             # Frontier expansion algorithm
  knowledge_graph.py      # Hypergraph structure, entity-passage graph
  embedding_store.py      # Parquet-backed embedding store
  ner.py                  # LangExtract-based mention extraction
  chunker.py              # Token chunking + Kamradt semantic unit splitting
  entity_normalization.py # Mention normalization + canonical entity merging
  reranker.py             # Qwen3 reranker
  utils.py                # LLM wrapper, hashing, logging utilities
benchmarks/
  graphrag_bench/          # GraphRAG-Bench benchmark runner
  multihop/                # Multi-hop QA benchmark runner (HotpotQA, 2WikiMultiHop, MuSiQue)
```

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install the spaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3. Set your LLM API credentials

Hyperflow uses an OpenAI-compatible chat completion API for both LangExtract-based indexing and final answer generation.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"
```

### 4. Embedding model

By default, the code uses `BAAI/bge-large-en-v1.5` via sentence-transformers (downloaded automatically from Hugging Face Hub).

## Quick Start

### GraphRAG-Bench

```bash
python benchmarks/graphrag_bench/run.py \
  --corpus_name medical \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --scoring_lambda 0.7
```

### Multi-hop QA

```bash
python benchmarks/multihop/run.py \
  --dataset_name hotpotqa \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --scoring_lambda 0.7
```

For a fast smoke test, limit the number of questions:

```bash
python benchmarks/graphrag_bench/run.py \
  --corpus_name medical \
  --question_limit 20
```

## Key Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--langextract_model` | Model used by LangExtract for mention extraction | `gpt-4o-mini` |
| `--langextract_max_char_buffer` | Max char buffer per LangExtract extraction call | `1000` |
| `--langextract_extraction_passes` | Sequential LangExtract passes for higher recall | `1` |
| `--expansion_max_hops` | Max BFS hops from seed entities | `3` |
| `--expansion_top_k` | New entities discovered per hop | `15` |
| `--hop_decay` | Score decay per hop | `0.5` |
| `--conductance_floor` | Query-SU similarity below this is suppressed | `0.3` |
| `--conductance_gamma` | Conductance power exponent | `1.0` |
| `--scoring_lambda` | Fusion weight (1.0 = pure dense, 0.0 = pure entity coverage) | `0.7` |
| `--semantic_unit_percentile` | Kamradt percentile for SU boundary detection | `60` |
| `--no_rerank` | Disable reranking | `false` |
| `--reranker_model` | Reranker model path | `Qwen/Qwen3-Reranker-4B` |
| `--reranker_candidate_top_k` | Candidates passed to reranker | `30` |
| `--question_limit` | Limit number of questions (for debugging) | all |
| `--skip_qa` / `--skip_eval` | Skip QA generation or evaluation | `false` |

## Index Store

Stored under `index_store/<dataset_name>/`:

| File | Content |
| --- | --- |
| `passage_embedding.parquet` | Passage text + embeddings |
| `entity_embedding.parquet` | Entity text + embeddings |
| `su_embedding.parquet` | Semantic unit text + embeddings |
| `ner_results.json` | Cached extraction results (passage → SU ids, SU hash → mention records) |
| `entity_nodes.json` | Canonical entity node metadata |

### Run Outputs

Stored under `results/<dataset_name>/<timestamp>/`:

- `predictions.json`
- `evaluation_results.json`
- `log.txt`

## Notes

- Hyperflow now uses an LLM during indexing through LangExtract to create mention-level entities with grounded descriptions.
- The reranker is loaded locally through Hugging Face Transformers and may require substantial GPU memory.
- The default code path uses CUDA when available for embedding inference, hypergraph operations, and reranking.

## License

This repository is released under the license provided in `LICENSE.txt`.
