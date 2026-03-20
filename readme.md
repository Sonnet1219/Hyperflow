# Hyperflow: Zero-Token Hypergraph Indexing for Single-Step Multi-Hop Retrieval

> Hyperflow turns a document collection into a reasoning-ready indexing hypergraph. Instead of spending large amounts of LLM tokens during offline graph construction, it uses NER-based hyperedge construction and flow-damped hypergraph diffusion to support single-step multi-hop retrieval for complex question answering.

## Overview

Many existing GraphRAG systems rely on LLMs in the offline indexing stage to extract relations, summarize communities, or generate graph structure. This makes indexing expensive and often introduces noisy, highly connected hub nodes that can dominate downstream reasoning.

Hyperflow is designed to address both issues:

- It builds a retrieval-oriented indexing hypergraph directly from the corpus, without asking an LLM to generate explicit relations one by one.
- It uses entity recognition and semantic linking to construct hyperedges at almost zero token cost during indexing.
- At query time, it performs hypergraph spectral diffusion with flow damping, which weakens the influence of hub nodes and activates multiple related evidence chains in one retrieval pass.
- This enables single-step multi-hop retrieval: instead of retrieving hop by hop, Hyperflow can connect scattered evidence across entities, sentences, and passages in one shot.

In short, Hyperflow aims to be a lower-cost and more robust alternative to LLM-heavy GraphRAG pipelines for multi-hop QA.

## Core Idea

Hyperflow has two stages: offline indexing and online retrieval.

### 1. Offline indexing: build an indexing hypergraph with near-zero token cost

Given a corpus of passages, Hyperflow:

1. Uses spaCy NER to extract entities from each passage.
2. Splits passages into sentence-level contexts.
3. Treats each sentence as a hyperedge that connects all entities appearing in that sentence.
4. Stores embeddings for passages, entities, and sentences.
5. Builds an auxiliary graph over passages and entities for final ranking.

The key point is that no LLM is required to explicitly generate relation triples or graph edges during indexing. Hyperedges come directly from entity-sentence co-occurrence, so the offline stage is fast and essentially token-free.

### 2. Online retrieval: single-step multi-hop reasoning over the hypergraph

For each question, Hyperflow:

1. Extracts seed entities from the query.
2. Semantically links them to indexed entities using embedding similarity.
3. Runs query-aware hypergraph spectral diffusion from these seed entities.
4. Scores passages by combining dense retrieval signals with activated entity evidence.
5. Applies personalized PageRank on the passage-entity graph.
6. Uses a reranker to select the final evidence passages.
7. Sends the retrieved evidence to a reader LLM to generate the answer.

This lets the system retrieve evidence chains that span multiple passages without explicit hop-by-hop planning.

## Core Algorithms

### Indexing hypergraph

Hyperflow models the corpus as an entity-sentence hypergraph:

- Vertices: entities
- Hyperedges: sentences
- Passage nodes: used later for ranking and evidence aggregation

If an entity appears in a sentence, the corresponding entity node is connected to that sentence hyperedge. This gives a sparse incidence structure that preserves contextual grouping without requiring explicit relation extraction.

### Query-aware hypergraph spectral diffusion

During retrieval, Hyperflow propagates relevance scores from seed entities over the hypergraph using a query-adaptive diffusion operator. The implementation follows the form:

`f^(t+1) = alpha * Theta_q * f^(t) + (1 - alpha) * f^(0)`

where `Theta_q` is built from:

- the entity-sentence incidence matrix `H`
- vertex and hyperedge degree normalization
- query-aware sentence weights `W_q`

This allows the retrieval process to spread from query entities to semantically relevant sentence contexts and then back to other related entities.

### Flow damping for hub suppression

A common issue in graph retrieval is that hub nodes or frequently reused contexts can dominate propagation. Hyperflow addresses this with a flow-inspired damping mechanism:

- hyperedges that accumulate too much propagated flow are progressively down-weighted
- repeated activation of the same popular bridge becomes less influential over time
- diffusion is encouraged to explore alternative but still query-relevant paths

This makes retrieval more stable and helps reduce the interference caused by oversized hub structures.

### Passage scoring and reranking

After diffusion, Hyperflow combines:

- dense passage retrieval scores
- bonuses from activated entities found in each passage
- personalized PageRank over the graph
- local reranking with `Qwen/Qwen3-Reranker-4B`

The result is a final list of evidence passages that are passed to the reader LLM.

## Why Hyperflow

Compared with a typical GraphRAG pipeline, Hyperflow emphasizes:

- Lower offline cost: no LLM-based relation extraction during indexing
- Better scalability: sparse hypergraph construction with embedding stores
- More robust retrieval: flow damping weakens hub-node effects
- Single-step multi-hop retrieval: multiple evidence chains can be activated in one pass
- Simpler graph construction: NER plus semantic linking instead of explicit relation generation

## Repository Structure

```text
hyperflow/
  config.py          # Global configuration
  diffusion.py       # Hypergraph spectral diffusion and flow damping
  embedding_store.py # Persistent embedding stores for passages/entities/sentences
  engine.py          # Indexing, retrieval, QA pipeline
  evaluate.py        # LLM-based and string-containment evaluation
  graph.py           # Graph construction, PPR, passage scoring
  ner.py             # spaCy-based entity extraction
  reranker.py        # Qwen reranker
  utils.py           # OpenAI-compatible LLM wrapper and utilities
run.py               # Main entry point
scripts/run.sh       # Example launch script
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

Hyperflow uses an OpenAI-compatible chat completion API for final answer generation and optional evaluation.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"
```

### 4. Prepare the embedding model

By default, the code expects a local sentence-transformer model at:

```text
model/all-mpnet-base-v2
```

### 5. Prepare the datasets

Each dataset should contain:

```text
dataset/<dataset_name>/chunks.json
dataset/<dataset_name>/questions.json
```

This repository currently includes example layouts for:

- `2wikimultihop`
- `hotpotqa`
- `musique`

## Quick Start

```bash
python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name 2wikimultihop \
  --llm_model gpt-4o-mini \
  --max_workers 16 \
  --passage_ratio 0.05 \
  --diffusion_alpha 0.85 \
  --diffusion_max_iter 10 \
  --flow_damping 0.5 \
  --activation_ratio 0.05
```

For a fast smoke test, you can limit the number of questions:

```bash
python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name 2wikimultihop \
  --llm_model gpt-4o-mini \
  --question_limit 20
```

If you only want predictions and want to skip the evaluation stage:

```bash
python run.py \
  --spacy_model en_core_web_trf \
  --embedding_model model/all-mpnet-base-v2 \
  --dataset_name 2wikimultihop \
  --llm_model gpt-4o-mini \
  --skip_evaluation
```

## Important Arguments

| Argument | Description |
| --- | --- |
| `--dataset_name` | Dataset folder under `dataset/` |
| `--spacy_model` | spaCy model used for NER |
| `--embedding_model` | Local sentence-transformer path |
| `--llm_model` | Reader/evaluator LLM name |
| `--passage_ratio` | Weight of dense passage retrieval in final passage scoring |
| `--diffusion_alpha` | Diffusion strength; larger values favor broader propagation |
| `--diffusion_max_iter` | Maximum number of diffusion iterations |
| `--convergence_tol` | Early-stop threshold for diffusion convergence |
| `--flow_damping` | Hyperedge flow damping factor; smaller values more strongly suppress repeatedly used paths |
| `--activation_ratio` | Adaptive threshold for activating diffused entities |
| `--use_context_modulation` | Use context-aware incidence weighting instead of binary incidence |
| `--reranker_model` | Local or Hub path for the reranker |
| `--reranker_candidate_top_k` | Number of retrieved candidates passed to the reranker |
| `--question_limit` | Limit the number of questions for debugging |
| `--skip_evaluation` | Skip answer evaluation |

## Outputs

### Offline indexing artifacts

Stored under `import/<dataset_name>/`:

- `passage_embedding.parquet`
- `entity_embedding.parquet`
- `sentence_embedding.parquet`
- `ner_results.json`
- `Hyperflow.graphml`

### Run outputs

Stored under `results/<dataset_name>/<timestamp>/`:

- `predictions.json`
- `evaluation_results.json`
- `log.txt`

## Current Pipeline in Code

The current implementation in this repository follows this pipeline:

1. Insert passage embeddings into a persistent store.
2. Run spaCy NER and cache entity extraction results.
3. Build entity-sentence incidence mappings.
4. Build sparse hypergraph structures and graph edges.
5. Extract query seed entities and semantically link them to indexed entities.
6. Perform hypergraph spectral diffusion with flow damping.
7. Score passages and run personalized PageRank.
8. Rerank final evidence passages.
9. Ask the reader LLM to answer from retrieved evidence.
10. Optionally evaluate predictions with an LLM judge and containment metric.

## Notes

- Hyperflow avoids LLM usage in the indexing stage, but it still uses an LLM for final answer generation and optional evaluation.
- The reranker is loaded locally through Hugging Face Transformers and may require substantial GPU memory depending on the chosen model.
- The default code path uses CUDA when available for embedding inference, diffusion, and reranking.

## License

This repository is released under the license provided in `LICENSE.txt`.
