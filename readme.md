# HyperSU: Semantic-Unit Hypergraph Construction and Query-Gated Hypergraph Expansion

HyperSU is a hypergraph-based retrieval framework for multi-hop question answering.
Its central object is not a flat passage index and not an LLM-generated relation graph,
but a corpus-induced semantic hypergraph:

- vertices are canonical entities
- hyperedges are semantic units
- retrieval is formulated as query-conditioned expansion on the entity-hyperedge incidence structure

Instead of asking an LLM to synthesize explicit n-ary relation hyperedges, HyperSU builds
hyperedges directly from semantic units in the source corpus, then performs gated frontier
propagation over the resulting sparse hypergraph.

The method description in this README follows that hypergraph-first view of the project,
including planner-guided sub-query gating and semantic-unit fallback expansion.

## Motivation

For multi-hop retrieval, the core difficulty is usually not only ranking passages, but
recovering the latent higher-order connectivity that links entities, intermediate evidence,
and final answer evidence.

HyperSU approaches this problem from the hypergraph side:

- Each semantic unit is treated as a higher-order connector rather than as an isolated text span.
- The corpus is organized as a sparse entity-semantic-unit hypergraph rather than as a set of
  independent passages.
- Query processing becomes hypergraph expansion, not only nearest-neighbor passage lookup.
- Complex questions are handled by sub-query-gated hyperedge activation, so different parts of
  the reasoning chain can excite different regions of the hypergraph.

This makes the hypergraph the primary retrieval object, while passages are the final evidence
projection target.

## Hypergraph View

HyperSU has two stages: hypergraph construction and hypergraph querying.

### 1. Hypergraph Construction

Given a corpus, HyperSU:

1. Splits the corpus into passages using token-budgeted semantic chunking.
2. Splits each passage into semantic units using embedding-based sentence grouping.
3. Runs LangExtract on each semantic unit to extract grounded entity mentions.
4. Merges mention-level variants into canonical entity nodes with normalized names and
   embedding similarity.
5. Treats each semantic unit as a hyperedge connecting all canonical entities that appear in it.
6. Stores passage, entity, and semantic-unit embeddings for later hypergraph querying.
7. Builds an auxiliary passage-entity projection map for final evidence scoring.

This yields an entity-semantic-unit hypergraph:

- Vertex set `V`: canonical entities
- Hyperedge set `E`: semantic units
- Incidence matrix `H`: `H[v, e] = 1` iff entity vertex `v` belongs to semantic-unit hyperedge `e`

The semantic unit is therefore the basic higher-order neighborhood in the system. One
hyperedge may connect multiple entities at once, which is exactly why the retrieval problem
is framed as hypergraph exploration rather than ordinary pairwise graph traversal.

### 2. Hypergraph Querying

For each question, HyperSU:

1. Decomposes complex questions into sub-queries with a planner.
2. Uses sub-query-to-SU relevance, rather than raw query-to-SU relevance, to assign gate
   strength to semantic-unit hyperedges.
3. Extracts seed entities from the query and maps them to canonical vertices in the hypergraph.
4. Runs hop-wise frontier expansion over the hypergraph to discover new vertices through
   gated hyperedges.
5. If query-entity extraction fails, falls back to the single most relevant semantic unit,
   performs one hyperedge pre-expansion step, and activates all entities contained in that
   hyperedge as pseudo-seeds.
6. Continues expansion from those activated entities, allowing later hops to naturally filter
   noisy early activations through gated hypergraph propagation.
7. Projects the activated entity set back to passages and ranks passages with a dual signal:
   semantic relevance plus activated-entity coverage.
8. Optionally reranks the final candidates with a local reranker.
9. Sends the retrieved passages to a reader LLM for answer generation.

## Hypergraph Formulation

HyperSU can be read as a three-part retrieval pipeline:

1. Hypergraph construction:
   build `H` from entity-hyperedge membership induced by semantic units.
2. Hypergraph querying:
   score and gate hyperedges, initialize seed vertices, and propagate activation.
3. Hypergraph-to-passage projection:
   map activated vertices onto passages for final evidence ranking.

This separation is useful because it makes clear that passages are not the structure over
which expansion happens. Expansion happens on the hypergraph; passages are only where the
retrieved evidence is finally collected.

## Hypergraph Design Choices

### Semantic Units as Hyperedges

HyperSU does not ask an LLM to invent hyperedges. Instead, it derives hyperedges from
corpus-local semantic units. This makes the hypergraph:

- cheaper to build
- easier to ground in source text
- better aligned with evidence connectivity than with explicit relation extraction

The result is a retrieval hypergraph whose hyperedges correspond to local evidence groupings
rather than abstract generated relations.

### Sub-Query-Gated Hyperedge Conductance

Using only `(query, SU)` similarity as the hyperedge gate can be too coarse for hard
multi-hop questions, because one global query often mixes multiple latent evidence needs.

HyperSU therefore uses `(sub_query, SU)` similarity as the hyperedge conductance signal.
The intuition is:

- a sub-query can match an intermediate evidence unit more sharply
- different hyperedges can be activated by different sub-queries
- hypergraph expansion becomes more stable on complex reasoning questions

This converts hyperedge activation from a single global score into a decomposition-aware
conductance mechanism.

### Hyperedge Pre-Expansion Fallback

When LangExtract fails to extract query entities, HyperSU does not stop at dense passage
retrieval.
Instead, it:

1. retrieves the most relevant semantic-unit hyperedge
2. activates all entities contained in that hyperedge
3. uses those entities as pseudo-seeds for the next expansion step

This intentionally allows some early noise.
The design assumption is that noisy entities introduced by the fallback step are acceptable,
because later hypergraph expansion is still gated and scored, so irrelevant branches are
naturally suppressed in subsequent hops.

### Hypergraph Frontier Expansion

HyperSU performs explicit hop-wise expansion on the entity-semantic-unit hypergraph.
At each hop it:

1. starts from the current frontier vertices
2. finds reachable hyperedges under the gate
3. propagates activation through those hyperedges
4. discovers new entity vertices
5. keeps the highest-value activated vertices for the next hop

This is a genuine hypergraph retrieval step: activation flows from vertices to hyperedges and
back to vertices, instead of traversing only pairwise edges.

### Hypergraph-to-Passage Projection

After hypergraph expansion, the activated vertices are projected back to passages.
Passages are then ranked by combining:

- semantic score: passage-query embedding similarity
- coverage score: how much activated vertex evidence a passage contains

The second term helps keep passages that may not look globally similar to the question, but
contain crucial bridge entities discovered by the hypergraph expansion process.

## Repository Layout

```text
hypersu/
  __init__.py
  config.py
  engine.py
  frontier.py
  knowledge_graph.py
  chunker.py
  ner.py
  entity_normalization.py
  embedding_store.py
  reranker.py
  utils.py
  planner.py

benchmarks/
  graphrag_bench/
    bench.py
    run.py
  multihop/
    evaluate.py
    run.py

figure/
  main-result.png
```

## Main Components

### `hypersu/chunker.py`

- token-aware corpus chunking
- semantic-unit construction from sentence embeddings
- semantic-unit length balancing by merge/split

### `hypersu/ner.py`

- LangExtract-based entity mention extraction
- grounded mention metadata
- query-time entity extraction

### `hypersu/entity_normalization.py`

- mention normalization
- canonical entity construction
- mention merging by name and embedding similarity

### `hypersu/knowledge_graph.py`

- entity-SU incidence mapping
- sparse hypergraph construction
- hypergraph storage and passage projection support

### `hypersu/frontier.py`

- hop-wise hypergraph frontier expansion
- conductance-style hyperedge gating
- multi-hop vertex activation

### `hypersu/planner.py`

- standalone query planner for complex-question decomposition
- generates retrieval-oriented sub-queries
- supports the sub-query-gated retrieval design described above

### `hypersu/engine.py`

- hypergraph construction pipeline
- hypergraph querying pipeline
- fallback initialization logic
- QA generation pipeline

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install spaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3. Set API credentials

LangExtract-based extraction and answer generation use an OpenAI-compatible API.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-base-url"
```

## Quick Start

### GraphRAG-Bench

```bash
python benchmarks/graphrag_bench/run.py \
  --corpus_name medical \
  --llm_model gpt-4o-mini \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --expansion_top_k 15 \
  --conductance_floor 0.5 \
  --conductance_gamma 1.0 \
  --scoring_lambda 0.7
```

Run a single corpus entry:

```bash
python benchmarks/graphrag_bench/run.py \
  --corpus_name novel \
  --corpus_entry Novel-30752 \
  --question_limit 20
```

### Multi-Hop QA Datasets

```bash
python benchmarks/multihop/run.py \
  --dataset_name hotpotqa \
  --llm_model gpt-4o-mini \
  --langextract_model gpt-4o-mini \
  --expansion_max_hops 3 \
  --expansion_top_k 15 \
  --scoring_lambda 0.7
```

## Important Arguments

### Chunking and semantic units

- `--chunk_size`: token budget for corpus chunking
- `--chunk_overlap`: overlap between neighboring chunks
- `--semantic_unit_percentile`: breakpoint percentile for semantic-unit splitting

### Entity extraction

- `--langextract_model`: model used by LangExtract
- `--langextract_max_char_buffer`: max char buffer per extraction call
- `--langextract_extraction_passes`: number of extraction passes

### Retrieval and expansion

- `--expansion_max_hops`: maximum number of hypergraph expansion hops
- `--expansion_top_k`: number of newly activated vertices kept per hop
- `--hop_decay`: hop-distance decay factor for propagated activation
- `--conductance_floor`: floor for hyperedge gate suppression
- `--conductance_gamma`: exponent for hyperedge conductance sharpening
- `--scoring_lambda`: fusion weight between dense passage relevance and hypergraph coverage

### Reranking

- `--no_rerank`: disable reranking in GraphRAG-Bench runs
- `--reranker_model`: reranker checkpoint
- `--reranker_candidate_top_k`: candidates passed to reranker
- `--reranker_batch_size`: reranker batch size
- `--reranker_max_length`: reranker max token length

## Outputs

Typical outputs are written under:

```text
index_store/
results/
```

Common files include:

- `passage_embedding.parquet`
- `entity_embedding.parquet`
- `su_embedding.parquet`
- `ner_results.json`
- `entity_nodes.json`
- `predictions.json`
- `generation_eval.json`
- `retrieval_eval.json`
- `log.txt`

## Notes

- HyperSU should be understood first as a semantic hypergraph retrieval method and only
  second as a downstream RAG pipeline.
- LangExtract is used to ground entity vertices, while semantic units define the hyperedges.
- The planner-guided gate is designed to control which hyperedges become conductive for
  expansion on complex multi-hop questions.
- The SU pre-expand fallback is a hypergraph initialization strategy for cases where query-side
  vertex extraction is sparse or fails.

## License

Please refer to the repository license file if one is provided.
