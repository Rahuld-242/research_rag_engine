# Retrieval Baseline Analysis

## Experiment 1 — Naive Dense Retrieval

### Objective

Establish a baseline for semantic retrieval using the existing dense embedding
and Qdrant vector-search pipeline before introducing more advanced retrieval
techniques.

### Configuration

- Embedding model: `BAAI/bge-small-en-v1.5`
- Embedding dimension: 384
- Vector database: Qdrant
- Similarity metric: Cosine
- Collection: `research_chunks_formula_off_test`
- Initial `top_k`: 5
- Extended inspection: 20 results

### Query

> Why can proposition-level retrieval outperform passage-level retrieval in dense retrieval?

### Initial Top-5 Observation

All five retrieved chunks came from:

`Dense Retrieval What Retrieval Granularity Should We Use.pdf`

This indicates that the dense retriever successfully identified:

- the correct document
- the correct topic
- the relevant terminology around proposition-level and passage-level retrieval

However, the top five chunks primarily described **that** proposition-level
retrieval performs better rather than directly explaining **why** it performs
better.

Therefore:

**Topical relevance was high, but answer relevance was weaker.**

### Extended Top-20 Analysis

Increasing `top_k` to 20 exposed more directly answer-bearing chunks.

#### Rank 6 — Highly Relevant

Score: `0.87546885`

The chunk states that propositions provide a higher density of query-relevant
information than sentences or passages, and that finer-grained retrieval makes
the correct answer more likely to occur within the retrieved content.

This is the strongest direct answer to the query.

#### Rank 9 — Highly Relevant

The chunk discusses the intuition behind different retrieval granularities and
contrasts coarse passage-level retrieval with finer-grained retrieval.

#### Rank 16 — Highly Relevant

The conclusion connects finer-grained indexing with improved cross-task
generalization and increased density of relevant information.

#### Rank 8 — Supporting / Caveat

This chunk discusses why the improvement from proposition-level retrieval may
be smaller for supervised retrievers trained using query-passage pairs.

It is relevant to the broader question but is less directly answer-bearing than
ranks 6, 9, and 16.

### Baseline Diagnosis

The baseline dense retriever is able to locate the correct document and
semantically relevant sections.

The primary weakness observed in this experiment is **ranking quality rather
than complete retrieval failure**.

The most directly answer-bearing chunk appeared at rank 6, just outside the
initial top-5 retrieval window.

This demonstrates an important distinction:

> Semantic similarity does not necessarily imply answer relevance.

### Preliminary Relevance Judgement

| Rank | Relevance |
|------|-----------|
| 6 | Highly relevant |
| 9 | Highly relevant |
| 16 | Highly relevant |
| 8 | Relevant / supporting |
| 1–5 | Topically relevant but less answer-focused |

These manual relevance judgements can later be used when implementing
retrieval metrics such as NDCG.

## Next Experiments

Future retrieval approaches will be compared against this dense baseline.

Planned comparisons include:

- reranking
- multi-query retrieval / query expansion
- sparse retrieval
- hybrid dense + sparse retrieval
- HNSW search tuning and exact-search comparison

A useful objective for subsequent experiments is to determine whether the
answer-bearing chunks currently appearing at ranks 6, 9, and 16 can be moved
into the top 3 results without unacceptable increases in retrieval latency.