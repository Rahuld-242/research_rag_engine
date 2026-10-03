# Retrieval Evaluation Notes

## Purpose

This file records the manual analysis used to build and validate the retrieval evaluation set for RAG_V2.

The goal is to separate two questions:

1. **Pipeline correctness** — does query embedding -> Qdrant search -> top-k retrieval work?
2. **Retrieval quality** — are the most useful answer-bearing chunks ranked highly enough?

The integration test covers the first question. This evaluation set is for the second.

## Baseline Setup

- Embedding model: `BAAI/bge-small-en-v1.5`
- Vector database: Qdrant
- Baseline retrieval: dense semantic search
- Manual inspection depth: `top_k = 20`
- Current baseline collection: `research_chunks_formula_off_test`
- Evaluation query count: `5`
- Generated retrieval results: `evaluations/results/dense_baseline.json`
- Human-readable inspection view: `evaluations/results/dense_baseline_dashboard.html`

Generated results are treated as run outputs and should not be edited manually. Relevance judgments will be stored separately so they can be reused across retrieval methods.

## Relevance Scale

Use graded relevance:

- `2` — directly answers the query or contains essential answer-bearing evidence
- `1` — relevant/supporting context, but not sufficient as the main answer
- `0` — not useful for answering the query

Where possible, store relevance judgments against a **stable chunk/point ID**, not a rank number. Rank can change when the retrieval method changes.

## Evaluation Queries

### Q1 — Dense Retrieval Granularity

> Why can proposition-level retrieval outperform passage-level retrieval in dense retrieval?

Baseline observation:

- Top results identify the correct paper and topic.
- The strongest answer-bearing chunk appears at rank 6.
- Ranks 9, 11, and 16 are also highly relevant.
- Main observed failure mode: **ranking quality**, not inability to locate the relevant evidence.

### Q2 — Direct Preference Optimization

> How does Direct Preference Optimization simplify preference learning compared with PPO-based RLHF?

Baseline observation:

- Strong answer-bearing chunks appear at the top of the ranking.
- Ranks 1, 2, and 3 directly explain how DPO avoids an explicit reward-model + RL loop.
- This query acts as a useful positive-control case for the dense retriever.

### Q3 — ReAct

> How does ReAct combine reasoning and acting to reduce hallucination and improve task solving?

Baseline observation:

- The correct paper dominates the retrieved results.
- The answer is distributed across multiple chunks rather than concentrated in a single top-ranked passage.
- Useful evidence appears in chunks describing interleaved reasoning, actions, observations, grounding, and reduced hallucination.
- This query is a useful **multi-evidence/context challenge**.

### Q4 — HELM Evaluation

> Why does HELM use multiple evaluation metrics instead of focusing only on accuracy?

Baseline observation:

- Rank 1 directly answers the question.
- Additional highly relevant chunks explain trade-offs across multiple desiderata and the motivation for a multi-metric view.
- This is another strong positive-control case.

### Q5 — Advanced RAG vs Naive RAG

> How does Advanced RAG address the retrieval limitations of Naive RAG?

Baseline observation:

- Rank 1 directly answers the question.
- Supporting chunks describe Naive RAG retrieval limitations, query optimization, and multi-query strategies.
- This is a strong positive-control case.

## Current Evaluation Workflow

```text
retrieval_queries.json
        ↓
evaluate_retrieval.py
        ↓
dense_baseline.json
        ↓
render_retrieval_results.py
        ↓
dense_baseline_dashboard.html
```

The dashboard is for readable inspection of retrieval results. Metric values will be added after relevance judgments are created.

## Relevance Judgment Storage

Planned structure:

```text
evaluations/
├── retrieval_queries.json
├── retrieval_evaluation_notes.md
├── evaluate_retrieval.py
├── render_retrieval_results.py
├── judgments/
│   └── relevance_judgments.json
└── results/
    ├── dense_baseline.json
    └── dense_baseline_dashboard.html
```

Judgments should be keyed by stable identifiers such as `(query_id, point_id)` so they remain valid even if ranks change in later experiments.

## Planned Metrics

After relevance judgments are available, implement:

- Precision@5
- Precision@10
- MRR
- NDCG@5
- NDCG@10

For binary metrics such as Precision@k and MRR, define a clear relevance threshold, likely `relevance > 0`.

NDCG will use the graded `0/1/2` relevance labels directly.

### Recall@k

True Recall@k is deferred for now.

The current labels are based on retrieved candidate sets rather than a complete corpus-level set of all relevant chunks. Reporting Recall@k at this stage could therefore be misleading.

Later, when multiple retrieval methods are compared, pooled candidate results can be judged to create a more complete relevance set.

## Planned Retrieval Comparisons

- Dense baseline
- Exact vector search vs HNSW
- Reranking
- Multi-query retrieval / query expansion
- Sparse retrieval
- Hybrid dense + sparse retrieval
- Parent / neighboring context expansion

Later retrieval methods should be compared against the same evaluation set rather than judged only by isolated examples.

The purpose is to measure whether more advanced methods improve ranking and answer relevance without introducing unacceptable latency or complexity.

## Current Checkpoint

The dense retrieval baseline is functioning end-to-end and has a repeatable evaluation workflow.

The five-query evaluation set has been finalized and the baseline results can be inspected through a browser-based dashboard.

The next task is to create `relevance_judgments.json`, label retrieved chunks using the `0/1/2` scale, and implement the first retrieval-quality metrics before modifying the retrieval method.
