
import json
from pathlib import Path

from evaluations.run_judge_batch import find_pending_evaluations
from sentence_transformers import CrossEncoder


EVALUATION_DIR = Path(__file__).parent


def rerank_candidates(candidate_pairs, model):
    # Prepare the query and chunk text for the cross-encoder.
    cross_encoder_pairs = [
        (query, result["text"])
        for query_id, query, result in candidate_pairs
    ]

    if not cross_encoder_pairs:
        return {}

    scores = model.predict(cross_encoder_pairs)

    if len(scores) != len(candidate_pairs):
        raise ValueError("Cross-encoder score count mismatch")

    # Group scored results under their query IDs.
    reranked_by_query = {}
    for (query_id, query, result), score in zip(candidate_pairs, scores):
        if query_id not in reranked_by_query:
            reranked_by_query[query_id] = [(result, float(score))]
        else:
            reranked_by_query[query_id].append((result, float(score)))

    # Rank the results for each query from highest to lowest score.
    for results in reranked_by_query.values():
        results.sort(key=lambda x: x[1], reverse=True)

    # Preserve the original 'rank' and add the new ranking information.
    final_results_by_query = {}
    for query_id in reranked_by_query:
        reranked_results = []
        for new_rank, (result, score) in enumerate(
            reranked_by_query[query_id], start=1
        ):
            result["reranked_rank"] = new_rank
            result["reranker_score"] = score
            reranked_results.append(result)

        final_results_by_query[query_id] = reranked_results

    return final_results_by_query


def main():
    baseline_path = EVALUATION_DIR / "results" / "dense_baseline.json"
    output_path = EVALUATION_DIR / "results" / "cross_encoder_reranked.json"

    candidate_pairs = find_pending_evaluations(
        baseline_path, completed_pairs=set()
    )
    print(f"Total candidate pairs: {len(candidate_pairs)}")

    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    model = CrossEncoder(model_name)

    final_results_by_query = rerank_candidates(candidate_pairs, model)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(final_results_by_query, file, indent=2)

    for query_id, results in final_results_by_query.items():
        print(query_id, len(results))

    print(f"Reranked results saved to: {output_path}")


if __name__ == "__main__":
    main()
