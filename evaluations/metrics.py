import json
import math

def load_grades_by_query(file_path: str) -> dict:
    
    with open(file_path, "r", encoding="utf-8") as file:
        batch_file = json.load(file)
        
    grades_by_query = {}
    
    results = batch_file["results"]
    
    if not results:
        raise ValueError("No relevance judgements found")
    
    for result in results:
        if result["query_id"] not in grades_by_query:
            grades_by_query[result["query_id"]] = [(result["rank"], result["grade"])]
        else:
            grades_by_query[result["query_id"]].append((result["rank"], result["grade"]))
    
    for query_id in grades_by_query:
        pairs = grades_by_query[query_id]
        ranks = [rank for rank, grade in pairs]

        # Check for invalid, missing or duplicate ranks
        if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(1, 21)):
            raise ValueError(f"Invalid or incomplete ranks for {query_id}")

        # Check for invalid relevance grades
        if any(type(grade) is not int or grade not in (0, 1, 2) for rank, grade in pairs):
            raise ValueError(f"Invalid relevance grade for {query_id}")
        grades_by_query[query_id].sort()
        grades_by_query[query_id]=[grade for rank, grade in grades_by_query[query_id]]
        
    return grades_by_query

def precision_at_k(
    grades: list[int], k: int, relevant_threshold: int = 1
) -> float:
    if k<=0 or k>len(grades):
        raise ValueError("invalid k value")
    count = 0
    k_grades = grades[:k]
    for grade in k_grades:
        if grade>=relevant_threshold:
            count+=1
    return count / k

def reciprocal_rank(
    grades: list[int], relevant_threshold: int = 2
) -> float:
    for idx, grade in enumerate(grades):
        if grade==relevant_threshold:
            return 1/(idx+1)
    return 0.0

def mean_reciprocal_rank(
    grades_by_query: dict[str, list[int]],
    relevant_threshold: int = 2
) -> float:
    rr_sum = 0.0
    if not grades_by_query:
        return 0.0
    for grades in grades_by_query.values():
        rr_sum += reciprocal_rank(grades, relevant_threshold)
    return rr_sum/len(grades_by_query)
    
def mean_precision_at_k(
    grades_by_query: dict[str, list[int]],
    k: int,
    relevant_threshold: int = 1
) -> float:
    prec_sum = 0.0
    if not grades_by_query:
        return 0.0
    for grades in grades_by_query.values():
        prec_sum += precision_at_k(grades, k, relevant_threshold)
    return prec_sum/len(grades_by_query)

def dcg_at_k(grades: list[int], k: int) -> float:
    if k<=0 or k>len(grades):
        raise ValueError("Invalid k value")
    dcg_k = 0.0
    k_grades = grades[:k]
    for idx, grade in enumerate(k_grades, start=1):
        dcg_k += (2**grade-1)/math.log2(idx+1)
    return dcg_k

def idcg_at_k(grades: list[int], k: int) -> float:
    sorted_grades = sorted(grades, reverse=True)
    return dcg_at_k(sorted_grades, k)

def ndcg_at_k(grades: list[int], k: int) -> float:
    dcg_k = dcg_at_k(grades, k)
    idcg_k = idcg_at_k(grades, k)
    if idcg_k == 0.0:
        return 0.0
    return dcg_k/idcg_k
        
def mean_ndcg_at_k(grades_by_query: dict[str, list[int]], k: int) -> float:
    ndcg_sum = 0.0
    if not grades_by_query:
        return 0.0
    for grades in grades_by_query.values():
        ndcg_sum += ndcg_at_k(grades, k)
    return ndcg_sum/len(grades_by_query)
        

if __name__ == "__main__":
    grades_by_query = load_grades_by_query(
        "evaluations/results/judge_proposals_gpt-oss-20b_v2_batch.json"
    )
    
    expected_queries = {"q1", "q2", "q3", "q4", "q5"}

    if set(grades_by_query) != expected_queries:
        raise ValueError("Missing or unexpected queries")

    print("Data integrity: PASSED (5 queries, 100 judgements)")

    print(f"Mean P@5: {mean_precision_at_k(grades_by_query, 5):.4f}")
    print(f"Mean P@10: {mean_precision_at_k(grades_by_query, 10):.4f}")
    print(f"Mean strict P@5: {mean_precision_at_k(grades_by_query, 5, 2):.4f}")
    print(f"MRR@20: {mean_reciprocal_rank(grades_by_query):.4f}")
    print(f"Mean NDCG@5: {mean_ndcg_at_k(grades_by_query, 5):.4f}")
    print(f"Mean NDCG@10: {mean_ndcg_at_k(grades_by_query, 10):.4f}")

    
        
    
    
    
