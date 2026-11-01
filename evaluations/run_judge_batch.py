import json
import os

from evaluations.judge_relevance import judge_relevance
from evaluations.llm import create_judge_llm

def load_completed_pairs(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    pairs = set()
    for result in data["results"]:
        pair = (result["query_id"], result["point_id"])
        pairs.add(pair)
        
    return pairs

def find_pending_evaluations(baseline_file_path: str, completed_pairs):
    with open(baseline_file_path, "r", encoding="utf-8") as file2:
        data2 = json.load(file2)
        
    queries = data2["queries"]
    
    pending_evaluations = []
    
    for query in queries:
        for result in query["results"]:
            pair=(query["query_id"], result["point_id"])
            if pair not in completed_pairs:
                pending_item= (
                    query["query_id"],
                    query["query"],
                    result
                )
                pending_evaluations.append(pending_item)
                
    return pending_evaluations

def save_checkpoint(file_path, judgement_record):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
    for result in data["results"]:
            if judgement_record["query_id"]==result["query_id"] and judgement_record["point_id"]==result["point_id"]:
                raise ValueError("Duplicate judgement detected")
        
    data["results"].append(judgement_record)
    data["metadata"]["total_judgements"] = len(data["results"])
    
    temp_path = str(file_path)+".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    
    os.replace(temp_path, file_path)
    
def run_batch(pending_evaluations, llm, checkpoint_path):
    for query_id, query_text, result in pending_evaluations:
        grading = judge_relevance(query_text, result["text"], llm)
        judgement_record = {
            "query_id": query_id,
            "point_id": result["point_id"],
            "rank": result["rank"],
            "text": result["text"],
            "grade": grading.grade,
            "reason": grading.reason
        }
        save_checkpoint(checkpoint_path, judgement_record)
        
        
if __name__ == "__main__":
    checkpoint_path = "evaluations\\results\\judge_proposals_gpt-oss-20b_v2_pilot.json"
    baseline_path = "evaluations\\results\\dense_baseline.json"
    
    completed_pairs = load_completed_pairs(checkpoint_path)
    
    pending_evaluations = find_pending_evaluations(baseline_path, completed_pairs)
        
    print(f"Pending evaluations: {len(pending_evaluations)}")
    
    if pending_evaluations:
        print(f"Next query: {pending_evaluations[0][0]}")
        print(f"Next rank: {pending_evaluations[0][2]['rank']}")
        
        llm = create_judge_llm("gpt-oss:20b")
        run_batch(pending_evaluations, llm, checkpoint_path)
    
    else:
        print("All evaluations are complete")
    
    

