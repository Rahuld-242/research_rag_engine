import json
import os

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
                
    return pending_evaluations, len(pending_evaluations)

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
        
judgement_record = {
    "query_id": "q1",
    "point_id": "test-point-005",
    "rank": 4,
    "text": "This is another dummy chunk used to test checkpoint persistence.",
    "grade": 1,
    "reason": "Test judgement to verify JSON checkpoint saving."
}
        
save_checkpoint("evaluations/results/judge_checkpoint_test.json", judgement_record)

