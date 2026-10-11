
import json
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

from evaluations.llm import create_judge_llm
from evaluations.judge_relevance import judge_relevance

# Configuration
model_name = "gpt-oss:20b"

input_path = Path("evaluations/results/dense_baseline.json")
output_path = Path(
    "evaluations/results/judge_proposals_gpt-oss-20b_q1_pilot.json"
)

# Load retrieval baseline
with input_path.open("r", encoding="utf-8") as file:
    json_file = json.load(file)

query_data = json_file["queries"][0]

# Create the model once
llm = create_judge_llm(model_name)

grading_results = []

# Evaluate the first three chunks of q1
for result in query_data["results"][:3]:
    grading_info_dict = {}

    grading_info_dict["query_id"] = query_data["query_id"]
    grading_info_dict["rank"] = result["rank"]
    grading_info_dict["point_id"] = result["point_id"]
    grading_info_dict["text"] = result["text"]

    grading_result = judge_relevance(
        query_data["query"],
        grading_info_dict["text"],
        llm
    )

    grading_info_dict["grade"] = grading_result.grade
    grading_info_dict["reason"] = grading_result.reason

    grading_results.append(grading_info_dict)

# Add experiment metadata
output_data = {
    "metadata": {
        "judge_model": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "query_id": query_data["query_id"],
        "total_judgements": len(grading_results),
        "label_type": "llm_proposal"
    },
    "results": grading_results
}

# Save proposed judgements
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)

pprint(grading_results)

print(f"\nSaved {len(grading_results)} judgements to {output_path}")
