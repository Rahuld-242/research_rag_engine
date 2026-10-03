import json
import os

from pathlib import Path

from dotenv import load_dotenv

from research_rag.indexing.embeddings import (
    load_embedding_model,
    embed_query
)

from research_rag.indexing.vector_store import create_qdrant_client
from research_rag.retrieval.search import search_qdrant

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

current_dir = Path(__file__).parent
file_path = current_dir / "retrieval_queries.json"

with open(file_path, "r") as file:
    retrieval_queries=json.load(file)
    
model = load_embedding_model("BAAI/bge-small-en-v1.5")

client = create_qdrant_client(QDRANT_URL)

collection_name = "research_chunks_formula_off_test"



all_results = []

for query in retrieval_queries:
    query_id = query["query_id"]
    query_text = query["query"]
    
    query_embedding = embed_query(query_text, model = model)
    
    search_result = search_qdrant(client = client, collection_name=collection_name, query_vector = query_embedding, top_k=20)
    
    query_results = []
    
    for rank, hit in enumerate(search_result, start=1):
        query_results.append(
            {
                "rank": rank,
                "point_id": str(hit.id),
                "score": hit.score,
                "document": hit.payload.get("file_name"),
                "text": hit.payload.get("text","")
            }
        )
    
    all_results.append(
        {
            "query_id": query_id,
            "topic": query.get("topic",""),
            "query": query_text,
            "results": query_results
        }
    )
    
results_dir = current_dir/"results"
results_dir.mkdir(exist_ok=True)

output_path = results_dir/"dense_baseline.json"

output_data = {
    "metadata": {
        "collection_name": collection_name,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "top_k": 20
    },
    "queries": all_results
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
    
print(f"Results saved to: {output_path}")
    
        
    
