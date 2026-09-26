import os

from dotenv import load_dotenv
from research_rag.indexing.embeddings import load_embedding_model, embed_query
from research_rag.indexing.vector_store import create_qdrant_client
from research_rag.retrieval.search import search_qdrant

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

model = load_embedding_model("BAAI/bge-small-en-v1.5")

client = create_qdrant_client(QDRANT_URL)

query = "Why can proposition-level retrieval outperform passage-level retrieval in dense retrieval?"

query_embedding = embed_query(query, model)

collection_name = "research_chunks_formula_off_test"

top_k_chunks = search_qdrant(client, collection_name, query_embedding, top_k=20)

print("Query: ", query)
print(f"\nRetrieved {len(top_k_chunks)} chunks")

for rank, hit in enumerate(top_k_chunks, start=1):
    
    print(f"\nRank: {rank}")
    print(f"Score: {hit.score}")
    print(f"Document: {hit.payload.get('file_name')}")
    print(f"Text: {hit.payload.get('text', '')[:500]}")