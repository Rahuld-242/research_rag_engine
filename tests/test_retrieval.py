import os

from dotenv import load_dotenv

from research_rag.indexing.embeddings import (
    load_embedding_model,
    embed_query,
)
from research_rag.indexing.vector_store import create_qdrant_client
from research_rag.retrieval.search import search_qdrant


def test_dense_retrieval():
    load_dotenv()

    qdrant_url = os.getenv("QDRANT_URL")
    assert qdrant_url is not None

    model = load_embedding_model("BAAI/bge-small-en-v1.5")
    client = create_qdrant_client(qdrant_url)

    query = (
        "Why can proposition-level retrieval outperform "
        "passage-level retrieval in dense retrieval?"
    )

    query_embedding = embed_query(query, model)

    results = search_qdrant(
        client=client,
        collection_name="research_chunks_formula_off_test",
        query_vector=query_embedding,
        top_k=5,
    )

    assert query_embedding.shape == (384,)
    assert len(results) == 5

    for hit in results:
        assert hit.score is not None
        assert hit.payload is not None
        assert "text" in hit.payload
        assert hit.payload["text"]