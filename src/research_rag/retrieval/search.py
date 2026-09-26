import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint



def search_qdrant(client: QdrantClient, collection_name: str, query_vector: np.ndarray, top_k: int = 5) -> list[ScoredPoint]:
    
    search_result = client.query_points(
        collection_name = collection_name,
        query = query_vector.tolist(),
        limit = top_k,
        with_payload = True
    ).points
    
    return search_result

