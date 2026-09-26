import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

if not QDRANT_URL:
    raise ValueError ("QDRANT_URL is not set")

def create_qdrant_client(url: str) -> QdrantClient:
    client = QdrantClient(url=url)
    return client

def does_collection_exist(client:QdrantClient, collection_name: str) -> bool:
    return client.collection_exists(collection_name=collection_name)

def ensure_qdrant_collection(collection_name: str, vector_size: int, client: QdrantClient):
    
    collection_exists=does_collection_exist(client = client, collection_name = collection_name)
    
    if not collection_exists:        
        client.create_collection(
            collection_name = collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
    collection_info = client.get_collection(collection_name = collection_name)
    
    return collection_info

def build_qdrant_points(
    chunk_records : list[dict],
    embeddings
) -> list[PointStruct]:
    if len(chunk_records)!=len(embeddings):
        raise ValueError(
            "Number of chunk records is not equal to the number of embeddings"
        )
        
    points = []   
        
    for chunk_record, embedding in zip(chunk_records, embeddings):
        point = PointStruct(
            id = chunk_record["chunk_id"],
            vector = embedding.tolist(),
            payload = chunk_record
        )
        
        points.append(point)
        
    return points
        
def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: list[PointStruct]
):
    return client.upsert(
        collection_name = collection_name,
        points = points
    )
        
        
        

