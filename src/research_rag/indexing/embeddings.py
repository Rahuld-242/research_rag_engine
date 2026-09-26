from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

def embed_documents(texts: list[str], model: SentenceTransformer, batch_size:int = 32):
    embeddings = model.encode_document(texts, batch_size = batch_size)
    return embeddings

def embed_chunk_records(chunk_records: list[dict], model: SentenceTransformer, batch_size = 32):
    texts = [record["text"] for record in chunk_records]
    embeddings = embed_documents(texts, model, batch_size = batch_size)
    return embeddings

def embed_query(query: str, model:SentenceTransformer):
    query_embedding = model.encode_query(inputs = query)
    return query_embedding

"""
model = load_embedding_model(
        "BAAI/bge-small-en-v1.5"
    )

texts = [
        "RAG retrieves relevant documents before generation.",
        "Transformers use attention mechanisms.",
    ]

embeddings = embed_documents(texts, model)

print(type(embeddings))
print(embeddings.shape)
print(embeddings[0][:10])
"""