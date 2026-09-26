from pathlib import Path

from research_rag.ingestion.records import (build_chunk_records, get_processed_document_ids)
from research_rag.indexing.embeddings import embed_chunk_records
from research_rag.indexing.vector_store import (
    create_qdrant_client,
    does_collection_exist,
    ensure_qdrant_collection,
    build_qdrant_points,
    upsert_points
)
from research_rag.ingestion.persistence import (load_chunk_records, save_chunk_records)
from research_rag.ingestion.discovery import find_unprocessed_pdf_files

def run_indexing_pipeline(
    chunk_records_path: Path,
    root_dir: Path,
    converter,
    chunker,
    model,
    qdrant_url: str,
    collection_name: str,
    vector_size: int,
    batch_size: int
):
        
    chunk_records_path = Path(chunk_records_path).resolve()
    
    if chunk_records_path.exists():
        persisted_chunk_records, load_errors = load_chunk_records(chunk_records_path)
    else:
        persisted_chunk_records=[]
        load_errors=[]
        print("No persisted chunk records found. Starting first indexing run...")
    
    if load_errors:
        print(f"Found {len(load_errors)} malformed lines in persisted chunk records")
        
        for load_error in load_errors:
            print(
                f"Line {load_error['line_number']}: "
                f"{load_error['error_message']}"
            )
    
    processed_document_ids = get_processed_document_ids(persisted_chunk_records)
    
    unprocessed_pdf_files = find_unprocessed_pdf_files(root_dir, processed_document_ids)
    
    if not unprocessed_pdf_files:
        print("No new PDF files to process")
        return
    
    client = create_qdrant_client(url = qdrant_url)
    
    chunk_records = build_chunk_records(unprocessed_pdf_files, root_dir = root_dir,converter = converter, chunker = chunker)
    
    embeddings = embed_chunk_records(chunk_records = chunk_records, model= model, batch_size = batch_size)
    
    ensure_qdrant_collection(collection_name = collection_name, vector_size = vector_size, client = client)
    
    collection_exists = does_collection_exist(client = client, collection_name = collection_name)
    
    if not collection_exists:
        raise RuntimeError(
            f"Qdrant collection {collection_name} was not created successfully"
        )
        
    points = build_qdrant_points(chunk_records = chunk_records, embeddings = embeddings)
    
    upsert_points(client = client, collection_name = collection_name, points = points)
    
    chunk_ids = [chunk_record["chunk_id"] for chunk_record in chunk_records]
    
    point_records = client.retrieve(collection_name = collection_name, ids = chunk_ids, with_payload = False, with_vectors = False)
    
    point_record_ids = [point_record.id for point_record in point_records]
    
    missing_ids = set(chunk_ids) - set(point_record_ids)
    
    if missing_ids:
        raise RuntimeError(
             f"Qdrant upsert verification failed. "
            f"{len(missing_ids)} chunk IDs were not found in collection "
            f"'{collection_name}'. "
            f"Missing IDs: {list(missing_ids)[:5]}"
        )
        
    save_chunk_records(chunk_records, chunk_records_path, mode="append")