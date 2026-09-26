import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from research_rag.ingestion.docling_parser import converter
from research_rag.ingestion.chunking import chunker
from research_rag.ingestion.discovery import find_pdf_files
from research_rag.ingestion.records import build_chunk_records

from research_rag.indexing.embeddings import (
    load_embedding_model,
    embed_chunk_records,
)

from research_rag.indexing.vector_store import (
    create_qdrant_client,
    ensure_qdrant_collection,
    build_qdrant_points,
    upsert_points,
)


load_dotenv()

TEST_COLLECTION = "research_chunks_sample_test"


def test_indexing_pipeline_with_sample_papers():

    qdrant_url = os.getenv("QDRANT_URL")

    if not qdrant_url:
        raise RuntimeError("QDRANT_URL is not set")

    root_dir = Path("raw_data/sample_papers").resolve()

    if not root_dir.exists():
        raise RuntimeError(
            f"Sample directory does not exist: {root_dir}"
        )

    client = create_qdrant_client(
        url=qdrant_url
    )

    # Start with a clean integration-test collection
    if client.collection_exists(
        collection_name=TEST_COLLECTION
    ):
        client.delete_collection(
            collection_name=TEST_COLLECTION
        )

    print("\nLoading embedding model...")

    model_start = perf_counter()

    model = load_embedding_model(
        "BAAI/bge-small-en-v1.5"
    )

    print(
        f"Embedding model loaded in "
        f"{perf_counter() - model_start:.1f}s"
    )

    total_start = perf_counter()

    # ---------------------------------------------------------
    # Discover sample PDFs
    # ---------------------------------------------------------

    print("\nDiscovering PDF files...")

    pdf_files = find_pdf_files(root_dir)

    print(
        f"PDF files discovered: {len(pdf_files)}"
    )

    assert len(pdf_files) > 0

    # ---------------------------------------------------------
    # Stage 1: Build chunk records
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n1/6 Building chunk records...")

    chunk_records = build_chunk_records(
        pdf_files=pdf_files,
        root_dir=root_dir,
        converter=converter,
        chunker=chunker,
    )

    print(
        f"Chunk records built: {len(chunk_records)} "
        f"in {perf_counter() - start:.1f}s"
    )

    assert len(chunk_records) > 0

    # ---------------------------------------------------------
    # Stage 2: Generate embeddings
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n2/6 Generating embeddings...")

    embeddings = embed_chunk_records(
        chunk_records=chunk_records,
        model=model,
        batch_size=32,
    )

    print(
        f"Embeddings generated: {len(embeddings)} "
        f"in {perf_counter() - start:.1f}s"
    )

    assert len(embeddings) == len(chunk_records)

    # ---------------------------------------------------------
    # Stage 3: Ensure Qdrant collection
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n3/6 Creating/verifying Qdrant collection...")

    ensure_qdrant_collection(
        client=client,
        collection_name=TEST_COLLECTION,
        vector_size=384,
    )

    print(
        f"Qdrant collection ready in "
        f"{perf_counter() - start:.1f}s"
    )

    assert client.collection_exists(
        collection_name=TEST_COLLECTION
    )

    # ---------------------------------------------------------
    # Stage 4: Build Qdrant points
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n4/6 Building Qdrant points...")

    points = build_qdrant_points(
        chunk_records=chunk_records,
        embeddings=embeddings,
    )

    print(
        f"Qdrant points built: {len(points)} "
        f"in {perf_counter() - start:.1f}s"
    )

    assert len(points) == len(chunk_records)

    # ---------------------------------------------------------
    # Stage 5: Upsert points
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n5/6 Upserting points into Qdrant...")

    upsert_points(
        client=client,
        collection_name=TEST_COLLECTION,
        points=points,
    )

    print(
        f"Qdrant upsert completed in "
        f"{perf_counter() - start:.1f}s"
    )

    # ---------------------------------------------------------
    # Stage 6: Verify current chunk IDs
    # ---------------------------------------------------------

    start = perf_counter()

    print("\n6/6 Verifying stored point IDs...")

    chunk_ids = [
        chunk_record["chunk_id"]
        for chunk_record in chunk_records
    ]

    point_records = client.retrieve(
        collection_name=TEST_COLLECTION,
        ids=chunk_ids,
        with_payload=False,
        with_vectors=False,
    )

    point_record_ids = [
        point_record.id
        for point_record in point_records
    ]

    missing_ids = (
        set(chunk_ids)
        - set(point_record_ids)
    )

    print(
        f"Verified {len(point_record_ids)} points "
        f"in {perf_counter() - start:.1f}s"
    )

    if missing_ids:
        raise RuntimeError(
            f"Qdrant upsert verification failed. "
            f"{len(missing_ids)} chunk IDs were not found "
            f"in collection '{TEST_COLLECTION}'. "
            f"Missing IDs: {list(missing_ids)[:5]}"
        )

    # ---------------------------------------------------------
    # Show actual stored points
    # ---------------------------------------------------------

    print("\nSample points stored in Qdrant:")

    stored_points, _ = client.scroll(
        collection_name=TEST_COLLECTION,
        limit=2,
        with_payload=True,
        with_vectors=True,
    )

    for index, point in enumerate(
        stored_points,
        start=1,
    ):
        print(f"\n--- Point {index} ---")
        print(f"ID: {point.id}")
        print(f"Vector: {point.vector}")
        print(f"Payload: {point.payload}")

    # ---------------------------------------------------------
    # Total duration
    # ---------------------------------------------------------

    total_time = perf_counter() - total_start

    print("\n======================================")
    print("INDEXING PIPELINE COMPLETED")
    print(f"PDF folder: {root_dir}")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Chunks indexed: {len(chunk_records)}")
    print(f"Points verified: {len(point_record_ids)}")
    print(f"Total pipeline time: {total_time:.1f}s")
    print("======================================\n")

    assert not missing_ids