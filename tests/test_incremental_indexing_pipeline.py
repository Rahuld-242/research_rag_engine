import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from research_rag.ingestion.docling_parser import converter
from research_rag.ingestion.chunking import chunker
from research_rag.ingestion.discovery import find_pdf_files
from research_rag.ingestion.ids import get_document_id
from research_rag.ingestion.persistence import load_chunk_records
from research_rag.ingestion.records import get_processed_document_ids

from research_rag.indexing.embeddings import load_embedding_model
from research_rag.indexing.vector_store import create_qdrant_client
from research_rag.indexing.indexing_pipeline import run_indexing_pipeline


load_dotenv()


TEST_COLLECTION = "research_chunks_incremental_test"

CHUNK_RECORDS_PATH = Path(
    "processed/incremental_test/chunk_records.jsonl"
).resolve()


def test_incremental_indexing_pipeline_with_sample_papers():

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    qdrant_url = os.getenv("QDRANT_URL")

    if not qdrant_url:
        raise RuntimeError(
            "QDRANT_URL is not set"
        )

    root_dir = Path(
        "raw_data/sample_papers"
    ).resolve()

    if not root_dir.exists():
        raise RuntimeError(
            f"Sample directory does not exist: {root_dir}"
        )

    # ---------------------------------------------------------
    # Discover current sample corpus
    # ---------------------------------------------------------

    pdf_files = find_pdf_files(root_dir)

    assert len(pdf_files) > 0

    current_document_ids = {
        get_document_id(
            pdf_file,
            root_dir,
        )
        for pdf_file in pdf_files
    }

    print("\n======================================")
    print("CURRENT SAMPLE CORPUS")
    print("======================================")
    print(f"PDF files found: {len(pdf_files)}")

    # ---------------------------------------------------------
    # Read persistence state before this run
    # ---------------------------------------------------------

    if CHUNK_RECORDS_PATH.exists():

        persisted_records_before, load_errors = (
            load_chunk_records(
                CHUNK_RECORDS_PATH
            )
        )

        assert load_errors == []

        processed_document_ids_before = (
            get_processed_document_ids(
                persisted_records_before
            )
        )

    else:

        persisted_records_before = []
        processed_document_ids_before = set()

        print(
            "No persisted integration-test records found. "
            "This will be the first indexing run."
        )

    new_document_ids = (
        current_document_ids
        - processed_document_ids_before
    )

    print(
        f"Already processed documents: "
        f"{len(processed_document_ids_before)}"
    )

    print(
        f"New documents to process: "
        f"{len(new_document_ids)}"
    )

    if new_document_ids:
        print("\nNew document IDs:")

        for document_id in sorted(
            new_document_ids
        ):
            print(f"  {document_id}")

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    start = perf_counter()

    model = load_embedding_model(
        "BAAI/bge-small-en-v1.5"
    )

    print(
        f"Embedding model loaded in "
        f"{perf_counter() - start:.1f}s"
    )

    # ---------------------------------------------------------
    # RUN 1
    # Process anything currently unprocessed
    # ---------------------------------------------------------

    print("\n======================================")
    print("RUN 1 - INDEX CURRENT CORPUS")
    print("======================================")

    start = perf_counter()

    run_indexing_pipeline(
        chunk_records_path=CHUNK_RECORDS_PATH,
        root_dir=root_dir,
        converter=converter,
        chunker=chunker,
        model=model,
        qdrant_url=qdrant_url,
        collection_name=TEST_COLLECTION,
        vector_size=384,
        batch_size=32,
    )

    print(
        f"Run 1 completed in "
        f"{perf_counter() - start:.1f}s"
    )

    # ---------------------------------------------------------
    # Verify persistence after run 1
    # ---------------------------------------------------------

    assert CHUNK_RECORDS_PATH.exists()

    persisted_records_after, load_errors = (
        load_chunk_records(
            CHUNK_RECORDS_PATH
        )
    )

    assert load_errors == []

    processed_document_ids_after = (
        get_processed_document_ids(
            persisted_records_after
        )
    )

    # Previously processed documents must remain present.
    assert (
        processed_document_ids_before
        <= processed_document_ids_after
    )

    # Any documents that were new at the start of this run
    # should now appear in persistence.
    assert (
        new_document_ids
        <= processed_document_ids_after
    )

    print(
        f"Persisted chunk records after run 1: "
        f"{len(persisted_records_after)}"
    )

    print(
        f"Processed documents after run 1: "
        f"{len(processed_document_ids_after)}"
    )

    # ---------------------------------------------------------
    # Verify Qdrant after run 1
    # ---------------------------------------------------------

    client = create_qdrant_client(
        url=qdrant_url
    )

    assert client.collection_exists(
        collection_name=TEST_COLLECTION
    )

    qdrant_count_after = client.count(
        collection_name=TEST_COLLECTION,
        exact=True,
    ).count

    print(
        f"Qdrant points after run 1: "
        f"{qdrant_count_after}"
    )

    # This is a dedicated collection for this integration test,
    # so persisted chunk records and Qdrant points should match.
    assert (
        qdrant_count_after
        == len(persisted_records_after)
    )

    # ---------------------------------------------------------
    # RUN 2
    # Run immediately again without changing sample_papers
    # ---------------------------------------------------------

    print("\n======================================")
    print("RUN 2 - VERIFY NO-CHANGE BEHAVIOR")
    print("======================================")

    records_before_second_run = list(
        persisted_records_after
    )

    qdrant_count_before_second_run = (
        qdrant_count_after
    )

    start = perf_counter()

    run_indexing_pipeline(
        chunk_records_path=CHUNK_RECORDS_PATH,
        root_dir=root_dir,
        converter=converter,
        chunker=chunker,
        model=model,
        qdrant_url=qdrant_url,
        collection_name=TEST_COLLECTION,
        vector_size=384,
        batch_size=32,
    )

    print(
        f"Run 2 completed in "
        f"{perf_counter() - start:.1f}s"
    )

    # ---------------------------------------------------------
    # Verify nothing changed
    # ---------------------------------------------------------

    persisted_records_second_run, load_errors = (
        load_chunk_records(
            CHUNK_RECORDS_PATH
        )
    )

    assert load_errors == []

    assert (
        persisted_records_second_run
        == records_before_second_run
    )

    qdrant_count_second_run = client.count(
        collection_name=TEST_COLLECTION,
        exact=True,
    ).count

    assert (
        qdrant_count_second_run
        == qdrant_count_before_second_run
    )

    print(
        "No duplicate chunk records were added."
    )

    print(
        "No duplicate Qdrant points were added."
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n======================================")
    print("INCREMENTAL INDEXING TEST COMPLETED")
    print("======================================")
    print(
        f"PDFs currently in sample corpus: "
        f"{len(pdf_files)}"
    )
    print(
        f"Documents already processed before run: "
        f"{len(processed_document_ids_before)}"
    )
    print(
        f"New documents processed this run: "
        f"{len(new_document_ids)}"
    )
    print(
        f"Total processed documents: "
        f"{len(processed_document_ids_after)}"
    )
    print(
        f"Persisted chunk records: "
        f"{len(persisted_records_second_run)}"
    )
    print(
        f"Qdrant points: "
        f"{qdrant_count_second_run}"
    )
    print("======================================\n")