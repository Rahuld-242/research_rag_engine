import os

import numpy as np
from dotenv import load_dotenv

from research_rag.indexing.vector_store import (
    create_qdrant_client,
    ensure_qdrant_collection,
    build_qdrant_points,
    upsert_points,
)


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

TEST_COLLECTION = "research_chunks_integration_test"


def test_qdrant_upsert_integration():
    client = create_qdrant_client(
        url=QDRANT_URL
    )

    # Start clean in case an earlier test run left the collection behind
    if client.collection_exists(
        collection_name=TEST_COLLECTION
    ):
        client.delete_collection(
            collection_name=TEST_COLLECTION
        )

    try:
        ensure_qdrant_collection(
            client=client,
            collection_name=TEST_COLLECTION,
            vector_size=384,
        )

        chunk_records = [
            {
                "chunk_id": (
                    "4e68efec-31c3-5be6-a98c-50cd8f85fdeb"
                ),
                "document_id": "paper_1.pdf",
                "chunk_index": 0,
                "text": "Dense retrieval uses vector embeddings.",
            },
            {
                "chunk_id": (
                    "86d7f2a8-1b4f-5b65-8e08-c10a73e5bc37"
                ),
                "document_id": "paper_1.pdf",
                "chunk_index": 1,
                "text": "Qdrant stores vectors and payloads.",
            },
        ]

        embeddings = np.zeros(
            (2, 384),
            dtype=np.float32,
        )

        embeddings[0][0] = 1.0
        embeddings[1][1] = 1.0

        points = build_qdrant_points(
            chunk_records=chunk_records,
            embeddings=embeddings,
        )

        upsert_points(
            client=client,
            collection_name=TEST_COLLECTION,
            points=points,
        )

        count_result = client.count(
            collection_name=TEST_COLLECTION,
            exact=True,
        )

        assert count_result.count == 2

    finally:
        if client.collection_exists(
            collection_name=TEST_COLLECTION
        ):
            client.delete_collection(
                collection_name=TEST_COLLECTION
            )
            
def test_qdrant_upsert_same_id_does_not_duplicate():
    client = create_qdrant_client(
        url=QDRANT_URL
    )

    if client.collection_exists(
        collection_name=TEST_COLLECTION
    ):
        client.delete_collection(
            collection_name=TEST_COLLECTION
        )

    try:
        ensure_qdrant_collection(
            client=client,
            collection_name=TEST_COLLECTION,
            vector_size=384,
        )

        chunk_records = [
            {
                "chunk_id": (
                    "4e68efec-31c3-5be6-a98c-50cd8f85fdeb"
                ),
                "document_id": "paper_1.pdf",
                "chunk_index": 0,
                "text": "Original chunk text",
            }
        ]

        embeddings = np.zeros(
            (1, 384),
            dtype=np.float32,
        )

        embeddings[0][0] = 1.0

        points = build_qdrant_points(
            chunk_records=chunk_records,
            embeddings=embeddings,
        )

        upsert_points(
            client=client,
            collection_name=TEST_COLLECTION,
            points=points,
        )

        first_count = client.count(
            collection_name=TEST_COLLECTION,
            exact=True,
        )

        assert first_count.count == 1

        # Same chunk_id, but changed payload and vector
        updated_chunk_records = [
            {
                "chunk_id": (
                    "4e68efec-31c3-5be6-a98c-50cd8f85fdeb"
                ),
                "document_id": "paper_1.pdf",
                "chunk_index": 0,
                "text": "Updated chunk text",
            }
        ]

        updated_embeddings = np.zeros(
            (1, 384),
            dtype=np.float32,
        )

        updated_embeddings[0][1] = 1.0

        updated_points = build_qdrant_points(
            chunk_records=updated_chunk_records,
            embeddings=updated_embeddings,
        )

        upsert_points(
            client=client,
            collection_name=TEST_COLLECTION,
            points=updated_points,
        )

        second_count = client.count(
            collection_name=TEST_COLLECTION,
            exact=True,
        )

        assert second_count.count == 1

    finally:
        if client.collection_exists(
            collection_name=TEST_COLLECTION
        ):
            client.delete_collection(
                collection_name=TEST_COLLECTION
            )