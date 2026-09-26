from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from research_rag.indexing.vector_store import (
    build_qdrant_points,
    create_qdrant_client,
    does_collection_exist,
    ensure_qdrant_collection,
    upsert_points,
)


def test_create_qdrant_client(monkeypatch):
    mock_client_class = MagicMock()
    mock_client_instance = MagicMock()

    mock_client_class.return_value = mock_client_instance

    monkeypatch.setattr(
        "research_rag.indexing.vector_store.QdrantClient",
        mock_client_class,
    )

    result = create_qdrant_client(
        url="http://localhost:6333"
    )

    mock_client_class.assert_called_once_with(
        url="http://localhost:6333"
    )

    assert result is mock_client_instance


def test_does_collection_exist():
    client = MagicMock()

    client.collection_exists.return_value = True

    result = does_collection_exist(
        client=client,
        collection_name="research_chunks",
    )

    assert result is True

    client.collection_exists.assert_called_once_with(
        collection_name="research_chunks"
    )


def test_ensure_qdrant_collection_creates_collection_if_missing(
    monkeypatch,
):
    client = MagicMock()

    mock_does_collection_exist = MagicMock(
        return_value=False
    )

    monkeypatch.setattr(
        "research_rag.indexing.vector_store.does_collection_exist",
        mock_does_collection_exist,
    )

    mock_collection_info = MagicMock()

    client.get_collection.return_value = mock_collection_info

    result = ensure_qdrant_collection(
        collection_name="research_chunks",
        vector_size=384,
        client=client,
    )

    mock_does_collection_exist.assert_called_once_with(
        client=client,
        collection_name="research_chunks",
    )

    client.create_collection.assert_called_once_with(
        collection_name="research_chunks",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    client.get_collection.assert_called_once_with(
        collection_name="research_chunks"
    )

    assert result is mock_collection_info


def test_ensure_qdrant_collection_does_not_create_if_exists(
    monkeypatch,
):
    client = MagicMock()

    mock_does_collection_exist = MagicMock(
        return_value=True
    )

    monkeypatch.setattr(
        "research_rag.indexing.vector_store.does_collection_exist",
        mock_does_collection_exist,
    )

    mock_collection_info = MagicMock()

    client.get_collection.return_value = mock_collection_info

    result = ensure_qdrant_collection(
        collection_name="research_chunks",
        vector_size=384,
        client=client,
    )

    mock_does_collection_exist.assert_called_once_with(
        client=client,
        collection_name="research_chunks",
    )

    client.create_collection.assert_not_called()

    client.get_collection.assert_called_once_with(
        collection_name="research_chunks"
    )

    assert result is mock_collection_info


def test_build_qdrant_points():
    chunk_records = [
        {
            "chunk_id": "4e68efec-31c3-5be6-a98c-50cd8f85fdeb",
            "document_id": "paper_1.pdf",
            "chunk_index": 0,
            "text": "First chunk text",
        },
        {
            "chunk_id": "86d7f2a8-1b4f-5b65-8e08-c10a73e5bc37",
            "document_id": "paper_1.pdf",
            "chunk_index": 1,
            "text": "Second chunk text",
        },
    ]

    embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    result = build_qdrant_points(
        chunk_records=chunk_records,
        embeddings=embeddings,
    )

    assert len(result) == 2

    assert isinstance(result[0], PointStruct)
    assert isinstance(result[1], PointStruct)

    assert (
        result[0].id
        == "4e68efec-31c3-5be6-a98c-50cd8f85fdeb"
    )

    assert (
        result[1].id
        == "86d7f2a8-1b4f-5b65-8e08-c10a73e5bc37"
    )

    assert result[0].vector == pytest.approx(
        [0.1, 0.2, 0.3]
    )

    assert result[1].vector == pytest.approx(
        [0.4, 0.5, 0.6]
    )

    assert result[0].payload == chunk_records[0]
    assert result[1].payload == chunk_records[1]


def test_build_qdrant_points_mismatched_lengths():
    chunk_records = [
        {
            "chunk_id": "4e68efec-31c3-5be6-a98c-50cd8f85fdeb",
            "document_id": "paper_1.pdf",
            "chunk_index": 0,
            "text": "First chunk text",
        }
    ]

    embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Number of chunk records is not equal "
            "to the number of embeddings"
        ),
    ):
        build_qdrant_points(
            chunk_records=chunk_records,
            embeddings=embeddings,
        )


def test_upsert_points():
    client = MagicMock()

    points = [
        PointStruct(
            id="4e68efec-31c3-5be6-a98c-50cd8f85fdeb",
            vector=[0.1, 0.2, 0.3],
            payload={
                "chunk_id": (
                    "4e68efec-31c3-5be6-a98c-50cd8f85fdeb"
                ),
                "text": "First chunk text",
            },
        )
    ]

    mock_upsert_result = MagicMock()

    client.upsert.return_value = mock_upsert_result

    result = upsert_points(
        client=client,
        collection_name="research_chunks",
        points=points,
    )

    client.upsert.assert_called_once_with(
        collection_name="research_chunks",
        points=points,
    )

    assert result is mock_upsert_result