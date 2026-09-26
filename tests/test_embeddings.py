import numpy as np
from unittest.mock import MagicMock, patch

from research_rag.indexing.embeddings import (
    load_embedding_model,
    embed_documents,
    embed_chunk_records,
)


def test_load_embedding_model():
    with patch(
        "research_rag.indexing.embeddings.SentenceTransformer"
    ) as mock_sentence_transformer:

        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        result = load_embedding_model("BAAI/bge-small-en-v1.5")

        mock_sentence_transformer.assert_called_once_with(
            "BAAI/bge-small-en-v1.5"
        )

        assert result is mock_model


def test_embed_documents():
    model = MagicMock()

    texts = [
        "RAG retrieves relevant documents.",
        "Transformers use attention mechanisms.",
    ]

    expected_embeddings = np.zeros((2, 384))

    model.encode_document.return_value = expected_embeddings

    result = embed_documents(
        texts,
        model,
        batch_size=32,
    )

    model.encode_document.assert_called_once_with(
        texts,
        batch_size=32,
    )

    assert result.shape == (2, 384)
    assert np.array_equal(result, expected_embeddings)


def test_embed_chunk_records():
    chunk_records = [
        {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "text": "RAG retrieves relevant documents.",
            "primary_topic": "RAG",
        },
        {
            "chunk_id": "chunk_2",
            "document_id": "doc_1",
            "text": "Embeddings represent text as vectors.",
            "primary_topic": "RAG",
        },
    ]

    expected_embeddings = np.zeros((2, 384))

    model = MagicMock()

    with patch(
        "research_rag.indexing.embeddings.embed_documents",
        return_value=expected_embeddings,
    ) as mock_embed_documents:

        result = embed_chunk_records(
            chunk_records,
            model,
            batch_size=32,
        )

        expected_texts = [
            "RAG retrieves relevant documents.",
            "Embeddings represent text as vectors.",
        ]

        mock_embed_documents.assert_called_once_with(
            expected_texts,
            model,
            batch_size=32,
        )

        assert result.shape == (2, 384)
        assert np.array_equal(result, expected_embeddings)