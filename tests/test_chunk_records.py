from unittest.mock import MagicMock
import uuid

from research_rag.ingestion.records import (
    build_chunk_record,
    build_chunk_records,
)


def test_build_chunk_record(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    root_dir = tmp_path

    base_record = {
        "document_id": "05_RAG_and_Retrieval/paper.pdf",
        "document_hash": "ABC123",
        "file_name": "paper.pdf",
        "file_path": str(pdf_path.resolve()),
        "primary_topic": "RAG and Retrieval",
        "subtopic": None,
        "topic_path": "RAG and Retrieval",
    }

    mock_build_document_record = MagicMock(
        return_value=base_record.copy()
    )

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_document_record",
        mock_build_document_record,
    )

    chunk = MagicMock()
    chunker = MagicMock()

    chunker.contextualize.return_value = (
        "1 Introduction\nDense retrievers are a popular class..."
    )

    result = build_chunk_record(
        pdf_path,
        root_dir,
        chunk,
        3,
        chunker,
    )

    assert result["document_id"] == "05_RAG_and_Retrieval/paper.pdf"
    assert result["document_hash"] == "ABC123"
    assert result["file_name"] == "paper.pdf"
    assert result["primary_topic"] == "RAG and Retrieval"

    assert result["chunk_index"] == 3

    assert "chunk_id" in result

    parsed_uuid = uuid.UUID(result["chunk_id"])
    assert str(parsed_uuid) == result["chunk_id"]

    assert result["text"] == (
        "1 Introduction\nDense retrievers are a popular class..."
    )

    mock_build_document_record.assert_called_once_with(
        pdf_path.resolve(),
        root_dir.resolve(),
    )

    chunker.contextualize.assert_called_once_with(chunk)


def test_build_chunk_record_same_input_same_chunk_id(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "paper.pdf"
    root_dir = tmp_path

    base_record = {
        "document_id": "05_RAG_and_Retrieval/paper.pdf",
        "document_hash": "ABC123",
        "file_name": "paper.pdf",
        "file_path": str(pdf_path.resolve()),
        "primary_topic": "RAG and Retrieval",
        "subtopic": None,
        "topic_path": "RAG and Retrieval",
    }

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_document_record",
        MagicMock(
            side_effect=lambda *args, **kwargs: base_record.copy()
        ),
    )

    chunk = MagicMock()
    chunker = MagicMock()
    chunker.contextualize.return_value = "sample text"

    result_1 = build_chunk_record(
        pdf_path,
        root_dir,
        chunk,
        3,
        chunker,
    )

    result_2 = build_chunk_record(
        pdf_path,
        root_dir,
        chunk,
        3,
        chunker,
    )

    assert result_1["chunk_id"] == result_2["chunk_id"]


def test_build_chunk_record_different_index_different_chunk_id(
    monkeypatch,
    tmp_path,
):
    pdf_path = tmp_path / "paper.pdf"
    root_dir = tmp_path

    base_record = {
        "document_id": "05_RAG_and_Retrieval/paper.pdf",
        "document_hash": "ABC123",
        "file_name": "paper.pdf",
        "file_path": str(pdf_path.resolve()),
        "primary_topic": "RAG and Retrieval",
        "subtopic": None,
        "topic_path": "RAG and Retrieval",
    }

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_document_record",
        MagicMock(
            side_effect=lambda *args, **kwargs: base_record.copy()
        ),
    )

    chunk = MagicMock()
    chunker = MagicMock()
    chunker.contextualize.return_value = "sample text"

    result_1 = build_chunk_record(
        pdf_path,
        root_dir,
        chunk,
        3,
        chunker,
    )

    result_2 = build_chunk_record(
        pdf_path,
        root_dir,
        chunk,
        4,
        chunker,
    )

    assert result_1["chunk_id"] != result_2["chunk_id"]


def test_build_chunk_records(monkeypatch, tmp_path):
    root_dir = tmp_path

    pdf_1 = tmp_path / "paper_1.pdf"
    pdf_2 = tmp_path / "paper_2.pdf"

    chunk_1 = MagicMock(name="chunk_1")
    chunk_2 = MagicMock(name="chunk_2")
    chunk_3 = MagicMock(name="chunk_3")

    chunked_docs = {
        pdf_1: [chunk_1, chunk_2],
        pdf_2: [chunk_3],
    }

    converter = MagicMock()
    chunker = MagicMock()

    mock_chunk_documents = MagicMock(
        return_value=chunked_docs
    )

    monkeypatch.setattr(
        "research_rag.ingestion.records.chunk_documents",
        mock_chunk_documents,
    )

    def fake_build_chunk_record(
        pdf_path,
        received_root_dir,
        chunk,
        chunk_index,
        received_chunker,
    ):
        return {
            "file_name": pdf_path.name,
            "chunk_index": chunk_index,
            "chunk": chunk,
        }

    mock_build_chunk_record = MagicMock(
        side_effect=fake_build_chunk_record
    )

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_chunk_record",
        mock_build_chunk_record,
    )

    result = build_chunk_records(
        root_dir,
        converter,
        chunker,
    )

    assert len(result) == 3

    assert result[0]["file_name"] == "paper_1.pdf"
    assert result[0]["chunk_index"] == 0
    assert result[0]["chunk"] is chunk_1

    assert result[1]["file_name"] == "paper_1.pdf"
    assert result[1]["chunk_index"] == 1
    assert result[1]["chunk"] is chunk_2

    assert result[2]["file_name"] == "paper_2.pdf"
    assert result[2]["chunk_index"] == 0
    assert result[2]["chunk"] is chunk_3

    mock_chunk_documents.assert_called_once_with(
        root_dir.resolve(),
        converter,
        chunker,
    )

    assert mock_build_chunk_record.call_count == 3

    expected_calls = [
        (
            pdf_1,
            root_dir.resolve(),
            chunk_1,
            0,
            chunker,
        ),
        (
            pdf_1,
            root_dir.resolve(),
            chunk_2,
            1,
            chunker,
        ),
        (
            pdf_2,
            root_dir.resolve(),
            chunk_3,
            0,
            chunker,
        ),
    ]

    actual_calls = [
        call.args
        for call in mock_build_chunk_record.call_args_list
    ]

    assert actual_calls == expected_calls