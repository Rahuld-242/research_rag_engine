from unittest.mock import MagicMock

from research_rag.ingestion.chunking import (
    chunk_document,
    chunk_documents,
)


def test_chunk_document():
    doc = MagicMock()
    chunker = MagicMock()

    chunk_1 = MagicMock()
    chunk_2 = MagicMock()

    chunker.chunk.return_value = iter([chunk_1, chunk_2])

    result = chunk_document(doc, chunker)

    assert result == [chunk_1, chunk_2]

    chunker.chunk.assert_called_once_with(
        dl_doc=doc
    )


def test_chunk_documents(monkeypatch, tmp_path):
    pdf_1 = tmp_path / "paper_1.pdf"
    pdf_2 = tmp_path / "paper_2.pdf"

    doc_1 = MagicMock()
    doc_2 = MagicMock()

    parsed_pdfs = {
        pdf_1: doc_1,
        pdf_2: doc_2,
    }

    chunks_1 = [MagicMock(), MagicMock()]
    chunks_2 = [MagicMock()]

    converter = MagicMock()
    chunker = MagicMock()

    mock_parse_pdfs = MagicMock(
        return_value=parsed_pdfs
    )

    monkeypatch.setattr(
        "research_rag.ingestion.chunking.parse_pdfs",
        mock_parse_pdfs
    )

    def fake_chunk_document(doc, received_chunker):

        assert received_chunker is chunker

        if doc is doc_1:
            return chunks_1

        if doc is doc_2:
            return chunks_2

    monkeypatch.setattr(
        "research_rag.ingestion.chunking.chunk_document",
        fake_chunk_document
    )

    result = chunk_documents(
        tmp_path,
        converter,
        chunker
    )

    assert result == {
        pdf_1: chunks_1,
        pdf_2: chunks_2,
    }

    mock_parse_pdfs.assert_called_once_with(
        tmp_path.resolve(),
        converter
    )