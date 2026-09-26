from pathlib import Path

from research_rag.ingestion.records import (
    build_extracted_document_record,
    build_extracted_document_records,
)


def test_build_extracted_document_record(monkeypatch, tmp_path):
    pdf_path = tmp_path / "paper1.pdf"
    pdf_path.touch()

    fake_metadata_record = {
        "document_id": "topic/paper1",
        "document_hash": "ABC123",
        "file_name": "paper1.pdf",
        "file_path": str(pdf_path.resolve()),
        "primary_topic": "Topic",
        "subtopic": None,
    }

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_document_record",
        lambda pdf_path, root_dir: fake_metadata_record.copy(),
    )

    monkeypatch.setattr(
        "research_rag.ingestion.records.extract_pdf_text",
        lambda pdf_path: "This is the extracted text.",
    )

    result = build_extracted_document_record(pdf_path, tmp_path)

    print("\nExtracted document record:")
    print(result)

    assert result["document_id"] == "topic/paper1"
    assert result["file_name"] == "paper1.pdf"
    assert result["text"] == "This is the extracted text."
    assert "primary_topic" in result


def test_build_extracted_document_records(monkeypatch, tmp_path):
    pdf1 = tmp_path / "paper1.pdf"
    pdf2 = tmp_path / "paper2.pdf"

    pdf1.touch()
    pdf2.touch()

    monkeypatch.setattr(
        "research_rag.ingestion.records.find_pdf_files",
        lambda root_dir: [pdf1, pdf2],
    )

    def fake_build_document_record(pdf_path, root_dir):
        return {
            "document_id": f"topic/{pdf_path.stem}",
            "document_hash": f"hash_{pdf_path.stem}",
            "file_name": pdf_path.name,
            "file_path": str(pdf_path.resolve()),
            "primary_topic": "RAG",
            "subtopic": "Dense Retrieval",
        }

    monkeypatch.setattr(
        "research_rag.ingestion.records.build_document_record",
        fake_build_document_record,
    )

    monkeypatch.setattr(
        "research_rag.ingestion.records.extract_pdf_text",
        lambda pdf_path: f"Text from {pdf_path.name}",
    )

    result = build_extracted_document_records(tmp_path)

    print("\nExtracted document records:")
    for record in result:
        print(record)

    assert len(result) == 2

    assert result[0]["file_name"] == "paper1.pdf"
    assert result[0]["primary_topic"] == "RAG"
    assert result[0]["subtopic"] == "Dense Retrieval"
    assert result[0]["text"] == "Text from paper1.pdf"

    assert result[1]["file_name"] == "paper2.pdf"
    assert result[1]["text"] == "Text from paper2.pdf"