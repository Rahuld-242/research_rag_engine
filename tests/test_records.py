from research_rag.ingestion.records import (
    build_document_record,
    build_document_records
)


def test_build_document_record_returns_document_metadata(tmp_path):
    root_dir = tmp_path / "AIML Research Papers"
    topic_dir = root_dir / "05_RAG_and_Retrieval"
    topic_dir.mkdir(parents=True)

    pdf_path = topic_dir / "RAG Survey.pdf"
    pdf_path.write_text("dummy pdf content", encoding="utf-8")

    record = build_document_record(pdf_path, root_dir)

    assert record["document_id"] == "05_RAG_and_Retrieval/RAG Survey.pdf"
    assert len(record["document_hash"]) == 16
    assert record["file_name"] == "RAG Survey.pdf"
    assert record["file_path"] == str(pdf_path.resolve())

    assert record["topic_parts"] == ["05_RAG_and_Retrieval"]
    assert record["topic_names"] == ["RAG and Retrieval"]
    assert record["primary_topic"] == "RAG and Retrieval"
    assert record["subtopic"] is None
    assert record["topic_path"] == "RAG and Retrieval"
    
def test_build_document_records_returns_records_for_all_pdfs(tmp_path):
    root_dir = tmp_path / "AIML Research Papers"

    rag_dir = root_dir / "05_RAG_and_Retrieval"
    dpo_dir = (
        root_dir
        / "03_Fine_Tuning_and_Alignment"
        / "02_Instruction_Tuning_and_Preference_Optimization"
    )

    rag_dir.mkdir(parents=True)
    dpo_dir.mkdir(parents=True)

    rag_pdf = rag_dir / "RAG Survey.pdf"
    dpo_pdf = dpo_dir / "DPO.pdf"
    notes_file = root_dir / "notes.txt"

    rag_pdf.write_text("dummy rag pdf", encoding="utf-8")
    dpo_pdf.write_text("dummy dpo pdf", encoding="utf-8")
    notes_file.write_text("not a pdf", encoding="utf-8")

    records = build_document_records(root_dir)

    document_ids = [record["document_id"] for record in records]

    assert len(records) == 2
    assert "05_RAG_and_Retrieval/RAG Survey.pdf" in document_ids
    assert (
        "03_Fine_Tuning_and_Alignment/"
        "02_Instruction_Tuning_and_Preference_Optimization/"
        "DPO.pdf"
    ) in document_ids
    assert "notes.txt" not in document_ids