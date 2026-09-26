from pathlib import Path

from research_rag.ingestion.extraction import extract_pdf_texts


def test_extract_pdf_texts(monkeypatch, tmp_path):
    pdf1 = tmp_path / "paper1.pdf"
    pdf2 = tmp_path / "paper2.pdf"

    pdf1.touch()
    pdf2.touch()

    def fake_find_pdf_files(root_dir):
        return [pdf1, pdf2]

    def fake_extract_pdf_text(pdf_path):
        if pdf_path == pdf1:
            return "Text from paper 1"
        if pdf_path == pdf2:
            return "Text from paper 2"

    monkeypatch.setattr(
        "research_rag.ingestion.extraction.find_pdf_files",
        fake_find_pdf_files,
    )

    monkeypatch.setattr(
        "research_rag.ingestion.extraction.extract_pdf_text",
        fake_extract_pdf_text,
    )

    result = extract_pdf_texts(tmp_path)

    assert result == {
        pdf1: "Text from paper 1",
        pdf2: "Text from paper 2",
    }