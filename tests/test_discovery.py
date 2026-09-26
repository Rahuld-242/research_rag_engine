from research_rag.ingestion.discovery import find_pdf_files

def test_find_pdf_files_returns_pdf_files_recursively(tmp_path):
    root_dir=tmp_path/"papers"
    nested_dir=root_dir/"05_RAG_and_Retrieval"
    nested_dir.mkdir(parents=True)
    
    pdf_1=root_dir/"paper_1.pdf"
    pdf_2=nested_dir/"paper_2.pdf"
    txt_file=root_dir/"notes.txt"
    
    pdf_1.write_text("dummy pdf1")
    pdf_2.write_text("dummy pdf2")
    txt_file.write_text("not a pdf")
    
    pdf_files=find_pdf_files(root_dir)
    
    assert len(pdf_files)==2
    assert pdf_1 in pdf_files
    assert pdf_2 in pdf_files
    assert txt_file not in pdf_files