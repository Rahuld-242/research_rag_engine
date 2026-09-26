from pathlib import Path
from pypdf import PdfReader
from research_rag.ingestion.discovery import find_pdf_files

def extract_pdf_text(pdf_path: Path) -> str:
    pdf_path = Path(pdf_path).resolve()
    
    pdf_file = PdfReader(pdf_path)
    text_list=[]
    
    for page in pdf_file.pages:
        page_text=page.extract_text()
        
        if page_text:
            text_list.append(page_text)
        
    pdf_file_text = "\n".join(text_list)
    
    return pdf_file_text

def extract_pdf_texts(root_dir: Path) -> dict[Path, str]:
    root_dir = Path(root_dir).resolve()
    
    pdf_files=find_pdf_files(root_dir)
    
    pdf_file_texts = {pdf_file: extract_pdf_text(pdf_file) for pdf_file in pdf_files}
    
    return pdf_file_texts